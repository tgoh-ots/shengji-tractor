use std::collections::HashMap;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Instant;

use anyhow::bail;
use slog::{debug, error, info, o, Logger};
use tokio::sync::{mpsc, oneshot, Mutex, Semaphore};

use shengji_core::bot::{
    apply_planned_bot_action, classify_next_bot_work, plan_next_bot_action, BotPause, BotStep,
    NextBotWork,
};
use shengji_core::game_state::GameState;
use shengji_core::interactive::{Action, InteractiveGame};
use shengji_mechanics::types::PlayerID;
use shengji_types::GameMessage;
use storage::Storage;

use crate::{
    security::{
        sanitize_chat_message, sanitize_text, ResourceLimits, MAX_NAME_BYTES, ROOM_NAME_BYTES,
    },
    serving_types::{JoinRoom, UserMessage, VersionedGame},
    state_dump::InMemoryStats,
    utils::{
        execute_immutable_operation, execute_operation, execute_operation_at_version,
        VersionedOperationOutcome,
    },
    ZSTD_COMPRESSOR,
};

#[derive(Debug, Default)]
struct RoomDriveState {
    /// At least one request arrived while this room's driver was running.  The
    /// active driver performs one more complete pass before releasing the room,
    /// which closes the otherwise-racy "request arrived just before exit" gap.
    dirty: bool,
}

/// Process-wide coordination for bot work.
///
/// There may be many WebSocket connections per room, each of which can observe
/// and trigger the same state transition.  `rooms` makes bot driving
/// single-flight per room with a dirty/re-run bit, while `search_slots` bounds
/// CPU-heavy planning globally so wall-clock search budgets remain meaningful on
/// small deployments.
#[derive(Clone, Debug)]
pub struct BotRuntime {
    rooms: Arc<StdMutex<HashMap<String, RoomDriveState>>>,
    search_slots: Arc<Semaphore>,
    max_parallel_searches: usize,
}

impl BotRuntime {
    pub fn new(max_parallel_searches: usize) -> Self {
        let max_parallel_searches = max_parallel_searches.max(1);
        Self {
            rooms: Arc::new(StdMutex::new(HashMap::new())),
            search_slots: Arc::new(Semaphore::new(max_parallel_searches)),
            max_parallel_searches,
        }
    }

    pub fn active_room_drivers(&self) -> usize {
        self.rooms.lock().unwrap_or_else(|p| p.into_inner()).len()
    }

    pub fn active_searches(&self) -> usize {
        self.max_parallel_searches
            .saturating_sub(self.search_slots.available_permits())
    }

    pub fn max_parallel_searches(&self) -> usize {
        self.max_parallel_searches
    }

    /// Request a drive for `room`.  The first caller receives a lease and starts
    /// the task; later callers only mark the running task dirty.
    fn request_drive(&self, room: &str) -> Option<RoomDriveLease> {
        let mut rooms = self.rooms.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(state) = rooms.get_mut(room) {
            state.dirty = true;
            return None;
        }
        rooms.insert(room.to_owned(), RoomDriveState::default());
        Some(RoomDriveLease {
            runtime: self.clone(),
            room: room.to_owned(),
            active: true,
        })
    }
}

impl Default for BotRuntime {
    fn default() -> Self {
        Self::new(1)
    }
}

/// RAII lease for one room's bot driver.  Dropping a cancelled/panicked task
/// clears the room, so a later user action can always restart it.
struct RoomDriveLease {
    runtime: BotRuntime,
    room: String,
    active: bool,
}

impl RoomDriveLease {
    /// Finish one driver pass.  Returns true when an overlapping request marked
    /// the room dirty and another pass is required.
    fn continue_if_dirty(&mut self) -> bool {
        let mut rooms = self.runtime.rooms.lock().unwrap_or_else(|p| p.into_inner());
        match rooms.get_mut(&self.room) {
            Some(state) if state.dirty => {
                state.dirty = false;
                true
            }
            _ => {
                rooms.remove(&self.room);
                self.active = false;
                false
            }
        }
    }
}

impl Drop for RoomDriveLease {
    fn drop(&mut self) {
        if self.active {
            self.runtime
                .rooms
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .remove(&self.room);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn entrypoint<
    S: Storage<VersionedGame, E> + 'static,
    E: std::fmt::Debug + Send + 'static,
>(
    tx: mpsc::UnboundedSender<Vec<u8>>,
    rx: mpsc::UnboundedReceiver<Vec<u8>>,
    ws_id: usize,
    logger: Logger,
    backend_storage: S,
    stats: Arc<Mutex<InMemoryStats>>,
    resource_limits: Arc<ResourceLimits>,
    bot_runtime: BotRuntime,
) {
    let _ = handle_user_connected(
        tx,
        rx,
        ws_id,
        logger,
        backend_storage,
        stats,
        resource_limits,
        bot_runtime,
    )
    .await;
}

async fn send_to_user(
    tx: &'_ mpsc::UnboundedSender<Vec<u8>>,
    msg: &GameMessage,
) -> Result<(), anyhow::Error> {
    send_to_user_with_compression(tx, msg, false).await
}

async fn send_to_user_with_compression(
    tx: &'_ mpsc::UnboundedSender<Vec<u8>>,
    msg: &GameMessage,
    disable_compression: bool,
) -> Result<(), anyhow::Error> {
    if let Ok(j) = serde_json::to_vec(&msg) {
        let data = if disable_compression {
            j
        } else {
            ZSTD_COMPRESSOR
                .lock()
                .unwrap()
                .compress(&j)
                .map_err(|_| anyhow::anyhow!("Unable to compress message"))?
        };

        if tx.send(data).is_ok() {
            return Ok(());
        }
    }
    Err(anyhow::anyhow!("Unable to send message to user {:?}", msg))
}

#[allow(clippy::too_many_arguments)]
async fn handle_user_connected<
    S: Storage<VersionedGame, E> + 'static,
    E: std::fmt::Debug + Send + 'static,
>(
    tx: mpsc::UnboundedSender<Vec<u8>>,
    mut rx: mpsc::UnboundedReceiver<Vec<u8>>,
    ws_id: usize,
    logger: Logger,
    backend_storage: S,
    stats: Arc<Mutex<InMemoryStats>>,
    resource_limits: Arc<ResourceLimits>,
    bot_runtime: BotRuntime,
) -> Result<(), anyhow::Error> {
    let (room, name, disable_compression) = loop {
        if let Some(msg) = rx.recv().await {
            let err = match serde_json::from_slice::<JoinRoom>(&msg) {
                Ok(JoinRoom {
                    room_name,
                    name,
                    disable_compression,
                }) => {
                    // Sanitize the player name server-side: strip control
                    // characters and enforce the length cap. The room name has a
                    // fixed length and is treated as an opaque id, so it is only
                    // length-validated (not sanitized) to preserve room keys.
                    let name = sanitize_text(&name);
                    if room_name.len() == ROOM_NAME_BYTES
                        && room_name.chars().all(|c| !c.is_control())
                        && !name.is_empty()
                        && name.len() < MAX_NAME_BYTES
                    {
                        break (room_name, name, disable_compression);
                    }
                    GameMessage::Error("invalid room or name".to_string())
                }
                Err(err) => GameMessage::Error(format!("couldn't deserialize message {err:?}")),
            };

            send_to_user(&tx, &err).await?;
        } else {
            Err(anyhow::anyhow!("no message on socket"))?;
        }
    };

    let logger = logger.new(o!("room" => room.clone(), "name" => name.clone()));

    // Enforce the global cap on the number of concurrent rooms. If this would be
    // a brand-new room and we are already at capacity, reject the join. Joining
    // an already-existing room is always allowed (it doesn't grow the room count).
    if let Ok(keys) = backend_storage.clone().get_all_keys().await {
        let room_key = room.as_bytes().to_vec();
        let room_exists = keys.contains(&room_key);
        if !room_exists && keys.len() >= resource_limits.max_total_rooms {
            let _ = send_to_user(
                &tx,
                &GameMessage::Error(
                    "The server is at capacity; please try again later.".to_string(),
                ),
            )
            .await;
            return Err(anyhow::anyhow!(
                "rejecting new room: at max_total_rooms capacity"
            ));
        }
    }

    let subscription = match backend_storage
        .clone()
        .subscribe(room.as_bytes().to_vec(), ws_id)
        .await
    {
        Ok(sub) => sub,
        Err(e) => {
            let _ = send_to_user(
                &tx,
                &GameMessage::Error(format!("Failed to join room: {e:?}")),
            )
            .await;
            return Err(anyhow::anyhow!("Failed to join room {:?}", e));
        }
    };

    // Subscribe to messages for the room. After this point, we should
    // no longer use tx! It's owned by the backend storage.
    let (subscribe_player_id_tx, subscribe_player_id_rx) = oneshot::channel::<PlayerID>();
    tokio::task::spawn(player_subscribe_task(
        logger.clone(),
        name.clone(),
        tx.clone(),
        subscribe_player_id_rx,
        subscription,
        disable_compression,
    ));

    let (player_id, join_span) = register_user(
        logger.clone(),
        name.clone(),
        ws_id,
        room.clone(),
        backend_storage.clone(),
        stats.clone(),
        resource_limits.clone(),
    )
    .await
    .map_err(|_| anyhow::anyhow!("Failed to register user"))?;

    let logger = logger.new(o!("player_id" => player_id.0));
    info!(logger, "Successfully registered user");
    let _ = subscribe_player_id_tx.send(player_id);

    // A persisted room may have been restored while a bot was responsible for
    // the next action.  Registration is the first event after wake/reconnect, so
    // request a drive here as well as after explicit user actions.
    request_bot_drive(
        logger.clone(),
        ws_id,
        room.clone(),
        backend_storage.clone(),
        bot_runtime.clone(),
    );

    run_game_for_player(
        logger.clone(),
        ws_id,
        player_id,
        room.clone(),
        name,
        backend_storage.clone(),
        rx,
        bot_runtime,
    )
    .await;

    // user_ws_rx stream will keep processing as long as the user stays
    // connected. Once they disconnect, then...
    user_disconnected(room, ws_id, backend_storage, logger, join_span).await;
    Ok(())
}

async fn player_subscribe_task(
    logger_: Logger,
    name_: String,
    tx: mpsc::UnboundedSender<Vec<u8>>,
    subscribe_player_id_rx: oneshot::Receiver<PlayerID>,
    mut subscription: mpsc::UnboundedReceiver<GameMessage>,
    disable_compression: bool,
) {
    debug!(logger_, "Subscribed to messages");
    if let Ok(player_id) = subscribe_player_id_rx.await {
        let logger_ = logger_.new(o!("player_id" => player_id.0));
        debug!(logger_, "Received player ID");
        while let Some(v) = subscription.recv().await {
            let should_send = match &v {
                GameMessage::State { .. }
                | GameMessage::Broadcast { .. }
                | GameMessage::Message { .. }
                | GameMessage::Error(_)
                | GameMessage::Header { .. } => true,
                GameMessage::Beep { target } | GameMessage::Kicked { target } => *target == name_,
                GameMessage::ReadyCheck { from } => *from != name_,
            };
            let v = if should_send {
                if let GameMessage::State { state } = v {
                    let g = InteractiveGame::new_from_state(state);
                    g.dump_state_for_player(player_id)
                        .ok()
                        .map(|state| GameMessage::State { state })
                } else {
                    Some(v)
                }
            } else {
                None
            };

            if let Some(v) = v {
                if send_to_user_with_compression(&tx, &v, disable_compression)
                    .await
                    .is_err()
                {
                    break;
                }
            }
        }
    }
    debug!(logger_, "Subscription task completed");
}

#[allow(clippy::too_many_arguments)]
async fn register_user<S: Storage<VersionedGame, E>, E: std::fmt::Debug + Send>(
    logger: Logger,
    name: String,
    ws_id: usize,
    room: String,
    backend_storage: S,
    stats: Arc<Mutex<InMemoryStats>>,
    resource_limits: Arc<ResourceLimits>,
) -> Result<(PlayerID, u64), ()> {
    let (player_id_tx, player_id_rx) = oneshot::channel();
    let logger_ = logger.clone();
    let name_ = name.clone();
    let limits = resource_limits.clone();
    execute_operation(
        ws_id,
        &room,
        backend_storage.clone(),
        move |g, version, associated_websockets| {
            // Enforce per-room player/observer caps for *new* participants. A
            // rejoin under an existing name is always allowed (it doesn't grow
            // the room). Whether a new registrant becomes a player or an
            // observer depends on the game phase: the lobby (Initialize) seats
            // players, all later phases add observers.
            if !g.has_participant_named(&name_) {
                if g.is_in_lobby() {
                    if g.num_players() >= limits.max_players_per_room {
                        bail!("This room is full (maximum players reached).");
                    }
                } else if g.num_observers() >= limits.max_observers_per_room {
                    bail!("This room has too many observers.");
                }
            }
            let (assigned_player_id, register_msgs) = g.register(name_)?;
            info!(logger_, "Joining room"; "player_id" => assigned_player_id.0);
            let mut clients_to_disconnect = vec![];
            let clients = associated_websockets.entry(assigned_player_id).or_default();
            // If the same user joined before, remove the previous entries
            // from the state-store.
            if !g.allows_multiple_sessions_per_user() {
                std::mem::swap(&mut clients_to_disconnect, clients);
            }
            clients.push(ws_id);

            player_id_tx
                .send((assigned_player_id, version, clients_to_disconnect))
                .map_err(|_| anyhow::anyhow!("Couldn't send player ID back".to_owned()))?;
            Ok(register_msgs
                .into_iter()
                .map(|(data, message)| GameMessage::Broadcast { data, message })
                .collect())
        },
        "register game",
    )
    .await;

    let header_messages = {
        let stats = stats.lock().await;
        stats.header_messages().to_vec()
    };
    let _ = backend_storage
        .clone()
        .publish_to_single_subscriber(
            room.as_bytes().to_vec(),
            ws_id,
            GameMessage::Header {
                messages: header_messages,
            },
        )
        .await;

    if let Ok((player_id, ws_id, websockets_to_disconnect)) = player_id_rx.await {
        for id in websockets_to_disconnect {
            info!(logger, "Disconnnecting existing client"; "kicked_ws_id" => id);
            let _ = backend_storage
                .clone()
                .publish_to_single_subscriber(
                    room.as_bytes().to_vec(),
                    id,
                    GameMessage::Kicked {
                        target: name.clone(),
                    },
                )
                .await;
        }
        Ok((player_id, ws_id))
    } else {
        Err(())
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_game_for_player<
    S: Storage<VersionedGame, E> + 'static,
    E: Send + std::fmt::Debug + 'static,
>(
    logger: Logger,
    ws_id: usize,
    player_id: PlayerID,
    room: String,
    name: String,
    backend_storage: S,
    mut rx: mpsc::UnboundedReceiver<Vec<u8>>,
    bot_runtime: BotRuntime,
) {
    debug!(logger, "Entering main game loop");
    // Handle the main game loop
    while let Some(result) = rx.recv().await {
        match serde_json::from_slice::<UserMessage>(&result) {
            Ok(msg) => {
                if let Err(e) = handle_user_action(
                    logger.clone(),
                    ws_id,
                    player_id,
                    &room,
                    name.clone(),
                    backend_storage.clone(),
                    msg,
                    bot_runtime.clone(),
                )
                .await
                {
                    let _ = backend_storage
                        .clone()
                        .publish_to_single_subscriber(
                            room.as_bytes().to_vec(),
                            ws_id,
                            GameMessage::Error(format!("Unexpected error {e:?}")),
                        )
                        .await;
                }
            }
            Err(e) => {
                error!(logger, "Failed to deserialize message"; "error" => format!("{e:?}"));
                let _ = backend_storage
                    .clone()
                    .publish_to_single_subscriber(
                        room.as_bytes().to_vec(),
                        ws_id,
                        GameMessage::Error(format!("couldn't deserialize message {e:?}")),
                    )
                    .await;
            }
        }
    }
    debug!(logger, "Exiting main game loop");
}

#[allow(clippy::too_many_arguments)]
async fn handle_user_action<S: Storage<VersionedGame, E> + 'static, E: Send + 'static>(
    logger: Logger,
    ws_id: usize,
    caller: PlayerID,
    room_name: &str,
    name: String,
    backend_storage: S,
    msg: UserMessage,
    bot_runtime: BotRuntime,
) -> Result<(), E> {
    match msg {
        UserMessage::Beep => {
            execute_immutable_operation(
                ws_id,
                room_name,
                backend_storage,
                move |game, _| {
                    let next_player_id = game.next_player()?;
                    let beeped_player_name = game.player_name(next_player_id)?.to_owned();
                    Ok(vec![
                        GameMessage::Message {
                            from: name,
                            message: "BEEP".to_owned(),
                        },
                        GameMessage::Beep {
                            target: beeped_player_name,
                        },
                    ])
                },
                "send appropriate beep",
            )
            .await;
        }
        UserMessage::Message(m) => {
            // Sanitize chat server-side: strip control characters, trim, and
            // reject empty or oversized messages. The frontend renders chat as
            // text, but we still neutralize control sequences and bound length
            // before storing/broadcasting.
            match sanitize_chat_message(&m) {
                Some(message) => {
                    backend_storage
                        .publish(
                            room_name.as_bytes().to_vec(),
                            GameMessage::Message {
                                from: name,
                                message,
                            },
                        )
                        .await?;
                }
                None => {
                    let _ = backend_storage
                        .publish_to_single_subscriber(
                            room_name.as_bytes().to_vec(),
                            ws_id,
                            GameMessage::Error("Chat message was empty or too long.".to_string()),
                        )
                        .await;
                }
            }
        }
        UserMessage::ReadyCheck => {
            backend_storage
                .clone()
                .publish(
                    room_name.as_bytes().to_vec(),
                    GameMessage::Message {
                        from: name.clone(),
                        message: "Is everyone ready?".to_owned(),
                    },
                )
                .await?;
            backend_storage
                .publish(
                    room_name.as_bytes().to_vec(),
                    GameMessage::ReadyCheck { from: name },
                )
                .await?;
        }
        UserMessage::Ready => {
            backend_storage
                .publish(
                    room_name.as_bytes().to_vec(),
                    GameMessage::Message {
                        from: name,
                        message: "I'm ready!".to_owned(),
                    },
                )
                .await?;
        }
        UserMessage::Kick(id) => {
            info!(logger, "Kicking user"; "other" => id.0);
            execute_operation(
                ws_id,
                room_name,
                backend_storage,
                move |game, _, _| {
                    let kicked_player_name = game.player_name(id)?.to_owned();
                    game.kick(caller, id)?;
                    Ok(vec![GameMessage::Kicked {
                        target: kicked_player_name,
                    }])
                },
                "kick user",
            )
            .await;
        }
        UserMessage::Action(action) => {
            // Sanitize free-text carried by actions before applying them. A bot
            // rename is the one action that carries an arbitrary display name, so
            // strip control characters here (reusing the same `sanitize_text`
            // helper as the join path); core then trims, length-bounds, and
            // uniqueness-checks it.
            let action = match action {
                Action::RenameBot { player, name } => Action::RenameBot {
                    player,
                    name: sanitize_text(&name),
                },
                other => other,
            };
            // Apply ONLY the human's action under the game lock (a cheap
            // operation), publish the resulting state, and release the lock. The
            // (potentially CPU-heavy) bot move computation is then driven OFF the
            // lock by `drive_bots_non_blocking` below, so chat and other players'
            // actions keep flowing while a Hard/Expert bot "thinks". See that
            // function for the snapshot -> spawn_blocking compute -> apply-under-
            // lock-with-recheck model.
            let op_logger = logger.clone();
            execute_operation(
                ws_id,
                room_name,
                backend_storage.clone(),
                move |game, _, _| {
                    let broadcasts = game.interact(action, caller, &op_logger)?;
                    Ok(broadcasts
                        .into_iter()
                        .map(|(data, message)| GameMessage::Broadcast { data, message })
                        .collect())
                },
                "handle user action",
            )
            .await;

            // Drive any bots that now need to act WITHOUT holding the game lock
            // across their computation. Each bot move is selected on a blocking
            // worker from a cloned snapshot, then applied under a brief lock with a
            // turn/state re-check. Pacing (the trick-clear and per-action beats)
            // and the done-bidding park are preserved.
            //
            // When a BOT holds the standing bid after the deck is drawn, the driver
            // PARKS (plans nothing) until every human seat clicks "Done bidding".
            // That click is an ordinary `Action::MarkBiddingDone` which flows
            // through this same branch and re-runs the driver — so once the LAST
            // human is done, the bot finalizes itself here, with no timer and no
            // forced finalization. An all-bot table has zero humans, so "all humans
            // done" is trivially true and the bot proceeds immediately (no deadlock).
            request_bot_drive(
                logger,
                ws_id,
                room_name.to_string(),
                backend_storage,
                bot_runtime,
            );
        }
    }
    Ok(())
}

fn request_bot_drive<S: Storage<VersionedGame, E> + 'static, E: Send + 'static>(
    logger: Logger,
    ws_id: usize,
    room_name: String,
    backend_storage: S,
    bot_runtime: BotRuntime,
) {
    if let Some(mut lease) = bot_runtime.request_drive(&room_name) {
        let runtime = bot_runtime.clone();
        tokio::task::spawn(async move {
            loop {
                drive_bots_non_blocking(
                    logger.clone(),
                    ws_id,
                    room_name.clone(),
                    backend_storage.clone(),
                    runtime.clone(),
                )
                .await;
                if !lease.continue_if_dirty() {
                    break;
                }
            }
        });
    } else {
        debug!(logger, "Coalesced bot-drive request into active room task");
    }
}

/// Resume bot-owned turns in rooms restored from the state dump.  Rooms parked
/// on a human remain untouched; all-bot or bot-to-act rooms continue without
/// waiting for an unrelated WebSocket action after process restart.
pub async fn resume_restored_bot_rooms<
    S: Storage<VersionedGame, E> + 'static,
    E: Send + std::fmt::Debug + 'static,
>(
    logger: Logger,
    backend_storage: S,
    bot_runtime: BotRuntime,
) {
    let keys = match backend_storage.clone().get_all_keys().await {
        Ok(keys) => keys,
        Err(error) => {
            error!(logger, "Unable to enumerate restored rooms for bot resume"; "error" => format!("{error:?}"));
            return;
        }
    };
    let mut requested = 0usize;
    for key in keys {
        let Ok(room_name) = String::from_utf8(key) else {
            continue;
        };
        requested += 1;
        request_bot_drive(
            logger.clone(),
            0,
            room_name,
            backend_storage.clone(),
            bot_runtime.clone(),
        );
    }
    if requested > 0 {
        info!(logger, "Requested bot resume for restored rooms"; "rooms" => requested);
    }
}

/// Default human-visible pause (milliseconds) before a bot clears a trick it
/// won, so the human can see the completed 4-card trick. This is the LONGER beat.
/// Overridable at runtime via the `SHENGJI_BOT_TRICK_PAUSE_MS` environment
/// variable.
const DEFAULT_BOT_TRICK_PAUSE_MS: u64 = 2500;

/// Default human-visible pause (milliseconds) after a single meaningful bot move
/// (a play, a bid, a kitty/landlord/exchange decision) so a human can register
/// what one bot did before the next acts. This is the SHORTER beat. Overridable
/// at runtime via the `SHENGJI_BOT_ACTION_PAUSE_MS` environment variable.
const DEFAULT_BOT_ACTION_PAUSE_MS: u64 = 1200;

/// Read the configured pause for the given beat, defaulting to the matching
/// constant if the env var is unset or unparseable. The trick-clear beat is the
/// longer pause (so the full 4-card trick is visible); the per-action beat is the
/// shorter pause (one move at a time).
fn bot_pause(kind: BotPause) -> std::time::Duration {
    let (var, default) = match kind {
        BotPause::TrickClear => ("SHENGJI_BOT_TRICK_PAUSE_MS", DEFAULT_BOT_TRICK_PAUSE_MS),
        BotPause::Action => ("SHENGJI_BOT_ACTION_PAUSE_MS", DEFAULT_BOT_ACTION_PAUSE_MS),
    };
    let ms = std::env::var(var)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default);
    std::time::Duration::from_millis(ms)
}

/// Read a CLONE of the current game state out of storage under a brief lock,
/// without mutating or re-versioning it. Used by `drive_bots_non_blocking` to
/// snapshot the state the bot needs BEFORE computing its move off the lock.
/// Returns `None` if the read failed (e.g. the room vanished).
#[derive(Clone)]
struct BotSnapshot {
    state: GameState,
    version: u64,
}

async fn snapshot_state<S: Storage<VersionedGame, E> + 'static, E: Send + 'static>(
    ws_id: usize,
    room_name: &str,
    backend_storage: S,
) -> Option<BotSnapshot> {
    let captured: Arc<StdMutex<Option<BotSnapshot>>> = Arc::new(StdMutex::new(None));
    let sink = captured.clone();
    // `execute_immutable_operation` runs the closure with `&InteractiveGame` and
    // does NOT bump the version or publish a State, so this is a pure read.
    execute_immutable_operation(
        ws_id,
        room_name,
        backend_storage,
        move |game, version| {
            if let Ok(state) = game.dump_state() {
                *sink.lock().unwrap_or_else(|p| p.into_inner()) =
                    Some(BotSnapshot { state, version });
            }
            Ok(vec![])
        },
        "snapshot game state",
    )
    .await;
    let state = captured.lock().unwrap_or_else(|p| p.into_inner()).take();
    state
}

/// Detached task that drives any bots that need to act AFTER a user action, WITHOUT
/// holding the game lock across the (potentially CPU-heavy) move computation. This
/// is what keeps chat and other players' actions responsive while a Hard/Expert bot
/// "thinks".
///
/// Each iteration has two parts:
///
/// **(A) Cheap burst, under one brief lock.** `advance_bots_burst_unpaced` applies
/// every UN-PACED bot step — bot draws and reset confirmations — in a SINGLE locked
/// operation (one State publish instead of ~75 during the deal). These selections
/// are cheap (never the Play-phase search), so doing them under the lock is fine and
/// avoids flooding clients with one State per draw. It stops as soon as the next bot
/// step is a paceable / trick-clear beat, a human's turn, or nothing.
///
/// **(B) One paceable step, computed OFF the lock.** For the next paceable beat (a
/// bid, a kitty/reveal/exchange decision, a play, or a bot finishing a trick) we:
///   1. **Snapshot** the state under a brief lock (a cheap clone), release the lock;
///   2. **Compute** the move with `plan_next_bot_action` on a
///      `tokio::task::spawn_blocking` worker — OFF the async runtime AND off the lock
///      — so neither the lock nor an async worker is blocked for the up-to-1s search;
///   3. **Apply** the precomputed step under a brief lock only if the room's
///      monotonic version still exactly matches the snapshot, then also re-check
///      the expected actor/action kind. If the world moved on, the stale step is
///      a true no-op and we re-snapshot/re-plan — no double-apply.
///
/// Pacing is preserved exactly:
///
/// * Planning begins immediately after the previous action.  Publication waits
///   only for any remainder of the visible action beat, so thinking and pacing
///   overlap.
/// * `BotStep::pause == Some(TrickClear)` keeps the completed trick visible until
///   the longer deadline measured from the last card's publication, then applies
///   `EndTrick`.
///
/// The done-bidding PARK is preserved: when a bot holds the standing bid and a human
/// could still outbid, both the burst and `plan_next_bot_action` make no move, so
/// this task simply ends. The last human's "Done bidding" click re-runs the driver
/// via the ordinary action path. The iteration bound caps chained steps within a
/// hand so a wedged state can never spin this task forever.
async fn drive_bots_non_blocking<S: Storage<VersionedGame, E> + 'static, E: Send + 'static>(
    logger: Logger,
    ws_id: usize,
    room_name: String,
    backend_storage: S,
    bot_runtime: BotRuntime,
) {
    // The time at which the previous visible bot action was applied.  We plan
    // the next action immediately and delay only its publication, allowing the
    // configured human-visible beat to overlap CPU search instead of adding to
    // it.  This is local to one continuous pass; a human-triggered later pass
    // starts with no artificial delay.
    let mut last_visible_apply: Option<Instant> = None;
    // A generous bound on chained paceable bot steps within a single hand (many
    // short per-action pauses can chain across a multi-trick run), so a wedged
    // state can never spin this task forever.
    for _ in 0..8192 {
        // Snapshot the current state with a read-only lock (NO version bump, NO
        // State republish), then classify the next bot step cheaply (no search).
        // This lets us avoid touching the WRITE lock — and thus avoid a spurious
        // State republish — whenever no bot has anything to do.
        let snapshot = match snapshot_state(ws_id, &room_name, backend_storage.clone()).await {
            Some(s) => s,
            None => break,
        };
        let work = {
            let game = InteractiveGame::new_from_state(snapshot.state.clone());
            classify_next_bot_work(&game, true).unwrap_or(NextBotWork::None)
        };

        match work {
            // Nothing for a bot to do (human's turn, parked, game over). Stop
            // WITHOUT writing, so we don't republish an unchanged State.
            NextBotWork::None => break,
            // The next step(s) are cheap un-paced burst steps (bot draws / reset).
            // Apply them all under ONE lock (one State publish for the whole
            // burst), then loop to re-classify.
            NextBotWork::Burst => {
                let burst_logger = logger.clone();
                let burst_made_progress = Arc::new(std::sync::Mutex::new(false));
                let progress_in_op = burst_made_progress.clone();
                execute_operation(
                    ws_id,
                    &room_name,
                    backend_storage.clone(),
                    move |game, _, _| {
                        let broadcasts =
                            shengji_core::bot::advance_bots_burst_unpaced(game, &burst_logger)?;
                        // The burst may legitimately produce no broadcasts (draws
                        // emit none) yet still apply moves; detect progress by
                        // whether the next work is no longer a burst.
                        if !matches!(
                            classify_next_bot_work(game, true).unwrap_or(NextBotWork::None),
                            NextBotWork::Burst
                        ) {
                            *progress_in_op.lock().unwrap() = true;
                        }
                        Ok(broadcasts
                            .into_iter()
                            .map(|(data, message)| GameMessage::Broadcast { data, message })
                            .collect())
                    },
                    "drive bots (burst)",
                )
                .await;
                // If the burst couldn't drain (e.g. a stale classification raced an
                // outside change), stop rather than spin republishing.
                if !*burst_made_progress.lock().unwrap() {
                    break;
                }
                continue;
            }
            // The next step is a paceable / trick-clear beat: fall through to the
            // off-lock compute + apply path below.
            NextBotWork::Paceable => {}
        }

        // Compute the next (paceable / trick-clear) bot step OFF the lock and OFF
        // the async runtime. The determinized search is CPU-bound with no await
        // points, so it must run on a blocking worker to avoid starving an async
        // worker thread.
        let wait_started = Instant::now();
        let search_permit = match bot_runtime.search_slots.clone().acquire_owned().await {
            Ok(permit) => permit,
            Err(_) => break,
        };
        let queue_ms = wait_started.elapsed().as_secs_f64() * 1000.0;
        if queue_ms >= 1.0 {
            debug!(logger, "Bot planner waited for global CPU slot"; "queue_ms" => queue_ms);
        }
        let plan_logger = logger.clone();
        let snapshot_version = snapshot.version;
        let snapshot_state = snapshot.state;
        let planned = tokio::task::spawn_blocking(move || {
            let game = InteractiveGame::new_from_state(snapshot_state);
            match plan_next_bot_action(&game, true) {
                Ok(step) => step,
                Err(e) => {
                    error!(plan_logger, "Failed to plan bot action"; "error" => format!("{e:?}"));
                    None
                }
            }
        })
        .await;
        drop(search_permit);

        let step: BotStep = match planned {
            Ok(Some(step)) => step,
            // No bot needs to act (human's turn, parked, game over), or the
            // blocking task itself failed to join.
            Ok(None) => break,
            Err(e) => {
                error!(logger, "Bot planning task panicked"; "error" => format!("{e:?}"));
                break;
            }
        };

        // Search has already run.  Wait only the REMAINDER of the visible beat
        // before publishing this step.  A trick-clear deadline is measured from
        // the last card appearing, not from the end of the next planning call.
        let not_before = match step.pause {
            Some(BotPause::TrickClear) => Some(
                last_visible_apply
                    .map(|t| t + bot_pause(BotPause::TrickClear))
                    .unwrap_or_else(|| Instant::now() + bot_pause(BotPause::TrickClear)),
            ),
            Some(BotPause::Action) => last_visible_apply.map(|t| t + bot_pause(BotPause::Action)),
            None => None,
        };
        if let Some(deadline) = not_before {
            let now = Instant::now();
            if deadline > now {
                tokio::time::sleep(deadline.duration_since(now)).await;
            }
        }

        // Apply the precomputed step under a brief lock, re-checking the world
        // hasn't moved on. This publishes the new state and broadcasts.
        let pause_after_apply = step.pause;
        let op_logger = logger.clone();
        // `apply_planned_bot_action` returns `None` when it DROPS the step (the
        // world changed since the snapshot); `Some(..)` when it applies (the
        // broadcast list may be empty for a kitty/exchange step that still mutates).
        let outcome = execute_operation_at_version(
            ws_id,
            &room_name,
            backend_storage.clone(),
            snapshot_version,
            move |game, _| match apply_planned_bot_action(game, &step, true, &op_logger)? {
                Some(broadcasts) => Ok(Some(
                    broadcasts
                        .into_iter()
                        .map(|(data, message)| GameMessage::Broadcast { data, message })
                        .collect(),
                )),
                None => Ok(None),
            },
            "drive bots (non-blocking)",
        )
        .await;

        // If the precomputed step was dropped by the re-check (the world changed
        // out from under us, e.g. a human finished the trick during the compute),
        // re-burst/re-snapshot and re-plan rather than pausing on a stale beat.
        match outcome {
            VersionedOperationOutcome::Applied => {}
            VersionedOperationOutcome::Stale => continue,
            VersionedOperationOutcome::Failed => break,
        }

        if matches!(pause_after_apply, Some(BotPause::Action)) {
            last_visible_apply = Some(Instant::now());
        } else if matches!(pause_after_apply, Some(BotPause::TrickClear)) {
            // The completed trick has now been cleared; the next lead does not
            // inherit the prior card's display deadline.
            last_visible_apply = None;
        }
    }
}

async fn user_disconnected<S: Storage<VersionedGame, E>, E: Send>(
    room: String,
    ws_id: usize,
    backend_storage: S,
    logger: slog::Logger,
    parent: u64,
) {
    execute_operation(
        ws_id,
        &room,
        backend_storage.clone(),
        move |_, _, associated_websockets| {
            for ws in associated_websockets.values_mut() {
                ws.retain(|w| *w != ws_id);
            }
            Ok(vec![])
        },
        "disconnect player",
    )
    .await;
    backend_storage
        .unsubscribe(room.as_bytes().to_vec(), ws_id)
        .await;
    info!(logger, "Websocket disconnected";
        "room" => room,
        "parent_span" => format!("{room}:{parent}"),
        "span" => format!("{room}:ws_{ws_id}")
    );
}

#[cfg(test)]
mod bot_runtime_tests {
    use super::BotRuntime;

    #[test]
    fn room_drives_are_singleflight_and_dirty_requests_are_not_lost() {
        let runtime = BotRuntime::new(2);
        let mut lease = runtime.request_drive("room").expect("first request starts");

        assert!(runtime.request_drive("room").is_none());
        assert!(runtime.request_drive("room").is_none());
        assert!(
            lease.continue_if_dirty(),
            "overlap requests require another pass"
        );
        assert!(
            !lease.continue_if_dirty(),
            "room is released after a clean pass"
        );

        assert!(runtime.request_drive("room").is_some());
    }

    #[test]
    fn dropping_a_driver_lease_releases_the_room() {
        let runtime = BotRuntime::new(1);
        let lease = runtime.request_drive("room").expect("first request starts");
        drop(lease);
        assert!(runtime.request_drive("room").is_some());
    }
}
