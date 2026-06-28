use std::sync::Arc;

use anyhow::bail;
use slog::{debug, error, info, o, Logger};
use tokio::sync::{mpsc, oneshot, Mutex};

use shengji_core::bot::BotPause;
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
    utils::{execute_immutable_operation, execute_operation},
    ZSTD_COMPRESSOR,
};

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
) {
    let _ = handle_user_connected(
        tx,
        rx,
        ws_id,
        logger,
        backend_storage,
        stats,
        resource_limits,
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

    run_game_for_player(
        logger.clone(),
        ws_id,
        player_id,
        room.clone(),
        name,
        backend_storage.clone(),
        rx,
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

async fn handle_user_action<S: Storage<VersionedGame, E> + 'static, E: Send + 'static>(
    logger: Logger,
    ws_id: usize,
    caller: PlayerID,
    room_name: &str,
    name: String,
    backend_storage: S,
    msg: UserMessage,
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
            // The bot driver may STOP early at a human-visible beat so a human can
            // follow the game one bot move at a time: either a bot just won a
            // now-complete trick (we DEFER clearing it so the full 4-card trick is
            // visible) or a bot just made a single meaningful move (play / bid /
            // kitty / landlord decision) after which we pause briefly. The closure
            // records WHICH beat it stopped on (if any); we read it after the lock
            // is released (and the just-produced state has been published).
            let pause_kind: Arc<std::sync::Mutex<Option<BotPause>>> =
                Arc::new(std::sync::Mutex::new(None));
            let pause_in_op = pause_kind.clone();
            let op_logger = logger.clone();
            execute_operation(
                ws_id,
                room_name,
                backend_storage.clone(),
                move |game, _, _| {
                    // Apply the human's action, then let any bot players that now
                    // need to act take their turns. Both happen within this single
                    // locked operation so the moves are applied atomically and the
                    // combined set of broadcasts is published together. We pass
                    // `defer_bot_trick_finish = true` so the bot driver pauses at
                    // the first human-visible beat (handled below) instead of
                    // bursting the whole sequence.
                    let mut broadcasts = game.interact(action, caller, &op_logger)?;
                    let result = shengji_core::bot::advance_bots(game, &op_logger, true)?;
                    if let Some(pause) = result.pause {
                        *pause_in_op.lock().unwrap() = Some(pause);
                    }
                    broadcasts.extend(result.messages);
                    Ok(broadcasts
                        .into_iter()
                        .map(|(data, message)| GameMessage::Broadcast { data, message })
                        .collect())
                },
                "handle user action",
            )
            .await;

            // If the bot driver paused at a human-visible beat, the just-produced
            // state has now been published (lock released). Spawn a detached task
            // that waits the appropriate beat WITHOUT holding the lock and then
            // resumes the bots (pausing again at each subsequent beat).
            //
            // When a BOT holds the standing bid after the deck is drawn, the driver
            // instead PARKS (no pause, no resume task) until every human seat clicks
            // "Done bidding". That click is an ordinary `Action::MarkBiddingDone`
            // which flows through this same branch and re-runs the bot driver — so
            // once the LAST human is done, the bot finalizes itself here with no
            // timer and no forced finalization needed. An all-bot table has zero
            // humans, so "all humans done" is trivially true and the bot proceeds
            // immediately (no deadlock).
            let initial_pause = *pause_kind.lock().unwrap();
            if initial_pause.is_some() {
                tokio::task::spawn(resume_deferred_bots_after_delay(
                    logger,
                    ws_id,
                    room_name.to_string(),
                    backend_storage,
                    initial_pause,
                ));
            }
        }
    }
    Ok(())
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

/// Detached task that resumes the deferred bot driver after a human-visible beat.
///
/// This is the second half of the production deferral (see
/// `shengji_core::bot::advance_bots` with `defer_bot_trick_finish = true`). It is
/// spawned ONLY after the just-produced state has already been published and the
/// game lock released, so the human sees that state during the pause. Every sleep
/// happens WITHOUT holding the lock.
///
/// It loops because the bot driver pauses at EACH beat: after one resume,
/// `finish_deferred_bot_trick` keeps driving the bots and may stop again at the
/// next beat (the next bot move, or the next bot-won trick), reporting WHICH beat
/// via [`AdvanceResult::pause`]. We sleep for the duration matching the CURRENT
/// pending beat, resume once, then continue with whatever beat the resume reports
/// next. The loop ends as soon as a resume reports no further beat (human's turn,
/// game over, no progress).
///
/// Safety: each resume re-derives, under the lock, what the bot driver wants to
/// do RIGHT NOW. If a human (or any other event) already finished the trick / it
/// is no longer a bot's turn, `finish_deferred_bot_trick` applies no stale move —
/// so there is no double-apply even if the world changed during the delay. The
/// iteration bound caps the number of chained beats within a single hand so a
/// wedged state can never spin this task forever.
///
/// Note: the draw-phase "a bot holds the standing bid, awaiting the humans"
/// situation is NOT handled here. The bot driver simply PARKS there until every
/// human clicks "Done bidding"; the last such click is an ordinary user action
/// that re-runs the driver and finalizes the bot. There is no timer and no forced
/// finalization, so this task only ever paces normal per-action / trick beats.
async fn resume_deferred_bots_after_delay<
    S: Storage<VersionedGame, E> + 'static,
    E: Send + 'static,
>(
    logger: Logger,
    ws_id: usize,
    room_name: String,
    backend_storage: S,
    initial_pause: Option<BotPause>,
) {
    // A generous bound on chained beats within a single hand (many short
    // per-action pauses can chain across a multi-trick run), so a wedged state
    // can never spin this task forever. `pending` carries the NEXT beat to sleep
    // on; the loop ends as soon as a resume reports no further beat.
    let mut pending = initial_pause;
    for _ in 0..4096 {
        let pause = match pending {
            Some(pause) => pause,
            None => break,
        };
        tokio::time::sleep(bot_pause(pause)).await;

        // The beat the NEXT resume stops on (if any), captured out of the
        // locked closure so we can pick the right sleep for the following loop.
        let next_pause: Arc<std::sync::Mutex<Option<BotPause>>> =
            Arc::new(std::sync::Mutex::new(None));
        let next_in_op = next_pause.clone();
        let op_logger = logger.clone();
        execute_operation(
            ws_id,
            &room_name,
            backend_storage.clone(),
            move |game, _, _| {
                // Re-check + apply under the lock. `finish_deferred_bot_trick`
                // is idempotent: it finishes a still-pending bot-won trick if
                // any, else advances the next single meaningful bot move, then
                // keeps deferring — reporting the next beat to wait on.
                let result = shengji_core::bot::finish_deferred_bot_trick(game, &op_logger)?;
                if let Some(pause) = result.pause {
                    *next_in_op.lock().unwrap() = Some(pause);
                }
                Ok(result
                    .messages
                    .into_iter()
                    .map(|(data, message)| GameMessage::Broadcast { data, message })
                    .collect())
            },
            "resume deferred bots",
        )
        .await;

        pending = *next_pause.lock().unwrap();
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
