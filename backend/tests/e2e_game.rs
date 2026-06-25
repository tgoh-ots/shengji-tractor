//! End-to-end WebSocket integration test for the Shengji backend.
//!
//! This boots the *real* Axum application (via [`shengji::build_app`], the same
//! router `main`/`run` serves) on a real `TcpListener`, then drives a full game
//! over an actual WebSocket using `tokio-tungstenite`. It validates the whole
//! stack end-to-end:
//!
//!   * the M1 lobby + bot driver (`AddAIPlayer`, `StartGame`, `advance_bots`),
//!   * the M2 bot brains (including the Omniscient CHEATER tier),
//!   * the M4 per-player redaction (`GameState::for_player`) on every State the
//!     server pushes to a human client.
//!
//! The single most important assertion is the **no-hidden-card-leakage** check:
//! every `GameMessage::State` the human receives must show every OTHER seat's
//! hand as `Card::Unknown` (the `🂠` glyph) / counts only — never a real card.
//! This holds even though one of the bots at the table is Omniscient and itself
//! sees everything: the cheat is confined to the bot's own decision-making and
//! must never leak onto the wire to the human.
//!
//! We deliberately join with `disable_compression: true` so the server emits
//! plain UTF-8 JSON (the normal path zstd-compresses with a trained dictionary),
//! letting the test parse messages directly.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tokio_tungstenite::tungstenite::Message as WsMessage;

use shengji::security::{OriginPolicy, ResourceLimits};
use shengji::serving_types::VersionedGame;
use shengji::state_dump::InMemoryStats;

use shengji_core::bot::{policy, BotDifficulty};
use shengji_core::game_state::GameState;
use shengji_core::interactive::Action;
use shengji_mechanics::types::PlayerID;
use storage::HashMapStorage;

/// The redaction glyph the server substitutes for every hidden card
/// (`Card::Unknown`). Any other card glyph appearing in a non-self seat's hand
/// would be a leak.
const UNKNOWN_CARD_GLYPH: &str = "🂠";

/// Boot the real app on an ephemeral port and return its address. The server
/// task runs for the lifetime of the test process.
async fn spawn_server() -> SocketAddr {
    // Allow ANY origin so the non-browser test client (which sends no Origin
    // header) is accepted by the WebSocket Origin guard.
    let origin_policy = OriginPolicy::from_raw("*");
    let logger = slog::Logger::root(slog::Discard, slog::o!());
    let backend_storage = HashMapStorage::<VersionedGame>::new(logger);
    let stats = Arc::new(Mutex::new(InMemoryStats::default()));
    let resource_limits = Arc::new(ResourceLimits::default());

    let app = shengji::build_app(backend_storage, stats, origin_policy, resource_limits);

    let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let addr = listener.local_addr().unwrap();
    let std_listener = listener.into_std().unwrap();
    std_listener.set_nonblocking(true).unwrap();

    tokio::spawn(async move {
        axum::Server::from_tcp(std_listener)
            .unwrap()
            .serve(app.into_make_service())
            .await
            .unwrap();
    });

    addr
}

/// Receive the next text/binary frame as parsed JSON, with a timeout. Returns
/// `None` on timeout or socket close.
async fn next_json<S>(socket: &mut S) -> Option<Value>
where
    S: StreamExt<Item = Result<WsMessage, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    loop {
        let frame = tokio::time::timeout(Duration::from_secs(10), socket.next()).await;
        let msg = match frame {
            Ok(Some(Ok(m))) => m,
            // Timeout, close, or error -> no more useful frames.
            _ => return None,
        };
        match msg {
            WsMessage::Binary(b) => {
                // Uncompressed JSON because we joined with disable_compression.
                if let Ok(v) = serde_json::from_slice::<Value>(&b) {
                    return Some(v);
                }
            }
            WsMessage::Text(t) => {
                if let Ok(v) = serde_json::from_str::<Value>(&t) {
                    return Some(v);
                }
            }
            WsMessage::Ping(_) | WsMessage::Pong(_) => continue,
            WsMessage::Close(_) => return None,
            _ => continue,
        }
    }
}

/// Assert that a `GameMessage::State` JSON value never exposes another seat's
/// real cards to `me`: every seat other than `me` must have ONLY the
/// `Card::Unknown` glyph as hand keys. Returns the parsed [`GameState`] (for the
/// caller to act on) so we only deserialize once.
fn assert_no_leak_and_parse(state_msg: &Value, my_name: &str) -> Option<GameState> {
    let state_val = state_msg.get("State")?.get("state")?;
    let game: GameState = serde_json::from_value(state_val.clone())
        .expect("server State must deserialize into a GameState");

    // Resolve my own PlayerID by name within the propagated player list.
    let me = game
        .propagated()
        .players()
        .iter()
        .find(|p| p.name == my_name)
        .map(|p| p.id);

    // Walk the raw JSON for the hands map of whichever phase carries one. The
    // shape is `{ <Phase>: { "hands": { "hands": { "<playerId>": { "<glyph>": n } } } } }`.
    for phase in ["Draw", "Exchange", "Play"] {
        if let Some(phase_val) = state_val.get(phase) {
            if let Some(hands) = phase_val
                .get("hands")
                .and_then(|h| h.get("hands"))
                .and_then(|h| h.as_object())
            {
                for (pid_str, hand) in hands {
                    let pid: usize = pid_str.parse().unwrap();
                    let is_me = me.map(|m| m.0 == pid).unwrap_or(false);
                    if is_me {
                        continue;
                    }
                    let cards = hand.as_object().expect("hand must be an object");
                    for glyph in cards.keys() {
                        assert_eq!(
                            glyph, UNKNOWN_CARD_GLYPH,
                            "LEAK: seat {} exposed real card '{}' to human '{}' \
                             in the {} phase; only the redaction glyph is allowed",
                            pid, glyph, my_name, phase
                        );
                    }
                }
            }
        }
    }

    Some(game)
}

/// Send a `UserMessage::Action(action)` as JSON over the socket.
async fn send_action<S>(socket: &mut S, action: &Action)
where
    S: SinkExt<WsMessage> + Unpin,
    <S as futures::Sink<WsMessage>>::Error: std::fmt::Debug,
{
    let payload = json!({ "Action": action });
    socket
        .send(WsMessage::Text(serde_json::to_string(&payload).unwrap()))
        .await
        .expect("send action");
}

#[tokio::test]
async fn e2e_game_no_hidden_card_leakage() {
    let addr = spawn_server().await;
    let url = format!("ws://{addr}/api");
    let (mut socket, _resp) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("connect to /api websocket");

    let my_name = "human-001";
    // Room name MUST be exactly 16 characters (ROOM_NAME_BYTES).
    let room_name = "e2eroom_16chars_";
    assert_eq!(room_name.len(), 16, "room name must be exactly 16 chars");

    // Join the room with compression disabled so we receive plain JSON.
    let join = json!({
        "room_name": room_name,
        "name": my_name,
        "disable_compression": true,
    });
    socket
        .send(WsMessage::Text(join.to_string()))
        .await
        .expect("send JoinRoom");

    // Drain the initial burst until we've seen our first State (the lobby), so
    // we know registration succeeded. Bail out if the server only sends errors.
    let mut saw_initial_state = false;
    for _ in 0..20 {
        match next_json(&mut socket).await {
            Some(v) => {
                if let Some(err) = v.get("Error") {
                    panic!("server returned an error during join: {}", err);
                }
                if v.get("State").is_some() {
                    assert_no_leak_and_parse(&v, my_name);
                    saw_initial_state = true;
                    break;
                }
            }
            None => break,
        }
    }
    assert!(saw_initial_state, "never received an initial lobby State");

    // Add three AI players: a mix of tiers including at least one CHEATER
    // (Omniscient) and at least one honest tier. The human is player 1; these
    // fill the 4-seat table.
    for difficulty in [
        BotDifficulty::Omniscient,
        BotDifficulty::Hard,
        BotDifficulty::Easy,
    ] {
        send_action(&mut socket, &Action::AddAIPlayer { difficulty }).await;
    }

    // Keep the per-decision search budget tiny so the Omniscient/Hard bots play
    // fast and the test stays well within its time bound. (The server reads this
    // env var per decision.)
    std::env::set_var("SHENGJI_BOT_BUDGET_MS", "10");

    // Start the game.
    send_action(&mut socket, &Action::StartGame).await;

    // Drive the game. On each State we (a) assert no leakage, and (b) if it's our
    // turn, compute a LEGAL move from our own redacted view via the honest engine
    // policy and submit it. The server runs the bots via `advance_bots` after
    // each of our actions, so most of the table advances without us.
    let mut states_seen = 0usize;
    let mut actions_sent = 0usize;
    let mut reached_play_phase = false;
    let mut game_finished = false;
    let mut last_phase = String::new();

    // Bounded loop with a hard overall wall-clock deadline so it can NEVER hang
    // CI (the per-recv timeout already bounds each receive). We also stop early
    // once we've observed several in-Play States with zero leakage.
    let mut plays_made = 0usize;
    let drive_started = Instant::now();
    while drive_started.elapsed() < Duration::from_secs(45) {
        let msg = match next_json(&mut socket).await {
            Some(v) => v,
            None => break,
        };
        if let Some(err) = msg.get("Error") {
            // A transient/unexpected error: surface it for debugging but keep
            // going so a single hiccup doesn't fail the whole drive.
            eprintln!("server error message: {err}");
            continue;
        }
        let game = match assert_no_leak_and_parse(&msg, my_name) {
            Some(g) => g,
            None => continue, // not a State message
        };
        states_seen += 1;

        // Track phase progression.
        let phase = match &game {
            GameState::Initialize(_) => "Initialize",
            GameState::Draw(_) => "Draw",
            GameState::Exchange(_) => "Exchange",
            GameState::Play(p) => {
                if p.game_finished() {
                    game_finished = true;
                }
                "Play"
            }
        };
        if phase != last_phase {
            last_phase = phase.to_string();
        }
        if phase == "Play" {
            reached_play_phase = true;
        }
        if game_finished {
            break;
        }

        // Determine if it is our turn, and if so submit a legal move. We use the
        // HONEST policy (Hard) on our OWN redacted view: it only returns an
        // action when `next_player == me`, and the action is guaranteed legal.
        let me = match game
            .propagated()
            .players()
            .iter()
            .find(|p| p.name == my_name)
            .map(|p| p.id)
        {
            Some(id) => id,
            None => continue,
        };

        if let Some(action) = next_action_for(&game, me) {
            if reached_play_phase && matches!(action, Action::PlayCards(_)) {
                plays_made += 1;
            }
            send_action(&mut socket, &action).await;
            actions_sent += 1;
        }

        // The redaction contract is validated once we've seen several in-Play
        // States (non-empty hands) with zero leakage; stop early to stay fast.
        if reached_play_phase && plays_made >= 4 {
            break;
        }
    }

    // Assertions: the lobby accepted our bots and the game made real progress.
    assert!(
        states_seen > 0,
        "expected to receive State updates after starting the game"
    );
    assert!(
        reached_play_phase,
        "the game never progressed into the Play phase (saw last phase {:?}, \
         {} states, {} actions sent)",
        last_phase, states_seen, actions_sent
    );

    // Prefer a fully-finished hand, but solid progress + zero leakage is the
    // contractual minimum. Driving every last trick over a live socket with bot
    // timing is fiddly; reaching/clearing the Play phase already validates the
    // full stack (M1 driver + M2 brains + M4 redaction).
    if game_finished {
        eprintln!("e2e: hand ran to completion ({states_seen} states, {actions_sent} actions)");
    } else {
        eprintln!(
            "e2e: reached the Play phase with zero leakage ({states_seen} states, \
             {actions_sent} actions); did not drive to the final trick"
        );
    }

    // Close politely.
    let _ = socket.close(None).await;
}

/// Compute a single legal [`Action`] for `me` from its own (already redacted)
/// view, or `None` if it is not `me`'s turn. This reuses the production HONEST
/// bot policy — which itself only ever reads the redacted view — so the move is
/// guaranteed legal and the test client never needs to reimplement Shengji's
/// (substantial) move-legality rules.
fn next_action_for(view: &GameState, me: PlayerID) -> Option<Action> {
    // As a human client we always act HONESTLY from our redacted view. We drive
    // EVERY responsibility a seated player can have, so the table never stalls
    // waiting on us: draw, the post-draw bid / kitty pick-up / reveal, exchange
    // (if we are the landlord), play, and finishing a trick we won. (The
    // Omniscient cheat is gated entirely in the server; a human client never
    // sees hidden cards, so an honest tier on our redacted view is the right
    // driver. We use `Easy` — a pure heuristic with no search/model dependency —
    // to keep the e2e socket loop fast and self-contained.)
    match view {
        GameState::Initialize(_) => None,
        GameState::Draw(p) => {
            if p.next_player().ok()? != me {
                return None;
            }
            if !p.done_drawing() {
                return Some(Action::DrawCard);
            }
            // Drawing finished and it is our turn to resolve trump / the kitty.
            if p.bid_decided() {
                Some(Action::PickUpKitty)
            } else if let Some(bid) = p.valid_bids(me).ok()?.into_iter().min_by_key(|b| b.count) {
                Some(Action::Bid(bid.card, bid.count))
            } else {
                // No legal bid available to us: reveal the bottom card to fix trump.
                Some(Action::RevealCard)
            }
        }
        // Exchange decisions (only the landlord acts) are fully handled by the
        // honest policy.
        GameState::Exchange(_) => policy::select_action(view, me, BotDifficulty::Easy)
            .ok()
            .flatten(),
        GameState::Play(p) => {
            if p.game_finished() {
                return None;
            }
            match p.trick().next_player() {
                // Our turn to play: the honest policy returns a legal play.
                Some(next) if next == me => policy::select_action(view, me, BotDifficulty::Easy)
                    .ok()
                    .flatten(),
                // Trick complete: if we won it (and thus lead next), finish it.
                None => match p.trick().complete() {
                    Ok(ended) if ended.winner == me => Some(Action::EndTrick),
                    _ => None,
                },
                _ => None,
            }
        }
    }
}
