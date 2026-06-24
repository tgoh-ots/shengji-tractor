# Security Notes (Milestone 4: Security Hardening)

This document records the server-side security hardening applied to the Shengji
backend (the in-memory Axum WebSocket/HTTP server in `backend/`), the chosen
defaults, and the **known remaining gaps** to track as follow-up work.

Scope of this milestone was the backend + configuration only. The frontend and
the bot policy (`core/src/bot/`) were intentionally not modified.

---

## 1. Protections added

### Origin validation on the WebSocket handshake (anti-CSWSH) — P0
- The CORS layer guards only `/api/rpc`; it does **not** protect the WebSocket
  upgrade. A new `axum::middleware::from_fn` middleware (`enforce_ws_origin` in
  `backend/src/main.rs`) runs **before** the `WebSocketUpgrade` extractor on the
  `GET /api` route and rejects the upgrade with **HTTP 403** when the browser
  `Origin` header is not in the allow-list.
- The allow-list is derived from the **same** `CORS_ALLOWED_ORIGINS` env var as
  CORS, so the two policies stay in sync. Default (when unset) is the localhost
  dev set: `http://localhost:3000,http://localhost:3030,http://127.0.0.1:3000,http://127.0.0.1:3030`.
- `CORS_ALLOWED_ORIGINS=*` allows any origin and logs a startup **warning**.
- A **missing** `Origin` header is allowed: native (non-browser) clients omit
  it, and the check exists to defend against cross-site requests *from browsers*
  (which always send `Origin` on WS upgrades). Rejecting missing-Origin would
  break legitimate non-browser clients without adding browser protection.
- Logic + unit tests: `backend/src/security.rs` (`OriginPolicy`). Integration
  tests asserting 403 on a disallowed origin and pass-through on an allowed
  origin: `backend/src/main.rs` (`ws_origin_tests`).

### Inbound message size cap — P0
- `WebSocketUpgrade::max_message_size(16 KiB)` rejects oversized frames at the
  protocol layer, plus a defense-in-depth length check in the receive loop that
  drops any frame `> MAX_INBOUND_MESSAGE_BYTES` before it is buffered/parsed.
- A legal move is at most a few hundred bytes; 16 KiB leaves headroom for the
  largest settings payloads. Constant: `security::MAX_INBOUND_MESSAGE_BYTES`.

### Per-connection inbound rate limit — P1
- A token-bucket limiter (`security::TokenBucket`) in the WS receive loop in
  `backend/src/main.rs`. Default: **40 msg/sec sustained, burst 60**
  (`WS_MSG_RATE_PER_SEC`, `WS_MSG_BURST`). A connection that exceeds its budget
  is **disconnected** (the receive loop breaks).

### Resource caps — P1
Enforced in `backend/src/shengji_handler.rs`:
- **Max players per room**: default **16** (`MAX_PLAYERS_PER_ROOM`). Enforced in
  `register_user` for new lobby (Initialize-phase) joins. A rejoin under an
  existing name is always allowed (it doesn't grow the room). This ceiling is
  well above any real Tractor table and keeps the bot self-play setup
  (host + 4 bots) within range.
- **Max observers per room**: default **16** (`MAX_OBSERVERS_PER_ROOM`).
  Enforced for new joins once the game has left the lobby.
- **Max total concurrent rooms**: default **1000** (`MAX_TOTAL_ROOMS`). A join
  that would create a brand-new room is rejected with a user-facing error when
  storage is already at capacity; joining an existing room is always allowed.
- **Room-creation rate limit (per connection)**: each WebSocket connection joins
  exactly one room for its lifetime, so per-connection room creation is naturally
  bounded to **1**. The meaningful global protection is `MAX_TOTAL_ROOMS` above.
  A per-**IP** room-creation rate limit is **not** implemented because the server
  is designed to sit behind a reverse proxy (nginx) and does not currently see
  the real client IP; see "Remaining gaps".

### Chat / name sanitization — P1
In `backend/src/shengji_handler.rs` + helpers in `backend/src/security.rs`:
- **Chat** (`UserMessage::Message`): control characters stripped, trimmed, and
  rejected if empty or `> 2048` bytes (`MAX_CHAT_MESSAGE_BYTES`). Oversized chat
  is rejected (not silently truncated) with a user-facing error.
- **Player name** (`JoinRoom`): control characters stripped + trimmed
  server-side; rejected if empty or `>= 32` bytes (`MAX_NAME_BYTES`, matching the
  pre-existing limit).
- **Room name** (`JoinRoom`): must be exactly 16 bytes (pre-existing
  `ROOM_NAME_BYTES`) and contain no control characters. It is treated as an
  opaque room key, so it is length/format-validated but not otherwise rewritten.
- The frontend renders these as text; this is the server-side enforcement layer.

### Security headers on HTTP responses — P1
`apply_security_headers` in `backend/src/main.rs` adds (via tower-http
`SetResponseHeaderLayer::if_not_present`, so route content-types are preserved):
- `Content-Security-Policy`: `default-src 'self'`; `connect-src 'self'` plus the
  configured `WEBSOCKET_HOST` origin (so the WS connection is allowed);
  `script-src 'self' 'wasm-unsafe-eval'` (the app loads its own JS and WASM);
  `style-src 'self' 'unsafe-inline'`; `img-src 'self' data:`; `font-src 'self'
  data:`; `object-src 'none'`; `base-uri 'self'`; `frame-ancestors 'none'`;
  `form-action 'self'`.
- `Strict-Transport-Security: max-age=63072000; includeSubDomains`
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Permissions-Policy`: disables accelerometer, camera, geolocation, gyroscope,
  magnetometer, microphone, payment, and usb.

### Move-validation tightening — P0
See "Move-validation assessment" below.

---

## 2. Idle / dead-connection reaping (verified)

- **Ping/pong**: the WS receive loop handles `Ping`/`Pong` frames (ignored at the
  application layer; the underlying tungstenite/axum stack answers protocol
  pings). When the peer closes or the socket errors, the receive loop breaks and
  `user_disconnected` unsubscribes the connection.
- **Storage pruning** (`storage/src/hash_map_storage.rs::prune`, called every
  60s by `periodically_dump_state`): removes rooms with no subscribers after
  1 hour and any room untouched for 2 hours, and drops closed subscriber
  channels. Disconnected sockets are also removed from a room's subscriber set
  on the next publish (`HashMapStorage::publish` retains only open senders).
- **Known limitation**: there is no server-initiated idle *timeout* that
  proactively closes a socket that stays open but silent (no app-level
  keepalive deadline). Dead TCP connections are reaped by the OS / proxy and by
  the pruning above, but a long-lived idle-but-open socket is not force-closed by
  the application. Tracked as a follow-up (see "Remaining gaps").

---

## 3. Move-validation assessment

The rules engine (`mechanics/src/trick.rs`, `core/src/game_state/play_phase.rs`)
already rejects the three unambiguous classes of illegal move, **before** any
state mutation, in the mutating `Trick::play_cards` path:

| Invariant | Status | Where |
|---|---|---|
| Acting **out of turn** | ENFORCED | `trick.rs` — `player_queue.front() != Some(id)` → `TrickError::OutOfOrder` |
| Playing cards you **don't hold** | ENFORCED (double-checked) | `trick.rs` `can_play_cards` → `Hands::contains`; again in `Hands::remove` |
| Wrong **card count** when following | ENFORCED | `TrickFormat::is_legal_play` size check |
| Acting in the **wrong phase** | ENFORCED | `InteractiveGame::interact` only matches valid `(Action, phase)` pairs; everything else `bail!("not supported in current phase")` |
| **Must-follow-suit** | ENFORCED (all policies) | `is_legal_play` led-suit count check |

Because these were already enforced, the safe tightening here was to **add
regression tests** (not to rewrite rules, which would risk breaking legal play /
the bot self-play test). Tests live in
`core/src/interactive.rs::move_validation_tests` and assert via
`InteractiveGame::interact` that (a) out-of-turn, (b) not-held, and (c)
wrong-phase actions are rejected.

### Remaining forced-play validation gaps (KNOWN — follow-up)

These are deliberate game-rule relaxations upstream; tightening them risks
rejecting legal moves and breaking the bot self-play test, so they were **not**
changed in this milestone:

1. **Followers are not forced to beat the current winner** ("must-beat-when-able"
   is not enforced for following players). This is standard Tractor behavior (a
   follower may always under-play), so it is correct *as a rule*, but it means
   the server does not stop a player from "wasting" a winning card.
2. **Format-following (must-play-pair/tractor-when-able) is policy-dependent.**
   Under the default `TrickDrawPolicy::NoProtections` it **is** enforced (you must
   play available pairs/tractors). Under `TrickDrawPolicy::NoFormatBasedDraw` the
   format requirement is **entirely disabled** — any same-suit cards of the right
   count are accepted. The `LongerTuplesProtected*` and `OnlyDrawTractorOnTractor`
   policies relax it partially by design.
3. **Throw ("bad throw") validation is throw-evaluation-policy dependent.** The
   leader's multi-unit throw is downgraded if any opponent can beat a unit
   (`ThrowEvaluationPolicy`), but the exhaustiveness of that check varies by
   policy (`All` / `Highest` / `TrickUnitLength`).

Recommendation: treat (1) as intended; (2)/(3) are configuration knobs whose
"loose" settings are opt-in. If a future milestone wants to guarantee strict
forced-play regardless of room settings, that is a rules-engine change in
`mechanics/src/trick.rs` and must be validated against the full
`mechanics`/`core` test suites and the bot self-play test.

---

## 4. Remaining gaps / follow-ups (not in scope this milestone)

- **Per-IP rate limiting / room-creation-per-IP**: requires propagating the real
  client IP from the reverse proxy (e.g. trusting `X-Forwarded-For` from a known
  proxy). Not wired up; would belong with a deployment-trust configuration.
- **Application-level idle timeout**: proactively close sockets that are open but
  silent past a deadline (see section 2).
- **Strict forced-play** regardless of room settings (see section 3, items 2/3).
- **Authentication**: there is no auth; any client that knows a room name can
  join it. This matches the upstream "players keep each other in check" design
  and is out of scope here.

---

## 5. Secrets / supply-chain hygiene

- **`Cargo.lock` is committed** (workspace root). This pins the exact dependency
  graph for reproducible, auditable builds; keep it committed.
- **No secrets in the repo.** `.env` and `*.key`/`*.pem`/`*.keystore` are
  gitignored; only `.env.example` (non-secret defaults) is tracked. Secrets
  (e.g. `SENTRY_DSN`, Upstash Redis tokens) belong in the host's secret store.
- **Dependency auditing** (run locally / in a later CI milestone — not set up
  here):
  - `cargo install cargo-audit && cargo audit` — scans `Cargo.lock` against the
    RustSec advisory DB for known-vulnerable crates.
  - `cargo install cargo-deny && cargo deny check` — license, advisory, and
    duplicate-/banned-dependency policy checks. An optional `deny.toml` can be
    added at the repo root to configure policy; it is intentionally left out of
    this milestone (config-only), but the recommended starting checks are
    `advisories`, `bans`, `licenses`, and `sources`.
  - Note: `cargo build` currently surfaces a future-incompatibility warning for
    the transitive `redis v0.23.3` crate (used only by the optional Redis
    storage backend). It is a dependency-side warning, not a `#![deny(warnings)]`
    failure in this codebase; bumping the Redis stack is a follow-up.
