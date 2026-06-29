# Code Audit — shengji-tractor

**Date:** 2026-06-29
**Scope:** Full repository (`backend/`, `core/` incl. bots, `mechanics/`, `storage/`, `frontend/` incl. wasm bindings, build/deploy/CI/training tooling).
**Method:** Multi-agent find-and-verify. Per-area auditors plus two cross-cutting passes (security deep-dive, dead-code-via-tooling). Every finding below was independently re-checked against the source; one finding was rejected during verification, several had severity/category adjusted. Cross-cutting and module findings on the same root issue have been merged (source ids listed per entry).

> 142 agents, ~119 distinct verified findings from 130 source ids. Counts by severity: **critical 1, high 4, medium ~12, low ~102**.

---

## Remediation status (updated 2026-06-29, shipped to production)

Two commits landed on `master` and were deployed to https://shengji-tractor.fly.dev
(verified live: `/full_state.json` now returns 404, lobby/app 200).

**Fixed — commit `fix: security + reliability act-first set` (act-first):**
- ✅ #1 critical `/full_state.json` leak — route removed; on-disk recovery dump
  still runs via the 60s `periodically_dump_state` timer; regression test added
  (`backend/tests/e2e_game.rs::full_state_json_route_is_not_exposed`).
- ✅ #2 `jack_variation` save/load data-loss bug.
- ✅ #3 `check_jacks_last_trick` panic guard.
- ✅ #4 zstd `FrameDecoder` poisoning (rebuild-on-error).
- ✅ #6 binary-WebSocket decode `try/catch` (folded in — completes #4).

**Fixed — commit `chore: hygiene sweep` (clippy gate + dead code/deps + toolchain):**
clippy gate restored (0 warnings/errors); dead code (`should_bid`,
`Suit::unicode_offset`, `_get_cards` gated) & unused deps removed
(`lazy_static`, `axum-macros`, `@sentry/tracing`, `hook-shell-script…`,
`tempdir`→`tempfile`); `rust-toolchain.toml` pins rustc 1.92.0 (#9); prettier
conflict resolved (`.prettierrc` deleted; codebase is on prettier defaults);
dependabot cargo+npm; `tsconfig` skipLibCheck; `release.sh` shebang; `substr`,
`rel="noreferrer"`. **Note:** `Action::Beep` (core-gamestate-2) was flagged dead
but is LIVE (BeepButton sends it) — intentionally kept.

**Accepted risk (will NOT fix):**
- ⏸️ #5 name-based seat takeover — accepted for a casual site (any name-only
  heuristic also blocks the legitimate owner / breaks fast-refresh reconnect; the
  real fix is a per-seat token, a wire-protocol change).

**Deferred (open follow-ups):**
- ⬜ #7 unauthenticated `/api/rpc` + unbounded trick-matcher CPU DoS (medium).
- ⬜ #8 frontend render-tree index-crash cluster (medium).
- ⬜ ~102 low-severity items below (cleanliness/best-practice tail), as desired.

---

## Top priorities

1. **🟥 CRITICAL — Unauthenticated `/full_state.json` leaks every player's hidden hand** *(Security; ids: backend-1, security-deep-1)*
   - **Files:** `backend/src/lib.rs:142,344-376`, `backend/src/state_dump.rs:102-185`
   - **Why:** The route returns the raw `GameState` (all hands, kitty, deck) for every active room to any anonymous HTTP caller, defeating the entire honesty/redaction model the project is built around. Real-time, systematic cheating in any game with one `curl`. CORS does not protect it (it is not an access control; non-browser clients get the full body). Each hit also forces prune + full serialize + fsync — a DoS amplifier on the single 256 MB machine.
   - **Fix (small):** Remove the route from the public router (keep `dump_state` as an internal function for the on-disk recovery dump only), or gate behind an env bearer token + localhost bind. Add a regression test mirroring `e2e_game_no_hidden_card_leakage`. Move the prune+disk-write side effects out of the HTTP handler.

2. **🟧 HIGH — Saved game settings silently drop `jack_variation` (data-loss bug)** *(Correctness; ids: frontend-core-4, frontend-core-5)*
   - **File:** `frontend/src/Initialize.tsx:1117-1119` (within the 228-line `setGameSettings` switch, 948-1179)
   - **Why:** `saveGameSettings` serializes the propagated key `jack_variation`, but the load switch matches `case "set_jack_variation"`, which can never equal it — so a saved non-default Jack Variation is never restored. High-impact, trivial fix, user-visible.
   - **Fix (trivial→medium):** Change the case to `"jack_variation"`; better, replace the entire switch with a declarative `{propagatedKey: actionName}` map iterated over `Object.entries`, and add a save/load round-trip test.

3. **🟧 HIGH — `check_jacks_last_trick` unguarded `.unwrap()` chain can panic the shared room** *(Correctness; id: core-gamestate-1)*
   - **File:** `core/src/game_state/play_phase.rs:613-642`
   - **Why:** Reachable from `EndGameEarly` → `StartNewGame` when no trick was played and `SingleJack` variation is active with landlord on Jack: `last_trick.unwrap()` panics, killing the server process. Untrusted client message sequence can crash the single shared machine.
   - **Fix (small):** `let Some(last_trick) = last_trick else { return false; };` and replace the remaining unwraps with early-returns.

4. **🟧 HIGH — Poisoned zstd `FrameDecoder` bricks the client session on one bad frame** *(Correctness; id: wasm-bindings-1)*
   - **File:** `frontend/shengji-wasm/src/lib.rs:108-125`
   - **Why:** `frame_decoder.take().unwrap()` removes the decoder; it is only restored on the success path. Any decode error leaves the cell `None`, so every subsequent `zstd_decompress` panics (aborting the WASM instance) — and this is the sole path for live binary game-state messages. One corrupt frame permanently disables the client.
   - **Fix (small):** Restore the decoder on all paths (reset on error / always write back `decoder.inner()`). Add a test: garbage frame, then a valid frame still decodes.

5. **🟧 HIGH — Name-based identity allows seat takeover + hidden-hand disclosure** *(Security; id: security-deep-2)*
   - **Files:** `core/src/game_state/mod.rs:96-105`, `backend/src/shengji_handler.rs:290-346`
   - **Why:** Identity is the display-name string. Anyone who knows the 16-char room name + a victim's name reclaims the seat, receives that seat's real hand, and forcibly disconnects the legitimate player. No per-session secret. Public rooms are enumerable via `/public_games.json` (security-deep-8), lowering the bar further.
   - **Fix (medium):** Issue a server-generated per-seat token on first join; require it to reclaim the seat; persist alongside room/name in localStorage. At minimum, don't silently disconnect a live session on a name collision.

6. **🟨 MEDIUM — Binary WebSocket decode path has no error handling** *(Correctness; ids: frontend-core-1, wasm-bindings-2)*
   - **File:** `frontend/src/WebsocketProvider.tsx:248-276`
   - **Why:** The text branch has try/catch; the binary branch (`decodeWireFormat` → `zstd_decompress` + `JSON.parse`, both can throw) does not. A throw inside the FileReader/queue callback can stall the message pump for pending items (and, via #4, panic). Pairs with #4: together they make a single bad frame survivable.
   - **Fix (trivial):** Wrap `f` in try/catch mirroring the text branch; use try/finally in the queue pump so the next item always drains.

7. **🟥 MEDIUM — Trick-format matcher / RPC compute are unbounded CPU on the shared machine** *(Security/Performance; ids: mechanics-2, backend-3 + security-deep-7)*
   - **Files:** `mechanics/src/format_match.rs:8-24,120-312`, `backend/src/wasm_rpc_handler.rs:6-58`, `backend/src/lib.rs:137`
   - **Why:** `find_format_matches` keeps an unbounded `visited` set and can enumerate tens of thousands of states on large single-suit plays; `compute_winner` re-runs the full matcher per follower. The unauthenticated `/api/rpc` runs this combinatorial work synchronously on the async runtime with no rate limit and no body-size cap beyond axum's ~2 MB default. A flood can pin worker threads on the single machine.
   - **Fix (medium):** Cap single-play / single-suit segment size early in `can_play_cards`; bound the matcher's visited-set/results; add `DefaultBodyLimit` + a per-IP rate limit and `spawn_blocking` (or a concurrency/timeout layer) to `/api/rpc`.

8. **🟨 MEDIUM — Frontend render-tree index crashes on malformed/transient derived state** *(Correctness; ids: frontend-ui-1, -2, -3, -4, -11)*
   - **Files:** `InlineCard.tsx:63-82`, `Trump.tsx:17-19`, `Exchange.tsx:144-163`, `Draw.tsx:262-298`, `Points.tsx:187-204`
   - **Why:** Several components dereference `[0]`/`[idx]`/`map[key]` with no presence guard; an unexpected card glyph or a missing player id throws during render and (via the Sentry error boundary) blanks the whole game UI for that client. `InlineCard` is the most exposed (renders wire-sourced card strings widely).
   - **Fix (small, bundled):** Make `unicodeToCard`/`InlineCard` return a neutral placeholder on unknown glyphs; use `.find(...)` + guards mirroring the safe `StatusRail`/trump-construction patterns already in the codebase.

---

## Security

| id(s) | title | sev | conf | file:lines |
|---|---|---|---|---|
| backend-1, security-deep-1 | Unauthenticated `/full_state.json` dumps every room's un-redacted hands | 🟥 critical | high | `backend/src/lib.rs:142,344-376`; `state_dump.rs:102-185` |
| security-deep-2 | Name-based identity → seat takeover + hand disclosure | 🟧 high | high | `core/src/game_state/mod.rs:96-105`; `shengji_handler.rs:290-346` |
| backend-2, security-deep-3 | No per-actor authorization on kick / chat-link / settings | 🟨 medium | high | `core/src/interactive.rs:52-59,491-507`; `mod.rs:114-130` |
| backend-3, security-deep-7 | Unauthenticated `/api/rpc` runs combinatorial work, no rate limit | 🟨 medium | medium | `wasm_rpc_handler.rs:6-58`; `lib.rs:137` |
| build-deps-config-7 | Legacy root `Dockerfile` builds `FROM …:latest` (supply-chain) | 🟨 medium | high | `Dockerfile:3` |
| security-deep-8 | `/public_games.json` enumeration eases room discovery (compounds takeover) | 🟦 low | medium | `state_dump.rs:187-213`; `lib.rs:150` |
| security-deep-6 | Periodic on-disk dump persists full un-redacted state in world-readable `/tmp` | 🟦 low | medium | `state_dump.rs:102-185`; `utils.rs:30-40` |
| build-deps-config-4, security-deep-5 | Hardcoded **upstream** Sentry DSN ships client errors to a third party | 🟦 low | high | `frontend/src/index.tsx:21-34` |
| build-deps-config-10 | `cargo audit` / `gitleaks` are advisory-only — no merge-blocking security gate | 🟦 low | medium | `.github/workflows/ci.yml:84-119` |
| backend-9 | `runtime.js` interpolates env-controlled `WEBSOCKET_HOST`/`VERSION` into JS unescaped | 🟦 low | low | `backend/src/lib.rs:104-119` |

**Notes.** `security-deep-3` subsumes `backend-2` (kick) and broadens to chat-link/settings; `ResetGame` was verified to require two distinct players' consent, so it is *not* a single-actor griefing vector (that claim was corrected). The Sentry DSN (security-deep-5) is mitigated in production because the CSP `connect-src` does not allow-list `ingest.sentry.io`, so reports silently fail — making the integration both privacy-leaky-by-intent and dead.

**Recommendations (effort):**
- backend-1: **small** — remove/gate the route; move side effects off the handler; add regression test.
- security-deep-2: **medium** — per-seat token issued on join, required on reclaim.
- backend-2/security-deep-3: **medium** — minimal host/landlord authorization model; restrict kick-of-others and lobby `Set*` actions; rate-limit reset.
- backend-3/security-deep-7: **medium** — body limit + input card-count cap + per-IP rate limit + `spawn_blocking`.
- build-deps-config-7: **small** — pin to a digest or republish under this fork; or delete the legacy Dockerfile + `build-image/`.
- security-deep-8: **small** — acceptable once seats are authenticated; otherwise prefer join-by-id token over exposing room names.
- security-deep-6: **small** — `0600` perms, non-`/tmp` dir; reconsider persisting hand data at all.
- build-deps-config-4/security-deep-5: **small/trivial** — source DSN from a build env var defaulting to empty, or drop Sentry init.
- build-deps-config-10: **small** — hard-fail gitleaks on PRs; fail `cargo audit` on high/critical with an explicit ignore-list.
- backend-9: **trivial** — `serde_json::to_string` the values before embedding.

---

## Correctness

| id(s) | title | sev | conf | file:lines |
|---|---|---|---|---|
| frontend-core-4, frontend-core-5 | Saved `jack_variation` never restored (228-line `setGameSettings` switch) | 🟧 high | high | `Initialize.tsx:948-1179,1117-1119` |
| core-gamestate-1 | `check_jacks_last_trick` unwrap chain panics the room | 🟧 high | medium | `play_phase.rs:613-642` |
| wasm-bindings-1 | Poisoned zstd `FrameDecoder` bricks the client session | 🟧 high | high | `shengji-wasm/src/lib.rs:108-125` |
| frontend-core-1, wasm-bindings-2 | Binary WS decode path has no try/catch (stalls pump) | 🟨 medium | high | `WebsocketProvider.tsx:248-276` |
| frontend-ui-1 | `InlineCard` → `unicodeToCard` throws on unknown glyph, blanks UI | 🟨 medium | high | `InlineCard.tsx:63-82` |
| frontend-ui-2 | `Trump.tsx` indexes `[0].value` on a filterable-empty array | 🟦 low | high | `Trump.tsx:17-19` |
| frontend-ui-3 | `Exchange.tsx` dereferences `players[-1]` when landlord/exchanger not seated | 🟦 low | medium | `Exchange.tsx:144-163` |
| frontend-ui-4 | `Draw.tsx` reads `players[landlord].level` without presence guard | 🟦 low | medium | `Draw.tsx:262-298` |
| frontend-ui-9 | Autodraw-speed select values mismatch labels and the real `700ms` default | 🟦 low | medium | `SettingsPane.tsx:185-199`; `Draw.tsx:113` |
| frontend-ui-11 | `Points.tsx` indexes `points[id].length` without guard (asymmetric vs penalties) | 🟦 low | medium | `Points.tsx:187-204` |
| frontend-ui-17 | Beeper effect omits deps (`beeper`/`interval`) with empty array (latent stale closure) | 🟦 low | medium | `Beeper.tsx:15-19` |
| frontend-core-9 | Effect resets `selected` using stale `trump`/`tractor_requirements` not in deps | 🟦 low | medium | `Play.tsx:154-184` |
| frontend-core-10 | `send()` 5s disconnect timer relies on a render-deferred ref (leaked-timer race) | 🟦 low | medium | `WebsocketProvider.tsx:293-308` |
| frontend-core-8 | `gameFinishedHandler` uses `name in result` (prototype-chain match corrupts local stats) | 🟦 low | medium | `websocketHandler.ts:141-169` |
| bots-heuristics-4 | Follow scorer never credits beating a trump-led trick with higher trump | 🟦 low | medium | `heuristics.rs:1306-1318` |
| bots-search-7 | PUCT uses `0.0` (not `-inf`) as value of never-visited candidates | 🟦 low | medium | `search.rs:373-397` |
| mechanics-1 | Negative `step_adjustments` → misleading validation (defended downstream) | 🟦 low | high | `scoring.rs:98-118` |
| core-gamestate-6 | `KittyPenalty::Power` uses `2usize.pow(unit_size)` with no overflow guard | 🟦 low | low | `play_phase.rs:353-356` |
| mechanics-6 | `Player::new` defaults `metalevel: 1` while test fixtures use `0` | 🟦 low | medium | `player.rs:14-50` |
| mechanics-12 | Bidding joker policy `num_decks - 1` underflows if `num_decks == 0` (WASM RPC path) | 🟦 low | medium | `bidding.rs:130-132` |
| backend-6 | Join-loop error replies ignore `disable_compression` (undecodable error for opt-out clients) | 🟦 low | medium | `shengji_handler.rs:95-124` |
| backend-10 | `load_dump_file` fixes `monotonic_id=1` (benign under HashMap backend) | 🟦 low | low | `state_dump.rs:50-59` |

**Notes.** `frontend-core-5` is the specific sub-bug of `frontend-core-4`; both are folded into Top Priority #2. `mechanics-1` was adjusted: the huge-usize cast is real but the malicious value is *rejected* downstream by the segment-geometry check, so it is a misleading-validation/cleanliness issue, not an exploit. `core-gamestate-9` (HashSet-randomized friend order) and `core-gamestate-8` (`set_done_bidding` no membership check) were reclassified to Best practices (no gameplay-correctness impact).

**Recommendations (effort):** mostly **trivial→small** — add presence guards/`?.`/`.find()` (the ui-* cluster, frontend-core-8/9, Points/Trump/Exchange/Draw); `let Some(..) else { return }` (core-gamestate-1); restore-on-all-paths (wasm-bindings-1); try/catch the binary branch (frontend-core-1); `checked_pow`/cap (core-gamestate-6); `inner_count + 1 < num_decks` (mechanics-12); validate `step_size >= 5` + fix the message (mechanics-1). bots-heuristics-4 / bots-search-7 are **small** and should be validated via `paired_eval` (bots-heuristics-4 touches the live NEW scorer; bots-search-7 is behind the off-by-default PUCT flag).

---

## Performance

| id(s) | title | sev | conf | file:lines |
|---|---|---|---|---|
| build-deps-config-16 | `Dockerfile.deploy` does cold Rust build every deploy (no dep caching) | 🟨 medium | medium | `Dockerfile.deploy:77-78` |
| bots-heuristics-2 | Candidate sort/dedup uses `format!("{:?}")` per comparison in the search hot path | 🟨 medium | high | `heuristics.rs:420-425,546-550` |
| backend-4 | Unbounded outbound mpsc channel → slow-reader memory growth | 🟦 low | medium | `backend/src/lib.rs:420-429` |
| backend-11 | Per-State redaction re-clones the full game per subscriber per update | 🟦 low | medium | `shengji_handler.rs:231-243` |
| bots-search-4 | Search hot loop clones the heavyweight `PlayPhase` per candidate-per-world | 🟦 low | medium | `search.rs:648,211-249,729-787` |
| bots-heuristics-1 | Search rebuilds `Knowledge` per candidate-scoring call | 🟦 low | medium | `heuristics.rs:1481-1515,311-383` |
| bots-search-6 | ONNX inference clones the feature batch (`flat.to_vec()`) per call | 🟦 low | high | `expert.rs:243` |
| bots-heuristics-7 | Determinizer first-pass deal is O(pool × seats) | 🟦 low | low | `determinize.rs:306-329` |
| bots-heuristics-8 | `boss_strength` recomputed multiple times per candidate | 🟦 low | low | `heuristics.rs:717-735,1292-1305` |
| mechanics-9 | `Hands` custom `Deserialize` round-trips through `serde_json::Value` | 🟦 low | medium | `hands.rs:30-67` |
| mechanics-10 | Per-follower clones + re-enumeration in `compute_winner` / `find_tractors_from_start` | 🟦 low | medium | `trick.rs:1092-1118,1448-1492` |

**Notes.** `bots-search-4` and `bots-heuristics-1` were adjusted down from medium: the search runs under a hard wall-clock budget that is **not** currently the binding constraint (the worlds cap is), so cutting clone/`Knowledge` cost would not directly buy more worlds — confirm with `budget_benchmark` before optimizing. `bots-heuristics-2` is the highest-value bot perf fix (it's a per-comparison heap alloc and the cheaper `as_char()` idiom already exists in `dedup_card_sets`).

**Recommendations (effort):** bots-heuristics-2 **small** (sort by card key, not Debug string); bots-search-6 **trivial** (take `Vec<f32>` by value); build-deps-config-16 **medium** (cargo-chef / manifests-first COPY); backend-4 **medium** (bounded/coalescing outbound channel); backend-11 / bots-search-4 / bots-heuristics-1 **medium→large**, low urgency at 4-player scale.

---

## Dead code

| id(s) | title | sev | conf | file:lines |
|---|---|---|---|---|
| build-deps-config-5 | `redis` stack always compiled into the deploy binary, never instantiated | 🟨 medium | high | `storage/Cargo.toml:14-18` |
| bots-search-1 | Full self-play driver duplicated verbatim (`eval.rs`, `budget_benchmark.rs`) vs shared harness | 🟨 medium | high | `core/examples/eval.rs:70-280`; `budget_benchmark.rs:174-294` |
| backend-5 | `RedisStorage::prune` is entirely commented out (no-op) | 🟦 low | high | `storage/src/redis_storage.rs:300-323` |
| core-gamestate-2 | `Action::Beep` variant is dead (never dispatched) | 🟦 low | high | `core/src/interactive.rs:581` |
| bots-heuristics-3 | `should_bid` is never called | 🟦 low | high | `heuristics.rs:1759-1766` |
| bots-search-11 | `SearchConfig::default()` exists only as docs; 3 copies of the budget numbers | 🟦 low | medium | `search.rs:111-129` |
| mechanics-3 | `Suit::unicode_offset` never used | 🟦 low | high | `mechanics/src/types.rs:764-771` |
| mechanics-5 | `Hands::_get_cards` is test-only but `pub` | 🟦 low | high | `mechanics/src/hands.rs:149-160` |
| frontend-ui-8 | Dead i18n keys (`team.declarer`, `rail.level`, `term.level`, …) never referenced | 🟦 low | high | `i18n.tsx:127-141` |
| build-deps-config-3, deadcode-tooling-4 | Unused `@sentry/tracing` v7 alongside `@sentry/react` v9 | 🟦 low | high | `frontend/package.json:8-9` |
| build-deps-config-8 | Unused devDep `hook-shell-script-webpack-plugin` | 🟦 low | high | `frontend/package.json:60` |
| build-deps-config-9 | Unused dep `axum-macros` | 🟦 low | high | `backend/Cargo.toml:16` |
| deadcode-tooling-3 | Unused dep `lazy_static` in `core/Cargo.toml` | 🟦 low | high | `core/Cargo.toml:14` |
| deadcode-tooling-6 | Stale TODO: "remove default deserialization … in a few days" (3+ months) | 🟦 low | high | `mechanics/src/trick.rs:602` |

**Notes.** `build-deps-config-5` was reclassified from Performance to Dead code (compiled-but-never-instantiated). `bots-search-1` and the redis prune (`backend-5`) overlap conceptually with the redis dead-weight item — kept distinct because they are different files/fixes. `bots-search-1` carries a latent honesty-invariant risk (a third copy of the redaction branch can drift); see Best practices.

**Recommendations (effort):** mostly **trivial** deletions/feature-gating. build-deps-config-5 **small** (put `redis` + `RedisStorage` behind an off-by-default cargo feature). bots-search-1 **medium** (refactor onto `harness::play_one_hand`/`Seat`). bots-search-11 **small** (single source of truth for the budget constants).

---

## Code cleanliness

| id(s) | title | sev | conf | file:lines |
|---|---|---|---|---|
| bots-heuristics-6 | `heuristics.rs` is a 3245-line module mixing scoring/bidding/kitty/tests | 🟦 low | medium | `heuristics.rs:1-3245` |
| frontend-core-15 | `Play` (≈530 lines) & `Initialize` (1586) monoliths with inline async-effect orchestration | 🟦 low | medium | `Play.tsx:54-581` |
| mechanics-11 | Oversized `play_cards` (~260 lines) mixes legality/throw-validation/mutation | 🟦 low | medium | `trick.rs:707-967` |
| bots-search-2 | Near-identical `play_one_hand_ab`/`GameOutcome`/`run_match` copy-pasted across 5 examples | 🟦 low | high | `expert_ab.rs`/`tournament.rs`/`easy_ab_benchmark.rs`/`enoch_benchmark.rs`/`heuristic_benchmark.rs` |
| bots-search-3 | Third full game-driver copy + honesty-bypass branch in `tests.rs` | 🟦 low | high | `core/src/bot/tests.rs:271-409` |
| bots-search-10 | Legacy `simulate_play.rs` (unwrap-heavy, undocumented dict.zstd regen tool) | 🟦 low | medium | `core/examples/simulate_play.rs:1-271` |
| bots-heuristics-5 | Duplicated ladder/pair-structure detection (4 copies of the 13-`Number` array) | 🟦 low | high | `heuristics.rs:1806-1893` |
| mechanics-4 | Duplicated `ALL_SUITS` trump-number successor block in `Trump::successor` | 🟦 low | high | `types.rs:178-244` |
| frontend-ui-16 | `CardInfo` fallback object hand-copied 7× with `effective_suit: 'Unknown' as any` | 🟦 low | high | `Card.tsx:74-209`; `cachePrefill.ts:147-155` |
| frontend-core-6 | `getCardsFromHand` half-finished stub (`suit: null as any`, unsorted) | 🟦 low | medium | `Play.tsx:321-340` |
| core-gamestate-3 | Duplicated match arms in `compute_player_level_deltas` | 🟦 low | high | `play_phase.rs:543-562` |
| core-gamestate-4 | Redundant duplicate landlord-level lookup in `reveal_card` | 🟦 low | high | `draw_phase.rs:168-190` |
| core-gamestate-7 | Misspelled `kitty_multipler` | 🟦 low | high | `play_phase.rs:353-443` |
| bots-search-8 | `easy_play` softmax max-fold seeds `f64::MIN` instead of `NEG_INFINITY` | 🟦 low | high | `harness.rs:239` |
| frontend-core-11 | Room-id `length === 16` magic number duplicated across 3 files | 🟦 low | high | `WebsocketProvider.tsx:155,205`; `AppStateProvider.tsx`; `Root.tsx` |
| frontend-core-12 | RPC host resolution: dead `!== undefined` check + asymmetric empty-string handling | 🟦 low | medium | `WasmOrRpcProvider.tsx:64-74` |
| frontend-ui-12 | `BidArea` unused outer `trump` + `any`-typed shadowing inner `trump` | 🟦 low | high | `BidArea.tsx:51-164` |
| frontend-ui-18 | `preventDefault()` on `<select>`/`<input>` `onChange` (no-op) | 🟦 low | high | `ScoringSettings.tsx:254-363`; `SettingsPane.tsx:227` |
| wasm-bindings-3 | zstd decoder init `.map_err(...).unwrap()` (redundant; bare `.unwrap()` on dict decode) | 🟦 low | high | `shengji-wasm/src/lib.rs:17-31` |
| wasm-bindings-6 | json-schema codegen crate unconditionally depends on `shengji-wasm`/`shengji-mechanics` (unused) | 🟦 low | medium | `json-schema-bin/Cargo.toml:8-16` |
| wasm-bindings-7 | WASM module on `window` + `any` casts at the boundary | 🟦 low | medium | `WasmOrRpcProvider.tsx:62,341,365` |
| wasm-bindings-8 | `BatchGetCardInfo` has no WASM export; batching re-implemented in JS | 🟦 low | medium | `shengji-wasm/src/lib.rs:101-106` |
| backend-8 | Misleading `version`→`ws_id` variable shadowing in `register_user` | 🟦 low | high | `shengji_handler.rs:301-342` |
| build-deps-config-14 | Conflicting Prettier config: empty `package.json prettier:{}` + real `.prettierrc` | 🟦 low | high | `frontend/package.json:34`; `.prettierrc` |
| build-deps-config-15 | Workspace `members` omits core/mechanics/storage/backend-types | 🟦 low | medium | `Cargo.toml:1-8` |
| build-deps-config-17 | Redundant `serde_json` in core dev-dependencies | 🟦 low | high | `core/Cargo.toml:14,27` |
| build-deps-config-20 | Mixed crate editions (2018 vs 2021) across the workspace | 🟦 low | medium | multiple `Cargo.toml` |
| deadcode-tooling-7 | Several `@types/*` + jest/ts-loader/typescript in prod `dependencies` | 🟦 low | high | `frontend/package.json:10-14` |

**Notes.** `bots-search-3` (and the eval.rs copy inside `bots-search-1`) means **three** independent copies of the honesty-bypass branch exist — all currently correct, but a latent honesty-invariant drift risk; consolidating onto `harness::play_cards_for` removes that. Both `wasm-bindings-3` and `frontend-core-6` were reclassified to cleanliness (the original "discarded message" / "defeats SuitGroup type" claims were overstated — the message survives `.unwrap()`; the null cast is discarded downstream).

**Recommendations (effort):** trivial fixes for the typo/dup-lookup/dup-arm/no-op-preventDefault/dead-check items; **small** helper extraction for the duplicated ladder/successor/`CardInfo`-fallback blocks; **medium** module/component splits (`heuristics.rs`, `Play`/`Initialize`, `play_cards`) and the benchmark-driver consolidation onto the shared harness.

---

## Best practices

| id(s) | title | sev | conf | file:lines |
|---|---|---|---|---|
| build-deps-config-1 | No Rust toolchain pinning despite `#![deny(warnings)]` (floating stable) | 🟨 medium | high | CI + both Dockerfiles |
| bots-search-5 | PUCT + learned-value blend ship with **zero** direct test coverage | 🟨 medium | high | `search.rs:349-445,797-900,283-314` |
| deadcode-tooling-1 | `cargo clippy` broken (17 violations under `deny(warnings)`) → unusable as a gate | 🟨 medium | high | `expert.rs:111,343-355`; `tests.rs`; `trick.rs:4055` |
| frontend-core-2 | Context value + `updateState` recreated every render → full-tree re-renders | 🟨 medium | high | `AppStateProvider.tsx:92-124` |
| frontend-core-3 | Both committed test files re-implement production logic (tautological) | 🟨 medium | high | `WebsocketProvider.test.tsx`; `WasmOrRpcProvider.test.tsx` |
| frontend-ui-5 | `JoinRoom` schedules `setTimeout(generateRoomName,0)` in render body | 🟨 medium | high | `JoinRoom.tsx:76-78` |
| backend-7, security-deep-4 | Global zstd compressor behind poisonable `std::Mutex` with `.unwrap()` | 🟦 low | medium | `shengji_handler.rs:64-73`; `lib.rs:83-87` |
| frontend-core-7 | `window.send` global (load-bearing for Draw/Exchange, not just debug) | 🟦 low | high | `WebsocketProvider.tsx:323-324` |
| frontend-core-13 | Module-level mutable beep/ready timestamps + blocking `confirm()` in dispatch | 🟦 low | medium | `websocketHandler.ts:101-129` |
| frontend-core-14 | `updateSelectionAndGrouping` uses `any` for trump/grouping; non-null `!` | 🟦 low | medium | `Play.tsx:66-70,86,591` |
| frontend-ui-7 | Large amount of user-facing copy bypasses i18n (untranslated in zh) | 🟦 low | high | `ScoringSettings.tsx` et al. |
| frontend-ui-10 | `Cards.tsx` uses `any[][]` for card-group state | 🟦 low | high | `Cards.tsx:25-30` |
| frontend-ui-13 | "Make observer/player" emoji buttons: raw English title, no a11y label | 🟦 low | medium | `Players.tsx:272-326` |
| frontend-ui-14 | "rules" link text hardcoded English even in zh mode | 🟦 low | high | `GameMode.tsx:12-21` |
| frontend-ui-15 | `SvgCard` `createElement` on a map lookup with no undefined fallback | 🟦 low | low | `SvgCard.tsx:244-251` |
| frontend-ui-19 | `ScoringSettings` uses `any[]` for `scoreTransitions` (typed `ScoreSegment` available) | 🟦 low | high | `ScoringSettings.tsx:18-21` |
| frontend-ui-20 | `JoinRoom` `_blank` rules link missing `rel="noreferrer"` (convention elsewhere) | 🟦 low | high | `JoinRoom.tsx:135-141` |
| frontend-ui-6 | Deprecated `String.prototype.substr` for room-id generation | 🟦 low | high | `JoinRoom.tsx:67-74` |
| bots-search-9 | `FEATURE_DIM` duplicated Rust/Python with only runtime CSV guard | 🟦 low | medium | `expert.rs:66`; `train_expert.py:60` |
| core-gamestate-8 | `set_done_bidding` doesn't validate id is a seated player (benign) | 🟦 low | medium | `draw_phase.rs:275-283` |
| core-gamestate-9 | Friend selection order-randomized via HashSet before storing (no gameplay impact) | 🟦 low | medium | `exchange_phase.rs:145-208` |
| mechanics-7 | `TrickFormat::matches` panics on empty slice (latent; `pub fn`) | 🟦 low | high | `trick.rs:462-468` |
| mechanics-8 | Magic-number iteration caps in scoring (`take(50)`, `1..1000`) undocumented | 🟦 low | medium | `scoring.rs:255,265,283-289` |
| wasm-bindings-4 | Panic hook installed in only 1 of 10 WASM exports | 🟦 low | high | `shengji-wasm/src/lib.rs:33-110` |
| wasm-bindings-5 | type-gen tool indexes `args[1]` + unwraps every IO step (no usage msg) | 🟦 low | high | `json-schema-bin/src/main.rs:45-62` |
| build-deps-config-2 | No `--locked` in any build — `Cargo.lock` not enforced | 🟦 low | high | CI + `Dockerfile.deploy` |
| build-deps-config-6 | Node version mismatch (`.node-version` 22 vs CI/Docker 20; no `engines`) | 🟦 low | high | `.node-version` vs CI/Dockerfiles |
| build-deps-config-11 | Dependabot covers only `github-actions` (not cargo/npm) | 🟦 low | high | `.github/dependabot.yml:1-8` |
| build-deps-config-12 | Deprecated `tempdir` crate in type-gen tool | 🟦 low | high | `json-schema-bin/Cargo.toml:14` |
| build-deps-config-13 | Python training deps unpinned at the top end | 🟦 low | medium | `training/requirements.txt:1-5` |
| build-deps-config-19 | `gen_shard` reads `$?` after a compound command (fragile; no `pipefail`) | 🟦 low | medium | `run_value_pipeline.sh:82-88` |
| deadcode-tooling-2 | `tsc --noEmit` fails — missing `skipLibCheck` (not in any npm script) | 🟦 low | high | `frontend/tsconfig.json` |
| deadcode-tooling-5 | `react-color` mis-shelved in devDependencies (used in prod UI) | 🟦 low | high | `frontend/package.json:46` |
| build-deps-config-18 | `release.sh` shebang typo `#/bin/sh` (inert interpreter line) | 🟦 low | high | `release.sh:1` |

**Notes.** `deadcode-tooling-1` (broken clippy) was adjusted from high to medium and recategorized: `cargo build`/`cargo test` still pass, so it is a tooling-gate hygiene problem, not a runtime defect — but all four lint classes are trivial mechanical fixes that restore a useful gate. `build-deps-config-1` adjusted high→medium (latent, requires a new Rust release to trigger), but is doubly important under `deny(warnings)`. `frontend-core-7` was corrected: `window.send` is actually *used* by `Draw.tsx`/`Exchange.tsx`, so removing it requires migrating those consumers to context first.

**Recommendations (effort):** **trivial** — shebang, `substr→padStart`, `rel="noreferrer"`, `skipLibCheck`, react-color move, dependabot/`--locked` additions, `tempdir→tempfile`, `serde_json::from_value` typing. **small** — commit `rust-toolchain.toml` + pin Docker bases; fix the 4 clippy classes; `useCallback`/`useMemo` the AppState context; move `JoinRoom` side effect into `useEffect`; extract pure resolvers so the two test files exercise real code. **medium** — i18n coverage sweep; add targeted unit tests for PUCT / value-blend / `evaluate_position` orientation before those paths are enabled.

---

## Rejected / non-issues

**1 finding was rejected during verification**, confirming the kept list was filtered rather than passed through wholesale:

- **bots-heuristics-9** (void inference treats any off-suit card in a follow as a hard void): rejected because it misread the rules. The legality check in `mechanics/src/trick.rs:330-334` only permits a mixed (in-suit + off-suit) follow once *all* in-suit cards are exhausted, so a mixed follow does prove a genuine void. The determinizer's `infer_voids` logic matches the engine's own canonical `voids_this_hand` log — it is correct, not over-eager.

Additionally, several kept findings were **down-graded or recategorized** during verification rather than accepted as written (a sign of calibration, not inflation): e.g. `mechanics-1` (the negative-step exploit is defended downstream), `bots-search-4`/`bots-heuristics-1` (search time budget is not the binding constraint), `deadcode-tooling-1` (clippy break does not affect build/test), and `security-deep-3` (`ResetGame` requires two-player consent).

---

## Coverage & caveats

**Audited:** All Rust crates (`backend`, `core` incl. the full bot stack, `mechanics`, `storage`, `backend-types`), the WASM bindings + RPC boundary, the React/TS frontend (state/transport + UI components), the json-schema codegen tool, and the build/deploy/CI/training tooling (Dockerfiles, CI, dependabot, training pipeline + shell runner, package manifests). The honesty invariant was explicitly probed end-to-end (the `observed_state`/`for_player` redaction path, the determinizer, the bot search/value path, and the WS push path) and found intact except for the `/full_state.json` bypass.

**Tooling that ran:** `cargo check` / `cargo test` (clean); `cargo +1.92.0 clippy` (reproduced the 17 `deny(warnings)` violations); ESLint + `noUnusedLocals` (clean on source); `npx tsc --noEmit` (fails only on third-party `.d.ts`, clean with `--skipLibCheck`); repo-wide grep for dead symbols/imports; `cargo metadata` for the workspace graph.

**Tooling that could NOT run / blind spots:**
- **Bot strength benchmarks were not executed** (the `paired_eval`/`budget_benchmark` harnesses need a `--release` build + a search budget and are not byte-reproducible). The performance findings (`bots-search-4`, `bots-heuristics-1`, `bots-heuristics-2`) are reasoned from code structure and the documented budget model — **they should be profiled under `budget_benchmark` before optimizing**, per the findings' own recommendations. Several carry `low` confidence for this reason.
- **No dynamic/runtime exploitation** was performed (e.g. actually curling a live `/full_state.json`, or driving the panic sequences). Confidence on the critical/high items rests on static path-tracing, which is high but not a live PoC.
- **Redis backend** is exercised only by reading (its tests need a local Redis, expected-fail in this env); findings there (`backend-5`, `build-deps-config-5`) are about dead/never-instantiated code, so this gap is immaterial.
- **Confidence is reported per finding.** Items marked `low` confidence (notably `backend-9`, `backend-10`, `core-gamestate-6`, `bots-heuristics-7`/`-8`, `bots-search-4`, `frontend-ui-15`) are plausible-but-narrow and should be treated as "worth a guard/cleanup" rather than confirmed live defects. The two `critical`/`high` security items (`/full_state.json`, name-based takeover) and the three reliability panics are the high-confidence, act-first set.
