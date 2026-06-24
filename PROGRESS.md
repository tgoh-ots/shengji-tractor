# Shengji Online — Project Progress & Recovery Log

> Durable status doc so work can resume after a crashed/restarted session.
> **Last updated: 2026-06-23.** Update this at every milestone boundary.

## What we're building
A web app to play Shengji (升级 / Tractor) with friends and/or AI bots (Easy/Medium/Hard,
plus a future **Expert** tier). Bots only ever see public info (no cheating). Modern redesigned
UI. Free hosting. Basic security hardening. No secrets in the repo.

Forked from **rbtying/shengji** (MIT). Repo: `/Users/tgoh/playground/shengji`.

## Locked decisions
- **Architecture:** FORK the Rust backend (battle-tested rules engine) — do NOT rewrite the rules.
- **AI:** in-process Rust bot, cheat-proof (reads only the per-player redacted view). Easy/Med/Hard
  = one engine with knobs; **Expert** = a learned net later.
- **Frontend:** restyle the existing React 19 app in place (Tailwind v4 + shadcn-style), add
  mobile + bilingual 中文/English. PRESERVE the WS protocol + WASM pipeline.
- **Hosting:** backend → **Fly.io** (~$2/mo always-on shared-cpu-1x/256MB, or `auto_stop`/pay-per-use; user opted into a small paid host since Koyeb's free tier was discontinued Feb 2026 after the Mistral acquisition). Frontend → Vercel. Render (free, cold-starts) and Oracle Cloud Always Free (free, always-on, VM setup) documented as alternatives in DEPLOY.md. Deploy uses a self-contained `Dockerfile.deploy` (host-agnostic).
- **Scope:** standard 4-player fixed-partnership Tractor (engine supports more; UI scoped to 4p).
- **Rules:** repo defaults, exposed as per-room settings.
- **AI data/strength:** NO public human Shengji corpus exists. Use **self-play** for strength, and
  **ShengJi+** (`github.com/themoon2000/shengji_plus`, Berkeley EECS-2023-127; weights checkpoint
  `1180000.zip`) as an offline **benchmark** and **teacher**. ShengJi+ has no license → use as
  teacher/benchmark and ship our OWN net (clean + free-host-friendly). Personal/non-commercial use.

## Architecture map
- `mechanics/` — pure stateless rules engine (bidding, tricks, tractors, scoring). DON'T change rules.
- `core/` — stateful engine: `game_state/{initialize,draw,exchange,play}_phase.rs`, `interactive.rs`
  (the `Action` enum + `InteractiveGame::interact`), `settings.rs` (`PropagatedState`).
  - `core/src/bot/` — **NEW (M1):** `BotDifficulty`, bot registry, `policy.rs` (move selection from
    the redacted view), `advance_bots` driver. **M2 expands this** (heuristics, determinizer,
    ISMCTS, self-play eval harness).
  - **Cheat boundary:** `GameState::for_player(id)` redacts other hands; bots get only this view.
- `backend/` — Axum WS server (`GET /api`), `shengji_handler.rs` (per-connection loop; calls
  `advance_bots` after each human action), `main.rs` (routing, CORS, static serving via
  `include_dir` of `frontend/dist`), binds `0.0.0.0:3030`.
- `frontend/` — React 19 + TS + Webpack 5 + `shengji-wasm` (WASM). `gen-types.d.ts` is auto-generated
  from the Rust types via `yarn types`.

## Toolchain / build (macOS)
- Source Rust each shell: `. "$HOME/.cargo/env"` (rustc 1.96, wasm-pack 0.13, wasm32 target present).
- yarn via corepack (1.22).
- **Build frontend BEFORE backend** (backend embeds `frontend/dist` via `include_dir`):
  `cd frontend && yarn build` (webpack + WASM), then `cargo build --bin shengji`.
- Dev serve-from-disk: `cargo run --features dynamic` (serves `../frontend/dist` at runtime).
- Run: `./target/debug/shengji` (binds :3030). Env: `CORS_ALLOWED_ORIGINS`, `WEBSOCKET_HOST`,
  `VERSION`, `DUMP_PATH`, `MESSAGE_PATH` (see `.env.example`).
- Tests: `cargo test --all` — green EXCEPT the 3 `storage/tests/redis_storage_tests` (need a live
  Redis; **expected to fail** with "Connection refused"). Game-logic + bot tests must pass.
- CONSTRAINT: backend has `#![deny(warnings)]` — all code must be warning-clean.
- Regenerate TS types after changing Rust types: `cd frontend && yarn types`.

## Milestone status
| # | Milestone | Status |
|---|---|---|
| M0 | Foundation: toolchains, baseline build, server boots+serves, hygiene (.gitignore/.env.example) | ✅ DONE |
| M1 | Bot-seat plumbing: AddAIPlayer/RemoveAIPlayer actions, BotDifficulty+registry, advance_bots driver, dumb-but-legal policy, self-play + cheat-boundary tests, regen types | ✅ DONE & verified |
| M4 | Security hardening (Origin allowlist, msg-size cap, rate limit, room/player caps, sanitization, CSP/HSTS headers; move-validation tests; SECURITY_NOTES.md) | ✅ DONE & verified (all tests green except redis; warning-clean) |
| M3 | Frontend redesign — lobby + in-game table (4-seat, StatusRail), responsive/mobile, card animations, ~70 i18n keys (中文/EN), Add-AI flow verified, four-color + a11y. Screenshots in frontend/.design-screens/. | ✅ DONE |
| M2 | AI brain: determinizer + heuristics + time-boxed ISMCTS; Easy/Med/Hard; self-play eval ladder (Hard>Med>Easy, monotonic). Files: core/src/bot/{determinize,heuristics,search,policy}.rs, core/examples/eval.rs. SHENGJI_BOT_BUDGET_MS env overrides search budget (default 1000ms). | ✅ DONE & verified (25 core tests; warning-clean) |
| M2.5 | ShengJi+ offline benchmark bridge (measure our bot vs the published RL agent) | ⏳ PENDING (after M2) |
| Expert | Expert tier: learned net (distill from ShengJi+ / self-play), ONNX → Rust via `tract` | ⏳ PENDING (after M2.5; does NOT block first deploy) |
| Omniscient | Opt-in PERFECT-INFORMATION cheater tier (perfect-info search; honesty-bypass centralized in `observed_state`, gated to Omniscient only; honest tiers stay honest). UI Add-AI label. backend/tests/e2e_game.rs (WS no-leak e2e). | ✅ DONE & verified (27 core tests incl honesty-inversion; Omniscient beats Hard 62%; WS e2e no-leak passes; FE builds) |
| M5 | Deploy — backend **LIVE on Fly.io** (app `shengji-tractor`, single machine, auto-stop pay-per-use) at **https://shengji-tractor.fly.dev** (serves frontend + WS; CSP/HSTS/headers verified live). Vercel frontend split = optional next. CI workflows written. | 🔄 backend LIVE; Vercel optional |
| M6 | End-to-end verification: AUTOMATED e2e tests (WS-driven full game via axum-test, asserting flow + no card leakage) + humans + each AI tier (incl Omniscient) + mobile + WS-leak spot-check | ⏳ PENDING (after M2/M3/M4) |

## Live deployment
- **Backend: LIVE on Fly.io** — app `shengji-tractor`, URL **https://shengji-tractor.fly.dev** (single service: serves embedded frontend + `/api` WebSocket). Verified live: `/`=200, `/stats`=200 (sha=fly), security headers (CSP/HSTS/X-Frame/nosniff) present, `/api`=400 (WS route up). **Single machine — DO NOT scale >1** (rooms are in-memory). `auto_stop` + `min_machines_running=0` (pay-per-use, ~1-2s wake; reconnect-by-name covers mid-game wakes).
- Redeploy: `export PATH=/opt/homebrew/bin:$PATH; fly deploy --ha=false` (builds `Dockerfile.deploy` on Fly's remote builder). Logs: `fly logs -a shengji-tractor`. Status: `fly status -a shengji-tractor`. Config: `fly.toml`.
- Prod env: `CORS_ALLOWED_ORIGINS=https://shengji-tractor.fly.dev` (also add the Vercel origin if the frontend is split out), `VERSION=fly`.
- **Frontend on Vercel: OPTIONAL** (faster loads) — `vercel` deploy with build env `WEBSOCKET_HOST=wss://shengji-tractor.fly.dev/api`, then add the Vercel origin to Fly's `CORS_ALLOWED_ORIGINS`. The single Fly service already serves the frontend, so this is a nicety.

## How to resume after a crash
1. Read this file; run `TaskList` for current task states.
2. Confirm build state: `. "$HOME/.cargo/env"; cargo test -p shengji-core` (should be green incl
   `bot::tests::*`) and `cd frontend && yarn build`.
3. Check `git status` / `git diff` for uncommitted work (we do NOT commit unless the user asks).
4. Background agents do not survive a session crash — re-check M3 (frontend) / M4 (security)
   completion by inspecting `frontend/` and `backend/` diffs; re-launch the incomplete ones.
5. Continue from the first PENDING milestone in the table above.

## Notes
- Do NOT commit to git unless the user asks.
- No secrets in the repo; `.env` is gitignored; the AI is local (no external API key to leak).

## Known follow-ups / caveats
- `core/examples/eval.rs` ladder is noisy at low budget in DEBUG (24-hand samples); Hard>Medium can invert at 30ms debug. Production uses 1000ms release; the strict `test_difficulty_ladder_monotonic` is `#[ignore]`d and verified in release — re-confirm in release during M5/CI.
- `test_difficulty_ladder_mixed_tier_self_play_quick` is SLOW in debug (~60s; full `cargo test --all` ~26min). For CI, gate the heavy self-play behind release or `--ignored`.
- E2E tests rely on the in-test `tokio::time::timeout`/deadline, NOT shell `timeout` (absent on macOS).
- backend now exposes `backend/src/lib.rs` (`build_app`) so integration tests can construct the app; `main.rs` builds on it.
