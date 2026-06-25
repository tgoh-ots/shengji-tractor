# Shengji Online — Project Progress & Recovery Log

> Durable status doc so work can resume after a crashed/restarted session.
> **Last updated: 2026-06-24.** Update this at every milestone boundary.
>
> **Latest — PROJECT COMPLETE & LIVE** at https://shengji-tractor.fly.dev. All milestones done.
> - **AI tiers = Easy / Hard / Expert / Omniscient** (Medium removed). **Expert** is now a **net-guided determinized search** (AlphaZero-lite): the distilled net (PyTorch→ONNX via `tract`, 36 honest features only; **teacher = our own Omniscient**'s choices) is used as the ROOT POLICY PRIOR (`0.6·net + 0.4·heuristic` candidate pruning) on top of Hard's search, with fast heuristic rollouts + the static leaf value. (Net-guided *rollouts* measured to HURT — a net call per ply starves the time-boxed search — so the net is the prior only; wired via a `Policy` enum on `SearchConfig`.) Strength: **Expert ≈ Hard** (competitive; edges it in higher-budget release runs, ~coin-flip within noise), clearly beats Easy (81%) + degenerates (92%). Ladder `Easy < Hard ≲ Expert < Omniscient`.
> - **M6 verified IN PRODUCTION:** no hidden-card leakage even with Expert+Omniscient seated; CSP/HSTS/X-Frame + WS Origin-guard (403 bad / 101 native) confirmed live.
> - **Two UX rounds shipped:** game-settings redesign, footer/changelog removed, dark-only, decluttered toolbar (About+Language only), plain Omniscient label, readable rules page, fixed hand-gap + points-bar, mobile brand fix.
> - **ShengJi+ benchmark:** head-to-head vs the trained RL net is **permanently infeasible — its weights are gone** (jiaruishan.com NXDOMAIN, no Wayback, never committed). Compared instead vs their reproducible rule-based `StrategicAgent`: variant is a strong match; **our bots are in the same league as their strategic baseline** (Expert ~85% vs degenerate ≈ their strategic ~80% vs random); honest prior = we're behind their unrunnable best DMC net (97.7% lvl-rate). Scratch-only (`/tmp`), repo untouched.
> - CI green; dependabot merged; de-branded (engine attribution kept in LICENSE/NOTICE/README/About).

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
| M2.5 | ShengJi+ offline benchmark | ✅ DONE — head-to-head with the trained RL net INFEASIBLE (weights permanently gone: jiaruishan.com NXDOMAIN, no Wayback, never committed). Compared vs ShengJi+'s reproducible rule-based StrategicAgent: variant strong-match; our bots in the SAME LEAGUE as their strategic baseline (Expert ~85% vs degenerate ≈ their strategic ~80% vs random); honest prior we're behind their unrunnable best DMC net (97.7% lvl). Scratch-only, repo untouched. |
| Expert | Expert tier: **net-guided determinized search** (AlphaZero-lite) — distilled net (teacher = our Omniscient) as the root policy prior on Hard's search; ONNX → Rust via `tract`. Expert ≈ Hard (competitive), beats Easy 81%. | ✅ DONE & verified |
| Omniscient | Opt-in PERFECT-INFORMATION cheater tier (perfect-info search; honesty-bypass centralized in `observed_state`, gated to Omniscient only; honest tiers stay honest). UI Add-AI label. backend/tests/e2e_game.rs (WS no-leak e2e). | ✅ DONE & verified (27 core tests incl honesty-inversion; Omniscient beats Hard 62%; WS e2e no-leak passes; FE builds) |
| M5 | Deploy — backend **LIVE on Fly.io** (app `shengji-tractor`, single machine, auto-stop pay-per-use) at **https://shengji-tractor.fly.dev** (serves frontend + WS; CSP/HSTS/headers verified live). Code pushed to public GitHub **tgoh-ots/shengji-tractor**. Vercel declined (single Fly URL). | ✅ DONE |
| M6 | End-to-end verification — VERIFIED LIVE IN PRODUCTION: WS game (observer + all 4 tiers) with zero hidden-card leakage; security headers + WS Origin guard + HTTPS confirmed live; mobile render checked + fixed. backend/tests/e2e_game.rs (WS no-leak) green. | ✅ DONE & verified |

## Live deployment
- **GitHub repo:** https://github.com/tgoh-ots/shengji-tractor (public). `origin` = HTTPS as `tgoh-ots` (the machine's SSH key maps to a different account, `tGoh98`, so push over HTTPS); `upstream` = rbtying/shengji. Pushing workflow files needs the gh token's `workflow` scope (already granted).
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
