# Shengji Online — Project Progress & Recovery Log

> Durable status doc so work can resume after a crashed/restarted session.
> **Last updated: 2026-06-29.** Update this at every milestone boundary.
>
> **Status: LIVE & actively iterated** at https://shengji-tractor.fly.dev.
> The original milestones (M0–M6) are all done; work since then has been a
> **bot-ladder overhaul**, a round of **UI/game-flow polish**, and a
> **security/reliability audit + hygiene sweep** (all shipped).

## Current state (2026-06-29)

### 2026-06-29 — Security/reliability audit + hygiene sweep (shipped & deployed)
A multi-agent code audit (`docs/code-audit-2026-06-29.md`, with a remediation-status
section) drove two commits, both pushed to `master` and deployed:
- **Act-first fixes**: closed a **critical** unauthenticated `/full_state.json`
  leak (it served every room's un-redacted hands — route removed; the on-disk
  recovery dump still runs on the 60s `periodically_dump_state` timer; regression
  test added). Guarded a jack-variation last-trick `unwrap` panic (a server-crash
  reachable from an untrusted message sequence). De-poisoned the wasm zstd decoder
  + added a binary-WebSocket `try/catch` so one corrupt frame is survivable. Fixed
  a `jack_variation` save/load data-loss bug.
- **Hygiene sweep**: restored a clean `cargo clippy` gate (0 warnings/errors);
  removed dead code (`should_bid`, `Suit::unicode_offset`, gated `_get_cards`) &
  unused deps (`lazy_static`, `axum-macros`, `@sentry/tracing`,
  `hook-shell-script…`, `tempdir`→`tempfile`); added **`rust-toolchain.toml`
  pinning rustc 1.92.0** (bare `cargo` now auto-selects it; makes the
  `#![deny(warnings)]` deploy build deterministic); resolved the prettier-config
  conflict (deleted the never-applied `.prettierrc`; codebase is on defaults);
  dependabot cargo+npm, `tsconfig` skipLibCheck, `release.sh` shebang.
- **Accepted risk** (will NOT fix): name-based seat takeover — casual site; the
  real fix is a per-seat-token wire-protocol change. **Open follow-ups**: harden
  the unauthenticated `/api/rpc` (CPU DoS) and a frontend render-index-crash
  cluster (both medium).

### Grandmaster tier + Omniscient fix (latest, shipped) — rebased on the strengthened Enoch (`7ca6b02`)
Added a 5th tier: **`Easy < Expert <= Enoch < Grandmaster <= Omniscient`**. Built on
top of the improved Enoch (full-memory search, void-aware play, flip-bidding,
declare gate). Self-play harness: `core/examples/gm_benchmark.rs` (multi-threaded,
Wilson-CI win-rate; a tier configured == its opponent scores ~50%).

**Grandmaster — a DIFFERENT play style at EQUAL strength (the stated goal).**
Enoch greedily obeys its hand-coded defensive playbook; Grandmaster is
**calculation-driven** — it uses the Enoch playbook only to PROPOSE candidates (and
to key the perfect-memory determinization), then commits to whatever its **full-hand
determinized rollouts** value highest, with a **neutral (non-playbook) leaf
evaluation**, so it breaks the playbook's instincts when the simulation disagrees.
Knobs: full-hand rollouts (`GM_ROLLOUT=0`), 8 candidates, 400-world cap, 3× budget
(`GM_BUDGET_MULT`); prior/rollout policy are env-selectable (`GM_PRIOR`,
`GM_ROLLOUT_POLICY`). Lives entirely in `policy.rs` (knobs + dispatch) + the enum in
`mod.rs`; search/determinize shared with Enoch.

- **Empirical (n=1200, paired):** Grandmaster is **statistically TIED with the
  strong Enoch (~50–52% win-rate)** — equal strength, different decisions. A careful
  policy sweep (prior × rollout ∈ {heuristic, net, enoch}) found **NO variant
  reliably out-scores Enoch**: GM shares Enoch's heuristic space, so it can only
  out-*search*, and the enormous deal variance washes that out. (An n=300 "neutral
  rollout = 57%" reading was a sampling fluke — it regressed to ~50% at n=1200.
  `GM_PRIOR=net` is clearly WORSE, ~38%, since it forgoes full-memory determinization
  and the net is weak.) The win-rate ceiling here is genuinely low.

**Omniscient — FIXED (was quietly broken).** It was running the *plain heuristic*
policy and, despite perfect information, **LOST to the playbook-driven Enoch (44.8%
/ −2.6 pts)** — better strategy beat better information. Fix: run the **Enoch
playbook policy** (prior + full-hand rollouts over the real hands) + a bigger budget
(`OMNI_BUDGET_MULT`, default 5×, capped ~15s) + 32 rollouts/candidate. Result vs the
strong Enoch: **~61% / +13 pts** — clearly the top of the ladder again. (Enoch
rollout beat neutral rollout for Omniscient, 61% vs 57%, the opposite of what helps
the honest tiers — with true hands the playbook's wisdom pays off.)

Honesty invariant preserved (`e2e_game_no_hidden_card_leakage` green); the 5-tier
ladder is surfaced in the lobby (`AddAIPlayer.tsx`, i18n en/zh, regenerated
`gen-types`). **Key lesson:** in this 2-deck game deal variance dominates, so
win-rate gaps between strong policies are tiny (even the cheater is only ~61%);
point-margin and large-n CIs are the reliable signals, and small samples mislead.

**Value-head experiment (learned leaf eval) — executed, result NEUTRAL.** Ran the
full `run_value_pipeline.sh` (DAgger data → multi-task policy+value ONNX → A/B) with
the fixed Omniscient as teacher. Found a CRITICAL bug (sharded group IDs collided →
trainer loaded ~10 of 226k decisions → junk net) — INDEPENDENTLY also caught + fixed
on master in `79f79e0` (per-shard `i*1e9` group-id offset in the concat). Retrained
net (226k decisions, value-RMSE 0.41) measured neutral
on Expert/Enoch/Grandmaster — the aliasing floor + 12-trick static leaf already
captures most signal. Value head stays default-OFF. **Full session writeup +
operational gotchas (paired_eval is single-threaded, value blend is a no-op at a
terminal leaf, duplicate-deck scoring bias, OMNI_BUDGET_MULT inflates the teacher,
WORKDIR staleness, long-job SIGTERM reaping): `docs/grandmaster-and-value-head-findings.md`.**

**Shipped to production** (Fly `version 35`, 2026-06-29) — the 5-tier ladder + the security
audit build are live at https://shengji-tractor.fly.dev. A full 5-tier round-robin
(`gm_benchmark`, all 10 pairings) confirms the clean ladder by overall win-rate:
**Easy 21% < Expert 40% < Enoch 55% ≲ Grandmaster 59% < Omniscient 75%** (matrix +
per-model characteristics table in `docs/grandmaster-and-value-head-findings.md`).

### Bot ladder overhaul (earlier major work — shipped)
The bot tiers were redesigned from `Easy/Hard/Expert/Omniscient` into a cleaner
**four-tier ladder `Easy < Expert <= Enoch < Omniscient`**. (See `CLAUDE.md` →
"Bot Architecture" for the full description; this is the status summary.)
- **`Hard` was REMOVED.** A `#[serde(alias = "Hard")]` on `Expert` keeps old
  persisted/in-flight state deserializing (as Expert) so live rooms created
  before the change don't break — guarded by the
  `legacy_hard_difficulty_deserializes_as_expert` regression test (do not delete).
- **Deeper search**: Expert/Enoch now run a time-boxed determinized (ISMCTS)
  search with **144 worlds / 12 rollout-tricks** and a **~2200ms** budget ceiling
  (`SHENGJI_BOT_BUDGET_MS` override); most decisions finish in tens of ms.
- **Re-distilled Expert net**: a small MLP (`core/src/bot/expert_model.onnx`,
  36→128→128→64→1) distilled from the Omniscient teacher via
  `core/examples/gen_training_data.rs` + `training/train_expert.py`, used as the
  ROOT POLICY PRIOR of the search (heuristic rollouts, static leaf value —
  AlphaZero-lite). Falls back to the hand-written heuristic prior if the net
  can't run. (`expert_model_old.onnx` is the previous net, kept for A/B.)
- **Enoch — new strongest HONEST tier**: the same search over the
  boss-/partner-aware heuristic PLUS a full-game **playbook** transcribed from a
  strong player (`docs/strategy/double-holder.txt`, `trip-holder.txt`):
  pair-prioritized trump declaring, point-scaled kitty discipline, **no
  high-trump opens**, tractor-first / long-suit leading, partner point-dumping, a
  defender low-trump hand-off, endgame kitty protection, and perfect play-memory.
- **Harder Easy**: still the weakest/beatable tier, but its blunder rate and
  softmax temperature were nudged down (ε 0.28→0.06, temp 3.5→1.1) so it follows
  the heuristic's top suggestions more often — without ever gaining search.
- **Stronger shared evaluation**: `heuristics.rs` `score_lead`/`score_follow` +
  `EvalCtx` gained boss-card detection (from public play history), point/partner/
  trump awareness. Used by all real tiers; the legacy scorer is kept for A/B.
- **HONESTY INVARIANT preserved**: only `Omniscient` sees hidden hands; the cheat
  is the single `sees_perfect_information()`-gated `observed_state()` branch.
  Enforced by `backend/tests/e2e_game.rs::e2e_game_no_hidden_card_leakage`.
- **Non-blocking bot driver**: the (expensive) search runs OFF the game lock via
  a plan/apply split (`bot/mod.rs` `classify_next_bot_work` /
  `plan_next_bot_action` / `apply_planned_bot_action`) driven by
  `shengji_handler.rs::drive_bots_non_blocking` (snapshot → `spawn_blocking`
  compute → brief apply-under-lock), so chat/UI don't lag while bots think.
- **Benchmark harnesses** in `core/examples/`: `tournament`, `expert_ab`,
  `enoch_benchmark`, `easy_ab_benchmark`, `heuristic_benchmark`, `eval`,
  `budget_benchmark` (run `--release`; budget via `SHENGJI_BOT_BUDGET_MS`).

### UI / game-flow polish (shipped)
- **Compass (N/S/E/W) trick layout** (`frontend/src/Table.tsx` / `Trick.tsx` /
  `style.css`): rotates the local player to the south seat; each player's played
  cards render at their own compass point reaching toward an empty center hub.
- **"Done bidding" flow** (`draw_phase.rs` / `bot/mod.rs`): after a bot bids, the
  landlord is finalized only once ALL HUMANS confirm done; the standing winner is
  treated as implicitly done so a human winner isn't frozen. Replaced the timed
  counter-bid grace.
- **Team colors** in-game and in the lobby (declarer/teammate/opponent/unknown);
  the "unknown" role only appears in Finding Friends until revealed.
- **Kitty-points game-log line** (`message.rs` `KittyScored`): logs the kitty's
  value × multiplier and which team won/kept it.
- **Bilingual, English-only-clean**: single en/zh toggle; English mode renders no
  Chinese anywhere (incl. game-mode names).
- **Rules page** (`frontend/static/rules.html` + `rules.js`): CSP-safe standalone
  page, language-aware (no Chinese in English mode), a "Finding Friends only"
  callout, centered real-SVG card examples, colored jokers.
- Plus: client auto-reconnect (silent re-join on dropped WebSocket), bot pacing
  pauses so humans can follow, never-empty played-cards broadcast, decluttered
  in-game team labels, lobby bot renaming, off-lock bot compute so chat doesn't
  lag, suppressed benign action-ordering toasts.

## What we're building
A web app to play Shengji (升级 / Tractor / Finding Friends) with friends and/or
AI bots. The honest bots only ever see public info (their own redacted view);
only the opt-in `Omniscient` cheater sees hidden hands. Modern redesigned
responsive bilingual UI. Cheap always-on hosting. Basic security hardening. No
secrets in the repo. Forked from **rbtying/shengji** (MIT). Repo:
`/Users/tgoh/playground/shengji`.

## Locked decisions
- **Architecture:** FORK the Rust rules engine (`mechanics/`) — do NOT rewrite the
  rules. Bots live in `core/src/bot/`.
- **AI:** in-process Rust bots, cheat-proof by construction (the honest tiers read
  only the per-player redacted view; the cheat is centralized in one gated
  branch). Tiers: `Easy / Expert / Enoch / Omniscient`. Expert's net is OUR OWN
  (distilled from our Omniscient teacher) and ships embedded as ONNX, run in pure
  Rust via `tract-onnx` (no C/onnxruntime dependency → builds in the musl deploy
  image).
- **Frontend:** restyle the React 19 app in place (Tailwind-based dark UI), mobile
  + bilingual 中文/English. PRESERVE the WS protocol + WASM pipeline.
- **Hosting:** backend → **Fly.io** single service (serves embedded frontend + WS;
  ~$2/mo, cheaper with auto-stop). Frontend on Vercel is OPTIONAL. Render / Oracle
  Cloud documented as alternatives in `DEPLOY.md`. Deploy uses a self-contained
  `Dockerfile.deploy`.
- **Scope:** standard 4-player fixed-partnership Tractor + Finding Friends.
- **AI data/strength:** NO public human Shengji corpus exists. Use **self-play**
  for strength and **ShengJi+** as an offline benchmark/teacher prior. We ship our
  own clean net.

## Architecture map
- `mechanics/` — pure stateless rules engine (bidding, tricks, tractors, scoring).
- `core/` — stateful engine: `game_state/{initialize,draw,exchange,play}_phase.rs`,
  `interactive.rs` (`Action` enum + `InteractiveGame::interact`), `message.rs`
  (broadcast / game-log), `settings.rs` (`PropagatedState`).
  - `core/src/bot/` — the AI: `mod.rs` (`BotDifficulty`, registry, `advance_bots`,
    the plan/apply non-blocking split, the single `observed_state` honesty gate),
    `policy.rs` (per-tier knobs + dispatch, search budget), `heuristics.rs` (the
    shared `score_lead`/`score_follow`/`EvalCtx` + the `*_enoch` playbook),
    `search.rs` (determinized search + `Policy` enum), `expert.rs` (ONNX net +
    feature encoding), `determinize.rs` (world sampling), `tests.rs`.
  - **Cheat boundary:** `GameState::for_player(id)` redacts other hands; honest
    bots get only this view via `observed_state`.
- `backend/` — Axum WS server (`GET /api`); `shengji_handler.rs` (per-connection
  loop + `drive_bots_non_blocking`); `lib.rs` (`build_app` for tests);
  `main.rs` (routing, CORS, static serving via `include_dir` of `frontend/dist`);
  binds `0.0.0.0:3030`.
- `frontend/` — React 19 + TS + Webpack 5 + `shengji-wasm`. `gen-types.d.ts` is
  auto-generated from the Rust types via `yarn types`. See `CLAUDE.md` →
  "Frontend Structure" for the per-file map (compass table, team colors, i18n,
  rules page, etc.).
- `training/` — the Python distillation pipeline (`train_expert.py`,
  `requirements.txt`, `README.md`); `data.csv` is the generated training set
  (gitignored / large).
- `docs/strategy/` — the transcribed human playbooks driving Enoch
  (`double-holder.txt`, `trip-holder.txt`, + the source `.docx`).

## Toolchain / build (macOS)
- Source Rust each shell: `. "$HOME/.cargo/env"`. yarn via corepack.
- **Build frontend BEFORE backend** (backend embeds `frontend/dist`):
  `cd frontend && yarn build`, then `cargo build --bin shengji`.
- Dev serve-from-disk: `cargo run --features dynamic` (serves `../frontend/dist`
  at runtime).
- Run: `./target/debug/shengji` (binds :3030). Env: `CORS_ALLOWED_ORIGINS`,
  `WEBSOCKET_HOST`, `VERSION`, `DUMP_PATH`, `MESSAGE_PATH` (see `.env.example`).
- Tests: `cargo test --all` — green EXCEPT the Redis storage tests (need a live
  Redis; **expected to fail**). Game-logic + bot tests must pass, incl.
  `e2e_game_no_hidden_card_leakage` and `legacy_hard_difficulty_deserializes_as_expert`.
- **The gate that matters: `cargo build --bin shengji`** — backend has
  `#![deny(warnings)]`, so the binary must build warning-clean. `cargo clippy`
  currently surfaces some pre-existing lints in `expert.rs`/`mechanics` from a
  newer toolchain — those are not the deny-warnings gate.
- Regenerate TS types after changing Rust types: `cd frontend && yarn types`.

## Milestone status (history — all DONE)
| # | Milestone | Status |
|---|---|---|
| M0 | Foundation: toolchains, baseline build, server boots+serves, hygiene | ✅ DONE |
| M1 | Bot-seat plumbing: AddAIPlayer/RemoveAIPlayer, BotDifficulty+registry, advance_bots driver, cheat-boundary tests | ✅ DONE |
| M2 | AI brain: determinizer + heuristics + time-boxed ISMCTS; self-play eval ladder | ✅ DONE |
| M2.5 | ShengJi+ offline benchmark (RL net weights permanently gone; compared vs their rule-based StrategicAgent — same league) | ✅ DONE |
| M3 | Frontend redesign — lobby + in-game table, responsive/mobile, i18n (中文/EN), Add-AI flow, four-color + a11y | ✅ DONE |
| M4 | Security hardening (Origin allowlist, msg-size cap, rate limit, room/player caps, sanitization, CSP/HSTS headers) | ✅ DONE |
| Expert | Expert tier: net-guided determinized search (distilled net = root prior) | ✅ DONE (since re-distilled — see "Current state") |
| Omniscient | Opt-in PERFECT-INFORMATION cheater tier; honesty bypass centralized + gated; WS no-leak e2e | ✅ DONE |
| M5 | Deploy — backend LIVE on Fly.io; code on public GitHub | ✅ DONE |
| M6 | End-to-end verification — verified live in production (no leakage, security headers/WS Origin guard, mobile) | ✅ DONE |
| Ladder overhaul | Drop Hard → 4 tiers, deeper search, re-distilled Expert, harder Easy, Enoch + playbook + perfect memory, stronger shared heuristic, non-blocking driver | ✅ DONE & shipped |
| UI/flow polish | Compass trick layout, Done-bidding flow, team colors, kitty-points log, English-only clean, rules page redo, auto-reconnect, bot pacing | ✅ DONE & shipped |

## Live deployment
- **GitHub repo:** https://github.com/tgoh-ots/shengji-tractor (public).
- **Backend: LIVE on Fly.io** — app `shengji-tractor`, URL
  **https://shengji-tractor.fly.dev** (single service: serves embedded frontend +
  `/api` WebSocket). **Single machine — DO NOT scale >1** (rooms are in-memory).
  `fly.toml` uses `auto_stop_machines = "suspend"` so the machine's RAM (and the
  in-memory rooms) survive idle→wake; reconnect-by-name covers mid-game wakes. A
  full redeploy (new image) still resets all rooms.
- **Redeploy — deploy from a CLEAN worktree at a specific SHA** so concurrent
  uncommitted work isn't shipped:
  ```bash
  git worktree add --detach /tmp/sj-deploy <sha>
  cd /tmp/sj-deploy
  export PATH=/opt/homebrew/bin:$PATH
  fly deploy --ha=false        # builds Dockerfile.deploy on Fly's remote builder
  cd -
  git worktree remove /tmp/sj-deploy --force
  ```
  Logs: `fly logs -a shengji-tractor`. Status: `fly status -a shengji-tractor`.
  Config: `fly.toml`. Full runbook: `DEPLOY.md`.
- Prod env: `CORS_ALLOWED_ORIGINS=https://shengji-tractor.fly.dev`, `VERSION=fly`.
- **Frontend on Vercel: OPTIONAL** (faster first paint) — the single Fly service
  already serves the frontend.

## How to resume after a crash
1. Read this file; check `git status` / `git log --oneline -20` for the latest
   committed work and any uncommitted changes (we do NOT commit unless asked).
2. Confirm build state:
   `. "$HOME/.cargo/env"; cargo test -p shengji-core` (bot tests green) and
   `cd frontend && yarn build`. Confirm `cargo build --bin shengji` is
   warning-clean. Run `e2e_game_no_hidden_card_leakage`.
3. The project is feature-complete and live; ongoing work is iterative
   (bot strength + UI polish). Pick up from the user's current request.

## Notes
- Do NOT commit to git unless the user asks.
- No secrets in the repo; `.env` is gitignored; the AI is local (no external API
  key to leak).
- Validate strength changes with the `core/examples/` benchmarks, not just unit
  tests; validate behavior changes with unit + e2e tests.

## Known follow-ups / caveats
- `cargo clippy` has pre-existing lints in `expert.rs`/`mechanics` from a newer
  toolchain — the real gate is `cargo build --bin shengji` under
  `#![deny(warnings)]`.
- Heavy self-play eval tests are slow in debug; run benchmarks in `--release`.
- Redis storage tests fail without a local Redis (expected).
- E2E tests rely on in-test `tokio::time::timeout`, not shell `timeout` (absent
  on macOS).
- `training/data.csv` is large and gitignored; regenerate with
  `gen_training_data` before retraining (see `CLAUDE.md` → "Retrain the Expert
  net").
