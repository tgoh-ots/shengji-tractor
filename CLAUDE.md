# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Shengji Online (升级 Online) is an online implementation of the Chinese trick-taking card game 升级 ("Tractor" or "Finding Friends"). It is an open-source fork of the rbtying/shengji engine (MIT), adding cheat-proof AI opponents, a redesigned responsive bilingual UI, and Fly.io deployment. The live site is **https://shengji-tractor.fly.dev** (live and actively iterated). It features a Rust backend with WebSocket support and a React TypeScript frontend with WebAssembly integration.

The headline differentiator is a **four-tier bot ladder** (`Easy < Expert <= Enoch < Omniscient`) with a hard **honesty invariant** (only the opt-in cheater tier may see hidden hands), a re-distilled learned-net Expert tier, an `Enoch` tier that follows a transcribed human strategy playbook, and a non-blocking bot driver so the search never lags chat/UI. See "Bot Architecture" below and `PROGRESS.md` for current status.

## Commands

> ⚠️ **Toolchain:** this repo's deps need **rustc ≥ 1.87**, but the machine's default
> `stable` may be older (it is 1.80.1 in the current dev env). Prefix EVERY `cargo`
> command below with the right toolchain — e.g. `cargo +1.92.0 test --all`,
> `cargo +1.92.0 build --bin shengji`. Running bare `cargo` on the old default
> fails with dependency build errors (`is_multiple_of`/edition mismatches). The
> bare `cargo …` shown below is for brevity; add `+1.92.0` (or your installed ≥1.87
> toolchain) in this environment.

### Development
```bash
# Run frontend in development mode with hot reloading
cd frontend && yarn watch

# Run backend in development mode
cd backend && cargo run --features dynamic

# Full development setup (run in separate terminals)
cd frontend && yarn watch
cd backend && cargo run --features dynamic
```

### Building
```bash
# Build production frontend
cd frontend && yarn build

# Build release backend
cargo build --release

# Full production build
cd frontend && yarn build && cd ../backend && cargo run
```

### Testing
```bash
# Run all Rust tests
cargo test --all

# Run specific Rust test
cargo test test_name

# Run frontend tests
cd frontend && yarn test

# Run frontend tests in watch mode
cd frontend && yarn test --watch

# The honesty-invariant gate (must stay green on any bot change): asserts no
# Omniscient/Expert seat ever leaks a hidden card over the WebSocket.
cargo test -p shengji --test e2e_game e2e_game_no_hidden_card_leakage

# Regression: legacy "Hard" bot difficulty must still deserialize (as Expert).
cargo test -p shengji-core legacy_hard_difficulty_deserializes_as_expert

# Committed-baseline strength gate (fast, search-less, paired CIs): fails only if a
# build drops a load-bearing relationship below its floor (Easy@NEW>OLD, NEW
# heuristic>=LEGACY). Re-measure + update floors if you change the shared scorer /
# Easy knobs. See docs/bot-eval-baseline.md.
cargo test -p shengji-core --test baseline_gate
# Coarse release-only search/net gate (Expert beats Easy):
SHENGJI_BOT_BUDGET_MS=60 cargo test -p shengji-core --release --test baseline_gate -- --ignored
```

### Bot benchmarks (headless self-play harnesses)
All live in `core/examples/`. Run in `--release` (debug is far slower) and set
`SHENGJI_BOT_BUDGET_MS` to control the per-decision search budget (production
default is 2200ms; benchmarks usually lower it).
```bash
# Full round-robin win-rate matrix across all four tiers
cargo run --release --example tournament

# Expert-net A/B: does the embedded expert_model.onnx beat the heuristic prior?
# (swap core/src/bot/expert_model.onnx between runs to compare nets)
cargo run --release --example expert_ab

# Enoch vs a chosen opponent tier (pair-aware declaring, kitty discipline, etc.)
cargo run --release --example enoch_benchmark

# Easy-tier knob A/B (softmax temperature + blunder rate: new vs old)
cargo run --release --example easy_ab_benchmark

# NEW boss-/partner-aware heuristic vs the frozen legacy scorer
cargo run --release --example heuristic_benchmark

# Original Easy/Expert/Enoch/Omniscient ladder eval; search-budget sweep
cargo run --release --example eval
cargo run --release --example budget_benchmark

# Paired-on-mirrored-deck A/B with Wilson + bootstrap CIs and a minimum-detectable
# -effect (the statistically-sound measurement substrate). Each deck is played in
# BOTH orientations to cancel deal luck. 3rd arg `which`: `fast` = search-less
# matchups; `search` = Expert-vs-Easy + Enoch-vs-Expert (honors SHENGJI_BOT_BUDGET_MS);
# `expert-easy` = ONLY Expert-vs-Easy (the value-blend A/B; half the wall-clock);
# `all` (default) = fast + search. See docs/bot-eval-baseline.md.
cargo run --release --example paired_eval -- 400 0x5EED fast
SHENGJI_BOT_BUDGET_MS=400 cargo run --release --example paired_eval -- 200 0x5EED search

# Kitty (扣底) Phase-1 audit: force candidate burials on the SAME deal, play each
# out with fixed greedy play, report whether the default burial leaks margin vs
# alternatives (directs whether a learned kitty model is worth building).
cargo run --release --example kitty_audit -- 300
```

All of the above share ONE driver — `core/src/bot/harness.rs` (`seeded_draw_phase`
+ the `Seat`/`PlayBrain` per-hand driver + the honesty boundary + the paired-AB
stats). Add a new benchmark by configuring `Seat`s, NOT by re-copying the loop.

> ⚠️ The benchmarks are NOT byte-reproducible run-to-run (Rust `HashMap` iteration
> order is per-process and leaks into tie-breaks). Compare CIs/distributions, never
> a byte-diff. See `docs/bot-eval-baseline.md`.

> Toolchain: build/test with `cargo +1.92.0 …` in this repo's dev env — deps need
> rustc ≥ 1.87 and the machine's default `stable` may be older.

Useful bot debug env vars: `SHENGJI_EXPERT_MODEL_PATH` (load a candidate ONNX at
runtime — A/B a net with no rebuild), `SHENGJI_SEARCH_TRACE=1` (log per-decision
worlds/budget/TIME-vs-WORLDS-bound from `search_play`), `SHENGJI_BOT_BUDGET_MS`
(per-decision search budget), `SHENGJI_VALUE_WEIGHT` (0..1, **default 0 = OFF**;
blend the learned VALUE head into the search leaf evaluator — needs a 2-output
value-head model, else it's a no-op; see the value-head note below),
`SHENGJI_SEARCH_PUCT=1` (**default OFF**; allocate the determinized search's
simulations by PUCT/UCB — concentrate the budget on the contested candidates —
instead of flat per-world averaging; A/B it via `paired_eval … search`).

### Code Quality
```bash
# Lint TypeScript
cd frontend && yarn lint

# Fix TypeScript lint issues
cd frontend && yarn lint --fix

# Lint Rust
cargo clippy

# Format TypeScript
cd frontend && yarn prettier --write

# Check TypeScript formatting
cd frontend && yarn prettier --check

# Format Rust
cargo fmt --all

# Check Rust formatting
cargo fmt --all -- --check
```

### Type Generation
```bash
# Generate TypeScript types from Rust schemas (run from frontend directory)
cd frontend && yarn types && yarn prettier --write && yarn lint --fix
```

### Retrain the Expert net (`core/src/bot/expert_model.onnx`)
The Expert tier's prior is a small MLP distilled from the Omniscient teacher.
Pipeline: Rust self-play data export → PyTorch training → ONNX → embedded in Rust
(`include_bytes!`, run via `tract-onnx` — no C/onnxruntime dependency, builds in
the musl deploy image). **This section (CLAUDE.md) is the source of truth for the
current pipeline**; `training/README.md` is older high-level background and may lag
(e.g. the value head, DAgger, and the runner are documented here, not there).
```bash
# 1. Generate distillation data. Writes training/data.csv (group, f0..f35, label,
#    value). GEN_TEACHER_BUDGET_MS (default 400) sets the teacher search budget =
#    label QUALITY, independent of any inherited SHENGJI_BOT_BUDGET_MS; it also
#    prints skipped-decision counts (degenerate / teacher-no-play / teacher-outside).
#    GEN_BEHAVIOUR (easy | expert | enoch | mix; default easy) picks which policy
#    ADVANCES the game = the recorded STATE DISTRIBUTION (a DAgger knob): `mix`
#    (GEN_MIX_SEARCH_FRAC, default 0.5) advances some games with the real search
#    tier so states match what the net actually serves — also sharpens the value
#    target. A search behaviour shares the teacher budget and is MUCH slower.
#    GEN_SEED (default 0xD157111) is the deal-RNG seed; DISTINCT seeds give disjoint
#    deals, so a run can be SHARDED across processes (GEN_SEED=base+i per shard) for
#    parallel/reproducible generation — this is what run_value_pipeline.sh does.
#    GEN_OUT (default training/data.csv) sets the output path.
GEN_GAMES=5000 GEN_TEACHER_BUDGET_MS=400 GEN_BEHAVIOUR=mix \
  cargo run --release --example gen_training_data

# 1b. (optional) Estimate the behavioral-cloning CEILING before training: the
#     label-aliasing floor + "unwinnable decision" rate (high => stop polishing the
#     policy, invest in a value head / kitty). Runs with just numpy (no torch).
python training/train_expert.py --data training/data.csv --analyze

# 2. Train + export ONNX. Shared trunk 36->128->128->64, then a POLICY head
#    (listwise softmax CE) and an OPTIONAL tanh VALUE head (MSE on the realized
#    terminal margin). With value targets present (the `value` CSV column from
#    step 1), it exports a 2-OUTPUT model (score, value); --value-weight 0 (or a
#    legacy policy-only CSV) exports the policy-only 1-output model.
cd training && pip install -r requirements.txt   # torch, onnx, numpy
python train_expert.py --data data.csv --out ../core/src/bot/expert_model.onnx

# 3. A/B the candidate net vs the embedded one WITHOUT rebuilding, via the runtime
#    model-path override (expert.rs::MODEL_PATH_ENV). A bad/missing path falls back
#    to the heuristic (it does NOT silently use the embedded net). For a value-head
#    net, also set SHENGJI_VALUE_WEIGHT>0 to engage the leaf-eval blend.
SHENGJI_EXPERT_MODEL_PATH=$PWD/candidate.onnx SHENGJI_VALUE_WEIGHT=0.5 \
  cargo run --release --example paired_eval -- 200 0x5EED search
```
The whole steps 1→3 chain (DAgger data-gen → train value head → A/B the blend) is
automated + **resumable** by `training/run_value_pipeline.sh` — a checkpointed,
parallel (sharded data-gen via `GEN_SEED`), reboot-safe runner. Re-run it to RESUME
(completed shards/train/A/B are skipped via markers in `$HOME/.shengji-value-run`);
`STATUS=1 bash training/run_value_pipeline.sh` reports progress.

CONTRACT: `FEATURE_DIM` (currently 36) is defined in `core/src/bot/expert.rs`
(`candidate_features`) AND hardcoded in `train_expert.py`; the value-augmented CSV
has `1 + FEATURE_DIM + 1 (label) + 1 (value)` columns (the trainer also accepts the
legacy policy-only width). The SAME `candidate_features` builds both the training
rows and the inference vector, so the encoding can't drift — but if you change it,
bump it in both places and retrain. The VALUE head: `gen_training_data` emits the
realized terminal margin (oriented per acting team), normalized by
`expert::VALUE_NORM` (the single Rust-side scale, shared by data-gen + the
`search::evaluate_position` blend; the Python trainer is scale-free); inference
reads ONNX output[1] and blends it behind `SHENGJI_VALUE_WEIGHT` with a static-eval
fallback (a policy-only model → no value output → blend auto-disabled). The
embedded model is policy-only by default, so the value path ships INERT until you
train + embed a value-head net. See `docs/bot-training-roadmap.md` (1-month plan).

## Architecture

### Rust Workspace Structure
- **backend/**: Axum web server handling WebSocket connections and game API. `shengji_handler.rs` holds the per-connection loop and the **non-blocking bot driver** (`drive_bots_non_blocking`); `lib.rs` exposes `build_app` for integration tests; binds `0.0.0.0:3030`.
- **core/**: Game state management, message types, serialization, and **the AI bots** (`core/src/bot/`). `game_state/{initialize,draw,exchange,play}_phase.rs` are the per-phase engines; `interactive.rs` holds the `Action` enum + `InteractiveGame`; `message.rs` holds broadcast/game-log messages; `settings.rs` holds `PropagatedState`.
- **mechanics/**: Pure stateless rules engine (bidding, tricks, tractors, scoring). DON'T change the rules.
- **storage/**: Storage abstraction layer supporting in-memory and Redis backends. (Redis tests fail without a local Redis — expected.)
- **frontend/shengji-wasm/**: WebAssembly bindings for client-side game mechanics.

### Bot Architecture (`core/src/bot/`)
A **four-tier, user-selectable strength ladder `Easy < Expert <= Enoch < Omniscient`** (the `Hard` tier was REMOVED). All four are surfaced in the lobby (`frontend/src/AddAIPlayer.tsx`), Expert is the default.

- **Easy** — the bare shared heuristic played noisily: occasional blunders (ε ≈ 0.06), a warm softmax over the top moves (temp ≈ 1.1), no card memory, no search. Clearly the weakest/beatable tier.
- **Expert** (default) — a small MLP (`core/src/bot/expert_model.onnx`, distilled from the **Omniscient** teacher via `training/train_expert.py` + `core/examples/gen_training_data.rs`) scores each legal candidate from **HONEST features only** and serves as the **prior** of a time-boxed **determinized (ISMCTS-style) search** over sampled worlds. AlphaZero-lite: net = root policy prior, fast heuristic = rollout policy, static leaf evaluator. If the net can't load/run, the prior transparently falls back to the shared hand-written heuristic, so Expert is never illegal/None.
- **Enoch** — the **strongest HONEST tier**. Same determinized search, but driven by the boss-/partner-aware heuristic plus a full-game **"playbook"** transcribed from a strong player (`docs/strategy/double-holder.txt`, `docs/strategy/trip-holder.txt`): pair-prioritized trump declaring, point-scaled kitty discipline, **no high-trump opens**, tractor-first / long-suit leading, partner point-dumping, a defender low-trump hand-off, endgame kitty protection, and perfect play-memory. Honest — consults only its own redacted per-player view.
- **Omniscient** — a deliberate, opt-in, clearly-badged **CHEATER**: a perfect-information search over the single true world (it sees every opponent's hand). For testing and an "impossible" practice opponent.

**HONESTY INVARIANT (core design rule):** Only `Omniscient` may see hidden hands. The cheat is centralized in ONE place — the `sees_perfect_information()` predicate (only true for Omniscient) gates the single `observed_state()` branch in `bot/mod.rs` that hands a bot the unredacted state; every honest tier gets `dump_state_for_player` (opponents' cards are `Card::Unknown`, kitty hidden). Adding a future honest tier cannot leak cards unless it opts in. The `backend/tests/e2e_game.rs::e2e_game_no_hidden_card_leakage` test enforces this every build — **any bot change must keep it passing.**

Key files:
- `heuristics.rs` — the shared evaluator: `score_lead`/`score_follow` + `EvalCtx`, with boss-card detection (`is_boss_card`/`boss_strength`) derived from public play history. The NEW boss-/partner-aware scorer is what all real tiers use (legacy scorer kept for A/B). Enoch's playbook lives in the `*_enoch` variants (`ranked_leads_enoch`, `ranked_follows_enoch`, `choose_kitty_enoch`, `bid_strength_enoch`).
- `policy.rs` — per-tier `Knobs` (ε, temperature, search worlds/candidates/rollout-tricks) and the dispatch in `select_action`. Search config: budget **~2200ms** (`search_budget_ms()`, override with `SHENGJI_BOT_BUDGET_MS`), **144 worlds**, **12 rollout-tricks** for Expert/Enoch. Most decisions finish in tens of ms; the budget is the safety ceiling.
- `search.rs` — the determinized search with the `Policy` enum (`Heuristic` / `Net` / `EnochHeuristic`) for the root prior and rollout policy.
- `expert.rs` — the ONNX net inference + the `candidate_features` honest-feature encoding (the contract shared with training).
- `determinize.rs` — samples plausible hidden hands consistent with the redacted view (`Knowledge`, void tracking). `sample_hidden_hands(.., full_memory, ..)`: Enoch passes `true` (perfect public memory — never re-deals a played card, full-hand voids); Easy/Expert pass `false`.
- `harness.rs` — the SHARED self-play benchmark engine: `seeded_draw_phase`, the `Seat`/`PlayBrain` per-hand driver (`play_one_hand`, honesty-correct `play_cards_for`), and the paired-on-mirrored-deck A/B + stats (`run_paired_ab`, Wilson/bootstrap CIs, MDE). Every `core/examples/` benchmark builds on this; `core/tests/baseline_gate.rs` gates on it.
- `mod.rs` — `BotDifficulty` (with `#[serde(alias = "Hard")]` on Expert so old states still deserialize), the `advance_bots` driver, and the **plan/apply split** for the non-blocking driver: `classify_next_bot_work` (cheaply decides whose turn / what kind of work), `plan_next_bot_action` (does the expensive search OFF the lock on a snapshot), `apply_planned_bot_action` (briefly re-acquires the lock to apply the precomputed move). This is why bot thinking doesn't lag chat/UI — see `shengji_handler.rs::drive_bots_non_blocking` (snapshot → `spawn_blocking` compute → apply-under-lock).
- `tests.rs` — bot unit tests incl. the cheat-boundary and `legacy_hard_difficulty_deserializes_as_expert` regression tests.

### Frontend Structure
React 19 + TypeScript + Webpack 5 + `shengji-wasm`. `gen-types.d.ts` is auto-generated from the Rust types via `yarn types`.
- **frontend/src/**: React components and application logic.
- **frontend/src/state/**: WebSocket connection and state management (`WebSocketProvider`, client auto-reconnect / silent re-join).
- **frontend/src/Play.tsx**: Main gameplay component (wires up the compass table).
- **frontend/src/Table.tsx** + **Trick.tsx**: the **compass (N/S/E/W) trick layout** — a CSS grid that rotates the local player to the south seat and lays each player's played cards out at their own compass point reaching toward an empty center hub. `Table.tsx` also derives the **team-color role** per seat (`declarer`/`teammate`/`opponent`/`unknown`); `style.css` holds the grid/seat/compass/team-color CSS.
- **frontend/src/Players.tsx**, **StatusRail.tsx**: lobby team coloring and the declarer/points/turn status rail.
- **frontend/src/AddAIPlayer.tsx**: lobby UI for adding bots — offers the four tiers `Easy / Expert / Enoch / Omniscient` (default Expert).
- **frontend/src/BidArea.tsx**: bidding UI incl. the **"Done bidding"** button (see Development Notes).
- **frontend/src/i18n.tsx**, **GameMode.tsx**: bilingual 中文/English (custom React-context i18n, `localStorage["lang"]`, single en/zh toggle — **English mode shows no Chinese**).
- **frontend/src/ChatMessage.tsx**, **Chat.tsx**: in-game chat + game-log rendering (incl. the kitty-points line).
- **frontend/src/Draw.tsx**, **Card.tsx**, **SvgCard.tsx**: card rendering (real SVG cards, four-color suits).
- **frontend/static/rules.html** + **rules.js** + **rules-cards.json**: the standalone CSP-safe rules page (copied to `dist/` by webpack; NOT compiled from `src/`). Language-aware (no Chinese in English mode), a "Finding Friends only" callout, centered SVG card examples.
- **frontend/json-schema-bin/**: utility for generating TypeScript types from Rust.

### Type Safety Strategy
The project maintains type safety between Rust and TypeScript by:
1. Defining types in Rust using serde serialization
2. Generating JSON schemas from Rust types
3. Converting schemas to TypeScript definitions via json-schema-bin
4. Sharing game logic through WebAssembly for client-side validation

### WebSocket Communication
- All game state updates flow through WebSocket connections
- Messages are typed and validated on both client and server
- State synchronization happens automatically via the WebSocketProvider

## Development Notes

### When modifying game mechanics:
1. Update logic in `mechanics/src/`
2. If changing message types, update `core/src/message.rs`
3. Regenerate TypeScript types with `yarn types`
4. Update frontend components to handle new mechanics

### When adding new features:
1. Implement server-side logic in appropriate Rust module
2. Add message types if needed in `core/`
3. Generate TypeScript types
4. Implement UI in React components
5. Ensure WebSocket message handling is updated

### When changing the bots (`core/src/bot/`):
1. The shared evaluator is `heuristics.rs` — change `score_lead`/`score_follow`/`EvalCtx` to affect ALL real tiers; the `*_enoch` variants only affect Enoch.
2. **Keep the honesty invariant**: never read another seat's cards outside the single `observed_state()` Omniscient branch. Re-run `e2e_game_no_hidden_card_leakage` after any bot change.
3. Don't remove the `#[serde(alias = "Hard")]` on `Expert` or the `legacy_hard_difficulty_deserializes_as_expert` test — live rooms created before `Hard` was dropped still send "Hard" and the wasm state-sync rejects the whole update if it fails to deserialize.
4. The bot move-selection (`plan_next_bot_action`) runs OFF the game lock; if you add expensive work, keep it in the plan step, not in `apply_planned_bot_action`, so chat/UI don't lag.
5. Validate strength changes with the `core/examples/` benchmarks (see Commands), not just unit tests. The statistically-sound path is the **paired-on-mirrored-deck harness** (`core/src/bot/harness.rs` via the `paired_eval` example) — it reports win-rate with CIs + a minimum-detectable-effect. Benchmarks are NOT byte-reproducible (per-process `HashMap` order), so compare **CIs, not byte-diffs** (see `docs/bot-eval-baseline.md`). Add a NEW benchmark by configuring `Seat`s on the shared harness — do NOT copy-paste a driver.
6. If you change the shared scorer (`score_lead`/`score_follow`) or the Easy knobs, **re-run the committed strength gate** `cargo +1.92.0 test -p shengji-core --test baseline_gate`; if a baseline genuinely moved, update the floors in `core/tests/baseline_gate.rs` AND `docs/bot-eval-baseline.md` in the same change.
7. Build/test with **`cargo +1.92.0`** (deps need rustc ≥ 1.87; the env default may be older) — the honesty test, the gate, and the examples all need it.
8. If you change the Expert net's feature encoding, update `FEATURE_DIM` in BOTH `core/src/bot/expert.rs` and `training/train_expert.py`, then regenerate data + retrain. (The VALUE head adds the `expert::VALUE_NORM` contract — shared by data-gen + `search::evaluate_position`; see the retrain section.)

### Notable game-flow details:
- **"Done bidding" flow** (`core/src/game_state/draw_phase.rs` + `bot/mod.rs`): after a bot bids, the landlord is finalized only once ALL HUMANS confirm "done bidding" (a new bid re-opens the window and clears everyone's done flag). The standing bid winner is treated as implicitly done (so a human winner isn't frozen waiting for themselves); bots are always implicitly done. This replaced the old timed counter-bid grace.
- **Kitty-points game-log line** (`core/src/message.rs`, `KittyScored`): logs the kitty's point value × multiplier and which team won the last trick / kept the kitty.
- **Bot pacing** (`shengji_handler.rs`): bots pause briefly (`SHENGJI_BOT_ACTION_PAUSE_MS`) after a meaningful move, and longer (`SHENGJI_BOT_TRICK_PAUSE_MS`) before clearing a trick they won, so humans can follow along. Draws are bursted (no per-draw pause).

### Testing approach:
- Unit test game mechanics in Rust (`mechanics/src/`)
- Integration test API endpoints in `backend/`
- Component testing for React UI elements
- Manual testing for WebSocket interactions and gameplay flow

### Deployment (Fly.io) — read this before deploying
The whole app is **one Rust binary** that serves BOTH the embedded frontend and the `/api` WebSocket on `0.0.0.0:3030`. Live app: `shengji-tractor` at https://shengji-tractor.fly.dev. See `DEPLOY.md` for the full runbook; `fly.toml` is the service config (uses `auto_stop_machines = "suspend"` so in-memory rooms survive idle→wake; **single machine — DO NOT scale >1**, rooms are in-memory).

**CRITICAL GOTCHAS:**
1. **Build the frontend BEFORE the backend** — the backend embeds `frontend/dist` via `include_dir` at compile time. (The Docker build does this for you; matters for local release builds.)
2. **Deploy from a CLEAN git worktree at a specific SHA**, so concurrent uncommitted work in the main checkout is NOT shipped:
   ```bash
   git worktree add --detach /tmp/sj-deploy <sha>
   cd /tmp/sj-deploy
   export PATH=/opt/homebrew/bin:$PATH
   fly deploy --ha=false        # builds Dockerfile.deploy on Fly's remote builder
   cd -                          # back to the main checkout
   git worktree remove /tmp/sj-deploy --force
   ```
3. A full redeploy (new image) resets all in-memory rooms; mid-game reconnect-by-name only survives the suspend/wake cycle, not a redeploy.

Logs: `fly logs -a shengji-tractor`. Status: `fly status -a shengji-tractor`.

### Before committing any changes:
Always run lints and formatting before committing:

**Rust:**
```bash
cargo fmt --all && cargo clippy
```

**Frontend (`.tsx`/`.ts`):**
```bash
cd frontend && yarn lint --fix && yarn prettier --write
```

> **The gate that actually matters is `cargo build --bin shengji`** — the backend has `#![deny(warnings)]`, so the binary must build with ZERO warnings. Note `cargo clippy` currently surfaces some pre-existing lints in `expert.rs`/`mechanics` from a newer toolchain; those are not the deny-warnings gate. The Redis storage tests fail without a local Redis (expected).
