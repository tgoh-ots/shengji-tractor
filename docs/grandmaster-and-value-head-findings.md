# Grandmaster tier + value-head experiment — findings & learnings

> Session log (2026-06-29), written for a parallel session. Branch `worktree-better`
> (commits `529eb35` Grandmaster+Omniscient, `5351d8f` value-pipeline bug fix), based
> on master `46f44e9`. Companion to `docs/bot-training-roadmap.md` and
> `docs/bot-eval-baseline.md`. Toolchain: **`cargo +1.92.0`** (default `stable` is 1.80, too old).

## TL;DR (read this first)

1. **Deal variance dominates everything.** In 2-deck Tractor, win-rate gaps between
   strong policies are tiny and need huge n to resolve. Even the perfect-information
   **Omniscient cheater only beats the (improved) Enoch ~61%**; honest tiers that
   share Enoch's heuristic tie it. **n=300 → ±5.6pp CI; you need n≥1000–1500 to call a
   ~3pp effect.** I was fooled once by an n=300 "57%" that regressed to 50% at n=1200.
   Track **point-margin** (more sensitive) alongside win-rate, and never trust small n.
2. **Two new tiers shipped (honest, default-OFF value head):** `Grandmaster` (apex
   honest, calculation-driven, ~tied with Enoch by design) and a **fixed `Omniscient`**
   (was *losing* to Enoch; now clearly top).
3. **The learned value head is NEUTRAL at this scale** (no significant gain on any
   tier). Bounded by the aliasing floor + noisy teacher, exactly as the roadmap warned.
   But I **found+fixed a critical bug that had silently broken the whole value pipeline**
   (it was training on ~10 decisions out of 226k). Future runs are now correct.

## What shipped (and the numbers)

Tier ladder (vs the strengthened Enoch from master `7ca6b02`, gm_benchmark, multi-threaded):

| Matchup | win-rate | margin | n |
|---|---|---|---|
| Omniscient (FIXED) vs Enoch | ~61% | +13 pts | 400 |
| Omniscient vs Grandmaster | ~53% | +10 pts | 400 |
| Grandmaster vs Enoch | ~50–52% (TIED) | ~0…+3 | 600–1200 |
| Grandmaster vs Expert | 64% | +12 | 400 |
| Enoch vs Expert | 64% | +9 | 600 |

Ladder: `Easy < Expert << Enoch ≈ Grandmaster < Omniscient`. (Note: vs the *pre*-`7ca6b02`
Enoch, Grandmaster was ~56% / +6 — the Enoch upgrade closed most of that gap.)

### Grandmaster (`core/src/bot/policy.rs`, enum in `mod.rs`)
- It's the Enoch determinized search with: **full-hand rollouts** (`GM_ROLLOUT=0` → roll
  each sampled world to the last card = exact terminal points, no truncation bias),
  **8 candidates**, **400-world cap**, and a **larger budget** (`GM_BUDGET_MULT`, default 3×).
- Identity = **calculation-driven**: Enoch-playbook PRIOR (proposes sensible moves + keys
  the perfect-memory determinization) but commits to whatever the full-hand simulation
  values highest → diverges from Enoch's defensive playbook when the sim disagrees.
- **It cannot reliably out-*score* Enoch — it shares Enoch's heuristic space, so it only
  out-*searches*, which deal variance washes out.** A careful prior×rollout policy sweep
  (`GM_PRIOR`/`GM_ROLLOUT_POLICY` ∈ heuristic|net|enoch, n=1200 paired) found **NO variant
  reliably beats Enoch**: neutral rollout ≈ tie (the n=300 57% was noise), `GM_PRIOR=net`
  is clearly *worse* (~38% — forgoes full-memory determinization + the net policy is weak).
  Per-user the goal became "different playstyle at equal strength," which this satisfies.
- Budget has steep diminishing returns: mult=3 ≈ 52%, mult=6 ≈ 54.75% (n=400) at ~13s/move.

### Omniscient fix (the clear win) — `core/src/bot/policy.rs`
- **Bug:** it searched the true world with `Policy::Heuristic` (the *plain* heuristic), so
  despite perfect info it **LOST to playbook-driven Enoch (44.8% / −2.6 pts)** — better
  *strategy* beat better *information*.
- **Fix:** run the **Enoch playbook policy** (`Policy::EnochHeuristic` prior + rollouts over
  the real hands) + bigger budget (`OMNI_BUDGET_MULT`, default 5×, capped ~15s) + 32
  rollouts/candidate → **~61% / +13 vs Enoch.** (Neutral rollout was *worse* for Omniscient,
  57% vs 61% — opposite of the honest tiers: with true hands the playbook's wisdom pays off.)
- ⚠️ `OMNI_BUDGET_MULT` multiplies whatever `SHENGJI_BOT_BUDGET_MS` is — including the
  **data-gen teacher budget** (the teacher = Omniscient). Set `OMNI_BUDGET_MULT=1` when
  generating training data, or the teacher runs 5× its intended `GEN_TEACHER_BUDGET_MS`.

## Value-head experiment (roadmap's "1-month" bet) — executed, result NEUTRAL

Ran `training/run_value_pipeline.sh` (DAgger `mix` data → multi-task policy+value ONNX →
paired A/B) with the fixed Omniscient as a 400ms teacher, 4000 games.

**CRITICAL BUG found (now fixed in `5351d8f`):** sharded data-gen numbers decision-groups
(CSV col 1) per-process from ~0; stage-3 concat didn't re-number → **group IDs collided
across shards**. The trainer (`train_expert.py::load_groups`) keeps only groups with exactly
one `label==1`, so each collided group (NUM_SHARDS teacher-picks) was DROPPED → **"Loaded 10
decisions" out of a 1.2M-row / 226k-decision dataset** → junk net (the pipeline's own A/B
then showed value *hurting*). Fix: offset each shard's group IDs above the running max in
the concat. Retrained → 226,668 decisions loaded, value-RMSE 0.412, policy top-1 51.4%.

**Measured (properly-trained net, n=300, value off vs on; tier-vs-Easy isolates the search side):**

| | value OFF | value ON (w=0.5) |
|---|---|---|
| Expert vs Easy | 60.0% / +8.4 | 58.3% / +7.8 |
| Enoch vs Easy | 73.0% / +19.9 | 75.0% / +21.7 |
| GM(trunc-12) vs Enoch | 53.0% / +0.5 | 50.0% / −0.9 |

**Neutral on every tier** (all Δ within noise). Why, concretely: val-RMSE 0.41 on a
high-variance terminal-margin target, and **at a 12-trick leaf the static eval already has
most points realized**, so the learned value adds ~nothing over it. Policy top-1 51.4%
confirms the perfect-info teacher's picks aren't well-learnable from honest features
(aliasing floor). Kept **default-OFF** (`SHENGJI_VALUE_WEIGHT=0`); `value_fixed.onnx` is an
artifact, not embedded.

**To actually get a value-head win** (none of these is a quick win): de-noised teacher
labels (the deferred exact-endgame solver — blocked on a complete legal-move enumerator in
`mechanics`), a **lower-variance / shallow-leaf target** (the value head's niche is *shallow*
rollouts where the static eval is weak — at 12 tricks it isn't), or far more compute. More
of the *same* (bigger dataset, same teacher) won't cross the aliasing floor.

## Operational gotchas (these cost me time — avoid them)

- **`paired_eval` is single-threaded; `gm_benchmark` is multi-threaded (10 cores).** Use
  `gm_benchmark` for fast search A/Bs (it takes arbitrary `tierA-tierB` pairs:
  easy/expert/enoch/gm/omni). `paired_eval` at 200ms × 300 pairs took ~40+ min for ONE run.
- **Measurement nondeterminism:** Rust `HashMap` iteration order is per-process and leaks
  into tie-breaks → not byte-reproducible. My ad-hoc **duplicate/mirrored-deck scoring was
  biased ~+5pp toward the "subject" seat** (the per-deal cancellation breaks under
  nondeterminism). Use **independent games with alternating landlord side** (a tier
  configured == its opponent then scores a clean ~50% — verified 49.8% at n=600), or
  master's `paired_eval` which quantifies it with a deck-level bootstrap CI.
- **The value blend is a NO-OP at a terminal leaf.** `search.rs::net_value_estimate` returns
  `None` when `sim.game_finished()`. Full-hand rollouts (Grandmaster, Omniscient) always hit
  terminal → value head never fires. To use the value head you must **truncate rollouts**
  (`GM_ROLLOUT=12`).
- **Pipeline `$WORKDIR` is resumable and persists across git states** — a stale workdir will
  reuse shards from a *different teacher/settings* and skip retraining (a model marker exists).
  Use a fresh `WORKDIR=` per experiment.
- **Long background jobs get reaped:** macOS has no `setsid`, and the harness SIGTERMs a
  background task's process tree when its launcher task is reaped (killed 4 data-gen shards
  once). Run long jobs as a single harness-tracked background task with `trap '' TERM HUP`
  at the top (inherited as SIG_IGN by children) so stray SIGTERM can't kill them.
- **torch venv** installs fine on this box's Python 3.13 (`run_value_pipeline.sh` stage 0).

## Key files / knobs
- `core/examples/gm_benchmark.rs` — multi-threaded tier-vs-tier win-rate + Wilson CI; honors
  `SHENGJI_EXPERT_MODEL_PATH`, `SHENGJI_VALUE_WEIGHT`, `SHENGJI_SEARCH_PUCT`, and the
  `GM_*` / `OMNI_*` knobs.
- `core/src/bot/policy.rs` — Grandmaster + Omniscient dispatch and all env knobs
  (`GM_ROLLOUT`/`GM_WORLDS`/`GM_CANDS`/`GM_PRIOR`/`GM_ROLLOUT_POLICY`/`GM_BUDGET_MULT`,
  `OMNI_BUDGET_MULT`/`OMNI_WORLDS`/`OMNI_PRIOR`/`OMNI_ROLLOUT_POLICY`).
- `training/run_value_pipeline.sh` — fixed; the resumable value pipeline.
- Artifacts from this run live in `~/.shengji-value-gm/` (dataset `data_fixed.csv`, net
  `value_fixed.onnx`) — NOT committed.
