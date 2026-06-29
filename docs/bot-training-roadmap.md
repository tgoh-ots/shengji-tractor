# Bot Training Improvement Roadmap

> **Status:** the **1-day** and **1-week** plans are IMPLEMENTED (2026-06-29); the **1-month** plan is the remaining work. · **Drafted:** 2026-06-28
> **Scope:** how to make the Shengji bot models *stronger*, and how to *prove* they got stronger.
> Grounded in the current code (file:line references throughout). Companion to `PROGRESS.md` and the committed eval baseline `docs/bot-eval-baseline.md`.

## Execution status (2026-06-29)

**Done (1-day + 1-week):**
- Runtime net override `SHENGJI_EXPERT_MODEL_PATH` (`expert.rs`) + unit tests — A/B a net without rebuilding.
- Label-aliasing `--analyze` in `train_expert.py` (torch made optional so it runs with just numpy).
- Per-decision search instrumentation `SHENGJI_SEARCH_TRACE` (`search.rs`) — TIME-vs-WORLDS bound diagnostic.
- Data-gen drop counters + decoupled `GEN_TEACHER_BUDGET_MS` (default 400ms) in `gen_training_data.rs`. Confirmed `teacher-outside-candidates ≈ 0` (no "union-in" rework needed).
- Determinizer **Enoch full-memory** (sampled worlds never re-deal an already-played card; full-hand void inference) — this landed independently on master in `7ca6b02` (a `#[serde(skip)] voids_this_hand` log feeds `infer_voids`, plus a LAMBDA playbook-prior blend in selection). My contribution here is the additional **global card-conservation test** (`tests.rs::test_determinizer_full_memory_conserves_played_cards`) that pins the invariant.
- **Shared self-play harness** `core/src/bot/harness.rs` (deal + `Seat`/`PlayBrain` driver + honesty boundary); all 6 benchmark examples migrated onto it.
- **Paired-on-mirrored-deck** A/B + Wilson/bootstrap CIs + MDE (`harness::run_paired_ab`), surfaced by the new `paired_eval` example.
- **Committed baseline + non-inferiority gate**: `core/tests/baseline_gate.rs` (fast search-less gate, release-only search gate) + `docs/bot-eval-baseline.md`.

**Finding:** the benchmarks are **not byte-reproducible** run-to-run (Rust `HashMap` iteration order is per-process and leaks into tie-breaks), so the planned "byte-identical golden diff" verification was replaced with **distributional / CI** equivalence. A future reproducibility hardening (deterministic candidate ordering) is noted in `docs/bot-eval-baseline.md`.

**Started (1-month):** the **learned value head** PIPELINE is implemented & validated end-to-end, shipping
default-OFF (`SHENGJI_VALUE_WEIGHT=0`) — what remains there is training a real value net on a large dataset and
paired-measuring the blend weight. **Remaining:** the rest of the 1-month plan (DAgger, PUCT, endgame solver,
kitty distillation).

Toolchain note: build/test with `cargo +1.92.0 …` in this environment (deps need rustc ≥ 1.87; the default `stable` here is 1.80.1).

---

## TL;DR — the biggest lever

**Make strength _measurable_ first (≈week 1), then bet the month on a learned value head.**

The binding constraint today is **not "a better net" — it's that you can't tell whether any change helped.**
Deal variance swamps skill variance in this game, and nothing in the repo can currently resolve a
strength change below roughly **±7–13 percentage points**:

- Only 2 of ~6 harnesses (`core/examples/easy_ab_benchmark.rs`, `budget_benchmark.rs`) compute *any*
  statistic, and it's an unpaired z-vs-0.5 that throws away the seed-mirrored design.
- There is **no committed baseline number anywhere** — no Elo / Wilson / bootstrap.
- The net is `include_bytes!`-embedded in `core/src/bot/expert.rs` with **no runtime override**, so every
  net A/B costs a full Rust rebuild and is read by eye.

At that resolution, the +3–5pp gains a value head / PUCT / DAgger each promise are **statistically
invisible** — you cannot distinguish a real win from luck, and a regression ships silently behind the
honesty-only test gate.

Once measurement exists, the **single highest-ceiling strength change is adding a learned value head and
using it as the search leaf evaluator** — the current leaf (`search.rs::evaluate_position`) is a hand-rolled
magic-constant term (joker 1.6 / trump 0.8 / Ace 1.0 / King 0.5) that **credits only your own high cards and
ignores opponents' winners entirely** — a textbook hoarding bias that governs every rollout.

---

## Why the current approach has a ceiling

- **Behavioral cloning of a _perfect-info_ teacher, from honest-only inputs.** `gen_training_data.rs` labels
  each position with the **Omniscient** (cheating) teacher's pick, but the net
  (`expert.rs::candidate_features`, 36 features) only ever sees a redacted view at serve time. For many
  positions the perfect-info-optimal move is **information-theoretically unidentifiable** from honest
  features — identical feature vectors carry different labels. This is an irreducible **aliasing floor** no
  amount of epochs / bigger net can cross, **and its size is currently unmeasured.** Worse, the teacher
  itself is weak: data-gen defaults `SHENGJI_BOT_BUDGET_MS=8`, so even the "perfect-info" labels are noisy.
- **Train/serve distribution mismatch.** Every recorded trajectory is driven by `BotDifficulty::Easy`
  (blunders, no search, no memory), yet the net is deployed as the *root prior of a 144-world search*. The
  tight endgames / trump-management spots where the prior matters most are under-represented in training.
- **No value net; crude leaf; the learned signal barely reaches the decision.** `search.rs` says it outright:
  "the net is a policy, not a value net." The net is *only* the root prior, further diluted
  `NET_W=0.6 / HEUR_W=0.4`. Rollouts + leaf are 100% heuristic — shallow (12 of ~25 tricks), noisy (15%
  2nd-best), scored by the own-hand-only `evaluate_position`. The "AlphaZero-lite" framing has **no value
  half.**
- **Bidding and 扣底 (kitty burial) are never trained at all** — data-gen records only play-phase rows. The
  two highest-leverage 升级 decisions fall back entirely to heuristics.
- **Net selection is driven by offline top-1 imitation accuracy** — a proxy never shown to correlate with
  win-rate in this repo.

---

## The 1-day plan — make strength measurable + bank free wins  ✅ DONE (2026-06-29)

Goal: by end of day you can A/B a net without recompiling, you have the first committed baseline, and cheap
diagnostics tell you where the month should go.

- [ ] **Runtime `SHENGJI_EXPERT_MODEL_PATH` override in `expert.rs::load_model`** (~½ day).
  Env-gated path that loads a candidate ONNX at runtime, keeping the embedded default and the `<64-byte`
  placeholder guard. Turns every net A/B from a full rebuild into a flag — the biggest iteration-speed unlock.
  *Verify:* point at the current net → bit-identical play; point at a zeroed net → heuristic fallback fires.
- [ ] **Label-aliasing floor** — a `--analyze` pass in `train_expert.py` bucketing rows by rounded feature
  key, reporting the fraction of near-identical vectors with conflicting labels (2–3 granularities). High
  floor → cloning is exhausted, invest in value/kitty; low floor → data is still worth it.
- [ ] **Per-decision instrumentation in `search.rs`** — log worlds-completed + elapsed-ms to answer "is search
  world-bound or time-bound at 2200ms?" (prerequisite for judging whether PUCT/value matter).
- [ ] **Drop-rate counters in `gen_training_data.rs`** as a 30-min fact-check only.
  ⚠️ The "teacher-pick-outside-candidates" rate is almost certainly ~0 (both teacher and candidate generator
  go through `heuristics::lead/follow_candidates`) — **do not** build the "union-in the teacher pick" rework
  on that premise.
- [ ] *(Free rider, only if retraining anyway)* unify `f26` onto the new scorer to kill the documented
  legacy/new prior skew. Real correctness cleanup, but win-rate effect is below the noise floor.

---

## The 1-week plan — the measurement substrate, then one real bet  ✅ DONE (2026-06-29)

Assumes the 1-day work is done. **Sequence matters.**

- [ ] **Extract one shared self-play harness** (`core/src/bot/harness.rs`), deleting the ~600 lines of driver
  copy-pasted across the 6 examples (~2 days). Expose a `Contestant` abstraction covering both the
  `BotDifficulty` path and the raw `Knobs`/`Policy` path that `budget_benchmark` needs, plus a config knob for
  decks/players/Finding-Friends. *Verify by golden-output diff at a fixed seed* before merge — RNG-order drift
  would silently move every number. This is the enabling refactor that makes every following item a 1-file
  change.
- [ ] **Paired-on-mirrored-deck analysis + Wilson/bootstrap CIs** (~3 days, in the shared harness). Replace
  the unpaired z-vs-0.5 with: play each deck both orientations, difference per-deck margins, paired
  t / Wilcoxon, Wilson interval on win-rate, **bootstrap over deck-pairs (not games)**, and print the
  minimum-detectable-effect so "no difference" is distinguishable from "underpowered." **2–5× variance
  reduction** — existing 120–200-game runs start resolving +3–5pp gains. *Verify:* re-run an A/B under old vs
  new analysis; the new CI should be materially narrower.
- [ ] **Commit a baseline manifest + non-inferiority CI gate** (~3 days). Check in paired win-rate + CI for the
  load-bearing matchups (net-on vs net-off, Expert vs Easy, Enoch vs Expert, heuristic NEW vs LEGACY) at a
  fixed seed set and CI-affordable budget; add a fast paired smoke test that fails **only** below the CI lower
  bound (won't flake on noise). ⚠️ The gate runs ~100ms but ships 2200ms — tightly gate the budget-independent
  Easy/heuristic tiers; treat the search-tier gate as a coarse net. *Verify:* inject `NET_W=0` → it fails;
  ~zero false-positives over ~10 no-op commits.
- [ ] **Determinizer correctness bug-fixes** (~2 days — fixes only, *defer* Bayesian weighting).
  `sample_hidden_hands` hardcodes limited-memory `Knowledge::from_play_view` even for Enoch, so **Enoch's
  sampled worlds can contain cards it knows were already played**; `infer_voids` scans only the last 2 tricks.
  Give Enoch full-memory parity + full-history void inference. Strict correctness wins for every honest tier.
  (The likelihood-weighting half is easy to get wrong — dumping ≠ void — and slower per world; A/B it later.)
- [ ] **Decouple + raise the teacher budget** in `gen_training_data.rs`: separate `GEN_TEACHER_BUDGET_MS` from
  the 8ms behaviour budget, set ~300–500ms. An 8ms Omniscient label is near-noise. Floor-raiser, not a
  headline — prep for any retrain.

---

## The 1-month plan — raise the ceiling (now falsifiable)

Assumes the substrate exists, so these are now measurable bets. Sequenced by dependency.

- [x] **Learned VALUE head → search leaf evaluator** — PIPELINE IMPLEMENTED & validated end-to-end (2026-06-29),
  shipping **default-OFF** (see below). Concretely:
  - `gen_training_data.rs` back-fills the realized terminal margin per decision (oriented for the acting team,
    normalized by `expert::VALUE_NORM`) as a new `value` CSV column — a full-playout MC target, not bootstrapped.
  - `train_expert.py` is now a multi-task net: shared trunk + a policy head (listwise CE) **and** a `tanh` value
    head (MSE), exporting a **2-output ONNX** (`score`, `value`) when value targets are present (else policy-only,
    back-compat). The trainer + `--analyze` accept both CSV layouts.
  - `expert.rs::value_candidates_net` reads ONNX `output[1]`; `search.rs::evaluate_position` blends it
    (`(1-w)·static + w·net_value`, oriented to my team, scaled by `VALUE_NORM`) behind `SHENGJI_VALUE_WEIGHT`
    (default **0 = OFF** → production byte-unchanged). A policy-only / legacy model has no `output[1]` → blend
    auto-disabled; verified by `embedded_model_has_no_value_output` (and a `tract` 2-output load test).
  - **REMAINING (the actual payoff):** train a value head on a LARGE on-policy dataset and paired-measure the
    weight (`SHENGJI_VALUE_WEIGHT` + `paired_eval ... search`). De-risks still to honor when measuring: calibrate
    on **Expert/Enoch-generated** states (not the Easy split — the current data-gen `BEHAVIOUR` is Easy, so a
    DAgger pass below should precede/accompany this), and consider shorter `rollout_tricks` once the value net
    carries the leaf. The leaf net-call is at the rollout TERMINAL (once per candidate per world), not per-ply.
    Realistic **+3–8pp** once measured. **This is the change worth a month.**
- [~] **DAgger loop** — the data-gen MECHANISM is implemented (2026-06-29); the iterate-and-gate loop is the
  manual run. `gen_training_data` now takes `GEN_BEHAVIOUR` (easy | expert | enoch | **mix**, + `GEN_MIX_SEARCH_FRAC`):
  the chosen policy ADVANCES the game (the teacher still labels), so `mix` records states from the real search
  distribution the net serves — and, because the per-GAME continuation is now strong play, it also sharpens the
  VALUE target (addresses the value head's "calibrate on Expert/Enoch states, not the Easy split" de-risk).
  Default `easy` is rng-stream-identical to before. **Remaining:** actually iterate —
  generate(`GEN_BEHAVIOUR=mix`) → train → paired-gate → repeat **1–2 rounds**, NOT open-ended. ⚠️ The labeler is
  still the unidentifiable cheater, so DAgger moves *where on the ceiling* you sit, not the ceiling; and
  search-driven generation is hours, not minutes (the search behaviour shares the teacher's budget).
- [ ] **PUCT/ISMCTS + progressive widening** (~6–9 days) — **only after the value head.** Flat averaging gives
  every truncated candidate equal world budget and never searches rank-7+. PUCT concentrates the 144 worlds on
  the contested candidates and uncaps width — but with the crude leaf it mostly reshuffles bad estimates, and
  it sacrifices the current paired-world variance reduction. Pays off *after* a real leaf exists.
- [ ] **Exact alpha-beta endgame solver** (~6–8 days, optional). Replaces noisy rollouts in the last ≤~6
  tricks with exact tail evaluation; biggest clean win is making **Omniscient near-optimal** (raises the
  honest-vs-cheater ceiling *and* de-noises endgame teacher labels). Needs a conservative card-count threshold
  + node cap or Tractor's tractor/pair/throw branching blows the budget; **re-verify the honesty invariant** —
  it touches the perfect-info path.
- [ ] **Kitty (扣底) distillation — Phase 1 only this month.** Export landlord bury decisions with the
  teacher's choice and **measure the heuristic-vs-teacher disagreement rate.** Build the second model only if
  that number is large.

---

## Where the largest improvement comes from

In order:

1. **Measurement substrate (week 1)** — not strength itself, but the multiplier that makes every other lever
   real instead of guesswork.
2. **Learned value head (month)** — the highest *strength* ceiling; replaces the single crudest component
   governing every rollout and fixes the concrete hoarding bias.
3. **Kitty / 扣底 learning (Phase-1 now, Phase-2 if justified)** — 升级 games are decided in the bid and the
   kitty, and the net touches the kitty *zero*. Clean perfect-info label, tiny candidate space, trivially
   honesty-safe. **But size it before you build it.**

**Adjudicated disagreement:** the domain-expert reviewer ranked kitty #1; the pragmatic + skeptic reviewers
ranked measurement #1. Measurement wins because even a perfect kitty model produces a gain you currently can't
detect — and Phase-1 kitty *is itself a measurement task*. The views collapse: substrate → Phase-1 kitty →
decide.

---

## AVOID — judged non-starters / traps

Don't burn the month on these:

- **CFR-flavored multi-world soft targets** — 10–50× generation blowup, and the target is only as good as a
  determinizer that samples uniformly and relaxes voids, so it approximates the **wrong** posterior. A
  36-feature MLP likely can't express a conditional-on-information-set policy. Worst ROI in the set.
- **DeepSets / set-transformer card encoder** — bigger model on a ~5000-game corpus → overfit; offline top-1
  gains have a documented history of not transferring to win-rate; and the dual-declared `FEATURE_DIM` (Rust
  const + Python literal, guarded only by a CSV-width check) makes a layout-vs-weights mismatch a
  silent-corruption trap. Wrong lever before the leaf + measurement are fixed.
- **A "strong honest co-teacher"** — it *is* today's determinized search at higher budget, bounded by the same
  crude leaf + uniform determinizer it's meant to fix. Improving the leaf directly dominates it.
- **Suit-permutation augmentation** — `f6/f7` and the `f28–f35` memory features are led-suit-relative; permute
  them inconsistently and you *inject* label noise. Payoff is offline top-1, decoupled from strength.
- **Standalone `NET_W/HEUR_W` sweeps, `f26`-unify retrains, soft visit-distillation** — all target effects
  inside the noise floor. Acceptable only as free riders on a retrain you're already running, never as their
  own deliverable.
- **Building the kitty/bidding ONNX blind, or PUCT/value-head before the measurement substrate exists** —
  flying blind on your most expensive, highest-variance changes.

**The one-line rule:** spend week 1 making strength *visible*; spend the month on the *value head* (+ Phase-1
kitty); refuse anything whose own success metric sits below your noise floor.

---

## Key file map (for whoever picks this up)

| Concern | File |
|---|---|
| Net inference + 36-feature encoding + `FEATURE_DIM` + `NET_W/HEUR_W` | `core/src/bot/expert.rs` |
| Distillation trainer (MLP, loss, export ONNX) | `training/train_expert.py` |
| Training-data generation (teacher labels, Easy-driven trajectories) | `core/examples/gen_training_data.rs` |
| Determinized search, rollouts, `evaluate_position` leaf | `core/src/bot/search.rs` |
| Per-tier `Knobs` (worlds / candidates / rollout-tricks / budget) | `core/src/bot/policy.rs` |
| Hidden-hand sampling, void inference, `Knowledge` | `core/src/bot/determinize.rs` |
| Honesty invariant (`sees_perfect_information`, `observed_state`) | `core/src/bot/mod.rs` |
| Benchmarks (the eval substrate to rebuild) | `core/examples/{tournament,expert_ab,enoch_benchmark,easy_ab_benchmark,heuristic_benchmark,eval,budget_benchmark}.rs` |
| Honesty gate (must stay green on any bot change) | `backend/tests/e2e_game.rs::e2e_game_no_hidden_card_leakage` |
