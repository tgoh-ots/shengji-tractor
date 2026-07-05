# Strongest-bot compute runtime and Week 1 execution plan

> **Status:** compute-execution supplement, recorded 2026-07-02
> **Parent roadmap:**
> [`strongest-bot-program-2026-07-02.md`](strongest-bot-program-2026-07-02.md)
> **Scope:** current-machine wall-clock only. This document deliberately excludes
> person-week and staffing estimates.
> **Authority:** the parent roadmap still controls sequencing, strength gates,
> and promotion. This supplement records runtime measurements, defines which
> campaigns fit inside seven days, and turns Stage 0–1 into an executable first
> week.
> **Promotion policy:** no result in this document authorizes automatic promotion
> or deployment. Final gates are presented to the human operator.

## Decision summary

On the current machine, neural-network fitting is not the main schedule risk.
The measured schema-v2 MLP fits in roughly 1.5 hours. The expensive work is:

1. generating honest trajectories and strong counterfactual labels;
2. evaluating many action/world continuations;
3. running product-budget Grandmaster and RIS searches; and
4. completing the locked matched-deal and robustness matrices.

The correct compute split is therefore:

- **within one week:** Phase 0, a complete Enoch candidate evaluation, one
  Grandmaster primary gate, a deliberately sparse Track C campaign, a bid-only
  campaign, pilots for belief/RIS and structured models, or Expert v2
  distillation that reuses Grandmaster's existing search statistics;
- **longer than one week:** dense multi-world Q labeling, the complete
  Grandmaster final-claim package, full belief/RIS, whole-set kitty training,
  the structured checkpoint league, and the final robustness matrix.

These are compute envelopes, not promises that a candidate will win. A failed
or neutral gate is still a completed experiment and remains in the record.

## Runtime accounting contract

Unless a row says otherwise, estimates assume:

- Apple M1 Max, 10 CPU cores, 64 GB memory, and PyTorch MPS;
- release Rust binaries;
- six parallel data generators or eight to ten deterministic evaluation seed
  shards, as appropriate;
- exclusive use of the machine by one heavy CPU campaign at a time;
- code and experiment definitions are ready before the clock starts;
- the first attempt completes without process loss or artifact repair;
- development screens use deterministic node/world caps where reproducibility
  matters;
- serving claims use the intended wall-clock budget; and
- model/data contracts, frozen seeds, and the no-fallback checks already pass.

Running multiple full-search arms concurrently is not free. Prior measurements
showed roughly a 20% slowdown from contention. The preferred pattern is one
authoritative comparison sharded across the available cores, followed by the
next comparison.

The estimates ignore implementation time because the purpose of this record is
to identify compute and training bottlenecks. They also exclude open-ended
research loops after a failed idea.

## Measured local anchors

The following measurements calibrate the estimates below. They are runtime
evidence, not promotion evidence.

| Operation | Measured wall time or throughput |
|---|---:|
| Searchless 5,000-deal evaluation | about 2m40s per arm |
| Dense 5,000-hand decision metrics | about 2m49s per arm |
| Enoch at 12 ms, 100 mirrored pairs | about 73s |
| Enoch at 12 ms, 300 mirrored pairs | about 3m40s |
| Fixed 4-world Enoch, 200 pairs | about 8m |
| Fixed 4-world Grandmaster, 200 pairs | about 12–15m |
| Enoch at the real 2.2 s budget | about 81 CPU-seconds per mirrored pair in the product smoke |
| 4,000-game schema-v3 corpus, six generators | about 26.6–30h |
| The same corpus | 231,881 decisions, 7.92M rows, 617,167 Q rows, about 5.2 GB |
| Compose/hash the 7.92M-row corpus | about 16m |
| Train one full 80-epoch policy/V/Q MLP on MPS | about 1h31m |
| Train policy-only or policy+V with early stopping | about 1h17m each |
| 200-pair, 150 ms two-arm model screen | about 55m |
| 400-pair, 150 ms two-arm model gate | about 1h49m |
| 800-pair, 150 ms two-arm model gate | about 3h40m, linearly projected |
| 2,000-pair, 150 ms two-arm comparison | about 9h, linearly projected |
| Strong Expert@30 ms candidate-to-terminal continuation | about 0.87s per continuation |
| Easy candidate-to-terminal continuation | about 3.44ms per continuation |

The durable historical findings are in
[`grandmaster-and-value-head-findings.md`](../grandmaster-and-value-head-findings.md).
The local raw timing anchors used for the 2026-07-02 calculation are:

- `~/.shengji-expert-iteration-20260701T025210Z-00331cf83d39-r02/`
  `round-00-bootstrap-value-q/run.log` and `controller.log`;
- `~/.shengji-policy-ablation-20260702T153120Z-00331cf/`
  `variants/*/{training,screen}.log` and `final-gate.log`; and
- `~/.shengji-value-gm/run.log` and the per-shard generation logs.

The 4,000-game corpus is especially instructive. Easy shards finished at about
13.2 seconds/game, while search-driven mixture shards took about 169.6
seconds/game and Enoch-continuation shards about 191.9 seconds/game. Continuation
strength, not gradient descent, controls the campaign duration.

## What fits inside seven days

Each row is an independent seven-day-sized compute unit. The rows do not all fit
on the same machine during the same week.

| Roadmap work | Current-machine wall time |
|---|---:|
| Phase 0 hashes, contracts, and searchless/style baselines | under 1h |
| Ready, isolated Enoch ablation campaign at deterministic fixed work | 2–6h |
| Enoch product-budget qualification screens | 12–30h |
| Enoch 1,500–2,000-pair gate plus independent confirmation | 18–30h |
| One complete Enoch candidate evaluation package | about 2–4d |
| Grandmaster fixed-work development sweep | 4–12h |
| One frozen Grandmaster candidate's primary four-comparison gate | 2–4d |
| Five MLP policy/Q/V ablations from a frozen 8,000–12,000-game corpus | about 13–25h |
| Five development model screens plus one 800-pair finalist | about 8–18h |
| Week-bounded sparse Track C corpus, five MLPs, and development gates under the cheap-base contract below | about 5–7d |
| Belief/particle/RIS pilot | 1–3d |
| Bid-only outcome-model campaign | 2–5d |
| One structured-model architecture/seed screen | 2–5d |
| Search-controlled Enoch kitty-burial comparison | under 1d |
| Expert v2 distillation that logs and reuses existing GM visits/Q/uncertainty | about 2–7d |

For a Grandmaster candidate, the primary four-comparison gate means Enoch-1 and
Enoch-0 at equal and intended product compute. It does not include the final two
2,000–2,500-pair confirmations or the broad robustness matrix.

## What does not fit inside seven days

| Roadmap work | Current-machine wall time |
|---|---:|
| Complete Grandmaster development, locked gate, final confirmations, and continuity package | about 1–2w |
| Track C at 5–10% selection with the larger world/action settings | about 1–3w |
| Dense Track C labeling nearly every observation | about 37–224d before model fitting |
| Full persistent-particle belief and RIS-MCTS campaign | about 10–24d capped; 3–6w at product budgets |
| Complete-set kitty search and outcome model | about 7–18d |
| Bid + kitty + joint outcome evaluation | about 2–4w; potentially 4–8w at product continuation budgets |
| Structured model plus frozen-checkpoint league | about 2–4w reusing data; 4–8w with a fresh corpus and full product cross-play |
| Complete final Grandmaster/Expert product/equal-compute robustness matrix | about 1–4w, depending on the Grandmaster budget |
| PSRO, conservative offline RL, or ReBeL-style research | open-ended; no reliable upper bound |

## Track C is controlled by label geometry

Historical corpora contain about 57–58 eligible decisions/game. The initial
8,000–12,000-game plan therefore exposes roughly 455,000–696,000 observations
before subsampling.

For a selected-observation fraction `s`, the counterfactual work is:

```text
games × observations_per_game × s × actions × worlds
```

At the planned 6–12 actions and 8–16 worlds, that is 48–192 full continuations
per selected observation. Dense labeling would require roughly 21.8M–133.6M
strong continuations.

Using the measured Expert@30 ms continuation as a lower-bound proxy for a frozen
strong continuation, six generators yield approximately:

| Selected observations | 8k games × 48 continuations | 12k games × 192 continuations |
|---:|---:|---:|
| 1% | 0.37d | 2.24d |
| 5% | 1.8d | 11.2d |
| 10% | 3.7d | 22.4d |
| 25% | 9.2d | 56d |
| 100% | 37d | 224d |

The 5–12-hour base-trajectory allowance is valid only for a week-bounded corpus
that advances games with a cheap, capped heterogeneous behavior mixture and
invokes the frozen Grandmaster teacher/continuation only on selected
observations. It is supported by historical Q-off shards taking about 20–24
minutes per 100 games, rather than by the 26.6–30-hour Q-bearing campaign. A
preflight shard must confirm that projected base generation remains within this
allowance. If the campaign instead advances every trajectory with the prior
strong search mixture, scaling the measured 4,000-game run puts base generation
alone around 53–90 hours for 8,000–12,000 games, and Track C no longer fits in a
week.

After base generation, add 13–25 hours for the five MLP ablations and 8–18
hours for development gates. Grandmaster continuation may be slower than the
Expert proxy, so these are not upper bounds.

The week-bounded Track C contract is consequently:

- at 8,000 games and 48 continuations/observation, select no more than about 5%
  of observations; or
- at 12,000 games and 192 continuations/observation, select no more than about
  1% of observations.

The stricter limits reserve time for base trajectories, all five model fits,
development gates, and a Grandmaster-slower-than-Expert contingency. The 10%
and 2% geometries are counterfactual-generation boundaries only; they are not
whole-campaign one-week budgets.

Selection must be tactical rather than uniform: score thresholds, final-kitty
decisions, ruffs, throws, bombs, pair/tractor preservation, partner feeds, and
Enoch/Grandmaster disagreements. If the learning curve has not saturated, later
campaigns may add shards without changing the frozen teacher or locked test
seeds.

Using Easy continuation would make generation much faster but would change the
learning target. It is not a valid way to claim the strong-continuation campaign
was completed.

## Compute-only end-to-end envelopes

Assuming implementations already exist, candidates pass on their first attempt,
and the conditional research tracks are actually run:

| Program shape | Current-machine wall time |
|---|---:|
| Aggressively capped, tactically sparse label program | about 2–4 months |
| Full product-budget campaigns and robustness gates | about 3–6 months |
| Dense multi-world labeling of nearly every observation | about 5–13+ months |

The dense range is dominated by Track C. Open-ended PSRO/offline-RL/ReBeL work
is excluded. Failed scientific hypotheses add new campaigns and therefore have
no fixed compute bound.

## Safe compute parallelism

The current machine can accelerate work without stronger hardware by:

- sharding matched-deal seeds across eight to ten processes and aggregating only
  complete paired units;
- generating immutable corpus shards across six processes;
- training the five MLP objectives sequentially on the single MPS device;
- reusing one immutable corpus and split across all P/Q/V ablations; and
- logging Grandmaster visit distributions, Q, uncertainty, and final choices
  during the original search instead of recomputing them for distillation.

Do not run two product-budget comparisons at once merely because two arms exist.
They contend for the same CPU and can change time-bounded search behavior. Do not
generate final Track C labels before the Grandmaster teacher freezes, and do not
consume locked test seeds during development.

## Week 1 objective

Week 1 executes Stage 0 and Track A only:

> Freeze the scientific control, evaluate independent Enoch improvements, and
> either freeze a confirmed Enoch-1 or record that Enoch-0 remains the primary
> baseline.

No Grandmaster policy tuning, final Q labels, model training, or locked
downstream seeds are allowed before this objective resolves.

The planned machine workload is roughly 40–84 hours, leaving substantial buffer
inside a 168-hour week for a failed shard, one predeclared rerun, analysis, and
human review.

### What completing Week 1 buys us

Week 1 has a useful result on either branch; it is not a bet that every proposed
heuristic will win.

- If a candidate passes both locked gates, the output is one source-, binary-,
  configuration-, evaluator-, environment-, and seed-bound **Enoch-1**. It has
  shown a positive paired level-utility result twice on independent untouched
  samples, without failing the predeclared serving, correctness, equal-compute,
  or robustness checks. That is a defensible new yardstick for later Expert and
  Grandmaster work, subject to human promotion review.
- If no candidate confirms, the output is an equally explicit decision to keep
  **Enoch-0**. The failed, neutral, and interacting changes remain measured and
  attributable, so later work does not quietly rebuild a losing combination or
  mistake noise for progress.
- On both branches, the durable product is a reproducible control bundle, a
  calibrated fixed-work and product-budget evaluator, fail-closed paired-run
  orchestration, frozen seed namespaces, tactical correctness fixtures, a
  ranked independent-ablation record, and an auditable terminal decision. These
  reduce the setup and trust cost of every downstream strength experiment.

Week 1 does **not** train a neural model, complete Grandmaster, prove that the
chosen bot is globally optimal, guarantee a stronger candidate, or authorize a
production deployment. Its strongest claim is narrower and more valuable: by
the end of the campaign, there is either a statistically confirmed Enoch-1 or a
well-supported reason to retain Enoch-0, and the exact evidence for that choice
can be replayed.

## Week 1 phases

### W1.0 — Freeze the experiment

**Target:** first half-day; under one hour of compute.

1. Materialize Enoch-0, Expert-0, Grandmaster-0, and the evaluator from
   production reference `c813c8a`.
2. Record source SHA, release-binary hash, embedded/override model hashes,
   manifest/golden hashes, compiler and OS, hardware summary, every search knob,
   and the exact development and locked seed sets.
3. Clear ambient `SHENGJI_*` variables and record the effective child
   environment.
4. Prove policy, V, and Q selection is independent and that the intended model
   loaded without fallback.
5. Separate development seeds from both locked Enoch gate seeds.

**Exit:** an immutable control manifest and replayable command set. Any failure
stops the week before strength experimentation.

### W1.1 — Calibrate the evaluator and baselines

**Target:** Day 1; under one hour for searchless work, followed by a small
product-budget smoke.

1. Run frozen-policy equivalence checks and repeated capped searches to verify
   deterministic candidate ordering and tie-breaking.
2. Run searchless outcome and style baselines.
3. Run small 1-, 10-, and 100-pair product-budget smokes to measure seconds/pair,
   timeout rate, completed worlds, and safe shard concurrency.
4. Verify that incomplete orientations, cancellation, illegal actions, honesty
   violations, and model fallback fail closed.

**Exit:** a baseline report and a fixed worker/shard configuration for every
later phase.

The artifacts keep two control identities distinct. The **permanent scientific
control** is the exact `c813c8a` Enoch-0 source and binary. Paired feature
experiments execute a **runtime evaluation control**: the current frozen
evaluator with an empty feature set, sharing the evaluator's non-optional
correctness infrastructure with the candidate. W1.0 and W1.1 bind that adapter
to the permanent reference through source, model, configuration, probe, and
determinism evidence, but never claim that their binaries or fingerprints are
identical. Locked and terminal records retain both identities. If no candidate
confirms, the downstream production/reference baseline remains the permanent
Enoch-0; the adapter is only the matched experimental comparator.

### W1.2 — Run independent cheap Enoch ablations

**Target:** Day 1; 2–6 hours of compute.

Every arm changes one named behavior and passes its deterministic fixtures
before self-play. Correctness infrastructure that cannot sensibly be disabled
independently—physical-copy conservation, authoritative trick decomposition,
and mechanics-level exhaustive small-state enumeration—must pass its invariant
and completeness suites before any dependent policy arm runs.

The initial policy queue is deliberately split into independent arms:

1. bid-ownership evidence in determinization;
2. compound pair/tractor follow evidence;
3. failed-throw `better_player` evidence;
4. friend-revelation evidence;
5. exact terminal level utility when a rollout finishes;
6. current Enoch kitty burial versus the default point/shape burial;
7. late retained-trump ruff-shape planning;
8. contextual empty-trick value;
9. relative live-suit control and higher halters;
10. known team-void boss discounting;
11. teammate entry and return-suit planning;
12. low-trump handoff protection;
13. one representative per structural action family;
14. progressive admission beyond the initial top-K; and
15. uncertain but legal throw admission.

If an implementation hypothesis is genuinely inseparable, its protocol must
name the package and explain why its components cannot be switched and screened
separately. It is otherwise excluded from the independent-ablation stage.

Run 200–300 matched pairs at deterministic fixed work, plus dense decision
metrics. Preserve every result, including neutral and negative arms.

**Exit:** a ranked ablation table. No two unproven changes are combined.

### W1.3 — Run 800-pair survivor screens

**Target:** Day 1–2; about 3–10 hours depending on the number of survivors.

1. Advance only candidates that meet their predeclared development rule and
   have zero correctness failures.
2. Run 800 matched pairs on fresh development seeds.
3. Report paired level utility, point margin, win rate, bootstrap intervals,
   MDE, latency, completed worlds, style metrics, and all failure counters.
4. Require this evidence before a change can enter a combination candidate.

**Exit:** a small set of independently supported changes or an explicit
no-survivor result.

### W1.4 — Build and regress the Enoch-1 candidate

**Target:** Day 2; about 3–6 hours of compute after fixtures.

1. Combine only the independently supported changes.
2. Re-run the full tactical fixture suite, mechanics tests, hidden-information
   test, and model-contract tests.
3. Run a 200–300-pair combination screen followed by an 800-pair development
   screen on fresh seeds.
4. Compare combination behavior with the sum of the individual ablations; if a
   shared-scorer interaction cancels or reverses a gain, remove the conflicting
   change and record why.

Both combination comparisons use one non-null acceptance rule declared before
either result. A self-contained W1.4 decision must reconstruct both merged
results, the independently supported survivor set, and the interaction delta.
W1.5 cannot be materialized from a pre-run lineage manifest alone.

**Exit:** exactly one candidate hash and configuration for product-budget
qualification. If no combination survives, Enoch-0 remains the baseline and
Week 1 skips to W1.8.

The candidate cannot be called Enoch-1 if a parent-roadmap Track A correctness,
fixture, or legal-coverage prerequisite remains incomplete. In that case W1.8
records a provisional candidate and Enoch-0 remains the downstream baseline.

### W1.5 — Product-budget qualification

**Target:** Day 2–4; about 12–30 hours.

Use disjoint qualification seed namespaces derived from the frozen Week 1
master seed: `qual/intended`, `qual/equal`, and one namespace per robustness
stratum. None overlaps development or either locked seed set.

Run this fixed matrix:

1. 800 matched pairs against frozen Enoch-0 at the intended Enoch budget;
2. 800 matched pairs at equal compute on its own disjoint seed namespace;
3. 300 pairs across representative low, middle, and high ranks (100 each);
4. 400 cross-play pairs covering four frozen partner/opponent checkpoint
   assignments (100 each);
5. 300 pairs across three representative deck/player-count configurations
   (100 each);
6. 200 Finding Friends pairs across two representative revelation contracts
   (100 each);
7. 300 pairs across three representative scoring/kitty-multiplier rule sets
   (100 each); and
8. 200 threshold-enriched pairs across four landlord/attacker score situations
   (50 each).

This is 3,300 matched pairs. The equal-compute arm is retained even when its
configuration happens to match the intended-budget arm: it is an independent
claim with an independent seed sample, not an inferred duplicate. At the
measured Enoch product throughput, eight to ten seed shards put the matrix
inside the 12–30-hour allowance, including overhead.

The candidate advances only if:

- intended-budget and equal-compute signed level-utility estimates are positive
  and each paired-bootstrap lower 95% bound is at least `-0.02`;
- both comparisons have a positive point-margin estimate and no win-rate point
  estimate below `-0.02` relative to Enoch-0;
- the pooled robustness estimate is non-negative for level utility;
- no robustness stratum has signed level-utility estimate at or below `-0.10`;
- p50/p95 latency and completed-world throughput remain within the predeclared
  serving envelope; and
- every illegal-action, honesty, fallback, model-contract, and incomplete-pair
  counter is zero.

Serving-envelope gates apply to homogeneous candidate/control assignments. The
four mixed-checkpoint cross-play strata still report latency and work telemetry,
but do not gate on a team-attributed latency number that combines candidate and
control checkpoint decisions. Their outcome contribution and every correctness
counter remain fully gating.

Small robustness strata are diagnostic rather than separate superiority tests;
the pooled rule prevents the candidate from advancing on an aggregate known
regression, while the `-0.10` stop catches a large localized failure.

**Exit:** a candidate eligible for the locked gate. Do not tune on the product
qualification seeds after reviewing the result.

### W1.6 — Locked Enoch-1 gate

**Target:** Day 3–5; about 9–15 hours.

1. Freeze the candidate source, binary, model state, evaluator, environment,
   protocol, and first locked seed.
2. Run 1,500–2,000 matched-deal pairs against Enoch-0 across both role
   orientations.
3. Apply the parent roadmap's primary criterion: the paired-bootstrap 95%
   lower bound for signed level-utility delta must be greater than zero.
4. Apply every secondary correctness, latency, margin, win-rate, and robustness
   requirement without changing the candidate.

**Exit:** first locked pass or fail. A failure may not be repaired and rerun on
the same seed.

### W1.7 — Independent locked confirmation

**Target:** Day 4–6; another 9–15 hours.

1. Use the independently predeclared seed set with the identical frozen
   candidate and evaluator.
2. Run the same 1,500–2,000-pair protocol and superiority rule.
3. Reconstruct the published deltas and intervals independently from the raw
   paired observations.

**Exit:** confirmed Enoch-1 or a recorded non-confirmation. A promising first
gate without confirmation is not a freeze.

### W1.8 — Freeze or retain the control

**Target:** Day 6–7; negligible compute.

1. Preserve every fixture, screen, gate, trace, manifest, hash, and failure.
2. If both locked gates passed, freeze Enoch-1 as the primary downstream
   yardstick while retaining Enoch-0 as the permanent scientific control.
3. If no candidate confirmed, record that Enoch-0 remains both baselines.
4. Present the evidence to the human operator. Do not promote or deploy
   automatically.
5. Only after this decision may Stage 2 rebaseline Expert and Grandmaster or
   begin authoritative Grandmaster policy selection.

**Exit:** one immutable Week 1 decision record and an unambiguous downstream
teacher/baseline identity.

## Week 1 stop and invalidation rules

- A fixture failure blocks that ablation from self-play.
- Any illegal action, hidden-information leak, model-contract failure, silent
  fallback, incomplete paired unit, or mismatched artifact invalidates the run.
- A development loss is recorded and stopped; it is not rescued by combining it
  with another unproven change.
- Locked seeds are never used to tune, repair, or select a revised candidate.
- If the candidate hash, executable, environment, or evaluator changes after
  W1.6 starts, both locked runs are invalid and require a new predeclared
  protocol and unused seeds.
- Machine contention that changes effective search work invalidates the affected
  comparison; authoritative product comparisons run one at a time.

## Week 1 required artifacts

1. Frozen-control and environment manifest.
2. Reproducibility and fail-closed preflight report.
3. Searchless outcome and style baselines.
4. Fixture report for every attempted ablation.
5. Independent 200–300-pair ablation results.
6. Every 800-pair survivor result.
7. Combination and product-budget qualification results.
8. First locked gate protocol, raw pairs, comparison, and audit.
9. Independent confirmation protocol, raw pairs, comparison, and audit.
10. Either an Enoch-1 freeze manifest or a no-confirmed-candidate decision.

## Implementation and execution status — 2026-07-04

The Week 1 package is implemented, but there is no valid completed
authoritative W1.1 calibration and the 40–84-hour strength campaign has **not**
been run. Four authoritative starts were stopped and retained as described
below. There is therefore no Enoch-1 result, promotion claim, or evidence yet
that any arm improves playing strength.

The implemented package includes:

- the 15 independently selectable feature bits and their determinization,
  terminal-utility, kitty, tactical-value, structural-family, progressive, and
  legal-throw paths in [`core/src/bot`](../../core/src/bot/);
- mechanics, redaction, authoritative trick-format, physical-card-conservation,
  and exhaustive small-state fixtures;
- the strict in-process mirrored evaluator in
  [`enoch_eval.rs`](../../core/examples/enoch_eval.rs) and the cross-version
  control probe in
  [`enoch_control_probe.rs`](../../core/examples/enoch_control_probe.rs),
  including canonical Enoch play and bid ordering plus a repeated-process
  regression gate for the inherited hash-order leak;
- frozen seed, phase, comparison, shard, merge, qualification, locked-gate, and
  terminal schemas in [`enoch_week1.py`](../../training/enoch_week1.py);
- W1.4 survivor/combination lineage and the reconstructable post-run
  interaction decision in
  [`enoch_week1_campaign.py`](../../training/enoch_week1_campaign.py);
- W1.0 freezing, W1.1 preflight, sealed fixtures, file-backed external evidence,
  environment identity, bounded shard execution, and fail-closed merging in the
  remaining [`training/enoch_week1_*`](../../training/) modules; and
- the authoritative W1.0/W1.1 operator in
  [`enoch_week1_operator.py`](../../training/enoch_week1_operator.py), which
  requires a clean committed source tree, binds its commit provenance, seals
  fixtures before smoke evidence, and holds one machine-global campaign lock
  continuously across environment probing, attestation, evidence construction,
  dry-run validation, execution, and completion validation; and
- adversarial tests for identity drift, seed reuse, incomplete work, missing or
  invented evidence, scenario/mode drift, mixed evidence, omitted comparisons,
  forged decisions, and cross-phase candidate changes.

Local implementation verification completed with:

- 188 passing core tests with 2 pre-existing ignored tests, 77 passing mechanics
  tests, 12 evaluator tests, and 5 control-probe tests;
- 90 passing Week 1 Python tests;
- clean formatting, all-target compilation, and strict Clippy checks;
- 31 sealed tactical/global fixture cases over 45 source files, including the
  hash-order regression fixture, with zero failures
  (`000b57d7…fc969dab`);
- a verified non-authoritative W1.0 smoke bundle for the exact
  `c813c8ad6a43ef0599effbca098dec45c55e9aa8` reference
  (`784dd67d…ea240f0`);
- a deliberately partial and non-authoritative probe preflight
  (`3e3ef30d…6966dbf`) plus release deterministic-search authority
  (`b7da970b…d5b3f`); and
- one product-budget raw smoke revalidated through the final environment,
  typed-evidence, shard, and merge contracts, with one complete pair, zero
  failure counters, and merged fingerprint `d3a1da3b…0311c7b`.

Those short runs verify implementation and orchestration only. The 200–2,000
pair development and locked results, the 3,300-pair qualification matrix, and
the terminal Week 1 decision remain to be generated on unused authoritative
seed ledgers.

### Retired authoritative attempts — 2026-07-04

Four starts are preserved under `.enoch-week1-runs/` and must not be
overwritten or presented as campaign evidence:

1. `authoritative-2026-07-04-e5ab4ec` stopped during W1.0 before any seed
   claim because the host Python 3.9 `tarfile` API lacks the newer extraction
   filter argument. The freezer was corrected with an equivalent safe legacy
   extraction path.
2. `authoritative-2026-07-04-1019134` completed and sealed W1.0
   (`c92704b2…c563a10e`; phase `801dd6db…faea56`), deterministic-search
   authority, and all fixtures. W1.1 then consumed exactly the 100
   `preflight/frozen-policy-equivalence` seeds and failed before any product
   smoke. Repeated fresh-process diagnosis showed both reference and current
   binaries had the same four-output support. The mismatch came from inherited
   per-process `HashMap` order in Enoch throw candidates, floating-point bid
   score accumulation, normal/exchange bid ties, and the harness's forced-bid
   fallback—not from the new mechanics changes.
3. `authoritative-2026-07-04-f87cb65` stopped during W1.0 before any seed
   claim. The three-file patch's check/apply/reverse commands ran from an
   extracted reference directory nested inside the main Git worktree. Git
   treated that directory as a subdirectory of the parent repository, filtered
   every patch path, and returned success after changing zero files; the
   post-apply hash guard caught the no-op. The freezer now forces standalone
   apply semantics with a resolved `GIT_CEILING_DIRECTORIES`, and a nested
   parent-worktree regression test verifies that all three target hashes change.
4. `authoritative-2026-07-04-314935b` successfully sealed W1.0
   (`d6145b7a…014816`; phase `bce68e32…ac670a`) and entered W1.1 with an
   empty ledger. The host's Santa policy then blocked the deterministic-search
   release test executable because Cargo's default target was below the
   `/private/tmp` source worktree. This was a pre-claim execution-location
   failure, not a failed fixture. The authority builder now overwrites any
   ambient `CARGO_TARGET_DIR` with a canonical, manifest-bound target below
   the run root, outside the immutable W1.0 bundle; cached and offline
   verification require that exact target.

A non-authoritative host diagnostic then compiled the exact release fixture in
that run-root-style target. The explicit `/private/tmp` Santa denial was gone,
but AMFI killed the unsigned binary; ad-hoc signing was also rejected as an
unknown certificate chain, and `security find-identity -p codesigning`
reported zero valid identities. W1.1 is therefore externally blocked on this
managed Mac until the exact release test is approved through the Santa App or a
trusted code-signing identity is supplied. Do not initialize the reserved
`0x5eed202607040004` protocol before that prerequisite is satisfied.

The replacement source applies the same narrow deterministic normalization to
current code and to a separately built probe-only `c813c8a` reference. The
freezer stages and hashes every untouched production binary before applying
that bound three-file patch, records its before/after hashes, and requires eight
fresh subprocesses per binary to be byte-identical within and across versions.
An independent check over the entire 100-seed equivalence prefix was
byte-identical (`140d71d5…75e9e`).

Protocols `dea0964f…bd644f`, `4acdc5c1…0e2ed7e`, and
`b968b3be…6e04a8`, with master seeds `0x5eed202607040001` through
`0x5eed202607040003`, are retired. No namespace from those protocols may be
reused. The replacement reserves master seed `0x5eed202607040004` in a new
run root; each registry contains 35,111 seeds, and all four registries are
pairwise disjoint.

## Operator sequence

The authoritative run starts from a new output directory and never reuses the
diagnostic artifacts above. Capture every operator command in an external
durable transcript. If a stage fails, preserve an immutable failure tombstone
with its protocol, seed claims, failed stage, and disposition before any retry
or replacement; some failures occur before a normal phase artifact is
published.

1. From a clean committed worktree, run
   `python3 training/enoch_week1_operator.py init --root <run-root> --workspace <worktree> --master-seed <new-u64>`.
   This freezes all namespaces and creates the only authoritative empty ledger.
2. Run
   `python3 training/enoch_week1_operator.py freeze-w1.0 --root <run-root> --workspace <worktree>`.
   Retain the whole immutable bundle, source-provenance record, and W1.0 phase
   manifest, not only the control manifest.
3. With the machine otherwise idle, run
   `python3 training/enoch_week1_operator.py run-w1.1 --root <run-root> --workspace <worktree> --operator-id <id> --available-parallelism <measured> --maximum-smoke-workers 8 --attest-no-machine-contention`.
   The operator runs the release deterministic-search authority and complete
   fixture gate before the first preflight seed claim, then runs full-coverage
   preflight and the exact 1/10/100 product smokes with a 1/4/8 worker ramp. It
   refuses partial preflight, incomplete smoke evidence, a mismatched real Rust
   environment identity, or any ambiguous consumed seed.
   On this host the measured declaration is `--available-parallelism 10`;
   keep the machine awake and plugged in, and do not make the no-contention
   attestation while builds, tests, or another experiment are active.
4. Reconstruct the sealed W1.0/W1.1 package with
   `python3 training/enoch_week1_operator.py verify --root <run-root>` before any
   W1.2 arm consumes a seed.
5. Predeclare each comparison, launch, runtime identity, exact evaluator
   environment identity, machine attestation, and typed external-evidence
   bundle. Authoritative execution must acquire
   `enoch_week1_runner.authoritative_campaign_lock` and pass its live opaque
   token through the environment probe and runner API; the bare runner CLI
   intentionally refuses W1.1--W1.7 execution without that token. Supply the
   measured available parallelism and no ambient experiment overrides.
6. Build W1.3 decisions from actual fixture and merged-result artifacts. Build
   W1.4's pre-run lineage and post-run candidate decision from both combination
   results; W1.5 cannot be created from lineage alone.
7. Run the full W1.5 matrix, freeze both locked manifests before W1.6, execute
   W1.6 and W1.7 one comparison at a time, and reconstruct every decision from
   raw merged evidence.
8. Build W1.8 only from the contiguous phase chain. Present the terminal record
   to the human operator; it cannot authorize deployment.
