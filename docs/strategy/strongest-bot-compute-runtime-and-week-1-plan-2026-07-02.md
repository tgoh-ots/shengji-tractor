# Strongest-bot compute runtime and Week 1 execution plan

> **Status:** compute-execution supplement, recorded 2026-07-02; authoritative
> Week 1 execution complete and independently reconstructed through W1.8 as of
> 2026-07-06; W1.4 ended `combination-regressed`, W1.5--W1.7 were skipped, and
> W1.8 retained Enoch-0
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
| One frozen Grandmaster candidate's primary two-to-four-comparison gate | about 1–4d |
| Five MLP policy/Q/V ablations from a frozen 8,000–12,000-game corpus | about 13–25h |
| Five development model screens plus one 800-pair finalist | about 8–18h |
| Week-bounded sparse Track C corpus, five MLPs, and development gates under the cheap-base contract below | about 5–7d |
| Belief/particle/RIS pilot | 1–3d |
| Bid-only outcome-model campaign | 2–5d |
| One structured-model architecture/seed screen | 2–5d |
| Search-controlled Enoch kitty-burial comparison | under 1d |
| Expert v2 distillation that logs and reuses existing GM visits/Q/uncertainty | about 2–7d |

For a Grandmaster candidate, the primary gate means the frozen downstream Enoch
baseline at equal and intended product compute, plus Enoch-0 at both budgets
only when it is a distinct identity. That produces two comparisons on the
active retained-Enoch-0 branch and four after a future confirmed Enoch-1. It
does not include the final two 2,000–2,500-pair confirmations or the broad
robustness matrix.

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

**Authoritative outcome (2026-07-06): complete and recovery-sealed.** Five
survivors ran for 800 fresh pairs each, totaling 4,000 pairs and 8,000
orientations in approximately `3h49m` of evaluation wall time on the same host.
Four changes are eligible for W1.4; this eligibility is not a superiority or
promotion claim. The detailed metrics, seal incident, recovery, and immutable
bindings are recorded in the authoritative execution status below.

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

**Authoritative outcome (2026-07-06): complete; candidate rejected.** The
four-change candidate passed the full regression gate and completed both
declared screens, but failed the common predeclared acceptance rule. The sealed
exit is `combination-regressed`, no candidate was selected, and W1.5 is
disallowed. Detailed metrics and immutable bindings appear below.

### W1.5 — Product-budget qualification

**Current applicability:** this phase is skipped for the active campaign because
the sealed W1.4 exit has `w1_5_allowed: false`. The protocol below remains the
contract for a future campaign that reaches W1.5.

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

**Authoritative outcome (2026-07-06): complete; Enoch-0 retained.** W1.8 sealed
`no-confirmed-candidate`, recorded the human-reviewed retain-control decision,
and left W1.5--W1.7 absent. Enoch-0 is both the permanent scientific control and
the downstream primary baseline. Stage 2 rebaselining is authorized;
production promotion and deployment are not.

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
7. Combination results and, only if W1.4 admits a candidate, product-budget
   qualification results.
8. If admitted, first locked-gate protocol, raw pairs, comparison, and audit.
9. If admitted, independent-confirmation protocol, raw pairs, comparison, and
   audit.
10. Either an Enoch-1 freeze manifest or a no-confirmed-candidate decision.
    The active run completed the latter in W1.8.

## Authoritative execution status — 2026-07-06

The active authoritative run is preserved at
`.enoch-week1-runs/authoritative-2026-07-05-d048b79`. It uses master seed
`0x5eed202607040004` and protocol
`a1e48199e6cb153c68f442cac9f28400798b994d154e03d34cb64420e21db2b7`.
W1.0--W1.4 and W1.8 are sealed and independently reconstructible. W1.4
completed all 1,100 declared pairs with zero invalidating counters, but the
candidate failed
its predeclared qualification rule. The sealed exit is
`combination-regressed`, no candidate was selected, and W1.5--W1.7 are not
allowed. W1.8 then sealed `no-confirmed-candidate`, retained Enoch-0 as both
baselines, and recorded human review. There is no Enoch-1. Stage 2 rebaselining
is authorized, but production promotion and deployment are not.

W1.0 sealed the production control
(`1aeb0c4f7d62d606eb554cf50aa83250a3672dd806587e695981196766e620f2`)
and phase
`6b2b63187981da256476256b673d2f809ed4ede3089790233a249b50fe5d937c`.
W1.1 then completed the deterministic-search authority, 100-seed
frozen-policy equivalence check, two 5,000-pair searchless baselines, all 31
fixtures, and the 1/10/100-pair product-smoke ramp. All 111 smoke pairs and 222
orientations completed with zero invalidating counters. The build-to-seal
interval on this host was about 2 hours 7 minutes; the individual product
smokes took about 3:05, 9:06, and 38:21. W1.1 selected eight workers/shards for
later phases and sealed report
`79d47b95f65c4c9b5377c5a0894abecd4d02069eb8c965176eb1f69587b8dc6f`
and phase
`24503b8909df96b6bb1cf47a91da3e830ea623c4ead6db98f9c4881b9043da63`.

W1.2 predeclared all 15 independent arms before claiming a seed, then ran each
for 300 mirrored pairs under the fixed 24-world, six-candidate, six-trick
configuration. The declaration-to-phase interval was 4:27:12
(`2026-07-05 15:49:36 PDT` through `20:16:48 PDT`) on this same machine. All
4,500 pairs and 9,000 orientations completed; every artifact-mismatch,
cancellation, fixture, hidden-information, honesty, illegal-action,
incomplete-pair, machine-contention, model-contract, fallback, and timeout
counter was zero. The five arms advancing to fresh 800-pair W1.3 screens are:

- `friend-revelation`;
- `uncertain-legal-throws`;
- `bid-ownership`;
- `team-void-boss`; and
- `compound-follow`.

The other ten arms are sealed as `stop-and-record`; their negative or
insufficient independent results remain part of the ranked table. Advancement
is only permission to run W1.3, not evidence of superiority or permission to
combine or promote an arm. W1.2 sealed declaration
`16c1a9d28d490beb624d313ed7aab0b177a2cbc458a8a07ff884b83599b29bc1`,
ranked table
`71b68e51cd46c4c6cb9a14b0893df046ef3beb7deba325e9e350a6c2b7ef5f06`,
14,711-claim immutable ledger snapshot
`2500ae1fbc5e1f03dc04b62ac92564be014f565443b1e6b0e337234cc2e598d8`,
and phase
`70d8a05c0e17a372e42c80e93d760ebf5d9fcb26c03c5474b06d5e3bbf214264`.
The offline verifier reproduced all four bindings from stored artifacts and
stored Git objects.

W1.3 then ran the five W1.2 survivors for 800 fresh mirrored pairs each: 4,000
pairs and 8,000 orientations total. Evaluation wall time was approximately
`3h49m` on the same host. Every pair completed, and all 11 artifact-mismatch,
cancellation, fixture, hidden-information, honesty, illegal-action,
incomplete-pair, machine-contention, model-contract, fallback, and timeout
counters were zero.

| Arm | Level utility (95% interval) | Point margin | Win-rate delta | p95 latency | Decision |
|---|---:|---:|---:|---:|---|
| `bid-ownership` | `+0.003125 [-0.010625, +0.01625]` | `+0.1875` | `+0.00125` | `231.2046 ms` | `advance-to-w1.4` |
| `compound-follow` | `+0.00375 [-0.031875, +0.03875]` | `+0.028125` | `+0.00625` | `237.5414 ms` | `advance-to-w1.4` |
| `friend-revelation` | `0.00000 [0.00000, 0.00000]` | `0.0000` | `0.0000` | `194.7366 ms` | `advance-to-w1.4` |
| `team-void-boss` | `-0.0075 [-0.038125, +0.02375]` | `-0.2375` | `-0.0075` | `209.0476 ms` | `stop-and-record` |
| `uncertain-legal-throws` | `+0.00875 [-0.006875, +0.02375]` | `+0.309375` | `+0.00125` | `211.9845 ms` | `advance-to-w1.4` |

The supported W1.4 input is exactly `bid-ownership`, `compound-follow`,
`friend-revelation`, and `uncertain-legal-throws`. `team-void-boss` is sealed
as `stop-and-record`. These decisions grant only permission to test the
four-change combination; they do not establish superiority and do not
authorize production promotion or deployment.

The original `171ee31a1528085f2378c8db211ba74ea25b9925` operator completed every
W1.3 evaluation and sealed the exact 18,711-claim ledger. During final sealing,
it mislabeled the external-evidence validator's counter map as a fingerprint,
wrote a semantically hashed but malformed supported-set, and stopped before a
phase was written. There was no retirement. The audited
`bc685a6f44961f32c80caacedf92a784a5bf0032` metadata-only recovery archived
the malformed file byte-for-byte, corrected exactly the five mislabeled fields
from their sealed external-evidence fingerprints, and wrote a recovery-bound
phase. It made zero new seed claims and invoked the evaluator zero times.

| Artifact | Fingerprint or file SHA-256 |
|---|---|
| Original W1.3 continuation provenance | `d947489a73996558289e0f2815ad3742d97c717ebc97f447a24a56f04a3ee16e` |
| Original W1.3 campaign declaration | `a9fe49ba40831844cafa8a56c9fe6663c659785385bac968d338486da676ac89` |
| Immutable W1.3 final ledger snapshot (18,711 total claims) | `999e43c97bd27daa372df882a0208a71862cd7bc484e8a87605df5363e38c897` |
| Archived malformed supported-set semantic fingerprint | `116b56f15c34d7dafe15579716d6adbc4b1a88e3eb0b75407d0a49e1f6ba0ee0` |
| Archived malformed supported-set raw file SHA-256 | `79cd17778ff68582b7886ec0f4e382d060175aedd67b94b84ac6462fcd6816f7` |
| Corrected supported-independent-change-set | `a6b2e8f0b79eb2199b141f68ce6d65a4716fbe6dbae2e2160738c0cd051ce025` |
| Corrected supported-set raw file SHA-256 | `e6021dee34db462c7d2a5b102de09377fe5b0bda360b51ae4595a9ca4ac2a6dd` |
| Seal-recovery provenance | `130bad03a3eff24ca57da15615df820f582b84f81e054e96bb7436bf6c02c267` |
| Seal-recovery manifest | `8cea1459b32e530e7211d69c5f7c0e4cad281eada081a64a9c29c97d6912a11e` |
| Recovery-aware W1.3 phase | `1059c449de8b4f181e1887f0064f4544ac8286b152094025246e11a3830b3a5e` |

### Measured W1.4 combination regression and screens

W1.4 evaluated exactly `bid-ownership`, `compound-follow`,
`friend-revelation`, and `uncertain-legal-throws` under source commit
`73d36330502f6a138c63ffa4092abd5447c28233`. The prerequisite regression gate
passed all 31 fixtures, the full mechanics suite, model contracts, strict
evaluator test, and frozen-model validation.

Both stages used the same predeclared rule: level-utility estimate at least
`0.0`, lower 95% bound at least `-0.05`, point margin at least `0.0`, win-rate
delta at least `-0.02`, candidate p95 latency at most `750 ms`, and zero
invalidating counters.

| Stage | Pairs | Level utility (95% interval) | Point margin | Win-rate delta | Candidate p95 | Rule result |
|---|---:|---:|---:|---:|---:|---|
| W1.4 qualification screen | 300 | `-0.006667 [-0.068333, +0.053333]` | `+0.116667` | `+0.008333` | `248.8802 ms` | Fail: estimate and lower-bound gates |
| W1.4 development screen | 800 | `+0.010625 [-0.023750, +0.046250]` | `+0.637500` | `+0.006875` | `249.1276 ms` | Pass |

Every pair completed and all 11 artifact-mismatch, cancellation, fixture,
hidden-information, honesty, illegal-action, incomplete-pair,
machine-contention, model-contract, fallback, and timeout counters were zero.
The four individual W1.3 estimates summed to `+0.015625`; the 800-pair
combination estimate was `+0.010625`, for a screen-minus-individual-sum
interaction of `-0.005000`. The exact rejection reasons are
`qualification:level-utility-estimate-below-rule` and
`qualification:level-utility-lower-95-below-rule`.

The declaration-to-phase interval was `1:18:14` (`2026-07-06 03:47:59 PDT`
through `05:06:13 PDT`), including `49:15` for the 800-pair screen. The offline
verifier then reconstructed the complete parent and W1.4 evidence in `5:10.98`,
without running comparisons or claiming seeds.

| Artifact | Fingerprint |
|---|---|
| W1.4 continuation provenance | `731b06228dc5a602b89a66d7e7902adf470a684e25ca802ffdc3cdbe184c34c1` |
| Campaign declaration | `2e8b9a6f530c5595b8b6ba32367fbc9fbcb971cb2e919f2fb8d907126db59765` |
| Campaign lineage | `b81c9709773431adea7e266d18badb2bcfb4f4fb4ce9d79e4b9444b5fee162fd` |
| Evaluated candidate | `abebed7aee684612282e37513b37592bf3fe64f96a34f597870fe72cf3bbe706` |
| Combination regression gate | `abef9967133b0c86af263f6550b7b9381abfbdcf6525c50209be5d5fa19c6241` |
| Qualification merged result | `d143d275bd559d4fe5d02949235b4c41a888dbd3298b6791651b27e9a07efd0a` |
| Screen merged result | `a804bdd6ca7ab76aaa4662a774440a6ac1ab84011ac0a22dbf11bb0401e43bd4` |
| Candidate decision | `cd80e82024d4d97b71f53d680883a5a0f462fa2792b13770355216a07ede95ae` |
| Immutable W1.4 ledger snapshot (19,811 claims) | `17d6eecf5a8119946de82853e34e314a1b3a4a4b18404c3209c217fd52c0fbea` |
| W1.4 exit | `d3a449cf29a98f14bcfdcdca7bafe1c5ba9bafbbebd56a21f9d4617f11eddf8b` |
| W1.4 phase | `ad353c33adda2a31dfb1a26c63e3802b8584aee0b711d457d8c1fa60dd88a399` |

### Measured W1.8 terminal decision

W1.8 ran as a metadata-only retain-control seal under source commit
`8bd141dea3fcacb015a6b5a48ebe99e90ae801ad`. It claimed zero seeds, invoked the
evaluator zero times, and kept the 19,811-claim ledger byte-identical to W1.4.
The authoritative seal took `7:45.60`; the separate offline read-only verifier
reconstructed the full lineage and terminal state in `5:44.72`.

The W1.8 seal binds 118 evidence fingerprints and the complete preserved
pre-W1.8 inventory: 5,103 files, 2,466,144,135 bytes, with fingerprint
`c3772664722f3fa46ca3b5d1489382c84fd3079a3a13bb006c38ba8923c267e4`.
The rejected evaluated candidate remains preserved, the selected-candidate
binding remains null, W1.5--W1.7 are absent, and Enoch-0 is both terminal
baselines.

| Artifact | Fingerprint |
|---|---|
| Retained Enoch-0 identity | `5243de72c6669e233c1528f42f5de8e4b578165f55b0964dd48c3326df551e62` |
| Rejected evaluated candidate | `abebed7aee684612282e37513b37592bf3fe64f96a34f597870fe72cf3bbe706` |
| W1.8 input plan | `430351610de11f15554643aa7e8d6dc499fd2f1834069f8d561b4892afce8289` |
| W1.8 continuation provenance | `d95a812b1391effbabbf52f2ca0b200a455db7eb6686b1b1c95b160b09584c32` |
| Immutable final ledger (19,811 claims) | `17d6eecf5a8119946de82853e34e314a1b3a4a4b18404c3209c217fd52c0fbea` |
| Human-review attestation | `b8745960c0d6d5d9af847c4bbfcfbcb708985ed80bccf96f5fa770cace138b32` |
| Terminal no-confirmed-candidate decision | `799be65915b7c60a280e03cbcd038e94bfc3af7340763c86dfe2b93070e5bd84` |
| W1.8 phase | `61eaba943d1ad417408f2ba3f190e1d4e5ea270bc840e582c6ea17c0c16a3cdf` |

Week 1 is complete. The immediate next step is Stage 2: rebaseline Expert and
Grandmaster against Enoch-0 and begin authoritative Grandmaster policy
selection. This result is rollout-evaluation evidence, not neural-network
training, and it authorizes neither promotion nor deployment.

## Historical implementation and execution status — 2026-07-04

The status below is retained as the incident record that led to the corrected
July 5 run. Its external-block statements are historical and are superseded by
the sealed W1.1/W1.2 record above.

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
7. If and only if the sealed W1.4 exit permits W1.5, run the full W1.5 matrix,
   freeze both locked manifests before W1.6, and execute W1.6 and W1.7 one
   comparison at a time. If W1.4 seals no candidate, do not materialize those
   phases and proceed directly to W1.8.
8. Build W1.8 only from the contiguous phase chain. Present the terminal record
   to the human operator; it cannot authorize deployment.
