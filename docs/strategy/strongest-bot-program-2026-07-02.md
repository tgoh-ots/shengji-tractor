# Strongest-bot program: Enoch, Grandmaster v2, and Expert v2

> **Status:** execution plan, recorded 2026-07-02
> **Production reference:** `c813c8a`, including the promoted schema-v2
> policy-only Expert model
> **Promotion policy:** train and evaluate freely, but never promote or deploy a
> model automatically. Present every final gate to the human operator.
> **Authority:** this is the current bot-strength roadmap. It supersedes the
> execution order, estimated gains, and recommendations in the older roadmap
> and handoff documents linked at the end.

## Objective

Build the strongest honest Shengji bot we can, without requiring it to play like
Enoch.

The desired product ladder is:

```text
Easy < current Expert < Expert v2 ≈ or > Enoch < Grandmaster v2 < Omniscient
```

There should ultimately be two distinct strong opponents:

1. **Enoch** — convention-heavy, partnership-aware, human-readable, and
   disciplined.
2. **Grandmaster v2** — calculation-first, posterior-aware, and willing to
   reject Enoch's playbook when expected level utility says to do so.

Expert v2 should be a faster learned approximation of the strongest search
system, suitable where Grandmaster v2's full compute budget is not practical.

## Strategic decision: improve Enoch first, while preserving the old yardstick

Enoch-1 will be completed and frozen before Grandmaster v2 development, model
training, or their authoritative gates. This is the better product sequence for
four reasons:

1. downstream systems are measured against the strongest validated opponent,
   not a baseline we already know how to improve;
2. Grandmaster currently inherits Enoch-derived proposals, so Enoch improvements
   can raise Grandmaster's floor rather than being reimplemented later;
3. the training league and Enoch/Grandmaster disagreement curriculum start from
   stronger decisions; and
4. it establishes how far a transparent human playbook can go before we spend
   heavily on learned systems.

Enoch-0 still cannot be discarded. Replacing every control with a changing
Enoch would make results difficult to interpret, so the program uses two
baselines:

- **Enoch-0:** the exact Enoch behavior at production reference `c813c8a`, with
  source, evaluator, environment, budgets, and hashes frozen. It remains the
  permanent scientific control for the whole campaign.
- **Enoch-1:** a separately developed candidate that incorporates independently
  proven playbook and mechanics improvements. After it passes its locked gate,
  it becomes the primary yardstick, search prior, league member, and downstream
  training reference. It never erases Enoch-0 results.

Every serious Grandmaster v2 or Expert v2 candidate will be selected against
Enoch-1 and will also report against Enoch-0. This gives us practical strength
against the best Enoch we have and continuity against a permanent control. If
no Enoch-1 candidate proves superior, Enoch-0 remains both baselines.

Changes to shared heuristics can affect both sides of an in-process comparison
and cancel out. Enoch experiments must therefore compare versioned policies or
frozen executables, not merely two tier names that share the changed scorer.

## Evidence that shapes this plan

The current deployed Expert is a 49-feature schema-v2 policy-only model. Its
independent 800-pair confirmation at a 150 ms budget measured candidate minus
the prior embedded model as:

| Metric | Estimate | Paired bootstrap 95% interval |
|---|---:|---:|
| Level utility | +0.040625 | [+0.001875, +0.080625] |
| Point margin | +1.1125 | [-0.221875, +2.4875] |
| Win rate | +0.023125 | [-0.001250, +0.048750] |

That is real evidence that policy learning can still help. It is not evidence
that the present value recipe works:

- A 200-pair policy-plus-state-V screen measured level utility `-0.0275`, point
  margin `-1.85`, and win rate `-0.0200` relative to policy-only.
- A 400-pair full policy/V/Q gate measured level utility `-0.02625` with 95%
  interval `[-0.085, +0.030]`, point margin `-1.9875` with interval
  `[-3.96875, -0.01875]`, and win rate `-0.01125`.

The next campaign must not repeat that V/Q recipe unchanged. In particular:

- current Q labels are too sparse and too dependent on one hidden world;
- mixed continuation policies make V/Q targets ambiguous unless policy identity
  is encoded or the continuation is frozen;
- state V is a poor unconditional replacement for a strong late search leaf;
- current schema features alias important post-play shape, throw, ruff, kitty,
  and team-void information;
- Grandmaster currently searches harder over much of Enoch's proposal and
  rollout space, so compute alone has shown steeply diminishing returns.

Two older positive/negative results also constrain the sequence:

- versioned evaluation measured the earlier Enoch playbook at about `+3.1pp`
  win rate and `+3.56` points versus frozen Legacy. Transparent rule changes can
  create real gains that symmetric tier comparisons hide;
- a 2,400-game DAgger policy-only re-distill was statistically tied with its
  embedded control. More of the same perfect-information policy cloning is not
  a standalone priority, even though the later schema-v2 policy-only campaign
  did produce the confirmed improvement above.

The most promising sequence is therefore Enoch-1 first, then stronger model-free
Grandmaster search, then multi-world action-value learning, and only then more
ambitious representation learning.

## Target designs

| Component | Enoch-1 | Grandmaster v2 | Expert v2 |
|---|---|---|---|
| Identity | Human playbook and conventions | Calculation-first apex bot | Fast learned approximation |
| Root proposals | Improved Enoch heuristics | Enoch safety proposals plus learned policy/Q and progressive widening | Learned policy, optionally learned Q |
| Hidden cards | Constrained sampled worlds | Stronger joint belief/particles; later RIS-MCTS | Same honest observation contract, lighter sampling |
| Search | Current search with proven tactical fixes | Paired candidate racing, deeper information-set search, exact terminal/endgame handling | Small fixed-budget search |
| Objective | Exact terminal level utility where available | Expected level utility with calibrated risk near thresholds | Distilled visits and multi-world Q |
| Rollouts | Enoch playbook | Partner/opponent-specific policy mixture | Cheap distilled rollout policy if useful |
| Style | Disciplined, conservative, legible | Tactical, adaptive, more willing to override conventions | Similar to Grandmaster v2 but less compute-heavy |

## Phase 0: freeze the experiment

Before changing play strength:

1. Freeze `Enoch-0`, `Expert-0`, `Grandmaster-0`, and the evaluator from
   `c813c8a`. Record source SHA, binary hash, model/manifest hashes, seed sets,
   exact environment, hardware, and all search knobs.
2. Preserve all earlier gate outputs, especially the first policy-only gate, its
   independent confirmation, and the adverse V/Q results. Never overwrite or
   extend a completed gate in place.
3. Establish direct baselines at both:
   - **equal compute**, to measure algorithmic strength; and
   - **intended product budgets**, currently roughly Enoch 1x and Grandmaster 3x,
     to measure the experience users will receive.
4. Harden independent head selection. Policy, V, and Q must be exportable and
   enabled separately; testing Q must not silently enable V.
5. Use a clean evaluator environment and predeclared protocols. No ambient
   `SHENGJI_*` knob may leak into an authoritative gate.
6. Record style baselines as well as wins: action disagreement, trump spend,
   speculative throws, partner overtakes, point feeds, bid timing, and kitty
   composition.
7. Remove nondeterministic candidate ordering and tie-breaking, including the
   known `HashMap` iteration leak. Until that is complete, fixed seeds are not
   byte-reproducible and all equivalence claims must remain distributional.
8. Preserve the established measurement mechanics: `version_ab` or frozen
   binaries for shared-scorer changes, `decision_metrics` on common observations
   with matched denominators, mirrored-deal-pair bootstrap intervals, Wilson
   intervals, minimum detectable effect, and both fast search-less and
   release-only search regression gates.
9. Make evaluator publication fail closed. Clear stale success/failure markers
   before preflight, freeze and hash the actual executable used by every path,
   preserve cancellation rather than falling back, and record the effective
   child environment and artifact lineage.
10. Prove that each candidate model loaded, passed its manifest/golden contract,
    and did not silently fall back before treating an A/B result as evidence.

## Track A: improve Enoch without losing its identity

This track should favor understandable rules with direct tactical justification.
Each item is an independent ablation before combinations are tested.

### A1. Correctness and information

- Add missing hard public constraints to hidden-world sampling: bid ownership,
  compound pair/tractor follow implications, failed-throw `better_player`
  evidence, friend revelations, and physical-copy multiplicity.
- Make actual trick decomposition authoritative for throws, ruffs, rainbows,
  bombs, and configured variants.
- Use exact terminal level outcomes whenever a rollout completes instead of a
  point-valued proxy.
- Preserve and extend the determinizer invariants: played cards are never
  re-dealt, configured physical-copy multiplicity is conserved, full-history
  void evidence survives state boundaries, and the end-to-end hidden-card
  leakage test remains green.

These changes are high priority because they improve every searched tier and do
not require a learned model.

### A2. Tactical playbook

- Complete the late ruff/kitty planner using post-play nontrumps, retained trump
  pairs/tractors, projected final-lead shape, actual kitty multiplier, and a
  separate estimate for throw failure versus a legal trump ruff.
- Replace absolute suit-length rules with relative live-suit control, higher
  halters, and team-specific voids.
- Make empty-trick value contextual: current pot points, score threshold,
  tempo, partner protection, hand shaping, and final-kitty exposure.
- Add explicit teammate-plan state: likely entries, return suit, inferred voids,
  point-feed opportunity, and whether an overtake would destroy a plan.
- Expand and fixture-test conventions such as return partner's suit, preserve an
  entry, trump handoff, and do not steal a locked partner.
- Re-run the Enoch kitty-burial comparison with a search evaluator. The earlier
  150-hand greedy audit found that `choose_kitty_enoch` buried about 19 points
  per hand and was slightly worse than the default zero-point/shape heuristic;
  repair this before proposing a learned kitty model if the result replicates.

### A3. Candidate coverage

- Represent every strategic action family before ranking within a family.
- Progressively admit lower-ranked legal actions instead of permanently
  truncating at the initial top-K.
- Search uncertain but legal throws rather than hiding all of them from strong
  tiers.
- Build a mechanics-level complete legal-move enumerator for singles, pairs,
  tractors, throws, rainbows, bombs, and configured variants. Validate every
  generated move through the mechanics legality checker and use exhaustive
  small-state completeness tests. Exact alpha-beta is blocked on this: a solver
  over the current heuristic candidate subset must never be described as exact.

### A4. Required tactical fixtures

Before self-play, pin the underlying decisions with deterministic fixtures:

- kitty multiplier follows the largest actual trick unit, not total card count;
- a clean ruff requires all trump and matching response structure;
- shedding the last nontrump can beat spending a low trump on an empty pot;
- a point-rich or threshold-critical pot can correctly override hand shaping;
- a low-trump handoff is suppressed when it destroys the only final ruff shape;
- a nontrump boss is discounted when a remaining opponent is publicly void; and
- weak-hand kitty burial protects points without breaking valuable shape.

### A5. Enoch-1 gate

Use dense decision metrics for quick tactical diagnosis, then matched-deal play.
Enoch-1 must beat frozen Enoch-0 rather than merely beat a shared Legacy scorer.
If a rule improves one tactical metric while regressing level utility, it stays
out or remains an explicit experiment. Only after the independent locked
confirmations pass is Enoch-1 frozen and allowed to become the primary yardstick
for every later phase.

## Track B: build model-free Grandmaster v2 after Enoch-1 freezes

This track begins only after Enoch-1 freezes. Grandmaster v2 inherits its proven
safety proposals and is selected primarily by direct play against it. The first
Grandmaster v2 milestone should be stronger without relying on a new neural
artifact. This creates a better teacher and tells us how much gain comes from
search rather than training.

### B1. Better allocation at the root

- Keep common sampled worlds across candidates to preserve paired variance
  reduction.
- Add sequential halving or confidence-bound racing so obviously weak actions
  stop consuming worlds while close actions receive more evidence.
- Add progressive widening over hierarchical action families.
- Pass actual learned or heuristic prior probabilities to PUCT experiments; do
  not substitute uniform mass over an already-pruned list.
- Separate proposal ranking, simulation allocation, and final decision. A Q
  estimate must not be counted in all three places with one fixed weight.
- Compare every racing/PUCT variant with flat common-world search at equal wall
  time; adaptive allocation can lose the flat search's paired-world variance
  reduction even when its visit pattern looks more selective.

### B2. Stronger continuation and endgames

- Use partner-specific and opponent-specific rollout policies rather than one
  greedy policy for every seat.
- Add a canonical transposition table, move ordering, iterative deepening, and
  forced-terminal detection to the exact endgame solver.
- Treat a solver that reaches only the last two or three cards as a useful
  correctness oracle, not an assumed material strength gain.
- Redesign and independently gate the nonterminal heuristic leaf using team
  control, likely opposing winners, trump control, point exposure, retained
  entries, role, and score threshold. The failed learned state-V experiment does
  not validate the existing own-high-card-heavy static leaf or its hoarding
  bias.
- Make candidate count, world count, and depth phase-dependent: posterior
  diversity early, action coverage and exact solving late.
- Tune risk by role and score threshold rather than applying one global
  variance/CVaR setting.

### B3. More search per second

- Profile full state cloning, feature construction, legal generation, and trick
  decomposition.
- Retain per-decision worlds-versus-time, timeout, and completed-search tracing
  so an apparent search improvement can be separated from doing less work.
- Move toward compact state plus apply/undo, incremental knowledge, cached
  evaluation context, and canonical feature/transposition caches.
- Batch model inference when learned leaves return.
- Reuse the public search tree and particle belief across moves where state
  versioning makes that safe.

Optimization is a strength project: additional correctly sampled worlds at the
same wall-clock budget are more valuable than increasing production latency
without bound.

## Track C: rebuild action-value learning around the actual decision

Once the model-free Grandmaster v2 search policy is frozen, use it as the
consistent data-generation teacher and continuation. Do not generate the final
Q corpus while that target is still changing.

### C1. Production corpus

Generate approximately 8,000–12,000 heterogeneous league games initially, then
scale further if learning curves have not saturated. For each selected honest
observation:

- sample 8–16 compatible hidden worlds;
- evaluate the same 6–12 actions in those worlds with common random numbers;
- use one declared frozen continuation policy per head;
- include the behavior action, search action, Enoch action, model/search
  disagreements, structural-family representatives, and hard negatives;
- evaluate every legal candidate in sufficiently small late states;
- record mean return, variance, sample count, win, point margin, level result,
  and final-kitty result.

Here `Enoch action` means frozen Enoch-1. Oversample score thresholds,
final-kitty decisions, ruffs, throws, bombs,
pair/tractor preservation, partner feeds, and states where Enoch and Grandmaster
disagree.

Every shard must have immutable lineage and integrity checks: globally unique
group IDs after concatenation; whole-game/family train-validation splits;
drop/action-coverage and teacher/search-outside-candidate counters; distinct
behavior, teacher, and serving budgets; sampler and continuation identity;
actual worlds, elapsed time, and timeout rate; source/model hashes; and a content
hash for each completed shard. Use a fresh fingerprinted workdir. Partial or
mismatched shards fail closed and are never resumed into a different campaign.

Use deterministic node/world caps where reproducible teacher labels matter, and
wall-clock limits where serving behavior matters. The old observation that a
teacher action was almost always inside the candidate set only showed that both
used the same generator; it did not prove complete legal-action coverage.

### C2. Objective and schema

- Train action-centered advantages and pairwise/listwise ranking, with a smaller
  absolute-Q calibration loss. The runtime needs to rank legal actions more than
  it needs an exact global scalar.
- Bump to a tactical schema that includes post-action nontrump count, side-suit
  count, retained trump pairs/tractors, exact candidate units, largest unit,
  relative suit control, suit-specific team voids, exchanger identity, kitty
  points, multiplier policy, and higher halters.
- Prefer a distributional or quantile Q target once the basic mean-Q experiment
  works, so search can reason about threshold and tail risk.
- Begin with an expanded MLP to test the target and data cheaply. Architecture
  complexity should not hide a bad learning target.
- For every schema, measure conflicting policy labels and conditional Q/value
  variance among identical or near-identical honest observations. This
  distinguishes representation aliasing from insufficient model capacity.
- Audit symmetry rather than blindly duplicating rows. Any augmentation must
  transform every led-suit/history-relative feature consistently, keep augmented
  families in one data split, and pass equivariance tests.

### C3. Required ablations

Train and gate these independently:

1. policy only;
2. policy plus Q;
3. policy plus state V;
4. policy plus Q plus V;
5. policy plus distributional Q.

Policy-plus-Q is the primary bet. V remains absent unless it wins its own gate.
If V is revisited, protect the winning policy trunk, calibrate by phase/role, and
blend or confidence-gate it instead of replacing every leaf unconditionally.

### C4. Search integration

Independently tune Q as:

- a root prior;
- an allocation/racing signal; and
- a final-selection blend.

Only combine uses that win separately. Search remains authoritative and the
model proposes or accelerates; it does not bypass mechanics or legality.

### C5. Bounded expert iteration

Only after the first P+Q candidate wins, run at most one or two predeclared
dataset-aggregation rounds on states reached by the current candidate. Freeze
and record the continuation policy, sampler, and prior champion for each round;
gate every round against that frozen champion. Do not run open-ended
self-referential retraining. Offline top-1 accuracy and loss can reject a broken
run, but they never select a champion.

## Track D: higher-ceiling belief and information-set search

Run this after the model-free search baseline is solid, and gate belief changes
independently from policy/value changes.

1. Fix persistent particles with multiplicity-correct conditioning, effective
   sample-size diagnostics, resampling, and rejuvenation.
2. Train on strong league and verified human trajectories rather than
   Easy-heavy play.
3. Replace independent card-destination marginals with a capacity-aware joint
   assignment model or whole-world scorer.
4. Encode complete attributed bid/play history with card embeddings.
5. Test stratified critical-card worlds and bid/kitty-informed posterior
   evidence before deploying a large model.
6. Experiment with RIS-MCTS, re-determinizing when control changes seats to
   reduce strategy fusion. Compare at equal wall time.

A belief candidate must improve joint-world calibration, effective independent
worlds per second, and downstream play. Row-level card-location accuracy alone
is not a promotion metric.

## Track E: bidding and kitty decisions

These phases have high leverage and should no longer be trained by merely
imitating the current heuristic.

The prior kitty evidence is a warning, not a final verdict: in a 150-hand greedy
audit the default burial heuristic had the best mean landlord margin,
`choose_kitty_enoch` buried about 19 points per hand and was slightly worse, and
naive minimum-point burial was about 25 points per hand worse than default. The
first experiment is the search-controlled Enoch burial replication in Track A.

- Search bid, pass, and wait as actions, including declaration timing,
  reinforcement, expected kitty, overbid risk, and downstream play value.
- Search complete burial sets with a structured/beam candidate generator and
  shared continuation seeds. Per-card additive labels cannot represent making
  a void or preserving a tractor.
- Train outcome-supervised bid Q and whole-burial-set kitty Q on signed level
  utility, with point/kitty auxiliaries.
- Ultimately evaluate declaration, burial, and play jointly rather than as
  unrelated phases.

Bid and kitty models are separate promotion units. Either may lose to the
heuristic and remain off without blocking the play-policy program.

## Track F: structured model and league training

After the multi-world Q target has proved useful with an MLP, test a structured
network:

- a permutation-aware set encoder for the hand;
- a candidate-unit encoder for singles, pairs, tractors, throws, rainbows, and
  bombs;
- a compact transformer or recurrent encoder for full public history;
- role, score, contract, kitty, and rule context tokens;
- separate policy and Q heads, plus an optional independently gated
  distributional V head.

Train against a frozen checkpoint league with heterogeneous partners and
opponents. Include Enoch-0, Enoch-1, prior Experts, Grandmasters, human-like
scripts, and candidate best responses. Evaluate cross-play and robustness, not
only homogeneous mirror self-play.

Verified strong-human replays may supply policy, history, and belief examples,
but counterfactual Q must be stripped unless independently recomputed. A small
active-labeling queue of high-regret, high-uncertainty, or Enoch/Grandmaster
disagreement states can collect pairwise choices from strong humans more
efficiently than indiscriminate replay imitation.

Longer-term research bets—population self-play/PSRO, conservative offline RL,
and public-belief/ReBeL-style training—belong here, after the supervised Q and
evaluation foundations are trustworthy.

## Track G: distill Grandmaster v2 into Expert v2

Once Grandmaster v2 is frozen and has beaten Enoch:

- collect its search visit distribution, multi-world action values, uncertainty,
  and final choices;
- train on soft action targets so strategically equivalent plays receive credit;
- emphasize Grandmaster/Enoch disagreement states and rare decisive tactics;
- optionally distill a very small rollout policy as a separate artifact;
- retain a modest honest search budget at serving time.

Expert v2 is successful if it approaches Enoch or Grandmaster v2 strength at a
substantially smaller compute budget. It does not need to replace the apex bot.

## Evaluation protocol

### Development screens

- Use 200–300 matched-deal pairs for cheap, predeclared ablations.
- Require 800 pairs before combining a change with other unproven changes.
- Use dense tactical metrics and fixture suites to diagnose why a change moved,
  but never substitute them for game outcome.
- Bootstrap over matched deal-pairs, not individual hands, and always report the
  minimum detectable effect. Offline top-1, V/Q error, calibration, and Q rank
  are diagnostics rather than strength criteria.

### Locked strength gates

For Enoch-1, run 1,500–2,000 matched-deal pairs against frozen Enoch-0, covering
both seat/team orientations. It becomes the primary yardstick only after an
independently seeded confirmation passes the same predeclared superiority rule.

For Grandmaster v2 and Expert v2, run 1,500–2,000 matched-deal pairs against
frozen Enoch-1 as the primary gate and against Enoch-0 for historical continuity.
Run both intended product budgets and equal-compute budgets.

For the final claim that a bot is stronger than Enoch, run two independently
seeded confirmations of 2,000–2,500 pairs against frozen Enoch-1, plus the
predeclared Enoch-0 continuity comparison. After the protocol is locked, do not
tune on those seeds.

Primary success criterion:

- the lower bound of the paired-bootstrap 95% interval for signed level-utility
  delta is greater than zero.

Secondary requirements:

- positive point-margin estimate and no material win-rate regression;
- zero illegal actions, honesty violations, model-contract failures, and silent
  fallback;
- acceptable p50/p95 move latency and completed-world throughput;
- robustness across landlord/attacker role, seat, rank, representative score
  thresholds, partners, frozen opponent checkpoints, decks/player counts,
  Finding Friends, and representative configured scoring/multiplier rules.

Report uncertainty even when the gate fails. A failed or neutral result remains
part of the experiment record and may not be overwritten by a later run.

### Style gate

Strength alone does not establish a second playstyle. After collapsing truly
equivalent actions, report:

- meaningful action disagreement with Enoch;
- percentage of roots where search overrides the Enoch prior;
- trump conservation/spend, throw aggression, and threshold risk;
- partner overtake, point-feed, suit-return, and handoff rates;
- bid timing and burial-shape differences.

The target is roughly 10–15% meaningful decision divergence while still beating
Enoch. This is a characterization target, not permission to choose weaker moves
for novelty.

## Execution dependencies

The strength work is deliberately Enoch-first. Only infrastructure that cannot
influence downstream policy selection runs alongside it:

| Stage | Work | Dependency |
|---|---|---|
| 0 | Freeze controls; harden evaluator, lineage, determinism, and style metrics | Starts immediately |
| 1 | Enoch A1–A5 ablations and locked Enoch-1 confirmation | Starts after Enoch-0 freeze; no downstream strength campaign yet |
| 2 | Rebaseline Expert/Grandmaster against Enoch-1; build model-free Grandmaster v2 | Starts only after Enoch-1 freezes |
| 3 | Corpus generation and P/Q/V training | Starts only after the Grandmaster v2 teacher freezes |
| 4 | Belief, bid/kitty, structured-model, and bounded-iteration experiments | Starts after the relevant simpler baseline is locked |
| 5 | Expert v2 distillation and final confirmation | Starts after a winning Grandmaster v2 freezes |

Evaluator and corpus-tooling reliability work may be prepared early, but it may
not generate final labels, tune downstream policies, or consume locked test
seeds before its upstream baseline freezes.

## Recommended order of execution

1. Freeze Enoch-0 and the complete evaluation contract.
2. Build, ablate, and independently confirm Enoch-1; freeze it as the primary
   yardstick.
3. Rebaseline current Expert and Grandmaster against Enoch-1.
4. Build and gate model-free Grandmaster v2; freeze the winning teacher.
5. Generate the multi-world, common-random-number Q corpus.
6. Train and gate policy/P+Q/P+V/P+Q+V independently.
7. Integrate the winning heads into Grandmaster v2 without double counting.
8. Run belief/RIS-MCTS and bid/kitty outcome-supervision experiments.
9. Test bounded expert iteration, the structured encoder, and the heterogeneous
   frozen-checkpoint league.
10. Distill the final Grandmaster v2 into Expert v2.
11. Run locked confirmations against Enoch-1, preserve the Enoch-0 continuity
    results, characterize style, and present all evidence to the human operator.

No step in this document authorizes automatic promotion or deployment.

## Related documents and precedence

The documents below remain useful as evidence or technical references, but none
of their older future-work ordering overrides this plan.

- [Policy-only Expert promotion](policy-only-promotion-2026-07-02.md)
- [Human refinement evaluation](human-refinement-evaluation-2026-06-30.md)
- [Bot overhaul and alternative training paths](../bot-overhaul-2026-06-30.md) — historical implementation audit
- [Action-value training design](../action-value-training.md) — technical rationale
- [Expert-iteration campaign contract](../expert-iteration-training.md) — operator/tooling reference
- [Grandmaster and value-head findings](../grandmaster-and-value-head-findings.md) — historical experiment evidence
- [Committed evaluation baseline](../bot-eval-baseline.md) — historical baseline and harness reference
