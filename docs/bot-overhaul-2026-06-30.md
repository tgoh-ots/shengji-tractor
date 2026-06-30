# Bot system overhaul: findings, implementation, and next bets

## Outcome

The bot stack now supports a higher-yield training path than pure behavioral
cloning: honest, game-grouped Monte Carlo state value (V), sparse same-world
action value (Q), typed multi-head model contracts, constrained belief proposals,
and paired promotion tests. The existing policy model remains a compatible
fallback. Experimental learned heads are not promoted merely because they train;
they must beat the embedded control under matched deals and latency.

This matters because the old objective had an intrinsic ceiling. It asked an
honest observation to imitate a perfect-information action. Two identical public
observations can have different teacher-optimal moves because the hidden cards
differ, so contradictory labels are unavoidable. More data or a larger policy
MLP cannot remove that information-theoretic aliasing.

## Main flaws found

1. **Wrong learning target.** Top-1 imitation accuracy rewarded guessing a
   cheating teacher's exact move, not choosing one of several actions with the
   same expected outcome. It also provided no calibrated leaf value for search.
2. **Weak outcome semantics.** The former value path mostly represented point
   margin. Shengji is won in levels and thresholds, so a one-point and a
   one-level change were conflated. The new primary return is signed win plus
   level utility, with score bucket/team-win/kitty auxiliaries.
3. **Sparse and mismatched data.** Easy-driven trajectories differed from the
   positions where search serves the model. Candidate-Q coverage was narrow and
   naive random train/validation splits leaked decisions from the same game.
4. **Search strategy fusion and information leaks.** Simulated later actors could
   reason from a fully materialized sampled world. Knowledge is now reconstructed
   structurally for each observer, even inside a sampled world, without cloning a
   redacted state at every ply.
5. **Incorrect hidden worlds.** Earlier sampling could forget old public plays,
   infer false voids after legal rainbows/bombs, assume a standard deck for
   special decks, mishandle the hidden kitty, and ignore cards exposed by a failed
   throw. Sampling now conserves the configured physical multiset, uses
   attributed public history and proven voids, treats exotic follows correctly,
   and pins explicitly rejected throw cards to their known holder. It does not
   yet condition ownership on bids, compound pair/tractor follow implications
   beyond a proven void, or a failed throw's `better_player` witness.
6. **Biased/slow matching.** A greedy matching order was neither uniform nor
   clearly a posterior. Neutral worlds now use exact shuffle/zip or conditional
   rejection; constrained fallback uses augmenting matching plus bounded
   permutation MCMC. Weighted belief sampling is explicitly approximate and has
   a small enumerable calibration test.
7. **Search spent compute in the wrong places.** Wide candidate construction ran
   at every rollout ply, deadlines did not cover all work, and a cutoff could
   ignore a just-completed trick. Rollouts now use a 64-action/two-unit bounded
   generator, propagate a strict deadline, and flush completed tricks.
8. **Legal action coverage was fragile.** Candidate caps could starve rainbows,
   throws, or bombs; tractors and alternate follows were underrepresented; some
   fallbacks used default instead of configured rules. Families are generated
   independently and merged round-robin, with mechanics validation and low-cap
   regression tests.
9. **Card strength and phase mechanics had correctness gaps.** Ace ordering,
   boss-card logic, finding-friend selection, kitty theft, advanced bidding, and
   special decks all had edge cases. These now use mechanics-accurate comparisons,
   exact deck copies, and explicit phase ownership. Humans receive an explicit
   kitty-theft "done bidding" window that bots cannot race.
10. **Runtime races and duplicate work.** Concurrent room events could launch
    overlapping searches and apply a result to a newer state. The backend now
    has per-room singleflight with dirty reruns, a global search semaphore, and
    exact monotonic-version compare/apply.
11. **Evaluation could not support promotion.** Results were mostly homogeneous
    self-play summaries. The harness now supports configurable decks, modes,
    ranks and players; mirrored paired games; level utility; bootstrap intervals;
    minimum detectable effect; cross-play; version A/B; and decision-level
    diagnostics.
12. **Model artifacts were weakly typed.** Serving could confuse policy, point-V,
    level-V, and Q outputs. Manifests now bind schema, dimensions, semantics,
    output names, hashes, and golden vectors; Rust/tract parity validators test
    PyTorch-to-ONNX numerical agreement.

## Implemented training alternatives

### 1. Outcome-first Monte Carlo V/Q (recommended primary path)

Schema 3 records stable game IDs, the behavior policy, terminal signed level
utility, state-V masks, and sparse same-world counterfactual Q. Whole games—not
rows—are split between training and validation. The serving path can replace the
static leaf with typed level-V and blend calibrated action-Q into PUCT and final
root selection. Near-constant Q is ignored rather than min-max amplified.

Why it should yield more than cloning: it gives equivalent good moves similar
credit, learns the actual game objective, and puts the learned signal at every
search leaf rather than only in root ordering.

Current limitation: the pilot's Q target uses few candidates and one sampled
world. The next data run should use several constrained worlds per observation,
common random numbers across candidates, hard-negative coverage, and a frozen
control continuation.

### 2. Honest card-location belief proposal (experimental, default off)

The belief pilot predicts relative opponent/kitty destinations under hard
capacity and void masks. It cannot make an illegal world legal. Training and
runtime share one feature encoder; the model is restricted to its declared
four-player/two-standard-deck/Tractor contract, hash and class order are checked,
inference failures trip a circuit breaker, and `SHENGJI_BELIEF_WEIGHT` defaults
to zero.

Schema v1's 20-feature layout remains supported, although old artifacts need a
regenerated strict lineage manifest. Schema v2 now has a strict
128-feature contract: the v1 prefix plus public progress, four ordered bids and
eight ordered play events. The learned logits are per-card destination
marginals; serving multiplies them over physical-copy assignments, which misses
joint correlations and is not a calibrated posterior. It therefore remains
explicitly weight-gated and default off.

Dataset sidecars now hash the exact CSV, record the scripted behavior-policy
domain, exclude publicly pinned holding targets, and bind the exact Rust encoder
source hash. Model manifests preserve that lineage and are explicitly
research-only/non-promoting/safe-sidecar artifacts. Current golden vectors test
synthetic tensor and ONNX parity; state-derived encoder goldens remain future
work and the manifest says so directly.

A bounded whole-world particle cache also exists, but is independently default
off behind `SHENGJI_PERSISTENT_BELIEF=1`. Its reveal transition is not yet
multiplicity-weighted for duplicate physical copies, so retained-particle reuse
is a biased experiment rather than the neutral production sampler. The default
continues to draw a fresh constrained world. Measure both experiments against
that fresh baseline on strength, joint-assignment calibration, and completed
worlds/second before considering promotion.

### 3. DAgger/reanalysis loop

`training/expert_iteration.py` now runs a bounded, resumable version of this
loop. Each round fingerprints its source, configuration, league, prior model and
offline inputs; trains policy/V/Q; validates ONNX/tract parity; and compares the
candidate with the prior round in separate processes on matched deals. It never
promotes or deploys a model automatically.

## Second implementation tranche

- The generator supports four different seat policies in one hand, so league
  data can vary partners and opponents among Easy, Expert, Enoch, Grandmaster
  and mixed policies rather than varying only one whole-table policy.
- Existing seat/suit invariance is now audited without materializing duplicate
  49-feature rows. Augmented families remain one train/validation split unit.
- Replay-verified human/offline datasets have a fail-closed ingestion boundary,
  deterministic volume caps and ID namespacing. Observed actions become policy
  labels; counterfactual Q is stripped unless independently recomputed.
- Search has opt-in adaptive budgets, empirical CVaR/standard-deviation risk
  scoring, and a mechanics-validated exact alpha-beta oracle for fully
  materialized endgames up to hard card/node/deadline caps.
- Separate bid and kitty feature contracts, honest seeded exporters, listwise
  trainers and a strict Rust parity validator now exist. Runtime jointly
  normalizes the relative logits and blends candidate ranks, never raw absolute
  scores; bid/pass remains heuristic. Serving is restricted to the exporter's
  exact four-player/two-standard-deck Tractor support and requires both a model
  path and an explicit nonzero weight.

Detailed commands, manifests and safety boundaries are in
`docs/expert-iteration-training.md`. All new serving behavior is neutral or off
by default. The contract pilots described below are not strength evidence.

## Contract-only measurements

The small local runs used to validate plumbing produced measurable diagnostics,
but far too few independent hands for a playing-strength conclusion:

- The outcome/V/Q pipeline completed 2/2 hands (3,161 candidate rows, 121
  decisions). After one epoch its validation policy top-1 was 35.09%, state-V
  RMSE 0.3999, and action-Q RMSE 0.3640. PyTorch/tract parity passed; a one-pair
  A/B was exactly tied and is not an effect estimate.
- Belief schema v2 completed 2/2 hands (837 rows). After one epoch its row-level
  top-1 was 31.85% and NLL 1.2261; parity passed with worst absolute error
  `4.47e-8`. These marginal metrics do not validate the realized joint sampler.
- The phase exporter completed 8/8 hands (17 bid rows and 1,888 kitty rows).
  One-epoch validation was 0% top-1/NLL 1.3958 for the tiny bid split and 43.75%
  top-1/NLL 3.2706 for kitty. Both artifacts passed Rust parity and a one-pair
  runtime smoke, but the labels only imitate the existing heuristic.

None of these artifacts is embedded, configured on Fly, or eligible for
promotion. A useful strength result needs the predeclared multi-thousand-hand
generation and matched-deal confidence gate described below.

## Further research beyond the implemented tranche

### Public-history sequence model

Schema v2 supplies an ordered public-history tail to the current MLP. The next
experiment is a compact transformer or recurrent encoder over the complete
attributed history plus a permutation-aware hand multiset encoder, potentially
sharing a public-state trunk among belief, V and Q. Keep legality outside the
network.

### Re-determinizing information-set search

The structural per-actor observation fix prevents a rollout actor from seeing
the root's sampled hidden cards, but a stronger experiment is RIS-MCTS: whenever
control changes seats, resample hidden cards from that actor's information set
while preserving public history. This reduces strategy fusion. It is expensive
and must be compared at equal wall-clock time, not equal simulations.

### League training and partner modelling

Heterogeneous scripted/checkpoint seat leagues are implemented. The remaining
research step is a latent partner-style head or short-history embedding and a
larger frozen-checkpoint population. Evaluate cross-play matrices and
exploitability, not only mirror win rate.

### Phase-specialized bidding and kitty agents

The separate candidate rankers and heuristic-imitation exporters are now wired.
The important remaining work is outcome supervision: bidding should predict
declaration value and the option value of waiting; kitty training should compare
several legal burials on the same deal with common continuations. Promote each
phase independently. The existing heuristic remains a strong baseline and may
win, which is a useful result.

### Distributional and risk-aware value

Search can already optimize empirical lower-tail CVaR and variance across its
sampled returns, behind neutral-default knobs. A learned categorical or quantile
distribution over level delta/score buckets remains worth testing. Calibration
(Brier/ECE and reliability by role) is required.

### Exact late-game oracle

A bounded mechanics-validated alpha-beta oracle now solves fully materialized
small Tractor endgames and aborts rather than returning a partial value when a
deadline or node cap is reached. The next step is a canonical transposition
table and a dedicated oracle-label corpus; serving remains opt-in. Within honest
search it is exact only for each sampled perfect-information world, not for the
original information-set game, so it still has strategy-fusion risk.

### Conservative offline RL

Verified replay ingestion is implemented, but it intentionally performs
conservative imitation and removes unproven Q. Once action propensities and
multi-world Q targets are reliable, try a bounded CQL/IQL-style objective on
legal candidate sets to penalize unsupported actions.

### Small-game CFR as a scientific probe

Full Shengji CFR is impractical. A reduced-deck/two-trick abstraction can still
measure strategy-fusion and belief-model error, provide an approximate best
response, and validate whether RIS-MCTS or learned beliefs move toward the known
solution. Treat it as a diagnostic laboratory, not a production training path.

## Promotion sequence

1. Generate a release-mode schema-3 corpus with mixed strong behavior, multiple
   counterfactual candidates, and several shared worlds per observation.
2. Train policy + level-V + Q; require whole-game validation, calibration, and
   PyTorch/tract parity.
3. Run candidate and embedded artifacts in separate processes on identical
   mirrored deals. Gate signed level-utility delta and latency/world throughput,
   with bootstrap confidence intervals and a predeclared minimum effect.
4. Run variant and partner cross-play, legality/honesty tests, restored-room and
   concurrency tests, then a canary. Do not enable the belief proposal unless it
   independently improves the paired gate.
5. Preserve instant fallback to the embedded model and neutral sampler.

Tiny smoke models prove contracts, not playing strength. None of the pilot
artifacts generated during this overhaul should replace the embedded production
model without the quantitative promotion run above.
