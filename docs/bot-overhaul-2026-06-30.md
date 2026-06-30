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
   throw. Sampling now conserves the configured physical multiset, persists
   attributed public history/voids/bids, treats exotic follows correctly, and
   pins rejected throw cards to their known holder.
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

The current 20-feature model is a scaffold, not a promotion candidate. It mostly
re-encodes constraints already known by the sampler and omits the signals most
likely to add value: attributed trick sequence, declarations, failed throws,
current-trick choices, and relative landlord/exchanger roles. Measure it against
the constrained-uniform sampler on both strength and completed worlds/second.

### 3. DAgger/reanalysis loop

The generator can collect Easy, Expert, Enoch, or mixed behavior while a stronger
teacher labels/reanalyzes. Run only one or two champion/challenger iterations:
collect current-policy states, train, paired-gate, freeze the winner. This reduces
train/serve state mismatch without allowing an unstable open-ended self-training
loop.

## Additional avenues worth exploring

### Public-history sequence model

Replace the aggregate belief features with a compact transformer or recurrent
encoder over attributed bids, tricks, failed throws, and current-trick actions,
plus a permutation-aware hand multiset encoder. Train card-location likelihood,
V and Q with a shared public-state trunk. This is the most plausible way for a
belief model to beat hard constraints rather than restate them. Keep legality
outside the network.

### Re-determinizing information-set search

The structural per-actor observation fix prevents a rollout actor from seeing
the root's sampled hidden cards, but a stronger experiment is RIS-MCTS: whenever
control changes seats, resample hidden cards from that actor's information set
while preserving public history. This reduces strategy fusion. It is expensive
and must be compared at equal wall-clock time, not equal simulations.

### League training and partner modelling

Homogeneous self-play overfits to one partner/opponent style. Maintain a small
league of frozen checkpoints and scripted policies, randomize seats, and train a
policy/value model against the mixture. Add an auxiliary latent partner-style
head or short-history embedding. Evaluate cross-play matrices and exploitability,
not only mirror win rate.

### Phase-specialized bidding and kitty agents

Play-card learning does not automatically solve bidding or burial. Build small,
separate candidate rankers using final level utility and exact phase features.
Bidding should predict both declaration value and the option value of waiting;
kitty training should compare several legal burials on the same deal with common
continuations. Promote each phase independently. The existing heuristic remains
a strong baseline and may win, which is a useful result.

### Distributional and risk-aware value

Predict a categorical distribution over level delta/score buckets rather than a
single mean. Search can then choose risk appropriate to role and threshold: a
defender protecting a lead and an attacker needing a multi-level swing should not
optimize the same variance profile. Calibration (Brier/ECE and reliability by
role) is required.

### Exact late-game oracle

Complete legal-move enumeration is now much better but still bounded for large
throws. Finish a mechanics-owned exhaustive enumerator, then add transposition-
cached alpha-beta for sufficiently small remaining hands. Use it first as a
teacher/target generator; serving it is optional. This would provide low-noise
tail values and detect heuristic/search mistakes.

### Conservative offline RL

Once action coverage is broader, try a conservative Q objective (CQL/IQL-style)
on legal candidate sets, with importance weighting by behavior propensity. This
can learn beyond exact teacher imitation while penalizing unsupported actions.
It is worth a bounded experiment only after propensities and multi-world Q targets
are reliable.

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
