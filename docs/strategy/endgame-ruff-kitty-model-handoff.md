# Endgame ruff, throw, and kitty strategy: model handoff

> Session handoff from 2026-06-30. This note follows the unsafe Joker-throw fix
> in commit `4f9e5a8` and records a subsequent strategy discussion. It separates
> exact engine rules from useful human heuristics, audits current bot coverage,
> and gives the next implementation session a prioritized plan.

## Executive summary

The discussion contains a real strategic lesson: late in the hand, the
landlord/banker team may deliberately concede point-free or low-value tricks,
shed its remaining nontrumps, and preserve enough trump **quantity and shape**
to ruff a large final nontrump lead. This is different from Bridge's reflex to
draw trump: Shengji rewards point capture, scoring thresholds, and the last
trick's multiplied kitty, not raw trick count.

The current bot only models this partially. It preserves expensive trump
locally and has a simple kitty-protection rule, but it does not plan toward a
post-play all-trump hand, retain a specific trump pair/tractor shape for the
final response, or value the actual configured kitty multiplier. This is worth
adding first to the Enoch heuristic/search path, because Grandmaster and
Omniscient use Enoch-derived root proposals by default. Omniscient also uses
Enoch rollouts, while Grandmaster deliberately uses neutral plain-heuristic
rollouts. Updating only the learned Expert ONNX model would not fix those named
tiers.

Do **not** train the literal chat claims without the mechanics corrections
below. In particular, multiplier depends on the largest trick unit, not total
cards, and merely holding one card of the thrown suit does not automatically
invalidate a throw.

## Exact mechanics to preserve

### Clean ruff of a nontrump lead

Under the default rules, a follower must play the required number of cards from
the led effective suit when able. If short, it must play every led-suit card it
holds before filling the remaining slots off-suit. A clean ruff therefore
requires the follower to be completely void in the led suit and to answer with
all trumps. See `mechanics/src/trick.rs`, `TrickFormat::is_legal_play`.

Raw trump count is not always enough. The response must also match the lead's
format:

- An all-single throw can be ruffed by the same number of trump singles.
- A pair requires a trump pair.
- A tractor requires a matching trump tractor.
- A compound such as tractor plus singleton requires the corresponding trump
  structure for every component under the default `ThrowEvaluationPolicy::All`.

The engine has a direct tractor-plus-single ruff example in
`mechanics/src/trick.rs` around `test_play_throw_tractor_with_other_tractor_in_game`.
Alternative throw-evaluation or bomb settings can change winner comparison, so
new logic must read propagated settings rather than assume every table uses the
defaults.

### Throw invalidation and ruffing are separate events

A compound lead is checked for invalidation immediately. For a regular throw,
the engine examines opponents' cards in the thrown effective suit:

- A singleton/repeated unit is halted only by a strictly higher same-suit card
  with at least the required multiplicity.
- A tractor is halted only by a strictly higher tractor satisfying at least the
  required tuple width and run length.
- On failure, only the threatened component is actually led; the other attempted
  cards return to the leader and are recorded in `bad_throw_cards`.

See `mechanics/src/trick.rs` in the regular-throw validation beginning near the
`Regular throw` comment.

Consequences:

- "If anyone still has one card of that suit, the throw is invalid" is only a
  rough shortcut for a weak all-single throw. The card must actually beat a
  vulnerable component; a lower card does not halt it.
- A void opponent's trumps do not invalidate a side-suit throw. The throw may
  survive validation and then lose to a legal all-trump ruff.
- The model should represent two different probabilities: **throw survives
  same-suit halting** and **surviving lead avoids a trump ruff**.
- Holding most of a side suit is useful evidence, not a proof. Exact safety
  depends on the remaining higher cards and structures.

### Last-trick kitty multiplier

`Trick::complete` reports `largest_trick_unit_size`. `PlayPhase::finish_trick`
then computes:

- `KittyPenalty::Times`: `2 * largest_trick_unit_size` (the default).
- `KittyPenalty::Power`: `2 ^ largest_trick_unit_size`.

This is implemented in `core/src/game_state/play_phase.rs`; it is **not** based
on the total number of cards in a compound throw.

| Final lead shape | Largest unit | Default Times | Power |
| --- | ---: | ---: | ---: |
| Four unrelated singletons | 1 | 2x | 2x |
| Two unrelated pairs | 2 | 4x | 4x |
| One triple | 3 | 6x | 8x |
| Four-card tractor or four-of-a-kind | 4 | 8x | 16x |

Thus the chat's "four cards means 8x" is true for a four-card tractor/quad under
the default rule, not for four unrelated singles. Its "2^n gives 8x for four"
statement is arithmetically and mechanically wrong.

There is also a wording mismatch worth fixing separately: UI/message text that
says "size of the last trick" is misleading if the intended rule remains
largest component size.

### The real objective is points and levels

Point-free tricks can often be conceded cheaply. Winning tempo still matters,
but spending trump merely to collect an empty trick can be worse than discarding
a nontrump, preserving final-trick structure, and protecting a multiplied kitty.
Conversely, this is not a blanket "never draw trump" rule: drawing trump remains
correct when it secures point tricks, protects a threshold, or removes the
opponents' only final-ruff resource.

## Tactical synthesis from the discussion

An offensive last-trick setup can look like this:

1. Keep a boss/Joker or another reliable entry for the penultimate trick.
2. Retain a long, controlled nontrump suit or structured unit for the final lead.
3. Account for higher same-suit cards/structures so the final throw is not
   halted.
4. Account separately for a defender who may have shaped down to an all-trump
   response with the matching structure.

The landlord/banker team's counter-plan can look like this:

1. Use the kitty burial to void a side suit where sensible.
2. During mid/late play, dump remaining nontrumps on low-value losses.
3. Preserve enough trump cards and, where needed, trump pairs/tractors to match a
   likely final lead.
4. If the deal is too weak to defend the last trick, bury zero or very few kitty
   points instead of creating a large multiplied liability.

The last point is already substantially represented by Enoch's weak-hand kitty
point budget. The missing part is the multi-trick hand-shaping plan.

## What the current bot already handles

- Commit `4f9e5a8` prevents unsafe Joker compounds from inheriting Joker/boss
  strength, filters speculative Joker throws from honest rankings, and removes
  known failed throws from perfect-information and exact search.
- `score_follow` generally avoids wasting high trumps/Jokers on lost tricks and
  protects trump pairs when a singleton ruff is sufficient.
- Enoch's kitty policy tries to create a complete side-suit void, protects trump
  and pairs, and gives a clearly weak hand a zero-point burial budget.
- Enoch has long-suit, known-void, partner-ruff, trump-drain, and late kitty
  protection heuristics.
- Grandmaster and Omniscient use full-hand rollouts, so a rollout that discovers
  the right line receives the exact terminal kitty result from the mechanics.
- Training schema 3 already records actor-team final-trick/kitty-win auxiliary
  labels, and terminal level utility already includes the true kitty swing.

Do not replace these with a hand-written ruff reward. The terminal objective is
already correct; the main failures are candidate representation, local policy,
and failure to discover the setup line.

## Concrete gaps found in the audit

### 1. No explicit post-play hand-shape value

The scorer does not ask:

- How many nontrumps remain after this candidate?
- Does this play leave an all-trump hand?
- How many side suits remain?
- Did it preserve or destroy the trump pair/tractor needed for a plausible final
  response?

Current discard ordering often keeps trump incidentally, but it cannot compare
multi-trick plans that differ mainly in final hand shape.

### 2. Empty tricks are still overvalued

In `score_follow`, taking a trick starts with a flat `+6`; an empty-pot ruff only
subtracts `3` before strength costs. This can still favor spending a low trump on
zero points instead of conceding and shedding a nontrump. The control reward
should become contextual rather than disappear entirely.

### 3. Kitty protection assumes only 2x

Both Enoch lead and follow protection compute `at_stake = kitty_points * 2`.
That is correct only for a largest unit of one. It misses pair/tractor finales
and ignores `KittyPenalty::Times` versus `Power`.

For an actual final-trick candidate, compute the exact unit decomposition and
configured multiplier. Earlier in the hand, use a conservative projected
multiplier based on retained structure rather than raw card count.

### 4. The late low-trump handoff can conflict with kitty defense

Enoch gives a low-trump handoff bonus when four or fewer cards remain. That can
spend one of the exact trump cards needed for the final all-trump response. Gate
the handoff when a valuable kitty is live, the player still has a nontrump to
shed, or the trump belongs to a retained pair/tractor.

### 5. Raw length is used as a proxy for structure

One Enoch branch treats any nontrump play with length at least four as a tractor,
and generic lead score scales with total length. Four singles, two pairs, and a
four-card tractor have different throw risk, ruff requirements, and kitty
multipliers. Decompose candidates with mechanics and score actual units.

### 6. Same-suit boss status ignores ruff risk

An uncatchable nontrump card is only boss within its own suit. If an opponent is
publicly known void, that card may be ruffed. Candidate scoring needs a separate
known/probable ruff-risk term, especially for weak singleton leads and final
throws.

### 7. Side-suit control is absolute, not relative

The long-suit bonus uses a fixed own-card threshold. Add an honest estimate of
the actor's share of all still-live cards in the candidate suit, remaining higher
halters by unit shape, and candidate-suit void facts by team. This better models
"I own most of this suit" without pretending it proves safety.

### 8. Learned features alias the tactic away

The shipped Expert model remains schema-v1 with 36 policy-only features. Schema
v2 adds progress and coarse duplicate/void aggregates, but still lacks candidate-
suit hand distribution, post-action all-trump shape, exact unit structure,
kitty points/multiplier, and exchanger identity. Its current void features refer
to the already-led suit and are zero while choosing a lead; aggregate void count
loses suit and team identity.

Grandmaster and Omniscient do not use the learned Expert policy by default.
Both use Enoch-derived root proposals; Omniscient defaults to Enoch rollouts,
while Grandmaster defaults to neutral plain-heuristic rollouts. Heuristic/search
work must therefore precede retraining.

### 9. Exact endgame is not currently a production backstop

The exact solver is opt-in (`SHENGJI_EXACT_ENDGAME_CARDS` defaults to zero) and
hard-capped at 12 total remaining cards. A four-card-per-seat finale has 16 cards
and is ineligible. Benchmark before raising the general cap; a dedicated
terminal-candidate fast path may be cheaper than broadening minimax.

## Recommended implementation order

### Phase A: heuristic and search behavior

1. Add a mechanics-backed candidate-shape helper that reports actual units,
   unit count, largest unit, tractor/repeated shape, and configured kitty
   multiplier when terminal.
2. Add a post-play own-hand summary: remaining nontrumps, all-trump flag,
   remaining side-suit count, trump count, and retained repeated/tractor capacity.
3. In late play, reward shedding a nontrump and completing an all-trump remainder
   when kitty/threshold leverage justifies it. Penalize breaking the only matching
   trump structure.
4. Reduce the flat reward for ruffing a zero-point trick; preserve strong rewards
   for point pots, threshold crossings, partner protection, or necessary tempo.
5. Replace hard-coded 2x kitty protection with actual/projected multiplier-aware
   exposure.
6. Gate the low-trump handoff against final-hand-shape and kitty-risk signals.
7. Add candidate-suit ruff risk and relative live-suit control. Keep throw
   invalidation risk and post-validation ruff risk as separate values.
8. Consider a dedicated exact terminal fast path for candidates that empty the
   leader's hand and force all followers to play their entire hands. Benchmark on
   the production one-vCPU shape before enabling a wider exact solver.

### Phase B: learned schema v3

Append new features; do not reinterpret the frozen v1/v2 prefix:

- Cards remaining after candidate; candidate empties hand.
- Post-play nontrump count/fraction and all-trump flag.
- Remaining number of side suits and shortest nonempty side-suit length.
- Remaining trump count and repeated/tractor capacity.
- Candidate unit count, largest unit size, tractor flag, and all-trump response
  flag when following.
- Actor is exchanger; kitty value is exact versus estimated.
- Honest exact/expected kitty points.
- Times versus Power policy and exact/projected multiplier.
- Candidate-suit own length/share of live cards.
- Candidate-suit teammate/opponent known-void counts.
- Remaining higher halters for each relevant unit shape.

Candidate-relative, suit-symmetric summaries are preferable to fixed suit
one-hots.

Keep terminal level utility as the primary target. For tactical late states,
evaluate all candidate Q values rather than the usual sparse cap, and oversample
or construct fixtures spanning:

- Zero-point versus high-point kitty.
- One, two, and four-card largest units.
- Times and Power multiplier policies.
- A defender one discard away from all-trump.
- Trump singles versus a required pair/tractor.
- Safe-in-suit leads with a known-void opponent.
- Weak final throws with one possible same-suit halter.

The existing final-trick/kitty-win label can remain as an auxiliary; add a
signed kitty-swing or multiplier bucket only if diagnostics show it helps.

### Phase C: evaluation and guardrails

Add deterministic decision fixtures before expensive self-play:

1. Four-single final throw scores 2x, while a four-card tractor/quad scores 8x
   under default Times.
2. A clean ruff requires all trump and matching structure.
3. With a valuable kitty and an empty current pot, shedding the last nontrump
   beats spending a low trump.
4. With a point-rich or threshold-critical pot, taking the trick can correctly
   override hand shaping.
5. A low-trump handoff is suppressed when it destroys the only final ruff shape.
6. A nontrump boss is discounted when a remaining opponent is known void.
7. Weak-hand kitty burial still chooses zero points; strong-hand behavior does
   not regress.
8. Honest tiers use only public void/card memory; Omniscient-only exact probes
   remain behind the existing information boundary.

Then run paired bot baselines and sufficiently large matched self-play. Deal
variance is high; do not promote from a small win-rate sample. Track point
margin, level utility, final-kitty win rate, empty-pot trump spend rate, and
weak/singleton throw rate alongside overall wins.

## Relevant files

- `mechanics/src/trick.rs` — follow legality, throw invalidation, format matching,
  winner comparison, largest trick unit.
- `core/src/game_state/play_phase.rs` — terminal kitty award and multiplier.
- `core/src/bot/heuristics.rs` — lead/follow scoring, Enoch playbook, kitty burial,
  safe-throw proposal.
- `core/src/bot/search.rs` — rollout horizon, terminal evaluation, exact-solver
  integration.
- `core/src/bot/endgame.rs` — bounded perfect-information solver and current caps.
- `core/src/bot/expert.rs` — v1/v2 feature contracts.
- `core/src/bot/policy.rs` — Grandmaster/Omniscient policy and rollout selection.
- `core/examples/gen_training_data.rs` — terminal level/Q and final-kitty targets.
- `training/train_expert.py` — schema/model heads and export contract.

## Handoff status

This session intentionally made no strategy/model code changes after the Joker-
throw fix. The next session should begin with Phase A targeted fixtures and a
mechanics-backed shape helper, not with ONNX retraining. Any feature change must
bump the schema/dimension, regenerate data, retrain the artifact, update its
manifest/golden vectors, and pass paired A/B before replacing the embedded model.
