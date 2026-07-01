# Human strategy refinement evaluation — 2026-06-30

## Decision

The tactical refinements are a credible improvement over `master` at `d1e55d1`.
The strongest evidence is a replicated greedy Enoch win-rate gain and large,
direct reductions in the targeted decision errors. The global provisional-level
leaf was harmful and remains disabled. The narrower terminal objectives and the
late ruff-shape reservation remain opt-in experiments; neither is enabled by
default or promoted here.

Candidate branch: `codex/human-refinements`, through `ee3080a`.

## What changed

- Honest lead selection rejects unproven multi-unit trump throws, including but
  not limited to Joker compounds. Perfect-information search still admits a
  throw when the materialized world proves it safe.
- A naked Joker lead is strongly discouraged unless it covers the configured
  late attacker threshold or belongs to the existing long-trump sequence plan.
- Lost-trick discards preserve nontrump aces, Jokers, high trump, pairs, and
  tractors; last-seat losses prefer the lowest nontrump nonpoint card.
- A partner point feed is suppressed when a remaining opponent is publicly
  known void and can ruff. Point feeding and edge guarding now require positive
  public evidence rather than treating missing information as proof.
- The late 65–75-style threshold rule uses the configured turnover score, not a
  hard-coded 80.
- Search and every direct/fallback live path apply the same honest lead mask.
- Benchmark tools now emit matched per-deck traces and paired bootstrap/sign-flip
  deltas. Decision metrics cover unsafe trump sets, naked Jokers, ace waste, and
  publicly ruffable partner feeds.

The documentation-only master commit `d1e55d1` proposed preserving late trump
response shape for the multiplied kitty. The candidate adds a mechanics-backed
post-play shape summary and an opt-in root candidate reservation. It is limited
to standard four-player, two-deck Tractor, requires a terminal-length rollout,
keeps the ordinary top candidate when only one search slot exists, and forces
the entire root into terminal contract units. This is terminal scoring of a
heuristic rollout, not an exact minimax solution.

## Greedy paired A/B

Every subject plays the same 5,000 or 2,500 deals in both seat orientations
against the frozen Legacy policy. Deltas are candidate minus `master`; this is a
cross-checkout against a fixed ruler, not direct candidate-versus-master play.

| Policy / corpus | Win delta | Point-margin delta | Level-utility delta |
| --- | ---: | ---: | ---: |
| Heuristic, 5,000, seed `0x5eed` | +0.65 pp `[-0.08,+1.36]` | +0.325 `[-0.115,+0.761]` | +0.018 `[+0.004,+0.030]` |
| Heuristic, 2,500, seed `0xc0ffee` | +0.94 pp `[-0.08,+1.98]` | +0.737 `[+0.116,+1.357]` | +0.019 `[-0.000,+0.037]` |
| Heuristic, 2,500, seed `0x48554d41` | +0.66 pp `[-0.36,+1.68]` | +0.683 `[+0.060,+1.314]` | +0.007 `[-0.012,+0.025]` |
| Enoch, 5,000, seed `0x5eed` | +1.20 pp `[+0.69,+1.71]` | +0.424 `[+0.099,+0.750]` | +0.011 `[+0.001,+0.021]` |
| Enoch, 2,500, seed `0xc0ffee` | +0.70 pp `[-0.02,+1.42]` | +0.206 `[-0.257,+0.672]` | +0.006 `[-0.008,+0.020]` |
| Enoch, 2,500, seed `0x48554d41` | +1.36 pp `[+0.66,+2.04]` | +0.225 `[-0.239,+0.682]` | +0.002 `[-0.012,+0.015]` |

The common heuristic is safely non-inferior and improves point margin in both
holdouts. Enoch's roughly one-point win-rate gain is resolved in the exploratory
corpus and second holdout; the first holdout narrowly crosses zero.

## Targeted decision metrics

Five thousand hands were replayed under a frozen Legacy roller. Tiny denominator
differences remain possible through tie ordering, so these are descriptive
rather than a strict matched-state causal estimate.

| Error | Master | Candidate |
| --- | ---: | ---: |
| Generic waste | 1.20% (1,764/147,015) | 0.51% (754/147,039) |
| Failed trump-set lead | 0/40,633 | 0/40,633 |
| Naked Joker lead | 61.89% (20,875/33,727) | 0.01% (2/33,727) |
| Nontrump ace waste | 1.28% (259/20,230) | 0.17% (34/20,244) |
| Publicly ruffable partner feed | 100% (80/80) | 0% (0/81) |

The trump-set metric was already zero on current master; broadening its hard mask
is defensive coverage. The Joker, ace, and ruffable-feed changes are large and
directly match the human player's complaints.

## Search experiments

- The global provisional level leaf was rejected. Turning it off improved point
  margin by +3.663 points/hand (`[+1.844,+5.444]`, `p=.0002`) and level utility
  by +0.070 (`[+0.015,+0.126]`, `p=.0163`) in 400 paired deals. It remains behind
  `SHENGJI_PROVISIONAL_LEVEL_OBJECTIVE=0`.
- A 200-pair production-path Expert screen was directionally positive versus
  master: +2.75 pp win `[-1.50,+7.00]`, +1.25 points `[-1.525,+3.987]`, and
  +0.068 levels `[-0.013,+0.145]`. It is a screen, not a promotion-quality gate.
- A 200-pair, four-world/four-candidate Enoch r12 screen was also directional:
  +1.25 pp win `[-3.50,+5.75]`, +1.562 points `[-1.137,+4.200]`, and +0.028
  levels `[-0.045,+0.100]` versus master.
- The first ruff-reservation run was effectively null: +0.25 pp win
  `[-0.50,+1.25]`, +0.062 points `[-0.287,+0.425]`, and +0.000 levels
  `[-0.010,+0.010]`. That revision still evaluated landlord reservations with
  the old static leaf, so it measures candidate coverage only and is not evidence
  for or against kitty defense.

Corrected, Grandmaster-shaped eight-candidate terminal experiments:

| Ablation, 200 paired deals | Win delta | Point-margin delta | Level-utility delta |
| --- | ---: | ---: | ---: |
| Ruff reservation + terminal scoring versus off | +0.75 pp `[-0.50,+2.00]` | +0.350 `[-0.275,+1.025]` | +0.020 `[+0.000,+0.043]` |
| Near-turnover attacker terminal objective versus off | +0.50 pp `[+0.00,+1.25]` | +0.150 `[-0.212,+0.562]` | +0.005 `[-0.007,+0.018]` |

The corrected ruff package was favorable but underpowered (`p=.454` for wins;
`p=.095` for level utility). It inserted 23 candidates across 400 hands, all for
seat 2, and changed only seven deck-level win indicators (five favorable, two
unfavorable). Roughly 1,200 paired deals would be needed for 80% power at the
observed +0.75 pp effect. It therefore remains default-off.

The isolated near-turnover objective was also directionally favorable, but only
two of 200 deck-level win indicators changed (both favorable). The discrete zero
bootstrap endpoint is not evidence of superiority (`p=.499`); 28 decks changed
on at least one outcome metric. This flag also remains default-off pending an
enriched late-state evaluation.

## Guardrails and limitations

- `SHENGJI_TERMINAL_LEVEL_OBJECTIVE` defaults off. It maximizes expected signed
  level utility in a narrowly terminal window; it is not a lexicographic proof
  that every increase in win probability dominates every extra level.
- `SHENGJI_LATE_RUFF_RESERVE` defaults off. Its post-play proxy does not yet
  match retained trump structure against a predicted final lead format.
- Exact kitty cards are used only by the exchanger. Teammates use an estimate
  from public card conservation; sampled hidden worlds supply terminal outcomes.
- The shipped policy model is unchanged. These heuristic/search changes do not
  require retraining. A future schema-v3 model with post-play shape, candidate
  units, suit-specific voids, and configured kitty multipliers would require new
  data, a new manifest, retraining, and a separate gate.

## Validation

- `cargo test -p shengji-core`: 161 passed, 2 ignored; baseline gate 2 passed,
  1 ignored.
- `cargo test -p shengji-mechanics`: 77 passed.
- `cargo clippy -p shengji-core --all-targets -- -D warnings`: passed.
- Deterministic tests cover honest/perfect-information throw separation,
  last-seat and ace retention, known-ruff partner feeds, configured threshold
  scoring, root-versus-rollout policy wiring, one-candidate safety, and both sides
  of the terminal ruff tradeoff: shed on an empty pot, but take a
  threshold-crossing point pot.

No model artifact was retrained or promoted, and no experimental flag was enabled
in production.
