# Policy-only Expert model promotion — 2026-07-02

## Decision

The human operator explicitly approved committing and deploying the confirmed
schema-v2 policy-only Expert model. Promotion is manual; automatic production
promotion remains disabled.

The embedded model changes from SHA-256 `04f678de...df4` to
`e4204bef...671c`. It consumes 49 honest schema-v2 features and exports only
`score`, so state-value and action-Q inference remain structurally absent.

## Evidence

The first fresh 400-pair gate was promising but missed its non-inferiority
threshold by 0.00125 on the level-utility lower confidence bound. It was not
reused or extended for the decision.

An independently seeded confirmation then completed 800 matched-deal pairs per
arm at a 150 ms bot budget with zero failed hands:

| Candidate minus embedded | Estimate | Paired bootstrap 95% |
|---|---:|---:|
| Level utility | +0.040625 | [+0.001875, +0.080625] |
| Point margin | +1.1125 | [-0.221875, +2.4875] |
| Win rate | +0.023125 | [-0.001250, +0.048750] |

The predeclared level-utility lower bound, `+0.001875`, cleared the `-0.020000`
gate. An independent reconstruction reproduced every published delta and
bootstrap interval exactly.

The full comparison and completion audit are preserved under
`docs/strategy/artifacts/`. `core/src/bot/expert_model.promotion.json` binds the
model, training manifest, golden vectors, protocol, comparison, source commit,
human approval, and rollback image.

## Serving contract and validation

The immutable trainer manifest is preserved as
`core/src/bot/expert_model.training.manifest.json`. The runtime companion is a
production derivative with the same schema, dimensions, output contract, model
hash, and golden hash; only serving/promotion metadata and the checked-in golden
path differ.

CI validates the committed model and golden vectors through tract. The embedded
model unit test parses the real companion manifest, asserts `V2Policy`, and
requires output 0 while rejecting value output 1.

## Deployment and rollback

The exact promotion commit is built into a single immutable Fly image. Because
the server stores rooms in one machine's memory, production uses an immediate
single-machine replacement (`--ha=false`) rather than canary or blue/green
traffic splitting. A full deploy disconnects clients and destroys current
rooms.

Rollback is the pinned Fly v39 image
`registry.fly.io/shengji-tractor:deployment-01KWDCJ3FM59FQWNGAVMN4NBFZ`,
digest `sha256:1b38f506...2528b`. Rollback restores software but cannot restore
rooms lost during cutover.
