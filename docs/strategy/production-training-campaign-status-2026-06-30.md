# Production-strength bot training campaign status — 2026-06-30

Audit timestamp: 2026-06-30 17:59:57 PDT

## Outcome

The schema-v3 production-strength campaign was stopped during round-0 data
generation. It produced no trained model and ran no formal gate. Consequently,
there is no production-campaign evaluation outcome to accept or reject, and no
model was promoted or deployed.

The two campaign attempts are distinct from the earlier policy-only re-distill
and tiny schema-v3 plumbing rehearsals described below.

## Campaign identity and progress

Attempt 1, `~/.shengji-expert-iteration-20260630-5abdd65`, failed during shard
generation: all 16 generators were killed and it produced no CSV completion
artifact, model, or gate.

Attempt 2, `~/.shengji-expert-iteration-20260630-5abdd65-v2`, used:

- Source commit `5abdd65d99c364297ec66c09227528b082c4daae`.
- Config SHA-256
  `87026a76f49d6ce8944f01e7a2c439b6996185100c43e62c5e8f4a89dd5de93e`.
- Expert-iteration runner SHA-256
  `3bc821a617bbe7ffc3e6a01df99ab42c9fcf1d85bce18746c979bfe210ce21f0`.
- Experiment manifest SHA-256
  `052f2c9b5125b25cafa1283a70a9a4413fde2b04293bf53c05cb70768ba731ca`.

It started at 15:12:27 PDT, restarted generation at 15:15:29, and was stopped
with eight partial `shard_0.csv`–`shard_7.csv` files. The latest write was
15:32:09. There are zero `.done` markers and no data sidecars, `data_full.csv`,
model, parity result, model manifest/goldens, A/B comparison, round manifest,
round 1, final gate, or completion marker. No campaign process remains active.
These partial shards are invalid and must not be reused.

## Gate contract that did not run

The planned research-only campaign had two rounds, each with 16 shards × 250
games and up to 80 epochs:

1. `bootstrap-value-q`.
2. `candidate-distribution`.

Each round would run 400 matched pairs at 150 ms and continue only if the paired
bootstrap 95% lower bound on signed level-utility delta was at least `-0.02`.
The final candidate would then face the embedded model for 400 matched pairs at
the 2,200 ms production budget, requiring a lower bound of at least `0.0` and no
candidate-load fallback. These are continuation/evaluation gates only: both the
config and generated manifests set `research_only: true` and
`automatic_production_promotion_allowed: false`.

The gate status is therefore **not run**, neither pass nor fail.

## Preserved earlier policy-only evaluation

This earlier re-distillation is not the stopped schema-v3 campaign, but it is the
first substantive learned-policy evaluation and its exact outcomes are preserved
here because the raw files live under temporary session storage.

- Training data: 2,400 games, 136,286 grouped decisions / 737,778 candidate rows.
- Data SHA-256:
  `e0a34a992d9bffe11af903184b60eadf39edface854438c9dc4124be8029d84f`.
- Candidate ONNX SHA-256:
  `9d99edf72fbd4ea55a14e766d9b2de0c3b211b11156e053b3b77ab05a1f78c16`.
- Best validation top-1 accuracy: 49.7%.

First 400-game evaluation versus Easy:

| Model | Win rate | 95% CI | Point margin |
| --- | ---: | ---: | ---: |
| Embedded | 55.00% (220/400) | [50.10%, 59.80%] | +5.79 |
| Candidate | 56.50% (226/400) | [51.60%, 61.27%] | +5.46 |

Confirmation, 1,000 games with seed 7000:

| Model | Win rate | 95% CI | Point margin |
| --- | ---: | ---: | ---: |
| Embedded | 57.20% (572/1,000) | [54.11%, 60.23%] | +6.67 |
| Candidate | 56.20% (562/1,000) | [53.11%, 59.25%] | +5.78 |

The confirmation was tied/slightly adverse, so the candidate was not embedded.
Raw record checksums:

- Training output `b2lbwxm4d.output`:
  `6a2f49b5cff2414c37b1f9873d0740696c6e8d10640e3b8bbe4968f684f02125`.
- First evaluation `bumkedz9n.output`:
  `d436a20ab692ea9576afc4c6670d16e124f6f197c5e9ffc58807ba8ce3f37a79`.
- Confirmation `bnaiknfiw.output`:
  `a82b5b125a3693817cf97e3b038411b34fe3d9671b9a5af7f0631914474be1d7`.

The tiny one-pair outputs under `/private/tmp/expert_iter_run_20260630` and
`/private/tmp/ab_gate_ci_20260630` are contract/plumbing rehearsals, not strength
evidence. Historical value-head runs were neutral and remain default-off; the
original large run was invalidated by a shard group-ID collision and must not be
used as promotion evidence.

## Production verification

Fly remains on release 39 (`4f9e5a8`), image digest
`sha256:1b38f50665afbf61d588742bf7dbc60bfdfb8d482fb7cfee48adf33e8be2528b`.
It has no secrets or external expert-model override. The deployed embedded model
is therefore still the 36-feature policy-only model, SHA-256
`04f678dec5329a23ef33735acfacd45b7c0815154012507bb2d74dc6bde23df4`.
Release 39 was the unrelated unsafe-Joker heuristic fix, not a trained-model
promotion.

Do not resume either stopped campaign workdir. The source predates later bot
fixes and the human-player refinements. Any future production-strength campaign
requires explicit approval, a fresh reviewed source SHA, a fresh workdir, and
preservation of its first formal gate artifacts. It must stop after all gated
evaluations for user review; model promotion and deployment remain forbidden.
