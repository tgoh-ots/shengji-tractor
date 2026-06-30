# Bot training

The committed Expert model remains the 36-feature policy-distillation baseline.
The new pipeline is additive: it retains a policy head while training honest
state-value and action-value heads, and never silently reinterprets a legacy
model.

## Contracts

The generator emits dataset schema 3. One row is one legal candidate; group
identifies a decision and game_id the whole trajectory. Validation splits by
whole game.

The serving outputs are:

1. score(o,a): listwise teacher or behavior policy logit.
2. state_value(o): state-only value. Candidate fields are masked and it is
   trained once per decision.
3. action_q(o,a): candidate value, trained only where a counterfactual exists.

Schema-v2 models use 49 honest features. The first 36 freeze the shipped policy
contract. Additions cover role, score/threshold, exact public progress, action
structure, full public card/void memory, and an exact mechanics-engine
candidate-takes-the-lead signal. Manifests must declare schema 2, dimension 49,
output names, and output semantics.

The primary threshold-aware target is:

    sign(actor team won) * (1 + levels awarded to the winner) / 5

It is clamped to [-1,1]. Adding one preserves a signal for a turnover/dead-zone
win that awards zero levels. The CSV also records final attacker points, scoring
bucket, actor-team win, and actor-team final-trick/kitty win for behavior V and
sparse Q. The trainer consumes these with offline categorical/binary heads and
reports bucket accuracy, Brier scores, and ten-bin ECE. Auxiliary heads are not
exported to serving ONNX; the manifest states that explicitly.

The old 36-feature optional V is normalized point margin. Runtime exposes
separate typed APIs for legacy point V, v2 level V, and v2 level Q, preventing a
level value from entering a point-valued blend.

## Seeded generation

~~~sh
GEN_GAMES=2 GEN_SEED=123 GEN_TEACHER_BUDGET_MS=10 \
GEN_BEHAVIOUR=easy GEN_BEHAVIOUR_BUDGET_MS=5 \
GEN_Q_CANDIDATES=2 GEN_Q_ROLLOUT_BEHAVIOUR=easy \
GEN_Q_ROLLOUT_BUDGET_MS=5 GEN_OUT=/tmp/shengji-v3.csv \
cargo +1.92.0 run --release -p shengji-core --example gen_training_data
python3.13 training/train_expert.py --data /tmp/shengji-v3.csv --analyze
~~~

Each deal seed is a stable SplitMix64 derivation of seed and game index, so
sharding/resume order cannot change cards. Deal and behavior-mixture RNGs are
isolated. Teacher, behavior, and Q continuation receive explicit per-call
budgets; no process-global budget is shared. Time-bounded search may still vary
across very different machines if simulation counts differ. Byte-identical
labels require deterministic caps or enough time to exhaust the world cap.

Q forces selected candidates into the same compatible real deal, after which
every player acts from its own redacted view. Behavior and teacher actions are
anchors when the Q cap permits. A blank Q cell means unsampled, not zero.

Generation cost is dominated by teacher search and counterfactual completion.
Candidate expansion can make early decisions wide; inspect rows per decision,
Q coverage, drop counters, and elapsed time before scaling.

## Train, export, validate

The macOS reference environment is fully pinned in requirements.lock.txt. The
runner also records Python version, platform, and pip freeze --all, refusing
resume after drift.

~~~sh
python3.13 -m venv /tmp/shengji-train
/tmp/shengji-train/bin/pip install -r training/requirements.lock.txt
/tmp/shengji-train/bin/python training/train_expert.py \
  --data /tmp/shengji-v3.csv --out /tmp/expert-v3.onnx \
  --policy-weight 1 --value-weight 1 --q-weight 1 --auxiliary-weight .25

cargo +1.92.0 run --release -p shengji-core \
  --example validate_expert_model -- \
  /tmp/expert-v3.onnx /tmp/expert-v3.onnx.manifest.json \
  /tmp/expert-v3.onnx.golden.json
~~~

Export writes ONNX, a semantic/config/data/split/hash manifest, and deterministic
PyTorch golden vectors. Rust runs those through tract and fails on shape,
finiteness, or numerical drift.

Legacy CSVs without trajectory IDs are accepted for analysis, but training
refuses to call a decision-level pseudo split leakage-free. Historical
reproduction needs --allow-legacy-group-split. Only widths 36 and 49 are valid.

## Resumable DMC pipeline

~~~sh
WORKDIR=/tmp/shengji-dmc training/run_dmc_pilot.sh

WORKDIR=$HOME/.shengji-action-value-run \
NUM_SHARDS=16 GAMES_PER_SHARD=250 PAR=8 \
training/run_value_pipeline.sh

WORKDIR=$HOME/.shengji-action-value-run STATUS=1 \
training/run_value_pipeline.sh
~~~

The DMC pilot also generates two belief games, trains a short experimental
belief model, and runs validate_belief_model. Set RUN_BELIEF_PILOT=0 to skip
that independent smoke stage.

The runner uses set -euo pipefail, writes shard content/config manifests,
validates headers and IDs, and marks completion only after all artifacts exist.
Its config fingerprints core/src, mechanics/src, Cargo manifests/lock, and
training sources by current file content, including dirty changes.

After training it validates tract parity, then launches embedded and candidate
models in separate processes so model OnceLock state cannot cross arms.
Identical deals/budgets face the same Easy control. Candidate-minus-embedded
per-deck win, margin, and level utility get paired bootstrap intervals;
AB_MIN_LEVEL_DELTA gates the lower endpoint of the paired level-utility
bootstrap interval, not the noisier point estimate.

## Offline belief model

~~~sh
BELIEF_GAMES=20 BELIEF_SEED=77 BELIEF_SNAPSHOT_EVERY=4 \
BELIEF_FEATURE_SCHEMA_VERSION=2 \
BELIEF_OUT=/tmp/belief.csv \
cargo +1.92.0 run --release -p shengji-core --example gen_belief_data

python3.13 training/train_belief.py --data /tmp/belief.csv \
  --out /tmp/belief.onnx

cargo +1.92.0 run --release -p shengji-core \
  --example validate_belief_model -- \
  /tmp/belief.onnx /tmp/belief.onnx.manifest.json \
  /tmp/belief.onnx.golden.json /tmp/belief.csv
~~~

For each honest snapshot and hidden card copy, the exporter records public
history/card/role/capacity/void features, a relative destination (next,
opposite, previous, kitty), and hard legality mask. The trainer splits by game,
masks illegal logits, and reports accuracy, NLL, multiclass Brier, ECE, and
illegal probability mass.

Two exact feature contracts are supported. Schema v1 is the frozen b0..b19
aggregate vector; pre-lineage artifacts must be regenerated with the current
strict manifest. Schema v2 has 128 ordered semantic features: the v1 prefix,
public progress, four public bid events, and eight public play events. Generate
v2 explicitly with `BELIEF_FEATURE_SCHEMA_VERSION=2`; the default remains v1
for compatibility.

The sidecar tuple must be exactly `(manifest, feature schema, dimension)` =
`(1,1,20)` or `(2,2,128)`, and its ordered `feature_names` must match every CSV
column and row-level feature schema. Both retain target order
next/opposite/previous/kitty and the
tractor:4p:2x-standard:kitty8:no-removed game contract. Training rejects any
cross-version mixture. The sidecar binds the exact CSV SHA-256 and records the
generator domain (`bidding=expert;exchange=easy;play=easy`); the model manifest
also binds the encoder contract and exact Rust encoder-source SHA-256 and carries
all fields forward. Publicly pinned failed-throw holdings are excluded from
targets because the constraint solver already assigns them deterministically.
`--allow-unsafe-no-sidecar` is restricted to v1 exploration and marks the
artifact non-servable. Export and golden vectors carry the selected schema
dynamically; validate_belief_model checks artifact hashes, lineage fields, and
ONNX/tract parity for both feature and legality-mask inputs.
Pass the dataset as a fourth validator argument to re-hash the CSV and sidecar
in place.

The manifest marks the result as an experimental serving candidate. The runtime
loader is opt-in and production defaults it off; promotion still requires
evaluation of belief-guided determinization.

The network emits per-card destination marginals. Runtime multiplies those
scores over physical-copy assignments, which does not capture correlations or
calibrate the realized joint sampler. Treat row metrics as proposal diagnostics,
not posterior calibration. The retained-particle experiment is separately off
unless `SHENGJI_PERSISTENT_BELIEF=1`; duplicate-copy reveal conditioning is not
yet multiplicity-weighted, so fresh constrained sampling remains the default.
Golden vectors currently prove deterministic tensor/ONNX parity using synthetic
inputs. The manifest labels that scope explicitly; a mechanics-state-derived
encoder golden corpus remains required before promotion.

Runtime belief weighting defaults off. A serving-path smoke must explicitly
enable it, for example:

~~~sh
SHENGJI_BELIEF_MODEL_PATH=/tmp/belief.onnx \
SHENGJI_BELIEF_MODEL_MANIFEST=/tmp/belief.onnx.manifest.json \
SHENGJI_BELIEF_WEIGHT=0.35 SHENGJI_BOT_BUDGET_MS=20 \
cargo +1.92.0 run --release -p shengji-core \
  --example model_control_eval -- 2 48879
~~~

## Bounded expert iteration and replay experiments

The existing value/Q runner is also the execution engine for a bounded
search-teacher loop. It can deterministically mix trajectory policies by shard,
train a candidate, compare it with the previous round on matched deals, and use
that candidate as the next round's search prior:

~~~sh
python3 training/expert_iteration.py plan \
  --config training/expert_iteration.example.json
python3 training/expert_iteration.py run \
  --config training/expert_iteration.example.json \
  --workdir "$HOME/.shengji-expert-iteration"
~~~

Optional seat/suit symmetry audits and replay-verified offline trajectories are
composed by `training/prepare_expert_data.py`. Augmented copies share a parent
trajectory split; offline Q values are stripped and offline volume is capped.
All such paths are research-only and never auto-promote a model. See
[`docs/expert-iteration-training.md`](../docs/expert-iteration-training.md) for
the exact configuration, replay attestation, symmetry, resumability, and
measurement contracts.

Specialized bid and kitty listwise candidates can be trained with
`training/train_phase.py` after `gen_phase_training_data` emits the documented
20-feature phase datasets. Listwise outputs are used only for offset/scale-
invariant rank blending; bid/pass stays heuristic. Runtime is limited to the
exporter's four-player/two-standard-deck Tractor domain and requires an explicit
`SHENGJI_PHASE_MODEL_WEIGHT`; unsupported contexts and the default fall back to
the existing heuristics. See the linked experiment guide for commands and all
search/belief/phase knobs.
