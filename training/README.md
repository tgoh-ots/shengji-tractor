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
AB_MIN_LEVEL_DELTA is a quantitative gate.

## Offline belief model

~~~sh
BELIEF_GAMES=20 BELIEF_SEED=77 BELIEF_SNAPSHOT_EVERY=4 \
BELIEF_OUT=/tmp/belief.csv \
cargo +1.92.0 run --release -p shengji-core --example gen_belief_data

python3.13 training/train_belief.py --data /tmp/belief.csv \
  --out /tmp/belief.onnx

cargo +1.92.0 run --release -p shengji-core \
  --example validate_belief_model -- \
  /tmp/belief.onnx /tmp/belief.onnx.manifest.json \
  /tmp/belief.onnx.golden.json
~~~

For each honest snapshot and hidden card copy, the exporter records public
history/card/role/capacity/void features, a relative destination (next,
opposite, previous, kitty), and hard legality mask. The trainer splits by game,
masks illegal logits, and reports accuracy, NLL, multiclass Brier, ECE, and
illegal probability mass.

This first contract is intentionally narrow: CSV schema 1, exactly b0..b19,
target order next/opposite/previous/kitty, and the
tractor:4p:2x-standard:kitty8:no-removed game contract. Training requires the
generator sidecar and validates its schema, width, target order, behavior, and
declared game contract before reading rows. `--allow-unsafe-no-sidecar` exists
only for exploration and marks the artifact non-servable. Export includes exact feature names and writes
PyTorch golden vectors; validate_belief_model checks the manifest and numerical
ONNX/tract parity for both feature and legality-mask inputs.

The manifest marks the result as an experimental serving candidate. The runtime
loader is opt-in and production defaults it off; promotion still requires
evaluation of belief-guided determinization.

Runtime belief weighting defaults off. A serving-path smoke must explicitly
enable it, for example:

~~~sh
SHENGJI_BELIEF_MODEL_PATH=/tmp/belief.onnx \
SHENGJI_BELIEF_MODEL_MANIFEST=/tmp/belief.onnx.manifest.json \
SHENGJI_BELIEF_WEIGHT=0.35 SHENGJI_BOT_BUDGET_MS=20 \
cargo +1.92.0 run --release -p shengji-core \
  --example model_control_eval -- 2 48879
~~~
