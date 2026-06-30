#!/usr/bin/env bash
# Small, end-to-end action-value smoke run. Override any value in the environment
# before invoking. The pilot skips the statistically meaningful A/B by default
# because eight games are only enough to validate artifacts and target flow.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export WORKDIR="${WORKDIR:-${TMPDIR:-/tmp}/shengji-dmc-pilot}"
export NUM_SHARDS="${NUM_SHARDS:-2}"
export GAMES_PER_SHARD="${GAMES_PER_SHARD:-4}"
export GEN_TEACHER_BUDGET_MS="${GEN_TEACHER_BUDGET_MS:-20}"
export GEN_BEHAVIOUR="${GEN_BEHAVIOUR:-easy}"
export GEN_BEHAVIOUR_BUDGET_MS="${GEN_BEHAVIOUR_BUDGET_MS:-5}"
export GEN_Q_CANDIDATES="${GEN_Q_CANDIDATES:-2}"
export GEN_Q_ROLLOUT_BEHAVIOUR="${GEN_Q_ROLLOUT_BEHAVIOUR:-easy}"
export GEN_Q_ROLLOUT_BUDGET_MS="${GEN_Q_ROLLOUT_BUDGET_MS:-5}"
export EPOCHS="${EPOCHS:-12}"
export EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-q}"
export RUN_AB="${RUN_AB:-0}"
export PAR="${PAR:-2}"
export RUN_BELIEF_PILOT="${RUN_BELIEF_PILOT:-1}"
export BELIEF_PILOT_GAMES="${BELIEF_PILOT_GAMES:-2}"
export BELIEF_PILOT_EPOCHS="${BELIEF_PILOT_EPOCHS:-2}"

bash "$REPO/training/run_value_pipeline.sh"

if [[ "$RUN_BELIEF_PILOT" == "1" ]]; then
  cd "$REPO"
  BELIEF_DIR="$WORKDIR/belief"
  mkdir -p "$BELIEF_DIR"
  CARGO="${CARGO:-cargo +1.92.0}"
  read -r -a CARGO_CMD <<<"$CARGO"

  BELIEF_GAMES="$BELIEF_PILOT_GAMES" \
    BELIEF_SEED="${BELIEF_SEED:-77}" \
    BELIEF_SNAPSHOT_EVERY="${BELIEF_SNAPSHOT_EVERY:-8}" \
    BELIEF_BEHAVIOUR_BUDGET_MS="${BELIEF_BEHAVIOUR_BUDGET_MS:-5}" \
    BELIEF_OUT="$BELIEF_DIR/data.csv" \
    BELIEF_MANIFEST="$BELIEF_DIR/data.csv.manifest.json" \
    "${CARGO_CMD[@]}" run --release -p shengji-core --example gen_belief_data

  "$WORKDIR/venv/bin/python" training/train_belief.py \
    --data "$BELIEF_DIR/data.csv" \
    --out "$BELIEF_DIR/model.onnx" \
    --epochs "$BELIEF_PILOT_EPOCHS"

  "${CARGO_CMD[@]}" run --release -p shengji-core \
    --example validate_belief_model -- \
    "$BELIEF_DIR/model.onnx" \
    "$BELIEF_DIR/model.onnx.manifest.json" \
    "$BELIEF_DIR/model.onnx.golden.json"

  SHENGJI_BELIEF_MODEL_PATH="$BELIEF_DIR/model.onnx" \
    SHENGJI_BELIEF_MODEL_MANIFEST="$BELIEF_DIR/model.onnx.manifest.json" \
    SHENGJI_BELIEF_WEIGHT="${SHENGJI_BELIEF_WEIGHT:-0.35}" \
    SHENGJI_BOT_BUDGET_MS="${BELIEF_RUNTIME_BUDGET_MS:-5}" \
    "${CARGO_CMD[@]}" run --release -p shengji-core \
    --example model_control_eval -- 1 48879 \
    >"$BELIEF_DIR/runtime-smoke.json"
fi
