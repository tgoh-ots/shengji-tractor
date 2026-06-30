#!/usr/bin/env bash
# Matched-deal embedded-model vs candidate-model evaluation in separate
# processes, so the Expert model OnceLock cannot contaminate either arm.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

MODEL="${1:?usage: run_model_ab.sh MODEL.onnx [OUTDIR]}"
OUTDIR="${2:-${AB_OUTDIR:-$PWD/model-ab}}"
# Candidate companions are derived from the candidate path. The caller may be
# using SHENGJI_EXPERT_MODEL_* for the teacher/prior, so those variables must
# never select the candidate's manifest by accident.
MANIFEST="${AB_CANDIDATE_MANIFEST:-$MODEL.manifest.json}"
GOLDEN="${AB_CANDIDATE_GOLDEN:-$MODEL.golden.json}"
BASELINE_MODEL="${AB_BASELINE_MODEL:-}"
BASELINE_MANIFEST="${AB_BASELINE_MANIFEST:-${BASELINE_MODEL:+$BASELINE_MODEL.manifest.json}}"
BASELINE_GOLDEN="${AB_BASELINE_GOLDEN:-${BASELINE_MODEL:+$BASELINE_MODEL.golden.json}}"
PAIRS="${AB_PAIRS:-200}"
SEED="${AB_SEED:-0x5EED}"
BUDGET_MS="${AB_BUDGET_MS:-150}"
CARGO="${CARGO:-cargo +1.92.0}"
read -r -a CARGO_CMD <<<"$CARGO"

[[ -s "$MODEL" && -s "$MANIFEST" && -s "$GOLDEN" ]] || {
  echo "model, companion manifest, and golden vectors are required" >&2
  exit 2
}
mkdir -p "$OUTDIR"
# A partially rerun evaluation is never allowed to retain an old success
# marker. The comparison is recreated atomically only after both arms pass.
python3 - "$OUTDIR" <<'PY'
from pathlib import Path
import sys
Path(sys.argv[1], "comparison.json").unlink(missing_ok=True)
PY

"${CARGO_CMD[@]}" build --release -p shengji-core \
  --example model_control_eval --example validate_expert_model
VALIDATOR="$REPO/target/release/examples/validate_expert_model"
CONTROL="$REPO/target/release/examples/model_control_eval"
"$VALIDATOR" "$MODEL" "$MANIFEST" "$GOLDEN" | tee "$OUTDIR/parity.txt"
if [[ -n "$BASELINE_MODEL" ]]; then
  [[ -s "$BASELINE_MODEL" && -s "$BASELINE_MANIFEST" && -s "$BASELINE_GOLDEN" ]] || {
    echo "baseline model, companion manifest, and golden vectors are required" >&2
    exit 2
  }
  "$VALIDATOR" "$BASELINE_MODEL" "$BASELINE_MANIFEST" "$BASELINE_GOLDEN" \
    | tee "$OUTDIR/baseline-parity.txt"
fi

# These are intentionally two OS processes. Each receives the identical deal
# sequence and compute budget, but initializes its own model OnceLock.
if [[ -n "$BASELINE_MODEL" ]]; then
  SHENGJI_EXPERT_MODEL_PATH="$BASELINE_MODEL" \
    SHENGJI_EXPERT_MODEL_MANIFEST="$BASELINE_MANIFEST" \
    SHENGJI_BOT_BUDGET_MS="$BUDGET_MS" \
    "$CONTROL" "$PAIRS" "$SEED" >"$OUTDIR/embedded.json" 2>"$OUTDIR/embedded.log"
else
  env -u SHENGJI_EXPERT_MODEL_PATH -u SHENGJI_EXPERT_MODEL_MANIFEST \
    SHENGJI_BOT_BUDGET_MS="$BUDGET_MS" \
    "$CONTROL" "$PAIRS" "$SEED" >"$OUTDIR/embedded.json" 2>"$OUTDIR/embedded.log"
fi
SHENGJI_EXPERT_MODEL_PATH="$MODEL" \
  SHENGJI_EXPERT_MODEL_MANIFEST="$MANIFEST" \
  SHENGJI_BOT_BUDGET_MS="$BUDGET_MS" \
  "$CONTROL" "$PAIRS" "$SEED" >"$OUTDIR/candidate.json" 2>"$OUTDIR/candidate.log"

COMPARE_ARGS=(
  training/model_ab.py compare
  --model "$MODEL"
  --manifest "$MANIFEST"
  --golden "$GOLDEN"
  --outdir "$OUTDIR"
  --pairs "$PAIRS"
  --seed "$SEED"
  --budget-ms "$BUDGET_MS"
)
if [[ -n "$BASELINE_MODEL" ]]; then
  COMPARE_ARGS+=(
    --baseline-model "$BASELINE_MODEL"
    --baseline-manifest "$BASELINE_MANIFEST"
    --baseline-golden "$BASELINE_GOLDEN"
  )
fi
if [[ -n "${AB_MIN_LEVEL_DELTA+x}" ]]; then
  COMPARE_ARGS+=(--minimum-level-delta "$AB_MIN_LEVEL_DELTA")
fi
python3 "${COMPARE_ARGS[@]}"

echo "Baseline arm ($([[ -n "$BASELINE_MODEL" ]] && echo candidate || echo embedded)): $OUTDIR/embedded.json"
echo "Candidate arm: $OUTDIR/candidate.json"
echo "Paired candidate-minus-baseline estimate: $OUTDIR/comparison.json"
