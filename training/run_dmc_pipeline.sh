#!/usr/bin/env bash
#
# Oracle (DouZero-style Deep Monte-Carlo) self-play training loop. RESUMABLE.
#
# Each ITERATION: parallel sharded self-play (every seat acts ε-greedy on the
# CURRENT Q-net) -> concat -> train Q (regress realized MC returns) -> next
# iteration acts with the freshly-trained Q. The behaviour bootstraps from the
# embedded Expert net at iter 0 (reasonable trajectories), then improves on its
# own value estimates — no teacher, no tree search.
#
#   bash training/run_dmc_pipeline.sh                 # start / resume
#   STATUS=1 bash training/run_dmc_pipeline.sh        # report progress
#   tail -f "$HOME/.shengji-oracle/run.log"
#
# Tunables (env): DMC_ITERS, DMC_SHARDS, DMC_GAMES_PER_SHARD, DMC_EPSILON,
# DMC_EPOCHS, DMC_BUDGET_MS (bid/exchange search budget), BASE_SEED, WORKDIR, CARGO.
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

WORKDIR="${WORKDIR:-$HOME/.shengji-oracle}"
ITERS="${DMC_ITERS:-5}"
NSHARDS="${DMC_SHARDS:-9}"
GAMES_PER_SHARD="${DMC_GAMES_PER_SHARD:-400}"   # 9 x 400 = 3600 games/iteration
EPSILON="${DMC_EPSILON:-0.12}"
EPOCHS="${DMC_EPOCHS:-60}"
BUDGET_MS="${DMC_BUDGET_MS:-50}"                # bid/exchange (Expert) budget only
BASE_SEED="${BASE_SEED:-770000}"
CARGO="${CARGO:-cargo +1.92.0}"
NCPU="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"
PAR="${PAR:-$(( NCPU > 1 ? NCPU - 1 : 1 ))}"

mkdir -p "$WORKDIR"
LOG="$WORKDIR/run.log"
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

VENV="$HOME/.shengji-value-run/venv"           # reuse the shared torch venv
GEN="$REPO/target/release/examples/gen_dmc_data"
EMBEDDED="$REPO/core/src/bot/expert_model.onnx"
PY="$VENV/bin/python"

if [ "${STATUS:-0}" = "1" ]; then
  echo "WORKDIR=$WORKDIR"
  for k in $(seq 0 $((ITERS - 1))); do
    echo "iter $k: $([ -f "$WORKDIR/iter_$k.done" ] && echo done || echo pending)  model=$([ -f "$WORKDIR/oracle_iter_$k.onnx" ] && echo yes || echo no)"
  done
  exit 0
fi

say "=== Oracle DMC === WORKDIR=$WORKDIR iters=$ITERS shards=${NSHARDS}x${GAMES_PER_SHARD} eps=$EPSILON par=$PAR"

# stage 0: venv + build
"$PY" -c 'import torch,numpy' 2>/dev/null || { say "venv missing torch (expected at $VENV)"; exit 1; }
$CARGO build --release -p shengji-core --example gen_dmc_data 2>&1 | tail -1 | tee -a "$LOG"
[ -x "$GEN" ] || { say "build FAILED"; exit 1; }

prev_pt=""
for k in $(seq 0 $((ITERS - 1))); do
  ITER_CSV="$WORKDIR/it_$k.csv"
  ITER_ONNX="$WORKDIR/oracle_iter_$k.onnx"
  # train_dmc writes the state_dict at splitext(--out)[0]+".pt"; with --out
  # "$ITER_ONNX.tmp" that is exactly "$ITER_ONNX.pt".
  ITER_PT="${ITER_ONNX}.pt"
  if [ -f "$WORKDIR/iter_$k.done" ]; then
    say "iter $k: already done (model $ITER_ONNX)"; prev_pt="$ITER_PT"; continue
  fi

  # behaviour net: embedded Expert at iter 0, else previous iteration's Q.
  if [ "$k" -eq 0 ]; then behaviour="$EMBEDDED"; else behaviour="$WORKDIR/oracle_iter_$((k-1)).onnx"; fi
  say "iter $k: self-play gen (behaviour=$(basename "$behaviour"))"

  gen_shard() {
    local j="$1"
    [ -f "$WORKDIR/it_${k}_s${j}.csv" ] && { echo "  shard $j cached"; return 0; }
    SHENGJI_EXPERT_MODEL_PATH="$behaviour" SHENGJI_BOT_BUDGET_MS="$BUDGET_MS" \
      DMC_GAMES="$GAMES_PER_SHARD" DMC_EPSILON="$EPSILON" \
      DMC_SEED="$((BASE_SEED + k * 1000 + j))" DMC_OUT="$WORKDIR/it_${k}_s${j}.csv" \
      "$GEN" >"$WORKDIR/it_${k}_s${j}.gen.log" 2>&1 \
      && echo "  shard $j ok ($(wc -l < "$WORKDIR/it_${k}_s${j}.csv") rows)" || echo "  shard $j FAILED"
  }
  export -f gen_shard
  export WORKDIR k behaviour BUDGET_MS GAMES_PER_SHARD EPSILON BASE_SEED GEN
  seq 0 $((NSHARDS - 1)) | xargs -P "$PAR" -I{} bash -c 'gen_shard "$1"' _ {} 2>&1 | tee -a "$LOG"

  # concat shards (DMC rows are independent — plain cat, keep one header).
  head -1 "$WORKDIR/it_${k}_s0.csv" > "$ITER_CSV"
  for j in $(seq 0 $((NSHARDS - 1))); do tail -n +2 "$WORKDIR/it_${k}_s${j}.csv" >> "$ITER_CSV"; done
  say "iter $k: dataset $(wc -l < "$ITER_CSV") rows -> training Q ($EPOCHS epochs)"

  init_arg=""; [ -n "$prev_pt" ] && [ -f "$prev_pt" ] && init_arg="--init $prev_pt"
  rm -f "$ITER_ONNX.tmp"
  "$PY" training/train_dmc.py --data "$ITER_CSV" --out "$ITER_ONNX.tmp" \
    --epochs "$EPOCHS" $init_arg 2>&1 | tee -a "$LOG" | grep -E 'Loaded|epoch|Best|Exported'
  if [ -f "$ITER_ONNX.tmp" ]; then mv "$ITER_ONNX.tmp" "$ITER_ONNX"; else say "iter $k: train FAILED"; exit 1; fi
  touch "$WORKDIR/iter_$k.done"
  prev_pt="$ITER_PT"
  say "iter $k: DONE -> $ITER_ONNX"
done

# Final model = last iteration's Q.
cp "$WORKDIR/oracle_iter_$((ITERS-1)).onnx" "$WORKDIR/oracle.onnx" 2>/dev/null || true
say "=== Oracle DMC DONE === final model: $WORKDIR/oracle.onnx"
