#!/usr/bin/env bash
#
# Athena (Expert-Iteration) self-play training loop. RESUMABLE.
#
# Each ITERATION: parallel sharded self-play where the current net-in-SEARCH
# (honest Expert determinized search) both plays AND labels each decision (the
# AlphaZero policy-improvement operator, but honest — see gen_exit_data) ->
# concat (group-id offset per shard) -> train a 2-output (policy+value) net ->
# next iteration's search uses the freshly-trained net as its prior. Cloning the
# SEARCH (not the Omniscient cheater) escapes the behavioral-cloning aliasing floor.
#
#   bash training/run_exit_pipeline.sh            # start / resume
#   STATUS=1 bash training/run_exit_pipeline.sh
#
# Tunables (env): EXIT_ITERS, EXIT_SHARDS, EXIT_GAMES_PER_SHARD, EXIT_BUDGET_MS
# (search budget = label quality + gen speed), EXIT_TIER (expert|enoch), EXIT_EPOCHS,
# BASE_SEED, WORKDIR, CARGO.
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

WORKDIR="${WORKDIR:-$HOME/.shengji-athena}"
ITERS="${EXIT_ITERS:-3}"
NSHARDS="${EXIT_SHARDS:-9}"
GAMES_PER_SHARD="${EXIT_GAMES_PER_SHARD:-200}"   # 9 x 200 = 1800 games/iteration
BUDGET_MS="${EXIT_BUDGET_MS:-150}"
TIER="${EXIT_TIER:-expert}"
EPOCHS="${EXIT_EPOCHS:-120}"
BASE_SEED="${BASE_SEED:-880000}"
CARGO="${CARGO:-cargo +1.92.0}"
NCPU="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"
PAR="${PAR:-$(( NCPU > 1 ? NCPU - 1 : 1 ))}"

mkdir -p "$WORKDIR"
LOG="$WORKDIR/run.log"
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

VENV="$HOME/.shengji-value-run/venv"
GEN="$REPO/target/release/examples/gen_exit_data"
EMBEDDED="$REPO/core/src/bot/expert_model.onnx"
PY="$VENV/bin/python"

if [ "${STATUS:-0}" = "1" ]; then
  echo "WORKDIR=$WORKDIR"
  for k in $(seq 0 $((ITERS - 1))); do
    echo "iter $k: $([ -f "$WORKDIR/iter_$k.done" ] && echo done || echo pending)  model=$([ -f "$WORKDIR/exit_iter_$k.onnx" ] && echo yes || echo no)"
  done
  exit 0
fi

say "=== Athena ExIt === WORKDIR=$WORKDIR iters=$ITERS shards=${NSHARDS}x${GAMES_PER_SHARD} budget=${BUDGET_MS}ms tier=$TIER par=$PAR"
"$PY" -c 'import torch,numpy' 2>/dev/null || { say "venv missing torch ($VENV)"; exit 1; }
$CARGO build --release -p shengji-core --example gen_exit_data --example paired_eval 2>&1 | tail -1 | tee -a "$LOG"
[ -x "$GEN" ] || { say "build FAILED"; exit 1; }

prev_pt=""
for k in $(seq 0 $((ITERS - 1))); do
  ITER_CSV="$WORKDIR/it_$k.csv"
  ITER_ONNX="$WORKDIR/exit_iter_$k.onnx"
  ITER_PT="${ITER_ONNX}.pt"
  if [ -f "$WORKDIR/iter_$k.done" ]; then
    say "iter $k: already done"; prev_pt="$ITER_PT"; continue
  fi
  if [ "$k" -eq 0 ]; then net="$EMBEDDED"; else net="$WORKDIR/exit_iter_$((k-1)).onnx"; fi
  say "iter $k: ExIt self-play+label gen (search prior=$(basename "$net"))"

  gen_shard() {
    local j="$1"
    [ -f "$WORKDIR/it_${k}_s${j}.csv" ] && { echo "  shard $j cached"; return 0; }
    SHENGJI_EXPERT_MODEL_PATH="$net" SHENGJI_BOT_BUDGET_MS="$BUDGET_MS" EXIT_TIER="$TIER" \
      GEN_GAMES="$GAMES_PER_SHARD" GEN_SEED="$((BASE_SEED + k * 1000 + j))" \
      GEN_OUT="$WORKDIR/it_${k}_s${j}.csv" \
      "$GEN" >"$WORKDIR/it_${k}_s${j}.gen.log" 2>&1 \
      && echo "  shard $j ok ($(wc -l < "$WORKDIR/it_${k}_s${j}.csv") rows)" || echo "  shard $j FAILED"
  }
  export -f gen_shard
  export WORKDIR k net BUDGET_MS TIER GAMES_PER_SHARD BASE_SEED GEN
  seq 0 $((NSHARDS - 1)) | xargs -P "$PAR" -I{} bash -c 'gen_shard "$1"' _ {} 2>&1 | tee -a "$LOG"

  # concat with per-shard group-id offset (each shard numbers groups from 0).
  # Offset 1e7 (NOT 1e9): on Debian's mawk, integral values > 2^31 print in
  # scientific notation ("3e+09"), which breaks the trainer's int() parse. 1e7
  # keeps offsets < 2^31 for up to ~200 shards while staying collision-free
  # (a shard has far fewer than 1e7 decisions).
  head -1 "$WORKDIR/it_${k}_s0.csv" > "$ITER_CSV"
  for j in $(seq 0 $((NSHARDS - 1))); do
    awk -F, -v OFS=, -v off="$((j * 10000000))" 'NR>1 { $1 = $1 + off; print }' \
      "$WORKDIR/it_${k}_s${j}.csv" >> "$ITER_CSV"
  done
  say "iter $k: dataset $(wc -l < "$ITER_CSV") rows -> train policy+value ($EPOCHS epochs)"

  init_arg=""  # train_expert has no warm-start; each iter trains fresh on the new labels.
  rm -f "$ITER_ONNX.tmp"
  "$PY" training/train_expert.py --data "$ITER_CSV" --out "$ITER_ONNX.tmp" \
    --epochs "$EPOCHS" --value-weight 1.0 2>&1 | tee -a "$LOG" | grep -E 'Loaded|Value head|Best|Exported'
  if [ -f "$ITER_ONNX.tmp" ]; then mv "$ITER_ONNX.tmp" "$ITER_ONNX"; else say "iter $k: train FAILED"; exit 1; fi
  touch "$WORKDIR/iter_$k.done"
  say "iter $k: DONE -> $ITER_ONNX"
done

cp "$WORKDIR/exit_iter_$((ITERS-1)).onnx" "$WORKDIR/athena.onnx" 2>/dev/null || true
say "=== Athena ExIt DONE === final model: $WORKDIR/athena.onnx"
