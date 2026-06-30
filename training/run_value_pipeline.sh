#!/usr/bin/env bash
#
# Resumable, parallel value-head training + A/B pipeline.
#
# Runs the full headline experiment end-to-end: DAgger (`GEN_BEHAVIOUR=mix`)
# data generation -> train the value head -> paired A/B of the search leaf-eval
# value blend (SHENGJI_VALUE_WEIGHT off vs on). It is a MULTI-HOUR job, so it is
# CHECKPOINTED and RESUMABLE: every stage writes durable markers in $WORKDIR, and
# **re-running this script simply continues from the last completed step**. Safe to
# kill / sleep / reboot and re-run.
#
#   bash training/run_value_pipeline.sh            # start (or resume)
#   tail -f "$HOME/.shengji-value-run/run.log"     # watch progress
#   STATUS=1 bash training/run_value_pipeline.sh   # just print which stages are done
#
# Tunables (env, with defaults): NUM_SHARDS, GAMES_PER_SHARD, GEN_TEACHER_BUDGET_MS,
# GEN_MIX_SEARCH_FRAC, BASE_SEED, AB_PAIRS, AB_BUDGET_MS, AB_SEED, EPOCHS, PAR,
# WORKDIR, CARGO.
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

WORKDIR="${WORKDIR:-$HOME/.shengji-value-run}"
NUM_SHARDS="${NUM_SHARDS:-14}"
GAMES_PER_SHARD="${GAMES_PER_SHARD:-150}"     # 14 x 150 = 2100 mix games
GEN_TEACHER_BUDGET_MS="${GEN_TEACHER_BUDGET_MS:-200}"
GEN_MIX_SEARCH_FRAC="${GEN_MIX_SEARCH_FRAC:-0.5}"
BASE_SEED="${BASE_SEED:-1000}"
AB_PAIRS="${AB_PAIRS:-200}"                   # 200 deck-pairs -> ~+-4pp 95% MDE
AB_BUDGET_MS="${AB_BUDGET_MS:-150}"
AB_SEED="${AB_SEED:-0x5EED}"
EPOCHS="${EPOCHS:-80}"
CARGO="${CARGO:-cargo +1.92.0}"
NCPU="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"
PAR="${PAR:-$(( NCPU > 1 ? NCPU - 1 : 1 ))}"

mkdir -p "$WORKDIR"
LOG="$WORKDIR/run.log"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

VENV="$WORKDIR/venv"
GEN="$REPO/target/release/examples/gen_training_data"
PEVAL="$REPO/target/release/examples/paired_eval"
FULL="$WORKDIR/data_full.csv"
MODEL="$WORKDIR/value.onnx"

# ---- STATUS=1: report progress and exit ----
if [ "${STATUS:-0}" = "1" ]; then
  done_shards=0
  for i in $(seq 0 $((NUM_SHARDS - 1))); do [ -f "$WORKDIR/shard_$i.done" ] && done_shards=$((done_shards + 1)); done
  echo "WORKDIR=$WORKDIR"
  echo "shards: $done_shards/$NUM_SHARDS done"
  echo "dataset: $([ -f "$FULL" ] && wc -l < "$FULL" || echo 0) rows ($([ -f "$FULL" ] && echo present || echo missing))"
  echo "model:   $([ -f "$MODEL" ] && echo present || echo missing)"
  echo "A/B off: $([ -f "$WORKDIR/ab_off.txt" ] && echo present || echo missing)"
  echo "A/B on:  $([ -f "$WORKDIR/ab_on.txt" ] && echo present || echo missing)"
  exit 0
fi

say "=== value-head pipeline === WORKDIR=$WORKDIR  shards=${NUM_SHARDS}x${GAMES_PER_SHARD}  teacher=${GEN_TEACHER_BUDGET_MS}ms  mix=${GEN_MIX_SEARCH_FRAC}  par=${PAR}  ncpu=${NCPU}"
say "(re-run this script any time to RESUME; completed stages are skipped)"

# ---- stage 0: venv with torch (reboot-safe; one-time) ----
if [ ! -x "$VENV/bin/python" ]; then
  say "stage0: creating venv + installing numpy/torch/onnx (one-time, ~minutes)..."
  python3 -m venv "$VENV" && "$VENV/bin/pip" -q install --upgrade pip >/dev/null 2>&1
  "$VENV/bin/pip" -q install numpy torch onnx 2>&1 | tail -1 | tee -a "$LOG"
fi
"$VENV/bin/python" -c 'import torch, numpy, onnx' 2>/dev/null \
  && say "stage0: venv OK" || { say "stage0: venv MISSING torch/onnx — fix the venv and re-run"; exit 1; }

# ---- stage 1: build release (idempotent) ----
say "stage1: building release examples..."
$CARGO build --release -p shengji-core --example gen_training_data --example paired_eval 2>&1 | tail -2 | tee -a "$LOG"
[ -x "$GEN" ] && [ -x "$PEVAL" ] || { say "stage1: build FAILED"; exit 1; }

# ---- stage 2: data-gen shards (parallel, resumable per shard) ----
gen_shard() {
  local i="$1"
  [ -f "$WORKDIR/shard_$i.done" ] && { echo "shard $i: already done"; return 0; }
  GEN_BEHAVIOUR=mix GEN_MIX_SEARCH_FRAC="$GEN_MIX_SEARCH_FRAC" \
    GEN_GAMES="$GAMES_PER_SHARD" GEN_TEACHER_BUDGET_MS="$GEN_TEACHER_BUDGET_MS" \
    GEN_SEED="$((BASE_SEED + i))" GEN_OUT="$WORKDIR/shard_$i.csv" \
    "$GEN" >"$WORKDIR/shard_$i.gen.log" 2>&1
  if [ $? -eq 0 ] && [ -s "$WORKDIR/shard_$i.csv" ]; then
    touch "$WORKDIR/shard_$i.done"; echo "shard $i: OK ($(wc -l < "$WORKDIR/shard_$i.csv") rows)"
  else echo "shard $i: FAILED (see shard_$i.gen.log)"; fi
}
export -f gen_shard
export WORKDIR GAMES_PER_SHARD GEN_TEACHER_BUDGET_MS GEN_MIX_SEARCH_FRAC BASE_SEED GEN
say "stage2: generating $NUM_SHARDS data shards (parallel x$PAR; resumable)..."
seq 0 $((NUM_SHARDS - 1)) | xargs -P "$PAR" -I{} bash -c 'gen_shard "$1"' _ {} 2>&1 | tee -a "$LOG"

missing=0
for i in $(seq 0 $((NUM_SHARDS - 1))); do [ -f "$WORKDIR/shard_$i.done" ] || missing=$((missing + 1)); done
if [ "$missing" -ne 0 ]; then
  say "stage2: $missing/$NUM_SHARDS shards still missing — RE-RUN this script to resume them."
  exit 1
fi

# ---- stage 3: concatenate shards (cheap; redo each run) ----
# CRITICAL: each shard numbers its `group` ids from 0, so a naive concat COLLIDES
# the group namespace — the trainer would merge unrelated decisions under a shared
# id, see sum(labels)>1, and drop nearly everything. Offset each shard's group id
# (column 1) by i*1e7 (>> per-shard group count) so groups are globally unique.
# Only $1 is modified, so the float feature columns are reprinted verbatim.
#
# NOTE: the offset is 1e7, NOT 1e9. On Linux the default awk is mawk, which prints
# integral values > 2^31 (~2.15e9) in scientific notation ("3e+09"); with a 1e9
# offset, shards 3+ would emit group ids like "3e+09" that the trainer's int()
# parse rejects, silently dropping most of the data (or erroring). 1e7 keeps every
# offset < 2^31 for up to ~200 shards while staying far above any per-shard group
# count, so it is collision-free AND mawk-safe. (macOS BSD awk prints big ints
# fine, which is why this only bit on Linux.)
say "stage3: concatenating $NUM_SHARDS shards (with per-shard group-id offsets) -> data_full.csv"
head -1 "$WORKDIR/shard_0.csv" > "$FULL"
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  awk -F, -v OFS=, -v off="$((i * 10000000))" 'NR>1 { $1 = $1 + off; print }' "$WORKDIR/shard_$i.csv" >> "$FULL"
done
say "stage3: dataset has $(wc -l < "$FULL") rows, $(awk -F, 'NR>1{print $1}' "$FULL" | sort -u | wc -l | tr -d ' ') distinct decisions"

# ---- stage 4: train value head (skip if model present) ----
if [ ! -f "$MODEL" ]; then
  say "stage4: training value head ($EPOCHS epochs)..."
  # Train to a .tmp and mv on success, so a kill mid-export can never leave a
  # corrupt $MODEL that a resume would wrongly skip over.
  rm -f "$MODEL.tmp"
  "$VENV/bin/python" training/train_expert.py --data "$FULL" --out "$MODEL.tmp" \
    --epochs "$EPOCHS" --value-weight 1.0 2>&1 | tee -a "$LOG" | grep -E 'Loaded|Value head|Best val|Exported|epoch'
  if [ -f "$MODEL.tmp" ]; then mv "$MODEL.tmp" "$MODEL"; else say "stage4: training FAILED (no model written)"; exit 1; fi
else
  say "stage4: model present (skip training): $MODEL"
fi

# ---- stage 5: paired A/B (blend off vs on; parallel; resumable per condition) ----
run_ab() {
  local w="$1" tag="$2" res="$WORKDIR/ab_$2.txt"
  [ -f "$res" ] && { echo "A/B $tag: already done"; return 0; }
  SHENGJI_EXPERT_MODEL_PATH="$MODEL" SHENGJI_VALUE_WEIGHT="$w" SHENGJI_BOT_BUDGET_MS="$AB_BUDGET_MS" \
    "$PEVAL" "$AB_PAIRS" "$AB_SEED" expert-easy > "$res.tmp" 2>&1 \
    && mv "$res.tmp" "$res" && echo "A/B $tag: OK" || echo "A/B $tag: FAILED"
}
export -f run_ab
export MODEL AB_BUDGET_MS AB_PAIRS AB_SEED PEVAL WORKDIR
say "stage5: paired A/B — value blend OFF (w=0) vs ON (w=0.5), Expert-vs-Easy, $AB_PAIRS pairs (parallel; resumable)..."
printf '0 off\n0.5 on\n' | xargs -P 2 -L1 bash -c 'run_ab "$1" "$2"' _ 2>&1 | tee -a "$LOG"

# ---- summary ----
say "================= SUMMARY ================="
for tag in off on; do
  if [ -f "$WORKDIR/ab_$tag.txt" ]; then
    say "--- value blend $tag ---"
    grep -A2 'Expert vs Easy' "$WORKDIR/ab_$tag.txt" | tee -a "$LOG"
  else
    say "--- value blend $tag: MISSING (re-run to finish) ---"
  fi
done
say "=== DONE === (artifacts in $WORKDIR; model at $MODEL)"
