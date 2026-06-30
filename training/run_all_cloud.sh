#!/usr/bin/env bash
#
# FULL-SCALE master runner for a big CPU box (after training/cloud_setup.sh).
# Runs every measurable phase at "expensive" scale, resumably, using all cores.
# Each sub-pipeline is independently checkpointed, so re-running resumes.
#
#   bash training/cloud_setup.sh          # once: rust + venv + build
#   nohup bash training/run_all_cloud.sh > ~/all.out 2>&1 &
#   tail -f ~/all.out
#
# Scale is tuned for ~64-128 vCPU. Override any env below. The whole point of the
# box: data-gen is CPU-bound and embarrassingly parallel, so MORE CORES = strictly
# more/better data, which is exactly what Oracle (needs DouZero-scale) and a
# well-calibrated Sage value head need — the two laptop-scale negative results.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
NCPU="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 16)"
PEVAL="$REPO/target/release/examples/paired_eval"
ORACLE_EVAL="$REPO/target/release/examples/oracle_eval"
say(){ echo "[$(date '+%F %T')] $*"; }

# ============================================================================
# Phase 1 — SAGE: a PROPERLY-calibrated value head + a real weight SWEEP.
#   Laptop finding: value blend at w=0.5 HURT on a 2,100-game/200ms-teacher net.
#   Fix on the box: 5-10x the data, a stronger teacher, then sweep the weight low.
# ============================================================================
say "=== SAGE (value head, full scale) ==="
# NOTE: wall-clock per phase ≈ GAMES_PER_SHARD × per-game-time (each core runs
# GAMES_PER_SHARD games serially, all cores in parallel); TOTAL data ≈ nproc ×
# GAMES_PER_SHARD. So on a many-core box keep per-shard counts modest — data still
# scales with cores. Search-gen (Sage/Athena) is ~decisions×budget per game
# (~15s at 300ms); DMC gen (Oracle) is ~0.05s/game.
NUM_SHARDS="${SAGE_SHARDS:-$NCPU}" GAMES_PER_SHARD="${SAGE_GPS:-500}" \
  GEN_TEACHER_BUDGET_MS="${SAGE_TEACHER_MS:-400}" GEN_MIX_SEARCH_FRAC=0.6 \
  AB_PAIRS="${SAGE_AB_PAIRS:-600}" AB_BUDGET_MS=300 EPOCHS=150 \
  WORKDIR="$HOME/.sage-cloud" bash training/run_value_pipeline.sh || say "sage pipeline returned nonzero (resume by re-running)"
# Weight sweep on the freshly-trained value net (paired vs Easy; pick the best w).
# Run the weights IN PARALLEL (each paired_eval is single-threaded; the box has
# cores to spare) and keep pairs/budget modest so the sweep is ~1h, not ~12h.
SAGE_NET="$HOME/.sage-cloud/value.onnx"
if [ -f "$SAGE_NET" ]; then
  sweep_w() {
    SHENGJI_EXPERT_MODEL_PATH="$SAGE_NET" SHENGJI_VALUE_WEIGHT="$1" SHENGJI_BOT_BUDGET_MS=200 \
      "$PEVAL" 400 0x5EED expert-easy 2>&1 | grep -E 'win-rate|margin' | sed "s/^/  [w=$1] /"
  }
  export -f sweep_w; export SAGE_NET PEVAL
  printf '0.0\n0.15\n0.3\n0.5\n' | xargs -P 4 -I{} bash -c 'sweep_w "$1"' _ {}
fi

# ============================================================================
# Phase 2 — ORACLE: DouZero-SCALE Deep Monte-Carlo (the laptop ran out of samples).
#   12 iterations x (nproc x 5000) games ~ tens of millions of (s,a,return) rows.
# ============================================================================
say "=== ORACLE (DMC, DouZero scale) ==="
DMC_ITERS="${ORACLE_ITERS:-12}" DMC_SHARDS="$NCPU" DMC_GAMES_PER_SHARD="${ORACLE_GPS:-2000}" \
  DMC_EPSILON=0.10 DMC_EPOCHS=80 WORKDIR="$HOME/.oracle-cloud" \
  bash training/run_dmc_pipeline.sh || say "oracle pipeline returned nonzero (resume by re-running)"
ORACLE_NET="$HOME/.oracle-cloud/oracle.onnx"
[ -f "$ORACLE_NET" ] && SHENGJI_EXPERT_MODEL_PATH="$ORACLE_NET" SHENGJI_BOT_BUDGET_MS=200 \
  "$ORACLE_EVAL" 400 0x0AC1E all 2>&1 | grep -E 'vs|win-rate|margin'

# ============================================================================
# Phase 3 — ATHENA: full multi-iteration honest-search ExIt, proper data + budget.
# ============================================================================
say "=== ATHENA (ExIt, full scale) ==="
EXIT_ITERS="${ATHENA_ITERS:-3}" EXIT_SHARDS="$NCPU" EXIT_GAMES_PER_SHARD="${ATHENA_GPS:-400}" \
  EXIT_BUDGET_MS="${ATHENA_MS:-300}" EXIT_EPOCHS=150 WORKDIR="$HOME/.athena-cloud" \
  bash training/run_exit_pipeline.sh || say "athena pipeline returned nonzero (resume by re-running)"
ATHENA_NET="$HOME/.athena-cloud/athena.onnx"
if [ -f "$ATHENA_NET" ]; then
  say "ATHENA deployed (Expert+athena-net) vs Easy + Enoch-vs-Athena"
  SHENGJI_EXPERT_MODEL_PATH="$ATHENA_NET" SHENGJI_BOT_BUDGET_MS=250 "$PEVAL" 400 0x5EED search 2>&1 | grep -E 'win-rate|margin'
fi

# ============================================================================
# Reference ladder anchors (Fast-Expert = embedded net, search-free, vs ladder).
# ============================================================================
say "=== reference: Fast-Expert (embedded net) vs Easy/Enoch/Omniscient ==="
SHENGJI_BOT_BUDGET_MS=200 "$ORACLE_EVAL" 300 0x0AC1E all 2>&1 | grep -E 'vs|win-rate' || true
say "=== ALL DONE === nets: $SAGE_NET  $ORACLE_NET  $ATHENA_NET"
