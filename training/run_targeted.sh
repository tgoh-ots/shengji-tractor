#!/usr/bin/env bash
# Targeted cloud experiment (pivot after the parallel session's learnings):
#  1. ATHENA cloning ENOCH (the strong honest teacher) — aiming for a search-free
#     Enoch-strength "Fast" tier (the most deployable win neither session has).
#  2. ORACLE DMC scale-up.
# Skips Sage (the parallel session already showed the value head is neutral, with a
# better teacher than ours). Benchmarks search-free + deployed vs Easy/Enoch.
trap '' TERM HUP   # don't let a stray reaper signal kill the long run (children inherit SIG_IGN)
set -u
cd /app
export CARGO=cargo CARGO_PROFILE_RELEASE_LTO=false

echo "=== [$(date -u +%T)] ATHENA: ExIt cloning ENOCH (3 iters) ==="
EXIT_TIER=enoch EXIT_ITERS=3 EXIT_SHARDS=15 EXIT_GAMES_PER_SHARD=300 \
  EXIT_BUDGET_MS=300 EXIT_EPOCHS=150 WORKDIR=/root/.athena-enoch \
  bash training/run_exit_pipeline.sh

AE=/root/.athena-enoch/athena.onnx
if [ -f "$AE" ]; then
  echo "=== [$(date -u +%T)] Athena(Enoch) SEARCH-FREE vs Easy / Enoch ==="
  SHENGJI_EXPERT_MODEL_PATH="$AE" target/release/examples/oracle_eval 400 0x0AC1E easy
  SHENGJI_EXPERT_MODEL_PATH="$AE" target/release/examples/oracle_eval 400 0x0AC1E enoch
  echo "=== [$(date -u +%T)] BASELINE embedded-Expert SEARCH-FREE vs Easy / Enoch ==="
  target/release/examples/oracle_eval 400 0x0AC1E easy
  target/release/examples/oracle_eval 400 0x0AC1E enoch
  echo "=== [$(date -u +%T)] Athena(Enoch) DEPLOYED (Expert search + net) vs Easy + Enoch-vs-it ==="
  SHENGJI_EXPERT_MODEL_PATH="$AE" SHENGJI_BOT_BUDGET_MS=250 \
    target/release/examples/paired_eval 400 0x5EED search
fi

echo "=== [$(date -u +%T)] ORACLE: DMC scale (8 iters) ==="
DMC_ITERS=8 DMC_SHARDS=15 DMC_GAMES_PER_SHARD=1000 DMC_EPOCHS=80 \
  WORKDIR=/root/.oracle-cloud bash training/run_dmc_pipeline.sh
OR=/root/.oracle-cloud/oracle.onnx
[ -f "$OR" ] && SHENGJI_EXPERT_MODEL_PATH="$OR" SHENGJI_BOT_BUDGET_MS=200 \
  target/release/examples/oracle_eval 400 0x0AC1E all

echo "=== [$(date -u +%T)] ALL DONE ==="
