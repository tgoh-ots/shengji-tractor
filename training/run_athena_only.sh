#!/usr/bin/env bash
# Re-run ONLY Athena cloning ENOCH (Oracle already measured: weak even at scale).
trap '' TERM HUP
set -u
cd /app
export CARGO=cargo CARGO_PROFILE_RELEASE_LTO=false
echo "=== [$(date -u +%T)] ATHENA cloning ENOCH (3 ExIt iters) ==="
# 80 games/shard × 15 = 1200 games/iter ≈ 66k decisions — memory-safe (the 248k-decision
# run OOM-killed train_expert's padded eval tensor on this no-swap box) and still ~9× the
# laptop's single-iter scale.
EXIT_TIER=enoch EXIT_ITERS=3 EXIT_SHARDS=15 EXIT_GAMES_PER_SHARD=80 \
  EXIT_BUDGET_MS=300 EXIT_EPOCHS=150 WORKDIR=/root/.athena-enoch \
  bash training/run_exit_pipeline.sh
AE=/root/.athena-enoch/athena.onnx
if [ -f "$AE" ]; then
  echo "### [SF] Athena(Enoch) vs Easy";   SHENGJI_EXPERT_MODEL_PATH="$AE" target/release/examples/oracle_eval 400 0x0AC1E easy
  echo "### [SF] Athena(Enoch) vs Enoch";  SHENGJI_EXPERT_MODEL_PATH="$AE" target/release/examples/oracle_eval 400 0x0AC1E enoch
  echo "### [SF] BASELINE embedded vs Easy";  target/release/examples/oracle_eval 400 0x0AC1E easy
  echo "### [SF] BASELINE embedded vs Enoch"; target/release/examples/oracle_eval 400 0x0AC1E enoch
  echo "### [DEPLOYED] Athena(Enoch) Expert+net search vs Easy + Enoch-vs-it"
  SHENGJI_EXPERT_MODEL_PATH="$AE" SHENGJI_BOT_BUDGET_MS=250 target/release/examples/paired_eval 400 0x5EED search
fi
echo "=== ALL DONE ==="
