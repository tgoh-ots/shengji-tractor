#!/usr/bin/env bash
#
# Turn-key setup for a fresh CPU cloud box to run the heavy self-play / data-gen
# phases of the superior-play bot training plan (docs/bot-superplay-plan.md).
#
# WHY A CLOUD CPU BOX (not a GPU): the bottleneck is Rust self-play data
# generation, which is embarrassingly parallel across cores. The nets are tiny and
# train in minutes on CPU; a GPU barely helps. So rent CORES, not a GPU.
#
# RECOMMENDED INSTANCES (any Ubuntu 22.04/24.04 x86-64 or arm64 box works):
#   - AWS        c7i.24xlarge (96 vCPU) / c7g.16xlarge (64 vCPU, arm)   spot ~$1-2/hr
#   - GCP        c4-standard-96 (96 vCPU)
#   - Hetzner    CCX63 (48 vCPU dedicated)  ~€0.5/hr  (cheapest for this)
#   - Azure      F72s_v2 (72 vCPU)
# 64-128 vCPU gives a ~6-13x speedup over the 10-core dev box on every gen phase.
#
# USAGE (from your laptop):
#   1. Provision the box, note its IP. Ensure ssh access.
#   2. Push the repo up (fast, excludes target/ and node_modules/):
#        rsync -az --exclude target --exclude node_modules --exclude .git \
#          ./ ubuntu@$IP:~/shengji/
#   3. ssh in and run this script:
#        ssh ubuntu@$IP 'cd ~/shengji && bash training/cloud_setup.sh'
#   4. Kick off (or resume) the value-head pipeline using ALL cores:
#        ssh ubuntu@$IP 'cd ~/shengji && \
#          NUM_SHARDS=$(nproc) GAMES_PER_SHARD=1000 GEN_TEACHER_BUDGET_MS=400 \
#          GEN_MIX_SEARCH_FRAC=0.6 AB_PAIRS=600 AB_BUDGET_MS=300 \
#          nohup bash training/run_value_pipeline.sh > ~/run.out 2>&1 &'
#   5. Watch:   ssh ubuntu@$IP 'tail -f ~/.shengji-value-run/run.log'
#      Status:  ssh ubuntu@$IP 'cd ~/shengji && STATUS=1 bash training/run_value_pipeline.sh'
#   6. Pull artifacts back when done:
#        rsync -az ubuntu@$IP:~/.shengji-value-run/value.onnx        ./candidate.onnx
#        rsync -az ubuntu@$IP:~/.shengji-value-run/data_full.csv     ./training/data.csv
#        rsync -az ubuntu@$IP:~/.shengji-value-run/ab_*.txt          ./
#
# The Oracle (DMC) and Athena (ExIt) phases have their own runners
# (training/run_dmc_pipeline.sh, training/run_exit_pipeline.sh — added in their
# phases); each is likewise resumable and parallel and runs the same way here.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
echo "== repo: $REPO  cores: $(nproc 2>/dev/null || echo '?') =="

# ---- system deps ----
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y build-essential curl git pkg-config libssl-dev python3-venv python3-pip
fi

# ---- rust toolchain (>= 1.87; pin 1.92 to match the repo) ----
if ! command -v rustup >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  # shellcheck disable=SC1090
  source "$HOME/.cargo/env"
fi
source "$HOME/.cargo/env" 2>/dev/null || true
rustup toolchain install 1.92.0 --profile minimal
echo "== rust: $(cargo +1.92.0 --version) =="

# ---- python venv with torch (CPU) ----
VENV="${WORKDIR:-$HOME/.shengji-value-run}/venv"
mkdir -p "$(dirname "$VENV")"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" -q install --upgrade pip
  # CPU torch wheel (no CUDA) — the nets are tiny; CPU is correct here.
  "$VENV/bin/pip" install numpy onnx --quiet
  "$VENV/bin/pip" install torch --index-url https://download.pytorch.org/whl/cpu --quiet
fi
"$VENV/bin/python" -c 'import torch,numpy,onnx; print("torch",torch.__version__)'

# ---- build the release examples once (warms the cache) ----
cargo +1.92.0 build --release -p shengji-core \
  --example gen_training_data --example paired_eval --example tournament
echo "== build OK. Ready. Kick off run_value_pipeline.sh with NUM_SHARDS=\$(nproc). =="
