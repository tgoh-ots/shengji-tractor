#!/usr/bin/env bash
# Matched-deal embedded-model vs candidate-model evaluation in separate
# processes, so the Expert model OnceLock cannot contaminate either arm.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

MODEL="${1:?usage: run_model_ab.sh MODEL.onnx [OUTDIR]}"
OUTDIR="${2:-${AB_OUTDIR:-$PWD/model-ab}}"
MANIFEST="${SHENGJI_EXPERT_MODEL_MANIFEST:-$MODEL.manifest.json}"
GOLDEN="${SHENGJI_EXPERT_MODEL_GOLDEN:-$MODEL.golden.json}"
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

"${CARGO_CMD[@]}" build --release -p shengji-core \
  --example model_control_eval --example validate_expert_model
VALIDATOR="$REPO/target/release/examples/validate_expert_model"
CONTROL="$REPO/target/release/examples/model_control_eval"
"$VALIDATOR" "$MODEL" "$MANIFEST" "$GOLDEN" | tee "$OUTDIR/parity.txt"

# These are intentionally two OS processes. Each receives the identical deal
# sequence and compute budget, but initializes its own model OnceLock.
env -u SHENGJI_EXPERT_MODEL_PATH -u SHENGJI_EXPERT_MODEL_MANIFEST \
  SHENGJI_BOT_BUDGET_MS="$BUDGET_MS" \
  "$CONTROL" "$PAIRS" "$SEED" >"$OUTDIR/embedded.json" 2>"$OUTDIR/embedded.log"
SHENGJI_EXPERT_MODEL_PATH="$MODEL" \
  SHENGJI_EXPERT_MODEL_MANIFEST="$MANIFEST" \
  SHENGJI_BOT_BUDGET_MS="$BUDGET_MS" \
  "$CONTROL" "$PAIRS" "$SEED" >"$OUTDIR/candidate.json" 2>"$OUTDIR/candidate.log"

python3 - "$MODEL" "$MANIFEST" "$GOLDEN" "$OUTDIR" "$PAIRS" "$SEED" "$BUDGET_MS" <<'PY'
import hashlib, json, os, random, statistics, sys
model, manifest, golden, out, pairs, seed, budget = sys.argv[1:]
def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
with open(os.path.join(out, "embedded.json")) as f:
    embedded = json.load(f)
with open(os.path.join(out, "candidate.json")) as f:
    candidate = json.load(f)
if embedded["complete_pairs"] != int(pairs) or candidate["complete_pairs"] != int(pairs):
    raise SystemExit("every arm must complete every pair for exact cross-process pairing")
def comparison(name):
    left = candidate[name]
    right = embedded[name]
    if len(left) != len(right):
        raise SystemExit(f"{name}: arm lengths differ")
    delta = [a - b for a, b in zip(left, right)]
    rng = random.Random(0xAB51)
    means = []
    for _ in range(5000):
        means.append(statistics.fmean(delta[rng.randrange(len(delta))] for _ in delta))
    means.sort()
    return {
        "candidate_minus_embedded": statistics.fmean(delta),
        "paired_bootstrap95": [means[125], means[4874]],
        "per_deck_delta": delta,
    }
payload = {
    "manifest_version": 1,
    "method": "two-process matched-deal difference-in-control-outcomes",
    "pairs": int(pairs),
    "seed": seed,
    "budget_ms": int(budget),
    "candidate_model_sha256": digest(model),
    "candidate_manifest_sha256": digest(manifest),
    "golden_sha256": digest(golden),
    "embedded_result_sha256": digest(os.path.join(out, "embedded.json")),
    "candidate_result_sha256": digest(os.path.join(out, "candidate.json")),
    "winrate": comparison("per_deck_winrate"),
    "point_margin": comparison("per_deck_margin"),
    "level_utility": comparison("per_deck_level_utility"),
}
with open(os.path.join(out, "comparison.json"), "w") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write("\n")
print(json.dumps({key: payload[key] for key in ("winrate", "point_margin", "level_utility")}, indent=2))
minimum = os.environ.get("AB_MIN_LEVEL_DELTA")
if minimum is not None and payload["level_utility"]["candidate_minus_embedded"] < float(minimum):
    raise SystemExit(f"candidate level delta fails AB_MIN_LEVEL_DELTA={minimum}")
PY

echo "Embedded arm: $OUTDIR/embedded.json"
echo "Candidate arm: $OUTDIR/candidate.json"
echo "Paired candidate-minus-embedded estimate: $OUTDIR/comparison.json"
