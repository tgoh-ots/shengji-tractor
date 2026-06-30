#!/usr/bin/env bash
# Resumable schema-v3 dataset / schema-v2 model policy + state-V + action-Q pipeline.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

WORKDIR="${WORKDIR:-$HOME/.shengji-action-value-run}"
NUM_SHARDS="${NUM_SHARDS:-8}"
GAMES_PER_SHARD="${GAMES_PER_SHARD:-100}"
GEN_TEACHER_BUDGET_MS="${GEN_TEACHER_BUDGET_MS:-200}"
GEN_BEHAVIOUR_BUDGET_MS="${GEN_BEHAVIOUR_BUDGET_MS:-80}"
GEN_BEHAVIOUR="${GEN_BEHAVIOUR:-mix}"
GEN_SEAT_BEHAVIOURS="${GEN_SEAT_BEHAVIOURS:-}"
GEN_MIX_SEARCH_FRAC="${GEN_MIX_SEARCH_FRAC:-0.5}"
GEN_Q_CANDIDATES="${GEN_Q_CANDIDATES:-2}"
GEN_Q_ROLLOUT_BEHAVIOUR="${GEN_Q_ROLLOUT_BEHAVIOUR:-easy}"
GEN_Q_ROLLOUT_BUDGET_MS="${GEN_Q_ROLLOUT_BUDGET_MS:-20}"
BASE_SEED="${BASE_SEED:-1000}"
EPOCHS="${EPOCHS:-80}"
POLICY_WEIGHT="${POLICY_WEIGHT:-1.0}"
VALUE_WEIGHT="${VALUE_WEIGHT:-1.0}"
Q_WEIGHT="${Q_WEIGHT:-1.0}"
AUXILIARY_WEIGHT="${AUXILIARY_WEIGHT:-0.25}"
POLICY_TARGET="${POLICY_TARGET:-teacher}"
EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-policy}"
AB_PAIRS="${AB_PAIRS:-200}"
AB_BUDGET_MS="${AB_BUDGET_MS:-150}"
AB_SEED="${AB_SEED:-0x5EED}"
AB_MIN_LEVEL_DELTA="${AB_MIN_LEVEL_DELTA:--0.05}"
RUN_AB="${RUN_AB:-1}"
MIN_SHARD_COMPLETION_RATE="${MIN_SHARD_COMPLETION_RATE:-0.80}"
MIN_DECISIONS_PER_GAME="${MIN_DECISIONS_PER_GAME:-1}"
MIN_Q_ROW_FRACTION="${MIN_Q_ROW_FRACTION:-0.01}"
DATA_ONLY="${DATA_ONLY:-0}"
LEAGUE_CONFIG="${LEAGUE_CONFIG:-}"
LEAGUE_ROUND="${LEAGUE_ROUND:-0}"
SYMMETRY_AUGMENT="${SYMMETRY_AUGMENT:-identity}"
OFFLINE_CONFIG="${OFFLINE_CONFIG:-}"
TEACHER_PRIOR_MODEL="${SHENGJI_EXPERT_MODEL_PATH:-}"
TEACHER_PRIOR_MANIFEST="${SHENGJI_EXPERT_MODEL_MANIFEST:-}"
AB_BASELINE_MODEL="${AB_BASELINE_MODEL:-}"
AB_BASELINE_MANIFEST="${AB_BASELINE_MANIFEST:-}"
AB_BASELINE_GOLDEN="${AB_BASELINE_GOLDEN:-}"
CARGO="${CARGO:-cargo +1.92.0}"
PYTHON="${PYTHON:-python3.13}"
NCPU="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"
PAR="${PAR:-$((NCPU > 1 ? NCPU - 1 : 1))}"

mkdir -p "$WORKDIR"
LOG="$WORKDIR/run.log"
VENV="${VENV:-$WORKDIR/venv}"
GEN="$REPO/target/release/examples/gen_training_data"
CONTROL="$REPO/target/release/examples/model_control_eval"
VALIDATOR="$REPO/target/release/examples/validate_expert_model"
GENERATED="$WORKDIR/data_generated.csv"
FULL="$WORKDIR/data_full.csv"
MODEL="$WORKDIR/action_value.onnx"
CONFIG="$WORKDIR/config.json"
read -r -a CARGO_CMD <<<"$CARGO"

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# Lock every resumability-relevant input. Reusing marker files under a changed
# teacher, schema, source tree, or trainer setting is a hard error, not a warning.
CONFIG_CANDIDATE="$WORKDIR/config.candidate.json"
export REPO NUM_SHARDS GAMES_PER_SHARD GEN_TEACHER_BUDGET_MS GEN_BEHAVIOUR
export GEN_BEHAVIOUR_BUDGET_MS GEN_MIX_SEARCH_FRAC GEN_Q_CANDIDATES GEN_SEAT_BEHAVIOURS
export GEN_Q_ROLLOUT_BEHAVIOUR GEN_Q_ROLLOUT_BUDGET_MS BASE_SEED
export EPOCHS POLICY_WEIGHT VALUE_WEIGHT Q_WEIGHT AUXILIARY_WEIGHT POLICY_TARGET EARLY_STOP_METRIC
export AB_PAIRS AB_BUDGET_MS AB_SEED AB_MIN_LEVEL_DELTA RUN_AB CARGO PYTHON
export MIN_SHARD_COMPLETION_RATE MIN_DECISIONS_PER_GAME MIN_Q_ROW_FRACTION
export DATA_ONLY LEAGUE_CONFIG LEAGUE_ROUND SYMMETRY_AUGMENT OFFLINE_CONFIG
export TEACHER_PRIOR_MODEL TEACHER_PRIOR_MANIFEST AB_BASELINE_MODEL
export AB_BASELINE_MANIFEST AB_BASELINE_GOLDEN VENV
python3 - "$CONFIG_CANDIDATE" <<'PY'
import hashlib, json, os, subprocess, sys
repo = os.environ["REPO"]
def digest(path):
    h = hashlib.sha256()
    with open(os.path.join(repo, path), "rb") as f:
        h.update(f.read())
    return h.hexdigest()
def tree_digest(paths):
    h = hashlib.sha256()
    files = []
    for relative in paths:
        absolute = os.path.join(repo, relative)
        if os.path.isdir(absolute):
            for root, _dirs, names in os.walk(absolute):
                files.extend(os.path.join(root, name) for name in names)
        else:
            files.append(absolute)
    for absolute in sorted(files):
        relative = os.path.relpath(absolute, repo)
        h.update(relative.encode() + b"\0")
        with open(absolute, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
    return h.hexdigest()
keys = [
    "NUM_SHARDS", "GAMES_PER_SHARD", "GEN_TEACHER_BUDGET_MS",
    "GEN_BEHAVIOUR", "GEN_SEAT_BEHAVIOURS", "GEN_BEHAVIOUR_BUDGET_MS", "GEN_MIX_SEARCH_FRAC",
    "GEN_Q_CANDIDATES", "GEN_Q_ROLLOUT_BEHAVIOUR", "GEN_Q_ROLLOUT_BUDGET_MS",
    "BASE_SEED", "EPOCHS", "POLICY_WEIGHT", "VALUE_WEIGHT", "Q_WEIGHT",
    "AUXILIARY_WEIGHT",
    "POLICY_TARGET", "EARLY_STOP_METRIC", "AB_PAIRS", "AB_BUDGET_MS",
    "AB_SEED", "AB_MIN_LEVEL_DELTA", "RUN_AB", "CARGO", "PYTHON",
    "MIN_SHARD_COMPLETION_RATE", "MIN_DECISIONS_PER_GAME", "MIN_Q_ROW_FRACTION",
    "DATA_ONLY", "LEAGUE_ROUND", "SYMMETRY_AUGMENT", "VENV",
]
config = {key.lower(): os.environ[key] for key in keys}
runtime_prefixes = ("SHENGJI_", "GM_", "OMNI_")
compiler_prefixes = (
    "CARGO_", "RUST", "CC_", "CXX_", "AR_", "PKG_CONFIG_",
)
compiler_names = {
    "AR", "CC", "CFLAGS", "CPPFLAGS", "CXX", "CXXFLAGS", "HOST", "LDFLAGS",
    "MACOSX_DEPLOYMENT_TARGET", "RUSTC", "RUSTDOC", "RUSTDOCFLAGS", "RUSTFLAGS",
    "CARGO_BUILD_TARGET", "CARGO_ENCODED_RUSTFLAGS", "CARGO_HOME",
    "CARGO_INCREMENTAL", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER",
    "RUSTUP_HOME", "RUSTUP_TOOLCHAIN", "SDKROOT", "TARGET",
}
def fingerprint_environment_value(name, value):
    file_suffixes = ("_PATH", "_MANIFEST", "_MODEL", "_GOLDEN")
    if value and name.startswith(runtime_prefixes) and name.endswith(file_suffixes):
        absolute = os.path.abspath(value)
        if not os.path.isfile(absolute):
            raise SystemExit(f"{name} points to a missing artifact: {absolute}")
        return {
            "path": absolute,
            "sha256": hashlib.sha256(open(absolute, "rb").read()).hexdigest(),
        }
    return value
influential_environment = {}
for name, value in sorted(os.environ.items()):
    if (
        name.startswith(runtime_prefixes)
        or name.startswith(compiler_prefixes)
        or name in compiler_names
    ):
        influential_environment[name] = fingerprint_environment_value(name, value)
def optional_artifact(path):
    if not path:
        return None
    absolute = os.path.abspath(path)
    if not os.path.isfile(absolute):
        raise SystemExit(f"configured artifact does not exist: {absolute}")
    return {"path": absolute, "sha256": digest(os.path.relpath(absolute, repo))} if absolute.startswith(repo + os.sep) else {
        "path": absolute,
        "sha256": hashlib.sha256(open(absolute, "rb").read()).hexdigest(),
    }
league = optional_artifact(os.environ.get("LEAGUE_CONFIG", ""))
offline = None
if os.environ.get("OFFLINE_CONFIG"):
    offline = json.loads(subprocess.check_output([
        sys.executable,
        os.path.join(repo, "training", "prepare_expert_data.py"),
        "fingerprint-offline", "--config", os.environ["OFFLINE_CONFIG"],
    ], text=True))
config.update({
    "manifest_version": 1,
    "git_head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip(),
    "generator_sha256": digest("core/examples/gen_training_data.rs"),
    "feature_code_sha256": digest("core/src/bot/expert.rs"),
    "trainer_sha256": digest("training/train_expert.py"),
    "pipeline_runner_sha256": digest("training/run_value_pipeline.sh"),
    "data_composer_sha256": digest("training/prepare_expert_data.py"),
    "expert_iteration_runner_sha256": digest("training/expert_iteration.py"),
    "validator_sha256": digest("core/examples/validate_expert_model.rs"),
    "python_lock_sha256": digest("training/requirements.lock.txt"),
    "ab_runner_sha256": digest("training/run_model_ab.sh"),
    "ab_contracts_sha256": digest("training/model_ab.py"),
    "pipeline_contracts_sha256": digest("training/pipeline_contracts.py"),
    "model_control_eval_sha256": digest("core/examples/model_control_eval.rs"),
    "generator_dependency_tree_sha256": tree_digest([
        "core/src", "mechanics/src", "core/Cargo.toml", "mechanics/Cargo.toml",
        "Cargo.toml", "Cargo.lock",
    ]),
    "league_config": league,
    "offline_inputs": offline,
    "teacher_prior_model": optional_artifact(os.environ.get("TEACHER_PRIOR_MODEL", "")),
    "teacher_prior_manifest": optional_artifact(os.environ.get("TEACHER_PRIOR_MANIFEST", "")),
    "ab_baseline_model": optional_artifact(os.environ.get("AB_BASELINE_MODEL", "")),
    "ab_baseline_manifest": optional_artifact(os.environ.get("AB_BASELINE_MANIFEST", "")),
    "ab_baseline_golden": optional_artifact(os.environ.get("AB_BASELINE_GOLDEN", "")),
    "influential_environment": influential_environment,
})
with open(sys.argv[1], "w") as f:
    json.dump(config, f, indent=2, sort_keys=True)
    f.write("\n")
PY
if [[ -f "$CONFIG" ]] && ! cmp -s "$CONFIG" "$CONFIG_CANDIDATE"; then
  say "ERROR: WORKDIR config differs from this run; use a fresh WORKDIR or restore the original settings."
  diff -u "$CONFIG" "$CONFIG_CANDIDATE" | tee -a "$LOG" || true
  exit 2
fi
if [[ ! -f "$CONFIG" ]]; then mv "$CONFIG_CANDIDATE" "$CONFIG"; else rm -f "$CONFIG_CANDIDATE"; fi

if [[ "${STATUS:-0}" == "1" ]]; then
  done_shards=0
  for ((i=0; i<NUM_SHARDS; i++)); do
    [[ -f "$WORKDIR/shard_$i.done" ]] && done_shards=$((done_shards + 1))
  done
  echo "WORKDIR=$WORKDIR"
  echo "config_sha256=$(python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$CONFIG")"
  echo "shards=$done_shards/$NUM_SHARDS"
  echo "dataset=$([[ -f "$FULL" ]] && echo present || echo missing)"
  echo "model=$([[ -f "$MODEL" && -f "$MODEL.manifest.json" && -f "$MODEL.golden.json" ]] && echo present || echo missing)"
  exit 0
fi

say "=== action-value pipeline === workdir=$WORKDIR shards=${NUM_SHARDS}x${GAMES_PER_SHARD} q=${GEN_Q_CANDIDATES}/${GEN_Q_ROLLOUT_BEHAVIOUR}"

if [[ "$DATA_ONLY" != "1" ]]; then
  if [[ ! -x "$VENV/bin/python" ]]; then
    say "stage0: creating training venv"
    "$PYTHON" -m venv "$VENV"
    "$VENV/bin/pip" install -r training/requirements.lock.txt 2>&1 | tee -a "$LOG"
  fi
  "$VENV/bin/python" -c 'import numpy, onnx, torch' || {
    say "stage0: venv is incomplete"; exit 1;
  }
  PY_ENV_LOCK="$WORKDIR/python-environment.freeze.txt"
  PY_ENV_CANDIDATE="$WORKDIR/python-environment.candidate.txt"
  "$VENV/bin/python" - "$VENV/bin/pip" >"$PY_ENV_CANDIDATE" <<'PY'
import platform, subprocess, sys
print("python=" + sys.version.replace("\n", " "))
print("platform=" + platform.platform())
print(subprocess.check_output([sys.argv[1], "freeze", "--all"], text=True), end="")
PY
  if [[ -f "$PY_ENV_LOCK" ]] && ! cmp -s "$PY_ENV_LOCK" "$PY_ENV_CANDIDATE"; then
    say "stage0: Python environment drifted from the recorded full freeze"
    diff -u "$PY_ENV_LOCK" "$PY_ENV_CANDIDATE" | tee -a "$LOG" || true
    exit 2
  fi
  if [[ ! -f "$PY_ENV_LOCK" ]]; then
    mv "$PY_ENV_CANDIDATE" "$PY_ENV_LOCK"
  else
    rm -f "$PY_ENV_CANDIDATE"
  fi
else
  say "stage0: data-only mode; training environment not required"
fi

say "stage1: building seeded generator and evaluator"
if [[ "$DATA_ONLY" == "1" ]]; then
  "${CARGO_CMD[@]}" build --release -p shengji-core \
    --example gen_training_data 2>&1 | tee -a "$LOG"
  [[ -x "$GEN" ]] || { say "stage1: generator binary missing"; exit 1; }
else
  "${CARGO_CMD[@]}" build --release -p shengji-core \
    --example gen_training_data --example model_control_eval \
    --example validate_expert_model 2>&1 | tee -a "$LOG"
  [[ -x "$GEN" && -x "$CONTROL" && -x "$VALIDATOR" ]] || {
    say "stage1: expected binaries missing"; exit 1;
  }
fi

artifact_manifest() {
  local csv="$1" generator_manifest="$2" output="$3" seed="$4" profile="$5"
  python3 - "$csv" "$generator_manifest" "$output" "$seed" "$CONFIG" "$profile" <<'PY'
import hashlib, json, os, sys
csv, generated, output, seed, config, profile = sys.argv[1:]
def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
with open(generated) as f:
    generator = json.load(f)
manifest = {
    "manifest_version": 1,
    "content_sha256": digest(csv),
    "bytes": os.path.getsize(csv),
    "lines": sum(1 for _ in open(csv, "rb")),
    "seed": int(seed),
    "config_sha256": digest(config),
    "league_profile": profile,
    "generator": generator,
}
temporary = output + ".tmp"
with open(temporary, "w") as f:
    json.dump(manifest, f, indent=2, sort_keys=True)
    f.write("\n")
os.replace(temporary, output)
PY
}
export -f artifact_manifest

gen_shard() {
  set -euo pipefail
  local i="$1"
  local seed=$((BASE_SEED + i))
  local csv="$WORKDIR/shard_$i.csv"
  local generated="$WORKDIR/shard_$i.generated.json"
  local artifact="$WORKDIR/shard_$i.artifact.json"
  local profile_name="defaults"
  local behaviour="$GEN_BEHAVIOUR"
  local behaviour_budget="$GEN_BEHAVIOUR_BUDGET_MS"
  local teacher_budget="$GEN_TEACHER_BUDGET_MS"
  local q_candidates="$GEN_Q_CANDIDATES"
  local q_rollout="$GEN_Q_ROLLOUT_BEHAVIOUR"
  local q_budget="$GEN_Q_ROLLOUT_BUDGET_MS"
  local seat_behaviours="$GEN_SEAT_BEHAVIOURS"
  local mix_fraction="$GEN_MIX_SEARCH_FRAC"
  if [[ -n "$LEAGUE_CONFIG" ]]; then
    local resolved
    resolved="$(python3 training/expert_iteration.py profile-tsv \
      --config "$LEAGUE_CONFIG" --round "$LEAGUE_ROUND" --shard "$i")"
    IFS=$'\t' read -r profile_name behaviour behaviour_budget teacher_budget \
      q_candidates q_rollout q_budget seat_behaviours mix_fraction <<<"$resolved"
    [[ "$seat_behaviours" == "-" ]] && seat_behaviours=""
    [[ -n "$profile_name" && -n "$q_budget" ]] || {
      echo "shard $i: invalid resolved league profile" >&2; return 2;
    }
  fi
  if [[ -f "$WORKDIR/shard_$i.done" && -s "$csv" && -s "$generated" && -s "$artifact" ]]; then
    echo "shard $i: already complete"
    return
  fi
  rm -f "$WORKDIR/shard_$i.done" "$csv" "$generated" "$artifact"
  local -a generator_env=(
    "GEN_BEHAVIOUR=$behaviour"
    "GEN_MIX_SEARCH_FRAC=$mix_fraction"
    "GEN_GAMES=$GAMES_PER_SHARD"
    "GEN_TEACHER_BUDGET_MS=$teacher_budget"
    "GEN_BEHAVIOUR_BUDGET_MS=$behaviour_budget"
    "GEN_Q_CANDIDATES=$q_candidates"
    "GEN_Q_ROLLOUT_BEHAVIOUR=$q_rollout"
    "GEN_Q_ROLLOUT_BUDGET_MS=$q_budget"
    "GEN_SEED=$seed"
    "GEN_OUT=$csv"
    "GEN_MANIFEST=$generated"
  )
  if [[ -n "$seat_behaviours" ]]; then
    env "${generator_env[@]}" "GEN_SEAT_BEHAVIOURS=$seat_behaviours" \
      "$GEN" >"$WORKDIR/shard_$i.gen.log" 2>&1
  else
    env -u GEN_SEAT_BEHAVIOURS "${generator_env[@]}" \
      "$GEN" >"$WORKDIR/shard_$i.gen.log" 2>&1
  fi
  artifact_manifest "$csv" "$generated" "$artifact" "$seed" "$profile_name"
  touch "$WORKDIR/shard_$i.done"
  echo "shard $i: OK ($(wc -l <"$csv") lines)"
}
export -f gen_shard
export WORKDIR BASE_SEED GEN_BEHAVIOUR GEN_MIX_SEARCH_FRAC GAMES_PER_SHARD
export GEN_TEACHER_BUDGET_MS GEN_BEHAVIOUR_BUDGET_MS GEN_Q_CANDIDATES
export GEN_Q_ROLLOUT_BEHAVIOUR GEN_Q_ROLLOUT_BUDGET_MS GEN CONFIG
export GEN_SEAT_BEHAVIOURS
export LEAGUE_CONFIG LEAGUE_ROUND

say "stage2: generating shards with parallelism=$PAR"
seq 0 $((NUM_SHARDS - 1)) | xargs -P "$PAR" -I{} bash -c 'gen_shard "$1"' _ {} \
  2>&1 | tee -a "$LOG"
for ((i=0; i<NUM_SHARDS; i++)); do
  [[ -f "$WORKDIR/shard_$i.done" ]] || { say "stage2: shard $i failed"; exit 1; }
done

say "stage3: validating and concatenating collision-free string IDs"
python3 - "$WORKDIR" "$NUM_SHARDS" "$GENERATED" "$CONFIG" \
  "$GAMES_PER_SHARD" "$MIN_SHARD_COMPLETION_RATE" \
  "$MIN_DECISIONS_PER_GAME" "$MIN_Q_ROW_FRACTION" <<'PY'
import csv, hashlib, json, os, sys
(
    workdir, count, output, config, games_per_shard,
    minimum_completion, minimum_decisions, minimum_q_fraction,
) = sys.argv[1:]
count = int(count)
games_per_shard = int(games_per_shard)
minimum_completion = float(minimum_completion)
minimum_decisions = float(minimum_decisions)
minimum_q_fraction = float(minimum_q_fraction)
if not 0.0 <= minimum_completion <= 1.0:
    raise SystemExit("MIN_SHARD_COMPLETION_RATE must be in [0,1]")
if minimum_decisions < 0.0:
    raise SystemExit("MIN_DECISIONS_PER_GAME cannot be negative")
if not 0.0 <= minimum_q_fraction <= 1.0:
    raise SystemExit("MIN_Q_ROW_FRACTION must be in [0,1]")
temporary = output + ".tmp"
def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()
header = None
game_ids, groups, sources = set(), set(), []
game_owner, group_owner = {}, {}
rows = 0
with open(temporary, "w", newline="") as out:
    writer = None
    for i in range(count):
        path = os.path.join(workdir, f"shard_{i}.csv")
        artifact = os.path.join(workdir, f"shard_{i}.artifact.json")
        with open(artifact) as f:
            source = json.load(f)
        if source.get("content_sha256") != digest(path):
            raise SystemExit(f"artifact content hash mismatch in {path}")
        if source.get("config_sha256") != digest(config):
            raise SystemExit(f"artifact config hash mismatch in {artifact}")
        generated = source.get("generator", {})
        if (
            generated.get("dataset_schema_version") != 3
            or generated.get("feature_schema_version") != 2
            or generated.get("feature_dim") != 49
        ):
            raise SystemExit(f"unsupported generator contract in {artifact}")
        local_rows = 0
        local_q_rows = 0
        local_games = set()
        local_groups = set()
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if header is None:
                header = reader.fieldnames
                writer = csv.DictWriter(out, fieldnames=header, lineterminator="\n")
                writer.writeheader()
            elif reader.fieldnames != header:
                raise SystemExit(f"header mismatch in {path}")
            for row in reader:
                if row["game_id"] in game_owner and game_owner[row["game_id"]] != i:
                    raise SystemExit(f"cross-shard game_id collision: {row['game_id']}")
                if row["group"] in group_owner and group_owner[row["group"]] != i:
                    raise SystemExit(f"cross-shard group collision: {row['group']}")
                game_owner[row["game_id"]] = i
                group_owner[row["group"]] = i
                game_ids.add(row["game_id"])
                groups.add(row["group"])
                local_games.add(row["game_id"])
                local_groups.add(row["group"])
                local_rows += 1
                local_q_rows += bool(row.get("q_target", "").strip())
                writer.writerow(row)
                rows += 1
        declared = {
            "games_requested": games_per_shard,
            "games_completed": len(local_games),
            "rows": local_rows,
            "decisions": len(local_groups),
            "q_rows": local_q_rows,
        }
        for name, actual in declared.items():
            if generated.get(name) != actual:
                raise SystemExit(
                    f"shard {i}: generator {name}={generated.get(name)!r}, CSV has {actual}"
                )
        if source.get("lines") != local_rows + 1:
            raise SystemExit(f"shard {i}: artifact line count does not match CSV")
        completion = len(local_games) / max(1, games_per_shard)
        if completion < minimum_completion:
            raise SystemExit(
                f"shard {i}: completion {completion:.3f} below {minimum_completion:.3f}"
            )
        decisions_per_game = len(local_groups) / max(1, len(local_games))
        if decisions_per_game < minimum_decisions:
            raise SystemExit(
                f"shard {i}: decisions/game {decisions_per_game:.3f} below "
                f"{minimum_decisions:.3f}"
            )
        if local_rows < 2 * len(local_groups):
            raise SystemExit(f"shard {i}: fewer than two rows per listwise decision")
        if generated.get("q_candidates") != "0":
            q_fraction = local_q_rows / max(1, local_rows)
            if q_fraction < minimum_q_fraction:
                raise SystemExit(
                    f"shard {i}: Q-row fraction {q_fraction:.3f} below "
                    f"{minimum_q_fraction:.3f}"
                )
        source["validated_quality"] = {
            "completion_rate": completion,
            "decisions_per_completed_game": decisions_per_game,
            "q_row_fraction": local_q_rows / max(1, local_rows),
            "minimum_completion_rate": minimum_completion,
            "minimum_decisions_per_game": minimum_decisions,
            "minimum_q_row_fraction_when_enabled": minimum_q_fraction,
        }
        sources.append(source)
os.replace(temporary, output)
manifest = {
    "manifest_version": 1,
    "content_sha256": digest(output),
    "config_sha256": digest(config),
    "rows": rows,
    "games": len(game_ids),
    "decisions": len(groups),
    "source_content_sha256": [source["content_sha256"] for source in sources],
    "source_quality": [source["validated_quality"] for source in sources],
}
with open(output + ".manifest.json.tmp", "w") as f:
    json.dump(manifest, f, indent=2, sort_keys=True)
    f.write("\n")
os.replace(output + ".manifest.json.tmp", output + ".manifest.json")
PY
say "stage3: composing symmetry/replay sources (augmentation=$SYMMETRY_AUGMENT)"
COMPOSE_ARGS=(
  training/prepare_expert_data.py compose
  --generated "$GENERATED"
  --out "$FULL"
  --augmentation "$SYMMETRY_AUGMENT"
)
if [[ -n "$OFFLINE_CONFIG" ]]; then
  COMPOSE_ARGS+=(--offline-config "$OFFLINE_CONFIG")
fi
python3 "${COMPOSE_ARGS[@]}" >"$WORKDIR/data-compose.json"
say "stage3: $(wc -l <"$FULL") lines; manifest=$FULL.manifest.json"

if [[ "$DATA_ONLY" == "1" ]]; then
  say "=== DATA ONLY DONE === dataset=$FULL manifest=$FULL.manifest.json"
  exit 0
fi

MODEL_REUSABLE=0
if [[ -s "$MODEL" && -s "$MODEL.manifest.json" && -s "$MODEL.golden.json" ]] && \
  python3 training/pipeline_contracts.py verify-model-resume \
    --model "$MODEL" --dataset "$FULL" --epochs "$EPOCHS" \
    --policy-weight "$POLICY_WEIGHT" --value-weight "$VALUE_WEIGHT" \
    --q-weight "$Q_WEIGHT" --auxiliary-weight "$AUXILIARY_WEIGHT" \
    --policy-target "$POLICY_TARGET" --early-stop-metric "$EARLY_STOP_METRIC"
then
  MODEL_REUSABLE=1
fi

if [[ "$MODEL_REUSABLE" != "1" ]]; then
  say "stage4: training policy/state-V/action-Q model"
  rm -f "$MODEL" "$MODEL.manifest.json" "$MODEL.golden.json"
  "$VENV/bin/python" training/train_expert.py \
    --data "$FULL" --out "$MODEL" --epochs "$EPOCHS" \
    --manifest-out "$MODEL.manifest.json" --golden-out "$MODEL.golden.json" \
    --policy-weight "$POLICY_WEIGHT" --value-weight "$VALUE_WEIGHT" \
    --q-weight "$Q_WEIGHT" --policy-target "$POLICY_TARGET" \
    --auxiliary-weight "$AUXILIARY_WEIGHT" \
    --early-stop-metric "$EARLY_STOP_METRIC" 2>&1 | tee -a "$LOG"
  [[ -s "$MODEL" && -s "$MODEL.manifest.json" && -s "$MODEL.golden.json" ]] || {
    say "stage4: model, manifest, or golden vectors missing"; exit 1;
  }
else
  say "stage4: model artifacts match the current dataset and training contract"
fi

say "stage5: validating PyTorch -> ONNX -> tract numerical parity"
"$VALIDATOR" "$MODEL" "$MODEL.manifest.json" "$MODEL.golden.json" 2>&1 | tee -a "$LOG"

if [[ "$RUN_AB" == "1" && "$AB_PAIRS" -gt 0 ]]; then
  RESUME_ARGS=(
    training/model_ab.py validate-resume
    --comparison "$WORKDIR/model-ab/comparison.json"
    --model "$MODEL"
    --manifest "$MODEL.manifest.json"
    --golden "$MODEL.golden.json"
    --pairs "$AB_PAIRS"
    --seed "$AB_SEED"
    --budget-ms "$AB_BUDGET_MS"
    --minimum-level-delta "$AB_MIN_LEVEL_DELTA"
  )
  if [[ -n "$AB_BASELINE_MODEL" ]]; then
    RESUME_ARGS+=(
      --baseline-model "$AB_BASELINE_MODEL"
      --baseline-manifest "$AB_BASELINE_MANIFEST"
      --baseline-golden "$AB_BASELINE_GOLDEN"
    )
  fi
  if [[ -s "$WORKDIR/model-ab/comparison.json" ]] && python3 "${RESUME_ARGS[@]}"
  then
    say "stage6: model A/B already present"
  else
    say "stage6: two-process matched-deal baseline vs candidate model A/B"
    AB_PAIRS="$AB_PAIRS" AB_SEED="$AB_SEED" AB_BUDGET_MS="$AB_BUDGET_MS" \
      AB_MIN_LEVEL_DELTA="$AB_MIN_LEVEL_DELTA" \
      AB_CANDIDATE_MANIFEST="$MODEL.manifest.json" \
      AB_CANDIDATE_GOLDEN="$MODEL.golden.json" \
      CARGO="$CARGO" training/run_model_ab.sh "$MODEL" "$WORKDIR/model-ab" \
      2>&1 | tee -a "$LOG"
  fi
else
  say "stage6: skipped (RUN_AB=$RUN_AB AB_PAIRS=$AB_PAIRS)"
fi

say "=== DONE === dataset=$FULL model=$MODEL manifest=$MODEL.manifest.json"
