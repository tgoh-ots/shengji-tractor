#!/usr/bin/env python3
"""Run bounded, resumable search-teacher expert-iteration experiments.

Each round delegates generation/training/parity/A-B work to
``run_value_pipeline.sh``.  A prior round's candidate model is loaded into the
next round's search policy and becomes that round's matched-deal baseline.  The
loop is intentionally bounded by the checked configuration; it never promotes
or deploys a model automatically.

Profiles can use one fallback behaviour for a whole trajectory or exactly four
seat behaviours for heterogeneous partner/opponent coverage. Shard assignment
remains deterministic and each generated row records its actor's policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


MANIFEST_VERSION = 1
BEHAVIOURS = {"easy", "expert", "enoch", "grandmaster", "mix"}
ROLLOUTS = {"easy", "expert", "enoch"}
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

PIPELINE_ENV = {
    "num_shards": "NUM_SHARDS",
    "games_per_shard": "GAMES_PER_SHARD",
    "base_seed": "BASE_SEED",
    "epochs": "EPOCHS",
    "policy_weight": "POLICY_WEIGHT",
    "value_weight": "VALUE_WEIGHT",
    "q_weight": "Q_WEIGHT",
    "auxiliary_weight": "AUXILIARY_WEIGHT",
    "policy_target": "POLICY_TARGET",
    "early_stop_metric": "EARLY_STOP_METRIC",
    "ab_pairs": "AB_PAIRS",
    "ab_budget_ms": "AB_BUDGET_MS",
    "ab_seed": "AB_SEED",
    "ab_min_level_delta": "AB_MIN_LEVEL_DELTA",
    "run_ab": "RUN_AB",
    "parallelism": "PAR",
    "cargo": "CARGO",
    "python": "PYTHON",
    "venv": "VENV",
    "min_shard_completion_rate": "MIN_SHARD_COMPLETION_RATE",
    "min_decisions_per_game": "MIN_DECISIONS_PER_GAME",
    "min_q_row_fraction": "MIN_Q_ROW_FRACTION",
}

# Expert-iteration configs are the authority for experiments. Search/model
# knobs and build-target flags inherited from an interactive shell can
# otherwise make two shards under one resume lock mean different things.
SANITIZED_ENV_PREFIXES = (
    "SHENGJI_",
    "GM_",
    "OMNI_",
    "CARGO_TARGET_",
    "CARGO_PROFILE_",
    "CC_",
    "CXX_",
    "AR_",
)
SANITIZED_ENV_NAMES = {
    "AR",
    "CC",
    "CFLAGS",
    "CPPFLAGS",
    "CXX",
    "CXXFLAGS",
    "HOST",
    "LDFLAGS",
    "MACOSX_DEPLOYMENT_TARGET",
    "RUSTC",
    "RUSTDOC",
    "RUSTDOCFLAGS",
    "RUSTFLAGS",
    "RUSTC_WRAPPER",
    "RUSTC_WORKSPACE_WRAPPER",
    "CARGO_BUILD_TARGET",
    "CARGO_ENCODED_RUSTFLAGS",
    "CARGO_INCREMENTAL",
    "RUSTUP_TOOLCHAIN",
    "SDKROOT",
    "TARGET",
    "AB_BASELINE_MODEL",
    "AB_BASELINE_MANIFEST",
    "AB_BASELINE_GOLDEN",
    "AB_CANDIDATE_MANIFEST",
    "AB_CANDIDATE_GOLDEN",
    "DATA_ONLY",
    "LEAGUE_CONFIG",
    "LEAGUE_ROUND",
    "OFFLINE_CONFIG",
    "STATUS",
    "SYMMETRY_AUGMENT",
    "WORKDIR",
    *PIPELINE_ENV.values(),
}


def _validate_pipeline_settings(settings, context):
    positive_integers = {
        "num_shards",
        "games_per_shard",
        "epochs",
        "ab_budget_ms",
        "parallelism",
    }
    nonnegative_integers = {"base_seed", "ab_pairs"}
    nonnegative_numbers = {
        "policy_weight",
        "value_weight",
        "q_weight",
        "auxiliary_weight",
        "min_shard_completion_rate",
        "min_decisions_per_game",
        "min_q_row_fraction",
    }
    for key in positive_integers:
        if key in settings and (
            not isinstance(settings[key], int)
            or isinstance(settings[key], bool)
            or settings[key] <= 0
        ):
            raise SystemExit(f"{context}.{key} must be a positive integer")
    for key in nonnegative_integers:
        if key in settings and (
            not isinstance(settings[key], int)
            or isinstance(settings[key], bool)
            or settings[key] < 0
        ):
            raise SystemExit(f"{context}.{key} must be a nonnegative integer")
    for key in nonnegative_numbers:
        if key in settings and (
            not isinstance(settings[key], (int, float))
            or isinstance(settings[key], bool)
            or settings[key] < 0
        ):
            raise SystemExit(f"{context}.{key} must be nonnegative")
    if settings.get("policy_target", "teacher") not in {"teacher", "behaviour"}:
        raise SystemExit(f"{context}.policy_target must be teacher or behaviour")
    if settings.get("early_stop_metric", "policy") not in {"policy", "value", "q"}:
        raise SystemExit(f"{context}.early_stop_metric must be policy, value, or q")
    if "run_ab" in settings and settings["run_ab"] not in (0, 1, False, True):
        raise SystemExit(f"{context}.run_ab must be 0 or 1")
    if "ab_min_level_delta" in settings and (
        not isinstance(settings["ab_min_level_delta"], (int, float))
        or isinstance(settings["ab_min_level_delta"], bool)
    ):
        raise SystemExit(f"{context}.ab_min_level_delta must be numeric")
    for key in ("cargo", "python", "venv"):
        if key in settings and not isinstance(settings[key], (str, int)):
            raise SystemExit(f"{context}.{key} must be a string/integer scalar")
    if "ab_seed" in settings:
        _parse_u64(settings["ab_seed"], f"{context}.ab_seed")
    for key in ("min_shard_completion_rate", "min_q_row_fraction"):
        if key in settings and not 0 <= float(settings[key]) <= 1:
            raise SystemExit(f"{context}.{key} must be in [0,1]")


def _parse_u64(value: str | int, context: str) -> int:
    if isinstance(value, bool):
        raise SystemExit(f"{context} must be an unsigned 64-bit integer")
    text = str(value).strip()
    base = 16 if text.lower().startswith("0x") else 10
    try:
        parsed = int(text, base)
    except ValueError as error:
        raise SystemExit(
            f"{context} must be decimal or 0x-prefixed hexadecimal"
        ) from error
    if not 0 <= parsed <= (1 << 64) - 1:
        raise SystemExit(f"{context} is outside unsigned 64-bit range")
    return parsed


def _sanitized_environment() -> tuple[dict[str, str], list[str]]:
    environment = os.environ.copy()
    removed = []
    for name in sorted(environment):
        if name in SANITIZED_ENV_NAMES or name.startswith(SANITIZED_ENV_PREFIXES):
            removed.append(name)
            environment.pop(name)
    return environment, removed


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: str | os.PathLike[str]) -> dict:
    try:
        with open(path, encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read expert-iteration config {path}: {error}") from error
    if not isinstance(config, dict) or config.get("manifest_version") != MANIFEST_VERSION:
        raise SystemExit("expert-iteration config manifest_version must be 1")
    if config.get("research_only") is not True:
        raise SystemExit("expert-iteration config must explicitly set research_only=true")
    allowed_top = {"manifest_version", "research_only", "defaults", "rounds", "offline_config"}
    unknown_top = sorted(set(config) - allowed_top)
    if unknown_top:
        raise SystemExit(f"unknown top-level config keys: {', '.join(unknown_top)}")
    rounds = config.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise SystemExit("expert-iteration config needs at least one round")
    names = set()
    for round_index, round_config in enumerate(rounds):
        context = f"rounds[{round_index}]"
        if not isinstance(round_config, dict):
            raise SystemExit(f"{context} must be an object")
        allowed_round = {
            "name",
            "league",
            "symmetry_augmentation",
            "offline_config",
            *PIPELINE_ENV,
        }
        unknown_round = sorted(set(round_config) - allowed_round)
        if unknown_round:
            raise SystemExit(f"{context}: unknown keys: {', '.join(unknown_round)}")
        name = round_config.get("name")
        if not isinstance(name, str) or not SAFE_NAME.fullmatch(name):
            raise SystemExit(f"{context}.name is missing or unsafe")
        if name in names:
            raise SystemExit(f"duplicate round name {name!r}")
        names.add(name)
        league = round_config.get("league")
        if not isinstance(league, list) or not league:
            raise SystemExit(f"{context}.league must be a non-empty array")
        profile_names = set()
        for profile_index, profile in enumerate(league):
            profile_context = f"{context}.league[{profile_index}]"
            if not isinstance(profile, dict):
                raise SystemExit(f"{profile_context} must be an object")
            allowed_profile = {
                "name",
                "slots",
                "behaviour",
                "behaviour_budget_ms",
                "teacher_budget_ms",
                "q_candidates",
                "q_rollout_behaviour",
                "q_rollout_budget_ms",
                "mix_search_fraction",
                "seat_behaviours",
            }
            unknown_profile = sorted(set(profile) - allowed_profile)
            if unknown_profile:
                raise SystemExit(
                    f"{profile_context}: unknown keys: {', '.join(unknown_profile)}"
                )
            profile_name = profile.get("name")
            if not isinstance(profile_name, str) or not SAFE_NAME.fullmatch(profile_name):
                raise SystemExit(f"{profile_context}.name is missing or unsafe")
            if profile_name in profile_names:
                raise SystemExit(f"{context}: duplicate profile name {profile_name!r}")
            profile_names.add(profile_name)
            slots = profile.get("slots")
            if not isinstance(slots, int) or isinstance(slots, bool) or slots <= 0:
                raise SystemExit(f"{profile_context}.slots must be a positive integer")
            behaviour = profile.get("behaviour")
            if behaviour not in BEHAVIOURS:
                raise SystemExit(f"{profile_context}.behaviour must be one of {sorted(BEHAVIOURS)}")
            rollout = profile.get("q_rollout_behaviour")
            if rollout not in ROLLOUTS:
                raise SystemExit(
                    f"{profile_context}.q_rollout_behaviour must be one of {sorted(ROLLOUTS)}"
                )
            for field in (
                "teacher_budget_ms",
                "behaviour_budget_ms",
                "q_rollout_budget_ms",
            ):
                value = profile.get(field)
                if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                    raise SystemExit(f"{profile_context}.{field} must be a positive integer")
            q_candidates = profile.get("q_candidates")
            if q_candidates != "all" and (
                not isinstance(q_candidates, int)
                or isinstance(q_candidates, bool)
                or q_candidates < 0
            ):
                raise SystemExit(f"{profile_context}.q_candidates must be >=0 or 'all'")
            mix_fraction = profile.get("mix_search_fraction", 0.5)
            if not isinstance(mix_fraction, (int, float)) or not 0 <= mix_fraction <= 1:
                raise SystemExit(f"{profile_context}.mix_search_fraction must be in [0,1]")
            seat_behaviours = profile.get("seat_behaviours")
            if seat_behaviours is not None and (
                not isinstance(seat_behaviours, list)
                or len(seat_behaviours) != 4
                or any(value not in BEHAVIOURS for value in seat_behaviours)
            ):
                raise SystemExit(
                    f"{profile_context}.seat_behaviours must be exactly four values from "
                    f"{sorted(BEHAVIOURS)}"
                )
        augmentation = round_config.get("symmetry_augmentation", "identity")
        if augmentation not in {"identity", "seat", "suit-cyclic", "seat-suit-cyclic"}:
            raise SystemExit(f"{context}.symmetry_augmentation is unsupported")
        if "offline_config" in round_config and not isinstance(
            round_config["offline_config"], str
        ):
            raise SystemExit(f"{context}.offline_config must be a path string")
    defaults = config.get("defaults", {})
    if not isinstance(defaults, dict):
        raise SystemExit("defaults must be an object")
    unknown = sorted(set(defaults) - set(PIPELINE_ENV))
    if unknown:
        raise SystemExit(f"unknown defaults keys: {', '.join(unknown)}")
    _validate_pipeline_settings(defaults, "defaults")
    for index, round_config in enumerate(rounds):
        _validate_pipeline_settings(round_config, f"rounds[{index}]")
    offline_config = config.get("offline_config")
    if offline_config is not None and not isinstance(offline_config, str):
        raise SystemExit("offline_config must be a path string")
    return config


def profile_for_shard(config: dict, round_index: int, shard_index: int) -> dict:
    try:
        league = config["rounds"][round_index]["league"]
    except IndexError as error:
        raise SystemExit(f"round index {round_index} is outside the configured rounds") from error
    if shard_index < 0:
        raise SystemExit("shard index cannot be negative")
    slot = shard_index % sum(profile["slots"] for profile in league)
    for profile in league:
        if slot < profile["slots"]:
            return profile
        slot -= profile["slots"]
    raise AssertionError("validated league schedule is non-empty")


def profile_tsv(config: dict, round_index: int, shard_index: int) -> str:
    profile = profile_for_shard(config, round_index, shard_index)
    values = [
        profile["name"],
        profile["behaviour"],
        str(profile["behaviour_budget_ms"]),
        str(profile["teacher_budget_ms"]),
        str(profile["q_candidates"]),
        profile["q_rollout_behaviour"],
        str(profile["q_rollout_budget_ms"]),
        ",".join(profile["seat_behaviours"])
        if profile.get("seat_behaviours") is not None
        else "-",
        str(profile.get("mix_search_fraction", 0.5)),
    ]
    if any("\t" in value or "\n" in value for value in values):
        raise SystemExit("profile values cannot contain tabs/newlines")
    return "\t".join(values)


def plan(config: dict, num_shards_override: int | None = None) -> dict:
    defaults = config.get("defaults", {})
    rounds = []
    for index, round_config in enumerate(config["rounds"]):
        num_shards = int(
            num_shards_override
            if num_shards_override is not None
            else round_config.get("num_shards", defaults.get("num_shards", 8))
        )
        if num_shards <= 0:
            raise SystemExit(f"round {round_config['name']}: num_shards must be positive")
        schedule = [profile_for_shard(config, index, shard)["name"] for shard in range(num_shards)]
        rounds.append(
            {
                "index": index,
                "name": round_config["name"],
                "num_shards": num_shards,
                "shard_schedule": schedule,
                "symmetry_augmentation": round_config.get(
                    "symmetry_augmentation", "identity"
                ),
                "prior": "embedded" if index == 0 else config["rounds"][index - 1]["name"],
            }
        )
    return {"manifest_version": 1, "research_only": True, "rounds": rounds}


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _resolve_optional(config_path: Path, value: str | None) -> str:
    if not value:
        return ""
    path = Path(value)
    if not path.is_absolute():
        path = config_path.parent / path
    return str(path.resolve())


def _effective_settings(config: dict, round_config: dict, round_index: int) -> dict:
    settings = dict(config.get("defaults", {}))
    for key in PIPELINE_ENV:
        if key in round_config:
            settings[key] = round_config[key]
    if "base_seed" not in round_config:
        settings["base_seed"] = int(settings.get("base_seed", 1000)) + round_index * 1_000_003
    if "ab_seed" not in round_config:
        base = _parse_u64(settings.get("ab_seed", "0x5EED"), "defaults.ab_seed")
        # A large odd Weyl increment gives each round a deterministic, distinct
        # matched-deal stream while keeping round zero backward-compatible.
        settings["ab_seed"] = (base + round_index * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    return settings


def _pipeline_env_value(key: str, value: object) -> str:
    if key == "run_ab":
        if value not in (True, False, 0, 1):
            raise SystemExit("run_ab must be a boolean or 0/1")
        return "1" if bool(value) else "0"
    return str(value)


def run_experiment(config_path_value: str, workdir_value: str) -> None:
    config_path = Path(config_path_value).resolve()
    config = load_config(config_path)
    repo = Path(__file__).resolve().parent.parent
    workdir = Path(workdir_value).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    base_environment, sanitized_names = _sanitized_environment()
    experiment_lock = workdir / "experiment.json"
    lock = {
        "manifest_version": 1,
        "research_only": True,
        "automatic_production_promotion_allowed": False,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "runner_sha256": sha256_file(__file__),
        "plan": plan(config),
        "environment_policy": {
            "mode": "sanitize-unconfigured-runtime-and-build-flags",
            "prefixes": list(SANITIZED_ENV_PREFIXES),
            "names": sorted(SANITIZED_ENV_NAMES),
            "removed_names": sanitized_names,
        },
    }
    if experiment_lock.exists():
        with open(experiment_lock, encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != lock:
            raise SystemExit(
                f"{workdir}: experiment fingerprint changed; use a fresh workdir or restore inputs"
            )
    else:
        _atomic_json(experiment_lock, lock)

    previous_model: Path | None = None
    for round_index, round_config in enumerate(config["rounds"]):
        round_dir = workdir / f"round-{round_index:02d}-{round_config['name']}"
        settings = _effective_settings(config, round_config, round_index)
        environment = base_environment.copy()
        environment["WORKDIR"] = str(round_dir)
        environment["LEAGUE_CONFIG"] = str(config_path)
        environment["LEAGUE_ROUND"] = str(round_index)
        environment["SYMMETRY_AUGMENT"] = round_config.get(
            "symmetry_augmentation", "identity"
        )
        offline_value = round_config.get("offline_config", config.get("offline_config"))
        environment["OFFLINE_CONFIG"] = _resolve_optional(config_path, offline_value)
        for key, value in settings.items():
            if key in PIPELINE_ENV:
                environment[PIPELINE_ENV[key]] = _pipeline_env_value(key, value)
        if previous_model is not None:
            previous_manifest = Path(f"{previous_model}.manifest.json")
            previous_golden = Path(f"{previous_model}.golden.json")
            environment["SHENGJI_EXPERT_MODEL_PATH"] = str(previous_model)
            environment["SHENGJI_EXPERT_MODEL_MANIFEST"] = str(previous_manifest)
            environment["AB_BASELINE_MODEL"] = str(previous_model)
            environment["AB_BASELINE_MANIFEST"] = str(previous_manifest)
            environment["AB_BASELINE_GOLDEN"] = str(previous_golden)
        else:
            for name in (
                "SHENGJI_EXPERT_MODEL_PATH",
                "SHENGJI_EXPERT_MODEL_MANIFEST",
                "AB_BASELINE_MODEL",
                "AB_BASELINE_MANIFEST",
                "AB_BASELINE_GOLDEN",
            ):
                environment.pop(name, None)

        print(
            f"=== expert iteration {round_index + 1}/{len(config['rounds'])}: "
            f"{round_config['name']} prior={'embedded' if previous_model is None else previous_model} ===",
            flush=True,
        )
        subprocess.run(
            ["bash", str(repo / "training" / "run_value_pipeline.sh")],
            cwd=repo,
            env=environment,
            check=True,
        )
        model = round_dir / "action_value.onnx"
        required = [
            model,
            Path(f"{model}.manifest.json"),
            Path(f"{model}.golden.json"),
            round_dir / "data_full.csv",
            round_dir / "data_full.csv.manifest.json",
            round_dir / "config.json",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise SystemExit(f"round {round_config['name']} is incomplete: {missing}")
        comparison = round_dir / "model-ab" / "comparison.json"
        if environment.get("RUN_AB", "1") == "1" and int(
            environment.get("AB_PAIRS", "200")
        ) > 0:
            if not comparison.is_file():
                raise SystemExit(f"round {round_config['name']}: promotion-gate result is missing")
            with open(comparison, encoding="utf-8") as handle:
                comparison_result = json.load(handle)
            gate = comparison_result.get("promotion_gate")
            if not isinstance(gate, dict) or gate.get("passed") is not True:
                raise SystemExit(f"round {round_config['name']}: promotion gate did not pass")
            interval = comparison_result.get("level_utility", {}).get(
                "paired_bootstrap95"
            )
            if (
                gate.get("metric")
                != "candidate_minus_baseline_level_utility_bootstrap95_lower"
                or not isinstance(interval, list)
                or len(interval) != 2
                or gate.get("observed_lower95") != interval[0]
            ):
                raise SystemExit(
                    f"round {round_config['name']}: promotion gate confidence contract is invalid"
                )
            if gate.get("minimum_level_delta") != float(
                environment.get("AB_MIN_LEVEL_DELTA", "-0.05")
            ):
                raise SystemExit(
                    f"round {round_config['name']}: promotion gate threshold does not match config"
                )
        round_manifest = {
            "manifest_version": 1,
            "research_only": True,
            "automatic_production_promotion_allowed": False,
            "round_index": round_index,
            "round_name": round_config["name"],
            "config_sha256": sha256_file(config_path),
            "pipeline_config_sha256": sha256_file(round_dir / "config.json"),
            "dataset_sha256": sha256_file(round_dir / "data_full.csv"),
            "dataset_manifest_sha256": sha256_file(
                round_dir / "data_full.csv.manifest.json"
            ),
            "model_sha256": sha256_file(model),
            "model_manifest_sha256": sha256_file(f"{model}.manifest.json"),
            "golden_sha256": sha256_file(f"{model}.golden.json"),
            "prior_model_sha256": sha256_file(previous_model) if previous_model else None,
            "promotion_gate_sha256": sha256_file(comparison) if comparison.is_file() else None,
            "league_schedule": plan(config)["rounds"][round_index]["shard_schedule"],
        }
        _atomic_json(round_dir / "round.manifest.json", round_manifest)
        _atomic_json(
            workdir / "latest.json",
            {
                "manifest_version": 1,
                "round_index": round_index,
                "round_name": round_config["name"],
                "model": str(model),
                "model_sha256": round_manifest["model_sha256"],
                "research_only": True,
            },
        )
        previous_model = model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile_parser = subparsers.add_parser(
        "profile-tsv", help="internal: resolve one deterministic shard profile"
    )
    profile_parser.add_argument("--config", required=True)
    profile_parser.add_argument("--round", type=int, required=True)
    profile_parser.add_argument("--shard", type=int, required=True)

    plan_parser = subparsers.add_parser("plan", help="validate and print the bounded schedule")
    plan_parser.add_argument("--config", required=True)
    plan_parser.add_argument("--num-shards", type=int)

    run_parser = subparsers.add_parser("run", help="execute/resume every configured round")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--workdir", required=True)

    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "profile-tsv":
        print(profile_tsv(config, args.round, args.shard))
    elif args.command == "plan":
        print(json.dumps(plan(config, args.num_shards), indent=2, sort_keys=True))
    elif args.command == "run":
        run_experiment(args.config, args.workdir)


if __name__ == "__main__":
    main()
