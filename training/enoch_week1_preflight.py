#!/usr/bin/env python3
"""Fail-closed W1.1 frozen-policy equivalence and searchless baseline runner."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence

try:
    from training import enoch_week1
    from training import enoch_week1_freeze
    from training import enoch_week1_runner
except ImportError:  # Direct ``python training/enoch_week1_preflight.py`` use.
    import enoch_week1  # type: ignore[no-redef]
    import enoch_week1_freeze  # type: ignore[no-redef]
    import enoch_week1_runner  # type: ignore[no-redef]


MANIFEST_VERSION = 1
PROBE_KIND = "enoch-control-probe"
PREFLIGHT_KIND = "enoch-week1-preflight"
STABLE_OPPONENT = "legacy-greedy/easy-phases-v1"
POLICIES = ("enoch-tier", "enoch-greedy")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

EQUIVALENCE_NAMESPACE = "preflight/frozen-policy-equivalence"
OUTCOME_BASELINE_NAMESPACE = "baseline/searchless-outcome"
STYLE_BASELINE_NAMESPACE = "baseline/style"
EQUIVALENCE_CONSUMER = "w1.1:frozen-policy-equivalence"
OUTCOME_BASELINE_CONSUMER = "w1.1:searchless-outcome"
STYLE_BASELINE_CONSUMER = "w1.1:style-baseline"
DEFAULT_PROBE_TIMEOUT_SECONDS = 6 * 60 * 60
CONTROL_REFERENCE_PROBE_ID = "binary/enoch-control-probe-reference"
CONTROL_CURRENT_PROBE_ID = "binary/enoch-control-probe-current"
CONTROL_EVALUATOR_ID = "binary/week1-evaluator"
CONTROL_SOURCE_FILE_LIST_ID = "source/week1-evaluator-file-list"
DETERMINISTIC_SEARCH_TESTS = (
    "bot::search::tests::root_candidate_order_is_canonical_and_deduplicated",
    "bot::search::tests::strict_fixed_work_search_repeats_cards_and_work_telemetry",
    "bot::search::tests::strict_telemetry_proves_progressive_and_control_have_equal_fixed_work",
    "bot::search::tests::strict_search_rejects_a_zero_sample_prior_fallback",
)
DETERMINISTIC_SEARCH_AUTHORITY_KIND = (
    "enoch-week1-deterministic-search-fixture-authority"
)
BASELINE_WORKER_REPORT_KIND = "enoch-week1-baseline-and-worker-report"


class PreflightError(RuntimeError):
    """Raised when a probe or equivalence contract fails closed."""


@dataclass(frozen=True)
class ProbeRun:
    document: dict[str, Any]
    stdout: bytes
    stdout_sha256: str


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while block := source.read(1024 * 1024):
                digest.update(block)
    except OSError as exc:
        raise PreflightError(f"could not hash probe binary {path}: {exc}") from exc
    return digest.hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise PreflightError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PreflightError(f"{label} must be an object")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PreflightError(f"{label} must be a nonnegative integer")
    return value


def _signed_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PreflightError(f"{label} must be an integer")
    return value


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise PreflightError(f"{label} must be a boolean")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise PreflightError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validate_seeds(seeds: Sequence[int], label: str) -> list[int]:
    if not seeds:
        raise PreflightError(f"{label} must contain at least one seed")
    checked: list[int] = []
    seen: set[int] = set()
    for index, seed in enumerate(seeds):
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 1 << 64:
            raise PreflightError(f"{label}[{index}] is not an unsigned 64-bit integer")
        if seed in seen:
            raise PreflightError(f"{label} contains duplicate seed {seed}")
        seen.add(seed)
        checked.append(seed)
    return checked


def _validate_style(value: Any, label: str) -> int:
    style = _mapping(value, label)
    count_keys = {
        "cards_played",
        "decisions",
        "follow_decisions",
        "joker_cards_played",
        "lead_decisions",
        "multi_card_plays",
        "point_cards_played",
        "point_value_played",
        "single_card_plays",
        "trump_cards_played",
    }
    _exact_keys(style, count_keys | {"decision_sha256"}, label)
    counts = {
        name: _nonnegative_int(style[name], f"{label}.{name}") for name in count_keys
    }
    _sha256(style["decision_sha256"], f"{label}.decision_sha256")
    if counts["lead_decisions"] + counts["follow_decisions"] != counts["decisions"]:
        raise PreflightError(f"{label} lead/follow counts do not sum to decisions")
    if counts["single_card_plays"] + counts["multi_card_plays"] != counts["decisions"]:
        raise PreflightError(f"{label} single/multi counts do not sum to decisions")
    if counts["point_cards_played"] > counts["cards_played"]:
        raise PreflightError(f"{label} point-card count exceeds cards played")
    if counts["trump_cards_played"] > counts["cards_played"]:
        raise PreflightError(f"{label} trump-card count exceeds cards played")
    if counts["joker_cards_played"] > counts["cards_played"]:
        raise PreflightError(f"{label} joker-card count exceeds cards played")
    return counts["decisions"]


def _validate_orientation(value: Any, focal_is_landlord: bool, label: str) -> dict[str, Any]:
    orientation = _mapping(value, label)
    _exact_keys(
        orientation,
        {
            "complete",
            "focal_is_landlord_team",
            "focal_level_utility",
            "focal_point_margin",
            "focal_won",
            "landlord_level_delta",
            "landlord_won",
            "non_landlord_level_delta",
            "non_landlord_points",
            "style",
        },
        label,
    )
    complete = _boolean(orientation["complete"], f"{label}.complete")
    if orientation["focal_is_landlord_team"] is not focal_is_landlord:
        raise PreflightError(f"{label} has the wrong focal orientation")
    decisions = _validate_style(orientation["style"], f"{label}.style")
    outcome_names = (
        "focal_level_utility",
        "focal_point_margin",
        "landlord_level_delta",
        "non_landlord_level_delta",
        "non_landlord_points",
    )
    if complete:
        for name in outcome_names:
            _signed_int(orientation[name], f"{label}.{name}")
        _boolean(orientation["focal_won"], f"{label}.focal_won")
        _boolean(orientation["landlord_won"], f"{label}.landlord_won")
    else:
        for name in (*outcome_names, "focal_won", "landlord_won"):
            if orientation[name] is not None:
                raise PreflightError(f"{label}.{name} must be null when incomplete")
    return {"complete": complete, "decisions": decisions, **orientation}


def validate_probe_document(
    value: Any,
    *,
    expected_policy: str,
    expected_seeds: Sequence[int],
) -> dict[str, Any]:
    """Validate the complete probe schema, seed order, summaries, and digest."""

    if expected_policy not in POLICIES:
        raise PreflightError(f"unsupported expected policy {expected_policy!r}")
    expected_seeds = _validate_seeds(expected_seeds, "expected seeds")
    document = _mapping(value, "probe output")
    _exact_keys(document, {"equivalence_sha256", "frozen_policy"}, "probe output")
    frozen = _mapping(document["frozen_policy"], "frozen_policy")
    _exact_keys(
        frozen,
        {
            "kind",
            "manifest_version",
            "opponent",
            "pairs",
            "policy",
            "seed_count",
            "seeds",
            "summary",
        },
        "frozen_policy",
    )
    if (
        frozen["kind"] != PROBE_KIND
        or _nonnegative_int(frozen["manifest_version"], "frozen_policy.manifest_version")
        != MANIFEST_VERSION
    ):
        raise PreflightError("probe kind or manifest version is unsupported")
    if frozen["opponent"] != STABLE_OPPONENT:
        raise PreflightError("probe did not use the explicit stable opponent")
    if frozen["policy"] != expected_policy:
        raise PreflightError(
            f"probe policy mismatch: expected {expected_policy!r}, got {frozen['policy']!r}"
        )
    if not isinstance(frozen["seeds"], list):
        raise PreflightError("probe seeds must be an array")
    reported_seeds = _validate_seeds(frozen["seeds"], "probe seeds")
    if reported_seeds != expected_seeds:
        raise PreflightError(
            f"probe seed mismatch: expected {expected_seeds!r}, got {reported_seeds!r}"
        )
    if (
        _nonnegative_int(frozen["seed_count"], "frozen_policy.seed_count")
        != len(expected_seeds)
    ):
        raise PreflightError("probe seed_count does not match the exact seed list")
    pairs = frozen["pairs"]
    if not isinstance(pairs, list) or len(pairs) != len(expected_seeds):
        raise PreflightError("probe pairs do not match the exact seed list")

    completed_hands = 0
    focal_decisions = 0
    focal_wins = 0
    for index, (pair_value, seed) in enumerate(zip(pairs, expected_seeds)):
        label = f"frozen_policy.pairs[{index}]"
        pair = _mapping(pair_value, label)
        _exact_keys(
            pair,
            {
                "complete",
                "focal_as_attacker",
                "focal_as_landlord",
                "focal_level_utility_sum",
                "focal_point_margin_sum",
                "focal_wins",
                "request_index",
                "seed",
            },
            label,
        )
        request_index = _nonnegative_int(pair["request_index"], f"{label}.request_index")
        pair_seed = _nonnegative_int(pair["seed"], f"{label}.seed")
        if pair_seed >= 1 << 64:
            raise PreflightError(f"{label}.seed is outside unsigned 64-bit range")
        if request_index != index or pair_seed != seed:
            raise PreflightError(f"{label} index/seed does not match the request")
        complete = _boolean(pair["complete"], f"{label}.complete")
        landlord = _validate_orientation(
            pair["focal_as_landlord"], True, f"{label}.focal_as_landlord"
        )
        attacker = _validate_orientation(
            pair["focal_as_attacker"], False, f"{label}.focal_as_attacker"
        )
        actual_complete = landlord["complete"] and attacker["complete"]
        if complete != actual_complete:
            raise PreflightError(f"{label}.complete disagrees with its orientations")
        completed_hands += int(landlord["complete"]) + int(attacker["complete"])
        focal_decisions += landlord["decisions"] + attacker["decisions"]
        if not complete:
            raise PreflightError(f"{label} is incomplete")
        expected_wins = int(landlord["focal_won"]) + int(attacker["focal_won"])
        expected_margin = landlord["focal_point_margin"] + attacker["focal_point_margin"]
        expected_levels = landlord["focal_level_utility"] + attacker["focal_level_utility"]
        if _nonnegative_int(pair["focal_wins"], f"{label}.focal_wins") != expected_wins:
            raise PreflightError(f"{label}.focal_wins is inconsistent")
        if (
            _signed_int(pair["focal_point_margin_sum"], f"{label}.focal_point_margin_sum")
            != expected_margin
        ):
            raise PreflightError(f"{label}.focal_point_margin_sum is inconsistent")
        if (
            _signed_int(pair["focal_level_utility_sum"], f"{label}.focal_level_utility_sum")
            != expected_levels
        ):
            raise PreflightError(f"{label}.focal_level_utility_sum is inconsistent")
        focal_wins += expected_wins

    summary = _mapping(frozen["summary"], "frozen_policy.summary")
    _exact_keys(
        summary,
        {
            "complete_pairs",
            "completed_hands",
            "focal_decisions",
            "focal_wins",
            "incomplete_pairs",
            "pairs_requested",
        },
        "frozen_policy.summary",
    )
    expected_summary = {
        "complete_pairs": len(expected_seeds),
        "completed_hands": completed_hands,
        "focal_decisions": focal_decisions,
        "focal_wins": focal_wins,
        "incomplete_pairs": 0,
        "pairs_requested": len(expected_seeds),
    }
    for name in expected_summary:
        _nonnegative_int(summary[name], f"frozen_policy.summary.{name}")
    if dict(summary) != expected_summary:
        raise PreflightError(
            f"probe summary mismatch: expected {expected_summary!r}, got {dict(summary)!r}"
        )
    expected_digest = enoch_week1.canonical_json_sha256(frozen)
    actual_digest = _sha256(document["equivalence_sha256"], "equivalence_sha256")
    if actual_digest != expected_digest:
        raise PreflightError("probe equivalence digest does not match frozen_policy")
    return dict(document)


def _safe_environment(
    environment: Mapping[str, str] | None = None,
    *,
    allowlist: Sequence[str] = enoch_week1.DEFAULT_EVALUATOR_ENV_ALLOWLIST,
) -> tuple[dict[str, str], tuple[str, ...]]:
    cleaned, removed = enoch_week1.sanitized_evaluator_environment(
        environment, allowlist=allowlist
    )
    if not cleaned.get("PATH"):
        raise PreflightError("PATH is required to execute probe binaries")
    cleaned.update({"LANG": "C", "LC_ALL": "C", "RUST_BACKTRACE": "1"})
    return cleaned, removed


def run_probe(
    binary: Path,
    policy: str,
    seeds: Sequence[int],
    *,
    environment: Mapping[str, str],
    timeout_seconds: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
) -> ProbeRun:
    seeds = _validate_seeds(seeds, "probe seeds")
    if policy not in POLICIES:
        raise PreflightError(f"unsupported probe policy {policy!r}")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or timeout_seconds <= 0
    ):
        raise PreflightError("probe timeout_seconds must be positive")
    command = [str(binary), "--policy", policy]
    for seed in seeds:
        command.extend(("--seed", str(seed)))
    try:
        completed = subprocess.run(
            command,
            env=dict(environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=float(timeout_seconds),
        )
    except subprocess.TimeoutExpired as exc:
        raise PreflightError(
            f"probe timed out after {timeout_seconds:g}s: {binary}"
        ) from exc
    except OSError as exc:
        raise PreflightError(f"could not execute probe binary {binary}: {exc}") from exc
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace")
        raise PreflightError(
            f"probe exited nonzero ({completed.returncode}): {binary}\n{stderr}"
        )
    try:
        decoded = completed.stdout.decode("utf-8")
        value = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"probe emitted invalid JSON: {binary}: {exc}") from exc
    document = validate_probe_document(
        value, expected_policy=policy, expected_seeds=seeds
    )
    canonical = enoch_week1.canonical_json_bytes(document) + b"\n"
    if completed.stdout != canonical:
        raise PreflightError(f"probe output is not canonical JSON: {binary}")
    return ProbeRun(
        document=document,
        stdout=completed.stdout,
        stdout_sha256=_sha256_bytes(completed.stdout),
    )


def _baseline_summary(document: Mapping[str, Any]) -> dict[str, Any]:
    frozen = document["frozen_policy"]
    style_names = (
        "cards_played",
        "decisions",
        "follow_decisions",
        "joker_cards_played",
        "lead_decisions",
        "multi_card_plays",
        "point_cards_played",
        "point_value_played",
        "single_card_plays",
        "trump_cards_played",
    )
    style_totals = {name: 0 for name in style_names}
    decision_digests: list[str] = []
    point_margin_sum = 0
    level_utility_sum = 0
    for pair in frozen["pairs"]:
        point_margin_sum += pair["focal_point_margin_sum"]
        level_utility_sum += pair["focal_level_utility_sum"]
        for orientation_name in ("focal_as_landlord", "focal_as_attacker"):
            style = pair[orientation_name]["style"]
            for name in style_names:
                style_totals[name] += style[name]
            decision_digests.append(style["decision_sha256"])
    return {
        "complete_pairs": frozen["summary"]["complete_pairs"],
        "decision_sequence_sha256": enoch_week1.canonical_json_sha256(decision_digests),
        "focal_level_utility_sum": level_utility_sum,
        "focal_point_margin_sum": point_margin_sum,
        "focal_wins": frozen["summary"]["focal_wins"],
        "style_totals": style_totals,
    }


def _load_json_array(path: Path, label: str) -> list[Any]:
    try:
        with path.open("r", encoding="utf-8") as source:
            value = json.load(source)
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"could not load {label} from {path}: {exc}") from exc
    if not isinstance(value, list):
        raise PreflightError(f"{label} must be a JSON array")
    return value


def _verify_control_bundle(
    protocol: Mapping[str, Any], control_bundle: Path
) -> dict[str, Any]:
    """Verify the complete W1.0 bundle and return its exact W1.1 identities."""

    control_bundle = control_bundle.resolve()
    manifest = enoch_week1.load_json_object(control_bundle / "control-manifest.json")
    control_fingerprint = enoch_week1.validate_w1_0_control_manifest(protocol, manifest)
    artifact_hashes = manifest["artifact_hashes"]
    required = {
        CONTROL_REFERENCE_PROBE_ID,
        CONTROL_CURRENT_PROBE_ID,
        CONTROL_EVALUATOR_ID,
        CONTROL_SOURCE_FILE_LIST_ID,
        "binary/enoch-0",
        "binary/expert-0",
        "binary/grandmaster-0",
        "model/expert_model.onnx",
        "source/production-reference",
    }
    missing = required - set(artifact_hashes)
    if missing:
        raise PreflightError(
            f"W1.0 control bundle lacks required artifacts: {sorted(missing)}"
        )
    artifact_paths: dict[str, Path] = {}
    for artifact_id, expected_hash in artifact_hashes.items():
        try:
            path = enoch_week1_freeze._bundle_artifact_path(  # noqa: SLF001
                control_bundle, artifact_id
            )
        except enoch_week1_freeze.FreezeError as exc:
            raise PreflightError(str(exc)) from exc
        if not path.is_file():
            raise PreflightError(f"W1.0 control artifact is missing: {artifact_id}")
        actual_hash = _sha256_file(path)
        if actual_hash != expected_hash:
            raise PreflightError(f"W1.0 control artifact hash mismatch: {artifact_id}")
        artifact_paths[artifact_id] = path.resolve()

    policies = manifest["policy_identities"]
    for policy_name in ("enoch-0", "expert-0", "grandmaster-0"):
        identity = policies[policy_name]
        if identity["binary_sha256"] != artifact_hashes[f"binary/{policy_name}"]:
            raise PreflightError(
                f"W1.0 {policy_name} binary identity is not its bundled artifact"
            )
        if identity["source_sha256"] != artifact_hashes["source/production-reference"]:
            raise PreflightError(
                f"W1.0 {policy_name} source identity is not the production archive"
            )
        expected_configuration = enoch_week1.canonical_json_sha256(
            manifest["search_knobs"][policy_name]
        )
        if identity["configuration_sha256"] != expected_configuration:
            raise PreflightError(f"W1.0 {policy_name} configuration identity mismatch")
    if (
        policies["expert-0"]["model_sha256"]
        != artifact_hashes["model/expert_model.onnx"]
    ):
        raise PreflightError("W1.0 Expert model identity is not its bundled model")

    source_records = _load_json_array(
        artifact_paths[CONTROL_SOURCE_FILE_LIST_ID], "W1.0 evaluator source file list"
    )
    evaluator_identity = manifest["evaluator_identity"]
    if evaluator_identity["binary_sha256"] != artifact_hashes[CONTROL_EVALUATOR_ID]:
        raise PreflightError("W1.0 evaluator binary identity is not its bundled artifact")
    if evaluator_identity["source_sha256"] != artifact_hashes[
        CONTROL_SOURCE_FILE_LIST_ID
    ]:
        raise PreflightError("W1.0 evaluator source identity is not its bundled file list")
    expected_evaluator_configuration = enoch_week1.canonical_json_sha256(
        enoch_week1.WEEK1_EVALUATOR_CONTRACT
    )
    if evaluator_identity["configuration_sha256"] != expected_evaluator_configuration:
        raise PreflightError("W1.0 evaluator configuration identity changed")

    index = enoch_week1.load_json_object(control_bundle / "bundle-index.json")
    expected_index = {
        "artifact_hashes": artifact_hashes,
        "control_manifest_fingerprint": control_fingerprint,
        "production_reference": manifest["production_reference"],
        "protocol_fingerprint": manifest["protocol_fingerprint"],
    }
    if index != expected_index:
        raise PreflightError("W1.0 bundle index does not reconstruct from the manifest")

    identity = {
        "artifact_hashes_sha256": enoch_week1.canonical_json_sha256(artifact_hashes),
        "control_manifest_fingerprint": control_fingerprint,
        "current_probe_artifact_id": CONTROL_CURRENT_PROBE_ID,
        "current_probe_sha256": artifact_hashes[CONTROL_CURRENT_PROBE_ID],
        "effective_environment_sha256": enoch_week1.canonical_json_sha256(
            manifest["effective_environment"]
        ),
        "enoch0_fingerprint": enoch_week1.canonical_json_sha256(policies["enoch-0"]),
        "evaluator_fingerprint": enoch_week1.canonical_json_sha256(evaluator_identity),
        "production_reference": manifest["production_reference"],
        "reference_probe_artifact_id": CONTROL_REFERENCE_PROBE_ID,
        "reference_probe_sha256": artifact_hashes[CONTROL_REFERENCE_PROBE_ID],
    }
    return {
        "artifact_paths": artifact_paths,
        "identity": identity,
        "manifest": manifest,
        "source_records": source_records,
    }


def _verify_probe_files(control: Mapping[str, Any]) -> None:
    identity = control["identity"]
    paths = control["artifact_paths"]
    checks = (
        (CONTROL_REFERENCE_PROBE_ID, identity["reference_probe_sha256"]),
        (CONTROL_CURRENT_PROBE_ID, identity["current_probe_sha256"]),
    )
    for artifact_id, expected in checks:
        if _sha256_file(paths[artifact_id]) != expected:
            raise PreflightError(f"frozen probe changed during W1.1: {artifact_id}")


def _namespace_entry(
    protocol: Mapping[str, Any], namespace: str
) -> Mapping[str, Any]:
    for entry in protocol["seed_registry"]["namespaces"]:
        if entry["name"] == namespace:
            return entry
    raise PreflightError(f"frozen protocol lacks namespace {namespace!r}")


def _protocol_prefix(
    protocol: Mapping[str, Any],
    namespace: str,
    requested_count: int | None,
    label: str,
) -> tuple[list[int], list[int], dict[str, Any]]:
    entry = _namespace_entry(protocol, namespace)
    capacity = entry["count"]
    count = capacity if requested_count is None else requested_count
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= capacity:
        raise PreflightError(
            f"{label} count must be between 1 and frozen namespace capacity {capacity}"
        )
    indices = list(range(count))
    seeds = list(entry["seeds"][:count])
    coverage = {
        "namespace_capacity": capacity,
        "status": "full" if count == capacity else "partial-prefix",
    }
    return indices, seeds, coverage


def _claim_seed_prefix(
    ledger_path: Path,
    protocol: Mapping[str, Any],
    *,
    namespace: str,
    indices: Sequence[int],
    seeds: Sequence[int],
    consumer: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    updated = enoch_week1.consume_seed_batch_once(
        ledger_path,
        protocol,
        [(namespace, index, consumer) for index in indices],
    )
    sequence_end = len(updated["consumed"])
    sequence_start = sequence_end - len(indices)
    claim = {
        "consumer": consumer,
        "ledger_fingerprint_after_claim": updated["ledger_fingerprint"],
        "namespace": namespace,
        "seed_count": len(indices),
        "seed_indices": list(indices),
        "seeds_sha256": enoch_week1.canonical_json_sha256(list(seeds)),
        "sequence_end_exclusive": sequence_end,
        "sequence_start": sequence_start,
    }
    return claim, updated


def _validate_selection(
    protocol: Mapping[str, Any],
    section: Mapping[str, Any],
    *,
    namespace: str,
    label: str,
) -> tuple[list[int], list[int]]:
    if section["namespace"] != namespace:
        raise PreflightError(f"{label} uses the wrong frozen namespace")
    count = _nonnegative_int(section["seed_count"], f"{label}.seed_count")
    if count == 0:
        raise PreflightError(f"{label} seed_count must be positive")
    entry = _namespace_entry(protocol, namespace)
    if count > entry["count"]:
        raise PreflightError(f"{label} exceeds its frozen namespace")
    indices = list(range(count))
    seeds = list(entry["seeds"][:count])
    if section["seed_indices"] != indices or section["seeds"] != seeds:
        raise PreflightError(f"{label} is not the exact frozen namespace prefix")
    expected_coverage = {
        "namespace_capacity": entry["count"],
        "status": "full" if count == entry["count"] else "partial-prefix",
    }
    if section["coverage"] != expected_coverage:
        raise PreflightError(f"{label} coverage declaration is inconsistent")
    return indices, seeds


def _historical_ledger_fingerprint(
    ledger: Mapping[str, Any], sequence_end: int
) -> str:
    body = {
        "consumed": list(ledger["consumed"][:sequence_end]),
        "ledger_kind": ledger["ledger_kind"],
        "manifest_version": ledger["manifest_version"],
        "protocol_fingerprint": ledger["protocol_fingerprint"],
        "seed_registry_sha256": ledger["seed_registry_sha256"],
    }
    return enoch_week1.canonical_json_sha256(body)


def _validate_seed_claim(
    protocol: Mapping[str, Any],
    ledger: Mapping[str, Any],
    claim: Mapping[str, Any],
    *,
    namespace: str,
    indices: Sequence[int],
    seeds: Sequence[int],
    consumer: str,
    label: str,
) -> tuple[int, int]:
    _exact_keys(
        claim,
        {
            "consumer",
            "ledger_fingerprint_after_claim",
            "namespace",
            "seed_count",
            "seed_indices",
            "seeds_sha256",
            "sequence_end_exclusive",
            "sequence_start",
        },
        f"{label}.seed_claim",
    )
    if (
        claim["namespace"] != namespace
        or claim["consumer"] != consumer
        or claim["seed_count"] != len(indices)
        or claim["seed_indices"] != list(indices)
        or claim["seeds_sha256"]
        != enoch_week1.canonical_json_sha256(list(seeds))
    ):
        raise PreflightError(f"{label} ledger claim differs from its seed selection")
    fingerprint = _sha256(
        claim["ledger_fingerprint_after_claim"],
        f"{label}.seed_claim.ledger_fingerprint_after_claim",
    )
    start = _nonnegative_int(claim["sequence_start"], f"{label}.sequence_start")
    end = _nonnegative_int(
        claim["sequence_end_exclusive"], f"{label}.sequence_end_exclusive"
    )
    if end - start != len(indices) or end > len(ledger["consumed"]):
        raise PreflightError(f"{label} ledger claim sequence range is invalid")
    expected_records = [
        {
            "consumer": consumer,
            "index": index,
            "namespace": namespace,
            "seed": seed,
            "sequence": start + offset,
        }
        for offset, (index, seed) in enumerate(zip(indices, seeds))
    ]
    if ledger["consumed"][start:end] != expected_records:
        raise PreflightError(f"{label} exact seed claims are absent from the ledger")
    if fingerprint != _historical_ledger_fingerprint(ledger, end):
        raise PreflightError(f"{label} historical ledger fingerprint mismatch")
    return start, end


def _validate_baseline_section(
    protocol: Mapping[str, Any],
    ledger: Mapping[str, Any],
    section: Mapping[str, Any],
    *,
    namespace: str,
    consumer: str,
    purpose: str,
    label: str,
    expected_binary_sha256: str,
) -> tuple[int, int]:
    _exact_keys(
        section,
        {
            "coverage",
            "namespace",
            "policy",
            "probe",
            "probe_binary_sha256",
            "probe_stdout_sha256",
            "purpose",
            "seed_claim",
            "seed_count",
            "seed_indices",
            "seeds",
            "summary",
        },
        label,
    )
    if section["policy"] != "enoch-greedy" or section["purpose"] != purpose:
        raise PreflightError(f"{label} is not the frozen searchless baseline")
    indices, seeds = _validate_selection(
        protocol, section, namespace=namespace, label=label
    )
    document = validate_probe_document(
        section["probe"], expected_policy="enoch-greedy", expected_seeds=seeds
    )
    if section["probe_binary_sha256"] != expected_binary_sha256:
        raise PreflightError(f"{label} did not use the frozen W1.0 reference probe")
    stdout_sha256 = _sha256(
        section["probe_stdout_sha256"], f"{label}.probe_stdout_sha256"
    )
    if stdout_sha256 != _sha256_bytes(enoch_week1.canonical_json_bytes(document) + b"\n"):
        raise PreflightError(f"{label} probe stdout hash is not reconstructible")
    if section["summary"] != _baseline_summary(document):
        raise PreflightError(f"{label} summary does not reconstruct from raw probe records")
    return _validate_seed_claim(
        protocol,
        ledger,
        _mapping(section["seed_claim"], f"{label}.seed_claim"),
        namespace=namespace,
        indices=indices,
        seeds=seeds,
        consumer=consumer,
        label=label,
    )


def validate_preflight_artifact(
    protocol: Mapping[str, Any],
    ledger: Mapping[str, Any],
    control_bundle: Path,
    artifact: Mapping[str, Any],
    *,
    require_full_coverage: bool = False,
) -> str:
    """Validate a W1.1 artifact, including its historical single-use claims."""

    protocol_fingerprint = enoch_week1.validate_protocol(protocol)
    enoch_week1.validate_seed_ledger(protocol, ledger)
    control = _verify_control_bundle(protocol, control_bundle)
    _verify_probe_files(control)
    preflight = _mapping(artifact, "preflight artifact")
    _exact_keys(
        preflight,
        {
            "artifact_sha256",
            "authoritative_for_w1_1_completion",
            "automatic_production_promotion_allowed",
            "coverage_status",
            "environment",
            "equivalence",
            "execution_contract",
            "final_ledger_fingerprint",
            "kind",
            "manifest_version",
            "protocol_fingerprint",
            "searchless_outcome_baseline",
            "seed_registry_sha256",
            "style_baseline",
            "w1_0_identity",
        },
        "preflight artifact",
    )
    if preflight["kind"] != PREFLIGHT_KIND or preflight["manifest_version"] != MANIFEST_VERSION:
        raise PreflightError("unsupported preflight artifact")
    if preflight["automatic_production_promotion_allowed"] is not False:
        raise PreflightError("preflight artifact cannot authorize production promotion")
    if preflight["protocol_fingerprint"] != protocol_fingerprint:
        raise PreflightError("preflight artifact belongs to another protocol")
    if preflight["seed_registry_sha256"] != protocol["seed_registry_sha256"]:
        raise PreflightError("preflight artifact belongs to another seed registry")
    if preflight["w1_0_identity"] != control["identity"]:
        raise PreflightError("preflight artifact is not bound to this W1.0 control bundle")
    execution_contract = _mapping(
        preflight["execution_contract"], "preflight execution contract"
    )
    _exact_keys(
        execution_contract,
        {"binary_reverified_after_each_stage", "probe_timeout_seconds"},
        "preflight execution contract",
    )
    if execution_contract["binary_reverified_after_each_stage"] is not True:
        raise PreflightError("preflight did not reverify binaries after every stage")
    timeout_seconds = execution_contract["probe_timeout_seconds"]
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or timeout_seconds <= 0
    ):
        raise PreflightError("preflight probe timeout must be positive")

    environment = _mapping(preflight["environment"], "preflight environment")
    _exact_keys(
        environment,
        {
            "blocked_prefixes",
            "effective_environment_sha256",
            "removed_names",
            "sanitized",
        },
        "preflight environment",
    )
    if (
        environment["blocked_prefixes"]
        != list(enoch_week1.BLOCKED_EVALUATOR_ENV_PREFIXES)
        or environment["sanitized"] is not True
    ):
        raise PreflightError("preflight environment was not sanitized")
    removed = environment["removed_names"]
    if (
        not isinstance(removed, list)
        or removed != sorted(set(removed))
        or not all(isinstance(name, str) for name in removed)
    ):
        raise PreflightError("preflight removed environment names are not canonical")
    _sha256(environment["effective_environment_sha256"], "effective environment hash")

    equivalence = _mapping(preflight["equivalence"], "equivalence")
    _exact_keys(
        equivalence,
        {
            "coverage",
            "current_binary_sha256",
            "current_probe",
            "current_stdout_sha256",
            "frozen_policy_sha256",
            "namespace",
            "policy",
            "reference_binary_sha256",
            "reference_probe",
            "reference_stdout_sha256",
            "seed_claim",
            "seed_count",
            "seed_indices",
            "seeds",
        },
        "equivalence",
    )
    if equivalence["policy"] not in POLICIES:
        raise PreflightError("equivalence policy is unsupported")
    equivalence_indices, equivalence_seeds = _validate_selection(
        protocol,
        equivalence,
        namespace=EQUIVALENCE_NAMESPACE,
        label="equivalence",
    )
    reference = validate_probe_document(
        equivalence["reference_probe"],
        expected_policy=equivalence["policy"],
        expected_seeds=equivalence_seeds,
    )
    current = validate_probe_document(
        equivalence["current_probe"],
        expected_policy=equivalence["policy"],
        expected_seeds=equivalence_seeds,
    )
    if reference["frozen_policy"] != current["frozen_policy"]:
        raise PreflightError("frozen-policy semantic divergence")
    reference_bytes = enoch_week1.canonical_json_bytes(reference) + b"\n"
    current_bytes = enoch_week1.canonical_json_bytes(current) + b"\n"
    if reference_bytes != current_bytes:
        raise PreflightError("frozen-policy byte divergence")
    for field in (
        "current_stdout_sha256",
        "frozen_policy_sha256",
        "reference_stdout_sha256",
    ):
        _sha256(equivalence[field], f"equivalence.{field}")
    if (
        equivalence["current_binary_sha256"]
        != control["identity"]["current_probe_sha256"]
        or equivalence["reference_binary_sha256"]
        != control["identity"]["reference_probe_sha256"]
    ):
        raise PreflightError("equivalence did not use the frozen W1.0 probe binaries")
    if (
        equivalence["reference_stdout_sha256"] != _sha256_bytes(reference_bytes)
        or equivalence["current_stdout_sha256"] != _sha256_bytes(current_bytes)
        or equivalence["frozen_policy_sha256"] != reference["equivalence_sha256"]
    ):
        raise PreflightError("equivalence hashes do not reconstruct from probe output")
    equivalence_range = _validate_seed_claim(
        protocol,
        ledger,
        _mapping(equivalence["seed_claim"], "equivalence.seed_claim"),
        namespace=EQUIVALENCE_NAMESPACE,
        indices=equivalence_indices,
        seeds=equivalence_seeds,
        consumer=EQUIVALENCE_CONSUMER,
        label="equivalence",
    )

    outcome_range = _validate_baseline_section(
        protocol,
        ledger,
        _mapping(preflight["searchless_outcome_baseline"], "outcome baseline"),
        namespace=OUTCOME_BASELINE_NAMESPACE,
        consumer=OUTCOME_BASELINE_CONSUMER,
        purpose="searchless-outcome",
        label="searchless_outcome_baseline",
        expected_binary_sha256=control["identity"]["reference_probe_sha256"],
    )
    style_range = _validate_baseline_section(
        protocol,
        ledger,
        _mapping(preflight["style_baseline"], "style baseline"),
        namespace=STYLE_BASELINE_NAMESPACE,
        consumer=STYLE_BASELINE_CONSUMER,
        purpose="style",
        label="style_baseline",
        expected_binary_sha256=control["identity"]["reference_probe_sha256"],
    )
    if not (
        equivalence_range[1] <= outcome_range[0]
        and outcome_range[1] <= style_range[0]
    ):
        raise PreflightError("preflight seed claims are not ordered by phase")
    coverage_statuses = (
        equivalence["coverage"]["status"],
        preflight["searchless_outcome_baseline"]["coverage"]["status"],
        preflight["style_baseline"]["coverage"]["status"],
    )
    expected_coverage_status = (
        "full" if all(status == "full" for status in coverage_statuses) else "partial-prefix"
    )
    if preflight["coverage_status"] != expected_coverage_status:
        raise PreflightError("aggregate preflight coverage status is inconsistent")
    expected_authoritative = expected_coverage_status == "full"
    if preflight["authoritative_for_w1_1_completion"] is not expected_authoritative:
        raise PreflightError("preflight completion authority disagrees with coverage")
    if require_full_coverage and not expected_authoritative:
        raise PreflightError("W1.1 completion requires full preflight namespace coverage")
    final_ledger_fingerprint = _sha256(
        preflight["final_ledger_fingerprint"], "final preflight ledger fingerprint"
    )
    if final_ledger_fingerprint != preflight["style_baseline"]["seed_claim"][
        "ledger_fingerprint_after_claim"
    ]:
        raise PreflightError("final preflight ledger fingerprint is not the final seed claim")

    fingerprint = _sha256(preflight["artifact_sha256"], "preflight artifact hash")
    body = dict(preflight)
    body.pop("artifact_sha256")
    if fingerprint != enoch_week1.canonical_json_sha256(body):
        raise PreflightError("preflight artifact fingerprint mismatch")
    return fingerprint


def _baseline_record(
    run: ProbeRun,
    *,
    reference_probe_sha256: str,
    namespace: str,
    purpose: str,
    indices: Sequence[int],
    seeds: Sequence[int],
    coverage: Mapping[str, Any],
    claim: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "coverage": dict(coverage),
        "namespace": namespace,
        "policy": "enoch-greedy",
        "probe": run.document,
        "probe_binary_sha256": reference_probe_sha256,
        "probe_stdout_sha256": run.stdout_sha256,
        "purpose": purpose,
        "seed_claim": dict(claim),
        "seed_count": len(seeds),
        "seed_indices": list(indices),
        "seeds": list(seeds),
        "summary": _baseline_summary(run.document),
    }


def build_preflight_artifact(
    protocol: Mapping[str, Any],
    ledger_path: Path,
    control_bundle: Path,
    *,
    equivalence_policy: str = "enoch-greedy",
    equivalence_count: int | None = None,
    outcome_count: int | None = None,
    style_count: int | None = None,
    allow_partial_prefix: bool = False,
    timeout_seconds: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run the three protocol-bound W1.1 probes with single-use seed claims.

    Counts default to each complete frozen namespace. Explicit smaller counts
    are supported for smoke tests and are recorded as partial-prefix coverage.
    A failed probe deliberately leaves its claimed seeds consumed.
    """

    protocol_fingerprint = enoch_week1.validate_protocol(protocol)
    control = _verify_control_bundle(protocol, control_bundle)
    reference_probe = control["artifact_paths"][CONTROL_REFERENCE_PROBE_ID]
    current_probe = control["artifact_paths"][CONTROL_CURRENT_PROBE_ID]
    _verify_probe_files(control)
    if not ledger_path.is_file():
        raise PreflightError(f"authoritative seed ledger does not exist: {ledger_path}")
    initial_ledger = enoch_week1.load_json_object(ledger_path)
    enoch_week1.validate_seed_ledger(protocol, initial_ledger)
    if equivalence_policy not in POLICIES:
        raise PreflightError(f"unsupported equivalence policy {equivalence_policy!r}")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or timeout_seconds <= 0
    ):
        raise PreflightError("probe timeout_seconds must be positive")
    equivalence_indices, equivalence_seeds, equivalence_coverage = _protocol_prefix(
        protocol,
        EQUIVALENCE_NAMESPACE,
        equivalence_count,
        "equivalence",
    )
    outcome_indices, outcome_seeds, outcome_coverage = _protocol_prefix(
        protocol,
        OUTCOME_BASELINE_NAMESPACE,
        outcome_count,
        "searchless outcome baseline",
    )
    style_indices, style_seeds, style_coverage = _protocol_prefix(
        protocol,
        STYLE_BASELINE_NAMESPACE,
        style_count,
        "style baseline",
    )
    coverage_records = (
        equivalence_coverage,
        outcome_coverage,
        style_coverage,
    )
    full_coverage = all(record["status"] == "full" for record in coverage_records)
    if not full_coverage and not allow_partial_prefix:
        raise PreflightError(
            "partial-prefix preflight is non-authoritative; pass allow_partial_prefix=True"
        )
    cleaned, removed = _safe_environment(
        environment,
        allowlist=protocol["evaluator_environment_policy"]["allowlist"],
    )

    equivalence_claim, _ = _claim_seed_prefix(
        ledger_path,
        protocol,
        namespace=EQUIVALENCE_NAMESPACE,
        indices=equivalence_indices,
        seeds=equivalence_seeds,
        consumer=EQUIVALENCE_CONSUMER,
    )
    reference = run_probe(
        reference_probe,
        equivalence_policy,
        equivalence_seeds,
        environment=cleaned,
        timeout_seconds=timeout_seconds,
    )
    _verify_probe_files(control)
    current = run_probe(
        current_probe,
        equivalence_policy,
        equivalence_seeds,
        environment=cleaned,
        timeout_seconds=timeout_seconds,
    )
    _verify_probe_files(control)
    if reference.document["frozen_policy"] != current.document["frozen_policy"]:
        raise PreflightError("frozen-policy semantic divergence")
    if reference.stdout != current.stdout:
        raise PreflightError("frozen-policy byte divergence")

    outcome_claim, _ = _claim_seed_prefix(
        ledger_path,
        protocol,
        namespace=OUTCOME_BASELINE_NAMESPACE,
        indices=outcome_indices,
        seeds=outcome_seeds,
        consumer=OUTCOME_BASELINE_CONSUMER,
    )
    outcome = run_probe(
        reference_probe,
        "enoch-greedy",
        outcome_seeds,
        environment=cleaned,
        timeout_seconds=timeout_seconds,
    )
    _verify_probe_files(control)

    style_claim, final_ledger = _claim_seed_prefix(
        ledger_path,
        protocol,
        namespace=STYLE_BASELINE_NAMESPACE,
        indices=style_indices,
        seeds=style_seeds,
        consumer=STYLE_BASELINE_CONSUMER,
    )
    style = run_probe(
        reference_probe,
        "enoch-greedy",
        style_seeds,
        environment=cleaned,
        timeout_seconds=timeout_seconds,
    )
    _verify_probe_files(control)

    reference_probe_sha256 = control["identity"]["reference_probe_sha256"]
    body: dict[str, Any] = {
        "authoritative_for_w1_1_completion": full_coverage,
        "automatic_production_promotion_allowed": False,
        "coverage_status": "full" if full_coverage else "partial-prefix",
        "environment": {
            "blocked_prefixes": list(enoch_week1.BLOCKED_EVALUATOR_ENV_PREFIXES),
            "effective_environment_sha256": enoch_week1.canonical_json_sha256(
                dict(sorted(cleaned.items()))
            ),
            "removed_names": list(removed),
            "sanitized": True,
        },
        "equivalence": {
            "coverage": equivalence_coverage,
            "current_binary_sha256": control["identity"]["current_probe_sha256"],
            "current_probe": current.document,
            "current_stdout_sha256": current.stdout_sha256,
            "frozen_policy_sha256": reference.document["equivalence_sha256"],
            "namespace": EQUIVALENCE_NAMESPACE,
            "policy": equivalence_policy,
            "reference_binary_sha256": reference_probe_sha256,
            "reference_probe": reference.document,
            "reference_stdout_sha256": reference.stdout_sha256,
            "seed_claim": equivalence_claim,
            "seed_count": len(equivalence_seeds),
            "seed_indices": equivalence_indices,
            "seeds": equivalence_seeds,
        },
        "execution_contract": {
            "binary_reverified_after_each_stage": True,
            "probe_timeout_seconds": timeout_seconds,
        },
        "final_ledger_fingerprint": final_ledger["ledger_fingerprint"],
        "kind": PREFLIGHT_KIND,
        "manifest_version": MANIFEST_VERSION,
        "protocol_fingerprint": protocol_fingerprint,
        "searchless_outcome_baseline": _baseline_record(
            outcome,
            reference_probe_sha256=reference_probe_sha256,
            namespace=OUTCOME_BASELINE_NAMESPACE,
            purpose="searchless-outcome",
            indices=outcome_indices,
            seeds=outcome_seeds,
            coverage=outcome_coverage,
            claim=outcome_claim,
        ),
        "seed_registry_sha256": protocol["seed_registry_sha256"],
        "style_baseline": _baseline_record(
            style,
            reference_probe_sha256=reference_probe_sha256,
            namespace=STYLE_BASELINE_NAMESPACE,
            purpose="style",
            indices=style_indices,
            seeds=style_seeds,
            coverage=style_coverage,
            claim=style_claim,
        ),
        "w1_0_identity": control["identity"],
    }
    artifact = {**body, "artifact_sha256": enoch_week1.canonical_json_sha256(body)}
    validate_preflight_artifact(
        protocol,
        final_ledger,
        control_bundle,
        artifact,
        require_full_coverage=full_coverage,
    )
    return artifact


def _canonical_search_fixture_command(test_name: str) -> list[str]:
    return [
        "cargo",
        "test",
        "--locked",
        "--release",
        "-p",
        "shengji-core",
        test_name,
        "--",
        "--exact",
    ]


def _workspace_source_records(workspace: Path) -> list[dict[str, str]]:
    try:
        paths = enoch_week1_freeze._canonical_source_paths(workspace)  # noqa: SLF001
    except enoch_week1_freeze.FreezeError as exc:
        raise PreflightError(str(exc)) from exc
    return [
        {
            "path": path.relative_to(workspace).as_posix(),
            "sha256": _sha256_file(path),
        }
        for path in paths
    ]


def build_deterministic_search_fixture_authority(
    protocol: Mapping[str, Any],
    control_bundle: Path,
    workspace: Path,
    *,
    timeout_seconds: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run the bounded release fixtures that authorize W1.1 search determinism."""

    enoch_week1.validate_protocol(protocol)
    control = _verify_control_bundle(protocol, control_bundle)
    workspace = workspace.resolve()
    if _workspace_source_records(workspace) != control["source_records"]:
        raise PreflightError(
            "deterministic-search fixtures are not running on the frozen W1.0 source"
        )
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or timeout_seconds <= 0
    ):
        raise PreflightError("deterministic-search fixture timeout must be positive")
    cleaned, removed = _safe_environment(
        environment,
        allowlist=protocol["evaluator_environment_policy"]["allowlist"],
    )
    records: list[dict[str, Any]] = []
    for test_name in DETERMINISTIC_SEARCH_TESTS:
        command = _canonical_search_fixture_command(test_name)
        try:
            completed = subprocess.run(
                command,
                cwd=workspace,
                env=dict(cleaned),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=float(timeout_seconds),
            )
        except subprocess.TimeoutExpired as exc:
            raise PreflightError(
                f"deterministic-search fixture timed out: {test_name}"
            ) from exc
        except OSError as exc:
            raise PreflightError(
                f"could not execute deterministic-search fixture {test_name}: {exc}"
            ) from exc
        try:
            output = completed.stdout.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PreflightError(
                f"deterministic-search fixture emitted non-UTF-8 output: {test_name}"
            ) from exc
        marker = f"test {test_name} ... ok"
        if completed.returncode != 0 or marker not in output:
            raise PreflightError(
                f"deterministic-search fixture failed or did not execute: {test_name}"
            )
        records.append(
            {
                "command": command,
                "output": output,
                "output_sha256": _sha256_bytes(completed.stdout),
                "test_name": test_name,
            }
        )
        _verify_probe_files(control)
    if _workspace_source_records(workspace) != control["source_records"]:
        raise PreflightError("frozen source changed while search fixtures were running")
    body = {
        "automatic_production_promotion_allowed": False,
        "control_manifest_fingerprint": control["identity"][
            "control_manifest_fingerprint"
        ],
        "effective_environment_sha256": enoch_week1.canonical_json_sha256(
            dict(sorted(cleaned.items()))
        ),
        "evaluator_binary_sha256": control["manifest"]["evaluator_identity"][
            "binary_sha256"
        ],
        "evaluator_source_sha256": control["manifest"]["evaluator_identity"][
            "source_sha256"
        ],
        "manifest_kind": DETERMINISTIC_SEARCH_AUTHORITY_KIND,
        "manifest_version": MANIFEST_VERSION,
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "removed_environment_names": list(removed),
        "source_files_sha256": enoch_week1.canonical_json_sha256(
            control["source_records"]
        ),
        "tests": records,
        "timeout_seconds": timeout_seconds,
    }
    authority = {
        **body,
        "deterministic_search_authority_fingerprint": enoch_week1.canonical_json_sha256(
            body
        ),
    }
    validate_deterministic_search_fixture_authority(
        protocol, control_bundle, authority
    )
    return authority


def validate_deterministic_search_fixture_authority(
    protocol: Mapping[str, Any],
    control_bundle: Path,
    authority: Mapping[str, Any],
) -> str:
    control = _verify_control_bundle(protocol, control_bundle)
    fixture = _mapping(authority, "deterministic-search fixture authority")
    _exact_keys(
        fixture,
        {
            "automatic_production_promotion_allowed",
            "control_manifest_fingerprint",
            "deterministic_search_authority_fingerprint",
            "effective_environment_sha256",
            "evaluator_binary_sha256",
            "evaluator_source_sha256",
            "manifest_kind",
            "manifest_version",
            "protocol_fingerprint",
            "removed_environment_names",
            "source_files_sha256",
            "tests",
            "timeout_seconds",
        },
        "deterministic-search fixture authority",
    )
    if (
        fixture["manifest_kind"] != DETERMINISTIC_SEARCH_AUTHORITY_KIND
        or fixture["manifest_version"] != MANIFEST_VERSION
    ):
        raise PreflightError("unsupported deterministic-search fixture authority")
    if fixture["automatic_production_promotion_allowed"] is not False:
        raise PreflightError("search fixture authority cannot authorize promotion")
    expected_bindings = {
        "control_manifest_fingerprint": control["identity"][
            "control_manifest_fingerprint"
        ],
        "evaluator_binary_sha256": control["manifest"]["evaluator_identity"][
            "binary_sha256"
        ],
        "evaluator_source_sha256": control["manifest"]["evaluator_identity"][
            "source_sha256"
        ],
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "source_files_sha256": enoch_week1.canonical_json_sha256(
            control["source_records"]
        ),
    }
    for field, expected in expected_bindings.items():
        if fixture[field] != expected:
            raise PreflightError(f"search fixture authority {field} mismatch")
    _sha256(
        fixture["effective_environment_sha256"],
        "search fixture effective environment hash",
    )
    timeout_seconds = fixture["timeout_seconds"]
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or timeout_seconds <= 0
    ):
        raise PreflightError("search fixture timeout must be positive")
    removed = fixture["removed_environment_names"]
    if not isinstance(removed, list) or removed != sorted(set(removed)):
        raise PreflightError("search fixture removed environment names are not canonical")
    tests = fixture["tests"]
    if not isinstance(tests, list) or len(tests) != len(DETERMINISTIC_SEARCH_TESTS):
        raise PreflightError("search fixture authority has incomplete test coverage")
    for record, test_name in zip(tests, DETERMINISTIC_SEARCH_TESTS):
        record = _mapping(record, f"search fixture {test_name}")
        _exact_keys(
            record,
            {"command", "output", "output_sha256", "test_name"},
            f"search fixture {test_name}",
        )
        if (
            record["test_name"] != test_name
            or record["command"] != _canonical_search_fixture_command(test_name)
        ):
            raise PreflightError("search fixture command/test identity changed")
        if not isinstance(record["output"], str):
            raise PreflightError("search fixture output must be text")
        if record["output_sha256"] != _sha256_bytes(record["output"].encode("utf-8")):
            raise PreflightError("search fixture raw output hash mismatch")
        if f"test {test_name} ... ok" not in record["output"]:
            raise PreflightError(f"search fixture lacks passing evidence: {test_name}")
    fingerprint = _sha256(
        fixture["deterministic_search_authority_fingerprint"],
        "deterministic-search authority fingerprint",
    )
    body = dict(fixture)
    body.pop("deterministic_search_authority_fingerprint")
    if fingerprint != enoch_week1.canonical_json_sha256(body):
        raise PreflightError("deterministic-search authority fingerprint mismatch")
    return fingerprint


def _standard_product_launch_configuration(
    control: Mapping[str, Any],
) -> dict[str, Any]:
    search_knobs = _mapping(
        control["manifest"]["search_knobs"], "W1.0 search knobs"
    )
    enoch_knobs = _mapping(search_knobs.get("enoch-0"), "W1.0 Enoch-0 search knobs")
    _exact_keys(
        enoch_knobs,
        {
            "budget_ms",
            "max_candidates",
            "max_worlds",
            "policy",
            "rollout_policy",
            "rollout_tricks",
        },
        "W1.0 Enoch-0 search knobs",
    )
    if (
        enoch_knobs["policy"] != "EnochHeuristic"
        or enoch_knobs["rollout_policy"] != "EnochHeuristic"
    ):
        raise PreflightError("W1.0 Enoch-0 is not the standard heuristic policy")
    try:
        return enoch_week1_runner.build_launch_configuration(
            candidate_arm_ids=[],
            worlds=enoch_knobs["max_worlds"],
            candidates=enoch_knobs["max_candidates"],
            rollout_tricks=enoch_knobs["rollout_tricks"],
            scenario_id="standard",
            budget_ms=enoch_knobs["budget_ms"],
        )
    except enoch_week1_runner.RunnerError as exc:
        raise PreflightError(f"invalid W1.0 product-budget configuration: {exc}") from exc


def _smoke_evidence_summary(
    protocol: Mapping[str, Any],
    control: Mapping[str, Any],
    smoke_evidence: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    expected = (
        ("smoke/product/001", 1),
        ("smoke/product/010", 10),
        ("smoke/product/100", 100),
    )
    if set(smoke_evidence) != {namespace for namespace, _ in expected}:
        raise PreflightError("W1.1 report requires exact 1/10/100 smoke evidence")
    reference_enoch0_fingerprint = control["identity"]["enoch0_fingerprint"]
    evaluator_fingerprint = control["identity"]["evaluator_fingerprint"]
    evaluator_identity = control["manifest"]["evaluator_identity"]
    expected_launch = _standard_product_launch_configuration(control)
    try:
        expected_identities = enoch_week1_runner.build_in_process_identity_bindings(
            evaluator_identity, expected_launch
        )
    except enoch_week1_runner.RunnerError as exc:
        raise PreflightError(f"invalid W1.1 runtime identity contract: {exc}") from exc
    if expected_identities["candidate"] != expected_identities["control"]:
        raise PreflightError("empty-arm W1.1 runtime candidate/control identities differ")
    runtime_control_fingerprint = enoch_week1.canonical_json_sha256(
        expected_identities["control"]
    )
    if runtime_control_fingerprint == reference_enoch0_fingerprint:
        raise PreflightError(
            "permanent reference Enoch-0 and runtime evaluation control were conflated"
        )
    summaries: list[dict[str, Any]] = []
    common_bindings: dict[str, Any] | None = None
    hundred_comparison: Mapping[str, Any] | None = None
    for namespace, pair_count in expected:
        evidence = _mapping(smoke_evidence[namespace], f"smoke evidence {namespace}")
        _exact_keys(
            evidence,
            {
                "comparison",
                "identity_bindings",
                "launch_configuration",
                "merged_result",
            },
            f"smoke evidence {namespace}",
        )
        launch = _mapping(
            evidence["launch_configuration"], f"smoke launch configuration {namespace}"
        )
        try:
            launch_fingerprint = enoch_week1_runner.validate_launch_configuration(launch)
        except enoch_week1_runner.RunnerError as exc:
            raise PreflightError(f"invalid smoke launch configuration {namespace}: {exc}") from exc
        if launch["candidate_arm_ids"]:
            raise PreflightError(f"{namespace} smoke must use the empty Enoch arm set")
        if dict(launch) != expected_launch:
            raise PreflightError(
                f"{namespace} does not use the frozen standard product-budget configuration"
            )
        identities = _mapping(
            evidence["identity_bindings"], f"smoke identity bindings {namespace}"
        )
        _exact_keys(
            identities,
            {"candidate", "control", "evaluator"},
            f"smoke identity bindings {namespace}",
        )
        if identities["evaluator"] != evaluator_identity:
            raise PreflightError(
                f"{namespace} is not bound to the frozen current evaluator identity"
            )
        if dict(identities) != expected_identities:
            raise PreflightError(
                f"{namespace} does not use identical empty-arm runtime identities"
            )
        comparison = _mapping(evidence["comparison"], f"smoke comparison {namespace}")
        merged = _mapping(evidence["merged_result"], f"smoke merged result {namespace}")
        comparison_fingerprint = enoch_week1.validate_comparison_protocol_manifest(
            protocol, comparison
        )
        merged_fingerprint = enoch_week1.validate_merged_result(
            protocol, comparison, merged
        )
        if (
            comparison["phase"] != "W1.1"
            or comparison["seed_namespace"] != namespace
            or comparison["pair_count"] != pair_count
        ):
            raise PreflightError(f"{namespace} is not its exact W1.1 product smoke")
        if (
            comparison["candidate_fingerprint"] != runtime_control_fingerprint
            or comparison["control_fingerprint"] != runtime_control_fingerprint
            or comparison["evaluator_fingerprint"] != evaluator_fingerprint
            or comparison["configuration_fingerprint"] != launch_fingerprint
        ):
            raise PreflightError(
                f"{namespace} is not current-evaluator empty-arm Enoch self-play"
            )
        if comparison["required_style_metrics"] != sorted(
            enoch_week1.WEEK1_STYLE_METRICS
        ):
            raise PreflightError(f"{namespace} lacks the complete W1.1 style metrics")
        metrics = merged["metrics"]
        failures = metrics["failure_counters"]
        nonzero = {name: count for name, count in failures.items() if count != 0}
        if nonzero:
            raise PreflightError(f"{namespace} has nonzero failure counters: {nonzero}")
        if (
            metrics["candidate_completed_worlds_mean"] <= 0
            or metrics["control_completed_worlds_mean"] <= 0
        ):
            raise PreflightError(f"{namespace} completed no search worlds")
        bindings = {
            field: comparison[field]
            for field in (
                "candidate_fingerprint",
                "configuration_fingerprint",
                "control_fingerprint",
                "environment_fingerprint",
                "evaluator_fingerprint",
                "required_style_metrics",
                "subject_id",
            )
        }
        if common_bindings is None:
            common_bindings = bindings
        elif bindings != common_bindings:
            raise PreflightError("1/10/100 smoke runs do not share one frozen configuration")
        summaries.append(
            {
                "comparison_protocol_fingerprint": comparison_fingerprint,
                "merged_result_fingerprint": merged_fingerprint,
                "metrics": metrics,
                "namespace": namespace,
                "pair_count": pair_count,
                "runtime_evaluation_control_fingerprint": runtime_control_fingerprint,
                "shard_count": len(comparison["shards"]),
            }
        )
        if pair_count == 100:
            hundred_comparison = comparison
    assert common_bindings is not None and hundred_comparison is not None
    worker_configuration = {
        "configuration_fingerprint": common_bindings["configuration_fingerprint"],
        "environment_fingerprint": common_bindings["environment_fingerprint"],
        "maximum_parallel_workers": len(hundred_comparison["shards"]),
        "pair_counts_per_shard": [
            len(shard["seed_indices"]) for shard in hundred_comparison["shards"]
        ],
        "selection_basis": "complete-zero-failure-100-pair-product-smoke",
        "shard_count": len(hundred_comparison["shards"]),
    }
    runtime_contract = {
        "identity_bindings": expected_identities,
        "launch_configuration": expected_launch,
        "reference_enoch0_fingerprint": reference_enoch0_fingerprint,
        "runtime_evaluation_control_fingerprint": runtime_control_fingerprint,
        "runtime_evaluator_fingerprint": evaluator_fingerprint,
    }
    return summaries, worker_configuration, runtime_contract


def _baseline_worker_report_body(
    protocol: Mapping[str, Any],
    ledger: Mapping[str, Any],
    control_bundle: Path,
    preflight_artifact: Mapping[str, Any],
    smoke_evidence: Mapping[str, Mapping[str, Any]],
    deterministic_search_authority: Mapping[str, Any],
) -> dict[str, Any]:
    preflight_fingerprint = validate_preflight_artifact(
        protocol,
        ledger,
        control_bundle,
        preflight_artifact,
        require_full_coverage=True,
    )
    authority_fingerprint = validate_deterministic_search_fixture_authority(
        protocol, control_bundle, deterministic_search_authority
    )
    control = _verify_control_bundle(protocol, control_bundle)
    smoke_summaries, worker_configuration, runtime_contract = _smoke_evidence_summary(
        protocol, control, smoke_evidence
    )
    return {
        "automatic_production_promotion_allowed": False,
        "control_manifest_fingerprint": control["identity"][
            "control_manifest_fingerprint"
        ],
        "deterministic_search_authority_fingerprint": authority_fingerprint,
        "fixed_worker_configuration": worker_configuration,
        "manifest_kind": BASELINE_WORKER_REPORT_KIND,
        "manifest_version": MANIFEST_VERSION,
        "preflight_artifact_sha256": preflight_fingerprint,
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "reference_enoch0_fingerprint": runtime_contract[
            "reference_enoch0_fingerprint"
        ],
        "runtime_evaluation_control_fingerprint": runtime_contract[
            "runtime_evaluation_control_fingerprint"
        ],
        "runtime_evaluator_fingerprint": runtime_contract[
            "runtime_evaluator_fingerprint"
        ],
        "runtime_identity_bindings": runtime_contract["identity_bindings"],
        "runtime_launch_configuration": runtime_contract["launch_configuration"],
        "searchless_outcome_summary": preflight_artifact[
            "searchless_outcome_baseline"
        ]["summary"],
        "smoke_summaries": smoke_summaries,
        "style_baseline_summary": preflight_artifact["style_baseline"]["summary"],
        "w1_0_artifact_hashes_sha256": control["identity"][
            "artifact_hashes_sha256"
        ],
    }


def build_w1_1_baseline_worker_report(
    protocol: Mapping[str, Any],
    ledger: Mapping[str, Any],
    control_bundle: Path,
    preflight_artifact: Mapping[str, Any],
    smoke_evidence: Mapping[str, Mapping[str, Any]],
    deterministic_search_authority: Mapping[str, Any],
) -> dict[str, Any]:
    body = _baseline_worker_report_body(
        protocol,
        ledger,
        control_bundle,
        preflight_artifact,
        smoke_evidence,
        deterministic_search_authority,
    )
    return {
        **body,
        "baseline_worker_report_fingerprint": enoch_week1.canonical_json_sha256(body),
    }


def validate_w1_1_baseline_worker_report(
    protocol: Mapping[str, Any],
    ledger: Mapping[str, Any],
    control_bundle: Path,
    preflight_artifact: Mapping[str, Any],
    smoke_evidence: Mapping[str, Mapping[str, Any]],
    deterministic_search_authority: Mapping[str, Any],
    report: Mapping[str, Any],
) -> str:
    result = _mapping(report, "W1.1 baseline-and-worker report")
    expected_body = _baseline_worker_report_body(
        protocol,
        ledger,
        control_bundle,
        preflight_artifact,
        smoke_evidence,
        deterministic_search_authority,
    )
    expected = {
        **expected_body,
        "baseline_worker_report_fingerprint": enoch_week1.canonical_json_sha256(
            expected_body
        ),
    }
    if dict(result) != expected:
        raise PreflightError(
            "W1.1 baseline-and-worker report does not reconstruct from its evidence"
        )
    return expected["baseline_worker_report_fingerprint"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--control-bundle", type=Path, required=True)
    parser.add_argument("--equivalence-policy", choices=POLICIES, default="enoch-greedy")
    parser.add_argument(
        "--equivalence-count",
        type=int,
        help="use this frozen namespace prefix (default: complete namespace)",
    )
    parser.add_argument(
        "--outcome-count",
        type=int,
        help="use this searchless-outcome prefix (default: complete namespace)",
    )
    parser.add_argument(
        "--style-count",
        type=int,
        help="use this style-baseline prefix (default: complete namespace)",
    )
    parser.add_argument(
        "--allow-partial-prefix",
        action="store_true",
        help="produce an explicitly non-authoritative smoke/test artifact",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_PROBE_TIMEOUT_SECONDS,
        help="per-probe timeout (default: 21600 seconds)",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        protocol = enoch_week1.load_json_object(args.protocol.resolve())
        artifact = build_preflight_artifact(
            protocol,
            args.ledger.resolve(),
            args.control_bundle.resolve(),
            equivalence_policy=args.equivalence_policy,
            equivalence_count=args.equivalence_count,
            outcome_count=args.outcome_count,
            style_count=args.style_count,
            allow_partial_prefix=args.allow_partial_prefix,
            timeout_seconds=args.timeout_seconds,
        )
        enoch_week1.atomic_write_json(args.output.resolve(), artifact)
    except (PreflightError, enoch_week1.ProtocolError, FileExistsError) as exc:
        print(f"preflight failed: {exc}", file=sys.stderr)
        return 2
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
