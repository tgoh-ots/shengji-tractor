#!/usr/bin/env python3
"""Fail-closed execution bridge for frozen Enoch Week-1 comparisons."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

try:
    import fcntl
except ImportError:  # pragma: no cover - Week-1 execution is Unix-only.
    fcntl = None

try:  # Support both module import and direct ``python training/...`` use.
    from . import enoch_week1, enoch_week1_evidence
except ImportError:  # pragma: no cover - direct-script path.
    import enoch_week1  # type: ignore[no-redef]
    import enoch_week1_evidence  # type: ignore[no-redef]


RUNNER_MANIFEST_VERSION = 1
DEFAULT_WORKERS = 8
MAX_WORKERS = 10
MAX_STDOUT_BYTES = 256 * 1024 * 1024
AUTHORITATIVE_PHASES = {"W1.1", "W1.2", "W1.3", "W1.4", "W1.5", "W1.6", "W1.7"}
CLEAN_ENVIRONMENT_PHASES = {"W1.2", "W1.3", "W1.4", "W1.5", "W1.6", "W1.7"}
REQUIRED_EXTERNAL_COUNTERS = {
    "artifact_mismatch",
    "fixture_failure",
    "hidden_information_leak",
    "honesty_violation",
    "machine_contention",
    "model_contract_failure",
}

# Python experiment names intentionally differ from several stable Rust feature
# names.  This complete mapping is frozen and hashed; feature strings are never
# guessed from slugs.
ARM_TO_RUST_FEATURE: tuple[tuple[str, str], ...] = (
    ("bid-ownership", "bid-ownership"),
    ("compound-follow", "compound-follow"),
    ("failed-throw-better-player", "failed-throw-witness"),
    ("friend-revelation", "friend-revelation"),
    ("terminal-level-utility", "terminal-level"),
    ("kitty-burial", "default-kitty"),
    ("late-ruff-shape", "ruff-shape"),
    ("contextual-empty-trick", "contextual-empty-trick"),
    ("relative-live-suit", "live-suit-control"),
    ("team-void-boss", "team-void"),
    ("teammate-entry-return", "entry-return"),
    ("low-trump-handoff", "handoff-protection"),
    ("structural-family-coverage", "structural-families"),
    ("progressive-admission", "progressive-admission"),
    ("uncertain-legal-throws", "uncertain-throws"),
)
ARM_TO_RUST_FEATURE_SHA256 = enoch_week1.canonical_json_sha256(ARM_TO_RUST_FEATURE)
_ARM_FEATURES = dict(ARM_TO_RUST_FEATURE)
RUST_STYLE_METRICS = enoch_week1.WEEK1_STYLE_METRICS
RUST_STYLE_METRICS_SHA256 = enoch_week1.canonical_json_sha256(RUST_STYLE_METRICS)
IN_PROCESS_ENOCH_MODEL_SHA256 = enoch_week1.canonical_json_sha256(
    {"model": "none", "reason": "heuristic-policy-tier"}
)
# This is the exact release target used by the frozen Week-1 evaluator.  Keep
# it data-only so operators can construct the Rust environment identity before
# a comparison manifest exists, while the hash below detects in-process drift.
EVALUATOR_CRATE_TARGET_CONTRACT: tuple[tuple[str, Any], ...] = (
    ("build_profile", "release"),
    ("crate_name", "shengji-core"),
    ("crate_version", "0.1.0"),
    ("target_arch", "aarch64"),
    ("target_endian", "little"),
    ("target_family", "unix"),
    ("target_os", "macos"),
    ("target_pointer_width", 64),
)
EVALUATOR_CRATE_TARGET_CONTRACT_SHA256 = (
    "cef718f5828558f49a3133b35d4d78eb3a99809c758ed01def24e3f0fb2874ae"
)
DEVELOPMENT_FRIEND_SCENARIO = "development-finding-friends"
QUALIFICATION_SCENARIO_BINDINGS: tuple[tuple[str, str, str], ...] = (
    ("qual-intended", "qual/intended", "intended"),
    ("qual-equal", "qual/equal", "equal"),
    ("qual-rank-low", "qual/rank/low", "rank-low"),
    ("qual-rank-middle", "qual/rank/middle", "rank-middle"),
    ("qual-rank-high", "qual/rank/high", "rank-high"),
    (
        "qual-crossplay-assignment-01",
        "qual/crossplay/assignment-01",
        "crossplay-assignment-01",
    ),
    (
        "qual-crossplay-assignment-02",
        "qual/crossplay/assignment-02",
        "crossplay-assignment-02",
    ),
    (
        "qual-crossplay-assignment-03",
        "qual/crossplay/assignment-03",
        "crossplay-assignment-03",
    ),
    (
        "qual-crossplay-assignment-04",
        "qual/crossplay/assignment-04",
        "crossplay-assignment-04",
    ),
    (
        "qual-configuration-slot-01",
        "qual/configuration/slot-01",
        "configuration-slot-01",
    ),
    (
        "qual-configuration-slot-02",
        "qual/configuration/slot-02",
        "configuration-slot-02",
    ),
    (
        "qual-configuration-slot-03",
        "qual/configuration/slot-03",
        "configuration-slot-03",
    ),
    (
        "qual-finding-friends-contract-01",
        "qual/finding-friends/contract-01",
        "finding-friends-contract-01",
    ),
    (
        "qual-finding-friends-contract-02",
        "qual/finding-friends/contract-02",
        "finding-friends-contract-02",
    ),
    (
        "qual-scoring-kitty-ruleset-01",
        "qual/scoring/kitty-ruleset-01",
        "scoring-kitty-ruleset-01",
    ),
    (
        "qual-scoring-kitty-ruleset-02",
        "qual/scoring/kitty-ruleset-02",
        "scoring-kitty-ruleset-02",
    ),
    (
        "qual-scoring-kitty-ruleset-03",
        "qual/scoring/kitty-ruleset-03",
        "scoring-kitty-ruleset-03",
    ),
    (
        "qual-threshold-situation-01",
        "qual/threshold/situation-01",
        "threshold-situation-01",
    ),
    (
        "qual-threshold-situation-02",
        "qual/threshold/situation-02",
        "threshold-situation-02",
    ),
    (
        "qual-threshold-situation-03",
        "qual/threshold/situation-03",
        "threshold-situation-03",
    ),
    (
        "qual-threshold-situation-04",
        "qual/threshold/situation-04",
        "threshold-situation-04",
    ),
)
QUALIFICATION_SCENARIO_BINDINGS_SHA256 = enoch_week1.canonical_json_sha256(
    QUALIFICATION_SCENARIO_BINDINGS
)
_QUALIFICATION_SCENARIOS = {
    (comparison_id, namespace): scenario
    for comparison_id, namespace, scenario in QUALIFICATION_SCENARIO_BINDINGS
}


class RunnerError(RuntimeError):
    """Raised when launch or evaluator evidence violates the frozen protocol."""


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise RunnerError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RunnerError(f"{label} must be lowercase SHA-256")
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RunnerError(f"{label} must be a nonnegative integer")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    value = _require_nonnegative_int(value, label)
    if value == 0:
        raise RunnerError(f"{label} must be positive")
    return value


def _require_finite(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RunnerError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise RunnerError(f"{label} must be a finite{' nonnegative' if nonnegative else ''} number")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while block := source.read(1024 * 1024):
                digest.update(block)
    except OSError as exc:
        raise RunnerError(f"cannot hash evaluator {path}: {exc}") from exc
    return digest.hexdigest()


def _assert_feature_mapping() -> None:
    python_arms = tuple(enoch_week1.ABLATION_ARMS)
    if tuple(name for name, _ in ARM_TO_RUST_FEATURE) != python_arms:
        raise RunnerError("Python-to-Rust arm mapping does not exactly cover the canonical registry")
    rust_names = tuple(name for _, name in ARM_TO_RUST_FEATURE)
    if len(set(rust_names)) != len(rust_names):
        raise RunnerError("Python-to-Rust arm mapping aliases Rust features")
    if enoch_week1.canonical_json_sha256(ARM_TO_RUST_FEATURE) != ARM_TO_RUST_FEATURE_SHA256:
        raise RunnerError("in-process mutation of the Python-to-Rust feature mapping detected")
    if (
        RUST_STYLE_METRICS_SHA256 != enoch_week1.WEEK1_STYLE_METRICS_SHA256
        or tuple(RUST_STYLE_METRICS) != tuple(enoch_week1.WEEK1_STYLE_METRICS)
    ):
        raise RunnerError("runner style schema differs from the frozen Week-1 registry")


def _validate_supported_style_metrics(metrics: Sequence[str]) -> None:
    unsupported = set(metrics) - set(RUST_STYLE_METRICS)
    if unsupported:
        raise RunnerError(
            f"comparison requests unsupported evaluator style metrics: {sorted(unsupported)}"
        )


def _validate_scenario_binding(
    comparison: Mapping[str, Any], launch: Mapping[str, Any]
) -> None:
    expected_registry = tuple(
        (
            f"qual-{entry['comparison_id']}",
            entry["namespace"],
            entry["comparison_id"],
        )
        for entry in enoch_week1.QUALIFICATION_MATRIX
    )
    if QUALIFICATION_SCENARIO_BINDINGS != expected_registry:
        raise RunnerError("qualification scenario mapping differs from the frozen matrix")
    if (
        enoch_week1.canonical_json_sha256(QUALIFICATION_SCENARIO_BINDINGS)
        != QUALIFICATION_SCENARIO_BINDINGS_SHA256
    ):
        raise RunnerError("qualification scenario mapping hash changed")
    if comparison["phase"] == "W1.5":
        key = (comparison["comparison_id"], comparison["seed_namespace"])
        expected_scenario = _QUALIFICATION_SCENARIOS.get(key)
        if expected_scenario is None:
            raise RunnerError("W1.5 comparison id/namespace lacks a frozen scenario binding")
        if launch["scenario_id"] != expected_scenario:
            raise RunnerError(
                f"W1.5 scenario must be {expected_scenario!r} for {key[0]!r}"
            )
    elif comparison["phase"] in {"W1.2", "W1.3"} and comparison[
        "subject_id"
    ] == "friend-revelation":
        if launch["scenario_id"] != DEVELOPMENT_FRIEND_SCENARIO:
            raise RunnerError(
                "friend-revelation development comparisons must use the frozen "
                "Finding Friends scenario"
            )
    elif launch["scenario_id"] != "standard":
        raise RunnerError("this non-W1.5 comparison must use the standard scenario")


def _validate_phase_work_contract(
    comparison: Mapping[str, Any], launch: Mapping[str, Any]
) -> None:
    phase = comparison["phase"]
    mode = launch["work_mode"]
    if phase in {"W1.2", "W1.3", "W1.4"} and mode != "fixed-work":
        raise RunnerError(f"{phase} comparisons require deterministic fixed work")
    if phase == "W1.5":
        comparison_id = comparison["comparison_id"]
        expected = "fixed-work" if comparison_id == "qual-equal" else "budget"
        if mode != expected:
            raise RunnerError(
                f"{comparison_id} requires {expected} mode under the frozen W1.5 contract"
            )
    if phase in {"W1.6", "W1.7"} and mode != "budget":
        raise RunnerError(f"{phase} locked gates require the frozen product-budget mode")


def _validate_phase_environment_contract(
    comparison: Mapping[str, Any], protocol: Mapping[str, Any]
) -> None:
    allowlist = protocol["evaluator_environment_policy"]["allowlist"]
    if comparison["phase"] in CLEAN_ENVIRONMENT_PHASES and allowlist:
        raise RunnerError(
            f"{comparison['phase']} requires an empty evaluator environment allowlist"
        )


def build_launch_configuration(
    *,
    candidate_arm_ids: Iterable[str],
    worlds: int,
    candidates: int,
    rollout_tricks: int,
    scenario_id: str = "standard",
    budget_ms: int | None = None,
    deadline_ms: int | None = None,
) -> dict[str, Any]:
    """Build the canonical configuration whose hash a comparison must freeze."""

    requested = list(candidate_arm_ids)
    order = {arm: index for index, arm in enumerate(enoch_week1.ABLATION_ARMS)}
    if any(arm not in order for arm in requested):
        unknown = sorted({arm for arm in requested if arm not in order})
        raise RunnerError(f"unknown Python Enoch arms: {unknown}")
    if len(requested) != len(set(requested)):
        raise RunnerError("candidate arm list contains duplicates")
    if requested != sorted(requested, key=order.__getitem__):
        raise RunnerError("candidate arms must use canonical registry order")
    if (budget_ms is None) == (deadline_ms is None):
        raise RunnerError("choose exactly one of budget_ms or deadline_ms")
    configuration = {
        "arm_feature_mapping_sha256": ARM_TO_RUST_FEATURE_SHA256,
        "budget_ms": budget_ms,
        "candidate_arm_ids": requested,
        "candidates": _require_positive_int(candidates, "candidate count"),
        "deadline_ms": deadline_ms,
        "rollout_tricks": _require_positive_int(rollout_tricks, "rollout trick count"),
        "scenario_id": scenario_id,
        "scenario_bindings_sha256": QUALIFICATION_SCENARIO_BINDINGS_SHA256,
        "work_mode": "budget" if budget_ms is not None else "fixed-work",
        "worlds": _require_positive_int(worlds, "world count"),
    }
    if budget_ms is not None:
        _require_positive_int(budget_ms, "budget_ms")
    if deadline_ms is not None:
        _require_positive_int(deadline_ms, "deadline_ms")
    validate_launch_configuration(configuration)
    return configuration


def validate_launch_configuration(configuration: Mapping[str, Any]) -> str:
    _assert_feature_mapping()
    if not isinstance(configuration, Mapping):
        raise RunnerError("launch configuration must be an object")
    _require_exact_keys(
        configuration,
        {
            "arm_feature_mapping_sha256",
            "budget_ms",
            "candidate_arm_ids",
            "candidates",
            "deadline_ms",
            "rollout_tricks",
            "scenario_id",
            "scenario_bindings_sha256",
            "work_mode",
            "worlds",
        },
        "launch configuration",
    )
    if configuration["arm_feature_mapping_sha256"] != ARM_TO_RUST_FEATURE_SHA256:
        raise RunnerError("launch configuration uses a different arm-feature mapping")
    arms = configuration["candidate_arm_ids"]
    if not isinstance(arms, list):
        raise RunnerError("candidate_arm_ids must be a list")
    order = {arm: index for index, arm in enumerate(enoch_week1.ABLATION_ARMS)}
    if any(not isinstance(arm, str) or arm not in order for arm in arms):
        raise RunnerError("candidate_arm_ids contains an unknown arm")
    if len(arms) != len(set(arms)) or arms != sorted(arms, key=order.__getitem__):
        raise RunnerError("candidate_arm_ids must be unique and canonically ordered")
    for field in ("worlds", "candidates", "rollout_tricks"):
        _require_positive_int(configuration[field], field)
    scenario_id = configuration["scenario_id"]
    if (
        configuration["scenario_bindings_sha256"]
        != QUALIFICATION_SCENARIO_BINDINGS_SHA256
    ):
        raise RunnerError("launch configuration uses a different scenario mapping")
    known_scenarios = {
        "standard",
        DEVELOPMENT_FRIEND_SCENARIO,
        *(entry[2] for entry in QUALIFICATION_SCENARIO_BINDINGS),
    }
    if not isinstance(scenario_id, str) or scenario_id not in known_scenarios:
        raise RunnerError(f"unknown frozen evaluator scenario: {scenario_id!r}")
    mode = configuration["work_mode"]
    if mode == "budget":
        _require_positive_int(configuration["budget_ms"], "budget_ms")
        if configuration["deadline_ms"] is not None:
            raise RunnerError("budget mode cannot declare deadline_ms")
    elif mode == "fixed-work":
        _require_positive_int(configuration["deadline_ms"], "deadline_ms")
        if configuration["budget_ms"] is not None:
            raise RunnerError("fixed-work mode cannot declare budget_ms")
    else:
        raise RunnerError("work_mode must be budget or fixed-work")
    return enoch_week1.canonical_json_sha256(configuration)


def rust_feature_spec(configuration: Mapping[str, Any]) -> str:
    validate_launch_configuration(configuration)
    names = [_ARM_FEATURES[arm] for arm in configuration["candidate_arm_ids"]]
    return ",".join(names) if names else "none"


def in_process_policy_configuration(
    candidate_arm_ids: Sequence[str], *, control: bool = False
) -> dict[str, Any]:
    """Canonical policy identity independent of scenario and work allocation."""

    arms = [] if control else list(candidate_arm_ids)
    order = {arm: index for index, arm in enumerate(enoch_week1.ABLATION_ARMS)}
    if any(arm not in order for arm in arms):
        raise RunnerError("in-process policy identity contains an unknown arm")
    if len(arms) != len(set(arms)) or arms != sorted(arms, key=order.__getitem__):
        raise RunnerError("in-process policy identity arms are not canonical")
    return {
        "arm_feature_mapping_sha256": ARM_TO_RUST_FEATURE_SHA256,
        "candidate_arm_ids": arms,
        "policy": "EnochHeuristic",
        "rust_feature_spec": ",".join(_ARM_FEATURES[arm] for arm in arms)
        if arms
        else "none",
        "version": 1,
    }


def build_in_process_identity_bindings(
    evaluator_identity: Mapping[str, Any], launch_configuration: Mapping[str, Any]
) -> dict[str, dict[str, str]]:
    """Construct the only candidate/control identities accepted by this runner."""

    validate_launch_configuration(launch_configuration)
    if not isinstance(evaluator_identity, Mapping):
        raise RunnerError("evaluator identity must be an object")
    _require_exact_keys(
        evaluator_identity,
        set(enoch_week1.EVALUATOR_IDENTITY_FIELDS),
        "evaluator identity",
    )
    for field in enoch_week1.EVALUATOR_IDENTITY_FIELDS:
        _require_sha256(evaluator_identity[field], f"evaluator {field}")
    common = {
        "binary_sha256": evaluator_identity["binary_sha256"],
        "model_sha256": IN_PROCESS_ENOCH_MODEL_SHA256,
        "source_sha256": evaluator_identity["source_sha256"],
    }
    candidate_configuration = in_process_policy_configuration(
        launch_configuration["candidate_arm_ids"]
    )
    control_configuration = in_process_policy_configuration([], control=True)
    return {
        "candidate": {
            **common,
            "configuration_sha256": enoch_week1.canonical_json_sha256(
                candidate_configuration
            ),
        },
        "control": {
            **common,
            "configuration_sha256": enoch_week1.canonical_json_sha256(
                control_configuration
            ),
        },
        "evaluator": dict(evaluator_identity),
    }


def build_external_failure_evidence(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Reject the retired naked-authority-digest evidence API.

    Call :func:`enoch_week1_evidence.build_verified_external_evidence` with
    actual artifact paths instead.
    """

    del args, kwargs
    raise RunnerError(
        "hash-only external evidence is forbidden; build verified file-backed evidence"
    )


def validate_external_failure_evidence(
    protocol: Mapping[str, Any],
    comparison: Mapping[str, Any],
    evidence: Mapping[str, Any] | None,
) -> dict[str, Mapping[str, Any]]:
    if evidence is None:
        return {}
    try:
        return enoch_week1_evidence.validate_verified_external_evidence(
            protocol, comparison, evidence
        )
    except (enoch_week1_evidence.EvidenceError, enoch_week1.ProtocolError) as exc:
        raise RunnerError(f"verified external evidence is invalid: {exc}") from exc


def _validate_declared_identity_contract(
    comparison: Mapping[str, Any],
    identities: Mapping[str, Any],
    launch_configuration: Mapping[str, Any],
) -> None:
    if not isinstance(identities, Mapping):
        raise RunnerError("identity bindings must be an object")
    _require_exact_keys(identities, {"candidate", "control", "evaluator"}, "identity bindings")
    for name in ("candidate", "control", "evaluator"):
        if not isinstance(identities[name], Mapping):
            raise RunnerError(f"{name} identity must be an object")
        expected = comparison[f"{name}_fingerprint"]
        if enoch_week1.canonical_json_sha256(identities[name]) != expected:
            raise RunnerError(f"{name} identity does not match the comparison fingerprint")
    expected_identities = build_in_process_identity_bindings(
        identities["evaluator"], launch_configuration
    )
    for name in ("candidate", "control"):
        _require_exact_keys(
            identities[name], set(enoch_week1.POLICY_IDENTITY_FIELDS), f"{name} identity"
        )
        for field in enoch_week1.POLICY_IDENTITY_FIELDS:
            _require_sha256(identities[name][field], f"{name} {field}")
        if dict(identities[name]) != expected_identities[name]:
            raise RunnerError(
                f"{name} identity is not the policy compiled into this evaluator/launch"
            )
    evaluator_identity = identities["evaluator"]
    _require_exact_keys(
        evaluator_identity,
        set(enoch_week1.EVALUATOR_IDENTITY_FIELDS),
        "evaluator identity",
    )
    for field in enoch_week1.EVALUATOR_IDENTITY_FIELDS:
        _require_sha256(evaluator_identity[field], f"evaluator {field}")


def validate_identity_bindings(
    comparison: Mapping[str, Any],
    identities: Mapping[str, Any],
    evaluator: Path,
    launch_configuration: Mapping[str, Any],
) -> None:
    _validate_declared_identity_contract(comparison, identities, launch_configuration)
    evaluator_identity = identities["evaluator"]
    if not evaluator.is_file() or not os.access(evaluator, os.X_OK):
        raise RunnerError(f"evaluator is missing or not executable: {evaluator}")
    if _sha256_file(evaluator) != evaluator_identity["binary_sha256"]:
        raise RunnerError("evaluator executable hash differs from its frozen identity")


def _assignment(comparison: Mapping[str, Any], shard_id: str) -> Mapping[str, Any]:
    for assignment in comparison["shards"]:
        if assignment["shard_id"] == shard_id:
            return assignment
    raise RunnerError(f"comparison does not declare shard {shard_id!r}")


def build_shard_command(
    evaluator: Path,
    protocol_path: Path,
    comparison: Mapping[str, Any],
    launch_configuration: Mapping[str, Any],
    shard_id: str,
) -> list[str]:
    validate_launch_configuration(launch_configuration)
    _validate_supported_style_metrics(comparison["required_style_metrics"])
    _validate_scenario_binding(comparison, launch_configuration)
    _validate_phase_work_contract(comparison, launch_configuration)
    assignment = _assignment(comparison, shard_id)
    command = [
        str(evaluator.resolve()),
        "--pairs",
        str(len(assignment["seed_indices"])),
        "--seeds-json",
        str(protocol_path.resolve()),
        "--seed-namespace",
        comparison["seed_namespace"],
    ]
    for index in assignment["seed_indices"]:
        command.extend(("--seed-index", str(index)))
    command.extend(
        (
            "--features",
            rust_feature_spec(launch_configuration),
            "--worlds",
            str(launch_configuration["worlds"]),
            "--candidates",
            str(launch_configuration["candidates"]),
            "--rollout-tricks",
            str(launch_configuration["rollout_tricks"]),
        )
    )
    for metric in comparison["required_style_metrics"]:
        command.extend(("--style-metric", metric))
    command.extend(("--scenario", launch_configuration["scenario_id"]))
    if launch_configuration["work_mode"] == "budget":
        command.extend(("--budget-ms", str(launch_configuration["budget_ms"])))
    else:
        command.extend(
            (
                "--fixed-work",
                "--deadline-ms",
                str(launch_configuration["deadline_ms"]),
            )
        )
    return command


def _verified_machine_attestation(
    comparison: Mapping[str, Any], evidence: Mapping[str, Any]
) -> Mapping[str, Any]:
    artifacts = evidence.get("artifacts")
    if not isinstance(artifacts, list):
        raise RunnerError("verified external evidence lacks artifact records")
    matches = [
        record
        for record in artifacts
        if isinstance(record, Mapping)
        and record.get("artifact_id")
        == enoch_week1_evidence.MACHINE_ATTESTATION_ARTIFACT
    ]
    if len(matches) != 1:
        raise RunnerError("verified external evidence must name one machine attestation")
    reference = matches[0]
    path_value = reference.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise RunnerError("verified machine attestation path is invalid")
    path = Path(path_value)
    if _sha256_file(path) != reference.get("file_sha256"):
        raise RunnerError("verified machine attestation file hash changed")
    try:
        attestation = enoch_week1.load_json_object(path)
        fingerprint = enoch_week1_evidence.validate_machine_contention_attestation(
            comparison, attestation
        )
    except (enoch_week1.ProtocolError, enoch_week1_evidence.EvidenceError) as exc:
        raise RunnerError(f"verified machine attestation is invalid: {exc}") from exc
    if fingerprint != reference.get("semantic_fingerprint"):
        raise RunnerError("verified machine attestation semantic fingerprint changed")
    return attestation


def probe_evaluator_environment_identity(
    *,
    evaluator: Path,
    protocol_path: Path,
    protocol: Mapping[str, Any],
    comparison: Mapping[str, Any],
    launch_configuration: Mapping[str, Any],
    child_environment: Mapping[str, str],
    evaluator_identity: Mapping[str, Any],
    available_parallelism: int,
    campaign_lock_token: AuthoritativeCampaignLockToken,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Probe and verify Rust's environment identity without running a hand."""

    _validate_campaign_lock_token(campaign_lock_token, protocol, comparison)
    evaluator = evaluator.resolve()
    if not evaluator.is_file() or not os.access(evaluator, os.X_OK):
        raise RunnerError(f"evaluator is missing or not executable: {evaluator}")
    if _sha256_file(evaluator) != evaluator_identity.get("binary_sha256"):
        raise RunnerError("environment probe executable differs from evaluator identity")
    if timeout_seconds is not None and (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or timeout_seconds <= 0
    ):
        raise RunnerError("environment probe timeout_seconds must be positive")
    expected = build_evaluator_environment_identity(
        evaluator_identity,
        protocol,
        child_environment,
        available_parallelism=available_parallelism,
    )
    expected_fingerprint = enoch_week1.canonical_json_sha256(expected)
    if comparison.get("environment_fingerprint") != expected_fingerprint:
        raise RunnerError(
            "comparison environment fingerprint differs from the reconstructed "
            "evaluator environment identity"
        )
    assignments = comparison.get("shards")
    if not isinstance(assignments, list) or not assignments:
        raise RunnerError("comparison has no shard available for environment probe")
    first_shard = assignments[0]
    if not isinstance(first_shard, Mapping) or not isinstance(
        first_shard.get("shard_id"), str
    ):
        raise RunnerError("comparison first shard is invalid")
    command = build_shard_command(
        evaluator,
        protocol_path,
        comparison,
        launch_configuration,
        first_shard["shard_id"],
    )
    command.append("--environment-identity-only")
    try:
        completed = subprocess.run(
            command,
            env=dict(child_environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RunnerError(f"evaluator environment identity probe failed: {exc}") from exc
    if completed.returncode != 0:
        raise RunnerError(
            "evaluator environment identity probe exited with status "
            f"{completed.returncode}"
        )
    if len(completed.stdout) > MAX_STDOUT_BYTES:
        raise RunnerError("evaluator environment identity probe stdout exceeds safety limit")
    try:
        payload = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunnerError(
            f"evaluator environment identity probe did not emit one JSON document: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise RunnerError("evaluator environment identity probe root must be an object")
    _require_exact_keys(
        payload,
        {"environment", "environment_identity_sha256"},
        "evaluator environment identity probe",
    )
    raw_environment = payload["environment"]
    if not isinstance(raw_environment, Mapping):
        raise RunnerError("evaluator environment identity probe contract must be an object")
    raw_fingerprint = _require_sha256(
        payload["environment_identity_sha256"],
        "evaluator environment identity probe fingerprint",
    )
    if enoch_week1.canonical_json_sha256(raw_environment) != raw_fingerprint:
        raise RunnerError("evaluator environment identity probe hash is inconsistent")
    if dict(raw_environment) != expected or raw_fingerprint != expected_fingerprint:
        differing = sorted(
            name
            for name in set(raw_environment) | set(expected)
            if raw_environment.get(name) != expected.get(name)
        )
        suffix = f": {', '.join(differing)}" if differing else ""
        raise RunnerError(
            "evaluator environment identity probe differs from the frozen contract" + suffix
        )
    return {
        "environment": dict(raw_environment),
        "environment_identity_sha256": raw_fingerprint,
    }


def _protocol_namespace(protocol: Mapping[str, Any], namespace: str) -> Mapping[str, Any]:
    for entry in protocol["seed_registry"]["namespaces"]:
        if entry["name"] == namespace:
            return entry
    raise RunnerError(f"protocol lacks namespace {namespace!r}")


def _validate_protocol_identity(
    protocol: Mapping[str, Any], comparison: Mapping[str, Any], raw: Mapping[str, Any]
) -> None:
    expected_keys = {
        "derivation_domain",
        "domain_status",
        "environment_allowlist",
        "environment_policy_verified",
        "manifest_version",
        "master_seed_u64",
        "namespace",
        "protocol_fingerprint",
        "protocol_kind",
        "registry_namespace_count",
        "seed_registry_sha256",
    }
    _require_exact_keys(raw, expected_keys, "raw protocol identity")
    namespace_entry = _protocol_namespace(protocol, comparison["seed_namespace"])
    expected = {
        "derivation_domain": enoch_week1.SEED_DERIVATION_DESCRIPTION["domain"],
        "domain_status": "verified-week1-protocol",
        "environment_allowlist": protocol["evaluator_environment_policy"]["allowlist"],
        "environment_policy_verified": True,
        "manifest_version": protocol["manifest_version"],
        "master_seed_u64": protocol["seed_registry"]["master_seed"],
        "namespace": comparison["seed_namespace"],
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "protocol_kind": protocol["protocol_kind"],
        "registry_namespace_count": namespace_entry["count"],
        "seed_registry_sha256": protocol["seed_registry_sha256"],
    }
    if dict(raw) != expected:
        raise RunnerError("evaluator protocol identity differs from the frozen protocol")


def _validate_search_identity(
    raw: Mapping[str, Any],
    launch: Mapping[str, Any],
    required_style_metrics: Sequence[str],
) -> None:
    if not isinstance(raw, Mapping):
        raise RunnerError("evaluator search identity must be an object")
    if raw.get("strict_no_fallback") is not True:
        raise RunnerError("evaluator search identity is not strict/no-fallback")
    candidate = raw.get("candidate")
    control = raw.get("control")
    work = raw.get("work")
    if not all(isinstance(value, Mapping) for value in (candidate, control, work)):
        raise RunnerError("evaluator search identity lacks candidate/control/work contracts")
    expected_features = rust_feature_spec(launch)
    candidate_features = candidate.get("features")
    control_features = control.get("features")
    if not isinstance(candidate_features, Mapping) or candidate_features.get(
        "canonical_spec"
    ) != expected_features:
        raise RunnerError("evaluator candidate feature identity differs from launch")
    if not isinstance(control_features, Mapping) or control_features.get(
        "canonical_spec"
    ) != "none":
        raise RunnerError("evaluator control is not frozen feature-empty Enoch-0")
    expected_mode = launch["work_mode"]
    expected_time = launch["budget_ms"] if expected_mode == "budget" else launch["deadline_ms"]
    checks = {
        "mode": expected_mode,
        "require_full_work": expected_mode == "fixed-work",
        "time_budget_ms": expected_time,
        "max_worlds": launch["worlds"],
        "max_candidates": launch["candidates"],
        "rollout_tricks": launch["rollout_tricks"],
    }
    for field, expected in checks.items():
        if work.get(field) != expected:
            raise RunnerError(f"evaluator search work field {field} differs from launch")
    if raw.get("runner_style_metrics") != list(required_style_metrics):
        raise RunnerError("evaluator style-metric identity differs from the comparison")


def _experiment_environment(environment: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(environment, Mapping):
        raise RunnerError("evaluator child environment must be an object")
    result: dict[str, str] = {}
    for name, value in sorted(environment.items()):
        if not isinstance(name, str) or not isinstance(value, str):
            raise RunnerError("evaluator child environment names and values must be strings")
        if name.startswith(enoch_week1.BLOCKED_EVALUATOR_ENV_PREFIXES):
            result[name] = value
    return result


def build_evaluator_environment_identity(
    evaluator_identity: Mapping[str, Any],
    protocol: Mapping[str, Any],
    child_environment: Mapping[str, str],
    *,
    available_parallelism: int,
) -> dict[str, Any]:
    """Build the exact contract emitted by Rust ``enoch_eval``.

    ``child_environment`` must be the result of
    :func:`enoch_week1.sanitized_evaluator_environment`.  Hardware
    parallelism is explicit because Rust records it at evaluator startup and
    it is therefore part of the comparison identity, not a runner default.
    """

    try:
        enoch_week1.validate_protocol(protocol)
    except enoch_week1.ProtocolError as exc:
        raise RunnerError(f"cannot build evaluator environment identity: {exc}") from exc
    if not isinstance(evaluator_identity, Mapping):
        raise RunnerError("evaluator identity must be an object")
    _require_exact_keys(
        evaluator_identity,
        set(enoch_week1.EVALUATOR_IDENTITY_FIELDS),
        "evaluator identity",
    )
    for field in enoch_week1.EVALUATOR_IDENTITY_FIELDS:
        _require_sha256(evaluator_identity[field], f"evaluator {field}")
    available_parallelism = _require_positive_int(
        available_parallelism, "available parallelism"
    )
    frozen_target = dict(EVALUATOR_CRATE_TARGET_CONTRACT)
    if (
        enoch_week1.canonical_json_sha256(frozen_target)
        != EVALUATOR_CRATE_TARGET_CONTRACT_SHA256
    ):
        raise RunnerError("frozen evaluator crate/target contract changed in process")
    _require_exact_keys(
        frozen_target,
        {
            "build_profile",
            "crate_name",
            "crate_version",
            "target_arch",
            "target_endian",
            "target_family",
            "target_os",
            "target_pointer_width",
        },
        "frozen evaluator crate/target contract",
    )
    experiment_environment = _experiment_environment(child_environment)
    allowlist = protocol["evaluator_environment_policy"]["allowlist"]
    retained_without_authority = set(experiment_environment) - set(allowlist)
    if retained_without_authority:
        raise RunnerError(
            "evaluator child environment retained non-allowlisted variables: "
            + ", ".join(sorted(retained_without_authority))
        )
    return {
        "available_parallelism": available_parallelism,
        "binary_sha256": evaluator_identity["binary_sha256"],
        "blocked_environment_prefixes": list(
            enoch_week1.BLOCKED_EVALUATOR_ENV_PREFIXES
        ),
        **frozen_target,
        "effective_experiment_environment": experiment_environment,
        "frozen_environment_allowlist": list(allowlist),
        "frozen_environment_policy_verified": True,
        "source_revision": evaluator_identity["source_sha256"],
    }


def _validate_environment_identity(
    comparison: Mapping[str, Any],
    protocol: Mapping[str, Any],
    raw: Mapping[str, Any],
    child_environment: Mapping[str, str],
    evaluator_identity: Mapping[str, Any],
    available_parallelism: int,
) -> None:
    if not isinstance(raw, Mapping):
        raise RunnerError("evaluator environment identity must be an object")
    expected = build_evaluator_environment_identity(
        evaluator_identity,
        protocol,
        child_environment,
        available_parallelism=available_parallelism,
    )
    expected_fingerprint = enoch_week1.canonical_json_sha256(expected)
    if comparison["environment_fingerprint"] != expected_fingerprint:
        raise RunnerError(
            "comparison environment fingerprint differs from the reconstructed "
            "evaluator environment identity"
        )
    if dict(raw) != expected:
        differing = sorted(
            name
            for name in set(raw) | set(expected)
            if raw.get(name) != expected.get(name)
        )
        raise RunnerError(
            "evaluator environment identity differs from the reconstructed contract: "
            + ", ".join(differing)
        )


def _validate_scenario_identity(
    comparison: Mapping[str, Any],
    launch: Mapping[str, Any],
    top: Mapping[str, Any],
    merge_identity: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not isinstance(top, Mapping):
        raise RunnerError("evaluator scenario identity must be an object")
    _require_exact_keys(
        top, {"contract", "expected_namespace", "id", "identity_sha256"}, "scenario identity"
    )
    expected_namespace = comparison["seed_namespace"] if comparison["phase"] == "W1.5" else None
    if top["id"] != launch["scenario_id"] or top["expected_namespace"] != expected_namespace:
        raise RunnerError("evaluator scenario id/namespace differs from the frozen launch")
    contract = top["contract"]
    if not isinstance(contract, Mapping):
        raise RunnerError("evaluator scenario contract must be an object")
    _require_exact_keys(
        contract,
        {
            "category",
            "expected_seed_namespace",
            "evaluator_pairs_are_one_declared_shard_subset",
            "id",
            "orientations",
            "qualification_total_pairs",
            "rules",
            "scenario_contract_version",
            "threshold_deal_selection",
        },
        "scenario contract",
    )
    if (
        contract["scenario_contract_version"] != 1
        or contract["id"] != launch["scenario_id"]
        or contract["expected_seed_namespace"] != expected_namespace
        or contract["evaluator_pairs_are_one_declared_shard_subset"] is not True
    ):
        raise RunnerError("evaluator scenario contract identity changed")
    expected_total_pairs = comparison["pair_count"] if comparison["phase"] == "W1.5" else None
    if contract["qualification_total_pairs"] != expected_total_pairs:
        raise RunnerError("scenario qualification pair total differs from the frozen comparison")
    identity_sha256 = _require_sha256(top["identity_sha256"], "scenario identity hash")
    if identity_sha256 != enoch_week1.canonical_json_sha256(contract):
        raise RunnerError("evaluator scenario contract hash mismatch")
    if merge_identity["scenario_identity_sha256"] != identity_sha256:
        raise RunnerError("merge scenario identity hash mismatch")
    if merge_identity["scenario"] != contract:
        raise RunnerError("merge scenario contract differs from top-level contract")
    search = merge_identity["search"]
    if not isinstance(search, Mapping) or search.get("scenario") != contract:
        raise RunnerError("search identity uses a different scenario contract")
    threshold = contract["threshold_deal_selection"]
    should_be_threshold = launch["scenario_id"].startswith("threshold-situation-")
    if should_be_threshold != (threshold is not None):
        raise RunnerError("scenario threshold-selection contract presence changed")
    return contract


def _validate_audit_clean(
    audit: Mapping[str, Any], label: str, *, work_mode: str
) -> None:
    if not isinstance(audit, Mapping):
        raise RunnerError(f"{label} audit must be an object")
    if audit.get("invalid_counter_total") != 0:
        raise RunnerError(f"{label} audit reports invalid/fallback counters")
    if audit.get("attribution_failures", 0) != 0:
        raise RunnerError(f"{label} audit reports attribution failures")
    for arm_name in ("candidate", "control"):
        arm = audit.get(arm_name)
        if not isinstance(arm, Mapping):
            raise RunnerError(f"{label} audit lacks {arm_name} arm evidence")
        bound_counts = arm.get("bound_counts")
        if not isinstance(bound_counts, Mapping):
            raise RunnerError(f"{label} {arm_name} audit lacks bound-count evidence")
        _require_exact_keys(
            bound_counts, {"time", "work"}, f"{label} {arm_name} bound counts"
        )
        time_bound = _require_nonnegative_int(
            bound_counts["time"], f"{label} {arm_name} time-bound count"
        )
        _require_nonnegative_int(
            bound_counts["work"], f"{label} {arm_name} work-bound count"
        )
        candidate_work = arm.get("candidate_work")
        if not isinstance(candidate_work, Mapping):
            raise RunnerError(f"{label} {arm_name} audit lacks candidate-work evidence")
        _require_exact_keys(
            candidate_work,
            {
                "candidate_pool",
                "evaluation_budget",
                "evaluations_completed",
                "initial_candidates",
            },
            f"{label} {arm_name} candidate work",
        )
        evaluation_budget = _require_nonnegative_int(
            candidate_work["evaluation_budget"],
            f"{label} {arm_name} candidate-work budget",
        )
        evaluations_completed = _require_nonnegative_int(
            candidate_work["evaluations_completed"],
            f"{label} {arm_name} completed candidate work",
        )
        for field in ("candidate_pool", "initial_candidates"):
            _require_nonnegative_int(
                candidate_work[field], f"{label} {arm_name} {field}"
            )
        if work_mode == "fixed-work" and time_bound != 0:
            raise RunnerError(
                f"{label} {arm_name} fixed-work audit reports {time_bound} "
                "time-bound decision(s); the safety deadline determined work"
            )
        if work_mode == "fixed-work" and evaluations_completed != evaluation_budget:
            raise RunnerError(
                f"{label} {arm_name} fixed-work audit completed "
                f"{evaluations_completed} of {evaluation_budget} candidate evaluations"
            )
        for field in (
            "internal_prior_fallback",
            "external_policy_fallback",
            "non_strict_decisions",
            "missing_search_telemetry",
            "decisions_without_action",
            "search_failure_count",
            "invalid_counter_total",
        ):
            if arm.get(field) != 0:
                raise RunnerError(f"{label} {arm_name} audit has nonzero {field}")


def _resolve_failure_counters(
    evidence: Mapping[str, Any], external: Mapping[str, Mapping[str, Any]], label: str
) -> dict[str, int]:
    if not isinstance(evidence, Mapping) or set(evidence) != set(
        enoch_week1.FAILURE_COUNTER_NAMES
    ):
        raise RunnerError(f"{label} failure evidence does not cover the frozen counter schema")
    counters: dict[str, int] = {}
    for name in enoch_week1.FAILURE_COUNTER_NAMES:
        item = evidence[name]
        if not isinstance(item, Mapping):
            raise RunnerError(f"{label} failure evidence {name} must be an object")
        _require_exact_keys(item, {"authority", "count"}, f"{label} failure evidence {name}")
        if not isinstance(item["authority"], str) or not item["authority"]:
            raise RunnerError(f"{label} failure evidence {name} lacks authority")
        count = item["count"]
        if name in REQUIRED_EXTERNAL_COUNTERS and name not in external:
            raise RunnerError(
                f"{label} counter {name} requires comparison-bound external evidence"
            )
        if count is None:
            if name not in external:
                raise RunnerError(f"{label} counter {name} is unknown without external evidence")
            count = external[name]["count"]
        counters[name] = _require_nonnegative_int(count, f"{label} counter {name}")
    nonzero = {name: count for name, count in counters.items() if count != 0}
    if nonzero:
        raise RunnerError(f"{label} contains invalidating evaluator counters: {nonzero}")
    return counters


def _derive_threshold_deal_seed(registry_seed: int, attempt: int) -> int:
    if attempt == 0:
        return registry_seed
    digest = hashlib.sha256()
    digest.update(b"shengji/enoch-week1/threshold-deal/v1\0")
    digest.update(registry_seed.to_bytes(8, "big"))
    digest.update(attempt.to_bytes(8, "big"))
    return int.from_bytes(digest.digest()[:8], "big")


def _validate_deal_selection(
    raw: Mapping[str, Any],
    registry_seed: int,
    scenario_contract: Mapping[str, Any],
) -> int:
    if not isinstance(raw, Mapping):
        raise RunnerError("deal selection evidence must be an object")
    _require_exact_keys(
        raw,
        {
            "attempts_examined",
            "selection_attempt_zero_based",
            "selector_non_landlord_points",
            "status",
        },
        "deal selection evidence",
    )
    attempt = _require_nonnegative_int(
        raw["selection_attempt_zero_based"], "deal selection attempt"
    )
    if raw["attempts_examined"] != attempt + 1:
        raise RunnerError("deal selection attempt count is inconsistent")
    threshold = scenario_contract["threshold_deal_selection"]
    if threshold is None:
        if (
            raw["status"] != "direct-registry-seed"
            or attempt != 0
            or raw["selector_non_landlord_points"] is not None
        ):
            raise RunnerError("non-threshold scenario did not use its registry seed directly")
        return registry_seed
    if raw["status"] != "candidate-independent-control-selected":
        raise RunnerError("threshold scenario lacks candidate-independent selection evidence")
    if not isinstance(threshold, Mapping):
        raise RunnerError("threshold scenario selection contract must be an object")
    _require_exact_keys(
        threshold,
        {
            "band",
            "candidate_independent",
            "derived_seed_domain",
            "maximum_attempts",
            "registry_seed_is_attempt_zero",
            "selector",
        },
        "threshold scenario selection contract",
    )
    if (
        threshold["selector"] != "homogeneous-searchless-Enoch-0-control-v1"
        or threshold["candidate_independent"] is not True
        or threshold["registry_seed_is_attempt_zero"] is not True
        or threshold["derived_seed_domain"]
        != "shengji/enoch-week1/threshold-deal/v1"
    ):
        raise RunnerError("threshold scenario selection is not frozen candidate-independent control")
    maximum_attempts = _require_positive_int(
        threshold.get("maximum_attempts"), "threshold maximum attempts"
    )
    if attempt >= maximum_attempts:
        raise RunnerError("threshold deal selection exceeded its frozen attempt bound")
    points = _require_nonnegative_int(
        raw["selector_non_landlord_points"], "threshold selector points"
    )
    band = threshold.get("band")
    if not isinstance(band, Mapping):
        raise RunnerError("threshold scenario lacks its score band")
    minimum = _require_nonnegative_int(
        band.get("attacker_points_minimum_inclusive"), "threshold minimum"
    )
    maximum = band.get("attacker_points_maximum_exclusive")
    if maximum is not None:
        maximum = _require_nonnegative_int(maximum, "threshold maximum")
    if points < minimum or (maximum is not None and points >= maximum):
        raise RunnerError("threshold selector outcome is outside its frozen score band")
    return _derive_threshold_deal_seed(registry_seed, attempt)


def _translate_pair_record(
    protocol: Mapping[str, Any],
    comparison: Mapping[str, Any],
    raw: Mapping[str, Any],
    expected_local_index: int,
    expected_registry_index: int,
    external: Mapping[str, Mapping[str, Any]],
    scenario_contract: Mapping[str, Any],
    work_mode: str,
) -> tuple[dict[str, Any], int]:
    expected_keys = {
        "audit",
        "candidate_level_utility",
        "candidate_point_margin",
        "candidate_win_rate",
        "candidate_wins",
        "complete",
        "control_wins",
        "deal_selection",
        "effective_deal_seed_hex",
        "effective_deal_seed_u64",
        "hands_completed",
        "hands_failed",
        "orientations",
        "paired_index",
        "registry_index",
        "registry_seed_hex",
        "registry_seed_u64",
        "runner_record_inputs",
        "seed_hex",
        "seed_u64",
    }
    _require_exact_keys(raw, expected_keys, "raw paired record")
    namespace = _protocol_namespace(protocol, comparison["seed_namespace"])
    expected_seed = namespace["seeds"][expected_registry_index]
    if raw["paired_index"] != expected_local_index:
        raise RunnerError("raw paired record local index mismatch")
    if raw["registry_index"] != expected_registry_index or raw["seed_u64"] != expected_seed:
        raise RunnerError("raw paired record registry index/seed mismatch")
    if raw["seed_hex"] != f"0x{expected_seed:016x}":
        raise RunnerError("raw paired record seed hex mismatch")
    if (
        raw["registry_seed_u64"] != expected_seed
        or raw["registry_seed_hex"] != f"0x{expected_seed:016x}"
    ):
        raise RunnerError("raw paired record registry seed identity mismatch")
    effective_seed = _validate_deal_selection(
        raw["deal_selection"], expected_seed, scenario_contract
    )
    if (
        raw["effective_deal_seed_u64"] != effective_seed
        or raw["effective_deal_seed_hex"] != f"0x{effective_seed:016x}"
    ):
        raise RunnerError("raw paired record effective deal seed mismatch")
    if raw["complete"] is not True or raw["hands_completed"] != 2 or raw["hands_failed"] != 0:
        raise RunnerError("raw paired record is not a complete two-orientation pair")
    candidate_wins = _require_nonnegative_int(raw["candidate_wins"], "candidate wins")
    control_wins = _require_nonnegative_int(raw["control_wins"], "control wins")
    if candidate_wins + control_wins != 2:
        raise RunnerError("raw paired win counts do not cover both orientations")
    orientations = raw["orientations"]
    if not isinstance(orientations, list) or len(orientations) != 2:
        raise RunnerError("raw paired record must retain exactly two orientations")
    if any(not isinstance(item, Mapping) or item.get("complete") is not True for item in orientations):
        raise RunnerError("raw paired record contains an incomplete orientation")
    roles = [item.get("candidate_is_landlord_team") for item in orientations]
    if any(not isinstance(role, bool) for role in roles) or roles.count(False) != 1 or roles.count(True) != 1:
        raise RunnerError("raw paired orientations do not cover both candidate roles")
    orientation_wins = sum(item.get("candidate_won") is True for item in orientations)
    if orientation_wins != candidate_wins:
        raise RunnerError("raw paired win count disagrees with orientation outcomes")
    _validate_audit_clean(
        raw["audit"], f"pair {expected_registry_index}", work_mode=work_mode
    )

    inputs = raw["runner_record_inputs"]
    if not isinstance(inputs, Mapping):
        raise RunnerError("raw paired record lacks runner_record_inputs")
    _require_exact_keys(
        inputs,
        {
            "candidate_completed_worlds",
            "candidate_latency_ms",
            "control_completed_worlds",
            "control_latency_ms",
            "failure_counter_evidence",
            "level_utility_delta",
            "point_margin_delta",
            "seed",
            "seed_index",
            "style_metrics",
            "win_rate_delta",
        },
        "runner record inputs",
    )
    if inputs["seed_index"] != expected_registry_index or inputs["seed"] != expected_seed:
        raise RunnerError("runner record inputs seed identity mismatch")
    level = _require_finite(inputs["level_utility_delta"], "level utility delta")
    margin = _require_finite(inputs["point_margin_delta"], "point margin delta")
    win = _require_finite(inputs["win_rate_delta"], "win-rate delta")
    if level != _require_finite(raw["candidate_level_utility"], "candidate level utility"):
        raise RunnerError("runner level-utility delta disagrees with raw outcome")
    if margin != _require_finite(raw["candidate_point_margin"], "candidate point margin"):
        raise RunnerError("runner point-margin delta disagrees with raw outcome")
    if win != _require_finite(raw["candidate_win_rate"], "candidate win rate") - 0.5:
        raise RunnerError("runner win-rate delta disagrees with raw outcome")
    if float(raw["candidate_win_rate"]) != candidate_wins / 2.0:
        raise RunnerError("raw candidate win rate disagrees with win counts")
    style = inputs["style_metrics"]
    if not isinstance(style, Mapping) or set(style) != set(comparison["required_style_metrics"]):
        raise RunnerError("runner style metrics do not exactly match the comparison declaration")
    for name, value in style.items():
        _require_finite(value, f"style metric {name}")
    counters = _resolve_failure_counters(
        inputs["failure_counter_evidence"], external, f"pair {expected_registry_index}"
    )
    record = {
        "candidate_completed_worlds": _require_nonnegative_int(
            inputs["candidate_completed_worlds"], "candidate completed worlds"
        ),
        "candidate_latency_ms": _require_finite(
            inputs["candidate_latency_ms"], "candidate latency", nonnegative=True
        ),
        "complete": True,
        "control_completed_worlds": _require_nonnegative_int(
            inputs["control_completed_worlds"], "control completed worlds"
        ),
        "control_latency_ms": _require_finite(
            inputs["control_latency_ms"], "control latency", nonnegative=True
        ),
        "effective_deal_seed": effective_seed,
        "failure_counters": counters,
        "level_utility_delta": level,
        "orientations_completed": 2,
        "point_margin_delta": margin,
        "seed": expected_seed,
        "seed_index": expected_registry_index,
        "style_metrics": dict(style),
        "win_rate_delta": win,
    }
    return record, effective_seed


def _validate_top_candidate_and_settings(
    raw: Mapping[str, Any],
    comparison: Mapping[str, Any],
    launch: Mapping[str, Any],
) -> None:
    candidate = raw["candidate"]
    control = raw["control"]
    settings = raw["settings"]
    if not all(isinstance(value, Mapping) for value in (candidate, control, settings)):
        raise RunnerError("evaluator candidate/control/settings contracts must be objects")
    feature_spec = rust_feature_spec(launch)
    if candidate.get("label") != f"Enoch(candidate:{feature_spec})":
        raise RunnerError("top-level candidate label differs from launch")
    if candidate.get("feature_input") != feature_spec:
        raise RunnerError("top-level candidate feature input differs from launch")
    if not isinstance(candidate.get("features"), Mapping) or candidate["features"].get(
        "canonical_spec"
    ) != feature_spec:
        raise RunnerError("top-level candidate feature identity differs from launch")
    if not isinstance(control.get("features"), Mapping) or control["features"].get(
        "canonical_spec"
    ) != "none":
        raise RunnerError("top-level control is not feature-empty Enoch-0")
    if control.get("label") != "Enoch-0":
        raise RunnerError("top-level control label changed")
    expected_settings = {
        "mode": launch["work_mode"],
        "require_full_work": launch["work_mode"] == "fixed-work",
        "time_budget_ms": (
            launch["budget_ms"]
            if launch["work_mode"] == "budget"
            else launch["deadline_ms"]
        ),
        "max_worlds": launch["worlds"],
        "max_candidates": launch["candidates"],
        "rollout_tricks": launch["rollout_tricks"],
        "runner_style_metrics": comparison["required_style_metrics"],
        "scenario_id": launch["scenario_id"],
    }
    for field, expected in expected_settings.items():
        if settings.get(field) != expected:
            raise RunnerError(f"top-level evaluator setting {field} differs from launch")


def _validate_per_deck_and_metrics(
    raw: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    effective_seeds: Sequence[int],
) -> None:
    per_deck = raw["per_deck"]
    if not isinstance(per_deck, Mapping):
        raise RunnerError("evaluator per_deck evidence must be an object")
    _require_exact_keys(
        per_deck,
        {
            "candidate_level_utility",
            "candidate_point_margin",
            "candidate_win_rate",
            "complete_registry_index",
            "complete_effective_deal_seed_hex",
            "complete_effective_deal_seed_u64",
            "complete_seed_hex",
            "complete_seed_u64",
        },
        "evaluator per_deck evidence",
    )
    expected = {
        "complete_registry_index": list(indices),
        "complete_seed_u64": [record["seed"] for record in records],
        "complete_seed_hex": [f"0x{record['seed']:016x}" for record in records],
        "complete_effective_deal_seed_u64": list(effective_seeds),
        "complete_effective_deal_seed_hex": [
            f"0x{seed:016x}" for seed in effective_seeds
        ],
        "candidate_win_rate": [record["win_rate_delta"] + 0.5 for record in records],
        "candidate_point_margin": [record["point_margin_delta"] for record in records],
        "candidate_level_utility": [record["level_utility_delta"] for record in records],
    }
    if dict(per_deck) != expected:
        raise RunnerError("evaluator per_deck arrays do not reconstruct from paired records")
    metrics = raw["metrics"]
    if not isinstance(metrics, Mapping) or set(metrics) != {
        "win_rate",
        "point_margin",
        "level_utility",
    }:
        raise RunnerError("evaluator aggregate metric schema changed")
    values = {
        "win_rate": expected["candidate_win_rate"],
        "point_margin": expected["candidate_point_margin"],
        "level_utility": expected["candidate_level_utility"],
    }
    for name, observations in values.items():
        metric = metrics[name]
        if not isinstance(metric, Mapping):
            raise RunnerError(f"evaluator metric {name} must be an object")
        estimate = math.fsum(observations) / len(observations)
        observed_estimate = metric.get("estimate")
        if (
            metric.get("paired_observations") != len(observations)
            or not isinstance(observed_estimate, (int, float))
            or not math.isclose(float(observed_estimate), estimate, rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise RunnerError(f"evaluator metric {name} does not match paired records")


def _validate_deal_selection_manifest(
    raw: Mapping[str, Any],
    merge_identity: Mapping[str, Any],
    launch: Mapping[str, Any],
    paired: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    records: Sequence[Mapping[str, Any]],
) -> None:
    manifest = raw["deal_selection"]
    if not isinstance(manifest, Mapping):
        raise RunnerError("top-level deal selection manifest must be an object")
    _require_exact_keys(
        manifest, {"contract", "records", "records_sha256", "scenario_id"}, "deal selection manifest"
    )
    threshold = launch["scenario_id"].startswith("threshold-situation-")
    expected_contract = (
        "candidate-independent-threshold-control-selection-v1"
        if threshold
        else "direct-registry-seed-v1"
    )
    if manifest["contract"] != expected_contract or manifest["scenario_id"] != launch[
        "scenario_id"
    ]:
        raise RunnerError("deal selection manifest contract/scenario mismatch")
    expected_records = [
        {
            "effective_deal_seed_hex": f"0x{record['effective_deal_seed']:016x}",
            "effective_deal_seed_u64": record["effective_deal_seed"],
            "paired_index": local,
            "registry_index": registry_index,
            "registry_seed_hex": f"0x{record['seed']:016x}",
            "registry_seed_u64": record["seed"],
            "selection_attempt_zero_based": paired[local]["deal_selection"][
                "selection_attempt_zero_based"
            ],
            "selector_non_landlord_points": paired[local]["deal_selection"][
                "selector_non_landlord_points"
            ],
        }
        for local, (registry_index, record) in enumerate(zip(indices, records))
    ]
    if manifest["records"] != expected_records:
        raise RunnerError("top-level deal selection records differ from paired evidence")
    expected_hash = enoch_week1.canonical_json_sha256(expected_records)
    if manifest["records_sha256"] != expected_hash:
        raise RunnerError("top-level deal selection record hash mismatch")
    if merge_identity["ordered_effective_deal_records_sha256"] != expected_hash:
        raise RunnerError("merge effective-deal record hash mismatch")


def translate_evaluator_output(
    protocol: Mapping[str, Any],
    comparison: Mapping[str, Any],
    shard_id: str,
    raw: Mapping[str, Any],
    launch_configuration: Mapping[str, Any],
    identities: Mapping[str, Any],
    child_environment: Mapping[str, str],
    external_evidence: Mapping[str, Any] | None = None,
    *,
    available_parallelism: int,
) -> dict[str, Any]:
    """Validate raw evaluator JSON and construct a sealed shard result."""

    enoch_week1.validate_protocol(protocol)
    enoch_week1.validate_comparison_protocol_manifest(protocol, comparison)
    _validate_phase_environment_contract(comparison, protocol)
    _validate_supported_style_metrics(comparison["required_style_metrics"])
    validate_launch_configuration(launch_configuration)
    _validate_scenario_binding(comparison, launch_configuration)
    _validate_phase_work_contract(comparison, launch_configuration)
    _validate_declared_identity_contract(comparison, identities, launch_configuration)
    external = validate_external_failure_evidence(
        protocol, comparison, external_evidence
    )
    if external_evidence is None:
        raise RunnerError("verified external evidence is required for every shard result")
    external_evidence_fingerprint = _require_sha256(
        external_evidence.get("verified_external_evidence_fingerprint"),
        "verified external evidence fingerprint",
    )
    assignment = _assignment(comparison, shard_id)
    if not isinstance(raw, Mapping):
        raise RunnerError("evaluator stdout JSON root must be an object")
    expected_top = {
        "audit",
        "candidate",
        "completion",
        "control",
        "deal_selection",
        "evaluator",
        "manifest_version",
        "merge_identity",
        "method",
        "metrics",
        "paired_records",
        "per_deck",
        "scenario",
        "seed_consumption",
        "settings",
        "valid",
    }
    _require_exact_keys(raw, expected_top, "evaluator output")
    if raw["manifest_version"] != 4 or raw["evaluator"] != "enoch-eval-v4-scenarios":
        raise RunnerError("unsupported evaluator output version/identity")
    if (
        raw["method"]
        != "direct in-process audited configured-subject mirrored-deal pairs"
    ):
        raise RunnerError("evaluator method identity changed")
    if raw["valid"] is not True:
        raise RunnerError("evaluator marked the shard invalid")
    merge_identity = raw["merge_identity"]
    if not isinstance(merge_identity, Mapping):
        raise RunnerError("evaluator merge identity must be an object")
    _require_exact_keys(
        merge_identity,
        {
            "environment",
            "environment_identity_sha256",
            "ordered_shard_seed_records_sha256",
            "ordered_effective_deal_records_sha256",
            "protocol",
            "protocol_compatibility_sha256",
            "scenario",
            "scenario_identity_sha256",
            "schema",
            "search",
            "search_identity_sha256",
            "merge_safe_seed_domain",
        },
        "evaluator merge identity",
    )
    if merge_identity["schema"] != "enoch-eval-deterministic-shard-merge-v3-scenarios":
        raise RunnerError("evaluator merge schema changed")
    if merge_identity["merge_safe_seed_domain"] is not True:
        raise RunnerError("evaluator did not verify a merge-safe seed domain")
    _validate_protocol_identity(protocol, comparison, merge_identity["protocol"])
    scenario_contract = _validate_scenario_identity(
        comparison, launch_configuration, raw["scenario"], merge_identity
    )
    compatibility_contract = {
        "scenario": scenario_contract,
        "seed_protocol": merge_identity["protocol"],
    }
    if enoch_week1.canonical_json_sha256(compatibility_contract) != merge_identity[
        "protocol_compatibility_sha256"
    ]:
        raise RunnerError("evaluator protocol/scenario compatibility hash mismatch")
    for value_field, hash_field in (
        ("search", "search_identity_sha256"),
        ("environment", "environment_identity_sha256"),
    ):
        if enoch_week1.canonical_json_sha256(merge_identity[value_field]) != merge_identity[
            hash_field
        ]:
            raise RunnerError(f"evaluator {value_field} identity hash mismatch")
    _validate_search_identity(
        merge_identity["search"],
        launch_configuration,
        comparison["required_style_metrics"],
    )
    _validate_top_candidate_and_settings(raw, comparison, launch_configuration)
    _validate_environment_identity(
        comparison,
        protocol,
        merge_identity["environment"],
        child_environment,
        identities["evaluator"],
        available_parallelism,
    )
    if merge_identity["environment_identity_sha256"] != comparison["environment_fingerprint"]:
        raise RunnerError("evaluator environment identity is not the comparison identity")

    consumption = raw["seed_consumption"]
    if not isinstance(consumption, Mapping):
        raise RunnerError("seed consumption must be an object")
    _require_exact_keys(
        consumption,
        {
            "ordered_seed_records_sha256",
            "pairs_requested",
            "records",
            "source_kind",
            "source_path",
        },
        "seed consumption",
    )
    if consumption["source_kind"] != "week1-protocol-registry":
        raise RunnerError("evaluator did not consume the full frozen Week-1 protocol")
    indices = assignment["seed_indices"]
    namespace = _protocol_namespace(protocol, comparison["seed_namespace"])
    expected_seed_records = [
        {
            "paired_index": local,
            "registry_index": index,
            "seed_hex": f"0x{namespace['seeds'][index]:016x}",
            "seed_u64": namespace["seeds"][index],
        }
        for local, index in enumerate(indices)
    ]
    if consumption["pairs_requested"] != len(indices) or consumption["records"] != expected_seed_records:
        raise RunnerError("evaluator seed consumption differs from the declared shard")
    expected_seed_hash = enoch_week1.canonical_json_sha256(expected_seed_records)
    if consumption["ordered_seed_records_sha256"] != expected_seed_hash:
        raise RunnerError("evaluator seed consumption hash mismatch")
    if merge_identity["ordered_shard_seed_records_sha256"] != expected_seed_hash:
        raise RunnerError("evaluator merge seed hash mismatch")

    completion = raw["completion"]
    if not isinstance(completion, Mapping):
        raise RunnerError("evaluator completion must be an object")
    required_completion = {
        "pairs_requested": len(indices),
        "pairs_complete": len(indices),
        "pairs_incomplete": 0,
        "hands_expected": 2 * len(indices),
        "hands_completed": 2 * len(indices),
        "hands_failed": 0,
    }
    for field, expected in required_completion.items():
        if completion.get(field) != expected:
            raise RunnerError(f"evaluator completion mismatch for {field}")
    if completion.get("audit_invalid_counter_total") != 0:
        raise RunnerError("evaluator completion reports invalid audit counters")
    _validate_audit_clean(
        raw["audit"], "shard", work_mode=launch_configuration["work_mode"]
    )
    paired = raw["paired_records"]
    if not isinstance(paired, list) or len(paired) != len(indices):
        raise RunnerError("evaluator paired record set is incomplete")
    translated = [
        _translate_pair_record(
            protocol,
            comparison,
            record,
            local,
            registry_index,
            external,
            scenario_contract,
            launch_configuration["work_mode"],
        )
        for local, (registry_index, record) in enumerate(zip(indices, paired))
    ]
    records = [record for record, _ in translated]
    effective_seeds = [seed for _, seed in translated]
    if len(effective_seeds) != len(set(effective_seeds)):
        raise RunnerError("evaluator scenario produced duplicate effective deal seeds")
    raw_candidate_wins = sum(record["candidate_wins"] for record in paired)
    raw_control_wins = sum(record["control_wins"] for record in paired)
    if completion.get("candidate_wins") != raw_candidate_wins or completion.get(
        "control_wins"
    ) != raw_control_wins:
        raise RunnerError("evaluator completion win totals disagree with raw pairs")
    _validate_per_deck_and_metrics(raw, records, indices, effective_seeds)
    _validate_deal_selection_manifest(
        raw,
        merge_identity,
        launch_configuration,
        paired,
        indices,
        records,
    )
    return enoch_week1.build_shard_result(
        protocol,
        comparison,
        shard_id,
        records,
        verified_external_evidence_fingerprint=external_evidence_fingerprint,
    )


def _move_artifact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise RunnerError(f"refusing to overwrite execution artifact: {destination}")
    os.replace(source, destination)


_CAMPAIGN_LOCK_TOKEN_SECRET = object()


class AuthoritativeCampaignLockToken:
    """Opaque proof that this process currently owns one campaign lock."""

    __slots__ = (
        "_active",
        "_comparison_fingerprint",
        "_owner_pid",
        "_phase",
        "_protocol_fingerprint",
    )

    def __init__(
        self,
        *,
        secret: object,
        protocol_fingerprint: str,
        comparison_fingerprint: str,
        phase: str,
    ) -> None:
        if secret is not _CAMPAIGN_LOCK_TOKEN_SECRET:
            raise RunnerError("campaign lock tokens can only be issued by the runner")
        self._active = True
        self._comparison_fingerprint = comparison_fingerprint
        self._owner_pid = os.getpid()
        self._phase = phase
        self._protocol_fingerprint = protocol_fingerprint


def _validate_campaign_lock_token(
    token: AuthoritativeCampaignLockToken,
    protocol: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> None:
    if not isinstance(token, AuthoritativeCampaignLockToken) or not token._active:
        raise RunnerError("campaign lock token is absent, invalid, or no longer active")
    if token._owner_pid != os.getpid():
        raise RunnerError("campaign lock token belongs to another process")
    if (
        token._protocol_fingerprint != protocol.get("protocol_fingerprint")
        or token._comparison_fingerprint
        != comparison.get("comparison_protocol_fingerprint")
        or token._phase != comparison.get("phase")
    ):
        raise RunnerError("campaign lock token belongs to another comparison")


@contextlib.contextmanager
def authoritative_campaign_lock(
    protocol: Mapping[str, Any], comparison: Mapping[str, Any]
) -> Iterable[AuthoritativeCampaignLockToken]:
    try:
        protocol_fingerprint = enoch_week1.validate_protocol(protocol)
        enoch_week1.validate_comparison_protocol_manifest(protocol, comparison)
    except enoch_week1.ProtocolError as exc:
        raise RunnerError(f"cannot lock invalid authoritative comparison: {exc}") from exc
    phase = comparison.get("phase")
    if phase not in AUTHORITATIVE_PHASES:
        raise RunnerError(f"phase {phase!r} is not an authoritative Week-1 campaign")
    if fcntl is None:
        raise RunnerError("authoritative campaigns require Unix advisory locking")
    comparison_fingerprint = _require_sha256(
        comparison.get("comparison_protocol_fingerprint"),
        "campaign lock comparison fingerprint",
    )
    # All authoritative product comparisons serialize machine-wide, including
    # comparisons rooted in different protocols or copied run directories.
    lock_path = Path("/tmp").resolve() / "enoch-week1-authoritative-campaign.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunnerError("another authoritative comparison campaign is active") from exc
        token = AuthoritativeCampaignLockToken(
            secret=_CAMPAIGN_LOCK_TOKEN_SECRET,
            protocol_fingerprint=protocol_fingerprint,
            comparison_fingerprint=comparison_fingerprint,
            phase=phase,
        )
        try:
            yield token
        finally:
            token._active = False
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _execute_shard(
    *,
    protocol: Mapping[str, Any],
    comparison: Mapping[str, Any],
    protocol_path: Path,
    evaluator: Path,
    launch_configuration: Mapping[str, Any],
    identities: Mapping[str, Any],
    external_evidence: Mapping[str, Any] | None,
    ledger_path: Path,
    output_dir: Path,
    child_environment: Mapping[str, str],
    available_parallelism: int,
    campaign_lock_token: AuthoritativeCampaignLockToken,
    shard_id: str,
    timeout_seconds: float | None,
) -> tuple[list[str], dict[str, Any], Path]:
    assignment = _assignment(comparison, shard_id)
    command = build_shard_command(
        evaluator, protocol_path, comparison, launch_configuration, shard_id
    )
    stdout_fd, stdout_name = tempfile.mkstemp(
        dir=output_dir, prefix=f".{shard_id}.", suffix=".stdout.tmp"
    )
    stderr_fd, stderr_name = tempfile.mkstemp(
        dir=output_dir, prefix=f".{shard_id}.", suffix=".stderr.tmp"
    )
    stdout_temp = Path(stdout_name)
    stderr_temp = Path(stderr_name)
    try:
        _validate_campaign_lock_token(campaign_lock_token, protocol, comparison)
        claims = [
            (
                comparison["seed_namespace"],
                index,
                f"runner:{comparison['comparison_protocol_fingerprint'][:16]}:{shard_id}",
            )
            for index in assignment["seed_indices"]
        ]
        # This is deliberately the final stateful operation before Popen.
        enoch_week1.consume_seed_batch_once(ledger_path, protocol, claims)
        with os.fdopen(stdout_fd, "wb") as stdout_file, os.fdopen(
            stderr_fd, "wb"
        ) as stderr_file:
            stdout_fd = -1
            stderr_fd = -1
            try:
                completed = subprocess.run(
                    command,
                    env=dict(child_environment),
                    stdout=stdout_file,
                    stderr=stderr_file,
                    check=False,
                    timeout=timeout_seconds,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                stdout_file.flush()
                stderr_file.flush()
                os.fsync(stdout_file.fileno())
                os.fsync(stderr_file.fileno())
                _move_artifact(
                    stdout_temp, output_dir / "failures" / f"{shard_id}.stdout"
                )
                _move_artifact(
                    stderr_temp, output_dir / "failures" / f"{shard_id}.stderr"
                )
                raise RunnerError(f"failed to execute {shard_id}: {exc}") from exc
            stdout_file.flush()
            stderr_file.flush()
            os.fsync(stdout_file.fileno())
            os.fsync(stderr_file.fileno())
        if completed.returncode != 0:
            _move_artifact(stdout_temp, output_dir / "failures" / f"{shard_id}.stdout")
            _move_artifact(stderr_temp, output_dir / "failures" / f"{shard_id}.stderr")
            raise RunnerError(f"evaluator {shard_id} exited with status {completed.returncode}")
        if stdout_temp.stat().st_size > MAX_STDOUT_BYTES:
            _move_artifact(stdout_temp, output_dir / "failures" / f"{shard_id}.stdout")
            _move_artifact(stderr_temp, output_dir / "failures" / f"{shard_id}.stderr")
            raise RunnerError(f"evaluator {shard_id} stdout exceeds safety limit")
        try:
            with stdout_temp.open("r", encoding="utf-8") as source:
                raw = json.load(source)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            _move_artifact(stdout_temp, output_dir / "failures" / f"{shard_id}.stdout")
            _move_artifact(stderr_temp, output_dir / "failures" / f"{shard_id}.stderr")
            raise RunnerError(f"evaluator {shard_id} stdout is not one JSON document: {exc}") from exc
        try:
            shard_result = translate_evaluator_output(
                protocol,
                comparison,
                shard_id,
                raw,
                launch_configuration,
                identities,
                child_environment,
                external_evidence,
                available_parallelism=available_parallelism,
            )
        except BaseException:
            _move_artifact(stdout_temp, output_dir / "failures" / f"{shard_id}.stdout")
            _move_artifact(stderr_temp, output_dir / "failures" / f"{shard_id}.stderr")
            raise
        raw_path = output_dir / f"{shard_id}.raw.json"
        stderr_path = output_dir / f"{shard_id}.stderr.txt"
        result_path = output_dir / f"{shard_id}.result.json"
        _move_artifact(stdout_temp, raw_path)
        _move_artifact(stderr_temp, stderr_path)
        enoch_week1.atomic_write_json(result_path, shard_result)
        return command, shard_result, result_path
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            if stdout_fd >= 0:
                os.close(stdout_fd)
            stdout_temp.unlink()
        with contextlib.suppress(FileNotFoundError):
            if stderr_fd >= 0:
                os.close(stderr_fd)
            stderr_temp.unlink()
        raise


def run_comparison(
    *,
    protocol_path: Path,
    comparison_path: Path,
    launch_configuration_path: Path,
    identities_path: Path,
    evaluator: Path,
    ledger_path: Path,
    output_dir: Path,
    external_evidence_path: Path | None = None,
    workers: int = DEFAULT_WORKERS,
    merge: bool = False,
    dry_run: bool = False,
    timeout_seconds: float | None = None,
    base_environment: Mapping[str, str] | None = None,
    environment_overrides: Mapping[str, str] | None = None,
    available_parallelism: int,
    campaign_lock_token: AuthoritativeCampaignLockToken | None = None,
) -> dict[str, Any]:
    """Plan or execute every declared shard for exactly one comparison."""

    protocol = enoch_week1.load_json_object(protocol_path)
    comparison = enoch_week1.load_json_object(comparison_path)
    launch = enoch_week1.load_json_object(launch_configuration_path)
    identities = enoch_week1.load_json_object(identities_path)
    external = (
        enoch_week1.load_json_object(external_evidence_path)
        if external_evidence_path is not None
        else None
    )
    enoch_week1.validate_protocol(protocol)
    enoch_week1.validate_comparison_protocol_manifest(protocol, comparison)
    _validate_phase_environment_contract(comparison, protocol)
    if comparison["phase"] in AUTHORITATIVE_PHASES:
        if campaign_lock_token is None:
            raise RunnerError(
                "authoritative execution requires a live campaign lock token"
            )
        _validate_campaign_lock_token(campaign_lock_token, protocol, comparison)
    launch_hash = validate_launch_configuration(launch)
    if launch_hash != comparison["configuration_fingerprint"]:
        raise RunnerError("launch configuration hash differs from comparison configuration")
    if comparison["phase"] in {"W1.2", "W1.3"} and launch["candidate_arm_ids"] != [
        comparison["subject_id"]
    ]:
        raise RunnerError("independent arm comparison must enable exactly its named arm")
    evaluator = evaluator.resolve()
    validate_identity_bindings(comparison, identities, evaluator, launch)
    external_counters = validate_external_failure_evidence(
        protocol, comparison, external
    )
    missing_external = REQUIRED_EXTERNAL_COUNTERS - set(external_counters)
    if missing_external:
        raise RunnerError(
            "comparison lacks required external failure evidence for "
            + ", ".join(sorted(missing_external))
        )
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= MAX_WORKERS:
        raise RunnerError(f"workers must be between 1 and {MAX_WORKERS}")
    if timeout_seconds is not None and (
        not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0
    ):
        raise RunnerError("timeout_seconds must be positive")
    available_parallelism = _require_positive_int(
        available_parallelism, "available parallelism"
    )
    if external is None:
        raise RunnerError("comparison lacks verified machine attestation evidence")
    machine_attestation = _verified_machine_attestation(comparison, external)
    if machine_attestation["available_parallelism"] != available_parallelism:
        raise RunnerError(
            "declared available parallelism differs from verified machine attestation"
        )
    if machine_attestation["worker_count"] != workers:
        raise RunnerError("declared worker count differs from verified machine attestation")
    child_environment, _ = enoch_week1.sanitized_evaluator_environment(
        base_environment,
        allowlist=protocol["evaluator_environment_policy"]["allowlist"],
        overrides=environment_overrides,
    )
    expected_environment_identity = build_evaluator_environment_identity(
        identities["evaluator"],
        protocol,
        child_environment,
        available_parallelism=available_parallelism,
    )
    expected_environment_fingerprint = enoch_week1.canonical_json_sha256(
        expected_environment_identity
    )
    if comparison["environment_fingerprint"] != expected_environment_fingerprint:
        raise RunnerError(
            "comparison environment fingerprint differs from the reconstructed "
            "evaluator environment identity"
        )
    environment_probe = probe_evaluator_environment_identity(
        evaluator=evaluator,
        protocol_path=protocol_path,
        protocol=protocol,
        comparison=comparison,
        launch_configuration=launch,
        child_environment=child_environment,
        evaluator_identity=identities["evaluator"],
        available_parallelism=available_parallelism,
        campaign_lock_token=campaign_lock_token,
        timeout_seconds=timeout_seconds,
    )
    commands = [
        build_shard_command(
            evaluator,
            protocol_path,
            comparison,
            launch,
            assignment["shard_id"],
        )
        for assignment in comparison["shards"]
    ]
    plan = {
        "arm_feature_mapping_sha256": ARM_TO_RUST_FEATURE_SHA256,
        "available_parallelism": available_parallelism,
        "commands": commands,
        "comparison_protocol_fingerprint": comparison[
            "comparison_protocol_fingerprint"
        ],
        "dry_run": dry_run,
        "external_failure_evidence_fingerprint": (
            external["verified_external_evidence_fingerprint"]
            if external is not None
            else None
        ),
        "launch_configuration_sha256": launch_hash,
        "evaluator_environment_identity_sha256": expected_environment_fingerprint,
        "environment_identity_probe": environment_probe,
        "manifest_kind": "enoch-week1-runner-execution",
        "manifest_version": RUNNER_MANIFEST_VERSION,
        "qualification_scenario_bindings_sha256": QUALIFICATION_SCENARIO_BINDINGS_SHA256,
        "rust_style_metrics_sha256": RUST_STYLE_METRICS_SHA256,
        "worker_limit": min(workers, len(commands)),
    }
    if dry_run:
        return plan
    output_dir.mkdir(parents=True, exist_ok=False)
    enoch_week1.atomic_write_json(output_dir / "comparison.json", comparison)
    enoch_week1.atomic_write_json(output_dir / "launch-configuration.json", launch)
    enoch_week1.atomic_write_json(output_dir / "identity-bindings.json", identities)
    if external is not None:
        enoch_week1.atomic_write_json(
            output_dir / "external-failure-evidence.json", external
        )
    results: dict[str, dict[str, Any]] = {}
    paths: dict[str, str] = {}
    errors: list[str] = []
    if campaign_lock_token is not None:
        _validate_campaign_lock_token(campaign_lock_token, protocol, comparison)
    with contextlib.nullcontext(campaign_lock_token):
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(workers, len(commands)), thread_name_prefix="enoch-week1"
        ) as executor:
            futures = {
                executor.submit(
                    _execute_shard,
                    protocol=protocol,
                    comparison=comparison,
                    protocol_path=protocol_path,
                    evaluator=evaluator,
                    launch_configuration=launch,
                    identities=identities,
                    external_evidence=external,
                    ledger_path=ledger_path,
                    output_dir=output_dir,
                    child_environment=child_environment,
                    available_parallelism=available_parallelism,
                    campaign_lock_token=campaign_lock_token,
                    shard_id=assignment["shard_id"],
                    timeout_seconds=timeout_seconds,
                ): assignment["shard_id"]
                for assignment in comparison["shards"]
            }
            for future in concurrent.futures.as_completed(futures):
                shard_id = futures[future]
                try:
                    _, shard_result, result_path = future.result()
                except BaseException as exc:
                    errors.append(f"{shard_id}: {exc}")
                else:
                    results[shard_id] = shard_result
                    paths[shard_id] = str(result_path)
    if errors:
        failure = {**plan, "dry_run": False, "errors": sorted(errors), "shard_results": paths}
        enoch_week1.atomic_write_json(output_dir / "execution-failed.json", failure)
        raise RunnerError("comparison failed closed: " + "; ".join(sorted(errors)))
    ordered_results = [results[item["shard_id"]] for item in comparison["shards"]]
    merged_path: str | None = None
    if merge:
        merged = enoch_week1.merge_shard_results(protocol, comparison, ordered_results)
        destination = output_dir / "merged-result.json"
        enoch_week1.atomic_write_json(destination, merged)
        merged_path = str(destination)
    completed = {
        **plan,
        "dry_run": False,
        "merged_result": merged_path,
        "shard_results": {name: paths[name] for name in sorted(paths)},
    }
    enoch_week1.atomic_write_json(output_dir / "execution-complete.json", completed)
    return completed


def _parse_override(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("environment override must be NAME=VALUE")
    name, setting = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("environment override name is empty")
    return name, setting


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "dry-run"))
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--launch-config", required=True, type=Path)
    parser.add_argument("--identities", required=True, type=Path)
    parser.add_argument("--external-evidence", type=Path)
    parser.add_argument("--evaluator", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--available-parallelism", required=True, type=int)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--env", action="append", default=[], type=_parse_override)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_comparison(
            protocol_path=args.protocol,
            comparison_path=args.comparison,
            launch_configuration_path=args.launch_config,
            identities_path=args.identities,
            evaluator=args.evaluator,
            ledger_path=args.ledger,
            output_dir=args.output,
            external_evidence_path=args.external_evidence,
            workers=args.workers,
            merge=args.merge,
            dry_run=args.command == "dry-run",
            timeout_seconds=args.timeout_seconds,
            environment_overrides=dict(args.env),
            available_parallelism=args.available_parallelism,
        )
    except (RunnerError, enoch_week1.ProtocolError, FileExistsError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
