#!/usr/bin/env python3
"""Authoritative W1.0/W1.1 operator with fail-closed resumability.

This is the narrow executable surface for the first two Week-1 phases.  It
does not make strength decisions.  It freezes one protocol/control bundle,
runs every cheap source-bound check before consuming evaluation seeds, then
executes the exact 1/10/100 product smokes and seals their typed lineage.

Completed immutable stages are validated and reused.  An interrupted stage is
retried only when its namespace has no ledger claims; any ambiguous consumed
seed fails closed and requires a fresh protocol rather than a silent rerun.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Iterator, Mapping, Sequence

try:
    import fcntl
except ImportError:  # pragma: no cover - authoritative operation requires Unix.
    fcntl = None  # type: ignore[assignment]

try:
    from training import enoch_week1
    from training import enoch_week1_evidence
    from training import enoch_week1_fixtures
    from training import enoch_week1_freeze
    from training import enoch_week1_preflight
    from training import enoch_week1_runner
except ImportError:  # pragma: no cover - direct script execution.
    import enoch_week1  # type: ignore[no-redef]
    import enoch_week1_evidence  # type: ignore[no-redef]
    import enoch_week1_fixtures  # type: ignore[no-redef]
    import enoch_week1_freeze  # type: ignore[no-redef]
    import enoch_week1_preflight  # type: ignore[no-redef]
    import enoch_week1_runner  # type: ignore[no-redef]


SMOKE_SPECS = (
    ("smoke/product/001", "001", 1, 1),
    ("smoke/product/010", "010", 10, 4),
    ("smoke/product/100", "100", 100, 8),
)
PREFLIGHT_NAMESPACES = (
    enoch_week1_preflight.EQUIVALENCE_NAMESPACE,
    enoch_week1_preflight.OUTCOME_BASELINE_NAMESPACE,
    enoch_week1_preflight.STYLE_BASELINE_NAMESPACE,
)


class OperatorError(RuntimeError):
    """Raised when an authoritative phase cannot proceed without ambiguity."""


@dataclass(frozen=True)
class RunLayout:
    root: Path

    @property
    def protocol(self) -> Path:
        return self.root / "protocol.json"

    @property
    def ledger(self) -> Path:
        return self.root / "seed-ledger.json"

    @property
    def control_bundle(self) -> Path:
        return self.root / "w1.0" / "control-bundle"

    @property
    def phase0(self) -> Path:
        return self.root / "w1.0" / "phase-manifest.json"

    @property
    def w1_1(self) -> Path:
        return self.root / "w1.1"

    @property
    def authority(self) -> Path:
        return self.w1_1 / "deterministic-search-authority.json"

    @property
    def fixtures(self) -> Path:
        return self.w1_1 / "fixtures"

    @property
    def preflight(self) -> Path:
        return self.w1_1 / "full-preflight.json"

    @property
    def report(self) -> Path:
        return self.w1_1 / "baseline-worker-report.json"

    @property
    def phase1(self) -> Path:
        return self.w1_1 / "phase-manifest.json"

    @property
    def operator_lock(self) -> Path:
        return self.root / ".authoritative-w1.0-w1.1.lock"

    @property
    def provenance(self) -> Path:
        return self.root / "w1.0" / "operator-source-provenance.json"

    def smoke(self, label: str) -> Path:
        return self.w1_1 / "smokes" / label


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _git(
    workspace: Path, *arguments: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        ["git", "-C", str(workspace), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if check and completed.returncode != 0:
        raise OperatorError(
            f"git {' '.join(arguments)} failed: {completed.stderr.strip()}"
        )
    return completed


def _git_identity(workspace: Path) -> dict[str, str]:
    workspace = workspace.resolve()
    top = Path(_git(workspace, "rev-parse", "--show-toplevel").stdout.strip()).resolve()
    if top != workspace:
        raise OperatorError(f"workspace must be the Git root: expected {top}")
    status = _git(
        workspace, "status", "--porcelain=v1", "--untracked-files=all"
    ).stdout
    if status:
        first = status.splitlines()[0]
        raise OperatorError(f"authoritative workspace is not clean: {first}")
    head = _git(workspace, "rev-parse", "HEAD^{commit}").stdout.strip()
    tree = _git(workspace, "rev-parse", "HEAD^{tree}").stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", head) or not re.fullmatch(
        r"[0-9a-f]{40}", tree
    ):
        raise OperatorError("Git HEAD/tree identity is not canonical")
    operator_relative = Path("training/enoch_week1_operator.py")
    _git(workspace, "ls-files", "--error-unmatch", operator_relative.as_posix())
    operator_path = workspace / operator_relative
    return {
        "git_commit": head,
        "git_tree": tree,
        "operator_source_path": operator_relative.as_posix(),
        "operator_source_sha256": _sha256_file(operator_path),
    }


def _require_safe_run_root(workspace: Path, root: Path) -> None:
    workspace = workspace.resolve()
    root = root.resolve()
    try:
        root.relative_to(workspace)
    except ValueError:
        return
    ignored = _git(
        workspace,
        "check-ignore",
        "--quiet",
        "--no-index",
        str(root),
        check=False,
    )
    if ignored.returncode != 0:
        raise OperatorError(
            "run root inside the repository must be ignored; use "
            f"{workspace / '.enoch-week1-runs' / '<run-id>'}"
        )


def _load_json(path: Path) -> dict[str, Any]:
    return enoch_week1.load_json_object(path)


def _load_json_value(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as source:
            return json.load(source)
    except (OSError, json.JSONDecodeError) as exc:
        raise OperatorError(f"could not load JSON artifact {path}: {exc}") from exc


def _write_or_match(path: Path, value: Mapping[str, Any], label: str) -> None:
    if path.exists():
        if _load_json(path) != dict(value):
            raise OperatorError(f"existing {label} does not reconstruct: {path}")
        return
    enoch_week1.atomic_write_json(path, value)


def _require_match(path: Path, value: Mapping[str, Any], label: str) -> None:
    if not path.is_file() or _load_json(path) != dict(value):
        raise OperatorError(f"completed {label} does not reconstruct: {path}")


@contextlib.contextmanager
def _operator_lock(path: Path) -> Iterator[None]:
    if fcntl is None:
        raise OperatorError("authoritative operation requires Unix advisory locking")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise OperatorError("another W1.0/W1.1 operator is active") from exc
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _ledger(protocol: Mapping[str, Any], layout: RunLayout) -> dict[str, Any]:
    ledger = _load_json(layout.ledger)
    enoch_week1.validate_seed_ledger(protocol, ledger)
    return ledger


def _consumed_in(
    ledger: Mapping[str, Any], namespaces: set[str]
) -> list[Mapping[str, Any]]:
    return [
        record
        for record in ledger["consumed"]
        if record["namespace"] in namespaces
    ]


def _require_unconsumed(
    protocol: Mapping[str, Any], layout: RunLayout, namespaces: set[str], label: str
) -> None:
    consumed = _consumed_in(_ledger(protocol, layout), namespaces)
    if consumed:
        coordinates = [f"{row['namespace']}[{row['index']}]" for row in consumed[:5]]
        suffix = "..." if len(consumed) > 5 else ""
        raise OperatorError(
            f"{label} has ledger claims but no validated completion artifact: "
            f"{', '.join(coordinates)}{suffix}; start a fresh protocol"
        )


def initialize(
    layout: RunLayout, workspace: Path, master_seed: int
) -> dict[str, str]:
    """Create or validate a protocol with an empty evaluator environment allowlist."""

    _require_safe_run_root(workspace, layout.root)
    source_identity = _git_identity(workspace)
    with _operator_lock(layout.operator_lock):
        protocol_exists = layout.protocol.exists()
        ledger_exists = layout.ledger.exists()
        if protocol_exists != ledger_exists:
            raise OperatorError("protocol/ledger initialization is partial and ambiguous")
        if protocol_exists:
            protocol = _load_json(layout.protocol)
            enoch_week1.validate_protocol(protocol)
            if protocol["seed_registry"]["master_seed"] != master_seed:
                raise OperatorError("existing run root belongs to another master seed")
            ledger = _ledger(protocol, layout)
            if ledger["consumed"]:
                raise OperatorError("cannot reinitialize a ledger that already consumed seeds")
        else:
            protocol = enoch_week1.build_protocol(
                master_seed, evaluator_env_allowlist=()
            )
            enoch_week1.atomic_write_json(layout.protocol, protocol)
            enoch_week1.atomic_write_json(
                layout.ledger, enoch_week1.new_seed_ledger(protocol)
            )
        return {
            "git_commit": source_identity["git_commit"],
            "ledger_fingerprint": _load_json(layout.ledger)["ledger_fingerprint"],
            "protocol_fingerprint": protocol["protocol_fingerprint"],
        }


def _build_provenance(
    protocol: Mapping[str, Any],
    control: Mapping[str, Any],
    source_identity: Mapping[str, str],
) -> dict[str, Any]:
    control_fingerprint = enoch_week1.validate_w1_0_control_manifest(
        protocol, control
    )
    body = {
        "automatic_production_promotion_allowed": False,
        "control_manifest_fingerprint": control_fingerprint,
        "evaluator_source_sha256": control["evaluator_identity"]["source_sha256"],
        "git_commit": source_identity["git_commit"],
        "git_tree": source_identity["git_tree"],
        "manifest_kind": "enoch-week1-operator-source-provenance",
        "manifest_version": 1,
        "operator_source_path": source_identity["operator_source_path"],
        "operator_source_sha256": source_identity["operator_source_sha256"],
        "production_reference": control["production_reference"],
        "protocol_fingerprint": protocol["protocol_fingerprint"],
    }
    return {
        **body,
        "source_provenance_fingerprint": enoch_week1.canonical_json_sha256(body),
    }


def _validate_provenance(
    protocol: Mapping[str, Any],
    control: Mapping[str, Any],
    provenance: Mapping[str, Any],
    source_identity: Mapping[str, str] | None = None,
) -> str:
    expected_keys = {
        "automatic_production_promotion_allowed",
        "control_manifest_fingerprint",
        "evaluator_source_sha256",
        "git_commit",
        "git_tree",
        "manifest_kind",
        "manifest_version",
        "operator_source_path",
        "operator_source_sha256",
        "production_reference",
        "protocol_fingerprint",
        "source_provenance_fingerprint",
    }
    if set(provenance) != expected_keys:
        raise OperatorError("operator source provenance schema changed")
    if (
        provenance["manifest_kind"]
        != "enoch-week1-operator-source-provenance"
        or provenance["manifest_version"] != 1
        or provenance["automatic_production_promotion_allowed"] is not False
    ):
        raise OperatorError("unsupported operator source provenance")
    expected_bindings = {
        "control_manifest_fingerprint": enoch_week1.validate_w1_0_control_manifest(
            protocol, control
        ),
        "evaluator_source_sha256": control["evaluator_identity"]["source_sha256"],
        "production_reference": control["production_reference"],
        "protocol_fingerprint": protocol["protocol_fingerprint"],
    }
    for field, expected in expected_bindings.items():
        if provenance[field] != expected:
            raise OperatorError(f"operator source provenance {field} mismatch")
    for field in ("git_commit", "git_tree"):
        if not isinstance(provenance[field], str) or not re.fullmatch(
            r"[0-9a-f]{40}", provenance[field]
        ):
            raise OperatorError(f"operator source provenance {field} is invalid")
    if provenance["operator_source_path"] != "training/enoch_week1_operator.py":
        raise OperatorError("operator source provenance path changed")
    if not isinstance(provenance["operator_source_sha256"], str) or not re.fullmatch(
        r"[0-9a-f]{64}", provenance["operator_source_sha256"]
    ):
        raise OperatorError("operator source provenance hash is invalid")
    body = dict(provenance)
    fingerprint = body.pop("source_provenance_fingerprint")
    if fingerprint != enoch_week1.canonical_json_sha256(body):
        raise OperatorError("operator source provenance fingerprint mismatch")
    if source_identity is not None:
        for field in (
            "git_commit",
            "git_tree",
            "operator_source_path",
            "operator_source_sha256",
        ):
            if provenance[field] != source_identity[field]:
                raise OperatorError(
                    f"workspace differs from frozen operator provenance: {field}"
                )
    return fingerprint


def _seal_phase0(
    protocol: Mapping[str, Any],
    layout: RunLayout,
    control: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    fingerprint = enoch_week1.validate_w1_0_control_manifest(protocol, control)
    provenance_fingerprint = _validate_provenance(protocol, control, provenance)
    expected = enoch_week1.build_phase_manifest(
        protocol,
        "W1.0",
        artifacts={
            "immutable-control-manifest": fingerprint,
            "operator-source-provenance": provenance_fingerprint,
        },
        declarations={
            "control_manifest_fingerprint": fingerprint,
            "implementation_source_commit": provenance["git_commit"],
            "production_reference": control["production_reference"],
        },
    )
    _write_or_match(layout.phase0, expected, "W1.0 phase manifest")
    enoch_week1.validate_phase_manifest(protocol, expected)
    return expected


def freeze_w1_0(
    layout: RunLayout, workspace: Path, reference: str
) -> dict[str, str]:
    """Build or validate the immutable W1.0 bundle and phase envelope."""

    _require_safe_run_root(workspace, layout.root)
    source_identity = _git_identity(workspace)
    with _operator_lock(layout.operator_lock):
        protocol = _load_json(layout.protocol)
        enoch_week1.validate_protocol(protocol)
        _ledger(protocol, layout)
        if layout.control_bundle.exists():
            control_fingerprint = enoch_week1_freeze.verify_bundle(
                layout.protocol, layout.control_bundle
            )
        else:
            enoch_week1_freeze.freeze_bundle(
                workspace.resolve(),
                layout.protocol.resolve(),
                layout.control_bundle.resolve(),
                reference,
            )
            control_fingerprint = enoch_week1_freeze.verify_bundle(
                layout.protocol, layout.control_bundle
            )
        control = _load_json(layout.control_bundle / "control-manifest.json")
        if _git_identity(workspace) != source_identity:
            raise OperatorError("Git source identity changed while W1.0 was freezing")
        provenance = _build_provenance(protocol, control, source_identity)
        _write_or_match(
            layout.provenance, provenance, "operator source provenance"
        )
        phase = _seal_phase0(protocol, layout, control, provenance)
        return {
            "control_manifest_fingerprint": control_fingerprint,
            "git_commit": provenance["git_commit"],
            "phase_manifest_fingerprint": phase["phase_manifest_fingerprint"],
        }


def _verify_frozen_source(workspace: Path, layout: RunLayout) -> None:
    frozen_path = (
        layout.control_bundle / "source" / "week1-evaluator-source-files.json"
    )
    frozen = _load_json_value(frozen_path)
    current = enoch_week1_preflight._workspace_source_records(  # noqa: SLF001
        workspace.resolve()
    )
    if current != frozen:
        raise OperatorError("workspace source differs from the frozen W1.0 evaluator")


def _ensure_authority(
    protocol: Mapping[str, Any], layout: RunLayout, workspace: Path, timeout: float
) -> dict[str, Any]:
    if layout.authority.exists():
        authority = _load_json(layout.authority)
        enoch_week1_preflight.validate_deterministic_search_fixture_authority(
            protocol, layout.control_bundle, authority
        )
        return authority
    authority = enoch_week1_preflight.build_deterministic_search_fixture_authority(
        protocol,
        layout.control_bundle,
        workspace.resolve(),
        timeout_seconds=timeout,
    )
    enoch_week1.atomic_write_json(layout.authority, authority)
    return authority


def _validate_fixture_gate(layout: RunLayout) -> dict[str, Any]:
    report_path = layout.fixtures / "fixture-report.json"
    # This evidence validator reopens every log, unlike the schema-only report
    # validator, so corrupted cached fixtures fail before preflight seeds burn.
    report, _, _, _ = enoch_week1_evidence._validate_fixture_bundle(  # noqa: SLF001
        report_path
    )
    frozen = _load_json_value(
        layout.control_bundle / "source" / "week1-evaluator-source-files.json"
    )
    fixture_sources = {row["path"]: row["sha256"] for row in report["source_files"]}
    for record in frozen:
        if fixture_sources.get(record["path"]) != record["sha256"]:
            raise OperatorError(
                f"fixture gate differs from frozen source: {record['path']}"
            )
    return dict(report)


def _ensure_fixtures(layout: RunLayout, workspace: Path) -> dict[str, Any]:
    if not layout.fixtures.exists():
        enoch_week1_fixtures.run_fixtures(
            workspace.resolve(), layout.fixtures.resolve()
        )
    return _validate_fixture_gate(layout)


def _ensure_full_preflight(
    protocol: Mapping[str, Any], layout: RunLayout, timeout: float
) -> dict[str, Any]:
    if layout.preflight.exists():
        artifact = _load_json(layout.preflight)
        enoch_week1_preflight.validate_preflight_artifact(
            protocol,
            _ledger(protocol, layout),
            layout.control_bundle,
            artifact,
            require_full_coverage=True,
        )
        return artifact
    _require_unconsumed(
        protocol, layout, set(PREFLIGHT_NAMESPACES), "full W1.1 preflight"
    )
    artifact = enoch_week1_preflight.build_preflight_artifact(
        protocol,
        layout.ledger,
        layout.control_bundle,
        timeout_seconds=timeout,
    )
    enoch_week1.atomic_write_json(layout.preflight, artifact)
    return artifact


def _runtime_contract(
    protocol: Mapping[str, Any],
    control: Mapping[str, Any],
    base_environment: Mapping[str, str],
    available_parallelism: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str], dict[str, Any]]:
    knobs = control["search_knobs"]["enoch-0"]
    launch = enoch_week1_runner.build_launch_configuration(
        candidate_arm_ids=[],
        worlds=knobs["max_worlds"],
        candidates=knobs["max_candidates"],
        rollout_tricks=knobs["rollout_tricks"],
        scenario_id="standard",
        budget_ms=knobs["budget_ms"],
    )
    evaluator_identity = control["evaluator_identity"]
    identities = enoch_week1_runner.build_in_process_identity_bindings(
        evaluator_identity, launch
    )
    child_environment, _ = enoch_week1.sanitized_evaluator_environment(
        base_environment,
        allowlist=protocol["evaluator_environment_policy"]["allowlist"],
    )
    environment_identity = enoch_week1_runner.build_evaluator_environment_identity(
        evaluator_identity,
        protocol,
        child_environment,
        available_parallelism=available_parallelism,
    )
    return launch, identities, child_environment, environment_identity


def _smoke_workers(
    nominal: int, maximum: int, available_parallelism: int
) -> int:
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
        raise OperatorError("maximum smoke workers must be a positive integer")
    if maximum > enoch_week1_runner.MAX_WORKERS:
        raise OperatorError(
            f"maximum smoke workers cannot exceed {enoch_week1_runner.MAX_WORKERS}"
        )
    return min(nominal, maximum, available_parallelism)


def _declarations(
    protocol: Mapping[str, Any],
    layout: RunLayout,
    control: Mapping[str, Any],
    base_environment: Mapping[str, str],
    available_parallelism: int,
    maximum_workers: int,
) -> dict[str, dict[str, Any]]:
    launch, identities, child_environment, environment_identity = _runtime_contract(
        protocol,
        control,
        base_environment,
        available_parallelism,
    )
    launch_fingerprint = enoch_week1_runner.validate_launch_configuration(launch)
    environment_fingerprint = enoch_week1.canonical_json_sha256(environment_identity)
    fingerprints = {
        name: enoch_week1.canonical_json_sha256(identity)
        for name, identity in identities.items()
    }
    declared: dict[str, dict[str, Any]] = {}
    for namespace, label, pair_count, nominal_workers in SMOKE_SPECS:
        workers = _smoke_workers(
            nominal_workers, maximum_workers, available_parallelism
        )
        comparison = enoch_week1.build_comparison_protocol_manifest(
            protocol,
            phase="W1.1",
            comparison_id=f"runtime-{label}",
            subject_id="runtime-enoch0-smoke",
            seed_namespace=namespace,
            pair_count=pair_count,
            shard_count=workers,
            candidate_fingerprint=fingerprints["candidate"],
            control_fingerprint=fingerprints["control"],
            evaluator_fingerprint=fingerprints["evaluator"],
            environment_fingerprint=environment_fingerprint,
            configuration_fingerprint=launch_fingerprint,
            required_style_metrics=enoch_week1.WEEK1_STYLE_METRICS,
        )
        root = layout.smoke(label) / "declaration"
        declared[namespace] = {
            "child_environment": child_environment,
            "comparison": comparison,
            "environment_identity": environment_identity,
            "identities": identities,
            "launch": launch,
            "root": root,
            "workers": workers,
        }
    return declared


def _expected_claim_consumers(
    comparison: Mapping[str, Any],
) -> dict[int, str]:
    prefix = comparison["comparison_protocol_fingerprint"][:16]
    expected: dict[int, str] = {}
    for assignment in comparison["shards"]:
        consumer = f"runner:{prefix}:{assignment['shard_id']}"
        for index in assignment["seed_indices"]:
            expected[index] = consumer
    return expected


def _validate_smoke_claims(
    protocol: Mapping[str, Any], layout: RunLayout, comparison: Mapping[str, Any]
) -> None:
    namespace = comparison["seed_namespace"]
    actual = {
        record["index"]: record["consumer"]
        for record in _ledger(protocol, layout)["consumed"]
        if record["namespace"] == namespace
    }
    expected = _expected_claim_consumers(comparison)
    if actual != expected:
        raise OperatorError(f"{namespace} ledger claims do not match its completed run")


def _attempts(smoke_root: Path) -> list[Path]:
    if not smoke_root.exists():
        return []
    return sorted(
        path
        for path in smoke_root.iterdir()
        if path.is_dir() and path.name.startswith("attempt-")
    )


def _completed_attempt(smoke_root: Path) -> Path | None:
    complete = [
        path
        for path in _attempts(smoke_root)
        if (path / "execution" / "execution-complete.json").is_file()
    ]
    if len(complete) > 1:
        raise OperatorError(f"multiple completed attempts exist under {smoke_root}")
    return complete[0] if complete else None


def _validate_environment_probe(
    artifact: Mapping[str, Any], expected_environment: Mapping[str, Any]
) -> str:
    if set(artifact) != {"environment", "environment_identity_sha256"}:
        raise OperatorError("environment identity probe schema changed")
    expected_hash = enoch_week1.canonical_json_sha256(expected_environment)
    if (
        artifact["environment"] != dict(expected_environment)
        or artifact["environment_identity_sha256"] != expected_hash
    ):
        raise OperatorError("environment identity probe differs from its declaration")
    return expected_hash


def _validate_completed_smoke(
    protocol: Mapping[str, Any],
    layout: RunLayout,
    declared: Mapping[str, Any],
    attempt: Path,
) -> dict[str, Any]:
    comparison = declared["comparison"]
    launch = declared["launch"]
    identities = declared["identities"]
    execution = attempt / "execution"
    for filename, expected in (
        ("comparison.json", comparison),
        ("launch-configuration.json", launch),
        ("identity-bindings.json", identities),
    ):
        if _load_json(execution / filename) != expected:
            raise OperatorError(f"completed smoke changed {filename}: {execution}")
    merged = _load_json(execution / "merged-result.json")
    enoch_week1.validate_merged_result(protocol, comparison, merged)
    machine = _load_json(attempt / "machine-attestation.json")
    enoch_week1_evidence.validate_machine_contention_attestation(
        comparison, machine
    )
    evidence = _load_json(attempt / "external-evidence.json")
    enoch_week1_evidence.validate_verified_external_evidence(
        protocol, comparison, evidence
    )
    expected_environment = _load_json(
        declared["root"] / "environment-identity.json"
    )
    if machine["worker_count"] != len(comparison["shards"]):
        raise OperatorError("machine attestation worker count differs from shards")
    if machine["available_parallelism"] != expected_environment[
        "available_parallelism"
    ]:
        raise OperatorError(
            "machine attestation parallelism differs from environment identity"
        )
    _validate_environment_probe(
        _load_json(attempt / "environment-probe.json"), expected_environment
    )
    _validate_smoke_claims(protocol, layout, comparison)
    return {
        "comparison": comparison,
        "identity_bindings": identities,
        "launch_configuration": launch,
        "merged_result": merged,
    }


def _next_attempt(smoke_root: Path) -> Path:
    attempts = _attempts(smoke_root)
    ordinal = 1
    if attempts:
        try:
            ordinal = max(int(path.name.removeprefix("attempt-")) for path in attempts) + 1
        except ValueError as exc:
            raise OperatorError(f"malformed attempt directory under {smoke_root}") from exc
    path = smoke_root / f"attempt-{ordinal:03d}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_seconds(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _finish_observation(started: datetime) -> datetime:
    ended = _utc_now()
    while _utc_seconds(ended) == _utc_seconds(started):
        time.sleep(0.05)
        ended = _utc_now()
    return ended


def _model_contract_paths(layout: RunLayout) -> dict[str, Path]:
    return {
        artifact_id: layout.control_bundle
        / "preflight"
        / f"{artifact_id.rsplit('/', 1)[-1]}.json"
        for artifact_id in enoch_week1_evidence.MODEL_CONTRACT_ARTIFACTS
    }


def _run_smoke(
    protocol: Mapping[str, Any],
    layout: RunLayout,
    declared: Mapping[str, Any],
    *,
    operator_id: str,
    available_parallelism: int,
    shard_timeout: float,
    base_environment: Mapping[str, str],
) -> dict[str, Any]:
    comparison = declared["comparison"]
    namespace = comparison["seed_namespace"]
    smoke_root = layout.smoke(namespace.rsplit("/", 1)[-1])
    with enoch_week1_runner.authoritative_campaign_lock(
        protocol, comparison
    ) as campaign_lock_token:
        root = declared["root"]
        label = namespace.rsplit("/", 1)[-1]
        completed = _completed_attempt(smoke_root)
        if completed is not None:
            _require_match(
                root / "comparison.json", comparison, f"{label} comparison"
            )
            _require_match(
                root / "launch.json", declared["launch"], f"{label} launch"
            )
            _require_match(
                root / "identities.json",
                declared["identities"],
                f"{label} identities",
            )
            _require_match(
                root / "environment-identity.json",
                declared["environment_identity"],
                f"{label} environment identity",
            )
            return _validate_completed_smoke(
                protocol, layout, declared, completed
            )
        _require_unconsumed(protocol, layout, {namespace}, namespace)
        started = _utc_now()
        probe_artifact = enoch_week1_runner.probe_evaluator_environment_identity(
            evaluator=(layout.control_bundle / "bin" / "enoch-week1-evaluator"),
            protocol_path=layout.protocol,
            protocol=protocol,
            comparison=comparison,
            launch_configuration=declared["launch"],
            child_environment=declared["child_environment"],
            evaluator_identity=declared["identities"]["evaluator"],
            available_parallelism=available_parallelism,
            campaign_lock_token=campaign_lock_token,
            timeout_seconds=shard_timeout,
        )
        expected_environment = declared["environment_identity"]
        _validate_environment_probe(probe_artifact, expected_environment)
        # The first comparison-path mutation happens only after the Rust probe
        # has confirmed actual environment/parallelism under the live lock.
        _write_or_match(root / "comparison.json", comparison, f"{label} comparison")
        _write_or_match(
            root / "launch.json", declared["launch"], f"{label} launch"
        )
        _write_or_match(
            root / "identities.json", declared["identities"], f"{label} identities"
        )
        _write_or_match(
            root / "environment-identity.json",
            declared["environment_identity"],
            f"{label} environment identity",
        )
        attempt = _next_attempt(smoke_root)
        enoch_week1.atomic_write_json(
            attempt / "environment-probe.json", probe_artifact
        )
        ended = _finish_observation(started)
        machine = enoch_week1_evidence.build_machine_contention_attestation(
            comparison,
            operator_id=operator_id,
            observation_started_utc=_utc_seconds(started),
            observation_ended_utc=_utc_seconds(ended),
            attested_at_utc=_utc_seconds(_utc_now()),
            worker_count=declared["workers"],
            available_parallelism=available_parallelism,
        )
        enoch_week1.atomic_write_json(
            attempt / "machine-attestation.json", machine
        )
        evidence = enoch_week1_evidence.build_verified_external_evidence(
            protocol,
            comparison,
            fixture_report_path=layout.fixtures / "fixture-report.json",
            source_identity_path=(
                layout.control_bundle
                / "source"
                / "week1-evaluator-source-files.json"
            ),
            control_manifest_path=layout.control_bundle / "control-manifest.json",
            runner_identities_path=root / "identities.json",
            model_contract_artifact_paths=_model_contract_paths(layout),
            machine_attestation_path=attempt / "machine-attestation.json",
        )
        enoch_week1.atomic_write_json(
            attempt / "external-evidence.json", evidence
        )
        # Reopen both operator assertions immediately before the runner can launch.
        enoch_week1_evidence.validate_machine_contention_attestation(
            comparison, machine
        )
        enoch_week1_evidence.validate_verified_external_evidence(
            protocol, comparison, evidence
        )
        common = {
            "protocol_path": layout.protocol,
            "comparison_path": root / "comparison.json",
            "launch_configuration_path": root / "launch.json",
            "identities_path": root / "identities.json",
            "evaluator": layout.control_bundle / "bin" / "enoch-week1-evaluator",
            "ledger_path": layout.ledger,
            "external_evidence_path": attempt / "external-evidence.json",
            "workers": declared["workers"],
            "merge": True,
            "timeout_seconds": shard_timeout,
            "base_environment": base_environment,
            "environment_overrides": {},
            "available_parallelism": available_parallelism,
            "campaign_lock_token": campaign_lock_token,
        }
        plan = enoch_week1_runner.run_comparison(
            **common,
            output_dir=attempt / "dry-run-unused",
            dry_run=True,
        )
        enoch_week1.atomic_write_json(attempt / "runner-plan.json", plan)
        enoch_week1_runner.run_comparison(
            **common,
            output_dir=attempt / "execution",
            dry_run=False,
        )
        return _validate_completed_smoke(protocol, layout, declared, attempt)


def _smoke_evidence_from_disk(
    protocol: Mapping[str, Any],
    layout: RunLayout,
    declarations: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for namespace, _, _, _ in SMOKE_SPECS:
        label = namespace.rsplit("/", 1)[-1]
        completed = _completed_attempt(layout.smoke(label))
        if completed is None:
            raise OperatorError(f"missing completed smoke: {namespace}")
        evidence[namespace] = _validate_completed_smoke(
            protocol, layout, declarations[namespace], completed
        )
    return evidence


def _seal_report_and_phase1(
    protocol: Mapping[str, Any],
    layout: RunLayout,
    preflight: Mapping[str, Any],
    authority: Mapping[str, Any],
    fixtures: Mapping[str, Any],
    provenance: Mapping[str, Any],
    smoke_evidence: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    ledger = _ledger(protocol, layout)
    expected_report = enoch_week1_preflight.build_w1_1_baseline_worker_report(
        protocol,
        ledger,
        layout.control_bundle,
        preflight,
        smoke_evidence,
        authority,
    )
    _write_or_match(layout.report, expected_report, "W1.1 baseline/worker report")
    enoch_week1_preflight.validate_w1_1_baseline_worker_report(
        protocol,
        ledger,
        layout.control_bundle,
        preflight,
        smoke_evidence,
        authority,
        expected_report,
    )
    phase0 = _load_json(layout.phase0)
    artifacts = {
        "baseline-and-worker-report": expected_report[
            "baseline_worker_report_fingerprint"
        ],
        "deterministic-search-authority": authority[
            "deterministic_search_authority_fingerprint"
        ],
        "fixture-gate": fixtures["fixture_report_fingerprint"],
        "full-preflight": preflight["artifact_sha256"],
        "operator-source-provenance": provenance[
            "source_provenance_fingerprint"
        ],
    }
    for namespace, summary in smoke_evidence.items():
        artifacts[namespace] = summary["merged_result"]["merged_result_fingerprint"]
    expected_phase = enoch_week1.build_phase_manifest(
        protocol,
        "W1.1",
        artifacts=artifacts,
        declarations={
            "baseline_worker_report_fingerprint": expected_report[
                "baseline_worker_report_fingerprint"
            ],
            "fixed_worker_configuration": expected_report[
                "fixed_worker_configuration"
            ],
            "full_preflight": True,
            "implementation_source_commit": provenance["git_commit"],
        },
        parent_phase_manifests=[phase0],
    )
    _write_or_match(layout.phase1, expected_phase, "W1.1 phase manifest")
    enoch_week1.validate_phase_chain(protocol, [phase0, expected_phase])
    return expected_report, expected_phase


def run_w1_1(
    layout: RunLayout,
    workspace: Path,
    *,
    operator_id: str,
    attest_no_machine_contention: bool,
    available_parallelism: int,
    maximum_smoke_workers: int = 8,
    probe_timeout_seconds: float = enoch_week1_preflight.DEFAULT_PROBE_TIMEOUT_SECONDS,
    shard_timeout_seconds: float = enoch_week1_preflight.DEFAULT_PROBE_TIMEOUT_SECONDS,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run or safely resume the complete authoritative W1.1 package."""

    if attest_no_machine_contention is not True:
        raise OperatorError(
            "--attest-no-machine-contention is required before product smokes"
        )
    if (
        isinstance(available_parallelism, bool)
        or not isinstance(available_parallelism, int)
        or available_parallelism <= 0
    ):
        raise OperatorError("available parallelism must be a positive integer")
    if shard_timeout_seconds <= 0 or probe_timeout_seconds <= 0:
        raise OperatorError("operator timeouts must be positive")
    base_environment = dict(os.environ if environment is None else environment)
    _require_safe_run_root(workspace, layout.root)
    source_identity = _git_identity(workspace)
    with _operator_lock(layout.operator_lock):
        protocol = _load_json(layout.protocol)
        enoch_week1.validate_protocol(protocol)
        if protocol["evaluator_environment_policy"]["allowlist"]:
            raise OperatorError("authoritative W1.1 requires an empty environment allowlist")
        _ledger(protocol, layout)
        enoch_week1_freeze.verify_bundle(layout.protocol, layout.control_bundle)
        control = _load_json(layout.control_bundle / "control-manifest.json")
        provenance = _load_json(layout.provenance)
        _validate_provenance(protocol, control, provenance, source_identity)
        _seal_phase0(protocol, layout, control, provenance)
        _verify_frozen_source(workspace, layout)

        # All cheap/source-bound work precedes the first ledger mutation.
        authority = _ensure_authority(
            protocol, layout, workspace, probe_timeout_seconds
        )
        fixtures = _ensure_fixtures(layout, workspace)
        _verify_frozen_source(workspace, layout)
        if _git_identity(workspace) != source_identity:
            raise OperatorError("Git source identity changed before W1.1 preflight")
        preflight = _ensure_full_preflight(
            protocol, layout, probe_timeout_seconds
        )
        declarations = _declarations(
            protocol,
            layout,
            control,
            base_environment,
            available_parallelism,
            maximum_smoke_workers,
        )
        for namespace, _, _, _ in SMOKE_SPECS:
            if _git_identity(workspace) != source_identity:
                raise OperatorError("Git source identity changed before a product smoke")
            _run_smoke(
                protocol,
                layout,
                declarations[namespace],
                operator_id=operator_id,
                available_parallelism=available_parallelism,
                shard_timeout=shard_timeout_seconds,
                base_environment=base_environment,
            )
        smoke_evidence = _smoke_evidence_from_disk(
            protocol, layout, declarations
        )
        if _git_identity(workspace) != source_identity:
            raise OperatorError("Git source identity changed before W1.1 sealing")
        report, phase = _seal_report_and_phase1(
            protocol,
            layout,
            preflight,
            authority,
            fixtures,
            provenance,
            smoke_evidence,
        )
        return {
            "baseline_worker_report_fingerprint": report[
                "baseline_worker_report_fingerprint"
            ],
            "fixed_worker_configuration": report["fixed_worker_configuration"],
            "phase_manifest_fingerprint": phase["phase_manifest_fingerprint"],
            "protocol_fingerprint": protocol["protocol_fingerprint"],
        }


def verify_complete(layout: RunLayout) -> dict[str, str]:
    """Reconstruct and validate completed W1.0/W1.1 without running programs."""

    with _operator_lock(layout.operator_lock):
        protocol = _load_json(layout.protocol)
        enoch_week1.validate_protocol(protocol)
        ledger = _ledger(protocol, layout)
        control_fingerprint = enoch_week1_freeze.verify_bundle(
            layout.protocol, layout.control_bundle
        )
        control = _load_json(layout.control_bundle / "control-manifest.json")
        provenance = _load_json(layout.provenance)
        _validate_provenance(protocol, control, provenance)
        phase0 = _load_json(layout.phase0)
        enoch_week1.validate_phase_manifest(protocol, phase0)
        if provenance["source_provenance_fingerprint"] not in {
            record["sha256"] for record in phase0["artifacts"]
        }:
            raise OperatorError("W1.0 phase does not bind operator source provenance")
        authority = _load_json(layout.authority)
        enoch_week1_preflight.validate_deterministic_search_fixture_authority(
            protocol, layout.control_bundle, authority
        )
        fixtures = _validate_fixture_gate(layout)
        preflight = _load_json(layout.preflight)
        enoch_week1_preflight.validate_preflight_artifact(
            protocol,
            ledger,
            layout.control_bundle,
            preflight,
            require_full_coverage=True,
        )
        # Declarations are discovered from their sealed files so verification
        # never relies on caller-supplied hardware values.
        declarations: dict[str, dict[str, Any]] = {}
        for namespace, label, _, _ in SMOKE_SPECS:
            root = layout.smoke(label) / "declaration"
            declarations[namespace] = {
                "comparison": _load_json(root / "comparison.json"),
                "identities": _load_json(root / "identities.json"),
                "launch": _load_json(root / "launch.json"),
                "root": root,
            }
        smokes = _smoke_evidence_from_disk(protocol, layout, declarations)
        report = _load_json(layout.report)
        report_fingerprint = (
            enoch_week1_preflight.validate_w1_1_baseline_worker_report(
                protocol,
                ledger,
                layout.control_bundle,
                preflight,
                smokes,
                authority,
                report,
            )
        )
        phase1 = _load_json(layout.phase1)
        enoch_week1.validate_phase_chain(protocol, [phase0, phase1])
        if report_fingerprint not in {
            record["sha256"] for record in phase1["artifacts"]
        }:
            raise OperatorError("W1.1 phase does not bind the baseline/worker report")
        if provenance["source_provenance_fingerprint"] not in {
            record["sha256"] for record in phase1["artifacts"]
        }:
            raise OperatorError("W1.1 phase does not bind operator source provenance")
        return {
            "control_manifest_fingerprint": control_fingerprint,
            "fixture_report_fingerprint": fixtures["fixture_report_fingerprint"],
            "ledger_fingerprint": ledger["ledger_fingerprint"],
            "phase_manifest_fingerprint": phase1["phase_manifest_fingerprint"],
            "report_fingerprint": report_fingerprint,
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    initialize_parser = subparsers.add_parser(
        "init", help="create a fresh authoritative protocol and empty ledger"
    )
    initialize_parser.add_argument("--root", required=True, type=Path)
    initialize_parser.add_argument("--workspace", type=Path, default=Path.cwd())
    initialize_parser.add_argument(
        "--master-seed", required=True, type=enoch_week1.parse_u64
    )

    freeze_parser = subparsers.add_parser(
        "freeze-w1.0", help="build/verify W1.0 and seal its phase manifest"
    )
    freeze_parser.add_argument("--root", required=True, type=Path)
    freeze_parser.add_argument("--workspace", type=Path, default=Path.cwd())
    freeze_parser.add_argument(
        "--reference", default=enoch_week1_freeze.DEFAULT_REFERENCE
    )

    w1_1_parser = subparsers.add_parser(
        "run-w1.1", help="run/resume full W1.1 and seal its report/phase manifest"
    )
    w1_1_parser.add_argument("--root", required=True, type=Path)
    w1_1_parser.add_argument("--workspace", type=Path, default=Path.cwd())
    w1_1_parser.add_argument("--operator-id", required=True)
    w1_1_parser.add_argument("--available-parallelism", required=True, type=int)
    w1_1_parser.add_argument("--maximum-smoke-workers", type=int, default=8)
    w1_1_parser.add_argument(
        "--probe-timeout-seconds",
        type=float,
        default=enoch_week1_preflight.DEFAULT_PROBE_TIMEOUT_SECONDS,
    )
    w1_1_parser.add_argument(
        "--shard-timeout-seconds",
        type=float,
        default=enoch_week1_preflight.DEFAULT_PROBE_TIMEOUT_SECONDS,
    )
    w1_1_parser.add_argument(
        "--attest-no-machine-contention",
        action="store_true",
        help=(
            "operator attests that no competing workload is active; the CLI "
            "also holds its exclusive campaign lock through every smoke"
        ),
    )

    verify_parser = subparsers.add_parser(
        "verify", help="reconstruct completed W1.0/W1.1 from disk"
    )
    verify_parser.add_argument("--root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    layout = RunLayout(args.root.expanduser().resolve())
    try:
        if args.command == "init":
            result: Mapping[str, Any] = initialize(
                layout, args.workspace, args.master_seed
            )
        elif args.command == "freeze-w1.0":
            result = freeze_w1_0(layout, args.workspace, args.reference)
        elif args.command == "run-w1.1":
            result = run_w1_1(
                layout,
                args.workspace,
                operator_id=args.operator_id,
                attest_no_machine_contention=args.attest_no_machine_contention,
                available_parallelism=args.available_parallelism,
                maximum_smoke_workers=args.maximum_smoke_workers,
                probe_timeout_seconds=args.probe_timeout_seconds,
                shard_timeout_seconds=args.shard_timeout_seconds,
            )
        else:
            result = verify_complete(layout)
    except (
        OperatorError,
        enoch_week1.ProtocolError,
        enoch_week1_evidence.EvidenceError,
        enoch_week1_fixtures.FixtureError,
        enoch_week1_freeze.FreezeError,
        enoch_week1_preflight.PreflightError,
        enoch_week1_runner.RunnerError,
        FileExistsError,
        OSError,
    ) as exc:
        print(f"operator failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
