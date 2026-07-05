#!/usr/bin/env python3
"""Typed, file-backed external evidence for Enoch Week-1 comparisons.

This module deliberately accepts artifact paths rather than caller-supplied
digests.  Every zero counter is derived only after the referenced artifact has
been opened, hashed, and validated against its own frozen schema and lineage.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

try:
    from . import enoch_week1, enoch_week1_fixtures
except ImportError:  # pragma: no cover - direct-script import path.
    import enoch_week1  # type: ignore[no-redef]
    import enoch_week1_fixtures  # type: ignore[no-redef]


MANIFEST_VERSION = 1
EVIDENCE_KIND = "enoch-week1-verified-external-evidence"
MACHINE_ATTESTATION_KIND = "enoch-week1-machine-contention-attestation"
EXTERNAL_COUNTERS = (
    "artifact_mismatch",
    "fixture_failure",
    "hidden_information_leak",
    "honesty_violation",
    "machine_contention",
    "model_contract_failure",
)
FIXTURE_REPORT_ARTIFACT = "fixtures/report"
SOURCE_IDENTITY_ARTIFACT = "fixtures/source-identity"
CONTROL_MANIFEST_ARTIFACT = "w1.0/control-manifest"
RUNNER_IDENTITIES_ARTIFACT = "runner/identity-bindings"
MACHINE_ATTESTATION_ARTIFACT = "operator/machine-contention-attestation"
MODEL_CONTRACT_ARTIFACTS = tuple(
    sorted(set(enoch_week1.MODEL_SELECTION_EVIDENCE_IDS.values()))
)
EXPECTED_ARTIFACT_IDS = tuple(
    sorted(
        (
            FIXTURE_REPORT_ARTIFACT,
            SOURCE_IDENTITY_ARTIFACT,
            CONTROL_MANIFEST_ARTIFACT,
            RUNNER_IDENTITIES_ARTIFACT,
            MACHINE_ATTESTATION_ARTIFACT,
            *MODEL_CONTRACT_ARTIFACTS,
        )
    )
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_OPERATOR_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_.@/-]*[A-Za-z0-9])?")
_UTC_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_MODEL_MARKERS: Mapping[str, tuple[str, ...]] = {
    "preflight/strict-evaluator-test": (
        "test bot::search::tests::strict_search_rejects_a_zero_sample_prior_fallback ... ok",
    ),
    "preflight/reference-model-contract-tests": (
        "test bot::expert::model_path_tests::model_path_override_round_trips ... ok",
        "test bot::expert::model_path_tests::manifest_rejects_width_drift_and_untyped_v2_outputs ... ok",
        "test bot::expert::model_path_tests::embedded_model_has_no_value_output ... ok",
    ),
    "preflight/expert-model-validation": (),
}


class EvidenceError(RuntimeError):
    """Raised when an external artifact cannot substantiate a zero counter."""


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise EvidenceError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise EvidenceError(f"{label} must be lowercase SHA-256")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvidenceError(f"{label} must be a nonnegative integer")
    return value


def _file_sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _load_json_artifact(path: Path, label: str) -> tuple[Any, str, Path]:
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise EvidenceError(f"{label} path cannot be resolved: {path}") from exc
    if not resolved.is_file():
        raise EvidenceError(f"{label} is not a regular file: {resolved}")
    try:
        content = resolved.read_bytes()
        value = json.loads(content.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"{label} is not one UTF-8 JSON document") from exc
    return value, _file_sha256_bytes(content), resolved


def _artifact_reference(
    artifact_id: str,
    artifact_kind: str,
    path: Path,
    file_sha256: str,
    semantic_fingerprint: str,
) -> dict[str, str]:
    return {
        "artifact_id": artifact_id,
        "artifact_kind": artifact_kind,
        "file_sha256": _sha256(file_sha256, f"{artifact_id} file hash"),
        "path": str(path),
        "semantic_fingerprint": _sha256(
            semantic_fingerprint, f"{artifact_id} semantic fingerprint"
        ),
    }


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not _UTC_RE.fullmatch(value):
        raise EvidenceError(f"{label} must be UTC YYYY-MM-DDTHH:MM:SSZ")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise EvidenceError(f"{label} is not a valid UTC timestamp") from exc
    return parsed


def build_machine_contention_attestation(
    comparison: Mapping[str, Any],
    *,
    operator_id: str,
    observation_started_utc: str,
    observation_ended_utc: str,
    attested_at_utc: str,
    worker_count: int,
    available_parallelism: int,
) -> dict[str, Any]:
    """Build a structured zero-contention operator attestation."""

    if not isinstance(comparison, Mapping):
        raise EvidenceError("machine attestation comparison must be an object")
    if not isinstance(operator_id, str) or not _OPERATOR_RE.fullmatch(operator_id):
        raise EvidenceError("machine attestation operator id is not canonical")
    worker_count = _nonnegative_int(worker_count, "machine attestation worker count")
    available_parallelism = _nonnegative_int(
        available_parallelism, "machine attestation available parallelism"
    )
    if worker_count == 0 or available_parallelism == 0:
        raise EvidenceError("machine attestation worker counts must be positive")
    if worker_count > available_parallelism:
        raise EvidenceError("machine attestation reports worker oversubscription")
    started = _parse_utc(observation_started_utc, "observation start")
    ended = _parse_utc(observation_ended_utc, "observation end")
    attested = _parse_utc(attested_at_utc, "attestation time")
    if not started < ended <= attested:
        raise EvidenceError("machine attestation timestamps are not ordered")
    body = {
        "attested_at_utc": attested_at_utc,
        "automatic_production_promotion_allowed": False,
        "available_parallelism": available_parallelism,
        "comparison_protocol_fingerprint": _sha256(
            comparison.get("comparison_protocol_fingerprint"),
            "machine attestation comparison fingerprint",
        ),
        "competing_process_count": 0,
        "environment_fingerprint": _sha256(
            comparison.get("environment_fingerprint"),
            "machine attestation environment fingerprint",
        ),
        "exclusive_campaign_lock_held": True,
        "machine_contention_count": 0,
        "manifest_kind": MACHINE_ATTESTATION_KIND,
        "manifest_version": MANIFEST_VERSION,
        "observation_ended_utc": observation_ended_utc,
        "observation_started_utc": observation_started_utc,
        "operator_id": operator_id,
        "worker_count": worker_count,
    }
    return {
        **body,
        "machine_attestation_fingerprint": enoch_week1.canonical_json_sha256(body),
    }


def validate_machine_contention_attestation(
    comparison: Mapping[str, Any], attestation: Mapping[str, Any]
) -> str:
    if not isinstance(attestation, Mapping):
        raise EvidenceError("machine contention attestation must be an object")
    _exact_keys(
        attestation,
        {
            "attested_at_utc",
            "automatic_production_promotion_allowed",
            "available_parallelism",
            "comparison_protocol_fingerprint",
            "competing_process_count",
            "environment_fingerprint",
            "exclusive_campaign_lock_held",
            "machine_attestation_fingerprint",
            "machine_contention_count",
            "manifest_kind",
            "manifest_version",
            "observation_ended_utc",
            "observation_started_utc",
            "operator_id",
            "worker_count",
        },
        "machine contention attestation",
    )
    if (
        attestation["manifest_kind"] != MACHINE_ATTESTATION_KIND
        or attestation["manifest_version"] != MANIFEST_VERSION
    ):
        raise EvidenceError("unsupported machine contention attestation")
    if attestation["automatic_production_promotion_allowed"] is not False:
        raise EvidenceError("machine attestation cannot authorize promotion")
    if attestation["comparison_protocol_fingerprint"] != comparison.get(
        "comparison_protocol_fingerprint"
    ):
        raise EvidenceError("machine attestation belongs to another comparison")
    if attestation["environment_fingerprint"] != comparison.get(
        "environment_fingerprint"
    ):
        raise EvidenceError("machine attestation belongs to another environment")
    if (
        attestation["competing_process_count"] != 0
        or attestation["machine_contention_count"] != 0
        or attestation["exclusive_campaign_lock_held"] is not True
    ):
        raise EvidenceError("machine attestation does not prove zero contention")
    rebuilt = build_machine_contention_attestation(
        comparison,
        operator_id=attestation["operator_id"],
        observation_started_utc=attestation["observation_started_utc"],
        observation_ended_utc=attestation["observation_ended_utc"],
        attested_at_utc=attestation["attested_at_utc"],
        worker_count=attestation["worker_count"],
        available_parallelism=attestation["available_parallelism"],
    )
    if dict(attestation) != rebuilt:
        raise EvidenceError("machine contention attestation fingerprint mismatch")
    return attestation["machine_attestation_fingerprint"]


def _validate_source_identity(value: Any) -> tuple[list[dict[str, str]], str]:
    if not isinstance(value, list) or not value:
        raise EvidenceError("fixture source identity must be a nonempty list")
    normalized: list[dict[str, str]] = []
    previous: str | None = None
    for record in value:
        if not isinstance(record, Mapping):
            raise EvidenceError("fixture source identity record must be an object")
        _exact_keys(record, {"path", "sha256"}, "fixture source identity record")
        path = record["path"]
        if (
            not isinstance(path, str)
            or not path
            or path.startswith("/")
            or ".." in Path(path).parts
            or (previous is not None and path <= previous)
        ):
            raise EvidenceError("fixture source identity paths are not canonical")
        normalized.append({"path": path, "sha256": _sha256(record["sha256"], path)})
        previous = path
    return normalized, enoch_week1.canonical_json_sha256(normalized)


def _validate_fixture_bundle(path: Path) -> tuple[Mapping[str, Any], str, str, Path]:
    raw, file_sha256, resolved = _load_json_artifact(path, "fixture report")
    if not isinstance(raw, Mapping):
        raise EvidenceError("fixture report root must be an object")
    try:
        fingerprint = enoch_week1_fixtures.validate_report(raw)
    except enoch_week1_fixtures.FixtureError as exc:
        raise EvidenceError(f"fixture report is invalid: {exc}") from exc
    root = resolved.parent
    for record in raw["records"]:
        log_path = (root / record["log_path"]).resolve()
        if root != log_path and root not in log_path.parents:
            raise EvidenceError("fixture log path escapes its artifact directory")
        try:
            log_hash = _file_sha256_bytes(log_path.read_bytes())
        except OSError as exc:
            raise EvidenceError(f"fixture log is missing: {log_path}") from exc
        if log_hash != record["output_sha256"]:
            raise EvidenceError(f"fixture log hash mismatch: {record['log_path']}")
    return raw, fingerprint, file_sha256, resolved


def _validate_model_contract_artifact(artifact_id: str, value: Any) -> None:
    if not isinstance(value, Mapping):
        raise EvidenceError(f"model-contract artifact {artifact_id} must be an object")
    _exact_keys(value, {"command_succeeded", "output"}, artifact_id)
    if value["command_succeeded"] is not True or not isinstance(value["output"], str):
        raise EvidenceError(f"model-contract artifact {artifact_id} did not succeed")
    for marker in _MODEL_MARKERS[artifact_id]:
        if marker not in value["output"]:
            raise EvidenceError(
                f"model-contract artifact {artifact_id} lacks required test marker"
            )


def _validate_runner_identities(
    comparison: Mapping[str, Any],
    control_manifest: Mapping[str, Any],
    identities: Any,
) -> str:
    if not isinstance(identities, Mapping):
        raise EvidenceError("runner identities must be an object")
    _exact_keys(identities, {"candidate", "control", "evaluator"}, "runner identities")
    for name in ("candidate", "control"):
        identity = identities[name]
        if not isinstance(identity, Mapping):
            raise EvidenceError(f"runner {name} identity must be an object")
        _exact_keys(identity, set(enoch_week1.POLICY_IDENTITY_FIELDS), f"runner {name}")
        for field in enoch_week1.POLICY_IDENTITY_FIELDS:
            _sha256(identity[field], f"runner {name} {field}")
    evaluator = identities["evaluator"]
    if not isinstance(evaluator, Mapping):
        raise EvidenceError("runner evaluator identity must be an object")
    _exact_keys(
        evaluator, set(enoch_week1.EVALUATOR_IDENTITY_FIELDS), "runner evaluator"
    )
    for field in enoch_week1.EVALUATOR_IDENTITY_FIELDS:
        _sha256(evaluator[field], f"runner evaluator {field}")
    for name in ("candidate", "control", "evaluator"):
        if enoch_week1.canonical_json_sha256(identities[name]) != comparison.get(
            f"{name}_fingerprint"
        ):
            raise EvidenceError(f"runner {name} identity differs from the comparison")
    if dict(evaluator) != dict(control_manifest["evaluator_identity"]):
        raise EvidenceError("runner evaluator identity differs from frozen W1.0")
    for name in ("candidate", "control"):
        if (
            identities[name]["binary_sha256"] != evaluator["binary_sha256"]
            or identities[name]["source_sha256"] != evaluator["source_sha256"]
        ):
            raise EvidenceError(f"runner {name} is not compiled into the frozen evaluator")
    if identities["candidate"]["model_sha256"] != identities["control"]["model_sha256"]:
        raise EvidenceError("runner candidate/control model identities differ")
    return enoch_week1.canonical_json_sha256(identities)


def _counter_records() -> dict[str, dict[str, Any]]:
    fixture_authorities = [FIXTURE_REPORT_ARTIFACT, SOURCE_IDENTITY_ARTIFACT]
    return {
        "artifact_mismatch": {
            "artifact_ids": [
                CONTROL_MANIFEST_ARTIFACT,
                RUNNER_IDENTITIES_ARTIFACT,
                SOURCE_IDENTITY_ARTIFACT,
            ],
            "count": 0,
            "validator_id": "w1.0-runner-identity-binding-v1",
        },
        "fixture_failure": {
            "artifact_ids": fixture_authorities,
            "count": 0,
            "validator_id": "sealed-fixture-report-v1",
        },
        "hidden_information_leak": {
            "artifact_ids": fixture_authorities,
            "count": 0,
            "validator_id": "fixture-hidden-information-source-v1",
        },
        "honesty_violation": {
            "artifact_ids": fixture_authorities,
            "count": 0,
            "validator_id": "fixture-honesty-source-v1",
        },
        "machine_contention": {
            "artifact_ids": [MACHINE_ATTESTATION_ARTIFACT],
            "count": 0,
            "validator_id": "operator-machine-contention-attestation-v1",
        },
        "model_contract_failure": {
            "artifact_ids": sorted(
                (
                    FIXTURE_REPORT_ARTIFACT,
                    SOURCE_IDENTITY_ARTIFACT,
                    CONTROL_MANIFEST_ARTIFACT,
                    *MODEL_CONTRACT_ARTIFACTS,
                )
            ),
            "count": 0,
            "validator_id": "w1.0-model-contract-and-fixture-source-v1",
        },
    }


def _validate_artifact_paths(
    protocol: Mapping[str, Any],
    comparison: Mapping[str, Any],
    *,
    fixture_report_path: Path,
    source_identity_path: Path,
    control_manifest_path: Path,
    runner_identities_path: Path,
    model_contract_artifact_paths: Mapping[str, Path],
    machine_attestation_path: Path,
) -> list[dict[str, str]]:
    enoch_week1.validate_protocol(protocol)
    enoch_week1.validate_comparison_protocol_manifest(protocol, comparison)
    if not isinstance(model_contract_artifact_paths, Mapping) or set(
        model_contract_artifact_paths
    ) != set(MODEL_CONTRACT_ARTIFACTS):
        raise EvidenceError("model-contract artifact path set is incomplete")

    fixture, fixture_fingerprint, fixture_file_hash, fixture_path = (
        _validate_fixture_bundle(fixture_report_path)
    )
    source_value, source_file_hash, source_path = _load_json_artifact(
        source_identity_path, "fixture source identity"
    )
    source_records, source_semantic_hash = _validate_source_identity(source_value)
    fixture_sources = {item["path"]: item["sha256"] for item in fixture["source_files"]}
    for record in source_records:
        if fixture_sources.get(record["path"]) != record["sha256"]:
            raise EvidenceError(
                f"fixture report source differs from frozen evaluator: {record['path']}"
            )

    control_value, control_file_hash, control_path = _load_json_artifact(
        control_manifest_path, "W1.0 control manifest"
    )
    if not isinstance(control_value, Mapping):
        raise EvidenceError("W1.0 control manifest root must be an object")
    try:
        control_fingerprint = enoch_week1.validate_w1_0_control_manifest(
            protocol, control_value
        )
    except enoch_week1.ProtocolError as exc:
        raise EvidenceError(f"W1.0 control manifest is invalid: {exc}") from exc
    if (
        control_value["artifact_hashes"]["source/week1-evaluator-file-list"]
        != source_file_hash
        or control_value["evaluator_identity"]["source_sha256"] != source_file_hash
    ):
        raise EvidenceError("fixture source identity is not the frozen W1.0 evaluator source")

    identities_value, identities_file_hash, identities_path = _load_json_artifact(
        runner_identities_path, "runner identity bindings"
    )
    identities_fingerprint = _validate_runner_identities(
        comparison, control_value, identities_value
    )

    machine_value, machine_file_hash, machine_path = _load_json_artifact(
        machine_attestation_path, "machine contention attestation"
    )
    if not isinstance(machine_value, Mapping):
        raise EvidenceError("machine contention attestation root must be an object")
    machine_fingerprint = validate_machine_contention_attestation(
        comparison, machine_value
    )

    references = [
        _artifact_reference(
            FIXTURE_REPORT_ARTIFACT,
            "sealed-fixture-report",
            fixture_path,
            fixture_file_hash,
            fixture_fingerprint,
        ),
        _artifact_reference(
            SOURCE_IDENTITY_ARTIFACT,
            "frozen-evaluator-source-list",
            source_path,
            source_file_hash,
            source_semantic_hash,
        ),
        _artifact_reference(
            CONTROL_MANIFEST_ARTIFACT,
            "w1.0-control-manifest",
            control_path,
            control_file_hash,
            control_fingerprint,
        ),
        _artifact_reference(
            RUNNER_IDENTITIES_ARTIFACT,
            "runner-identity-bindings",
            identities_path,
            identities_file_hash,
            identities_fingerprint,
        ),
        _artifact_reference(
            MACHINE_ATTESTATION_ARTIFACT,
            "machine-contention-attestation",
            machine_path,
            machine_file_hash,
            machine_fingerprint,
        ),
    ]
    for artifact_id in MODEL_CONTRACT_ARTIFACTS:
        value, file_hash, path = _load_json_artifact(
            model_contract_artifact_paths[artifact_id], artifact_id
        )
        _validate_model_contract_artifact(artifact_id, value)
        if control_value["artifact_hashes"].get(artifact_id) != file_hash:
            raise EvidenceError(
                f"model-contract artifact {artifact_id} differs from frozen W1.0"
            )
        references.append(
            _artifact_reference(
                artifact_id,
                "w1.0-model-contract-preflight",
                path,
                file_hash,
                file_hash,
            )
        )
    references.sort(key=lambda item: item["artifact_id"])
    return references


def build_verified_external_evidence(
    protocol: Mapping[str, Any],
    comparison: Mapping[str, Any],
    *,
    fixture_report_path: Path,
    source_identity_path: Path,
    control_manifest_path: Path,
    runner_identities_path: Path,
    model_contract_artifact_paths: Mapping[str, Path],
    machine_attestation_path: Path,
) -> dict[str, Any]:
    """Validate actual artifacts and seal their derived zero counters."""

    artifacts = _validate_artifact_paths(
        protocol,
        comparison,
        fixture_report_path=fixture_report_path,
        source_identity_path=source_identity_path,
        control_manifest_path=control_manifest_path,
        runner_identities_path=runner_identities_path,
        model_contract_artifact_paths=model_contract_artifact_paths,
        machine_attestation_path=machine_attestation_path,
    )
    body = {
        "artifacts": artifacts,
        "automatic_production_promotion_allowed": False,
        "comparison_protocol_fingerprint": comparison[
            "comparison_protocol_fingerprint"
        ],
        "counters": _counter_records(),
        "environment_fingerprint": comparison["environment_fingerprint"],
        "manifest_kind": EVIDENCE_KIND,
        "manifest_version": MANIFEST_VERSION,
        "protocol_fingerprint": protocol["protocol_fingerprint"],
    }
    evidence = {
        **body,
        "verified_external_evidence_fingerprint": enoch_week1.canonical_json_sha256(
            body
        ),
    }
    validate_verified_external_evidence(protocol, comparison, evidence)
    return evidence


def validate_verified_external_evidence(
    protocol: Mapping[str, Any],
    comparison: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    """Re-open every referenced path and rederive all external zero counters."""

    if not isinstance(evidence, Mapping):
        raise EvidenceError("verified external evidence must be an object")
    _exact_keys(
        evidence,
        {
            "artifacts",
            "automatic_production_promotion_allowed",
            "comparison_protocol_fingerprint",
            "counters",
            "environment_fingerprint",
            "manifest_kind",
            "manifest_version",
            "protocol_fingerprint",
            "verified_external_evidence_fingerprint",
        },
        "verified external evidence",
    )
    if evidence["manifest_kind"] != EVIDENCE_KIND or evidence[
        "manifest_version"
    ] != MANIFEST_VERSION:
        raise EvidenceError("unsupported verified external evidence")
    if evidence["automatic_production_promotion_allowed"] is not False:
        raise EvidenceError("external evidence cannot authorize promotion")
    if evidence["protocol_fingerprint"] != protocol.get("protocol_fingerprint"):
        raise EvidenceError("external evidence belongs to another protocol")
    if evidence["comparison_protocol_fingerprint"] != comparison.get(
        "comparison_protocol_fingerprint"
    ):
        raise EvidenceError("external evidence belongs to another comparison")
    if evidence["environment_fingerprint"] != comparison.get("environment_fingerprint"):
        raise EvidenceError("external evidence belongs to another environment")
    fingerprint = _sha256(
        evidence["verified_external_evidence_fingerprint"],
        "verified external evidence fingerprint",
    )
    body = dict(evidence)
    body.pop("verified_external_evidence_fingerprint")
    if fingerprint != enoch_week1.canonical_json_sha256(body):
        raise EvidenceError("verified external evidence fingerprint mismatch")
    artifacts = evidence["artifacts"]
    if not isinstance(artifacts, list):
        raise EvidenceError("verified external artifacts must be a list")
    artifact_keys = {
        "artifact_id",
        "artifact_kind",
        "file_sha256",
        "path",
        "semantic_fingerprint",
    }
    for record in artifacts:
        if not isinstance(record, Mapping):
            raise EvidenceError("verified artifact reference must be an object")
        _exact_keys(record, artifact_keys, "verified artifact reference")
        if not isinstance(record["artifact_id"], str) or not record["artifact_id"]:
            raise EvidenceError("verified artifact id must be a nonempty string")
        if not isinstance(record["artifact_kind"], str) or not record["artifact_kind"]:
            raise EvidenceError("verified artifact kind must be a nonempty string")
        if not isinstance(record["path"], str) or not record["path"]:
            raise EvidenceError("verified artifact path must be a nonempty string")
        _sha256(record["file_sha256"], "verified artifact file hash")
        _sha256(
            record["semantic_fingerprint"],
            "verified artifact semantic fingerprint",
        )
    if [record["artifact_id"] for record in artifacts] != list(EXPECTED_ARTIFACT_IDS):
        raise EvidenceError("verified external artifact set is incomplete or unsorted")
    paths = {record["artifact_id"]: Path(record["path"]) for record in artifacts}
    expected_artifacts = _validate_artifact_paths(
        protocol,
        comparison,
        fixture_report_path=paths[FIXTURE_REPORT_ARTIFACT],
        source_identity_path=paths[SOURCE_IDENTITY_ARTIFACT],
        control_manifest_path=paths[CONTROL_MANIFEST_ARTIFACT],
        runner_identities_path=paths[RUNNER_IDENTITIES_ARTIFACT],
        model_contract_artifact_paths={
            artifact_id: paths[artifact_id]
            for artifact_id in MODEL_CONTRACT_ARTIFACTS
        },
        machine_attestation_path=paths[MACHINE_ATTESTATION_ARTIFACT],
    )
    if artifacts != expected_artifacts:
        raise EvidenceError("verified external artifact references do not reconstruct")
    expected_counters = _counter_records()
    if evidence["counters"] != expected_counters:
        raise EvidenceError("verified external counters do not reconstruct")
    return dict(evidence["counters"])
