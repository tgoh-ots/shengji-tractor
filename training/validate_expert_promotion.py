#!/usr/bin/env python3
"""Fail closed when committed Expert promotion evidence drifts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (defaults to this script's parent repository)",
    )
    args = parser.parse_args()
    root = args.repo.resolve()
    bot = root / "core/src/bot"
    evidence = root / "docs/strategy/artifacts"

    model = bot / "expert_model.onnx"
    serving_manifest_path = bot / "expert_model.onnx.manifest.json"
    training_manifest_path = bot / "expert_model.training.manifest.json"
    golden = bot / "expert_model.onnx.golden.json"
    attestation_path = bot / "expert_model.promotion.json"
    comparison_path = evidence / "policy-only-confirmation-2026-07-02.comparison.json"
    audit_path = evidence / "policy-only-confirmation-2026-07-02.audit.json"
    protocol_path = evidence / "policy-only-confirmation-2026-07-02.protocol.json"

    attestation = load(attestation_path)
    candidate = attestation["candidate"]
    production = attestation["production_artifact"]
    confirmation = attestation["confirmation"]
    approval = attestation["approval"]
    serving_manifest = load(serving_manifest_path)
    training_manifest = load(training_manifest_path)
    comparison = load(comparison_path)
    audit = load(audit_path)
    protocol = load(protocol_path)

    require(attestation["manifest_version"] == 1, "unsupported promotion manifest")
    require(approval["authority"] == "explicit_human", "promotion lacks human approval")
    require(approval["instruction"] == "commit then deploy", "unexpected approval instruction")
    require(
        approval["automatic_production_promotion_allowed"] is False,
        "automatic production promotion must remain disabled",
    )
    require(sha256(model) == candidate["model_sha256"], "promoted model hash drift")
    require(sha256(golden) == candidate["golden_sha256"], "golden vector hash drift")
    require(
        sha256(training_manifest_path) == candidate["training_manifest_sha256"],
        "immutable training manifest hash drift",
    )
    require(
        sha256(serving_manifest_path) == production["manifest_sha256"],
        "production serving manifest hash drift",
    )
    require(
        training_manifest["model_sha256"] == candidate["model_sha256"],
        "training manifest points at a different model",
    )
    require(
        training_manifest["golden_sha256"] == candidate["golden_sha256"],
        "training manifest points at different golden vectors",
    )
    require(training_manifest["outputs"] == candidate["outputs"] == ["score"], "candidate is not policy-only")
    require(training_manifest["feature_schema_version"] == 2, "candidate schema version drift")
    require(training_manifest["feature_dim"] == 49, "candidate feature width drift")
    require(training_manifest["training"]["policy_weight"] == 1, "policy objective drift")
    require(training_manifest["training"]["value_weight"] == 0, "value objective must be absent")
    require(training_manifest["training"]["q_weight"] == 0, "Q objective must be absent")
    require(training_manifest["training"]["auxiliary_weight"] == 0, "auxiliary objective must be absent")
    require(
        serving_manifest["model_sha256"] == candidate["model_sha256"],
        "serving manifest points at a different model",
    )
    require(
        serving_manifest["golden_sha256"] == candidate["golden_sha256"],
        "serving manifest points at different golden vectors",
    )
    require(
        serving_manifest["promotion"]["source_candidate_manifest_sha256"]
        == candidate["training_manifest_sha256"],
        "serving manifest lost candidate-manifest lineage",
    )
    require(
        serving_manifest["promotion"]["attestation_path"] == attestation_path.name,
        "serving manifest points at the wrong promotion attestation",
    )
    require(serving_manifest["outputs"] == ["score"], "production model must be policy-only")
    require(
        serving_manifest["output_semantics"] == ["policy_logit"],
        "unexpected production output semantics",
    )
    require(serving_manifest["research_only"] is False, "serving manifest is not production")
    require(
        serving_manifest["serving_status"] == "embedded_production",
        "serving status is not embedded_production",
    )
    require(
        serving_manifest["automatic_production_promotion_allowed"] is False,
        "serving manifest must not authorize automatic promotion",
    )
    require(training_manifest["research_only"] is True, "training provenance was rewritten")
    require(
        training_manifest["serving_status"] == "experimental_candidate",
        "training manifest no longer describes the evaluated candidate",
    )
    expected_serving = copy.deepcopy(training_manifest)
    expected_serving["golden_path"] = golden.name
    expected_serving["research_only"] = False
    expected_serving["serving_status"] = "embedded_production"
    expected_serving["promotion"] = {
        "approval": "explicit_human",
        "attestation_path": attestation_path.name,
        "source_candidate_manifest_sha256": candidate["training_manifest_sha256"],
    }
    require(
        serving_manifest == expected_serving,
        "serving manifest differs from training manifest beyond approved promotion fields",
    )

    require(sha256(comparison_path) == confirmation["comparison_sha256"], "comparison hash drift")
    require(sha256(audit_path) == confirmation["audit_sha256"], "completion audit hash drift")
    require(sha256(protocol_path) == confirmation["protocol_sha256"], "protocol hash drift")
    require(comparison["pairs"] == confirmation["pairs_per_arm"], "pair count drift")
    require(comparison["seed"] == confirmation["seed"], "evaluation seed drift")
    require(comparison["budget_ms"] == confirmation["budget_ms"], "budget drift")
    require(
        comparison["candidate_model_sha256"] == candidate["model_sha256"],
        "comparison evaluated a different model",
    )
    require(
        comparison["candidate_manifest_sha256"] == candidate["training_manifest_sha256"],
        "comparison evaluated a different candidate manifest",
    )
    require(
        comparison["golden_sha256"] == candidate["golden_sha256"],
        "comparison validated different golden vectors",
    )
    require(comparison["baseline_kind"] == "embedded-model", "confirmation baseline kind drift")
    require(comparison["baseline_model_sha256"] is None, "confirmation did not use embedded baseline")
    gate = comparison["promotion_gate"]
    require(gate["passed"] is True, "committed comparison did not pass")
    require(gate["metric"] == confirmation["gate"]["metric"], "gate metric drift")
    require(gate["minimum_level_delta"] == confirmation["gate"]["minimum"], "gate threshold drift")
    require(gate["observed_lower95"] == confirmation["gate"]["observed"], "gate result drift")

    require(protocol["manifest_version"] == 1, "protocol version drift")
    require(protocol["research_only"] is True, "confirmation protocol was not research-only")
    require(
        protocol["automatic_production_promotion_allowed"] is False,
        "confirmation protocol allowed automatic promotion",
    )
    require(protocol["lineage"]["source_commit"] == candidate["source_commit"], "source lineage drift")
    require(protocol["candidate"]["model_sha256"] == candidate["model_sha256"], "protocol model drift")
    require(
        protocol["candidate"]["manifest_sha256"] == candidate["training_manifest_sha256"],
        "protocol candidate-manifest drift",
    )
    require(protocol["candidate"]["golden_sha256"] == candidate["golden_sha256"], "protocol golden drift")
    require(
        protocol["baseline"]["model_sha256"]
        == attestation["rollback"]["previous_embedded_model_sha256"],
        "protocol baseline model drift",
    )
    require(protocol["evaluation"]["pairs_per_arm"] == confirmation["pairs_per_arm"], "protocol pairs drift")
    require(int(protocol["evaluation"]["seed"], 16) == confirmation["seed"], "protocol seed drift")
    require(protocol["evaluation"]["budget_ms"] == confirmation["budget_ms"], "protocol budget drift")
    require(
        protocol["decision_rule"]["minimum_level_delta"] == confirmation["gate"]["minimum"],
        "protocol threshold drift",
    )

    require(audit["manifest_version"] == 1, "completion audit version drift")
    require(audit["research_only"] is True, "completion audit lost research-only status")
    require(
        audit["automatic_production_promotion_allowed"] is False,
        "completion audit allowed automatic promotion",
    )
    require(audit["promotion_or_deployment_performed"] is False, "confirmation self-promoted")
    require(audit["status"] == "complete-gate-passed-no-promotion", "completion status drift")
    require(audit["candidate"]["model_sha256"] == candidate["model_sha256"], "audit model drift")
    require(
        audit["candidate"]["manifest_sha256"] == candidate["training_manifest_sha256"],
        "audit candidate-manifest drift",
    )
    require(audit["candidate"]["golden_sha256"] == candidate["golden_sha256"], "audit golden drift")
    require(audit["evaluation"]["pairs_per_arm"] == confirmation["pairs_per_arm"], "audit pairs drift")
    require(audit["evaluation"]["seed"] == confirmation["seed"], "audit seed drift")
    require(audit["evaluation"]["budget_ms"] == confirmation["budget_ms"], "audit budget drift")
    require(audit["evaluation"]["candidate_failed_hands"] == 0, "candidate arm reported failures")
    require(audit["evaluation"]["embedded_failed_hands"] == 0, "embedded arm reported failures")
    require(audit["gate"]["passed"] is True, "completion audit gate did not pass")
    require(audit["gate"]["minimum_level_delta"] == confirmation["gate"]["minimum"], "audit threshold drift")
    require(audit["gate"]["observed_lower95"] == confirmation["gate"]["observed"], "audit result drift")
    require(
        audit["lineage"]["source_commit"] == candidate["source_commit"],
        "audit source lineage drift",
    )
    require(
        audit["lineage"]["embedded_model_sha256_after_run"]
        == attestation["rollback"]["previous_embedded_model_sha256"],
        "audit embedded baseline drift",
    )
    require(
        audit["artifacts"]["comparison_sha256"] == confirmation["comparison_sha256"],
        "audit comparison hash drift",
    )
    require(
        audit["artifacts"]["protocol_sha256"] == confirmation["protocol_sha256"],
        "audit protocol hash drift",
    )

    print(
        "PASS: promoted Expert model, serving/training manifests, golden vectors, "
        "human approval, and confirmation evidence are hash-consistent"
    )


if __name__ == "__main__":
    main()
