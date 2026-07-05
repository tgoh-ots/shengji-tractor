#!/usr/bin/env python3
"""Self-contained W1.4 combination-campaign lineage artifacts.

The comparison schema deliberately describes one experiment at a time.  This
module closes the cross-comparison gap for W1.4: one canonical artifact proves
that the short combination qualification and the 800-pair combination screen
run the same supported candidate, and that every arm in that candidate first
survived an independent development campaign.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Mapping, Sequence

try:  # Support package imports and direct ``python training/...`` imports.
    from . import enoch_week1, enoch_week1_runner
except ImportError:  # pragma: no cover - direct-script import path.
    import enoch_week1  # type: ignore[no-redef]
    import enoch_week1_runner  # type: ignore[no-redef]


MANIFEST_VERSION = 1
MANIFEST_KIND = "enoch-week1-w1.4-campaign-lineage"
FINGERPRINT_FIELD = "w1_4_campaign_lineage_fingerprint"
CANDIDATE_DECISION_KIND = "enoch-week1-w1.4-candidate-decision"
CANDIDATE_DECISION_FINGERPRINT_FIELD = "w1_4_candidate_decision_fingerprint"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

W1_4_STAGES: tuple[tuple[str, str, int, int], ...] = (
    ("qualification", "dev/combination/qualification", 200, 300),
    ("screen", "dev/combination/screen", 800, 800),
)

_RUN_KEYS = {
    "comparison",
    "identity_bindings",
    "launch_configuration",
    "merged_result",
}
_STAGE_KEYS = {
    "comparison",
    "identity_bindings",
    "launch_configuration",
    "stage_id",
}
_SURVIVOR_KEYS = {
    "ablation",
    "advancement_decision",
    "arm_id",
    "fixture_report",
    "survivor_screen",
}


def _fail(message: str) -> None:
    raise enoch_week1.ProtocolError(message)


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        _fail(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        _fail(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validate_launch_identity_run(
    protocol: Mapping[str, Any],
    run: Mapping[str, Any],
    *,
    label: str,
    expected_phase: str,
    expected_arm_ids: Sequence[str],
    require_result: bool,
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any] | None,
]:
    expected_keys = _RUN_KEYS if require_result else _RUN_KEYS - {"merged_result"}
    _require_exact_keys(run, expected_keys, label)
    comparison = run["comparison"]
    launch = run["launch_configuration"]
    identities = run["identity_bindings"]

    enoch_week1.validate_comparison_protocol_manifest(protocol, comparison)
    if comparison["phase"] != expected_phase:
        _fail(f"{label} must be a {expected_phase} comparison")

    launch_fingerprint = enoch_week1_runner.validate_launch_configuration(launch)
    if launch_fingerprint != comparison["configuration_fingerprint"]:
        _fail(f"{label} launch does not match its comparison configuration")
    if launch["candidate_arm_ids"] != list(expected_arm_ids):
        _fail(f"{label} does not run its exact canonical candidate arm set")
    if launch["work_mode"] != "fixed-work":
        _fail(f"{label} must use deterministic fixed-work mode")
    if expected_phase in {"W1.2", "W1.3"} and list(expected_arm_ids) == [
        "friend-revelation"
    ]:
        expected_scenario = enoch_week1_runner.DEVELOPMENT_FRIEND_SCENARIO
    else:
        expected_scenario = "standard"
    if launch["scenario_id"] != expected_scenario:
        _fail(f"{label} must use the frozen {expected_scenario!r} scenario")

    _require_exact_keys(
        identities, {"candidate", "control", "evaluator"}, f"{label} identities"
    )
    expected_identities = enoch_week1_runner.build_in_process_identity_bindings(
        identities["evaluator"], launch
    )
    for identity_name in ("candidate", "control", "evaluator"):
        if dict(identities[identity_name]) != expected_identities[identity_name]:
            _fail(
                f"{label} {identity_name} identity is not the policy compiled into "
                "its evaluator and launch"
            )
        actual_fingerprint = enoch_week1.canonical_json_sha256(
            identities[identity_name]
        )
        if actual_fingerprint != comparison[f"{identity_name}_fingerprint"]:
            _fail(f"{label} {identity_name} identity fingerprint mismatch")

    merged = run.get("merged_result")
    if require_result:
        enoch_week1.validate_merged_result(protocol, comparison, merged)
    return comparison, launch, identities, merged


def _identity_common_fields(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: identity[field]
        for field in ("binary_sha256", "model_sha256", "source_sha256")
    }


def _validate_survivor(
    protocol: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    _require_exact_keys(evidence, _SURVIVOR_KEYS, "W1.4 survivor evidence")
    arm_id = evidence["arm_id"]
    if arm_id not in enoch_week1.ABLATION_ARMS:
        _fail(f"W1.4 survivor evidence names an unknown arm: {arm_id!r}")

    ablation_comparison, _, ablation_identities, ablation_merged = (
        _validate_launch_identity_run(
            protocol,
            evidence["ablation"],
            label=f"{arm_id} ablation",
            expected_phase="W1.2",
            expected_arm_ids=[arm_id],
            require_result=True,
        )
    )
    if ablation_comparison["subject_id"] != arm_id:
        _fail(f"{arm_id} ablation subject does not name its independent arm")

    decision = evidence["advancement_decision"]
    enoch_week1.validate_w1_3_advancement_decision(
        protocol,
        ablation_comparison,
        ablation_merged,
        evidence["fixture_report"],
        decision,
    )
    if decision["decision"] != "advance-to-w1.3":
        _fail(f"{arm_id} did not pass its predeclared W1.3 advancement rule")

    survivor_comparison, _, survivor_identities, survivor_merged = (
        _validate_launch_identity_run(
            protocol,
            evidence["survivor_screen"],
            label=f"{arm_id} survivor screen",
            expected_phase="W1.3",
            expected_arm_ids=[arm_id],
            require_result=True,
        )
    )
    if survivor_comparison["subject_id"] != arm_id:
        _fail(f"{arm_id} survivor-screen subject does not name its independent arm")
    if survivor_comparison["pair_count"] != 800:
        _fail(f"{arm_id} survivor screen must contain exactly 800 matched pairs")
    if survivor_comparison["development_rule"] != ablation_comparison[
        "development_rule"
    ]:
        _fail(f"{arm_id} survivor screen changed its predeclared development rule")

    for field in ("candidate", "control", "evaluator"):
        if dict(ablation_identities[field]) != dict(survivor_identities[field]):
            _fail(f"{arm_id} changed {field} identity before its survivor screen")
    if ablation_comparison["environment_fingerprint"] != survivor_comparison[
        "environment_fingerprint"
    ]:
        _fail(f"{arm_id} changed environment before its survivor screen")

    failures = enoch_week1._development_rule_failures(  # Internal shared rule semantics.
        ablation_comparison["development_rule"], survivor_merged["metrics"]
    )
    if failures:
        _fail(
            f"{arm_id} is not an independently supported 800-pair survivor: "
            + ", ".join(failures)
        )

    return {
        "ablation_comparison_protocol_fingerprint": ablation_comparison[
            "comparison_protocol_fingerprint"
        ],
        "ablation_merged_result_fingerprint": ablation_merged[
            "merged_result_fingerprint"
        ],
        "advancement_decision_fingerprint": decision[
            "advancement_decision_fingerprint"
        ],
        "arm_id": arm_id,
        "candidate_identity": dict(survivor_identities["candidate"]),
        "control_identity": dict(survivor_identities["control"]),
        "environment_fingerprint": survivor_comparison["environment_fingerprint"],
        "evaluator_identity": dict(survivor_identities["evaluator"]),
        "fixture_report_fingerprint": evidence["fixture_report"][
            "fixture_report_fingerprint"
        ],
        "survivor_comparison_protocol_fingerprint": survivor_comparison[
            "comparison_protocol_fingerprint"
        ],
        "survivor_merged_result_fingerprint": survivor_merged[
            "merged_result_fingerprint"
        ],
    }


def _campaign_body(
    protocol: Mapping[str, Any],
    stages: Sequence[Mapping[str, Any]],
    survivor_evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    protocol_fingerprint = enoch_week1.validate_protocol(protocol)
    if protocol["evaluator_environment_policy"]["allowlist"]:
        _fail("W1.4 campaign requires an empty evaluator environment allowlist")
    if not isinstance(stages, Sequence) or isinstance(stages, (str, bytes)):
        _fail("W1.4 stages must be a sequence")
    if len(stages) != len(W1_4_STAGES):
        _fail("W1.4 campaign must contain exactly qualification and screen stages")

    validated_stages: list[dict[str, Any]] = []
    stage_identity_bindings: list[Mapping[str, Any]] = []
    stage_launches: list[Mapping[str, Any]] = []
    stage_comparisons: list[Mapping[str, Any]] = []
    candidate_arm_ids: list[str] | None = None
    for supplied, (stage_id, namespace, minimum_pairs, maximum_pairs) in zip(
        stages, W1_4_STAGES
    ):
        _require_exact_keys(supplied, _STAGE_KEYS, f"W1.4 {stage_id} stage")
        if supplied["stage_id"] != stage_id:
            _fail("W1.4 stages are missing or not in canonical order")
        supplied_launch = supplied["launch_configuration"]
        if not isinstance(supplied_launch, Mapping):
            _fail(f"W1.4 {stage_id} launch configuration must be an object")
        launch_arms = supplied_launch.get("candidate_arm_ids")
        if not isinstance(launch_arms, list) or not launch_arms:
            _fail("W1.4 candidate arm set must be nonempty")
        comparison, launch, identities, _ = _validate_launch_identity_run(
            protocol,
            {
                "comparison": supplied["comparison"],
                "identity_bindings": supplied["identity_bindings"],
                "launch_configuration": supplied["launch_configuration"],
            },
            label=f"W1.4 {stage_id} stage",
            expected_phase="W1.4",
            expected_arm_ids=launch_arms,
            require_result=False,
        )
        if comparison["seed_namespace"] != namespace:
            _fail(f"W1.4 {stage_id} stage uses the wrong seed namespace")
        if not minimum_pairs <= comparison["pair_count"] <= maximum_pairs:
            _fail(f"W1.4 {stage_id} stage uses the wrong exact pair count")
        if candidate_arm_ids is None:
            candidate_arm_ids = list(launch_arms)
        elif list(launch_arms) != candidate_arm_ids:
            _fail("W1.4 stages do not share one identical candidate arm set")
        stage_comparisons.append(comparison)
        stage_launches.append(launch)
        stage_identity_bindings.append(identities)
        validated_stages.append(
            {
                "comparison": copy.deepcopy(dict(comparison)),
                "identity_bindings": copy.deepcopy(dict(identities)),
                "launch_configuration": copy.deepcopy(dict(launch)),
                "stage_id": stage_id,
            }
        )

    assert candidate_arm_ids is not None  # Established by the exact two-stage loop.
    if dict(stage_launches[0]) != dict(stage_launches[1]):
        _fail("W1.4 stages do not share one identical launch configuration")
    if dict(stage_identity_bindings[0]) != dict(stage_identity_bindings[1]):
        _fail("W1.4 stages do not share candidate/control/evaluator identities")
    if stage_comparisons[0]["environment_fingerprint"] != stage_comparisons[1][
        "environment_fingerprint"
    ]:
        _fail("W1.4 stages do not share one identical environment")
    if stage_comparisons[0]["subject_id"] != stage_comparisons[1]["subject_id"]:
        _fail("W1.4 stages do not share one candidate subject")
    if stage_comparisons[0]["comparison_id"] == stage_comparisons[1]["comparison_id"]:
        _fail("W1.4 stages must have distinct comparison ids")
    if stage_comparisons[0]["seed_namespace"] == stage_comparisons[1]["seed_namespace"]:
        _fail("W1.4 stages must use distinct seed namespaces")
    if stage_comparisons[0]["seed_set_sha256"] == stage_comparisons[1][
        "seed_set_sha256"
    ]:
        _fail("W1.4 stages must use distinct seed sets")

    if not isinstance(survivor_evidence, Sequence) or isinstance(
        survivor_evidence, (str, bytes)
    ):
        _fail("W1.4 survivor evidence must be a sequence")
    if not survivor_evidence:
        _fail("W1.4 candidate requires at least one independently supported survivor")
    if any(not isinstance(entry, Mapping) for entry in survivor_evidence):
        _fail("W1.4 survivor evidence entries must be objects")
    supplied_arm_ids = [entry.get("arm_id") for entry in survivor_evidence]
    if supplied_arm_ids != candidate_arm_ids:
        _fail(
            "W1.4 survivor evidence must be unique, canonical, and exactly match "
            "the candidate arm set"
        )
    if len(set(supplied_arm_ids)) != len(supplied_arm_ids):
        _fail("W1.4 survivor evidence contains duplicate arms")

    survivor_summaries = [
        _validate_survivor(protocol, evidence) for evidence in survivor_evidence
    ]
    combined_identities = stage_identity_bindings[0]
    combined_environment = stage_comparisons[0]["environment_fingerprint"]
    fixture_fingerprints = {
        summary["fixture_report_fingerprint"] for summary in survivor_summaries
    }
    if len(fixture_fingerprints) != 1:
        _fail("W1.4 survivors do not share one sealed fixture report")
    for summary in survivor_summaries:
        if summary["control_identity"] != dict(combined_identities["control"]):
            _fail("W1.4 candidate control identity differs from survivor evidence")
        if summary["evaluator_identity"] != dict(combined_identities["evaluator"]):
            _fail("W1.4 candidate evaluator identity differs from survivor evidence")
        if summary["environment_fingerprint"] != combined_environment:
            _fail("W1.4 candidate environment differs from survivor evidence")
        if _identity_common_fields(summary["candidate_identity"]) != _identity_common_fields(
            combined_identities["candidate"]
        ):
            _fail("W1.4 candidate executable/model/source differs from survivor evidence")

    body = {
        "arm_registry_sha256": enoch_week1.ARM_REGISTRY_SHA256,
        "automatic_production_promotion_allowed": False,
        "candidate_arm_ids": list(candidate_arm_ids),
        "candidate_fingerprint": stage_comparisons[0]["candidate_fingerprint"],
        "control_fingerprint": stage_comparisons[0]["control_fingerprint"],
        "environment_fingerprint": combined_environment,
        "evaluator_fingerprint": stage_comparisons[0]["evaluator_fingerprint"],
        "fixture_report_fingerprint": next(iter(fixture_fingerprints)),
        "launch_configuration_sha256": enoch_week1.canonical_json_sha256(
            stage_launches[0]
        ),
        "manifest_kind": MANIFEST_KIND,
        "manifest_version": MANIFEST_VERSION,
        "protocol_fingerprint": protocol_fingerprint,
        "stages": validated_stages,
        "subject_id": stage_comparisons[0]["subject_id"],
        "survivor_evidence": copy.deepcopy([dict(item) for item in survivor_evidence]),
        "survivor_summaries": [
            {
                key: value
                for key, value in summary.items()
                if key
                not in {
                    "candidate_identity",
                    "control_identity",
                    "environment_fingerprint",
                    "evaluator_identity",
                }
            }
            for summary in survivor_summaries
        ],
    }
    return body


def build_w1_4_campaign_lineage(
    protocol: Mapping[str, Any],
    *,
    qualification_comparison: Mapping[str, Any],
    qualification_launch_configuration: Mapping[str, Any],
    qualification_identity_bindings: Mapping[str, Any],
    screen_comparison: Mapping[str, Any],
    screen_launch_configuration: Mapping[str, Any],
    screen_identity_bindings: Mapping[str, Any],
    survivor_evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the sole authoritative lineage package for one W1.4 candidate."""

    stages = [
        {
            "comparison": qualification_comparison,
            "identity_bindings": qualification_identity_bindings,
            "launch_configuration": qualification_launch_configuration,
            "stage_id": "qualification",
        },
        {
            "comparison": screen_comparison,
            "identity_bindings": screen_identity_bindings,
            "launch_configuration": screen_launch_configuration,
            "stage_id": "screen",
        },
    ]
    body = _campaign_body(protocol, stages, survivor_evidence)
    artifact = {
        **body,
        FINGERPRINT_FIELD: enoch_week1.canonical_json_sha256(body),
    }
    validate_w1_4_campaign_lineage(protocol, artifact)
    return artifact


def validate_w1_4_campaign_lineage(
    protocol: Mapping[str, Any], artifact: Mapping[str, Any]
) -> str:
    """Reconstruct all W1.4 lineage claims from one self-contained artifact."""

    expected_keys = {
        "arm_registry_sha256",
        "automatic_production_promotion_allowed",
        "candidate_arm_ids",
        "candidate_fingerprint",
        "control_fingerprint",
        "environment_fingerprint",
        "evaluator_fingerprint",
        FINGERPRINT_FIELD,
        "fixture_report_fingerprint",
        "launch_configuration_sha256",
        "manifest_kind",
        "manifest_version",
        "protocol_fingerprint",
        "stages",
        "subject_id",
        "survivor_evidence",
        "survivor_summaries",
    }
    _require_exact_keys(artifact, expected_keys, "W1.4 campaign lineage")
    if artifact["manifest_kind"] != MANIFEST_KIND:
        _fail("unexpected W1.4 campaign lineage kind")
    if artifact["manifest_version"] != MANIFEST_VERSION:
        _fail("unsupported W1.4 campaign lineage version")
    if artifact["automatic_production_promotion_allowed"] is not False:
        _fail("W1.4 campaign lineage cannot authorize production promotion")
    if artifact["arm_registry_sha256"] != enoch_week1.ARM_REGISTRY_SHA256:
        _fail("W1.4 campaign lineage arm registry mismatch")
    for field in (
        "candidate_fingerprint",
        "control_fingerprint",
        "environment_fingerprint",
        "evaluator_fingerprint",
        "fixture_report_fingerprint",
        "launch_configuration_sha256",
        FINGERPRINT_FIELD,
    ):
        _require_sha256(artifact[field], field)

    expected_body = _campaign_body(
        protocol, artifact["stages"], artifact["survivor_evidence"]
    )
    actual_body = copy.deepcopy(dict(artifact))
    supplied_fingerprint = actual_body.pop(FINGERPRINT_FIELD)
    if actual_body != expected_body:
        _fail("W1.4 campaign lineage does not reconstruct from its evidence")
    expected_fingerprint = enoch_week1.canonical_json_sha256(expected_body)
    if supplied_fingerprint != expected_fingerprint:
        _fail("W1.4 campaign lineage fingerprint mismatch")
    return expected_fingerprint


def _candidate_decision_body(
    protocol: Mapping[str, Any],
    campaign_lineage: Mapping[str, Any],
    qualification_merged_result: Mapping[str, Any] | None,
    screen_merged_result: Mapping[str, Any] | None,
) -> dict[str, Any]:
    lineage_fingerprint = validate_w1_4_campaign_lineage(
        protocol, campaign_lineage
    )
    if qualification_merged_result is None or screen_merged_result is None:
        _fail("W1.4 candidate decision requires both exact merged results")
    if not isinstance(qualification_merged_result, Mapping) or not isinstance(
        screen_merged_result, Mapping
    ):
        _fail("W1.4 candidate merged results must be objects")

    stages = campaign_lineage["stages"]
    qualification_comparison = stages[0]["comparison"]
    screen_comparison = stages[1]["comparison"]
    if (
        stages[0]["stage_id"] != "qualification"
        or stages[1]["stage_id"] != "screen"
    ):
        _fail("W1.4 candidate decision lineage stages are not canonical")
    qualification_result_fingerprint = enoch_week1.validate_merged_result(
        protocol, qualification_comparison, qualification_merged_result
    )
    screen_result_fingerprint = enoch_week1.validate_merged_result(
        protocol, screen_comparison, screen_merged_result
    )

    development_rule = qualification_comparison["development_rule"]
    if development_rule is None:
        _fail("W1.4 comparisons require one non-null predeclared development rule")
    enoch_week1.validate_development_rule(development_rule)
    if screen_comparison["development_rule"] != development_rule:
        _fail("W1.4 comparisons do not share one predeclared development rule")
    development_rule_sha256 = enoch_week1.canonical_json_sha256(development_rule)

    failures: list[str] = []
    for stage_id, merged in (
        ("qualification", qualification_merged_result),
        ("screen", screen_merged_result),
    ):
        stage_failures = enoch_week1._development_rule_failures(
            development_rule, merged["metrics"]
        )
        failures.extend(f"{stage_id}:{reason}" for reason in stage_failures)
    screen_estimate = float(
        screen_merged_result["metrics"]["level_utility"]["estimate"]
    )
    if screen_estimate <= 0.0:
        failures.append("screen:combination-level-utility-not-positive")

    survivor_estimates = [
        {
            "arm_id": evidence["arm_id"],
            "level_utility_estimate": float(
                evidence["survivor_screen"]["merged_result"]["metrics"][
                    "level_utility"
                ]["estimate"]
            ),
        }
        for evidence in campaign_lineage["survivor_evidence"]
    ]
    individual_sum = sum(
        item["level_utility_estimate"] for item in survivor_estimates
    )
    qualification_estimate = float(
        qualification_merged_result["metrics"]["level_utility"]["estimate"]
    )
    interaction = {
        "combination_qualification_level_utility_estimate": qualification_estimate,
        "combination_screen_level_utility_estimate": screen_estimate,
        "individual_survivor_level_utility_estimates": survivor_estimates,
        "individual_survivor_level_utility_sum": individual_sum,
        "screen_minus_individual_sum": screen_estimate - individual_sum,
    }
    return {
        "automatic_production_promotion_allowed": False,
        "campaign_lineage": copy.deepcopy(dict(campaign_lineage)),
        "campaign_lineage_fingerprint": lineage_fingerprint,
        "candidate_fingerprint": campaign_lineage["candidate_fingerprint"],
        "control_fingerprint": campaign_lineage["control_fingerprint"],
        "decision": "eligible-for-qualification" if not failures else "reject-candidate",
        "development_rule_sha256": development_rule_sha256,
        "environment_fingerprint": campaign_lineage["environment_fingerprint"],
        "evaluator_fingerprint": campaign_lineage["evaluator_fingerprint"],
        "interaction_diagnostic": interaction,
        "manifest_kind": CANDIDATE_DECISION_KIND,
        "manifest_version": MANIFEST_VERSION,
        "protocol_fingerprint": campaign_lineage["protocol_fingerprint"],
        "qualification_merged_result": copy.deepcopy(
            dict(qualification_merged_result)
        ),
        "qualification_merged_result_fingerprint": (
            qualification_result_fingerprint
        ),
        "reasons": failures
        or ["both-combination-stages-pass-the-predeclared-rule"],
        "screen_merged_result": copy.deepcopy(dict(screen_merged_result)),
        "screen_merged_result_fingerprint": screen_result_fingerprint,
    }


def build_w1_4_candidate_decision(
    protocol: Mapping[str, Any],
    campaign_lineage: Mapping[str, Any],
    *,
    qualification_merged_result: Mapping[str, Any] | None,
    screen_merged_result: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Seal the two completed W1.4 stages and their interaction decision."""

    body = _candidate_decision_body(
        protocol,
        campaign_lineage,
        qualification_merged_result,
        screen_merged_result,
    )
    decision = {
        **body,
        CANDIDATE_DECISION_FINGERPRINT_FIELD: enoch_week1.canonical_json_sha256(
            body
        ),
    }
    validate_w1_4_candidate_decision(protocol, decision)
    return decision


def validate_w1_4_candidate_decision(
    protocol: Mapping[str, Any], decision: Mapping[str, Any]
) -> str:
    """Reconstruct a self-contained post-run W1.4 candidate decision."""

    expected_keys = {
        "automatic_production_promotion_allowed",
        "campaign_lineage",
        "campaign_lineage_fingerprint",
        "candidate_fingerprint",
        CANDIDATE_DECISION_FINGERPRINT_FIELD,
        "control_fingerprint",
        "decision",
        "development_rule_sha256",
        "environment_fingerprint",
        "evaluator_fingerprint",
        "interaction_diagnostic",
        "manifest_kind",
        "manifest_version",
        "protocol_fingerprint",
        "qualification_merged_result",
        "qualification_merged_result_fingerprint",
        "reasons",
        "screen_merged_result",
        "screen_merged_result_fingerprint",
    }
    _require_exact_keys(decision, expected_keys, "W1.4 candidate decision")
    if decision["manifest_kind"] != CANDIDATE_DECISION_KIND:
        _fail("unexpected W1.4 candidate decision kind")
    if decision["manifest_version"] != MANIFEST_VERSION:
        _fail("unsupported W1.4 candidate decision version")
    if decision["automatic_production_promotion_allowed"] is not False:
        _fail("W1.4 candidate decision cannot authorize production promotion")
    if decision["decision"] not in {
        "eligible-for-qualification",
        "reject-candidate",
    }:
        _fail("invalid W1.4 candidate decision")
    for field in (
        "campaign_lineage_fingerprint",
        "candidate_fingerprint",
        CANDIDATE_DECISION_FINGERPRINT_FIELD,
        "control_fingerprint",
        "development_rule_sha256",
        "environment_fingerprint",
        "evaluator_fingerprint",
        "protocol_fingerprint",
        "qualification_merged_result_fingerprint",
        "screen_merged_result_fingerprint",
    ):
        _require_sha256(decision[field], field)
    expected_body = _candidate_decision_body(
        protocol,
        decision["campaign_lineage"],
        decision["qualification_merged_result"],
        decision["screen_merged_result"],
    )
    actual_body = copy.deepcopy(dict(decision))
    supplied_fingerprint = actual_body.pop(CANDIDATE_DECISION_FINGERPRINT_FIELD)
    if actual_body != expected_body:
        _fail("W1.4 candidate decision does not reconstruct from its evidence")
    expected_fingerprint = enoch_week1.canonical_json_sha256(expected_body)
    if supplied_fingerprint != expected_fingerprint:
        _fail("W1.4 candidate decision fingerprint mismatch")
    return expected_fingerprint
