#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from training import enoch_week1, enoch_week1_runner
from training.enoch_week1_preflight import (
    DETERMINISTIC_SEARCH_TESTS,
    EQUIVALENCE_NAMESPACE,
    OUTCOME_BASELINE_NAMESPACE,
    PreflightError,
    STYLE_BASELINE_NAMESPACE,
    _protocol_prefix,
    build_preflight_artifact,
    build_w1_1_baseline_worker_report,
    run_probe,
    validate_preflight_artifact,
    validate_w1_1_baseline_worker_report,
)


def _style() -> dict[str, int | str]:
    return {
        "cards_played": 2,
        "decision_sha256": "0" * 64,
        "decisions": 2,
        "follow_decisions": 1,
        "joker_cards_played": 0,
        "lead_decisions": 1,
        "multi_card_plays": 0,
        "point_cards_played": 1,
        "point_value_played": 5,
        "single_card_plays": 2,
        "trump_cards_played": 1,
    }


def _orientation(*, landlord: bool, won: bool, margin: int) -> dict[str, object]:
    return {
        "complete": True,
        "focal_is_landlord_team": landlord,
        "focal_level_utility": 1 if won else -1,
        "focal_point_margin": margin,
        "focal_won": won,
        "landlord_level_delta": 1,
        "landlord_won": won if landlord else not won,
        "non_landlord_level_delta": 0,
        "non_landlord_points": 70,
        "style": _style(),
    }


def _probe_bytes(
    seeds: list[int],
    *,
    policy: str = "enoch-greedy",
    margin_offset: int = 0,
) -> bytes:
    pairs: list[dict[str, object]] = []
    for index, seed in enumerate(seeds):
        landlord = _orientation(landlord=True, won=True, margin=10 + margin_offset)
        attacker = _orientation(landlord=False, won=False, margin=-4)
        pairs.append(
            {
                "complete": True,
                "focal_as_attacker": attacker,
                "focal_as_landlord": landlord,
                "focal_level_utility_sum": 0,
                "focal_point_margin_sum": 6 + margin_offset,
                "focal_wins": 1,
                "request_index": index,
                "seed": seed,
            }
        )
    frozen = {
        "kind": "enoch-control-probe",
        "manifest_version": 1,
        "opponent": "legacy-greedy/easy-phases-v1",
        "pairs": pairs,
        "policy": policy,
        "seed_count": len(seeds),
        "seeds": seeds,
        "summary": {
            "complete_pairs": len(seeds),
            "completed_hands": 2 * len(seeds),
            "focal_decisions": 4 * len(seeds),
            "focal_wins": len(seeds),
            "incomplete_pairs": 0,
            "pairs_requested": len(seeds),
        },
    }
    document = {
        "equivalence_sha256": enoch_week1.canonical_json_sha256(frozen),
        "frozen_policy": frozen,
    }
    return enoch_week1.canonical_json_bytes(document) + b"\n"


def _completed(stdout: bytes, returncode: int = 0) -> subprocess.CompletedProcess[bytes]:
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=b"failure")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Week1PreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = enoch_week1.build_protocol(0xE10C_2026)

    def _seeds(self, namespace: str, count: int) -> list[int]:
        for entry in self.protocol["seed_registry"]["namespaces"]:
            if entry["name"] == namespace:
                return list(entry["seeds"][:count])
        self.fail(f"missing namespace {namespace}")

    def _initialize_ledger(self, path: Path) -> None:
        enoch_week1.atomic_write_json(
            path, enoch_week1.new_seed_ledger(self.protocol)
        )

    def _initialize_control_bundle(self, root: Path) -> Path:
        bundle = root / "control-bundle"
        (bundle / "bin").mkdir(parents=True)
        (bundle / "models").mkdir()
        (bundle / "source").mkdir()
        (bundle / "preflight").mkdir()
        (bundle / "protocol").mkdir()
        files = {
            "binary/enoch-control-probe-reference": bundle
            / "bin"
            / "enoch-control-probe-reference",
            "binary/enoch-control-probe-current": bundle
            / "bin"
            / "enoch-control-probe-current",
            "binary/week1-evaluator": bundle / "bin" / "enoch-week1-evaluator",
            "binary/enoch-0": bundle / "bin" / "enoch-0",
            "binary/expert-0": bundle / "bin" / "expert-0",
            "binary/grandmaster-0": bundle / "bin" / "grandmaster-0",
            "model/expert_model.onnx": bundle / "models" / "expert_model.onnx",
            "source/production-reference": bundle
            / "source"
            / "production-reference.tar",
            "protocol/week1-seed-protocol": bundle
            / "protocol"
            / "week1-seed-protocol.json",
            "preflight/expert-model-validation": bundle
            / "preflight"
            / "expert-model-validation.json",
            "preflight/reference-model-contract-tests": bundle
            / "preflight"
            / "reference-model-contract-tests.json",
            "preflight/strict-evaluator-test": bundle
            / "preflight"
            / "strict-evaluator-test.json",
        }
        for artifact_id, path in files.items():
            path.write_bytes(f"fixture:{artifact_id}".encode("utf-8"))
        files["protocol/week1-seed-protocol"].write_bytes(
            enoch_week1.canonical_json_bytes(self.protocol) + b"\n"
        )
        source_records = [
            {"path": "core/examples/enoch_eval.rs", "sha256": "f" * 64}
        ]
        source_list = bundle / "source" / "week1-evaluator-source-files.json"
        source_list.write_bytes(enoch_week1.canonical_json_bytes(source_records) + b"\n")
        files["source/week1-evaluator-file-list"] = source_list
        artifact_hashes = {
            artifact_id: _file_sha256(path) for artifact_id, path in files.items()
        }
        search_knobs = {
            "enoch-0": {
                "budget_ms": 2200,
                "max_candidates": 6,
                "max_worlds": 144,
                "policy": "EnochHeuristic",
                "rollout_policy": "EnochHeuristic",
                "rollout_tricks": 12,
            },
            "expert-0": {
                "budget_ms": 2200,
                "max_candidates": 6,
                "max_worlds": 144,
                "policy": "Net",
                "rollout_policy": "Heuristic",
                "rollout_tricks": 12,
            },
            "grandmaster-0": {
                "budget_multiplier": 3.0,
                "max_candidates": 8,
                "max_worlds": 400,
                "policy": "EnochHeuristic",
                "rollout_policy": "Heuristic",
                "rollout_tricks": "full-hand",
            },
        }
        production_source = artifact_hashes["source/production-reference"]
        policies = {
            name: enoch_week1.build_frozen_policy_identity(
                source_sha256=production_source,
                binary_sha256=artifact_hashes[f"binary/{name}"],
                model_sha256=(
                    artifact_hashes["model/expert_model.onnx"]
                    if name == "expert-0"
                    else enoch_week1.canonical_json_sha256(
                        {"model": "none", "reason": "heuristic-policy-tier"}
                    )
                ),
                configuration_sha256=enoch_week1.canonical_json_sha256(
                    search_knobs[name]
                ),
            )
            for name in ("enoch-0", "expert-0", "grandmaster-0")
        }
        evaluator = enoch_week1.build_frozen_evaluator_identity(
            source_sha256=artifact_hashes["source/week1-evaluator-file-list"],
            binary_sha256=artifact_hashes["binary/week1-evaluator"],
            configuration_sha256=enoch_week1.canonical_json_sha256(
                enoch_week1.WEEK1_EVALUATOR_CONTRACT
            ),
        )
        manifest = enoch_week1.build_w1_0_control_manifest(
            self.protocol,
            production_reference="c813c8a",
            policy_identities=policies,
            evaluator_identity=evaluator,
            artifact_hashes=artifact_hashes,
            compiler="rustc-test",
            operating_system="test-os",
            hardware_summary="test-machine",
            search_knobs=search_knobs,
            effective_environment={"LANG": "C"},
            replay_commands=["test"],
            model_selection_contract={
                "fallback_disabled": True,
                "intended_model_loaded": True,
                "policy_selection_independent": True,
                "q_selection_independent": True,
                "value_selection_independent": True,
            },
            model_selection_evidence=enoch_week1.MODEL_SELECTION_EVIDENCE_IDS,
        )
        enoch_week1.atomic_write_json(bundle / "control-manifest.json", manifest)
        enoch_week1.atomic_write_json(
            bundle / "bundle-index.json",
            {
                "artifact_hashes": manifest["artifact_hashes"],
                "control_manifest_fingerprint": manifest[
                    "control_manifest_fingerprint"
                ],
                "production_reference": manifest["production_reference"],
                "protocol_fingerprint": manifest["protocol_fingerprint"],
            },
        )
        return bundle

    def _runtime_smoke_evidence(
        self,
        bundle: Path,
        *,
        candidate_arm_ids: tuple[str, ...] = (),
        use_reference_policy_identity: bool = False,
    ) -> tuple[dict[str, dict[str, object]], dict[str, object], dict[str, object]]:
        manifest = enoch_week1.load_json_object(bundle / "control-manifest.json")
        knobs = manifest["search_knobs"]["enoch-0"]
        launch = enoch_week1_runner.build_launch_configuration(
            candidate_arm_ids=candidate_arm_ids,
            worlds=knobs["max_worlds"],
            candidates=knobs["max_candidates"],
            rollout_tricks=knobs["rollout_tricks"],
            budget_ms=knobs["budget_ms"],
        )
        evaluator_identity = manifest["evaluator_identity"]
        if use_reference_policy_identity:
            reference = manifest["policy_identities"]["enoch-0"]
            identities: dict[str, object] = {
                "candidate": copy.deepcopy(reference),
                "control": copy.deepcopy(reference),
                "evaluator": copy.deepcopy(evaluator_identity),
            }
        else:
            identities = enoch_week1_runner.build_in_process_identity_bindings(
                evaluator_identity, launch
            )
        candidate_fingerprint = enoch_week1.canonical_json_sha256(
            identities["candidate"]
        )
        control_fingerprint = enoch_week1.canonical_json_sha256(
            identities["control"]
        )
        evaluator_fingerprint = enoch_week1.canonical_json_sha256(
            identities["evaluator"]
        )
        configuration_fingerprint = enoch_week1.canonical_json_sha256(launch)
        environment_fingerprint = enoch_week1.canonical_json_sha256(
            {"effective_experiment_environment": {}}
        )
        external_fingerprint = enoch_week1.canonical_json_sha256(
            {"evidence": "preflight-smoke-fixture"}
        )
        smoke: dict[str, dict[str, object]] = {}
        for namespace, pair_count, shard_count in (
            ("smoke/product/001", 1, 1),
            ("smoke/product/010", 10, 2),
            ("smoke/product/100", 100, 4),
        ):
            comparison = enoch_week1.build_comparison_protocol_manifest(
                self.protocol,
                phase="W1.1",
                comparison_id=f"runtime-{pair_count:03d}",
                subject_id="runtime-enoch0-smoke",
                seed_namespace=namespace,
                pair_count=pair_count,
                shard_count=shard_count,
                candidate_fingerprint=candidate_fingerprint,
                control_fingerprint=control_fingerprint,
                evaluator_fingerprint=evaluator_fingerprint,
                environment_fingerprint=environment_fingerprint,
                configuration_fingerprint=configuration_fingerprint,
                required_style_metrics=enoch_week1.WEEK1_STYLE_METRICS,
            )
            seeds = self._seeds(namespace, pair_count)
            records = {
                index: {
                    "candidate_completed_worlds": knobs["max_worlds"],
                    "candidate_latency_ms": 1.0,
                    "complete": True,
                    "control_completed_worlds": knobs["max_worlds"],
                    "control_latency_ms": 1.0,
                    "effective_deal_seed": seed,
                    "failure_counters": {
                        name: 0 for name in enoch_week1.FAILURE_COUNTER_NAMES
                    },
                    "level_utility_delta": 0.0,
                    "orientations_completed": 2,
                    "point_margin_delta": 0.0,
                    "seed": seed,
                    "seed_index": index,
                    "style_metrics": {
                        name: 0.0 for name in enoch_week1.WEEK1_STYLE_METRICS
                    },
                    "win_rate_delta": 0.0,
                }
                for index, seed in enumerate(seeds)
            }
            shards = [
                enoch_week1.build_shard_result(
                    self.protocol,
                    comparison,
                    assignment["shard_id"],
                    [records[index] for index in assignment["seed_indices"]],
                    verified_external_evidence_fingerprint=external_fingerprint,
                )
                for assignment in comparison["shards"]
            ]
            smoke[namespace] = {
                "comparison": comparison,
                "identity_bindings": copy.deepcopy(identities),
                "launch_configuration": copy.deepcopy(launch),
                "merged_result": enoch_week1.merge_shard_results(
                    self.protocol, comparison, shards
                ),
            }
        return smoke, launch, identities

    def _minimal_report_dependencies(self) -> tuple[dict[str, object], dict[str, object]]:
        return (
            {
                "searchless_outcome_baseline": {"summary": {"pair_count": 5_000}},
                "style_baseline": {"summary": {"pair_count": 5_000}},
            },
            {},
        )

    def test_deterministic_search_authority_runs_the_end_to_end_repeat_fixture(
        self,
    ) -> None:
        self.assertIn(
            "bot::search::tests::strict_fixed_work_search_repeats_cards_and_work_telemetry",
            DETERMINISTIC_SEARCH_TESTS,
        )

    def test_baseline_worker_report_separates_reference_and_runtime_identities(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = self._initialize_control_bundle(Path(directory))
            smoke, launch, identities = self._runtime_smoke_evidence(bundle)
            preflight, authority = self._minimal_report_dependencies()
            with mock.patch(
                "training.enoch_week1_preflight.validate_preflight_artifact",
                return_value="a" * 64,
            ), mock.patch(
                "training.enoch_week1_preflight.validate_deterministic_search_fixture_authority",
                return_value="b" * 64,
            ):
                report = build_w1_1_baseline_worker_report(
                    self.protocol, {}, bundle, preflight, smoke, authority
                )
                fingerprint = validate_w1_1_baseline_worker_report(
                    self.protocol, {}, bundle, preflight, smoke, authority, report
                )
            manifest = enoch_week1.load_json_object(bundle / "control-manifest.json")
            reference_fingerprint = enoch_week1.canonical_json_sha256(
                manifest["policy_identities"]["enoch-0"]
            )

        runtime_fingerprint = enoch_week1.canonical_json_sha256(identities["control"])
        self.assertEqual(report["reference_enoch0_fingerprint"], reference_fingerprint)
        self.assertEqual(
            report["runtime_evaluation_control_fingerprint"], runtime_fingerprint
        )
        self.assertNotEqual(reference_fingerprint, runtime_fingerprint)
        self.assertEqual(report["runtime_launch_configuration"], launch)
        self.assertEqual(report["runtime_identity_bindings"], identities)
        self.assertEqual(fingerprint, report["baseline_worker_report_fingerprint"])

    def test_baseline_worker_report_rejects_reference_substitution_and_nonempty_arms(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = self._initialize_control_bundle(Path(directory))
            preflight, authority = self._minimal_report_dependencies()
            reference_smoke, _, _ = self._runtime_smoke_evidence(
                bundle, use_reference_policy_identity=True
            )
            armed_smoke, _, _ = self._runtime_smoke_evidence(
                bundle, candidate_arm_ids=("bid-ownership",)
            )
            with mock.patch(
                "training.enoch_week1_preflight.validate_preflight_artifact",
                return_value="a" * 64,
            ), mock.patch(
                "training.enoch_week1_preflight.validate_deterministic_search_fixture_authority",
                return_value="b" * 64,
            ):
                with self.assertRaisesRegex(
                    PreflightError, "identical empty-arm runtime identities"
                ):
                    build_w1_1_baseline_worker_report(
                        self.protocol,
                        {},
                        bundle,
                        preflight,
                        reference_smoke,
                        authority,
                    )
                with self.assertRaisesRegex(PreflightError, "empty Enoch arm set"):
                    build_w1_1_baseline_worker_report(
                        self.protocol,
                        {},
                        bundle,
                        preflight,
                        armed_smoke,
                        authority,
                    )

    def test_equivalence_and_searchless_baseline_use_exact_sanitized_seeds(self) -> None:
        equivalence_seeds = self._seeds(EQUIVALENCE_NAMESPACE, 2)
        outcome_seeds = self._seeds(OUTCOME_BASELINE_NAMESPACE, 1)
        style_seeds = self._seeds(STYLE_BASELINE_NAMESPACE, 1)
        equivalence = _probe_bytes(equivalence_seeds)
        outcome = _probe_bytes(outcome_seeds)
        style = _probe_bytes(style_seeds)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = self._initialize_control_bundle(root)
            ledger_path = root / "seed-ledger.json"
            self._initialize_ledger(ledger_path)
            with mock.patch(
                "training.enoch_week1_preflight.subprocess.run",
                side_effect=[
                    _completed(equivalence),
                    _completed(equivalence),
                    _completed(outcome),
                    _completed(style),
                ],
            ) as run:
                artifact = build_preflight_artifact(
                    self.protocol,
                    ledger_path,
                    bundle,
                    equivalence_policy="enoch-greedy",
                    equivalence_count=2,
                    outcome_count=1,
                    style_count=1,
                    allow_partial_prefix=True,
                    environment={
                        "PATH": "/usr/bin:/bin",
                        "SHENGJI_BOT_BUDGET_MS": "1",
                        "GM_WORLDS": "999",
                    },
                )
            ledger = enoch_week1.load_json_object(ledger_path)
            validated_fingerprint = validate_preflight_artifact(
                self.protocol, ledger, bundle, artifact
            )
            with self.assertRaisesRegex(PreflightError, "requires full preflight"):
                validate_preflight_artifact(
                    self.protocol,
                    ledger,
                    bundle,
                    artifact,
                    require_full_coverage=True,
                )

        self.assertEqual(run.call_count, 4)
        first_command = run.call_args_list[0].args[0]
        self.assertEqual(
            first_command[1:],
            [
                "--policy",
                "enoch-greedy",
                "--seed",
                str(equivalence_seeds[0]),
                "--seed",
                str(equivalence_seeds[1]),
            ],
        )
        for call in run.call_args_list:
            environment = call.kwargs["env"]
            self.assertFalse(
                any(name.startswith(("SHENGJI_", "GM_", "OMNI_", "GEN_")) for name in environment)
            )
        self.assertEqual(artifact["equivalence"]["seeds"], equivalence_seeds)
        self.assertEqual(
            artifact["searchless_outcome_baseline"]["seeds"], outcome_seeds
        )
        self.assertEqual(artifact["style_baseline"]["seeds"], style_seeds)
        self.assertEqual(
            artifact["searchless_outcome_baseline"]["policy"], "enoch-greedy"
        )
        self.assertEqual(
            artifact["searchless_outcome_baseline"]["summary"][
                "focal_point_margin_sum"
            ],
            6,
        )
        self.assertEqual(
            artifact["style_baseline"]["summary"]["style_totals"]["decisions"],
            4,
        )
        self.assertEqual(
            [record["namespace"] for record in ledger["consumed"]],
            [
                EQUIVALENCE_NAMESPACE,
                EQUIVALENCE_NAMESPACE,
                OUTCOME_BASELINE_NAMESPACE,
                STYLE_BASELINE_NAMESPACE,
            ],
        )
        self.assertEqual(artifact["equivalence"]["coverage"]["status"], "partial-prefix")
        self.assertEqual(artifact["coverage_status"], "partial-prefix")
        self.assertFalse(artifact["authoritative_for_w1_1_completion"])
        self.assertFalse(artifact["automatic_production_promotion_allowed"])
        self.assertEqual(
            artifact["final_ledger_fingerprint"], ledger["ledger_fingerprint"]
        )
        self.assertEqual(
            validated_fingerprint,
            artifact["artifact_sha256"],
        )
        body = dict(artifact)
        fingerprint = body.pop("artifact_sha256")
        self.assertEqual(fingerprint, enoch_week1.canonical_json_sha256(body))

    def test_semantic_divergence_fails_before_baseline(self) -> None:
        equivalence_seeds = self._seeds(EQUIVALENCE_NAMESPACE, 1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = self._initialize_control_bundle(root)
            ledger_path = root / "seed-ledger.json"
            self._initialize_ledger(ledger_path)
            with mock.patch(
                "training.enoch_week1_preflight.subprocess.run",
                side_effect=[
                    _completed(_probe_bytes(equivalence_seeds)),
                    _completed(_probe_bytes(equivalence_seeds, margin_offset=1)),
                ],
            ) as run:
                with self.assertRaisesRegex(PreflightError, "semantic divergence"):
                    build_preflight_artifact(
                        self.protocol,
                        ledger_path,
                        bundle,
                        equivalence_policy="enoch-greedy",
                        equivalence_count=1,
                        outcome_count=1,
                        style_count=1,
                        allow_partial_prefix=True,
                        environment={"PATH": "/usr/bin:/bin"},
                    )
            ledger = enoch_week1.load_json_object(ledger_path)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(len(ledger["consumed"]), 1)
        self.assertEqual(ledger["consumed"][0]["namespace"], EQUIVALENCE_NAMESPACE)

    def test_default_count_is_full_and_explicit_count_is_recorded_partial(self) -> None:
        indices, seeds, coverage = _protocol_prefix(
            self.protocol, EQUIVALENCE_NAMESPACE, None, "equivalence"
        )
        self.assertEqual(len(indices), 100)
        self.assertEqual(len(seeds), 100)
        self.assertEqual(coverage, {"namespace_capacity": 100, "status": "full"})
        _, _, partial = _protocol_prefix(
            self.protocol, EQUIVALENCE_NAMESPACE, 3, "equivalence"
        )
        self.assertEqual(partial, {"namespace_capacity": 100, "status": "partial-prefix"})

    def test_preflight_seed_prefix_is_single_use(self) -> None:
        equivalence_seeds = self._seeds(EQUIVALENCE_NAMESPACE, 1)
        outcome_seeds = self._seeds(OUTCOME_BASELINE_NAMESPACE, 1)
        style_seeds = self._seeds(STYLE_BASELINE_NAMESPACE, 1)
        outputs = [
            _completed(_probe_bytes(equivalence_seeds)),
            _completed(_probe_bytes(equivalence_seeds)),
            _completed(_probe_bytes(outcome_seeds)),
            _completed(_probe_bytes(style_seeds)),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = self._initialize_control_bundle(root)
            ledger_path = root / "ledger.json"
            self._initialize_ledger(ledger_path)
            with mock.patch(
                "training.enoch_week1_preflight.subprocess.run", side_effect=outputs
            ):
                build_preflight_artifact(
                    self.protocol,
                    ledger_path,
                    bundle,
                    equivalence_count=1,
                    outcome_count=1,
                    style_count=1,
                    allow_partial_prefix=True,
                    environment={"PATH": "/usr/bin:/bin"},
                )
            with mock.patch("training.enoch_week1_preflight.subprocess.run") as run:
                with self.assertRaisesRegex(
                    enoch_week1.SeedReuseError, "seed already consumed"
                ):
                    build_preflight_artifact(
                        self.protocol,
                        ledger_path,
                        bundle,
                        equivalence_count=1,
                        outcome_count=1,
                        style_count=1,
                        allow_partial_prefix=True,
                        environment={"PATH": "/usr/bin:/bin"},
                    )
            run.assert_not_called()

    def test_validator_rejects_rehashed_tampering_and_the_wrong_ledger(self) -> None:
        equivalence_seeds = self._seeds(EQUIVALENCE_NAMESPACE, 1)
        outcome_seeds = self._seeds(OUTCOME_BASELINE_NAMESPACE, 1)
        style_seeds = self._seeds(STYLE_BASELINE_NAMESPACE, 1)
        outputs = [
            _completed(_probe_bytes(equivalence_seeds)),
            _completed(_probe_bytes(equivalence_seeds)),
            _completed(_probe_bytes(outcome_seeds)),
            _completed(_probe_bytes(style_seeds)),
        ]
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        bundle = self._initialize_control_bundle(root)
        ledger_path = root / "ledger.json"
        self._initialize_ledger(ledger_path)
        with mock.patch(
            "training.enoch_week1_preflight.subprocess.run", side_effect=outputs
        ):
            artifact = build_preflight_artifact(
                self.protocol,
                ledger_path,
                bundle,
                equivalence_count=1,
                outcome_count=1,
                style_count=1,
                allow_partial_prefix=True,
                environment={"PATH": "/usr/bin:/bin"},
            )
        ledger = enoch_week1.load_json_object(ledger_path)

        tampered = copy.deepcopy(artifact)
        tampered["searchless_outcome_baseline"]["namespace"] = STYLE_BASELINE_NAMESPACE
        tampered_body = dict(tampered)
        tampered_body.pop("artifact_sha256")
        tampered["artifact_sha256"] = enoch_week1.canonical_json_sha256(tampered_body)
        with self.assertRaisesRegex(PreflightError, "wrong frozen namespace"):
            validate_preflight_artifact(self.protocol, ledger, bundle, tampered)

        empty_ledger = enoch_week1.new_seed_ledger(self.protocol)
        with self.assertRaisesRegex(PreflightError, "sequence range is invalid"):
            validate_preflight_artifact(
                self.protocol, empty_ledger, bundle, artifact
            )

    def test_authoritative_builder_requires_an_existing_valid_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = self._initialize_control_bundle(root)
            missing = root / "missing-ledger.json"
            with mock.patch("training.enoch_week1_preflight.subprocess.run") as run:
                with self.assertRaisesRegex(PreflightError, "ledger does not exist"):
                    build_preflight_artifact(
                        self.protocol,
                        missing,
                        bundle,
                        equivalence_count=1,
                        outcome_count=1,
                        style_count=1,
                        allow_partial_prefix=True,
                        environment={"PATH": "/usr/bin:/bin"},
                    )
            run.assert_not_called()

    def test_nonzero_exit_fails_closed(self) -> None:
        with mock.patch(
            "training.enoch_week1_preflight.subprocess.run",
            return_value=_completed(b"{}\n", returncode=7),
        ):
            with self.assertRaisesRegex(PreflightError, r"exited nonzero \(7\)"):
                run_probe(
                    Path("probe"),
                    "enoch-greedy",
                    [1],
                    environment={"PATH": "/usr/bin:/bin"},
                )

    def test_bad_json_fails_closed(self) -> None:
        with mock.patch(
            "training.enoch_week1_preflight.subprocess.run",
            return_value=_completed(b"not-json\n"),
        ):
            with self.assertRaisesRegex(PreflightError, "invalid JSON"):
                run_probe(
                    Path("probe"),
                    "enoch-greedy",
                    [1],
                    environment={"PATH": "/usr/bin:/bin"},
                )

    def test_seed_mismatch_and_duplicate_request_fail_closed(self) -> None:
        with mock.patch(
            "training.enoch_week1_preflight.subprocess.run",
            return_value=_completed(_probe_bytes([2])),
        ):
            with self.assertRaisesRegex(PreflightError, "seed mismatch"):
                run_probe(
                    Path("probe"),
                    "enoch-greedy",
                    [1],
                    environment={"PATH": "/usr/bin:/bin"},
                )
        with mock.patch("training.enoch_week1_preflight.subprocess.run") as run:
            with self.assertRaisesRegex(PreflightError, "duplicate seed"):
                run_probe(
                    Path("probe"),
                    "enoch-greedy",
                    [1, 1],
                    environment={"PATH": "/usr/bin:/bin"},
                )
        run.assert_not_called()

    def test_noncanonical_json_is_rejected_even_when_semantically_valid(self) -> None:
        canonical = _probe_bytes([4])
        document = json.loads(canonical)
        noncanonical = json.dumps(document, indent=2).encode("utf-8") + b"\n"
        with mock.patch(
            "training.enoch_week1_preflight.subprocess.run",
            return_value=_completed(noncanonical),
        ):
            with self.assertRaisesRegex(PreflightError, "not canonical JSON"):
                run_probe(
                    Path("probe"),
                    "enoch-greedy",
                    [4],
                    environment={"PATH": "/usr/bin:/bin"},
                )


if __name__ == "__main__":
    unittest.main()
