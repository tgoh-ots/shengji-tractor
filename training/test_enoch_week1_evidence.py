#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

from training import enoch_week1
from training.enoch_week1_evidence import (
    EvidenceError,
    MODEL_CONTRACT_ARTIFACTS,
    build_machine_contention_attestation,
    build_verified_external_evidence,
    validate_verified_external_evidence,
)
from training.enoch_week1_fixtures import ARM_FIXTURES, GLOBAL_FIXTURES


class Week1EvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.protocol = enoch_week1.build_protocol(0xE71D_EACE)

        self.source_records = [
            {"path": "Cargo.toml", "sha256": self._hash("cargo-source")},
            {
                "path": "core/src/bot/search.rs",
                "sha256": self._hash("search-source"),
            },
        ]
        self.source_path = self.root / "week1-evaluator-source-files.json"
        enoch_week1.atomic_write_json(self.source_path, self.source_records)
        source_file_hash = self._file_sha256(self.source_path)

        self.model_paths = {
            "preflight/expert-model-validation": self.root
            / "expert-model-validation.json",
            "preflight/reference-model-contract-tests": self.root
            / "reference-model-contract-tests.json",
            "preflight/strict-evaluator-test": self.root
            / "strict-evaluator-test.json",
        }
        enoch_week1.atomic_write_json(
            self.model_paths["preflight/expert-model-validation"],
            {"command_succeeded": True, "output": "expert model validated"},
        )
        reference_markers = (
            "test bot::expert::model_path_tests::model_path_override_round_trips ... ok",
            "test bot::expert::model_path_tests::manifest_rejects_width_drift_and_untyped_v2_outputs ... ok",
            "test bot::expert::model_path_tests::embedded_model_has_no_value_output ... ok",
        )
        enoch_week1.atomic_write_json(
            self.model_paths["preflight/reference-model-contract-tests"],
            {"command_succeeded": True, "output": "\n".join(reference_markers)},
        )
        enoch_week1.atomic_write_json(
            self.model_paths["preflight/strict-evaluator-test"],
            {
                "command_succeeded": True,
                "output": (
                    "test bot::search::tests::"
                    "strict_search_rejects_a_zero_sample_prior_fallback ... ok"
                ),
            },
        )

        evaluator_binary = self._hash("week1-evaluator-binary")
        evaluator_identity = enoch_week1.build_frozen_evaluator_identity(
            source_sha256=source_file_hash,
            binary_sha256=evaluator_binary,
            configuration_sha256=enoch_week1.canonical_json_sha256(
                enoch_week1.WEEK1_EVALUATOR_CONTRACT
            ),
        )
        no_model = enoch_week1.canonical_json_sha256(
            {"model": "none", "reason": "heuristic-policy-tier"}
        )
        self.runner_identities = {
            "candidate": enoch_week1.build_frozen_policy_identity(
                source_sha256=source_file_hash,
                binary_sha256=evaluator_binary,
                model_sha256=no_model,
                configuration_sha256=self._hash("runner-candidate-config"),
            ),
            "control": enoch_week1.build_frozen_policy_identity(
                source_sha256=source_file_hash,
                binary_sha256=evaluator_binary,
                model_sha256=no_model,
                configuration_sha256=self._hash("runner-control-config"),
            ),
            "evaluator": evaluator_identity,
        }
        self.identities_path = self.root / "runner-identities.json"
        enoch_week1.atomic_write_json(self.identities_path, self.runner_identities)
        self.environment_fingerprint = self._hash("environment")
        self.comparison = enoch_week1.build_comparison_protocol_manifest(
            self.protocol,
            phase="W1.1",
            comparison_id="typed-evidence-smoke",
            subject_id="enoch-1",
            seed_namespace="smoke/product/001",
            pair_count=1,
            shard_count=1,
            candidate_fingerprint=enoch_week1.canonical_json_sha256(
                self.runner_identities["candidate"]
            ),
            control_fingerprint=enoch_week1.canonical_json_sha256(
                self.runner_identities["control"]
            ),
            evaluator_fingerprint=enoch_week1.canonical_json_sha256(
                evaluator_identity
            ),
            environment_fingerprint=self.environment_fingerprint,
            configuration_fingerprint=self._hash("launch-configuration"),
        )

        search_knobs = {
            "enoch-0": {"policy": "enoch-0"},
            "expert-0": {"policy": "expert-0"},
            "grandmaster-0": {"policy": "grandmaster-0"},
        }
        production_source = self._hash("production-source")
        expert_model = self._hash("expert-model")
        policy_identities = {
            "enoch-0": enoch_week1.build_frozen_policy_identity(
                source_sha256=production_source,
                binary_sha256=self._hash("enoch-0-binary"),
                model_sha256=no_model,
                configuration_sha256=enoch_week1.canonical_json_sha256(
                    search_knobs["enoch-0"]
                ),
            ),
            "expert-0": enoch_week1.build_frozen_policy_identity(
                source_sha256=production_source,
                binary_sha256=self._hash("expert-0-binary"),
                model_sha256=expert_model,
                configuration_sha256=enoch_week1.canonical_json_sha256(
                    search_knobs["expert-0"]
                ),
            ),
            "grandmaster-0": enoch_week1.build_frozen_policy_identity(
                source_sha256=production_source,
                binary_sha256=self._hash("grandmaster-0-binary"),
                model_sha256=no_model,
                configuration_sha256=enoch_week1.canonical_json_sha256(
                    search_knobs["grandmaster-0"]
                ),
            ),
        }
        artifact_hashes = {
            "binary/enoch-0": policy_identities["enoch-0"]["binary_sha256"],
            "binary/expert-0": policy_identities["expert-0"]["binary_sha256"],
            "binary/grandmaster-0": policy_identities["grandmaster-0"][
                "binary_sha256"
            ],
            "binary/week1-evaluator": evaluator_binary,
            "model/expert_model.onnx": expert_model,
            "protocol/week1-seed-protocol": self._hash("protocol-file"),
            "source/production-reference": production_source,
            "source/week1-evaluator-file-list": source_file_hash,
        }
        artifact_hashes.update(
            {
                artifact_id: self._file_sha256(path)
                for artifact_id, path in self.model_paths.items()
            }
        )
        control_manifest = enoch_week1.build_w1_0_control_manifest(
            self.protocol,
            production_reference="c813c8a",
            policy_identities=policy_identities,
            evaluator_identity=evaluator_identity,
            artifact_hashes=artifact_hashes,
            compiler="rustc-test",
            operating_system="test-os",
            hardware_summary="test-machine",
            search_knobs=search_knobs,
            effective_environment={"LANG": "C"},
            replay_commands=["replay"],
            model_selection_contract={
                "fallback_disabled": True,
                "intended_model_loaded": True,
                "policy_selection_independent": True,
                "q_selection_independent": True,
                "value_selection_independent": True,
            },
            model_selection_evidence=enoch_week1.MODEL_SELECTION_EVIDENCE_IDS,
        )
        self.control_path = self.root / "control-manifest.json"
        enoch_week1.atomic_write_json(self.control_path, control_manifest)

        self.fixture_path = self._write_fixture_bundle()
        attestation = build_machine_contention_attestation(
            self.comparison,
            operator_id="test-operator",
            observation_started_utc="2026-07-02T10:00:00Z",
            observation_ended_utc="2026-07-02T10:10:00Z",
            attested_at_utc="2026-07-02T10:11:00Z",
            worker_count=2,
            available_parallelism=4,
        )
        self.machine_path = self.root / "machine-attestation.json"
        enoch_week1.atomic_write_json(self.machine_path, attestation)
        self.evidence = self._build_evidence()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _hash(label: str) -> str:
        return enoch_week1.canonical_json_sha256({"test": label})

    @staticmethod
    def _file_sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _write_fixture_bundle(self) -> Path:
        fixture_root = self.root / "fixture-bundle"
        logs = fixture_root / "logs"
        logs.mkdir(parents=True)
        cases = [
            ("global", fixture_id, package, test_name)
            for fixture_id, package, test_name in GLOBAL_FIXTURES
        ]
        cases.extend(
            (arm, test_name.rsplit("::", 1)[-1], "shengji-core", test_name)
            for arm, tests in ARM_FIXTURES.items()
            for test_name in tests
        )
        records = []
        for index, (scope, fixture_id, package, test_name) in enumerate(cases):
            log_name = f"{index:03d}-{scope}-{fixture_id}.log"
            content = f"fixture {index} passed\n".encode()
            (logs / log_name).write_bytes(content)
            records.append(
                {
                    "command": [
                        "cargo",
                        "test",
                        "--locked",
                        "-p",
                        package,
                        test_name,
                        "--",
                        "--exact",
                        "--test-threads=1",
                    ],
                    "exit_code": 0,
                    "failed": 0,
                    "fixture_id": fixture_id,
                    "log_path": f"logs/{log_name}",
                    "output_sha256": hashlib.sha256(content).hexdigest(),
                    "package": package,
                    "passed": 1,
                    "scope": scope,
                    "sequence": index,
                    "test_name": test_name,
                }
            )
        body = {
            "arm_registry_sha256": enoch_week1.ARM_REGISTRY_SHA256,
            "automatic_production_promotion_allowed": False,
            "failure_count": 0,
            "manifest_kind": "enoch-week1-fixture-report",
            "manifest_version": 1,
            "records": records,
            "records_sha256": enoch_week1.canonical_json_sha256(records),
            "source_files": self.source_records,
            "source_files_sha256": enoch_week1.canonical_json_sha256(
                self.source_records
            ),
        }
        report = {
            **body,
            "fixture_report_fingerprint": enoch_week1.canonical_json_sha256(body),
        }
        path = fixture_root / "fixture-report.json"
        enoch_week1.atomic_write_json(path, report)
        return path

    def _build_evidence(self, **overrides):
        arguments = {
            "fixture_report_path": self.fixture_path,
            "source_identity_path": self.source_path,
            "control_manifest_path": self.control_path,
            "runner_identities_path": self.identities_path,
            "model_contract_artifact_paths": self.model_paths,
            "machine_attestation_path": self.machine_path,
        }
        arguments.update(overrides)
        return build_verified_external_evidence(
            self.protocol, self.comparison, **arguments
        )

    def test_actual_artifacts_rederive_every_external_zero(self) -> None:
        counters = validate_verified_external_evidence(
            self.protocol, self.comparison, self.evidence
        )
        self.assertEqual(set(counters), {
            "artifact_mismatch",
            "fixture_failure",
            "hidden_information_leak",
            "honesty_violation",
            "machine_contention",
            "model_contract_failure",
        })
        for record in counters.values():
            self.assertEqual(record["count"], 0)
            self.assertIn("artifact_ids", record)
            self.assertNotIn("authority_artifact_sha256", record)
        self.assertEqual(set(self.model_paths), set(MODEL_CONTRACT_ARTIFACTS))

    def test_fixture_logs_source_and_runner_identities_fail_closed(self) -> None:
        first_log = next((self.fixture_path.parent / "logs").iterdir())
        original_log = first_log.read_bytes()
        first_log.write_bytes(b"tampered")
        try:
            with self.assertRaisesRegex(EvidenceError, "fixture log hash mismatch"):
                validate_verified_external_evidence(
                    self.protocol, self.comparison, self.evidence
                )
        finally:
            first_log.write_bytes(original_log)

        wrong_identities = copy.deepcopy(self.runner_identities)
        wrong_identities["candidate"]["source_sha256"] = self._hash("wrong-source")
        wrong_path = self.root / "wrong-identities.json"
        enoch_week1.atomic_write_json(wrong_path, wrong_identities)
        with self.assertRaisesRegex(EvidenceError, "differs from the comparison"):
            self._build_evidence(runner_identities_path=wrong_path)

        wrong_source_records = copy.deepcopy(self.source_records)
        wrong_source_records[-1]["sha256"] = self._hash("wrong-fixture-source")
        wrong_source_path = self.root / "wrong-source.json"
        enoch_week1.atomic_write_json(wrong_source_path, wrong_source_records)
        with self.assertRaisesRegex(EvidenceError, "fixture report source differs"):
            self._build_evidence(source_identity_path=wrong_source_path)

    def test_model_markers_machine_binding_and_counter_schema_fail_closed(self) -> None:
        bad_model_path = self.root / "bad-strict-evaluator.json"
        enoch_week1.atomic_write_json(
            bad_model_path, {"command_succeeded": True, "output": "no test marker"}
        )
        bad_model_paths = dict(self.model_paths)
        bad_model_paths["preflight/strict-evaluator-test"] = bad_model_path
        with self.assertRaisesRegex(EvidenceError, "lacks required test marker"):
            self._build_evidence(model_contract_artifact_paths=bad_model_paths)

        wrong_comparison = dict(self.comparison)
        wrong_comparison["environment_fingerprint"] = self._hash("wrong-env")
        wrong_machine = build_machine_contention_attestation(
            wrong_comparison,
            operator_id="test-operator",
            observation_started_utc="2026-07-02T10:00:00Z",
            observation_ended_utc="2026-07-02T10:10:00Z",
            attested_at_utc="2026-07-02T10:11:00Z",
            worker_count=2,
            available_parallelism=4,
        )
        wrong_machine_path = self.root / "wrong-machine.json"
        enoch_week1.atomic_write_json(wrong_machine_path, wrong_machine)
        with self.assertRaisesRegex(EvidenceError, "another environment"):
            self._build_evidence(machine_attestation_path=wrong_machine_path)

        invented_authority = copy.deepcopy(self.evidence)
        invented_authority["counters"]["fixture_failure"][
            "artifact_ids"
        ] = ["invented/digest"]
        body = dict(invented_authority)
        body.pop("verified_external_evidence_fingerprint")
        invented_authority[
            "verified_external_evidence_fingerprint"
        ] = enoch_week1.canonical_json_sha256(body)
        with self.assertRaisesRegex(EvidenceError, "counters do not reconstruct"):
            validate_verified_external_evidence(
                self.protocol, self.comparison, invented_authority
            )


if __name__ == "__main__":
    unittest.main()
