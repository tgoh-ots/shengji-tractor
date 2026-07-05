#!/usr/bin/env python3

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from training import enoch_week1 as enoch_week1_module
from training import enoch_week1_campaign, enoch_week1_runner
from training.enoch_week1_fixtures import ARM_FIXTURES, GLOBAL_FIXTURES
from training.enoch_week1 import (
    ARM_REGISTRY_SHA256,
    BLOCKED_EVALUATOR_ENV_PREFIXES,
    MODEL_SELECTION_EVIDENCE_IDS,
    LOCKED_SUPERIORITY_RULE_SHA256,
    QUALIFICATION_MATRIX,
    QUALIFICATION_THRESHOLDS,
    ProtocolError,
    SEED_NAMESPACE_COUNTS,
    WEEK1_EVALUATOR_CONTRACT,
    WEEK1_STYLE_METRICS,
    SeedReuseError,
    atomic_write_json,
    build_comparison_protocol_manifest,
    build_development_rule,
    build_frozen_evaluator_identity,
    build_frozen_policy_identity,
    build_locked_gate_decision,
    build_locked_gate_manifests,
    build_phase_manifest,
    build_protocol,
    build_shard_result,
    build_w1_0_control_manifest,
    build_w1_3_advancement_decision,
    build_w1_5_qualification_decision,
    build_w1_5_qualification_manifest,
    build_week1_decision_artifact,
    canonical_arm_registry,
    canonical_json_bytes,
    canonical_json_sha256,
    consume_seed_batch_once,
    consume_seed_once,
    derive_seed,
    load_json_object,
    main,
    merge_shard_results,
    new_seed_ledger,
    sanitized_evaluator_environment,
    validate_comparison_protocol_manifest,
    validate_locked_gate_decision,
    validate_locked_gate_pair,
    validate_merged_result,
    validate_phase_manifest,
    validate_phase_chain,
    validate_protocol,
    validate_shard_result,
    validate_seed_ledger,
    validate_w1_0_control_manifest,
    validate_w1_3_advancement_decision,
    validate_w1_5_qualification_decision,
    validate_w1_5_qualification_manifest,
    validate_week1_decision_artifact,
)


class Week1ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = build_protocol(0x5EED_2026_0702)

    def test_namespace_contract_has_exact_plan_counts(self) -> None:
        counts = dict(SEED_NAMESPACE_COUNTS)
        self.assertEqual(len(counts), len(SEED_NAMESPACE_COUNTS))
        self.assertEqual(counts["preflight/frozen-policy-equivalence"], 100)
        self.assertEqual(counts["baseline/searchless-outcome"], 5_000)
        self.assertEqual(counts["smoke/product/001"], 1)
        self.assertEqual(counts["dev/ablation/bid-ownership"], 300)
        self.assertEqual(counts["dev/survivor/uncertain-legal-throws"], 800)
        self.assertEqual(counts["qual/intended"], 800)
        self.assertEqual(counts["qual/equal"], 800)
        self.assertEqual(counts["qual/threshold/situation-04"], 50)
        self.assertEqual(counts["locked/gate-1"], 2_000)
        self.assertEqual(counts["locked/confirmation"], 2_000)
        self.assertEqual(sum(counts.values()), 35_111)

    def test_seed_derivation_is_stable_and_domain_separated(self) -> None:
        first = derive_seed(123, "qual/intended", 0)
        self.assertEqual(first, derive_seed(123, "qual/intended", 0))
        self.assertNotEqual(first, derive_seed(123, "qual/intended", 1))
        self.assertNotEqual(first, derive_seed(123, "qual/equal", 0))
        self.assertNotEqual(first, derive_seed(124, "qual/intended", 0))
        # A fixed vector catches accidental changes to framing, encoding, or truncation.
        self.assertEqual(first, 9_567_484_213_682_665_997)

    def test_protocol_is_valid_and_all_seeds_are_globally_disjoint(self) -> None:
        fingerprint = validate_protocol(self.protocol)
        self.assertEqual(fingerprint, self.protocol["protocol_fingerprint"])
        seeds = [
            seed
            for namespace in self.protocol["seed_registry"]["namespaces"]
            for seed in namespace["seeds"]
        ]
        self.assertEqual(len(seeds), 35_111)
        self.assertEqual(len(seeds), len(set(seeds)))

    def test_protocol_fingerprint_rejects_tampering(self) -> None:
        tampered = copy.deepcopy(self.protocol)
        tampered["seed_registry"]["namespaces"][0]["seeds"][0] ^= 1
        with self.assertRaisesRegex(ProtocolError, "derived seeds changed"):
            validate_protocol(tampered)

        # Rehashing a changed protocol still cannot alter the immutable namespace contract.
        tampered = copy.deepcopy(self.protocol)
        tampered["seed_registry"]["namespaces"][0]["count"] = 99
        tampered["seed_registry"]["namespaces"][0]["seeds"].pop()
        tampered["seed_registry_sha256"] = canonical_json_sha256(tampered["seed_registry"])
        body = dict(tampered)
        body.pop("protocol_fingerprint")
        tampered["protocol_fingerprint"] = canonical_json_sha256(body)
        with self.assertRaisesRegex(ProtocolError, "namespace contract changed"):
            validate_protocol(tampered)

    def test_canonical_json_hashing_is_order_independent_and_strict(self) -> None:
        left = {"z": [3, 2, 1], "a": {"β": True}}
        right = {"a": {"β": True}, "z": [3, 2, 1]}
        self.assertEqual(canonical_json_bytes(left), canonical_json_bytes(right))
        self.assertEqual(canonical_json_sha256(left), canonical_json_sha256(right))
        self.assertNotIn(b" ", canonical_json_bytes(left))
        with self.assertRaises(ProtocolError):
            canonical_json_bytes({"bad": float("nan")})

    def test_atomic_write_is_canonical_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "manifest.json"
            atomic_write_json(path, {"b": 2, "a": 1})
            self.assertEqual(path.read_bytes(), b'{"a":1,"b":2}\n')
            with self.assertRaises(FileExistsError):
                atomic_write_json(path, {"replacement": True})
            atomic_write_json(path, {"replacement": True}, overwrite=True)
            self.assertEqual(load_json_object(path), {"replacement": True})

    def test_environment_sanitizer_removes_all_blocked_families(self) -> None:
        environment = {
            "PATH": "/bin",
            "SHENGJI_KEEP": "yes",
            "SHENGJI_DROP": "bad",
            "GM_MODE": "bad",
            "OMNI_MODEL": "bad",
            "GEN_LIMIT": "bad",
        }
        cleaned, removed = sanitized_evaluator_environment(
            environment,
            allowlist=["SHENGJI_KEEP"],
            overrides={"SHENGJI_KEEP": "explicit", "LANG": "C"},
        )
        self.assertEqual(cleaned["PATH"], "/bin")
        self.assertEqual(cleaned["SHENGJI_KEEP"], "explicit")
        self.assertEqual(cleaned["LANG"], "C")
        self.assertEqual(
            removed, ("GEN_LIMIT", "GM_MODE", "OMNI_MODEL", "SHENGJI_DROP")
        )
        self.assertTrue(
            all(name.startswith(BLOCKED_EVALUATOR_ENV_PREFIXES) for name in removed)
        )
        with self.assertRaisesRegex(ProtocolError, "not explicitly allowlisted"):
            sanitized_evaluator_environment({}, overrides={"SHENGJI_SURPRISE": "1"})

    def test_seed_ledger_rejects_reuse_and_tampering(self) -> None:
        ledger = new_seed_ledger(self.protocol)
        validate_seed_ledger(self.protocol, ledger)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "consumed.json"
            consume_seed_once(path, self.protocol, "smoke/product/010", 0, "smoke-run")
            with self.assertRaises(SeedReuseError):
                consume_seed_once(path, self.protocol, "smoke/product/010", 0, "other-run")

            updated = consume_seed_batch_once(
                path,
                self.protocol,
                [
                    ("smoke/product/010", 1, "smoke-run"),
                    ("smoke/product/010", 2, "smoke-run"),
                ],
            )
            self.assertEqual(len(updated["consumed"]), 3)
            validate_seed_ledger(self.protocol, updated)

            tampered = copy.deepcopy(updated)
            tampered["consumed"][0]["consumer"] = "changed"
            with self.assertRaisesRegex(ProtocolError, "fingerprint mismatch"):
                validate_seed_ledger(self.protocol, tampered)

    def test_batch_rejects_in_batch_reuse_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "consumed.json"
            with self.assertRaises(SeedReuseError):
                consume_seed_batch_once(
                    path,
                    self.protocol,
                    [
                        ("smoke/product/100", 0, "run-a"),
                        ("smoke/product/100", 0, "run-b"),
                    ],
                )
            self.assertFalse(path.exists())

    def test_cli_initializes_and_validates_protocol_and_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            protocol_path = Path(temporary) / "protocol.json"
            ledger_path = Path(temporary) / "ledger.json"
            self.assertEqual(
                main(
                    [
                        "init-seeds",
                        "--master-seed",
                        "0x1234",
                        "--output",
                        str(protocol_path),
                        "--ledger",
                        str(ledger_path),
                        "--allow-env",
                        "SHENGJI_BOT_BUDGET_MS",
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "validate",
                        "--protocol",
                        str(protocol_path),
                        "--ledger",
                        str(ledger_path),
                    ]
                ),
                0,
            )
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            self.assertEqual(
                protocol["evaluator_environment_policy"]["allowlist"],
                ["SHENGJI_BOT_BUDGET_MS"],
            )


class Week1OrchestrationTests(unittest.TestCase):
    @staticmethod
    def _hash(label: str) -> str:
        return canonical_json_sha256({"test-identity": label})

    @classmethod
    def _w1_0_arguments(cls) -> dict:
        expert_knobs = {"policy": "Net", "worlds": 144}
        grandmaster_knobs = {"policy": "EnochHeuristic", "worlds": 400}
        expert_model = cls._hash("production-expert-model")
        expert = build_frozen_policy_identity(
            source_sha256=cls.production_source_hash,
            binary_sha256=cls._hash("expert-0-binary"),
            model_sha256=expert_model,
            configuration_sha256=canonical_json_sha256(expert_knobs),
        )
        grandmaster = build_frozen_policy_identity(
            source_sha256=cls.production_source_hash,
            binary_sha256=cls._hash("grandmaster-0-binary"),
            model_sha256=cls.no_model_hash,
            configuration_sha256=canonical_json_sha256(grandmaster_knobs),
        )
        evaluator = cls.evaluator_identity
        evaluator_source = evaluator["source_sha256"]
        artifacts = {
            "binary/enoch-0": cls.reference_control_identity["binary_sha256"],
            "binary/expert-0": expert["binary_sha256"],
            "binary/grandmaster-0": grandmaster["binary_sha256"],
            "binary/week1-evaluator": evaluator["binary_sha256"],
            "model/expert_model.onnx": expert_model,
            "preflight/expert-model-validation": cls._hash("model-validation"),
            "preflight/reference-model-contract-tests": cls._hash("model-tests"),
            "preflight/strict-evaluator-test": cls._hash("strict-test"),
            "protocol/week1-seed-protocol": cls._hash("protocol-file"),
            "source/production-reference": cls.production_source_hash,
            "source/week1-evaluator-file-list": evaluator_source,
        }
        return {
            "production_reference": "c813c8a",
            "policy_identities": {
                "enoch-0": cls.reference_control_identity,
                "expert-0": expert,
                "grandmaster-0": grandmaster,
            },
            "evaluator_identity": evaluator,
            "artifact_hashes": artifacts,
            "compiler": "rustc test",
            "operating_system": "test-os",
            "hardware_summary": "test-machine",
            "search_knobs": {
                "enoch-0": cls.control_search_knobs,
                "expert-0": expert_knobs,
                "grandmaster-0": grandmaster_knobs,
            },
            "effective_environment": {"LANG": "C"},
            "replay_commands": ["replay"],
            "model_selection_contract": {
                "fallback_disabled": True,
                "intended_model_loaded": True,
                "policy_selection_independent": True,
                "q_selection_independent": True,
                "value_selection_independent": True,
            },
            "model_selection_evidence": MODEL_SELECTION_EVIDENCE_IDS,
        }

    @staticmethod
    def _rehash(value: dict, fingerprint_field: str) -> None:
        body = dict(value)
        body.pop(fingerprint_field)
        value[fingerprint_field] = canonical_json_sha256(body)

    @classmethod
    def _fixture_report(cls, *, source_digest: str | None = None) -> dict:
        cases = [
            ("global", fixture_id, package, test_name)
            for fixture_id, package, test_name in GLOBAL_FIXTURES
        ]
        cases.extend(
            (arm, test_name.rsplit("::", 1)[-1], "shengji-core", test_name)
            for arm, tests in ARM_FIXTURES.items()
            for test_name in tests
        )
        records = [
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
                "log_path": f"logs/{index:03d}-{scope}-{fixture_id}.log",
                "output_sha256": cls._hash(f"fixture-output:{index}"),
                "package": package,
                "passed": 1,
                "scope": scope,
                "sequence": index,
                "test_name": test_name,
            }
            for index, (scope, fixture_id, package, test_name) in enumerate(cases)
        ]
        source_files = [
            {
                "path": "core/src/bot/search.rs",
                "sha256": source_digest or cls._hash("fixture-source"),
            }
        ]
        body = {
            "arm_registry_sha256": ARM_REGISTRY_SHA256,
            "automatic_production_promotion_allowed": False,
            "failure_count": 0,
            "manifest_kind": "enoch-week1-fixture-report",
            "manifest_version": 1,
            "records": records,
            "records_sha256": canonical_json_sha256(records),
            "source_files": source_files,
            "source_files_sha256": canonical_json_sha256(source_files),
        }
        return {
            **body,
            "fixture_report_fingerprint": canonical_json_sha256(body),
        }

    @classmethod
    def _record(
        cls,
        comparison: dict,
        index: int,
        *,
        level: float = 0.2,
        failures: dict[str, int] | None = None,
    ) -> dict:
        namespace = comparison["seed_namespace"]
        seed_entry = next(
            entry
            for entry in cls.protocol["seed_registry"]["namespaces"]
            if entry["name"] == namespace
        )
        counters = {
            "illegal_action": 0,
            "honesty_violation": 0,
            "model_fallback": 0,
            "model_contract_failure": 0,
            "incomplete_pair": 0,
            "hidden_information_leak": 0,
            "artifact_mismatch": 0,
            "cancellation": 0,
            "fixture_failure": 0,
            "machine_contention": 0,
            "timeout": 0,
        }
        counters.update(failures or {})
        return {
            "candidate_completed_worlds": 4,
            "candidate_latency_ms": 10.0,
            "complete": True,
            "control_completed_worlds": 4,
            "control_latency_ms": 10.0,
            "effective_deal_seed": seed_entry["seeds"][index],
            "failure_counters": counters,
            "level_utility_delta": level,
            "orientations_completed": 2,
            "point_margin_delta": 1.0,
            "seed": seed_entry["seeds"][index],
            "seed_index": index,
            "style_metrics": {
                name: 0.25 for name in comparison["required_style_metrics"]
            },
            "win_rate_delta": 0.1,
        }

    @classmethod
    def _build_merged(cls, comparison: dict, *, level: float = 0.2):
        external_evidence_fingerprint = cls._hash(
            "verified-external-evidence:"
            + comparison["comparison_protocol_fingerprint"]
        )
        shards = []
        for assignment in comparison["shards"]:
            records = [
                cls._record(comparison, index, level=level)
                for index in assignment["seed_indices"]
            ]
            shards.append(
                build_shard_result(
                    cls.protocol,
                    comparison,
                    assignment["shard_id"],
                    records,
                    verified_external_evidence_fingerprint=(
                        external_evidence_fingerprint
                    ),
                )
            )
        return shards, merge_shard_results(cls.protocol, comparison, shards)

    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = build_protocol(0xA11CE_2026_0702)
        cls.production_source_hash = cls._hash("production-source")
        cls.evaluator_source_hash = cls._hash("evaluator-file-list")
        cls.evaluator_identity = build_frozen_evaluator_identity(
            source_sha256=cls.evaluator_source_hash,
            binary_sha256=cls._hash("week1-evaluator-binary"),
            configuration_sha256=canonical_json_sha256(WEEK1_EVALUATOR_CONTRACT),
        )
        cls.combination_launch = enoch_week1_runner.build_launch_configuration(
            candidate_arm_ids=["bid-ownership"],
            worlds=4,
            candidates=4,
            rollout_tricks=2,
            deadline_ms=30_000,
        )
        campaign_identities = enoch_week1_runner.build_in_process_identity_bindings(
            cls.evaluator_identity, cls.combination_launch
        )
        cls.candidate_identity = campaign_identities["candidate"]
        cls.control_identity = campaign_identities["control"]
        cls.control_search_knobs = enoch_week1_runner.in_process_policy_configuration(
            [], control=True
        )
        cls.no_model_hash = enoch_week1_runner.IN_PROCESS_ENOCH_MODEL_SHA256
        cls.reference_control_identity = build_frozen_policy_identity(
            source_sha256=cls.production_source_hash,
            binary_sha256=cls._hash("production-enoch-0-binary"),
            model_sha256=cls.no_model_hash,
            configuration_sha256=canonical_json_sha256(cls.control_search_knobs),
        )
        cls.reference_control_fingerprint = canonical_json_sha256(
            cls.reference_control_identity
        )
        cls.candidate_fingerprint = canonical_json_sha256(cls.candidate_identity)
        cls.control_fingerprint = canonical_json_sha256(cls.control_identity)
        cls.evaluator_fingerprint = canonical_json_sha256(cls.evaluator_identity)
        cls.environment_fingerprint = cls._hash("environment")

        cls.development_rule = build_development_rule(
            "positive-independent-arm",
            minimum_level_utility_estimate=0.0,
            minimum_level_utility_lower_95=-0.02,
            minimum_point_margin_estimate=0.0,
            minimum_win_rate_estimate=-0.02,
            maximum_candidate_p95_latency_ms=20.0,
            minimum_candidate_completed_worlds_mean=2.0,
            style_metric_bounds={"throw-rate": {"minimum": 0.1, "maximum": 0.5}},
        )
        cls.ablation_comparison = build_comparison_protocol_manifest(
            cls.protocol,
            phase="W1.2",
            comparison_id="ablation-bid-ownership",
            subject_id="bid-ownership",
            seed_namespace="dev/ablation/bid-ownership",
            pair_count=200,
            shard_count=2,
            candidate_fingerprint=cls.candidate_fingerprint,
            control_fingerprint=cls.control_fingerprint,
            evaluator_fingerprint=cls.evaluator_fingerprint,
            environment_fingerprint=cls.environment_fingerprint,
            configuration_fingerprint=canonical_json_sha256(cls.combination_launch),
            development_rule=cls.development_rule,
            required_style_metrics=WEEK1_STYLE_METRICS,
        )
        cls.ablation_shards, cls.ablation_merged = cls._build_merged(
            cls.ablation_comparison
        )
        cls.fixture_report = cls._fixture_report()
        advancement_decision = build_w1_3_advancement_decision(
            cls.protocol,
            cls.ablation_comparison,
            cls.ablation_merged,
            fixture_report=cls.fixture_report,
        )
        survivor_comparison = build_comparison_protocol_manifest(
            cls.protocol,
            phase="W1.3",
            comparison_id="survivor-bid-ownership",
            subject_id="bid-ownership",
            seed_namespace="dev/survivor/bid-ownership",
            pair_count=800,
            shard_count=1,
            candidate_fingerprint=cls.candidate_fingerprint,
            control_fingerprint=cls.control_fingerprint,
            evaluator_fingerprint=cls.evaluator_fingerprint,
            environment_fingerprint=cls.environment_fingerprint,
            configuration_fingerprint=canonical_json_sha256(cls.combination_launch),
            development_rule=cls.development_rule,
        )
        _, survivor_merged = cls._build_merged(survivor_comparison)
        combination_qualification = build_comparison_protocol_manifest(
            cls.protocol,
            phase="W1.4",
            comparison_id="combination-qualification",
            subject_id="enoch-1-candidate",
            seed_namespace="dev/combination/qualification",
            pair_count=250,
            shard_count=1,
            candidate_fingerprint=cls.candidate_fingerprint,
            control_fingerprint=cls.control_fingerprint,
            evaluator_fingerprint=cls.evaluator_fingerprint,
            environment_fingerprint=cls.environment_fingerprint,
            configuration_fingerprint=canonical_json_sha256(cls.combination_launch),
            development_rule=cls.development_rule,
        )
        combination_screen = build_comparison_protocol_manifest(
            cls.protocol,
            phase="W1.4",
            comparison_id="combination-screen",
            subject_id="enoch-1-candidate",
            seed_namespace="dev/combination/screen",
            pair_count=800,
            shard_count=1,
            candidate_fingerprint=cls.candidate_fingerprint,
            control_fingerprint=cls.control_fingerprint,
            evaluator_fingerprint=cls.evaluator_fingerprint,
            environment_fingerprint=cls.environment_fingerprint,
            configuration_fingerprint=canonical_json_sha256(cls.combination_launch),
            development_rule=cls.development_rule,
        )
        cls.w1_4_campaign_lineage = enoch_week1_campaign.build_w1_4_campaign_lineage(
            cls.protocol,
            qualification_comparison=combination_qualification,
            qualification_launch_configuration=cls.combination_launch,
            qualification_identity_bindings=campaign_identities,
            screen_comparison=combination_screen,
            screen_launch_configuration=cls.combination_launch,
            screen_identity_bindings=campaign_identities,
            survivor_evidence=[
                {
                    "ablation": {
                        "comparison": cls.ablation_comparison,
                        "identity_bindings": campaign_identities,
                        "launch_configuration": cls.combination_launch,
                        "merged_result": cls.ablation_merged,
                    },
                    "advancement_decision": advancement_decision,
                    "arm_id": "bid-ownership",
                    "fixture_report": cls.fixture_report,
                    "survivor_screen": {
                        "comparison": survivor_comparison,
                        "identity_bindings": campaign_identities,
                        "launch_configuration": cls.combination_launch,
                        "merged_result": survivor_merged,
                    },
                }
            ],
        )
        _, combination_qualification_merged = cls._build_merged(
            combination_qualification
        )
        _, combination_screen_merged = cls._build_merged(combination_screen)
        cls.w1_4_candidate_decision = (
            enoch_week1_campaign.build_w1_4_candidate_decision(
                cls.protocol,
                cls.w1_4_campaign_lineage,
                qualification_merged_result=combination_qualification_merged,
                screen_merged_result=combination_screen_merged,
            )
        )

        configuration_hashes = {
            entry["namespace"]: cls._hash(f"configuration:{entry['namespace']}")
            for entry in QUALIFICATION_MATRIX
        }
        configuration_hashes["qual/intended"] = cls.candidate_identity[
            "configuration_sha256"
        ]
        cls.qualification_manifest = build_w1_5_qualification_manifest(
            cls.protocol,
            candidate_fingerprint=cls.candidate_fingerprint,
            control_fingerprint=cls.control_fingerprint,
            evaluator_fingerprint=cls.evaluator_fingerprint,
            environment_fingerprint=cls.environment_fingerprint,
            configuration_fingerprints=configuration_hashes,
            serving_envelope={
                "candidate_completed_worlds_mean_min": 2.0,
                "candidate_p50_latency_ms_max": 20.0,
                "candidate_p95_latency_ms_max": 20.0,
            },
            intended_equal_byte_identical=False,
            w1_4_candidate_decision=cls.w1_4_candidate_decision,
            shard_count=1,
        )
        cls.qualification_results = {}
        for comparison in cls.qualification_manifest["comparisons"]:
            _, merged = cls._build_merged(comparison)
            cls.qualification_results[comparison["comparison_id"]] = merged
        cls.qualification_decision = build_w1_5_qualification_decision(
            cls.protocol, cls.qualification_manifest, cls.qualification_results
        )

        cls.secondary_rule = build_development_rule(
            "locked-secondary-requirements",
            minimum_point_margin_estimate=0.0,
            minimum_win_rate_estimate=-0.02,
            maximum_candidate_p95_latency_ms=20.0,
            minimum_candidate_completed_worlds_mean=2.0,
        )
        cls.locked_configuration_fingerprint = cls._hash(
            "locked-standard-scenario-configuration"
        )
        cls.primary_manifest, cls.confirmation_manifest = build_locked_gate_manifests(
            cls.protocol,
            cls.qualification_decision,
            qualification_manifest=cls.qualification_manifest,
            qualification_merged_results=cls.qualification_results,
            candidate_identity=cls.candidate_identity,
            control_identity=cls.control_identity,
            evaluator_identity=cls.evaluator_identity,
            environment_sha256=cls.environment_fingerprint,
            locked_configuration_fingerprint=cls.locked_configuration_fingerprint,
            secondary_requirements=cls.secondary_rule,
            pair_count=1_500,
            shard_count=2,
        )
        _, cls.primary_merged = cls._build_merged(cls.primary_manifest["comparison"])
        _, cls.confirmation_merged = cls._build_merged(
            cls.confirmation_manifest["comparison"]
        )
        cls.primary_decision = build_locked_gate_decision(
            cls.protocol, cls.primary_manifest, cls.primary_merged
        )
        cls.confirmation_decision = build_locked_gate_decision(
            cls.protocol, cls.confirmation_manifest, cls.confirmation_merged
        )

    def test_canonical_arm_registry_and_control_manifest(self) -> None:
        arms = canonical_arm_registry()
        self.assertEqual(len(arms), 15)
        self.assertEqual(arms[0]["arm_id"], "bid-ownership")
        self.assertEqual(canonical_json_sha256(arms), ARM_REGISTRY_SHA256)

        control_manifest = build_w1_0_control_manifest(
            self.protocol,
            **self._w1_0_arguments(),
        )
        validate_w1_0_control_manifest(self.protocol, control_manifest)
        mismatched = copy.deepcopy(control_manifest)
        mismatched["artifact_hashes"]["binary/enoch-0"] = self._hash("wrong-binary")
        self._rehash(mismatched, "control_manifest_fingerprint")
        with self.assertRaisesRegex(ProtocolError, "binary identity"):
            validate_w1_0_control_manifest(self.protocol, mismatched)
        blocked = self._w1_0_arguments()
        blocked["effective_environment"] = {"SHENGJI_LEAK": "1"}
        with self.assertRaisesRegex(ProtocolError, "blocked variable"):
            build_w1_0_control_manifest(
                self.protocol,
                **blocked,
            )

    def test_phase_manifest_binds_predecessors_and_artifact_hashes(self) -> None:
        initial = build_phase_manifest(
            self.protocol,
            "W1.0",
            artifacts={"control": self._hash("control")},
            declarations={"status": "frozen"},
        )
        validate_phase_manifest(self.protocol, initial)
        baseline = build_phase_manifest(
            self.protocol,
            "W1.1",
            artifacts={"baseline": self._hash("baseline")},
            declarations={"workers": 8},
            parent_phase_manifests=[initial],
        )
        validate_phase_manifest(self.protocol, baseline)
        self.assertEqual(validate_phase_chain(self.protocol, [initial, baseline])[-1], baseline["phase_manifest_fingerprint"])
        with self.assertRaisesRegex(ProtocolError, "predecessor"):
            build_phase_manifest(
                self.protocol,
                "W1.1",
                artifacts={"baseline": self._hash("other-baseline")},
                declarations={"workers": 1},
            )
        forged_parent = copy.deepcopy(baseline)
        forged_parent["parent_phases"][0]["phase_manifest_fingerprint"] = self._hash(
            "invented-parent"
        )
        body = dict(forged_parent)
        body.pop("phase_manifest_fingerprint")
        forged_parent["phase_manifest_fingerprint"] = canonical_json_sha256(body)
        with self.assertRaisesRegex(ProtocolError, "does not match"):
            validate_phase_chain(self.protocol, [initial, forged_parent])

    def test_shard_validation_and_merge_are_exact_and_hash_bound(self) -> None:
        validate_comparison_protocol_manifest(self.protocol, self.ablation_comparison)
        expected_external_evidence = self.ablation_merged[
            "verified_external_evidence_fingerprint"
        ]
        for shard in self.ablation_shards:
            validate_shard_result(self.protocol, self.ablation_comparison, shard)
            self.assertEqual(
                shard["verified_external_evidence_fingerprint"],
                expected_external_evidence,
            )
        validate_merged_result(
            self.protocol, self.ablation_comparison, self.ablation_merged
        )
        with self.assertRaisesRegex(ProtocolError, "every declared shard"):
            merge_shard_results(
                self.protocol, self.ablation_comparison, self.ablation_shards[:1]
            )
        tampered = copy.deepcopy(self.ablation_shards[0])
        tampered["candidate_fingerprint"] = self._hash("wrong-candidate")
        with self.assertRaisesRegex(ProtocolError, "candidate_fingerprint"):
            validate_shard_result(self.protocol, self.ablation_comparison, tampered)
        incomplete = copy.deepcopy(self.ablation_shards[0])
        incomplete["records"].pop()
        with self.assertRaisesRegex(ProtocolError, "exact declared seed coverage"):
            validate_shard_result(self.protocol, self.ablation_comparison, incomplete)

        missing_evidence = copy.deepcopy(self.ablation_shards[0])
        missing_evidence.pop("verified_external_evidence_fingerprint")
        with self.assertRaisesRegex(ProtocolError, "shard result keys differ"):
            validate_shard_result(
                self.protocol, self.ablation_comparison, missing_evidence
            )

        mixed_evidence = copy.deepcopy(self.ablation_shards)
        mixed_evidence[1]["verified_external_evidence_fingerprint"] = self._hash(
            "other-verified-external-evidence"
        )
        self._rehash(mixed_evidence[1], "shard_result_fingerprint")
        with self.assertRaisesRegex(ProtocolError, "same verified external evidence"):
            merge_shard_results(
                self.protocol, self.ablation_comparison, mixed_evidence
            )

        unbound_merged = copy.deepcopy(self.ablation_merged)
        unbound_merged.pop("verified_external_evidence_fingerprint")
        with self.assertRaisesRegex(ProtocolError, "merged result keys differ"):
            validate_merged_result(
                self.protocol, self.ablation_comparison, unbound_merged
            )

    def test_w1_3_advancement_applies_predeclared_rule_and_fixtures(self) -> None:
        fixture_report = self._fixture_report()
        advanced = build_w1_3_advancement_decision(
            self.protocol,
            self.ablation_comparison,
            self.ablation_merged,
            fixture_report=fixture_report,
        )
        self.assertEqual(advanced["decision"], "advance-to-w1.3")
        validate_w1_3_advancement_decision(
            self.protocol,
            self.ablation_comparison,
            self.ablation_merged,
            fixture_report,
            advanced,
        )
        other_source = self._fixture_report(source_digest=self._hash("other-source"))
        with self.assertRaisesRegex(ProtocolError, "does not reconstruct"):
            validate_w1_3_advancement_decision(
                self.protocol,
                self.ablation_comparison,
                self.ablation_merged,
                other_source,
                advanced,
            )
        failed_report = copy.deepcopy(fixture_report)
        failed_report["failure_count"] = 1
        self._rehash(failed_report, "fixture_report_fingerprint")
        with self.assertRaisesRegex(ProtocolError, "fixture report contains failures"):
            build_w1_3_advancement_decision(
                self.protocol,
                self.ablation_comparison,
                self.ablation_merged,
                fixture_report=failed_report,
            )

    def test_w1_5_matrix_thresholds_and_decision_are_fixed(self) -> None:
        self.assertEqual(self.qualification_manifest["pair_count"], 3_300)
        self.assertEqual(len(self.qualification_manifest["active_matrix"]), 21)
        self.assertEqual(
            self.qualification_manifest["thresholds"], QUALIFICATION_THRESHOLDS
        )
        self.assertEqual(
            self.qualification_manifest["w1_4_candidate_decision_fingerprint"],
            self.w1_4_candidate_decision["w1_4_candidate_decision_fingerprint"],
        )
        self.assertEqual(
            self.qualification_decision["w1_4_candidate_decision_fingerprint"],
            self.w1_4_candidate_decision["w1_4_candidate_decision_fingerprint"],
        )
        validate_w1_5_qualification_manifest(
            self.protocol, self.qualification_manifest
        )
        self.assertEqual(
            self.qualification_decision["decision"], "eligible-for-locked-gate"
        )
        for summary in self.qualification_decision["comparison_summaries"]:
            self.assertEqual(
                summary["serving_envelope_applicable"],
                summary["category"] != "crossplay",
            )
        validate_w1_5_qualification_decision(
            self.protocol,
            self.qualification_manifest,
            self.qualification_results,
            self.qualification_decision,
        )

    def test_development_style_schema_and_all_outcome_uncertainty_are_mandatory(self) -> None:
        metrics = self.ablation_merged["metrics"]
        for name in ("level_utility", "point_margin", "win_rate"):
            self.assertEqual(len(metrics[name]["paired_bootstrap_95"]), 2)
            self.assertEqual(
                metrics[name]["paired_bootstrap_lower_95"],
                metrics[name]["paired_bootstrap_95"][0],
            )
            self.assertEqual(
                metrics[name]["paired_bootstrap_upper_95"],
                metrics[name]["paired_bootstrap_95"][1],
            )
            self.assertIsNotNone(metrics[name]["mde_95_80"])

        with self.assertRaisesRegex(ProtocolError, "frozen Week-1 style schema"):
            build_comparison_protocol_manifest(
                self.protocol,
                phase="W1.2",
                comparison_id="ablation-bid-ownership-empty-style",
                subject_id="bid-ownership",
                seed_namespace="dev/ablation/bid-ownership",
                pair_count=200,
                shard_count=2,
                candidate_fingerprint=self.candidate_fingerprint,
                control_fingerprint=self.control_fingerprint,
                evaluator_fingerprint=self.evaluator_fingerprint,
                environment_fingerprint=self.environment_fingerprint,
                configuration_fingerprint=self.candidate_identity[
                    "configuration_sha256"
                ],
                development_rule=self.development_rule,
                required_style_metrics=[],
            )

        with self.assertRaisesRegex(ProtocolError, "cannot be omitted"):
            build_w1_5_qualification_manifest(
                self.protocol,
                candidate_fingerprint=self.candidate_fingerprint,
                control_fingerprint=self.control_fingerprint,
                evaluator_fingerprint=self.evaluator_fingerprint,
                environment_fingerprint=self.environment_fingerprint,
                configuration_fingerprints=self.qualification_manifest[
                    "configuration_fingerprints"
                ],
                serving_envelope=self.qualification_manifest["serving_envelope"],
                intended_equal_byte_identical=True,
                w1_4_candidate_decision=self.w1_4_candidate_decision,
            )

    def test_locked_gates_share_frozen_candidate_but_not_seeds(self) -> None:
        validate_locked_gate_pair(
            self.protocol, self.primary_manifest, self.confirmation_manifest
        )
        self.assertEqual(
            self.primary_manifest["frozen_lock_sha256"],
            self.confirmation_manifest["frozen_lock_sha256"],
        )
        self.assertNotEqual(
            self.primary_manifest["comparison"]["seed_set_sha256"],
            self.confirmation_manifest["comparison"]["seed_set_sha256"],
        )
        self.assertEqual(self.primary_decision["decision"], "pass")
        self.assertEqual(self.confirmation_decision["decision"], "pass")
        self.assertEqual(
            self.primary_manifest["superiority_rule_sha256"],
            LOCKED_SUPERIORITY_RULE_SHA256,
        )
        validate_locked_gate_decision(
            self.protocol,
            self.confirmation_manifest,
            self.confirmation_merged,
            self.confirmation_decision,
        )
        wrong_candidate = dict(self.candidate_identity)
        wrong_candidate["binary_sha256"] = self._hash("post-qualification-binary")
        with self.assertRaisesRegex(ProtocolError, "differs from the qualified"):
            build_locked_gate_manifests(
                self.protocol,
                self.qualification_decision,
                qualification_manifest=self.qualification_manifest,
                qualification_merged_results=self.qualification_results,
                candidate_identity=wrong_candidate,
                control_identity=self.control_identity,
                evaluator_identity=self.evaluator_identity,
                environment_sha256=self.environment_fingerprint,
                locked_configuration_fingerprint=self.locked_configuration_fingerprint,
                secondary_requirements=self.secondary_rule,
                pair_count=1_500,
                shard_count=2,
            )
        with self.assertRaisesRegex(ProtocolError, "must differ from intended"):
            build_locked_gate_manifests(
                self.protocol,
                self.qualification_decision,
                qualification_manifest=self.qualification_manifest,
                qualification_merged_results=self.qualification_results,
                candidate_identity=self.candidate_identity,
                control_identity=self.control_identity,
                evaluator_identity=self.evaluator_identity,
                environment_sha256=self.environment_fingerprint,
                locked_configuration_fingerprint=self.qualification_manifest[
                    "configuration_fingerprints"
                ]["qual/intended"],
                secondary_requirements=self.secondary_rule,
                pair_count=1_500,
                shard_count=2,
            )

        mismatched_confirmation = copy.deepcopy(self.confirmation_manifest)
        other_locked_hash = self._hash("different-confirmation-launch")
        mismatched_confirmation["frozen_lock"][
            "locked_configuration_fingerprint"
        ] = other_locked_hash
        mismatched_confirmation["frozen_lock_sha256"] = canonical_json_sha256(
            mismatched_confirmation["frozen_lock"]
        )
        comparison = mismatched_confirmation["comparison"]
        comparison["configuration_fingerprint"] = other_locked_hash
        comparison_body = dict(comparison)
        comparison_body.pop("comparison_protocol_fingerprint")
        comparison["comparison_protocol_fingerprint"] = canonical_json_sha256(
            comparison_body
        )
        manifest_body = dict(mismatched_confirmation)
        manifest_body.pop("locked_gate_manifest_fingerprint")
        mismatched_confirmation["locked_gate_manifest_fingerprint"] = canonical_json_sha256(
            manifest_body
        )
        with self.assertRaisesRegex(ProtocolError, "do not share the frozen candidate"):
            validate_locked_gate_pair(
                self.protocol, self.primary_manifest, mismatched_confirmation
            )

    def test_locked_gate_rejects_timeout_and_changed_standard_deal_seed(self) -> None:
        comparison = self.primary_manifest["comparison"]
        timeout_shards = []
        for assignment in comparison["shards"]:
            records = [
                self._record(
                    comparison,
                    index,
                    failures={"timeout": 1} if index == 0 else None,
                )
                for index in assignment["seed_indices"]
            ]
            timeout_shards.append(
                build_shard_result(
                    self.protocol,
                    comparison,
                    assignment["shard_id"],
                    records,
                    verified_external_evidence_fingerprint=self.primary_merged[
                        "verified_external_evidence_fingerprint"
                    ],
                )
            )
        timeout_merged = merge_shard_results(
            self.protocol, comparison, timeout_shards
        )
        timeout_decision = build_locked_gate_decision(
            self.protocol, self.primary_manifest, timeout_merged
        )
        self.assertEqual(timeout_decision["decision"], "fail")
        self.assertIn("nonzero-failure-counter:timeout", timeout_decision["reasons"])

        assignment = comparison["shards"][0]
        changed_deals = [
            self._record(comparison, index) for index in assignment["seed_indices"]
        ]
        changed_deals[0]["effective_deal_seed"] ^= 1
        with self.assertRaisesRegex(ProtocolError, "changed its frozen deal seed"):
            build_shard_result(
                self.protocol,
                comparison,
                assignment["shard_id"],
                changed_deals,
                verified_external_evidence_fingerprint=self.primary_merged[
                    "verified_external_evidence_fingerprint"
                ],
            )

    def test_locked_gate_comparison_identity_and_schema_are_exact(self) -> None:
        for field, value, message in (
            ("comparison_id", "unrelated-id", "comparison id"),
            ("subject_id", "unrelated-subject", "subject must be enoch-1"),
        ):
            drifted = copy.deepcopy(self.primary_manifest)
            drifted["comparison"][field] = value
            self._rehash(
                drifted["comparison"], "comparison_protocol_fingerprint"
            )
            self._rehash(drifted, "locked_gate_manifest_fingerprint")
            with self.assertRaisesRegex(ProtocolError, message):
                validate_locked_gate_pair(
                    self.protocol, drifted, self.confirmation_manifest
                )

        drifted_schema = copy.deepcopy(self.confirmation_manifest)
        drifted_schema["comparison"]["required_style_metrics"] = ["throw-rate"]
        self._rehash(
            drifted_schema["comparison"], "comparison_protocol_fingerprint"
        )
        self._rehash(drifted_schema, "locked_gate_manifest_fingerprint")
        with self.assertRaisesRegex(ProtocolError, "comparison schemas differ"):
            validate_locked_gate_pair(
                self.protocol, self.primary_manifest, drifted_schema
            )

    def test_locked_superiority_rule_is_hash_guarded(self) -> None:
        original_rule = copy.deepcopy(enoch_week1_module.LOCKED_SUPERIORITY_RULE)
        mutated = copy.deepcopy(self.primary_manifest)
        try:
            enoch_week1_module.LOCKED_SUPERIORITY_RULE["threshold"] = 999.0
            mutated["superiority_rule"]["threshold"] = 999.0
            mutated["superiority_rule_sha256"] = canonical_json_sha256(
                mutated["superiority_rule"]
            )
            self._rehash(mutated, "locked_gate_manifest_fingerprint")
            with self.assertRaisesRegex(ProtocolError, "mutation.*superiority rule"):
                validate_locked_gate_pair(
                    self.protocol, mutated, self.confirmation_manifest
                )
        finally:
            enoch_week1_module.LOCKED_SUPERIORITY_RULE.clear()
            enoch_week1_module.LOCKED_SUPERIORITY_RULE.update(original_rule)

    def test_terminal_decision_freezes_only_after_two_gates_and_never_promotes(self) -> None:
        control_manifest = build_w1_0_control_manifest(
            self.protocol,
            **self._w1_0_arguments(),
        )
        control_manifest_fingerprint = control_manifest["control_manifest_fingerprint"]

        def phase_chain(last: int) -> list[dict]:
            manifests: list[dict] = []
            special = {
                0: control_manifest_fingerprint,
                5: self.qualification_decision["qualification_decision_fingerprint"],
                6: self.primary_decision["locked_gate_decision_fingerprint"],
                7: self.confirmation_decision["locked_gate_decision_fingerprint"],
            }
            for index in range(last + 1):
                digest = special.get(index, self._hash(f"phase-{index}-exit"))
                manifests.append(
                    build_phase_manifest(
                        self.protocol,
                        f"W1.{index}",
                        artifacts={f"phase-{index}-exit": digest},
                        declarations={"status": "complete"},
                        parent_phase_manifests=(manifests[-1:] if manifests else ()),
                    )
                )
            return manifests

        confirmed_chain = phase_chain(7)
        frozen = build_week1_decision_artifact(
            self.protocol,
            phase_manifests=confirmed_chain,
            control_manifest=control_manifest,
            enoch0_fingerprint=self.reference_control_fingerprint,
            candidate_fingerprint=self.candidate_fingerprint,
            qualification_manifest=self.qualification_manifest,
            qualification_merged_results=self.qualification_results,
            qualification_decision=self.qualification_decision,
            primary_gate_decision=self.primary_decision,
            confirmation_gate_decision=self.confirmation_decision,
            primary_gate_manifest=self.primary_manifest,
            primary_merged_result=self.primary_merged,
            confirmation_gate_manifest=self.confirmation_manifest,
            confirmation_merged_result=self.confirmation_merged,
            prerequisites_complete=True,
            evidence_fingerprints=[
                self.qualification_decision["qualification_decision_fingerprint"]
            ],
        )
        self.assertEqual(frozen["decision"], "freeze-enoch-1")
        self.assertEqual(
            frozen["evaluation_control_fingerprint"], self.control_fingerprint
        )
        self.assertNotEqual(
            frozen["evaluation_control_fingerprint"],
            frozen["permanent_scientific_control_fingerprint"],
        )
        self.assertFalse(frozen["production_promotion_authorized"])
        validate_week1_decision_artifact(self.protocol, frozen)
        wrong_control = copy.deepcopy(frozen)
        wrong_control["permanent_scientific_control_fingerprint"] = self._hash(
            "unrelated-control"
        )
        self._rehash(wrong_control, "week1_decision_fingerprint")
        with self.assertRaisesRegex(ProtocolError, "differs from W1.0 Enoch-0"):
            validate_week1_decision_artifact(self.protocol, wrong_control)
        same_as_control = copy.deepcopy(frozen)
        same_as_control["candidate_fingerprint"] = self.control_fingerprint
        same_as_control["downstream_primary_fingerprint"] = self.control_fingerprint
        same_as_control["primary_gate_decision"]["candidate_fingerprint"] = (
            self.control_fingerprint
        )
        same_as_control["confirmation_gate_decision"]["candidate_fingerprint"] = (
            self.control_fingerprint
        )
        for field in ("primary_gate_decision", "confirmation_gate_decision"):
            self._rehash(same_as_control[field], "locked_gate_decision_fingerprint")
        self._rehash(same_as_control, "week1_decision_fingerprint")
        with self.assertRaisesRegex(
            ProtocolError, "does not reconstruct|differs from manifest|must differ"
        ):
            validate_week1_decision_artifact(self.protocol, same_as_control)

        missing_raw_gate = copy.deepcopy(frozen)
        missing_raw_gate["primary_merged_result"] = None
        self._rehash(missing_raw_gate, "week1_decision_fingerprint")
        with self.assertRaisesRegex(ProtocolError, "lacks its manifest or merged result"):
            validate_week1_decision_artifact(self.protocol, missing_raw_gate)

        no_candidate = build_week1_decision_artifact(
            self.protocol,
            phase_manifests=phase_chain(4),
            control_manifest=control_manifest,
            enoch0_fingerprint=self.reference_control_fingerprint,
            candidate_fingerprint=None,
            primary_gate_decision=None,
            confirmation_gate_decision=None,
            prerequisites_complete=True,
            no_candidate_reason="no-survivor",
        )
        self.assertEqual(no_candidate["decision"], "no-confirmed-candidate")
        self.assertEqual(
            no_candidate["downstream_primary_fingerprint"],
            self.reference_control_fingerprint,
        )
        self.assertGreaterEqual(len(no_candidate["evidence_fingerprints"]), 10)
        with self.assertRaisesRegex(ProtocolError, "phase evidence through W1.4"):
            build_week1_decision_artifact(
                self.protocol,
                phase_manifests=phase_chain(3),
                control_manifest=control_manifest,
                enoch0_fingerprint=self.reference_control_fingerprint,
                candidate_fingerprint=None,
                primary_gate_decision=None,
                confirmation_gate_decision=None,
                prerequisites_complete=True,
                no_candidate_reason="no-survivor",
            )
        tampered = copy.deepcopy(frozen)
        tampered["production_promotion_authorized"] = True
        body = dict(tampered)
        body.pop("week1_decision_fingerprint")
        tampered["week1_decision_fingerprint"] = canonical_json_sha256(body)
        with self.assertRaisesRegex(ProtocolError, "cannot authorize"):
            validate_week1_decision_artifact(self.protocol, tampered)


if __name__ == "__main__":
    unittest.main()
