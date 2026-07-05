#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from training import enoch_week1
from training.enoch_week1_evidence import (
    build_machine_contention_attestation,
    build_verified_external_evidence,
)
from training.enoch_week1_fixtures import ARM_FIXTURES, GLOBAL_FIXTURES
from training.enoch_week1_runner import (
    ARM_TO_RUST_FEATURE,
    ARM_TO_RUST_FEATURE_SHA256,
    AUTHORITATIVE_PHASES,
    RunnerError,
    _validate_phase_environment_contract,
    authoritative_campaign_lock,
    build_evaluator_environment_identity,
    build_in_process_identity_bindings,
    build_launch_configuration,
    build_shard_command,
    run_comparison,
    rust_feature_spec,
    translate_evaluator_output,
    validate_identity_bindings,
)


FAKE_EVALUATOR = r'''#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

Path(os.environ["FAKE_ARGS_LOG"]).write_text(
    json.dumps(
        {
            "args": sys.argv[1:],
            "blocked_leak_present": "SHENGJI_LEAK" in os.environ,
            "ledger_claim_count_at_launch": (
                len(json.loads(Path(os.environ["FAKE_LEDGER"]).read_text())["consumed"])
                if os.environ.get("FAKE_LEDGER")
                and Path(os.environ["FAKE_LEDGER"]).exists()
                else None
            ),
        },
        sort_keys=True,
    ),
    encoding="utf-8",
)
if "--environment-identity-only" in sys.argv[1:]:
    if os.environ.get("FAKE_PROBE_EXIT"):
        print("forced probe failure", file=sys.stderr)
        raise SystemExit(int(os.environ["FAKE_PROBE_EXIT"]))
    sys.stdout.write(Path(os.environ["FAKE_ENVIRONMENT_PROBE"]).read_text(encoding="utf-8"))
    raise SystemExit(0)
if os.environ.get("FAKE_EXIT"):
    print("forced evaluator failure", file=sys.stderr)
    raise SystemExit(int(os.environ["FAKE_EXIT"]))
if os.environ.get("FAKE_NON_JSON"):
    print("not-json")
    raise SystemExit(0)
sys.stdout.write(Path(os.environ["FAKE_PAYLOAD"]).read_text(encoding="utf-8"))
'''


class Week1RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.protocol = enoch_week1.build_protocol(0x2026_0702)
        cls.protocol_path = cls.root / "protocol.json"
        enoch_week1.atomic_write_json(cls.protocol_path, cls.protocol)

        cls.evaluator = cls.root / "enoch_eval"
        cls.evaluator.write_text(FAKE_EVALUATOR, encoding="utf-8")
        cls.evaluator.chmod(0o755)
        cls.evaluator_binary_sha256 = cls._file_sha256(cls.evaluator)
        cls.source_records = [
            {
                "path": "core/src/bot/search.rs",
                "sha256": cls._hash("runner-source-record"),
            }
        ]
        cls.source_identity_path = cls.root / "source-identity.json"
        enoch_week1.atomic_write_json(
            cls.source_identity_path, cls.source_records
        )
        cls.evaluator_identity = {
            "binary_sha256": cls.evaluator_binary_sha256,
            "configuration_sha256": enoch_week1.canonical_json_sha256(
                enoch_week1.WEEK1_EVALUATOR_CONTRACT
            ),
            "source_sha256": cls._file_sha256(cls.source_identity_path),
        }
        cls.launch = build_launch_configuration(
            candidate_arm_ids=["failed-throw-better-player"],
            worlds=4,
            candidates=6,
            rollout_tricks=12,
            deadline_ms=30_000,
        )
        cls.identities = build_in_process_identity_bindings(
            cls.evaluator_identity, cls.launch
        )
        cls.candidate_identity = cls.identities["candidate"]
        cls.control_identity = cls.identities["control"]
        cls.identities_path = cls.root / "identities.json"
        enoch_week1.atomic_write_json(cls.identities_path, cls.identities)
        cls.launch_path = cls.root / "launch.json"
        enoch_week1.atomic_write_json(cls.launch_path, cls.launch)
        cls.environment_contract = build_evaluator_environment_identity(
            cls.evaluator_identity,
            cls.protocol,
            {},
            available_parallelism=1,
        )
        cls.environment_probe = {
            "environment": cls.environment_contract,
            "environment_identity_sha256": enoch_week1.canonical_json_sha256(
                cls.environment_contract
            ),
        }
        cls.environment_probe_path = cls.root / "environment-probe.json"
        enoch_week1.atomic_write_json(
            cls.environment_probe_path, cls.environment_probe
        )
        cls.comparison = enoch_week1.build_comparison_protocol_manifest(
            cls.protocol,
            phase="W1.1",
            comparison_id="runner-fake-smoke",
            subject_id="runner-fake",
            seed_namespace="smoke/product/001",
            pair_count=1,
            shard_count=1,
            candidate_fingerprint=enoch_week1.canonical_json_sha256(
                cls.candidate_identity
            ),
            control_fingerprint=enoch_week1.canonical_json_sha256(
                cls.control_identity
            ),
            evaluator_fingerprint=enoch_week1.canonical_json_sha256(
                cls.evaluator_identity
            ),
            environment_fingerprint=enoch_week1.canonical_json_sha256(
                cls.environment_contract
            ),
            configuration_fingerprint=enoch_week1.canonical_json_sha256(cls.launch),
            required_style_metrics=["throw-rate"],
        )
        cls.comparison_path = cls.root / "comparison.json"
        enoch_week1.atomic_write_json(cls.comparison_path, cls.comparison)
        cls.external = cls._build_external_evidence()
        cls.external_path = cls.root / "external.json"
        enoch_week1.atomic_write_json(cls.external_path, cls.external)
        cls.payload = cls._build_payload()
        cls.payload_path = cls.root / "payload.json"
        enoch_week1.atomic_write_json(cls.payload_path, cls.payload)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    @staticmethod
    def _hash(label: str) -> str:
        return enoch_week1.canonical_json_sha256({"test": label})

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            digest.update(source.read())
        return digest.hexdigest()

    @classmethod
    def _build_external_evidence(cls) -> dict:
        cls.model_evidence_paths = {
            "preflight/expert-model-validation": cls.root / "expert-model.json",
            "preflight/reference-model-contract-tests": cls.root
            / "reference-model-tests.json",
            "preflight/strict-evaluator-test": cls.root / "strict-evaluator.json",
        }
        enoch_week1.atomic_write_json(
            cls.model_evidence_paths["preflight/expert-model-validation"],
            {"command_succeeded": True, "output": "expert model validated"},
        )
        enoch_week1.atomic_write_json(
            cls.model_evidence_paths["preflight/reference-model-contract-tests"],
            {
                "command_succeeded": True,
                "output": "\n".join(
                    (
                        "test bot::expert::model_path_tests::model_path_override_round_trips ... ok",
                        "test bot::expert::model_path_tests::manifest_rejects_width_drift_and_untyped_v2_outputs ... ok",
                        "test bot::expert::model_path_tests::embedded_model_has_no_value_output ... ok",
                    )
                ),
            },
        )
        enoch_week1.atomic_write_json(
            cls.model_evidence_paths["preflight/strict-evaluator-test"],
            {
                "command_succeeded": True,
                "output": (
                    "test bot::search::tests::"
                    "strict_search_rejects_a_zero_sample_prior_fallback ... ok"
                ),
            },
        )

        no_model = enoch_week1.canonical_json_sha256(
            {"model": "none", "reason": "heuristic-policy-tier"}
        )
        production_source = cls._hash("production-source")
        expert_model = cls._hash("expert-model")
        search_knobs = {
            "enoch-0": {"policy": "enoch-0"},
            "expert-0": {"policy": "expert-0"},
            "grandmaster-0": {"policy": "grandmaster-0"},
        }
        policies = {
            name: enoch_week1.build_frozen_policy_identity(
                source_sha256=production_source,
                binary_sha256=cls._hash(f"{name}-binary"),
                model_sha256=expert_model if name == "expert-0" else no_model,
                configuration_sha256=enoch_week1.canonical_json_sha256(
                    search_knobs[name]
                ),
            )
            for name in ("enoch-0", "expert-0", "grandmaster-0")
        }
        artifact_hashes = {
            **{
                f"binary/{name}": identity["binary_sha256"]
                for name, identity in policies.items()
            },
            "binary/week1-evaluator": cls.evaluator_identity["binary_sha256"],
            "model/expert_model.onnx": expert_model,
            "protocol/week1-seed-protocol": cls._hash("protocol-file"),
            "source/production-reference": production_source,
            "source/week1-evaluator-file-list": cls._file_sha256(
                cls.source_identity_path
            ),
            **{
                artifact_id: cls._file_sha256(path)
                for artifact_id, path in cls.model_evidence_paths.items()
            },
        }
        control_manifest = enoch_week1.build_w1_0_control_manifest(
            cls.protocol,
            production_reference="c813c8a",
            policy_identities=policies,
            evaluator_identity=cls.evaluator_identity,
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
        cls.control_manifest_path = cls.root / "control-manifest.json"
        enoch_week1.atomic_write_json(
            cls.control_manifest_path, control_manifest
        )

        fixture_root = cls.root / "fixtures"
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
        fixture_body = {
            "arm_registry_sha256": enoch_week1.ARM_REGISTRY_SHA256,
            "automatic_production_promotion_allowed": False,
            "failure_count": 0,
            "manifest_kind": "enoch-week1-fixture-report",
            "manifest_version": 1,
            "records": records,
            "records_sha256": enoch_week1.canonical_json_sha256(records),
            "source_files": cls.source_records,
            "source_files_sha256": enoch_week1.canonical_json_sha256(
                cls.source_records
            ),
        }
        fixture_report = {
            **fixture_body,
            "fixture_report_fingerprint": enoch_week1.canonical_json_sha256(
                fixture_body
            ),
        }
        cls.fixture_report_path = fixture_root / "fixture-report.json"
        enoch_week1.atomic_write_json(cls.fixture_report_path, fixture_report)

        machine = build_machine_contention_attestation(
            cls.comparison,
            operator_id="runner-test-operator",
            observation_started_utc="2026-07-02T10:00:00Z",
            observation_ended_utc="2026-07-02T10:10:00Z",
            attested_at_utc="2026-07-02T10:11:00Z",
            worker_count=1,
            available_parallelism=1,
        )
        cls.machine_path = cls.root / "machine-attestation.json"
        enoch_week1.atomic_write_json(cls.machine_path, machine)
        return build_verified_external_evidence(
            cls.protocol,
            cls.comparison,
            fixture_report_path=cls.fixture_report_path,
            source_identity_path=cls.source_identity_path,
            control_manifest_path=cls.control_manifest_path,
            runner_identities_path=cls.identities_path,
            model_contract_artifact_paths=cls.model_evidence_paths,
            machine_attestation_path=cls.machine_path,
        )

    @classmethod
    def _arm_audit(cls) -> dict:
        return {
            "bound_counts": {"time": 0, "work": 2},
            "candidate_work": {
                "candidate_pool": 12,
                "evaluation_budget": 48,
                "evaluations_completed": 48,
                "initial_candidates": 12,
            },
            "decision_count": 2,
            "decisions_without_action": 0,
            "external_policy_fallback": 0,
            "internal_prior_fallback": 0,
            "invalid_counter_total": 0,
            "missing_search_telemetry": 0,
            "non_strict_decisions": 0,
            "search_failure_count": 0,
        }

    @classmethod
    def _pair_audit(cls) -> dict:
        return {
            "attribution_failures": 0,
            "candidate": cls._arm_audit(),
            "control": cls._arm_audit(),
            "invalid_counter_total": 0,
        }

    @classmethod
    def _build_payload(cls) -> dict:
        seed_entry = next(
            entry
            for entry in cls.protocol["seed_registry"]["namespaces"]
            if entry["name"] == "smoke/product/001"
        )
        seed = seed_entry["seeds"][0]
        protocol_identity = {
            "derivation_domain": enoch_week1.SEED_DERIVATION_DESCRIPTION["domain"],
            "domain_status": "verified-week1-protocol",
            "environment_allowlist": [],
            "environment_policy_verified": True,
            "manifest_version": 1,
            "master_seed_u64": cls.protocol["seed_registry"]["master_seed"],
            "namespace": "smoke/product/001",
            "protocol_fingerprint": cls.protocol["protocol_fingerprint"],
            "protocol_kind": cls.protocol["protocol_kind"],
            "registry_namespace_count": 1,
            "seed_registry_sha256": cls.protocol["seed_registry_sha256"],
        }
        scenario_contract = {
            "category": "standard",
            "evaluator_pairs_are_one_declared_shard_subset": True,
            "expected_seed_namespace": None,
            "id": "standard",
            "orientations": [],
            "qualification_total_pairs": None,
            "rules": {},
            "scenario_contract_version": 1,
            "threshold_deal_selection": None,
        }
        scenario_hash = enoch_week1.canonical_json_sha256(scenario_contract)
        search = {
            "candidate": {"features": {"canonical_spec": "failed-throw-witness"}},
            "control": {"features": {"canonical_spec": "none"}},
            "runner_style_metrics": ["throw-rate"],
            "scenario": scenario_contract,
            "strict_no_fallback": True,
            "work": {
                "max_candidates": 6,
                "max_worlds": 4,
                "mode": "fixed-work",
                "require_full_work": True,
                "rollout_tricks": 12,
                "time_budget_ms": 30_000,
            },
        }
        seed_records = [
            {
                "paired_index": 0,
                "registry_index": 0,
                "seed_hex": f"0x{seed:016x}",
                "seed_u64": seed,
            }
        ]
        seed_hash = enoch_week1.canonical_json_sha256(seed_records)
        evidence = {
            name: {
                "authority": f"fake-evaluator:{name}",
                "count": (
                    None
                    if name
                    in {
                        "artifact_mismatch",
                        "fixture_failure",
                        "hidden_information_leak",
                        "honesty_violation",
                        "machine_contention",
                    }
                    else 0
                ),
            }
            for name in enoch_week1.FAILURE_COUNTER_NAMES
        }
        pair = {
            "audit": cls._pair_audit(),
            "candidate_level_utility": 0.25,
            "candidate_point_margin": 5.0,
            "candidate_win_rate": 0.5,
            "candidate_wins": 1,
            "complete": True,
            "control_wins": 1,
            "deal_selection": {
                "attempts_examined": 1,
                "selection_attempt_zero_based": 0,
                "selector_non_landlord_points": None,
                "status": "direct-registry-seed",
            },
            "effective_deal_seed_hex": f"0x{seed:016x}",
            "effective_deal_seed_u64": seed,
            "hands_completed": 2,
            "hands_failed": 0,
            "orientations": [
                {
                    "candidate_is_landlord_team": True,
                    "candidate_won": True,
                    "complete": True,
                },
                {
                    "candidate_is_landlord_team": False,
                    "candidate_won": False,
                    "complete": True,
                },
            ],
            "paired_index": 0,
            "registry_index": 0,
            "registry_seed_hex": f"0x{seed:016x}",
            "registry_seed_u64": seed,
            "runner_record_inputs": {
                "candidate_completed_worlds": 8,
                "candidate_latency_ms": 1.5,
                "control_completed_worlds": 8,
                "control_latency_ms": 1.25,
                "failure_counter_evidence": evidence,
                "level_utility_delta": 0.25,
                "point_margin_delta": 5.0,
                "seed": seed,
                "seed_index": 0,
                "style_metrics": {"throw-rate": 0.1},
                "win_rate_delta": 0.0,
            },
            "seed_hex": f"0x{seed:016x}",
            "seed_u64": seed,
        }
        deal_records = [
            {
                "effective_deal_seed_hex": f"0x{seed:016x}",
                "effective_deal_seed_u64": seed,
                "paired_index": 0,
                "registry_index": 0,
                "registry_seed_hex": f"0x{seed:016x}",
                "registry_seed_u64": seed,
                "selection_attempt_zero_based": 0,
                "selector_non_landlord_points": None,
            }
        ]
        deal_records_hash = enoch_week1.canonical_json_sha256(deal_records)
        return {
            "audit": cls._pair_audit(),
            "candidate": {
                "feature_input": "failed-throw-witness",
                "features": {"canonical_spec": "failed-throw-witness"},
                "label": "Enoch(candidate:failed-throw-witness)",
            },
            "completion": {
                "audit_invalid_counter_total": 0,
                "candidate_wins": 1,
                "control_wins": 1,
                "hands_completed": 2,
                "hands_expected": 2,
                "hands_failed": 0,
                "pairs_complete": 1,
                "pairs_incomplete": 0,
                "pairs_requested": 1,
            },
            "control": {
                "features": {"canonical_spec": "none"},
                "label": "Enoch-0",
            },
            "deal_selection": {
                "contract": "direct-registry-seed-v1",
                "records": deal_records,
                "records_sha256": deal_records_hash,
                "scenario_id": "standard",
            },
            "evaluator": "enoch-eval-v4-scenarios",
            "manifest_version": 4,
            "merge_identity": {
                "environment": cls.environment_contract,
                "environment_identity_sha256": enoch_week1.canonical_json_sha256(
                    cls.environment_contract
                ),
                "merge_safe_seed_domain": True,
                "ordered_shard_seed_records_sha256": seed_hash,
                "ordered_effective_deal_records_sha256": deal_records_hash,
                "protocol": protocol_identity,
                "protocol_compatibility_sha256": enoch_week1.canonical_json_sha256(
                    {"scenario": scenario_contract, "seed_protocol": protocol_identity}
                ),
                "scenario": scenario_contract,
                "scenario_identity_sha256": scenario_hash,
                "schema": "enoch-eval-deterministic-shard-merge-v3-scenarios",
                "search": search,
                "search_identity_sha256": enoch_week1.canonical_json_sha256(search),
            },
            "method": "direct in-process audited configured-subject mirrored-deal pairs",
            "metrics": {
                "level_utility": {"estimate": 0.25, "paired_observations": 1},
                "point_margin": {"estimate": 5.0, "paired_observations": 1},
                "win_rate": {"estimate": 0.5, "paired_observations": 1},
            },
            "paired_records": [pair],
            "per_deck": {
                "candidate_level_utility": [0.25],
                "candidate_point_margin": [5.0],
                "candidate_win_rate": [0.5],
                "complete_effective_deal_seed_hex": [f"0x{seed:016x}"],
                "complete_effective_deal_seed_u64": [seed],
                "complete_registry_index": [0],
                "complete_seed_hex": [f"0x{seed:016x}"],
                "complete_seed_u64": [seed],
            },
            "seed_consumption": {
                "ordered_seed_records_sha256": seed_hash,
                "pairs_requested": 1,
                "records": seed_records,
                "source_kind": "week1-protocol-registry",
                "source_path": str(cls.protocol_path.resolve()),
            },
            "scenario": {
                "contract": scenario_contract,
                "expected_namespace": None,
                "id": "standard",
                "identity_sha256": scenario_hash,
            },
            "settings": {
                "max_candidates": 6,
                "max_worlds": 4,
                "mode": "fixed-work",
                "require_full_work": True,
                "rollout_tricks": 12,
                "runner_style_metrics": ["throw-rate"],
                "scenario_id": "standard",
                "time_budget_ms": 30_000,
            },
            "valid": True,
        }

    def _environment(self, log: Path, **extra: str) -> dict[str, str]:
        environment = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "FAKE_ARGS_LOG": str(log),
            "FAKE_PAYLOAD": str(self.payload_path),
            "FAKE_ENVIRONMENT_PROBE": str(self.environment_probe_path),
            "SHENGJI_LEAK": "must-be-removed",
        }
        environment.update(extra)
        return environment

    def _run_comparison(self, **kwargs: object) -> dict:
        with authoritative_campaign_lock(self.protocol, self.comparison) as token:
            return run_comparison(
                **kwargs,
                campaign_lock_token=token,
            )

    def test_mapping_is_complete_and_never_heuristic(self) -> None:
        self.assertEqual(
            AUTHORITATIVE_PHASES,
            {"W1.1", "W1.2", "W1.3", "W1.4", "W1.5", "W1.6", "W1.7"},
        )
        self.assertEqual(
            tuple(name for name, _ in ARM_TO_RUST_FEATURE),
            enoch_week1.ABLATION_ARMS,
        )
        self.assertEqual(
            enoch_week1.canonical_json_sha256(ARM_TO_RUST_FEATURE),
            ARM_TO_RUST_FEATURE_SHA256,
        )
        self.assertEqual(rust_feature_spec(self.launch), "failed-throw-witness")
        kitty = build_launch_configuration(
            candidate_arm_ids=["kitty-burial"],
            worlds=1,
            candidates=1,
            rollout_tricks=1,
            budget_ms=10,
        )
        self.assertEqual(rust_feature_spec(kitty), "default-kitty")

    def test_authoritative_execution_requires_one_protocol_global_live_lock(self) -> None:
        ledger = self.root / "missing-lock-ledger.json"
        output = self.root / "missing-lock-output"
        log = self.root / "missing-lock-log.json"
        with self.assertRaisesRegex(RunnerError, "live campaign lock token"):
            run_comparison(
                protocol_path=self.protocol_path,
                comparison_path=self.comparison_path,
                launch_configuration_path=self.launch_path,
                identities_path=self.identities_path,
                evaluator=self.evaluator,
                ledger_path=ledger,
                output_dir=output,
                external_evidence_path=self.external_path,
                workers=1,
                dry_run=True,
                base_environment=self._environment(log),
                available_parallelism=1,
            )
        self.assertFalse(ledger.exists())
        self.assertFalse(output.exists())
        self.assertFalse(log.exists())

        other_protocol = enoch_week1.build_protocol(0xD1571C7)
        other_comparison = enoch_week1.build_comparison_protocol_manifest(
            other_protocol,
            phase="W1.1",
            comparison_id="runner-other-protocol",
            subject_id="runner-fake",
            seed_namespace="smoke/product/001",
            pair_count=1,
            shard_count=1,
            candidate_fingerprint=self.comparison["candidate_fingerprint"],
            control_fingerprint=self.comparison["control_fingerprint"],
            evaluator_fingerprint=self.comparison["evaluator_fingerprint"],
            environment_fingerprint=self.comparison["environment_fingerprint"],
            configuration_fingerprint=self.comparison["configuration_fingerprint"],
            required_style_metrics=["throw-rate"],
        )
        previous_tmpdir = os.environ.get("TMPDIR")
        with authoritative_campaign_lock(self.protocol, self.comparison):
            os.environ["TMPDIR"] = str(self.root / "alternate-tmp")
            try:
                with self.assertRaisesRegex(RunnerError, "another authoritative"):
                    with authoritative_campaign_lock(
                        other_protocol, other_comparison
                    ):
                        self.fail("distinct protocol lock unexpectedly succeeded")
            finally:
                if previous_tmpdir is None:
                    os.environ.pop("TMPDIR", None)
                else:
                    os.environ["TMPDIR"] = previous_tmpdir

    def test_w1_5_scenario_is_exactly_bound_to_comparison_and_namespace(self) -> None:
        intended_launch = build_launch_configuration(
            candidate_arm_ids=["failed-throw-better-player"],
            worlds=144,
            candidates=6,
            rollout_tricks=12,
            scenario_id="intended",
            budget_ms=2_200,
        )
        comparison = enoch_week1.build_comparison_protocol_manifest(
            self.protocol,
            phase="W1.5",
            comparison_id="qual-intended",
            subject_id="enoch-1",
            seed_namespace="qual/intended",
            pair_count=800,
            shard_count=8,
            candidate_fingerprint=enoch_week1.canonical_json_sha256(
                self.candidate_identity
            ),
            control_fingerprint=enoch_week1.canonical_json_sha256(
                self.control_identity
            ),
            evaluator_fingerprint=enoch_week1.canonical_json_sha256(
                self.evaluator_identity
            ),
            environment_fingerprint=enoch_week1.canonical_json_sha256(
                self.environment_contract
            ),
            configuration_fingerprint=enoch_week1.canonical_json_sha256(
                intended_launch
            ),
        )
        command = build_shard_command(
            self.evaluator,
            self.protocol_path,
            comparison,
            intended_launch,
            "shard-000",
        )
        scenario_position = command.index("--scenario")
        self.assertEqual(command[scenario_position + 1], "intended")
        wrong = dict(intended_launch)
        wrong["scenario_id"] = "standard"
        with self.assertRaisesRegex(RunnerError, "scenario must be 'intended'"):
            build_shard_command(
                self.evaluator,
                self.protocol_path,
                comparison,
                wrong,
                "shard-000",
            )

    def test_phase_modes_and_friend_development_scenario_are_mandatory(self) -> None:
        friend_launch = build_launch_configuration(
            candidate_arm_ids=["friend-revelation"],
            worlds=16,
            candidates=4,
            rollout_tricks=4,
            scenario_id="development-finding-friends",
            deadline_ms=30_000,
        )
        friend = enoch_week1.build_comparison_protocol_manifest(
            self.protocol,
            phase="W1.2",
            comparison_id="ablation-friend-revelation",
            subject_id="friend-revelation",
            seed_namespace="dev/ablation/friend-revelation",
            pair_count=200,
            shard_count=2,
            candidate_fingerprint=self.comparison["candidate_fingerprint"],
            control_fingerprint=self.comparison["control_fingerprint"],
            evaluator_fingerprint=self.comparison["evaluator_fingerprint"],
            environment_fingerprint=self.comparison["environment_fingerprint"],
            configuration_fingerprint=enoch_week1.canonical_json_sha256(friend_launch),
            development_rule=enoch_week1.build_development_rule(
                "friend-development", minimum_level_utility_estimate=0.0
            ),
        )
        command = build_shard_command(
            self.evaluator, self.protocol_path, friend, friend_launch, "shard-000"
        )
        self.assertEqual(command[command.index("--scenario") + 1], "development-finding-friends")

        inert = dict(friend_launch)
        inert["scenario_id"] = "standard"
        with self.assertRaisesRegex(RunnerError, "Finding Friends scenario"):
            build_shard_command(
                self.evaluator, self.protocol_path, friend, inert, "shard-000"
            )

        budget = build_launch_configuration(
            candidate_arm_ids=["friend-revelation"],
            worlds=16,
            candidates=4,
            rollout_tricks=4,
            scenario_id="development-finding-friends",
            budget_ms=2_200,
        )
        with self.assertRaisesRegex(RunnerError, "W1.2 comparisons require.*fixed work"):
            build_shard_command(
                self.evaluator, self.protocol_path, friend, budget, "shard-000"
            )

        equal_budget = build_launch_configuration(
            candidate_arm_ids=["failed-throw-better-player"],
            worlds=16,
            candidates=4,
            rollout_tricks=4,
            scenario_id="equal",
            budget_ms=2_200,
        )
        equal = enoch_week1.build_comparison_protocol_manifest(
            self.protocol,
            phase="W1.5",
            comparison_id="qual-equal",
            subject_id="enoch-1",
            seed_namespace="qual/equal",
            pair_count=800,
            shard_count=8,
            candidate_fingerprint=self.comparison["candidate_fingerprint"],
            control_fingerprint=self.comparison["control_fingerprint"],
            evaluator_fingerprint=self.comparison["evaluator_fingerprint"],
            environment_fingerprint=self.comparison["environment_fingerprint"],
            configuration_fingerprint=enoch_week1.canonical_json_sha256(equal_budget),
        )
        with self.assertRaisesRegex(RunnerError, "qual-equal requires fixed-work"):
            build_shard_command(
                self.evaluator, self.protocol_path, equal, equal_budget, "shard-000"
            )

        locked_fixed_work = build_launch_configuration(
            candidate_arm_ids=["failed-throw-better-player"],
            worlds=144,
            candidates=6,
            rollout_tricks=12,
            deadline_ms=30_000,
        )
        locked = enoch_week1.build_comparison_protocol_manifest(
            self.protocol,
            phase="W1.6",
            comparison_id="locked-primary",
            subject_id="enoch-1",
            seed_namespace="locked/gate-1",
            pair_count=1_500,
            shard_count=10,
            candidate_fingerprint=self.comparison["candidate_fingerprint"],
            control_fingerprint=self.comparison["control_fingerprint"],
            evaluator_fingerprint=self.comparison["evaluator_fingerprint"],
            environment_fingerprint=self.comparison["environment_fingerprint"],
            configuration_fingerprint=enoch_week1.canonical_json_sha256(
                locked_fixed_work
            ),
        )
        with self.assertRaisesRegex(RunnerError, "W1.6 locked gates require"):
            build_shard_command(
                self.evaluator,
                self.protocol_path,
                locked,
                locked_fixed_work,
                "shard-000",
            )

    def test_identity_arm_set_and_authoritative_environment_are_runtime_bound(self) -> None:
        changed_launch = build_launch_configuration(
            candidate_arm_ids=["kitty-burial"],
            worlds=4,
            candidates=6,
            rollout_tricks=12,
            deadline_ms=30_000,
        )
        with self.assertRaisesRegex(RunnerError, "not the policy compiled"):
            validate_identity_bindings(
                self.comparison, self.identities, self.evaluator, changed_launch
            )

        allowlisted = enoch_week1.build_protocol(
            0xBAD5EED,
            evaluator_env_allowlist=["SHENGJI_LATE_RUFF_RESERVE"],
        )
        authoritative = dict(self.comparison)
        authoritative["phase"] = "W1.2"
        with self.assertRaisesRegex(RunnerError, "empty evaluator environment allowlist"):
            _validate_phase_environment_contract(authoritative, allowlisted)
        _validate_phase_environment_contract(self.comparison, allowlisted)

    def test_environment_identity_known_vector_and_drift_fail_closed(self) -> None:
        rebuilt = build_evaluator_environment_identity(
            self.evaluator_identity,
            self.protocol,
            {},
            available_parallelism=1,
        )
        self.assertEqual(rebuilt, self.environment_contract)
        vector_protocol = enoch_week1.build_protocol(
            0xE10C, evaluator_env_allowlist=["SHENGJI_ALPHA"]
        )
        vector_identity = build_evaluator_environment_identity(
            {
                "binary_sha256": "0" * 64,
                "configuration_sha256": "2" * 64,
                "source_sha256": "1" * 64,
            },
            vector_protocol,
            {"SHENGJI_ALPHA": 'café/quote"'},
            available_parallelism=10,
        )
        self.assertEqual(
            enoch_week1.canonical_json_sha256(vector_identity),
            "552326c51b0a5acc67e3df86bb97a337a34472e900e64242991638894c2b86e7",
        )

        log = self.root / "environment-drift-log.json"
        with self.assertRaisesRegex(
            RunnerError, "declared available parallelism differs from verified"
        ):
            self._run_comparison(
                protocol_path=self.protocol_path,
                comparison_path=self.comparison_path,
                launch_configuration_path=self.launch_path,
                identities_path=self.identities_path,
                evaluator=self.evaluator,
                ledger_path=self.root / "environment-drift-ledger.json",
                output_dir=self.root / "environment-drift-output",
                external_evidence_path=self.external_path,
                workers=1,
                dry_run=True,
                base_environment=self._environment(log),
                available_parallelism=2,
            )
        self.assertFalse((self.root / "environment-drift-ledger.json").exists())
        self.assertFalse((self.root / "environment-drift-output").exists())
        self.assertFalse(log.exists())

        cleaned, _ = enoch_week1.sanitized_evaluator_environment(
            self._environment(log)
        )
        for field, changed in (
            ("available_parallelism", 2),
            ("effective_experiment_environment", {"SHENGJI_LEAK": "1"}),
        ):
            payload = copy.deepcopy(self.payload)
            payload["merge_identity"]["environment"][field] = changed
            payload["merge_identity"]["environment_identity_sha256"] = (
                enoch_week1.canonical_json_sha256(
                    payload["merge_identity"]["environment"]
                )
            )
            with self.subTest(field=field), self.assertRaisesRegex(
                RunnerError, field
            ):
                translate_evaluator_output(
                    self.protocol,
                    self.comparison,
                    "shard-000",
                    payload,
                    self.launch,
                    self.identities,
                    cleaned,
                    self.external,
                    available_parallelism=1,
                )

    def test_probe_and_worker_drift_fail_before_ledger_mutation(self) -> None:
        worker_ledger = self.root / "worker-drift-ledger.json"
        worker_output = self.root / "worker-drift-output"
        worker_log = self.root / "worker-drift-log.json"
        with self.assertRaisesRegex(RunnerError, "worker count differs"):
            self._run_comparison(
                protocol_path=self.protocol_path,
                comparison_path=self.comparison_path,
                launch_configuration_path=self.launch_path,
                identities_path=self.identities_path,
                evaluator=self.evaluator,
                ledger_path=worker_ledger,
                output_dir=worker_output,
                external_evidence_path=self.external_path,
                workers=2,
                base_environment=self._environment(worker_log),
                available_parallelism=1,
            )
        self.assertFalse(worker_ledger.exists())
        self.assertFalse(worker_output.exists())
        self.assertFalse(worker_log.exists())

        wrong_probe = copy.deepcopy(self.environment_probe)
        wrong_probe["environment"]["available_parallelism"] = 2
        wrong_probe["environment_identity_sha256"] = enoch_week1.canonical_json_sha256(
            wrong_probe["environment"]
        )
        wrong_probe_path = self.root / "wrong-environment-probe.json"
        enoch_week1.atomic_write_json(wrong_probe_path, wrong_probe)
        probe_ledger = self.root / "probe-drift-ledger.json"
        probe_output = self.root / "probe-drift-output"
        probe_log = self.root / "probe-drift-log.json"
        with self.assertRaisesRegex(RunnerError, "probe differs from the frozen contract"):
            self._run_comparison(
                protocol_path=self.protocol_path,
                comparison_path=self.comparison_path,
                launch_configuration_path=self.launch_path,
                identities_path=self.identities_path,
                evaluator=self.evaluator,
                ledger_path=probe_ledger,
                output_dir=probe_output,
                external_evidence_path=self.external_path,
                workers=1,
                base_environment=self._environment(
                    probe_log, FAKE_ENVIRONMENT_PROBE=str(wrong_probe_path)
                ),
                available_parallelism=1,
            )
        self.assertFalse(probe_ledger.exists())
        self.assertFalse(probe_output.exists())
        invocation = json.loads(probe_log.read_text(encoding="utf-8"))
        self.assertIn("--environment-identity-only", invocation["args"])
        self.assertIsNone(invocation["ledger_claim_count_at_launch"])

    def test_dry_run_emits_exact_seed_arguments_without_claiming(self) -> None:
        ledger = self.root / "dry-ledger.json"
        output = self.root / "dry-output"
        log = self.root / "dry-args.json"
        result = self._run_comparison(
            protocol_path=self.protocol_path,
            comparison_path=self.comparison_path,
            launch_configuration_path=self.launch_path,
            identities_path=self.identities_path,
            evaluator=self.evaluator,
            ledger_path=ledger,
            output_dir=output,
            external_evidence_path=self.external_path,
            workers=1,
            dry_run=True,
            base_environment=self._environment(log, FAKE_LEDGER=str(ledger)),
            available_parallelism=1,
        )
        expected = build_shard_command(
            self.evaluator,
            self.protocol_path,
            self.comparison,
            self.launch,
            "shard-000",
        )
        self.assertEqual(result["commands"], [expected])
        self.assertEqual(
            expected[1:13],
            [
                "--pairs",
                "1",
                "--seeds-json",
                str(self.protocol_path.resolve()),
                "--seed-namespace",
                "smoke/product/001",
                "--seed-index",
                "0",
                "--features",
                "failed-throw-witness",
                "--worlds",
                "4",
            ],
        )
        self.assertIn("--style-metric", expected)
        self.assertEqual(result["environment_identity_probe"], self.environment_probe)
        self.assertFalse(ledger.exists())
        self.assertFalse(output.exists())
        invocation = json.loads(log.read_text(encoding="utf-8"))
        self.assertEqual(
            invocation["args"], expected[1:] + ["--environment-identity-only"]
        )
        self.assertFalse(invocation["blocked_leak_present"])
        self.assertIsNone(invocation["ledger_claim_count_at_launch"])

    def test_fake_evaluator_success_claims_translates_and_merges(self) -> None:
        ledger = self.root / "success-ledger.json"
        output = self.root / "success-output"
        log = self.root / "success-args.json"
        result = self._run_comparison(
            protocol_path=self.protocol_path,
            comparison_path=self.comparison_path,
            launch_configuration_path=self.launch_path,
            identities_path=self.identities_path,
            evaluator=self.evaluator,
            ledger_path=ledger,
            output_dir=output,
            external_evidence_path=self.external_path,
            workers=1,
            merge=True,
            base_environment=self._environment(log, FAKE_LEDGER=str(ledger)),
            available_parallelism=1,
        )
        invocation = json.loads(log.read_text(encoding="utf-8"))
        self.assertFalse(invocation["blocked_leak_present"])
        self.assertEqual(invocation["ledger_claim_count_at_launch"], 1)
        self.assertEqual(invocation["args"], result["commands"][0][1:])
        consumed = enoch_week1.load_json_object(ledger)
        self.assertEqual(len(consumed["consumed"]), 1)
        self.assertEqual(consumed["consumed"][0]["namespace"], "smoke/product/001")
        self.assertTrue((output / "shard-000.raw.json").is_file())
        self.assertTrue((output / "shard-000.stderr.txt").is_file())
        self.assertTrue((output / "shard-000.result.json").is_file())
        self.assertTrue((output / "merged-result.json").is_file())
        self.assertTrue((output / "external-failure-evidence.json").is_file())
        self.assertEqual(
            result["external_failure_evidence_fingerprint"],
            self.external["verified_external_evidence_fingerprint"],
        )
        shard_result = enoch_week1.load_json_object(
            output / "shard-000.result.json"
        )
        merged_result = enoch_week1.load_json_object(output / "merged-result.json")
        expected_external_fingerprint = self.external[
            "verified_external_evidence_fingerprint"
        ]
        self.assertEqual(
            shard_result["verified_external_evidence_fingerprint"],
            expected_external_fingerprint,
        )
        self.assertEqual(
            merged_result["verified_external_evidence_fingerprint"],
            expected_external_fingerprint,
        )
        self.assertEqual(
            shard_result["records"][0]["effective_deal_seed"],
            shard_result["records"][0]["seed"],
        )

    def test_nonzero_exit_preserves_output_and_consumes_claim(self) -> None:
        ledger = self.root / "failure-ledger.json"
        output = self.root / "failure-output"
        log = self.root / "failure-args.json"
        with self.assertRaisesRegex(RunnerError, "exited with status 7"):
            self._run_comparison(
                protocol_path=self.protocol_path,
                comparison_path=self.comparison_path,
                launch_configuration_path=self.launch_path,
                identities_path=self.identities_path,
                evaluator=self.evaluator,
                ledger_path=ledger,
                output_dir=output,
                external_evidence_path=self.external_path,
                workers=1,
                base_environment=self._environment(log, FAKE_EXIT="7"),
                available_parallelism=1,
            )
        self.assertEqual(len(enoch_week1.load_json_object(ledger)["consumed"]), 1)
        self.assertTrue((output / "failures" / "shard-000.stdout").is_file())
        self.assertTrue((output / "failures" / "shard-000.stderr").is_file())
        self.assertTrue((output / "execution-failed.json").is_file())

    def test_non_json_stdout_fails_closed_after_claim(self) -> None:
        ledger = self.root / "nonjson-ledger.json"
        output = self.root / "nonjson-output"
        log = self.root / "nonjson-args.json"
        with self.assertRaisesRegex(RunnerError, "not one JSON document"):
            self._run_comparison(
                protocol_path=self.protocol_path,
                comparison_path=self.comparison_path,
                launch_configuration_path=self.launch_path,
                identities_path=self.identities_path,
                evaluator=self.evaluator,
                ledger_path=ledger,
                output_dir=output,
                external_evidence_path=self.external_path,
                workers=1,
                base_environment=self._environment(log, FAKE_NON_JSON="1"),
                available_parallelism=1,
            )
        self.assertEqual(len(enoch_week1.load_json_object(ledger)["consumed"]), 1)
        self.assertEqual(
            (output / "failures" / "shard-000.stdout").read_text(encoding="utf-8"),
            "not-json\n",
        )

    def test_translation_rejects_missing_evidence_seed_fallback_and_incomplete(self) -> None:
        cleaned, _ = enoch_week1.sanitized_evaluator_environment(
            self._environment(self.root / "unused-log.json")
        )
        with self.assertRaisesRegex(RunnerError, "external evidence"):
            translate_evaluator_output(
                self.protocol,
                self.comparison,
                "shard-000",
                self.payload,
                self.launch,
                self.identities,
                cleaned,
                None,
                available_parallelism=1,
            )

        fallback = copy.deepcopy(self.payload)
        fallback["paired_records"][0]["runner_record_inputs"][
            "failure_counter_evidence"
        ]["model_fallback"]["count"] = 1
        with self.assertRaisesRegex(RunnerError, "invalidating evaluator counters"):
            translate_evaluator_output(
                self.protocol,
                self.comparison,
                "shard-000",
                fallback,
                self.launch,
                self.identities,
                cleaned,
                self.external,
                available_parallelism=1,
            )

        wrong_seed = copy.deepcopy(self.payload)
        wrong_seed["paired_records"][0]["seed_u64"] ^= 1
        with self.assertRaisesRegex(RunnerError, "index/seed mismatch"):
            translate_evaluator_output(
                self.protocol,
                self.comparison,
                "shard-000",
                wrong_seed,
                self.launch,
                self.identities,
                cleaned,
                self.external,
                available_parallelism=1,
            )

        incomplete = copy.deepcopy(self.payload)
        incomplete["paired_records"][0]["complete"] = False
        with self.assertRaisesRegex(RunnerError, "not a complete"):
            translate_evaluator_output(
                self.protocol,
                self.comparison,
                "shard-000",
                incomplete,
                self.launch,
                self.identities,
                cleaned,
                self.external,
                available_parallelism=1,
            )

        wrong_effective_seed = copy.deepcopy(self.payload)
        wrong_effective_seed["paired_records"][0]["effective_deal_seed_u64"] ^= 1
        with self.assertRaisesRegex(RunnerError, "effective deal seed mismatch"):
            translate_evaluator_output(
                self.protocol,
                self.comparison,
                "shard-000",
                wrong_effective_seed,
                self.launch,
                self.identities,
                cleaned,
                self.external,
                available_parallelism=1,
            )

        time_bound = copy.deepcopy(self.payload)
        time_bound["paired_records"][0]["audit"]["candidate"]["bound_counts"][
            "time"
        ] = 1
        with self.assertRaisesRegex(
            RunnerError, "fixed-work audit reports 1 time-bound decision"
        ):
            translate_evaluator_output(
                self.protocol,
                self.comparison,
                "shard-000",
                time_bound,
                self.launch,
                self.identities,
                cleaned,
                self.external,
                available_parallelism=1,
            )

        incomplete_work = copy.deepcopy(self.payload)
        incomplete_work["paired_records"][0]["audit"]["candidate"]["candidate_work"][
            "evaluations_completed"
        ] -= 1
        with self.assertRaisesRegex(RunnerError, "completed 47 of 48"):
            translate_evaluator_output(
                self.protocol,
                self.comparison,
                "shard-000",
                incomplete_work,
                self.launch,
                self.identities,
                cleaned,
                self.external,
                available_parallelism=1,
            )


if __name__ == "__main__":
    unittest.main()
