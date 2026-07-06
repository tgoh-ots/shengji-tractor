#!/usr/bin/env python3
"""Adversarial tests for the recovery-aware W1.4 continuation."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from training import enoch_week1
from training import enoch_week1_campaign
from training import enoch_week1_w1_4_operator as w14


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class PlanAndProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = Path(__file__).with_name("enoch_week1_w1_4_plan.json")
        cls.plan = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_exact_plan_candidate_rule_and_stages(self) -> None:
        self.assertEqual(
            w14.validate_committed_plan(self.plan),
            enoch_week1.canonical_json_sha256(self.plan),
        )
        self.assertEqual(self.plan["candidate_arm_ids"], list(w14.CANDIDATE_ARM_IDS))
        self.assertNotIn(w14.STOPPED_ARM_ID, self.plan["candidate_arm_ids"])
        self.assertEqual(self.plan["development_rule"], w14._canonical_rule())
        self.assertEqual([item["pair_count"] for item in self.plan["stages"]], [300, 800])
        self.assertEqual(
            [item["comparison_protocol_fingerprint"] for item in self.plan["stages"]],
            [
                "b8df3bc4f03bece9ac24e86e5f78fa971b00e82f2b079d38db711b9cd8abe8b1",
                "99899173addfae419add144d73f9760fbe9499a67666e285dfae744d91d99f88",
            ],
        )
        self.assertEqual(
            self.plan["campaign_lineage_fingerprint"],
            "b81c9709773431adea7e266d18badb2bcfb4f4fb4ce9d79e4b9444b5fee162fd",
        )
        self.assertEqual(
            [item["shard_pair_counts"] for item in self.plan["stages"]],
            [[38, 38, 38, 38, 37, 37, 37, 37], [100] * 8],
        )
        self.assertEqual(self.plan["final_consumed_count"], 19_811)
        self.assertEqual(
            self.plan["source_control_contract"], w14._source_control_contract()
        )
        self.assertEqual(
            self.plan["rust_toolchain_contract"], w14._rust_toolchain_contract()
        )

    def test_source_control_environment_and_git_argv_ignore_ambient_path(self) -> None:
        environment = w14._source_control_environment()
        self.assertEqual(
            enoch_week1.canonical_json_sha256(environment),
            self.plan["source_control_contract"]["environment_sha256"],
        )
        self.assertEqual(environment["PATH"], "/usr/bin:/bin:/usr/sbin:/sbin")
        self.assertNotIn("GIT_DIR", environment)
        completed = mock.Mock(returncode=0, stdout="ok\n", stderr="")
        with mock.patch.object(w14, "_validate_trusted_git"), mock.patch.object(
            w14, "_validate_trusted_git_repository"
        ), mock.patch.object(w14.subprocess, "run", return_value=completed) as run:
            self.assertEqual(w14._git_text(Path.cwd(), "status"), "ok\n")
        command = run.call_args.args[0]
        self.assertEqual(command[0], "/usr/bin/git")
        self.assertEqual(
            command[1 : 1 + len(w14._source_control_global_options())],
            w14._source_control_global_options(),
        )
        self.assertEqual(run.call_args.kwargs["env"], environment)

    def test_pinned_regression_environment_drops_ambient_tool_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "run"
            run_root.mkdir()
            attempt = w14._require_safe_w1_4_directory(
                run_root / "w1.4/regression/attempts/attempt-001", create=True
            )
            environment = w14._regression_environment(
                {},
                {
                    "PATH": "/evil",
                    "RUSTC": "/evil/rustc",
                    "RUSTC_WRAPPER": "/evil/wrapper",
                    "RUSTFLAGS": "--cfg=fake",
                },
                attempt,
            )
            commands = w14._expected_regression_commands(
                w14.W14Layout(run_root), Path.cwd(), attempt
            )
        self.assertEqual(environment, w14._pinned_regression_environment(attempt))
        self.assertEqual(
            w14.shutil.which("cargo", path=environment["PATH"]),
            str(w14.TRUSTED_CARGO),
        )
        self.assertNotIn("RUSTC_WRAPPER", environment)
        self.assertNotIn("RUSTFLAGS", environment)
        self.assertEqual(commands["full-fixtures"][1:5], ["-I", "-B", "-c", w14.RECOVERY_RUNPY_BOOTSTRAP])
        for command_id in ("full-mechanics", "model-contracts", "strict-evaluator"):
            self.assertEqual(commands[command_id][0], str(w14.TRUSTED_CARGO))

    def test_cargo_config_dangling_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "workspace"
            (workspace / ".cargo").mkdir(parents=True)
            (workspace / ".cargo/config").symlink_to(workspace / "missing")
            with self.assertRaisesRegex(w14.W14OperatorError, "not allowed"):
                w14._cargo_config_records(workspace)

    def test_plan_rejects_order_rule_pair_seed_and_work_drift(self) -> None:
        mutations = []
        reordered = copy.deepcopy(self.plan)
        reordered["candidate_arm_ids"].reverse()
        mutations.append(reordered)
        rule = copy.deepcopy(self.plan)
        rule["development_rule"]["minimum_level_utility_estimate"] = -1.0
        mutations.append(rule)
        pairs = copy.deepcopy(self.plan)
        pairs["stages"][0]["pair_count"] = 299
        mutations.append(pairs)
        seed = copy.deepcopy(self.plan)
        seed["stages"][1]["seed_set_sha256"] = _digest("wrong")
        mutations.append(seed)
        work = copy.deepcopy(self.plan)
        work["fixed_work"]["worlds"] = 25
        mutations.append(work)
        stopped = copy.deepcopy(self.plan)
        stopped["candidate_arm_ids"].insert(3, w14.STOPPED_ARM_ID)
        mutations.append(stopped)
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                with self.assertRaises((w14.W14OperatorError, TypeError)):
                    w14.validate_committed_plan(mutation)

    def test_canonical_launch_and_identity_hashes_are_exact(self) -> None:
        launch = w14._canonical_launch()
        self.assertEqual(launch["candidate_arm_ids"], list(w14.CANDIDATE_ARM_IDS))
        self.assertEqual(launch["scenario_id"], "standard")
        self.assertEqual(launch["work_mode"], "fixed-work")
        self.assertEqual(
            enoch_week1.canonical_json_sha256(launch),
            self.plan["launch_configuration_sha256"],
        )

    def test_provenance_contract_is_exactly_three_additive_files(self) -> None:
        self.assertEqual(
            w14.CONTINUATION_PATHS,
            (
                "training/enoch_week1_w1_4_operator.py",
                "training/enoch_week1_w1_4_plan.json",
                "training/test_enoch_week1_w1_4_operator.py",
            ),
        )
        for frozen in (
            "training/enoch_week1_campaign.py",
            "training/enoch_week1_w1_3_operator.py",
            "training/enoch_week1_w1_3_seal_recovery.py",
        ):
            self.assertIn(frozen, w14.CRITICAL_BASE_MODULES)

    def test_stored_provenance_rejects_extra_field_before_git_use(self) -> None:
        artifact = {"continuation_provenance_fingerprint": _digest("x"), "extra": 1}
        with self.assertRaisesRegex(w14.W14OperatorError, "keys differ"):
            w14.validate_continuation_provenance(
                artifact, Path.cwd(), self.plan, live_source=False
            )


class RecoveryAndLockTests(unittest.TestCase):
    def test_only_recovery_aware_parent_verifier_is_accepted(self) -> None:
        completed = mock.Mock(
            returncode=0,
            stderr=b"",
            stdout=json.dumps(w14.EXPECTED_PARENT_RESULT, sort_keys=True).encode("utf-8"),
        )
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            w14, "_validate_trusted_git"
        ), mock.patch.object(
            w14, "_validate_trusted_git_repository"
        ), mock.patch.object(
            w14.subprocess, "run", return_value=completed
        ) as run, mock.patch.object(
            w14.w13, "verify_w1_3"
        ) as ordinary:
            layout = w14.W14Layout(Path(temporary))
            result = w14.verify_recovered_w1_3(layout, Path.cwd(), Path("sealed-d8"))
        self.assertEqual(result, w14.EXPECTED_PARENT_RESULT)
        self.assertIn("enoch_week1_w1_3_seal_recovery.py", " ".join(run.call_args.args[0]))
        ordinary.assert_not_called()

    def test_nonexact_recovery_result_is_rejected(self) -> None:
        wrong = copy.deepcopy(w14.EXPECTED_PARENT_RESULT)
        wrong["phase_manifest_fingerprint"] = _digest("ordinary-phase")
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            w14, "_validate_trusted_git"
        ), mock.patch.object(
            w14, "_validate_trusted_git_repository"
        ), mock.patch.object(
            w14.subprocess,
            "run",
            return_value=mock.Mock(
                returncode=0,
                stderr=b"",
                stdout=json.dumps(wrong, sort_keys=True).encode("utf-8"),
            ),
        ):
            with self.assertRaisesRegex(w14.W14OperatorError, "result changed"):
                w14.verify_recovered_w1_3(
                    w14.W14Layout(Path(temporary)), Path.cwd(), Path("sealed-d8")
                )

    def test_declare_verifies_parent_before_taking_operator_lock(self) -> None:
        events = []

        @contextlib.contextmanager
        def lock(_path):
            events.append("lock")
            yield

        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            w14, "verify_recovered_w1_3", side_effect=lambda *_a: events.append("verify")
        ), mock.patch.object(
            w14.base_operator, "_operator_lock", side_effect=lock
        ), mock.patch.object(
            w14, "load_committed_plan", side_effect=w14.W14OperatorError("stop")
        ):
            with self.assertRaisesRegex(w14.W14OperatorError, "stop"):
                w14.declare_w1_4(
                    w14.W14Layout(Path(temporary)), Path.cwd(), Path("sealed-d8")
                )
        self.assertEqual(events, ["verify", "lock"])


class ClaimsAndRetirementTests(unittest.TestCase):
    def _stage(self, stage_id: str = "qualification") -> dict:
        return {
            "comparison": {
                "comparison_protocol_fingerprint": _digest(stage_id),
                "seed_namespace": f"dev/combination/{stage_id}",
                "shards": [{"shard_id": "shard-000", "seed_indices": [0]}],
            },
            "sequence": 1 if stage_id == "qualification" else 2,
            "stage_id": stage_id,
        }

    def test_expected_consumers_are_stage_specific(self) -> None:
        qualification = self._stage()["comparison"]
        screen = self._stage("screen")["comparison"]
        self.assertNotEqual(
            w14._expected_claim_consumers(qualification)[0],
            w14._expected_claim_consumers(screen)[0],
        )

    def test_claimed_namespace_without_completion_retires_not_reruns(self) -> None:
        stage = self._stage()
        ledger = {
            "consumed": [
                {
                    "consumer": "runner:x:shard-000",
                    "index": 0,
                    "namespace": "dev/combination/qualification",
                    "seed": 1,
                    "sequence": 0,
                }
            ]
        }
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            w14, "_completed_attempt", return_value=None
        ), mock.patch.object(
            w14, "_stage_attempts", return_value=[]
        ), mock.patch.object(
            w14, "_retire", side_effect=w14.W14OperatorError("retired")
        ) as retire:
            with self.assertRaisesRegex(w14.W14OperatorError, "retired"):
                w14._retire_if_claimed_incomplete(
                    w14.W14Layout(Path(temporary)), {}, ledger, stage
                )
        self.assertEqual(retire.call_count, 1)
        self.assertIn("lacks-one-valid-completion", retire.call_args.kwargs["reason"])

    def test_out_of_order_screen_completion_retires(self) -> None:
        declaration = {
            "stages": [self._stage(), self._stage("screen")]
        }
        screen_claim = {
            "consumer": w14._expected_claim_consumers(
                declaration["stages"][1]["comparison"]
            )[0],
            "index": 0,
            "namespace": "dev/combination/screen",
            "seed": 1,
            "sequence": 0,
        }
        ledger = {"consumed": [screen_claim]}
        state = {"protocol": {}, "parent_ledger": {}}
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            w14, "_load_json", return_value=ledger
        ), mock.patch.object(
            w14.enoch_week1, "validate_seed_ledger"
        ), mock.patch.object(
            w14, "_completed_attempt", side_effect=[None, Path(temporary) / "screen"]
        ), mock.patch.object(
            w14, "_validate_claims"
        ), mock.patch.object(
            w14, "_validate_claim_frontier"
        ), mock.patch.object(
            w14, "_retire", side_effect=w14.W14OperatorError("retired")
        ) as retire:
            with self.assertRaisesRegex(w14.W14OperatorError, "retired"):
                w14._scan_resume_frontier(
                    w14.W14Layout(Path(temporary)), state, declaration
                )
        self.assertEqual(retire.call_args.kwargs["reason"], "claimed-w1.4-completion-is-out-of-order")

    def test_claim_frontier_stage_order_is_qualification_then_screen(self) -> None:
        declaration = {"stages": [self._stage(), self._stage("screen")]}
        ledger = {"consumed": []}
        with mock.patch.object(w14, "_validate_parent_prefix"), mock.patch.object(
            w14, "_expected_claim_consumers", return_value={}
        ):
            w14._validate_claim_frontier(ledger, {}, declaration, ["qualification"])
            w14._validate_claim_frontier(
                ledger, {}, declaration, ["qualification", "screen"]
            )

    def test_cached_claimed_completion_preflight_domain_error_retires(self) -> None:
        stage = self._stage()
        ledger = {"consumed": [], "ledger_fingerprint": _digest("ledger")}
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            w14,
            "_validate_completed_stage",
            side_effect=w14.enoch_week1_preflight.PreflightError("tampered preflight"),
        ), mock.patch.object(
            w14, "_load_json", return_value=ledger
        ), mock.patch.object(
            w14, "_retire", side_effect=w14.W14OperatorError("retired")
        ) as retire:
            with self.assertRaisesRegex(w14.W14OperatorError, "retired"):
                w14._validate_or_retire_completion(
                    w14.W14Layout(Path(temporary)),
                    {"protocol": {}},
                    {},
                    {},
                    stage,
                    Path(temporary) / "attempt-001",
                    base_environment={},
                )
        self.assertEqual(
            retire.call_args.kwargs["reason"],
            "claimed-w1.4-completion-marker-is-invalid",
        )

    def test_claimed_missing_completion_evidence_retires(self) -> None:
        stage = self._stage()
        ledger = {"consumed": [], "ledger_fingerprint": _digest("ledger")}
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            w14,
            "_validate_completed_stage",
            side_effect=FileNotFoundError("missing claimed evidence"),
        ), mock.patch.object(
            w14, "_load_json", return_value=ledger
        ), mock.patch.object(
            w14, "_retire", side_effect=w14.W14OperatorError("retired")
        ) as retire:
            with self.assertRaisesRegex(w14.W14OperatorError, "retired"):
                w14._validate_or_retire_completion(
                    w14.W14Layout(Path(temporary)),
                    {"protocol": {}},
                    {},
                    {},
                    stage,
                    Path(temporary) / "attempt-001",
                    base_environment={},
                )
        self.assertEqual(
            retire.call_args.kwargs["reason"],
            "claimed-w1.4-completion-marker-is-invalid",
        )

    def test_claimed_malformed_attempt_entry_still_writes_retirement(self) -> None:
        stage = self._stage()
        ledger = {
            "ledger_fingerprint": _digest("ledger"),
            "consumed": [
                {
                    "consumer": "runner:x:shard-000",
                    "index": 0,
                    "namespace": "dev/combination/qualification",
                    "seed": 1,
                    "sequence": 0,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            layout = w14.W14Layout(Path(temporary))
            attempts = layout.stage(1, "qualification") / "attempts"
            attempts.mkdir(parents=True)
            (attempts / "malformed-entry").write_text("not a directory", encoding="utf-8")
            with self.assertRaisesRegex(w14.W14OperatorError, "retired"):
                w14._retire_if_claimed_incomplete(
                    layout,
                    {"protocol_fingerprint": _digest("protocol")},
                    ledger,
                    stage,
                )
            retirement = json.loads(layout.retirement.read_text(encoding="utf-8"))
        self.assertEqual(retirement["reason"], "malformed-or-multiple-claimed-w1.4-attempt-entries")
        self.assertEqual(retirement["attempt_directories"], ["malformed-entry"])

    def test_claimed_symlinked_attempt_root_never_writes_external_tombstone(self) -> None:
        stage = self._stage()
        ledger = {
            "ledger_fingerprint": _digest("ledger"),
            "consumed": [
                {
                    "consumer": "runner:x:shard-000",
                    "index": 0,
                    "namespace": "dev/combination/qualification",
                    "seed": 1,
                    "sequence": 0,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            layout = w14.W14Layout(root / "run")
            outside_attempt = root / "outside" / "attempt-999"
            outside_attempt.mkdir(parents=True)
            attempts = layout.stage(1, "qualification") / "attempts"
            attempts.parent.mkdir(parents=True)
            attempts.symlink_to(outside_attempt.parent, target_is_directory=True)
            with self.assertRaisesRegex(w14.W14OperatorError, "retired"):
                w14._retire_if_claimed_incomplete(
                    layout,
                    {"protocol_fingerprint": _digest("protocol")},
                    ledger,
                    stage,
                )
            self.assertTrue(layout.retirement.is_file())
            self.assertFalse((outside_attempt / "failure-tombstone.json").exists())

    def test_symlinked_w1_4_root_never_writes_external_retirement(self) -> None:
        stage = self._stage()
        ledger = {"ledger_fingerprint": _digest("ledger"), "consumed": []}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = root / "run"
            outside = root / "outside"
            run_root.mkdir()
            outside.mkdir()
            (run_root / "w1.4").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(w14.W14OperatorError, "symlink"):
                w14._retire(
                    w14.W14Layout(run_root),
                    {"protocol_fingerprint": _digest("protocol")},
                    ledger,
                    stage,
                    attempt=None,
                    reason="test-retirement",
                )
            self.assertFalse((outside / "protocol-retired.json").exists())


class RegressionAndDecisionTests(unittest.TestCase):
    def test_regression_reconstructs_but_never_builds_seedful_preflight(self) -> None:
        source = inspect.getsource(w14._reconstruct_sealed_preflight)
        self.assertIn("validate_preflight_artifact", source)
        self.assertIn("require_full_coverage=True", source)
        self.assertIn("validate_w1_1_baseline_worker_report", source)
        self.assertNotIn("build_preflight_artifact", source)
        self.assertNotIn("run_w1_1", source)

    def test_nonzero_build_failure_is_operational_not_terminal(self) -> None:
        completed = mock.Mock(
            returncode=101,
            stdout=b"error: linking with cc failed: No space left on device\n",
        )
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            w14.subprocess, "run", return_value=completed
        ):
            run_root = Path(temporary) / "run"
            run_root.mkdir()
            attempt = w14._require_safe_w1_4_directory(
                run_root / "w1.4" / "regression" / "attempts" / "attempt-001",
                create=True,
            )
            with self.assertRaises(w14.W14OperatorError) as raised:
                w14._run_regression_command(
                    "full-mechanics",
                    ["cargo", "test"],
                    cwd=run_root,
                    environment={"PATH": "/bin"},
                    attempt=attempt,
                    control_manifest_fingerprint=_digest("control"),
                    rust_toolchain_contract_fingerprint=_digest("toolchain"),
                )
        self.assertNotIsInstance(raised.exception, w14.RegressionTestFailure)

    def test_only_machine_recognizable_failed_test_is_terminal(self) -> None:
        completed = mock.Mock(
            returncode=101,
            stdout=(
                b"test result: FAILED. 76 passed; 1 failed; 0 ignored; "
                b"0 measured; 0 filtered out\n"
            ),
        )
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            w14.subprocess, "run", return_value=completed
        ):
            run_root = Path(temporary) / "run"
            run_root.mkdir()
            attempt = w14._require_safe_w1_4_directory(
                run_root / "w1.4" / "regression" / "attempts" / "attempt-001",
                create=True,
            )
            with self.assertRaises(w14.RegressionTestFailure) as raised:
                w14._run_regression_command(
                    "full-mechanics",
                    ["cargo", "test"],
                    cwd=run_root,
                    environment={"PATH": "/bin"},
                    attempt=attempt,
                    control_manifest_fingerprint=_digest("control"),
                    rust_toolchain_contract_fingerprint=_digest("toolchain"),
                )
        self.assertEqual(
            raised.exception.record["semantic_failure_marker"],
            "cargo-test-failed-verdict",
        )
        self.assertEqual(
            raised.exception.record["rust_toolchain_contract_fingerprint"],
            _digest("toolchain"),
        )
        self.assertEqual(
            raised.exception.record["control_manifest_fingerprint"],
            _digest("control"),
        )

    def test_regression_revalidates_source_and_toolchain_around_every_command(self) -> None:
        source = inspect.getsource(w14._ensure_regression_gate_locked)
        self.assertIn("source_before = require_live_source()", source)
        self.assertIn("source_after = require_live_source()", source)
        self.assertIn("control_after = require_control()", source)
        self.assertIn("after_failure = require_toolchain()", source)
        self.assertIn("source_after_failure = require_live_source()", source)
        self.assertIn("control_after_failure = require_control()", source)
        build = source.index("_build_regression_gate")
        final_check = source.index("require_live_source()", build)
        completion = source.index("regression-complete.json", build)
        self.assertLess(final_check, completion)

    def test_parent_context_scopes_pinned_git_adapters_with_finally_restore(self) -> None:
        source = inspect.getsource(w14._load_parent_state)
        self.assertIn("recovery._git_text = _git_text", source)
        self.assertIn("w13._git_bytes = _git_bytes", source)
        self.assertIn("finally:", source)
        self.assertIn("= original_helpers", source)

    def test_terminal_failure_verifier_requires_nonnull_semantic_marker(self) -> None:
        source = inspect.getsource(w14._validate_prerequisite_failure_branch)
        self.assertIn("semantic_marker is None", source)
        self.assertIn("not isinstance(exit_code, int)", source)
        self.assertIn("isinstance(exit_code, bool)", source)

    def test_regression_acquires_machine_global_lock_without_nesting_stage_lock(self) -> None:
        events = []

        @contextlib.contextmanager
        def campaign_lock(_protocol, comparison):
            events.append(("lock", comparison))
            yield "opaque-token"

        declaration = {
            "stages": [
                {
                    "stage_id": "qualification",
                    "comparison": {"comparison_protocol_fingerprint": _digest("q")},
                }
            ]
        }
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            w14.enoch_week1_runner,
            "authoritative_campaign_lock",
            side_effect=campaign_lock,
        ), mock.patch.object(
            w14,
            "_ensure_regression_gate_locked",
            return_value={"combination_regression_gate_fingerprint": _digest("gate")},
        ) as locked:
            result = w14._ensure_regression_gate(
                w14.W14Layout(Path(temporary)),
                Path.cwd(),
                {"protocol": {}},
                {},
                declaration,
                w14.EXPECTED_PARENT_RESULT,
                base_environment={},
            )
        self.assertEqual(result["combination_regression_gate_fingerprint"], _digest("gate"))
        self.assertEqual(events[0][0], "lock")
        locked.assert_called_once()

    def test_interrupted_top_level_gate_write_reuses_completed_attempt(self) -> None:
        gate = {"combination_regression_gate_fingerprint": _digest("gate")}
        ledger = {"consumed": [], "ledger_fingerprint": w14.PARENT_LEDGER}
        with tempfile.TemporaryDirectory() as temporary:
            layout = w14.W14Layout(Path(temporary))
            attempt = layout.regression_attempts / "attempt-001"
            attempt.mkdir(parents=True)
            with mock.patch.object(
                w14, "_load_json", side_effect=[ledger, gate]
            ), mock.patch.object(
                w14, "validate_regression_gate", return_value=_digest("gate")
            ), mock.patch.object(
                w14, "_write_or_match"
            ) as write, mock.patch.object(
                w14, "_next_attempt"
            ) as next_attempt:
                (attempt / "regression-complete.json").write_text("{}", encoding="utf-8")
                result = w14._ensure_regression_gate_locked(
                    layout,
                    Path.cwd(),
                    {"protocol": {}},
                    {},
                    {},
                    w14.EXPECTED_PARENT_RESULT,
                    base_environment={},
                )
        self.assertEqual(result, gate)
        next_attempt.assert_not_called()
        self.assertEqual(write.call_args.args[0], layout.regression)

    def test_failed_regression_is_terminal_only_for_revalidated_live_source(self) -> None:
        source = inspect.getsource(w14._ensure_regression_gate_locked)
        revalidation = source.index("validate_continuation_provenance")
        classification = source.index("isinstance(failure_exc, RegressionTestFailure)")
        self.assertLess(revalidation, classification)
        self.assertIn("live_source=True", source[revalidation:classification])

    def test_prerequisite_failure_verifier_reopens_the_live_ledger(self) -> None:
        source = inspect.getsource(w14._validate_prerequisite_failure_branch)
        self.assertIn("_load_json(layout.base.ledger)", source)
        self.assertIn("if live_ledger != ledger", source)

    def test_external_validator_counter_map_is_never_used_as_fingerprint(self) -> None:
        expected = _digest("external")
        evidence = {"verified_external_evidence_fingerprint": expected}
        with mock.patch.object(
            w14.enoch_week1_evidence,
            "validate_verified_external_evidence",
            return_value={"illegal_action": {"count": 0}},
        ):
            self.assertEqual(
                w14._validated_external_fingerprint({}, {}, evidence), expected
            )

    def test_missing_scalar_external_fingerprint_is_rejected(self) -> None:
        with mock.patch.object(
            w14.enoch_week1_evidence,
            "validate_verified_external_evidence",
            return_value={},
        ):
            with self.assertRaisesRegex(w14.W14OperatorError, "lowercase SHA-256"):
                w14._validated_external_fingerprint({}, {}, {})

    def _decision(self, value: str) -> dict:
        return {
            "candidate_fingerprint": _digest("candidate"),
            "decision": value,
            enoch_week1_campaign.CANDIDATE_DECISION_FINGERPRINT_FIELD: _digest(
                f"decision:{value}"
            ),
        }

    def _exit_inputs(self):
        declaration = {
            "campaign_declaration_fingerprint": _digest("declaration"),
            "campaign_lineage_fingerprint": _digest("lineage"),
        }
        regression = {"combination_regression_gate_fingerprint": _digest("gate")}
        ledger = {"ledger_fingerprint": _digest("ledger")}
        return declaration, regression, ledger

    def test_eligible_exit_selects_exact_candidate_and_binds_regression(self) -> None:
        declaration, regression, ledger = self._exit_inputs()
        decision = self._decision("eligible-for-qualification")
        with mock.patch.object(
            w14.enoch_week1_campaign,
            "validate_w1_4_candidate_decision",
            return_value=decision[
                enoch_week1_campaign.CANDIDATE_DECISION_FINGERPRINT_FIELD
            ],
        ):
            artifact = w14.build_exit_artifact(
                {}, declaration, regression, decision, ledger
            )
        self.assertEqual(artifact["status"], "single-candidate")
        self.assertEqual(
            artifact["selected_candidate_fingerprint"], decision["candidate_fingerprint"]
        )
        self.assertTrue(artifact["w1_5_allowed"])
        self.assertEqual(
            artifact["combination_regression_gate_fingerprint"],
            regression["combination_regression_gate_fingerprint"],
        )

    def test_rejected_exit_selects_nothing_and_skips_w1_5(self) -> None:
        declaration, regression, ledger = self._exit_inputs()
        decision = self._decision("reject-candidate")
        with mock.patch.object(
            w14.enoch_week1_campaign,
            "validate_w1_4_candidate_decision",
            return_value=decision[
                enoch_week1_campaign.CANDIDATE_DECISION_FINGERPRINT_FIELD
            ],
        ):
            artifact = w14.build_exit_artifact(
                {}, declaration, regression, decision, ledger
            )
        self.assertEqual(artifact["status"], "combination-regressed")
        self.assertIsNone(artifact["selected_candidate_fingerprint"])
        self.assertEqual(artifact["no_candidate_reason"], "combination-regressed")
        self.assertFalse(artifact["w1_5_allowed"])
        self.assertIn("not-bare-candidate-decision", artifact["w1_5_dependency"])

    def test_terminal_prerequisite_failure_has_typed_no_candidate_exit(self) -> None:
        declaration, _regression, ledger = self._exit_inputs()
        declaration["campaign_lineage"] = {"candidate_fingerprint": _digest("candidate")}
        failure = {"preclaim_failure_fingerprint": _digest("failure")}
        artifact = w14._build_prerequisite_failure_exit(
            declaration, failure, ledger
        )
        self.assertEqual(artifact["status"], "candidate-prerequisites-incomplete")
        self.assertEqual(artifact["final_consumed_count"], 18_711)
        self.assertIsNone(artifact["candidate_decision_fingerprint"])
        self.assertIsNone(artifact["selected_candidate_fingerprint"])
        self.assertFalse(artifact["w1_5_allowed"])

    def test_phase_artifacts_bind_regression_decision_exit_and_both_results(self) -> None:
        declaration = {
            "campaign_declaration_fingerprint": _digest("declaration"),
            "campaign_lineage_fingerprint": _digest("lineage"),
            "stages": [
                {
                    "stage_id": stage_id,
                    "comparison": {"comparison_protocol_fingerprint": _digest(stage_id)},
                }
                for stage_id in w14.STAGE_IDS
            ],
        }
        stage_evidence = {
            stage_id: {
                "external_evidence_fingerprint": _digest(f"external:{stage_id}"),
                "merged_result": {"merged_result_fingerprint": _digest(f"merge:{stage_id}")},
            }
            for stage_id in w14.STAGE_IDS
        }
        artifacts = w14._phase_artifacts(
            {"continuation_provenance_fingerprint": _digest("provenance")},
            declaration,
            {"combination_regression_gate_fingerprint": _digest("gate")},
            {
                enoch_week1_campaign.CANDIDATE_DECISION_FINGERPRINT_FIELD: _digest(
                    "decision"
                )
            },
            {"w1_4_exit_fingerprint": _digest("exit")},
            {"ledger_fingerprint": _digest("ledger")},
            stage_evidence,
        )
        self.assertEqual(artifacts["w1.4-combination-regression-gate"], _digest("gate"))
        self.assertIn("w1.4/qualification/merged-result", artifacts)
        self.assertIn("w1.4/screen/merged-result", artifacts)
        self.assertEqual(artifacts["single-candidate-or-no-survivor"], _digest("exit"))

    def test_phase_is_the_last_successful_seal_write(self) -> None:
        source = inspect.getsource(w14._seal_w1_4)
        phase_write = source.index("_write_or_match(layout.phase")
        self.assertGreater(phase_write, source.index("_write_or_match(layout.decision"))
        self.assertGreater(phase_write, source.index("layout.exit, expected_exit"))

    def test_regression_failure_wrapper_preserves_existing_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            attempt = Path(temporary)
            (attempt / "logs").mkdir()
            (attempt / "logs" / "full-fixtures.log").write_text("failed", encoding="utf-8")
            w14._write_regression_failure(attempt, RuntimeError("fixture failed"))
            artifact = json.loads(
                (attempt / "preclaim-failure.json").read_text(encoding="utf-8")
            )
        self.assertEqual(artifact["seed_claim_count"], 0)
        self.assertEqual(artifact["preserved_logs"][0]["path"], "logs/full-fixtures.log")


class LedgerTests(unittest.TestCase):
    def test_final_ledger_requires_exact_count(self) -> None:
        declaration = {
            "stages": [
                {
                    "stage_id": stage_id,
                    "comparison": {
                        "seed_namespace": f"dev/combination/{stage_id}"
                    },
                }
                for stage_id in w14.STAGE_IDS
            ]
        }
        with mock.patch.object(w14.enoch_week1, "validate_seed_ledger"), mock.patch.object(
            w14, "_validate_parent_prefix"
        ):
            with self.assertRaisesRegex(w14.W14OperatorError, "19,811"):
                w14._expected_final_ledger(
                    {}, {"consumed": [], "ledger_fingerprint": _digest("ledger")}, {}, declaration
                )

    def test_downstream_extension_allows_only_qual_and_locked(self) -> None:
        snapshot = {
            "consumed": [{"namespace": "dev/combination/screen"}],
            "ledger_fingerprint": _digest("snapshot"),
        }
        allowed = {"consumed": snapshot["consumed"] + [{"namespace": "qual/intended"}]}
        forbidden = {"consumed": snapshot["consumed"] + [{"namespace": "dev/ablation/x"}]}
        with mock.patch.object(w14.enoch_week1, "validate_seed_ledger"), mock.patch.object(
            w14.enoch_week1, "canonical_json_sha256", return_value=snapshot["ledger_fingerprint"]
        ):
            w14._validate_ledger_extension({}, allowed, snapshot)
            with self.assertRaisesRegex(w14.W14OperatorError, "non-W1.5"):
                w14._validate_ledger_extension({}, forbidden, snapshot)

    def test_no_candidate_exit_forbids_every_live_ledger_extension(self) -> None:
        snapshot = {"consumed": [], "ledger_fingerprint": _digest("snapshot")}
        extension = {
            "consumed": [{"namespace": "qual/intended"}],
            "ledger_fingerprint": _digest("extension"),
        }
        with mock.patch.object(w14.enoch_week1, "validate_seed_ledger"):
            with self.assertRaisesRegex(w14.W14OperatorError, "no-candidate"):
                w14._validate_post_w1_4_live_ledger(
                    {},
                    extension,
                    snapshot,
                    {"status": "combination-regressed", "w1_5_allowed": False},
                )

    def test_only_single_candidate_exit_may_validate_downstream_extension(self) -> None:
        with mock.patch.object(w14, "_validate_ledger_extension") as validate:
            w14._validate_post_w1_4_live_ledger(
                {},
                {"ledger_fingerprint": _digest("current")},
                {"ledger_fingerprint": _digest("snapshot")},
                {"status": "single-candidate", "w1_5_allowed": True},
            )
        validate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
