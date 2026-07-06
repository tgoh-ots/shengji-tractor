#!/usr/bin/env python3
"""Adversarial tests for the metadata-only authoritative W1.8 seal."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from training import enoch_week1
from training import enoch_week1_w1_8_operator as w18


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _terminal_stub() -> dict:
    terminal = {
        "candidate_fingerprint": w18.EVALUATED_CANDIDATE_FINGERPRINT,
        "candidate_status": "not-confirmed",
        "decision": "no-confirmed-candidate",
        "downstream_primary_fingerprint": w18.PERMANENT_ENOCH0_FINGERPRINT,
        "evidence_fingerprints": [_digest(f"evidence-{index}") for index in range(118)],
        "human_operator_review_required": True,
        "no_candidate_reason": "combination-regressed",
        "permanent_scientific_control_fingerprint": w18.PERMANENT_ENOCH0_FINGERPRINT,
        "phase_chain_sha256": w18.PHASE_CHAIN_SHA256,
        "production_promotion_authorized": False,
        "stage2_rebaseline_authorized_after_human_review": True,
        "week1_decision_fingerprint": w18.TERMINAL_DECISION_FINGERPRINT,
    }
    for field in (
        "qualification_decision",
        "qualification_manifest",
        "qualification_merged_results",
        "primary_gate_decision",
        "primary_gate_manifest",
        "primary_merged_result",
        "confirmation_gate_decision",
        "confirmation_gate_manifest",
        "confirmation_merged_result",
        "evaluation_control_fingerprint",
        "evaluation_control_relation",
    ):
        terminal[field] = None
    return terminal


def _state() -> dict:
    return {
        "control": {"control": True},
        "decision": {"candidate_fingerprint": w18.EVALUATED_CANDIDATE_FINGERPRINT},
        "enoch0_fingerprint": w18.PERMANENT_ENOCH0_FINGERPRINT,
        "exit": {"selected_candidate_fingerprint": None},
        "ledger": {"ledger_fingerprint": w18.FINAL_LEDGER_FINGERPRINT},
        "phases": [{"phase": "W1.4"}],
        "protocol": {"protocol": True},
    }


class PlanAndSourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan_path = Path(__file__).with_name("enoch_week1_w1_8_plan.json")
        cls.plan = json.loads(cls.plan_path.read_text(encoding="utf-8"))

    def test_exact_committed_plan(self) -> None:
        fingerprint = w18.validate_committed_plan(self.plan)
        self.assertEqual(fingerprint, enoch_week1.canonical_json_sha256(self.plan))
        self.assertEqual(
            self.plan["terminal_decision_fingerprint"],
            "799be65915b7c60a280e03cbcd038e94bfc3af7340763c86dfe2b93070e5bd84",
        )
        self.assertEqual(self.plan["candidate_fingerprint"], w18.EVALUATED_CANDIDATE_FINGERPRINT)
        self.assertIsNone(self.plan["selected_candidate_fingerprint"])
        self.assertEqual(
            self.plan["permanent_scientific_control_fingerprint"],
            w18.PERMANENT_ENOCH0_FINGERPRINT,
        )
        self.assertFalse(self.plan["parent_w1_5_allowed"])
        self.assertEqual(self.plan["skipped_phases"], ["W1.5", "W1.6", "W1.7"])

    def test_plan_rejects_every_terminal_choice_mutation(self) -> None:
        mutations = {
            "decision": "freeze-enoch-1",
            "candidate_status": "confirmed",
            "candidate_fingerprint": None,
            "no_candidate_reason": "no-survivor",
            "permanent_scientific_control_fingerprint": _digest("other-control"),
            "production_promotion_authorized": True,
            "parent_w1_5_allowed": True,
            "skipped_phases": ["W1.5"],
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                changed = copy.deepcopy(self.plan)
                changed[field] = value
                with self.assertRaisesRegex(w18.W18OperatorError, "changed fields"):
                    w18.validate_committed_plan(changed)

    def test_source_bundle_is_exactly_three_additive_files(self) -> None:
        self.assertEqual(
            w18.CONTINUATION_PATHS,
            (
                "training/enoch_week1_w1_8_operator.py",
                "training/enoch_week1_w1_8_plan.json",
                "training/test_enoch_week1_w1_8_operator.py",
            ),
        )
        self.assertIn("training/enoch_week1.py", w18.CRITICAL_BASE_MODULES)
        self.assertIn(
            "training/enoch_week1_w1_4_operator.py", w18.CRITICAL_BASE_MODULES
        )

    def test_operator_has_no_seed_consumption_or_evaluator_surface(self) -> None:
        source = inspect.getsource(w18.seal_w1_8)
        self.assertNotIn("consume_seed", source)
        self.assertNotIn("enoch_week1_runner", source)
        self.assertNotIn("cargo", source.lower())

    def test_stored_provenance_requires_the_executing_operator_bytes(self) -> None:
        workspace = Path(__file__).resolve().parents[1]
        operator_path = workspace / w18.OPERATOR_RELATIVE
        actual = w18._sha256_regular(operator_path, "operator")  # noqa: SLF001
        w18._require_executing_operator(workspace, actual)  # noqa: SLF001
        with self.assertRaisesRegex(w18.W18OperatorError, "differs from stored"):
            w18._require_executing_operator(workspace, _digest("other"))  # noqa: SLF001


class ParentAuthorityTests(unittest.TestCase):
    def test_only_exact_authoritative_w14_result_is_accepted(self) -> None:
        with mock.patch.object(
            w18.w14, "verify_w1_4", return_value=copy.deepcopy(w18.EXPECTED_W14_RESULT)
        ) as verifier:
            result = w18._verify_authoritative_w1_4(  # noqa: SLF001
                w18.W18Layout(Path("/tmp/run")), Path("/tmp/work"), Path("/tmp/w12")
            )
        self.assertEqual(result, w18.EXPECTED_W14_RESULT)
        verifier.assert_called_once()

    def test_self_consistent_but_different_w14_result_is_rejected(self) -> None:
        changed = copy.deepcopy(w18.EXPECTED_W14_RESULT)
        changed["phase_manifest_fingerprint"] = _digest("fake-phase")
        with mock.patch.object(w18.w14, "verify_w1_4", return_value=changed):
            with self.assertRaisesRegex(w18.W18OperatorError, "verifier result changed"):
                w18._verify_authoritative_w1_4(  # noqa: SLF001
                    w18.W18Layout(Path("/tmp/run")),
                    Path("/tmp/work"),
                    Path("/tmp/w12"),
                )

    def test_all_skipped_phase_paths_are_forbidden_including_symlinks(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            layout = w18.W18Layout(root)
            for phase in w18.SKIPPED_PHASES:
                path = root / phase.lower()
                path.symlink_to(root / "missing")
                with self.subTest(phase=phase), self.assertRaisesRegex(
                    w18.W18OperatorError, "must not be materialized"
                ):
                    w18._assert_skipped_phases_absent(layout)  # noqa: SLF001
                path.unlink()


class TerminalAndPhaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = json.loads(
            Path(__file__).with_name("enoch_week1_w1_8_plan.json").read_text(
                encoding="utf-8"
            )
        )

    def test_terminal_preserves_rejected_candidate_but_retains_enoch0(self) -> None:
        terminal = _terminal_stub()
        state = _state()
        with mock.patch.object(
            w18.enoch_week1,
            "build_week1_decision_artifact",
            return_value=terminal,
        ) as build, mock.patch.object(
            w18.enoch_week1, "validate_week1_decision_artifact"
        ):
            result = w18._build_terminal(state, self.plan)  # noqa: SLF001
        self.assertEqual(result, terminal)
        kwargs = build.call_args.kwargs
        self.assertEqual(
            kwargs["candidate_fingerprint"], w18.EVALUATED_CANDIDATE_FINGERPRINT
        )
        self.assertEqual(kwargs["no_candidate_reason"], "combination-regressed")
        self.assertIsNone(kwargs["primary_gate_decision"])
        self.assertIsNone(kwargs["confirmation_gate_decision"])
        self.assertEqual(
            result["downstream_primary_fingerprint"], w18.PERMANENT_ENOCH0_FINGERPRINT
        )

    def test_terminal_rejects_freeze_or_later_phase_evidence(self) -> None:
        for field, value in (
            ("decision", "freeze-enoch-1"),
            ("candidate_status", "confirmed"),
            ("downstream_primary_fingerprint", w18.EVALUATED_CANDIDATE_FINGERPRINT),
            ("qualification_decision", {"decision": "eligible-for-locked-gate"}),
            ("production_promotion_authorized", True),
        ):
            with self.subTest(field=field):
                terminal = _terminal_stub()
                terminal[field] = value
                with mock.patch.object(
                    w18.enoch_week1,
                    "build_week1_decision_artifact",
                    return_value=terminal,
                ), mock.patch.object(
                    w18.enoch_week1, "validate_week1_decision_artifact"
                ):
                    with self.assertRaises(w18.W18OperatorError):
                        w18._build_terminal(_state(), self.plan)  # noqa: SLF001

    def test_human_review_is_explicit_and_never_promotes(self) -> None:
        provenance = {"continuation_provenance_fingerprint": _digest("provenance")}
        terminal = _terminal_stub()
        review = w18._build_review(  # noqa: SLF001
            _state(), self.plan, provenance, terminal, "codex-w1.8-test"
        )
        self.assertTrue(review["human_operator_review_completed"])
        self.assertEqual(review["human_review_decision"], "retain-enoch-0")
        self.assertEqual(review["human_review_source"], "interactive-user-instruction")
        self.assertFalse(review["production_promotion_authorized"])
        self.assertFalse(review["automatic_production_promotion_allowed"])
        self.assertTrue(review["stage2_rebaseline_authorized"])
        self.assertEqual(review["skipped_phases"], list(w18.SKIPPED_PHASES))
        self.assertEqual(
            w18._validate_review(  # noqa: SLF001
                review, _state(), self.plan, provenance, terminal
            ),
            review["human_review_fingerprint"],
        )

    def test_phase_binds_exact_exit_parent_review_provenance_and_ledger(self) -> None:
        state = _state()
        provenance = {"continuation_provenance_fingerprint": _digest("provenance")}
        terminal = _terminal_stub()
        review = w18._build_review(  # noqa: SLF001
            state, self.plan, provenance, terminal, "codex-w1.8-test"
        )

        def build_phase(_protocol, phase, *, artifacts, declarations, parent_phase_manifests):
            self.assertEqual(phase, "W1.8")
            return {
                "artifacts": [
                    {"artifact_id": key, "sha256": value}
                    for key, value in sorted(artifacts.items())
                ],
                "automatic_production_promotion_allowed": False,
                "declarations": declarations,
                "declared_exit_artifact": "freeze-or-no-confirmed-candidate-decision",
                "parent_phases": [
                    {
                        "phase": "W1.4",
                        "phase_manifest_fingerprint": w18.PARENT_PHASE_FINGERPRINT,
                    }
                ],
                "phase_manifest_fingerprint": _digest("phase"),
            }

        with mock.patch.object(
            w18.enoch_week1, "build_phase_manifest", side_effect=build_phase
        ), mock.patch.object(w18.enoch_week1, "validate_phase_manifest"):
            phase = w18._build_phase(  # noqa: SLF001
                state, self.plan, provenance, terminal, review
            )
        artifacts = {item["artifact_id"]: item["sha256"] for item in phase["artifacts"]}
        self.assertEqual(
            artifacts["freeze-or-no-confirmed-candidate-decision"],
            w18.TERMINAL_DECISION_FINGERPRINT,
        )
        self.assertEqual(artifacts["w1.4-phase-manifest"], w18.PARENT_PHASE_FINGERPRINT)
        self.assertEqual(
            artifacts["w1.4-single-candidate-or-no-survivor"],
            w18.PARENT_EXIT_FINGERPRINT,
        )
        self.assertEqual(artifacts["w1.8-final-ledger"], w18.FINAL_LEDGER_FINGERPRINT)
        self.assertEqual(
            phase["parent_phases"],
            [
                {
                    "phase": "W1.4",
                    "phase_manifest_fingerprint": w18.PARENT_PHASE_FINGERPRINT,
                }
            ],
        )


class FilesystemSafetyTests(unittest.TestCase):
    def test_parent_and_workspace_component_symlinks_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            external = root / "external"
            external.mkdir()
            (external / "artifact.json").write_text("{}", encoding="utf-8")
            (root / "w1.4").symlink_to(external, target_is_directory=True)
            with self.assertRaisesRegex(w18.W18OperatorError, "parent"):
                w18._require_safe_run_file(  # noqa: SLF001
                    w18.W18Layout(root), root / "w1.4/artifact.json", "artifact"
                )
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            workspace = Path(temporary)
            external = workspace / "external"
            external.mkdir()
            (external / "module.py").write_text("pass\n", encoding="utf-8")
            (workspace / "training").symlink_to(external, target_is_directory=True)
            with self.assertRaisesRegex(w18.W18OperatorError, "parent"):
                w18._require_workspace_file(  # noqa: SLF001
                    workspace, "training/module.py", "module"
                )

    def test_coordination_locks_reject_symlink_and_fifo_leaves(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            layout = w18.W18Layout(root)
            sentinel = root / "sentinel"
            sentinel.write_text("unchanged", encoding="utf-8")
            layout.base.operator_lock.symlink_to(sentinel)
            with self.assertRaisesRegex(w18.W18OperatorError, "regular"):
                with w18._secure_lock(  # noqa: SLF001
                    layout,
                    layout.base.operator_lock,
                    "operator",
                    nonblocking=True,
                ):
                    self.fail("unsafe lock acquired")
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "unchanged")
            layout.base.operator_lock.unlink()
            if hasattr(os, "mkfifo"):
                ledger_lock = layout.base.ledger.with_name(
                    f"{layout.base.ledger.name}.lock"
                )
                os.mkfifo(ledger_lock)
                with self.assertRaisesRegex(w18.W18OperatorError, "regular"):
                    with w18._secure_lock(  # noqa: SLF001
                        layout, ledger_lock, "seed-ledger", nonblocking=False
                    ):
                        self.fail("unsafe ledger lock acquired")

    def test_coordination_locks_must_preexist(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            layout = w18.W18Layout(Path(temporary))
            with self.assertRaisesRegex(w18.W18OperatorError, "lock is missing"):
                with w18._secure_lock(  # noqa: SLF001
                    layout,
                    layout.base.operator_lock,
                    "operator",
                    nonblocking=True,
                ):
                    self.fail("missing lock was created")
            self.assertFalse(layout.base.operator_lock.exists())

    def test_directory_creation_is_fsynced(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            layout = w18.W18Layout(Path(temporary))
            with mock.patch.object(w18, "_fsync_directory") as fsync:
                w18._require_w18_tree(layout, create=True)  # noqa: SLF001
            fsync.assert_called_once_with(layout.root, "Week 1 run root")

    def test_symlinked_w18_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            external = root / "external"
            external.mkdir()
            (root / "w1.8").symlink_to(external, target_is_directory=True)
            with self.assertRaisesRegex(w18.W18OperatorError, "not a real directory"):
                w18._require_w18_tree(  # noqa: SLF001
                    w18.W18Layout(root), create=False
                )

    def test_symlinked_leaf_cannot_overwrite_external_sentinel(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            layout = w18.W18Layout(root)
            w18._require_w18_tree(layout, create=True)  # noqa: SLF001
            sentinel = root / "sentinel"
            sentinel.write_text("unchanged", encoding="utf-8")
            layout.input.symlink_to(sentinel)
            with self.assertRaisesRegex(w18.W18OperatorError, "symlink|regular"):
                w18._write_or_match(  # noqa: SLF001
                    layout, layout.input, {"changed": True}, "input"
                )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "unchanged")

    def test_hard_linked_leaf_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            layout = w18.W18Layout(root)
            w18._require_w18_tree(layout, create=True)  # noqa: SLF001
            sentinel = root / "sentinel"
            sentinel.write_text("unchanged", encoding="utf-8")
            os.link(sentinel, layout.input)
            with self.assertRaisesRegex(w18.W18OperatorError, "hard-linked"):
                w18._write_or_match(  # noqa: SLF001
                    layout, layout.input, {"changed": True}, "input"
                )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "unchanged")

    def test_undeclared_or_special_w18_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            layout = w18.W18Layout(root)
            w18._require_w18_tree(layout, create=True)  # noqa: SLF001
            (layout.directory / "undeclared.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(w18.W18OperatorError, "undeclared"):
                w18._require_w18_tree(layout, create=False)  # noqa: SLF001
        if hasattr(os, "mkfifo"):
            with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
                root = Path(temporary)
                layout = w18.W18Layout(root)
                w18._require_w18_tree(layout, create=True)  # noqa: SLF001
                os.mkfifo(layout.directory / "input.json")
                with self.assertRaisesRegex(w18.W18OperatorError, "non-file"):
                    w18._require_w18_tree(layout, create=False)  # noqa: SLF001

    def test_noncanonical_or_mismatched_partial_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            layout = w18.W18Layout(Path(temporary))
            w18._require_w18_tree(layout, create=True)  # noqa: SLF001
            layout.input.write_text('{\n  "value": 1\n}\n', encoding="utf-8")
            with self.assertRaisesRegex(w18.W18OperatorError, "canonical"):
                w18._load_w18_json(layout, layout.input, "input")  # noqa: SLF001
            with self.assertRaisesRegex(w18.W18OperatorError, "canonical"):
                w18._write_or_match(  # noqa: SLF001
                    layout, layout.input, {"value": 1}, "input"
                )
            layout.input.write_bytes(
                enoch_week1.canonical_json_bytes({"value": 1}) + b"\n"
            )
            with self.assertRaisesRegex(w18.W18OperatorError, "does not reconstruct"):
                w18._write_or_match(  # noqa: SLF001
                    layout, layout.input, {"value": 2}, "input"
                )

    def test_full_inventory_detects_mutated_nested_shard(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            for index in range(5):
                (root / f"w1.{index}").mkdir()
            (root / "protocol.json").write_text("{}\n", encoding="utf-8")
            (root / "seed-ledger.json").write_text("{}\n", encoding="utf-8")
            shard = root / "w1.4/shard-000.raw.json"
            shard.write_text('{"value":1}\n', encoding="utf-8")
            before = w18._full_parent_inventory(w18.W18Layout(root))  # noqa: SLF001
            shard.write_text('{"value":2}\n', encoding="utf-8")
            after = w18._full_parent_inventory(w18.W18Layout(root))  # noqa: SLF001
            self.assertNotEqual(before, after)


class SealFlowTests(unittest.TestCase):
    @staticmethod
    @contextlib.contextmanager
    def _lock(*_args, **_kwargs):
        yield

    def _patches(self, state, plan, writes):
        provenance = {"continuation_provenance_fingerprint": _digest("provenance")}
        terminal = _terminal_stub()
        review = {"human_review_fingerprint": _digest("review")}
        phase = {"phase_manifest_fingerprint": _digest("phase")}
        summary = {"status": w18.STATUS}
        return (
            provenance,
            terminal,
            review,
            phase,
            summary,
            mock.patch.object(
                w18, "_verify_authoritative_w1_4", return_value=w18.EXPECTED_W14_RESULT
            ),
            mock.patch.object(w18, "load_committed_plan", return_value=plan),
            mock.patch.object(w18, "_load_parent_state", return_value=state),
            mock.patch.object(w18, "_protected_snapshot", return_value={"fixed": "yes"}),
            mock.patch.object(
                w18, "build_continuation_provenance", return_value=provenance
            ),
            mock.patch.object(w18, "_build_terminal", return_value=terminal),
            mock.patch.object(w18, "_build_review", return_value=review),
            mock.patch.object(w18, "_build_phase", return_value=phase),
            mock.patch.object(w18, "_verify_locked", return_value=summary),
            mock.patch.object(
                w18,
                "_write_or_match",
                side_effect=lambda _layout, path, _value, _label: writes.append(path.name),
            ),
            mock.patch.object(w18, "_load_json", return_value=state["ledger"]),
            mock.patch.object(w18, "_require_safe_run_file", side_effect=lambda _layout, path, _label: path),
            mock.patch.object(w18, "_full_parent_inventory", return_value={"fixed": "yes"}),
            mock.patch.object(w18, "_validate_parent_inventory"),
            mock.patch.object(w18, "_secure_lock", side_effect=self._lock),
            mock.patch.object(w18, "_w18_snapshot", return_value={"fixed": "yes"}),
        )

    def test_human_attestation_is_mandatory_before_parent_verification(self) -> None:
        with mock.patch.object(w18, "_verify_authoritative_w1_4") as verifier:
            with self.assertRaisesRegex(w18.W18OperatorError, "attest-human"):
                w18.seal_w1_8(
                    w18.W18Layout(Path("/tmp/run")),
                    Path("/tmp/work"),
                    Path("/tmp/w12"),
                    operator_id="operator",
                    attest_human_reviewed_retain_enoch0=False,
                )
        verifier.assert_not_called()

    def test_phase_is_last_write_and_no_seed_api_is_called(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            layout = w18.W18Layout(Path(temporary))
            state = _state()
            plan = {"plan": True}
            writes = []
            patches = self._patches(state, plan, writes)
            contexts = patches[5:]
            with contextlib.ExitStack() as stack:
                for patcher in contexts:
                    stack.enter_context(patcher)
                with mock.patch.object(
                    w18.enoch_week1,
                    "consume_seed_batch_once",
                    side_effect=AssertionError("seed claim attempted"),
                ):
                    result = w18.seal_w1_8(
                        layout,
                        Path("/tmp/work"),
                        Path("/tmp/w12"),
                        operator_id="codex-w1.8-test",
                        attest_human_reviewed_retain_enoch0=True,
                    )
            self.assertEqual(result, {"status": w18.STATUS})
            self.assertEqual(
                writes,
                [
                    "input.json",
                    "continuation-provenance.json",
                    "freeze-or-no-confirmed-candidate-decision.json",
                    "human-review-attestation.json",
                    "final-ledger.json",
                    "phase-manifest.json",
                ],
            )

    def test_existing_phase_is_verified_without_repair_writes(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            layout = w18.W18Layout(Path(temporary))
            w18._require_w18_tree(layout, create=True)  # noqa: SLF001
            layout.phase.write_text("{}", encoding="utf-8")
            state = _state()
            plan = {"plan": True}
            writes = []
            patches = self._patches(state, plan, writes)
            with contextlib.ExitStack() as stack:
                for patcher in patches[5:]:
                    stack.enter_context(patcher)
                result = w18.seal_w1_8(
                    layout,
                    Path("/tmp/work"),
                    Path("/tmp/w12"),
                    operator_id="codex-w1.8-test",
                    attest_human_reviewed_retain_enoch0=True,
                )
            self.assertEqual(result, {"status": w18.STATUS})
            self.assertEqual(writes, [])

    def test_operator_id_is_stable_and_restricted(self) -> None:
        self.assertEqual(w18._operator_id("codex-w1.8:review"), "codex-w1.8:review")  # noqa: SLF001
        for value in ("", " contains-space", "x" * 129, None):
            with self.subTest(value=value), self.assertRaises(w18.W18OperatorError):
                w18._operator_id(value)  # noqa: SLF001

    def test_offline_guard_blocks_every_mutating_surface_and_restores_it(self) -> None:
        original_atomic = w18.enoch_week1.atomic_write_json
        original_consume = w18.enoch_week1.consume_seed_batch_once
        original_write = w18._write_or_match  # noqa: SLF001
        with w18._offline_api_guard():  # noqa: SLF001
            for function in (
                w18.enoch_week1.atomic_write_json,
                w18.enoch_week1.consume_seed_batch_once,
                w18._write_or_match,  # noqa: SLF001
                w18.w14.run_w1_4,
            ):
                with self.assertRaisesRegex(w18.W18OperatorError, "offline"):
                    function()
        self.assertIs(w18.enoch_week1.atomic_write_json, original_atomic)
        self.assertIs(w18.enoch_week1.consume_seed_batch_once, original_consume)
        self.assertIs(w18._write_or_match, original_write)  # noqa: SLF001


if __name__ == "__main__":
    unittest.main()
