#!/usr/bin/env python3
"""Adversarial tests for the authoritative W1.3 continuation."""

from __future__ import annotations

import copy
import contextlib
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from training import enoch_week1
from training import enoch_week1_runner
from training import enoch_week1_w1_3_operator as w13


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _metrics(
    *,
    level: float = 0.1,
    lower: float = 0.0,
    margin: float = 1.0,
    win_rate: float = 0.01,
    failure: str | None = None,
) -> dict[str, object]:
    counters = {name: 0 for name in enoch_week1.FAILURE_COUNTER_NAMES}
    if failure is not None:
        counters[failure] = 1
    return {
        "bootstrap": {
            "algorithm": enoch_week1.BOOTSTRAP_ALGORITHM,
            "replicates": enoch_week1.BOOTSTRAP_REPLICATES,
        },
        "candidate_completed_worlds_mean": 1_400.0,
        "candidate_latency_ms": {"p50": 200.0, "p95": 250.0},
        "control_completed_worlds_mean": 1_400.0,
        "control_latency_ms": {"p50": 200.0, "p95": 250.0},
        "failure_counters": counters,
        "level_utility": {
            "estimate": level,
            "mde_95_80": 0.01,
            "paired_bootstrap_95": [lower, 0.2],
            "paired_bootstrap_lower_95": lower,
            "paired_bootstrap_upper_95": 0.2,
        },
        "pair_count": 800,
        "point_margin": {
            "estimate": margin,
            "mde_95_80": 0.1,
            "paired_bootstrap_95": [0.0, 2.0],
            "paired_bootstrap_lower_95": 0.0,
            "paired_bootstrap_upper_95": 2.0,
        },
        "point_margin_estimate": margin,
        "style_metric_estimates": {
            name: 0.1 for name in enoch_week1.WEEK1_STYLE_METRICS
        },
        "win_rate": {
            "estimate": win_rate,
            "mde_95_80": 0.01,
            "paired_bootstrap_95": [-0.01, 0.03],
            "paired_bootstrap_lower_95": -0.01,
            "paired_bootstrap_upper_95": 0.03,
        },
        "win_rate_estimate": win_rate,
    }


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = Path(__file__).with_name("enoch_week1_w1_3_plan.json")
        cls.plan = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_exact_committed_plan(self) -> None:
        self.assertEqual(
            w13.validate_committed_plan(self.plan),
            enoch_week1.canonical_json_sha256(self.plan),
        )
        self.assertEqual(
            [arm["arm_id"] for arm in self.plan["arms"]],
            list(w13.SURVIVOR_ARM_IDS),
        )
        self.assertEqual([arm["ordinal"] for arm in self.plan["arms"]], [1, 2, 4, 10, 15])
        self.assertEqual(
            [arm["scenario_id"] for arm in self.plan["arms"]],
            [
                "standard",
                "standard",
                "development-finding-friends",
                "standard",
                "standard",
            ],
        )
        self.assertEqual(self.plan["ranking_rule"], list(w13.RANKING_RULE))

    def test_rejects_arm_order_seed_scenario_rule_and_work_drift(self) -> None:
        mutations = []
        reordered = copy.deepcopy(self.plan)
        reordered["arms"][0], reordered["arms"][1] = (
            reordered["arms"][1],
            reordered["arms"][0],
        )
        mutations.append(reordered)
        seed = copy.deepcopy(self.plan)
        seed["arms"][0]["seed_set_sha256"] = _digest("wrong-seeds")
        mutations.append(seed)
        scenario = copy.deepcopy(self.plan)
        scenario["arms"][2]["scenario_id"] = "standard"
        mutations.append(scenario)
        rule = copy.deepcopy(self.plan)
        rule["arms"][0]["development_rule"] = None
        mutations.append(rule)
        work = copy.deepcopy(self.plan)
        work["fixed_work"]["worlds"] = 25
        mutations.append(work)
        ranking = copy.deepcopy(self.plan)
        ranking["ranking_rule"].reverse()
        mutations.append(ranking)
        for mutation in mutations:
            with self.subTest(mutation=mutations.index(mutation)):
                with self.assertRaises((w13.W13OperatorError, TypeError)):
                    w13.validate_committed_plan(mutation)


class HistoricalVerifierTests(unittest.TestCase):
    def test_isolated_command_reintroduces_only_validated_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            layout = w13.W13Layout(root)
            sealed = root / "sealed-d8"
            sealed.mkdir()
            provenance = {
                "continuation_provenance_fingerprint": w13.SEALED_W12_PROVENANCE
            }
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=json.dumps(w13.EXPECTED_W12_VERIFY_RESULT).encode("utf-8"),
                stderr=b"",
            )
            with mock.patch.object(w13, "_load_json", return_value=provenance), mock.patch.object(
                w13, "_validate_w1_2_workspace", return_value=sealed
            ) as validate, mock.patch.object(
                w13.subprocess, "run", return_value=completed
            ) as run:
                self.assertEqual(
                    w13.verify_sealed_w1_2(layout, sealed),
                    w13.EXPECTED_W12_VERIFY_RESULT,
                )
            self.assertEqual(validate.call_count, 2)
            command = run.call_args.args[0]
            self.assertEqual(command[:5], [
                w13.sys.executable,
                "-I",
                "-B",
                "-c",
                w13.W12_RUNPY_BOOTSTRAP,
            ])
            self.assertEqual(command[5], str(sealed))
            self.assertEqual(command[6], str(sealed / "training/enoch_week1_w1_2_operator.py"))
            self.assertNotIn("shell", run.call_args.kwargs)
            self.assertEqual(run.call_args.kwargs["cwd"], sealed)

    def test_rejects_nonexact_subprocess_result_and_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = w13.W13Layout(Path(temporary))
            sealed = Path(temporary) / "sealed"
            provenance = {
                "continuation_provenance_fingerprint": w13.SEALED_W12_PROVENANCE
            }
            for stdout, stderr in (
                (b"{}", b""),
                (json.dumps(w13.EXPECTED_W12_VERIFY_RESULT).encode(), b"warning"),
            ):
                completed = subprocess.CompletedProcess([], 0, stdout, stderr)
                with self.subTest(stderr=stderr), mock.patch.object(
                    w13, "_load_json", return_value=provenance
                ), mock.patch.object(
                    w13, "_validate_w1_2_workspace", return_value=sealed
                ), mock.patch.object(
                    w13.subprocess, "run", return_value=completed
                ), self.assertRaises(w13.W13OperatorError):
                    w13.verify_sealed_w1_2(layout, sealed)


class ProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = json.loads(
            Path(__file__).with_name("enoch_week1_w1_3_plan.json").read_text()
        )

    def _identity(self) -> dict[str, object]:
        return {
            "base_tree_manifest_sha256": _digest("base-manifest"),
            "changed_paths": [
                {
                    "new_blob": _digest(relative)[:40],
                    "old_blob": None,
                    "path": relative,
                    "sha256": _digest(Path(relative).name),
                    "status": "A",
                }
                for relative in w13.CONTINUATION_PATHS
            ],
            "continuation_git_commit": "b" * 40,
            "continuation_git_tree": "c" * 40,
            "critical_base_module_blobs": [
                {"blob": "d" * 40, "path": relative}
                for relative in w13.CRITICAL_BASE_MODULES
            ],
            "git_tree_manifest_sha256": _digest("head-manifest"),
        }

    def _runtime(self) -> list[dict[str, str]]:
        return [
            {"path": relative, "sha256": _digest(Path(relative).name)}
            for _module, relative in w13.RUNTIME_MODULES
        ]

    def _git_bytes(self, _workspace: Path, *arguments: str) -> bytes:
        spec = arguments[-1]
        relative = spec.split(":", 1)[-1]
        return Path(relative).name.encode("utf-8")

    def test_stored_provenance_binds_base_tree_and_runtime_git_bytes(self) -> None:
        workspace = Path(w13.__file__).resolve().parents[1]
        layout = w13.W13Layout(Path("/unused"))
        identity = self._identity()
        patches = (
            mock.patch.object(w13, "_continuation_git_identity", return_value=identity),
            mock.patch.object(w13, "_git_text", return_value="a" * 40 + "\n"),
            mock.patch.object(w13, "_git_bytes", side_effect=self._git_bytes),
            mock.patch.object(
                w13,
                "_sha256_file",
                side_effect=lambda path: _digest(Path(path).name),
            ),
            mock.patch.object(w13, "_runtime_import_records", return_value=self._runtime()),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            artifact = w13.build_continuation_provenance(
                layout, workspace, self.plan
            )
            w13.validate_continuation_provenance(
                artifact,
                layout,
                workspace,
                self.plan,
                live_source=False,
            )

            wrong_tree = copy.deepcopy(artifact)
            wrong_tree["base_git_tree"] = "e" * 40
            body = dict(wrong_tree)
            body.pop("continuation_provenance_fingerprint")
            wrong_tree["continuation_provenance_fingerprint"] = (
                enoch_week1.canonical_json_sha256(body)
            )
            with self.assertRaisesRegex(w13.W13OperatorError, "base Git tree"):
                w13.validate_continuation_provenance(
                    wrong_tree,
                    layout,
                    workspace,
                    self.plan,
                    live_source=False,
                )

            wrong_runtime = copy.deepcopy(artifact)
            wrong_runtime["runtime_imports"][0]["sha256"] = _digest("forged")
            body = dict(wrong_runtime)
            body.pop("continuation_provenance_fingerprint")
            wrong_runtime["continuation_provenance_fingerprint"] = (
                enoch_week1.canonical_json_sha256(body)
            )
            with self.assertRaisesRegex(w13.W13OperatorError, "runtime hashes"):
                w13.validate_continuation_provenance(
                    wrong_runtime,
                    layout,
                    workspace,
                    self.plan,
                    live_source=False,
                )


class DeclarationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = enoch_week1.build_protocol(0x5EED_2026_0704_0004)
        cls.plan = json.loads(
            Path(__file__).with_name("enoch_week1_w1_3_plan.json").read_text(
                encoding="utf-8"
            )
        )

    def _state(self) -> tuple[dict[str, object], str]:
        evaluator = {
            "binary_sha256": "acd0aec3fca93bfd2a5d97fdfa7f39c882bbf3582bd3a77c938713e10701a707",
            "configuration_sha256": "b6f0439e0f2d24c8eee1043d2d4a30c32341e4bedc53434f58ce85098f704035",
            "source_sha256": "ed160a2a0036dcc3898867aafb1d0daa606b427a822c5c4e0cdea08d5bcb0051",
        }
        fake_environment = {"test": "sealed-environment"}
        env_hash = enoch_week1.canonical_json_sha256(fake_environment)
        parent_arms = {}
        for plan_arm in self.plan["arms"]:
            launch = enoch_week1_runner.build_launch_configuration(
                candidate_arm_ids=[plan_arm["arm_id"]],
                worlds=24,
                candidates=6,
                rollout_tricks=6,
                scenario_id=plan_arm["scenario_id"],
                deadline_ms=30_000,
            )
            identities = enoch_week1_runner.build_in_process_identity_bindings(
                evaluator, launch
            )
            parent_arms[plan_arm["arm_id"]] = {
                "environment_identity": fake_environment,
                "identity_bindings": identities,
                "launch_configuration": launch,
                "rule": plan_arm["development_rule"],
            }
        state = {
            "control": {
                "control_manifest_fingerprint": _digest("control"),
                "evaluator_identity": evaluator,
            },
            "fixture": {
                "failure_count": 0,
                "fixture_report_fingerprint": w13.FIXTURE_REPORT_FINGERPRINT,
                "records": [{}] * 31,
                "source_files_sha256": _digest("fixture-source"),
            },
            "parent_arms": parent_arms,
            "report": {
                "runtime_evaluation_control_fingerprint": _digest("runtime-control"),
                "runtime_evaluator_fingerprint": _digest("runtime-evaluator"),
            },
        }
        return state, env_hash

    def test_declaration_has_five_exact_800_pair_eight_shard_comparisons(self) -> None:
        state, env_hash = self._state()
        provenance = {"continuation_provenance_fingerprint": _digest("provenance")}
        with mock.patch.object(w13, "ENVIRONMENT_FINGERPRINT", env_hash):
            declaration = w13.build_campaign_declaration(
                self.protocol,
                self.plan,
                provenance,
                state,
                environment_identity_override=next(
                    iter(state["parent_arms"].values())
                )["environment_identity"],
            )
        self.assertEqual(
            [arm["arm_id"] for arm in declaration["arms"]],
            list(w13.SURVIVOR_ARM_IDS),
        )
        for arm, plan_arm in zip(declaration["arms"], self.plan["arms"]):
            comparison = arm["comparison"]
            self.assertEqual(comparison["phase"], "W1.3")
            self.assertEqual(comparison["seed_namespace"], plan_arm["seed_namespace"])
            self.assertEqual(comparison["seed_set_sha256"], plan_arm["seed_set_sha256"])
            self.assertEqual(comparison["pair_count"], 800)
            self.assertEqual(
                [len(shard["seed_indices"]) for shard in comparison["shards"]],
                [100] * 8,
            )
            self.assertEqual(comparison["development_rule"], plan_arm["development_rule"])


class DecisionAndSetTests(unittest.TestCase):
    def _arm(self, arm_id: str = "bid-ownership") -> dict[str, object]:
        plan = json.loads(
            Path(__file__).with_name("enoch_week1_w1_3_plan.json").read_text()
        )
        plan_arm = next(arm for arm in plan["arms"] if arm["arm_id"] == arm_id)
        return {
            "arm_id": arm_id,
            "comparison": {
                "development_rule": plan_arm["development_rule"],
                "phase": "W1.3",
            },
            "development_rule": plan_arm["development_rule"],
            "development_rule_sha256": plan_arm["development_rule_sha256"],
            "parent_evidence": {
                "advancement_decision_fingerprint": plan_arm[
                    "w1_2_advancement_decision_fingerprint"
                ],
                "comparison_protocol_fingerprint": plan_arm[
                    "w1_2_comparison_protocol_fingerprint"
                ],
                "merged_result_fingerprint": plan_arm[
                    "w1_2_merged_result_fingerprint"
                ],
            },
        }

    def test_support_decision_passes_or_stops_by_carried_rule(self) -> None:
        arm = self._arm()
        fixture = {"failure_count": 0}
        with mock.patch.object(
            w13.enoch_week1,
            "validate_comparison_protocol_manifest",
            return_value=_digest("comparison"),
        ), mock.patch.object(
            w13.enoch_week1,
            "validate_merged_result",
            return_value=_digest("merged"),
        ), mock.patch.object(
            w13.enoch_week1_fixtures,
            "validate_report",
            return_value=w13.FIXTURE_REPORT_FINGERPRINT,
        ):
            passed = w13.build_support_decision(
                {}, arm, {"metrics": _metrics()}, fixture
            )
            stopped = w13.build_support_decision(
                {}, arm, {"metrics": _metrics(level=-0.01)}, fixture
            )
            self.assertEqual(passed["decision"], "advance-to-w1.4")
            self.assertEqual(stopped["decision"], "stop-and-record")
            self.assertIn("level-utility-estimate-below-rule", stopped["reasons"])
            with self.assertRaisesRegex(
                w13.W13OperatorError, "nonzero invalidating counter"
            ):
                w13.build_support_decision(
                    {},
                    arm,
                    {"metrics": _metrics(failure="illegal_action")},
                    fixture,
                )

    def test_supported_set_records_explicit_no_survivor(self) -> None:
        declaration_arms = []
        evidence = {}
        for sequence, arm_id in enumerate(w13.SURVIVOR_ARM_IDS, start=1):
            arm = self._arm(arm_id)
            arm.update(
                {
                    "ordinal": w13.EXPECTED_PARENT_ARM_FINGERPRINTS[arm_id][
                        "ordinal"
                    ],
                    "sequence": sequence,
                }
            )
            arm["comparison"].update(
                {
                    "candidate_fingerprint": _digest(f"candidate:{arm_id}"),
                    "comparison_protocol_fingerprint": _digest(
                        f"comparison:{arm_id}"
                    ),
                    "environment_fingerprint": _digest("environment"),
                }
            )
            declaration_arms.append(arm)
            metrics = _metrics(level=-0.01)
            evidence[arm_id] = {
                "arm_id": arm_id,
                "external_evidence_fingerprint": _digest(f"external:{arm_id}"),
                "machine_attestation_fingerprint": _digest(f"machine:{arm_id}"),
                "merged_result": {
                    "merged_result_fingerprint": _digest(f"merged:{arm_id}"),
                    "metrics": metrics,
                },
                "raw_output_sha256s": [],
                "runner_execution": {"arm": arm_id},
                "shard_results": [
                    {"shard_result_fingerprint": _digest(f"shard:{arm_id}")}
                ],
                "support_decision": {
                    "decision": "stop-and-record",
                    "reasons": ["level-utility-estimate-below-rule"],
                    "support_decision_fingerprint": _digest(f"decision:{arm_id}"),
                },
            }
        declaration = {
            "arms": declaration_arms,
            "campaign_declaration_fingerprint": _digest("declaration"),
        }
        fixture = {"fixture_report_fingerprint": w13.FIXTURE_REPORT_FINGERPRINT}
        ledger = {"ledger_fingerprint": _digest("final-ledger")}
        protocol = {"seed_registry_sha256": _digest("seed-registry")}
        with mock.patch.object(
            w13.enoch_week1,
            "validate_merged_result",
            side_effect=lambda _p, _c, merged: merged["merged_result_fingerprint"],
        ), mock.patch.object(
            w13,
            "validate_support_decision",
            side_effect=lambda decision, *_args: decision[
                "support_decision_fingerprint"
            ],
        ), mock.patch.object(
            w13,
            "_validate_supported_final_ledger",
            return_value=ledger["ledger_fingerprint"],
        ):
            artifact = w13.build_supported_change_set(
                protocol, declaration, fixture, ledger, evidence
            )
        self.assertEqual(artifact["status"], "no-survivor")
        self.assertEqual(artifact["summary"]["supported_arm_ids"], [])
        self.assertEqual(
            artifact["summary"]["stopped_arm_ids"], list(w13.SURVIVOR_ARM_IDS)
        )
        self.assertEqual(len(artifact["arm_results"]), 5)
        self.assertEqual(artifact["final_consumed_count"], w13.FINAL_COUNT)
        self.assertEqual(
            artifact["final_ledger_fingerprint"], ledger["ledger_fingerprint"]
        )
        tampered = copy.deepcopy(artifact)
        tampered["summary"]["total_pair_count"] = 3_999
        with mock.patch.object(
            w13.enoch_week1,
            "validate_merged_result",
            side_effect=lambda _p, _c, merged: merged["merged_result_fingerprint"],
        ), mock.patch.object(
            w13,
            "validate_support_decision",
            side_effect=lambda decision, *_args: decision[
                "support_decision_fingerprint"
            ],
        ), mock.patch.object(
            w13,
            "_validate_supported_final_ledger",
            return_value=ledger["ledger_fingerprint"],
        ), self.assertRaisesRegex(w13.W13OperatorError, "does not reconstruct"):
            w13.validate_supported_change_set(
                tampered, protocol, declaration, fixture, ledger, evidence
            )


class LedgerAndPhaseTests(unittest.TestCase):
    def _ledger(self, consumed: list[dict[str, object]]) -> dict[str, object]:
        body = {
            "consumed": consumed,
            "ledger_kind": "enoch-week1-seed-consumption-ledger",
            "manifest_version": 1,
            "protocol_fingerprint": _digest("protocol"),
            "seed_registry_sha256": _digest("registry"),
        }
        return {**body, "ledger_fingerprint": enoch_week1.canonical_json_sha256(body)}

    def _declaration(self) -> dict[str, object]:
        arms = []
        for sequence, arm_id in enumerate(w13.SURVIVOR_ARM_IDS, start=1):
            fingerprint = _digest(f"comparison:{arm_id}")
            arms.append(
                {
                    "arm_id": arm_id,
                    "comparison": {
                        "comparison_protocol_fingerprint": fingerprint,
                        "seed_namespace": f"dev/survivor/{arm_id}",
                        "shards": [
                            {"seed_indices": [0], "shard_id": "shard-000"}
                        ],
                    },
                    "seed_namespace": f"dev/survivor/{arm_id}",
                    "sequence": sequence,
                }
            )
        return {"arms": arms}

    def _claim(self, arm_id: str, sequence: int) -> dict[str, object]:
        fingerprint = _digest(f"comparison:{arm_id}")
        return {
            "consumer": f"runner:{fingerprint[:16]}:shard-000",
            "index": 0,
            "namespace": f"dev/survivor/{arm_id}",
            "seed": sequence + 10,
            "sequence": sequence,
        }

    def test_final_ledger_exact_prefix_count_and_forbidden_namespace(self) -> None:
        parent_record = {
            "consumer": "parent",
            "index": 0,
            "namespace": "smoke/product/001",
            "seed": 1,
            "sequence": 0,
        }
        parent = self._ledger([parent_record])
        claims = [
            self._claim(arm_id, sequence)
            for sequence, arm_id in enumerate(w13.SURVIVOR_ARM_IDS, start=1)
        ]
        final = self._ledger([parent_record, *claims])
        with mock.patch.object(w13, "PRECLAIM_COUNT", 1), mock.patch.object(
            w13, "FINAL_COUNT", 6
        ), mock.patch.object(
            w13, "SEALED_W12_LEDGER", parent["ledger_fingerprint"]
        ), mock.patch.object(w13.enoch_week1, "validate_seed_ledger"):
            w13._expected_final_ledger(  # noqa: SLF001
                {}, final, parent, self._declaration()
            )
            rewritten_parent = dict(parent_record)
            rewritten_parent["consumer"] = "rewritten"
            with self.assertRaisesRegex(w13.W13OperatorError, "prefix changed"):
                w13._expected_final_ledger(  # noqa: SLF001
                    {},
                    self._ledger([rewritten_parent, *claims]),
                    parent,
                    self._declaration(),
                )

        forbidden = {
            "consumer": "runner:forbidden:shard-000",
            "index": 0,
            "namespace": "dev/survivor/kitty-burial",
            "seed": 99,
            "sequence": 6,
        }
        with mock.patch.object(w13, "PRECLAIM_COUNT", 1), mock.patch.object(
            w13, "FINAL_COUNT", 7
        ), mock.patch.object(
            w13, "SEALED_W12_LEDGER", parent["ledger_fingerprint"]
        ), mock.patch.object(
            w13.enoch_week1, "validate_seed_ledger"
        ), self.assertRaisesRegex(w13.W13OperatorError, "undeclared W1.3"):
            w13._expected_final_ledger(  # noqa: SLF001
                {},
                self._ledger([parent_record, *claims, forbidden]),
                parent,
                self._declaration(),
            )

    def test_append_only_extension_rejects_rewritten_snapshot(self) -> None:
        first = {
            "consumer": "parent",
            "index": 0,
            "namespace": "smoke/product/001",
            "seed": 1,
            "sequence": 0,
        }
        snapshot = self._ledger([first])
        downstream = {
            "consumer": "runner:downstream:shard-000",
            "index": 0,
            "namespace": "dev/combination/qualification",
            "seed": 2,
            "sequence": 1,
        }
        extension = self._ledger([first, downstream])
        with mock.patch.object(w13.enoch_week1, "validate_seed_ledger"):
            w13._validate_ledger_extension({}, extension, snapshot)  # noqa: SLF001
            rewritten = dict(first)
            rewritten["consumer"] = "rewritten"
            with self.assertRaisesRegex(w13.W13OperatorError, "append-only"):
                w13._validate_ledger_extension(  # noqa: SLF001
                    {}, self._ledger([rewritten]), snapshot
                )
            with self.assertRaisesRegex(w13.W13OperatorError, "non-W1.4"):
                w13._validate_ledger_extension(  # noqa: SLF001
                    {},
                    self._ledger([first, self._claim("bid-ownership", 1)]),
                    snapshot,
                )

    def test_out_of_order_completed_frontier_retires(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = w13.W13Layout(Path(temporary))
            declaration = self._declaration()
            second = declaration["arms"][1]
            execution = (
                layout.arm(second["sequence"], second["arm_id"])
                / "attempts"
                / "attempt-001"
                / "execution"
            )
            execution.mkdir(parents=True)
            (execution / "execution-complete.json").write_text("{}\n")
            ledger = self._ledger([self._claim(second["arm_id"], 0)])
            layout.base.ledger.write_text(json.dumps(ledger))
            with mock.patch.object(
                w13.enoch_week1, "validate_seed_ledger"
            ), self.assertRaisesRegex(w13.W13OperatorError, "protocol retired"):
                w13._scan_resume_frontier(  # noqa: SLF001
                    layout,
                    {"protocol_fingerprint": _digest("protocol")},
                    self._ledger([]),
                    declaration,
                )
            retirement = json.loads(layout.retirement.read_text())
            self.assertEqual(
                retirement["reason"], "claimed-w1.3-completion-is-out-of-order"
            )

    def test_phase_tamper_is_not_accepted(self) -> None:
        expected = {
            "artifacts": [
                {
                    "artifact_id": "supported-independent-change-set",
                    "sha256": _digest("set"),
                }
            ],
            "phase_manifest_fingerprint": _digest("phase"),
        }
        tampered = copy.deepcopy(expected)
        tampered["artifacts"][0]["sha256"] = _digest("forged")
        supported = {"supported_change_set_fingerprint": _digest("set")}
        state = {"fixture": {}, "phase2": {}, "protocol": {}}
        with mock.patch.object(
            w13, "_build_phase3", return_value=expected
        ), self.assertRaisesRegex(w13.W13OperatorError, "does not reconstruct"):
            w13._validate_phase3(  # noqa: SLF001
                tampered, state, {}, {}, supported, {}, {}
            )


class RetirementTests(unittest.TestCase):
    def _arm(self) -> dict[str, object]:
        return {
            "arm_id": "bid-ownership",
            "seed_namespace": "dev/survivor/bid-ownership",
            "sequence": 1,
        }

    def _ledger(self) -> dict[str, object]:
        return {
            "consumed": [
                {
                    "consumer": "runner:x:shard-000",
                    "index": 0,
                    "namespace": "dev/survivor/bid-ownership",
                    "seed": 1,
                    "sequence": 0,
                }
            ],
            "ledger_fingerprint": _digest("ledger"),
        }

    def test_claimed_malformed_attempt_entry_still_retires(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = w13.W13Layout(Path(temporary))
            attempts = layout.arm(1, "bid-ownership") / "attempts"
            attempts.mkdir(parents=True)
            (attempts / "malformed-entry").write_text("bad", encoding="utf-8")
            with self.assertRaisesRegex(w13.W13OperatorError, "protocol retired"):
                w13._retire_if_claimed_incomplete(  # noqa: SLF001
                    layout,
                    {"protocol_fingerprint": _digest("protocol")},
                    self._ledger(),
                    self._arm(),
                )
            retirement = json.loads(layout.retirement.read_text(encoding="utf-8"))
            self.assertEqual(retirement["attempt_directories"], ["malformed-entry"])
            self.assertEqual(
                retirement["reason"],
                "malformed-or-multiple-claimed-w1.3-attempt-entries",
            )

    def test_cached_value_and_overflow_errors_retire(self) -> None:
        for exception in (ValueError("bad value"), OverflowError("too large")):
            with self.subTest(exception=type(exception).__name__), tempfile.TemporaryDirectory() as temporary:
                layout = w13.W13Layout(Path(temporary))
                execution = (
                    layout.arm(1, "bid-ownership")
                    / "attempts"
                    / "attempt-001"
                    / "execution"
                )
                execution.mkdir(parents=True)
                (execution / "execution-complete.json").write_text("{}\n")
                layout.base.ledger.write_text(json.dumps(self._ledger()))
                arm = {**self._arm(), "comparison": {}}
                with mock.patch.object(
                    w13.enoch_week1, "validate_seed_ledger"
                ), mock.patch.object(
                    w13, "_validate_completed_arm", side_effect=exception
                ), mock.patch.object(
                    w13, "_retire", side_effect=w13.W13OperatorError("retired")
                ) as retire, self.assertRaisesRegex(w13.W13OperatorError, "retired"):
                    w13._run_arm(  # noqa: SLF001
                        layout,
                        Path(temporary),
                        {"protocol": {}},
                        {},
                        {},
                        {},
                        arm,
                        operator_id="test",
                        base_environment={},
                    )
                self.assertEqual(
                    retire.call_args.kwargs["reason"],
                    "claimed-w1.3-completion-marker-is-invalid",
                )

    def test_just_finished_overflow_error_uses_same_retirement_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = w13.W13Layout(Path(temporary))
            attempt = layout.arm(1, "bid-ownership") / "attempts" / "attempt-001"
            empty = {"consumed": [], "ledger_fingerprint": _digest("empty")}
            claimed = self._ledger()
            arm = {
                **self._arm(),
                "comparison": {},
                "environment_identity": {},
                "identity_bindings": {"evaluator": {}},
                "launch_configuration": {},
            }
            declaration = {
                "execution_contract": {
                    "available_parallelism": 10,
                    "timeout_seconds": 3600,
                    "worker_count": 8,
                }
            }
            protocol = {"evaluator_environment_policy": {"allowlist": []}}
            lock = contextlib.nullcontext(object())
            with mock.patch.object(
                w13, "_load_json", side_effect=[empty, empty, claimed, claimed]
            ), mock.patch.object(
                w13.enoch_week1, "validate_seed_ledger"
            ), mock.patch.object(
                w13, "_completed_attempt", return_value=None
            ), mock.patch.object(
                w13, "_retire_if_claimed_incomplete"
            ), mock.patch.object(
                w13, "_seal_abandoned_preclaim_attempts"
            ), mock.patch.object(
                w13, "validate_continuation_provenance"
            ), mock.patch.object(
                w13.enoch_week1_runner,
                "authoritative_campaign_lock",
                return_value=lock,
            ), mock.patch.object(
                w13, "_next_attempt", return_value=attempt
            ), mock.patch.object(
                w13.enoch_week1_runner,
                "probe_evaluator_environment_identity",
                return_value={},
            ), mock.patch.object(
                w13.w12_operator, "_validate_environment_probe"
            ), mock.patch.object(
                w13.enoch_week1, "atomic_write_json"
            ), mock.patch.object(
                w13.enoch_week1_evidence,
                "build_machine_contention_attestation",
                return_value={},
            ), mock.patch.object(
                w13.enoch_week1_evidence,
                "build_verified_external_evidence",
                return_value={},
            ), mock.patch.object(
                w13.enoch_week1_evidence,
                "validate_machine_contention_attestation",
            ), mock.patch.object(
                w13.enoch_week1_evidence,
                "validate_verified_external_evidence",
            ), mock.patch.object(
                w13.enoch_week1_runner,
                "run_comparison",
                side_effect=[{}, {}],
            ) as runner, mock.patch.object(
                w13,
                "_validate_completed_arm",
                side_effect=OverflowError("malformed completion"),
            ), mock.patch.object(
                w13, "_retire", side_effect=w13.W13OperatorError("retired")
            ) as retire, self.assertRaisesRegex(w13.W13OperatorError, "retired"):
                w13._run_arm(  # noqa: SLF001
                    layout,
                    Path(temporary),
                    {"protocol": protocol},
                    {},
                    {},
                    declaration,
                    arm,
                    operator_id="test",
                    base_environment={},
                )
            self.assertEqual(runner.call_count, 2)
            self.assertEqual(
                retire.call_args.kwargs["reason"],
                "claimed-w1.3-completion-marker-is-invalid",
            )

    def test_shared_just_finished_boundary_retires_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = w13.W13Layout(Path(temporary))
            attempt = layout.arm(1, "bid-ownership") / "attempts" / "attempt-001"
            with mock.patch.object(
                w13,
                "_validate_completed_arm",
                side_effect=ValueError("malformed numeric field"),
            ), mock.patch.object(
                w13, "_retire", side_effect=w13.W13OperatorError("retired")
            ) as retire, self.assertRaisesRegex(w13.W13OperatorError, "retired"):
                w13._validate_or_retire_completion(  # noqa: SLF001
                    layout,
                    {},
                    self._ledger(),
                    {},
                    self._arm(),
                    attempt,
                    base_environment={},
                )
            self.assertEqual(
                retire.call_args.kwargs["reason"],
                "claimed-w1.3-completion-marker-is-invalid",
            )


if __name__ == "__main__":
    unittest.main()
