#!/usr/bin/env python3
"""Adversarial unit tests for the orchestration-only W1.2 operator."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from training import enoch_week1
from training import enoch_week1_w1_2_operator as w12


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _metrics(
    *,
    lower: float = 0.0,
    level: float = 0.0,
    margin: float = 0.0,
    win_rate: float = 0.0,
) -> dict[str, object]:
    return {
        "failure_counters": {
            name: 0 for name in enoch_week1.FAILURE_COUNTER_NAMES
        },
        "level_utility": {
            "estimate": level,
            "paired_bootstrap_lower_95": lower,
        },
        "point_margin": {"estimate": margin},
        "win_rate": {"estimate": win_rate},
    }


def _declaration_and_evidence() -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    arms: list[dict[str, object]] = []
    evidence: dict[str, dict[str, object]] = {}
    for ordinal, arm_id in enumerate(enoch_week1.ABLATION_ARMS, start=1):
        comparison = {
            "comparison_protocol_fingerprint": _digest(f"comparison:{arm_id}")
        }
        launch = {"arm": arm_id}
        identities = {"candidate": arm_id, "control": "none"}
        arms.append(
            {
                "arm_id": arm_id,
                "comparison": comparison,
                "identity_bindings": identities,
                "launch_configuration": launch,
                "ordinal": ordinal,
            }
        )
        evidence[arm_id] = {
            "advancement_decision": {
                "advancement_decision_fingerprint": _digest(
                    f"decision:{arm_id}"
                ),
                "decision": "advance-to-w1.3",
            },
            "comparison": comparison,
            "environment_probe": {"arm": arm_id},
            "external_evidence_fingerprint": _digest(f"external:{arm_id}"),
            "identity_bindings": identities,
            "launch_configuration": launch,
            "machine_attestation_fingerprint": _digest(f"machine:{arm_id}"),
            "merged_result": {
                "merged_result_fingerprint": _digest(f"merged:{arm_id}"),
                "metrics": _metrics(),
            },
            "raw_output_sha256s": [
                {
                    "sha256": _digest(f"raw:{arm_id}"),
                    "shard_id": "shard-000",
                }
            ],
            "runner_execution": {"arm": arm_id},
            "shard_results": [
                {"shard_result_fingerprint": _digest(f"shard:{arm_id}")}
            ],
        }
    declaration: dict[str, object] = {
        "arms": arms,
        "campaign_declaration_fingerprint": _digest("declaration"),
        "execution_contract": {
            "pair_count": 300,
            "ranking_rule": list(w12.RANKING_RULE),
        },
        "operator_source_provenance_fingerprint": _digest("provenance"),
        "parent_phase": {"phase_manifest_fingerprint": _digest("parent")},
    }
    return declaration, evidence


class CommittedPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan_path = Path(__file__).with_name("enoch_week1_w1_2_plan.json")
        cls.plan = json.loads(cls.plan_path.read_text(encoding="utf-8"))

    def test_committed_plan_covers_the_exact_canonical_fifteen_arms(self) -> None:
        fingerprint = w12.validate_committed_plan(self.plan)

        self.assertEqual(fingerprint, enoch_week1.canonical_json_sha256(self.plan))
        self.assertEqual(
            [arm["arm_id"] for arm in self.plan["arms"]],
            list(enoch_week1.ABLATION_ARMS),
        )
        self.assertEqual(len(self.plan["arms"]), 15)
        self.assertEqual(
            len({arm["comparison_id"] for arm in self.plan["arms"]}), 15
        )
        self.assertEqual(
            len(
                {
                    arm["development_rule"]["rule_id"]
                    for arm in self.plan["arms"]
                }
            ),
            15,
        )

    def test_plan_rejects_missing_duplicate_reordered_or_extra_arms(self) -> None:
        mutations: dict[str, dict[str, object]] = {}

        missing = copy.deepcopy(self.plan)
        missing["arms"].pop()
        mutations["missing"] = missing

        duplicate = copy.deepcopy(self.plan)
        duplicate["arms"][-1] = copy.deepcopy(duplicate["arms"][0])
        mutations["duplicate"] = duplicate

        reordered = copy.deepcopy(self.plan)
        reordered["arms"][0], reordered["arms"][1] = (
            reordered["arms"][1],
            reordered["arms"][0],
        )
        mutations["reordered"] = reordered

        extra = copy.deepcopy(self.plan)
        extra["arms"].append(copy.deepcopy(extra["arms"][-1]))
        mutations["extra"] = extra

        for label, mutation in mutations.items():
            with self.subTest(label=label), self.assertRaises(
                w12.W12OperatorError
            ):
                w12.validate_committed_plan(mutation)

    def test_plan_schema_rejects_unknown_fields_and_mutable_execution_contracts(self) -> None:
        mutations: dict[str, dict[str, object]] = {}

        unknown = copy.deepcopy(self.plan)
        unknown["post_result_override"] = True
        mutations["unknown top-level field"] = unknown

        wrong_pair_count = copy.deepcopy(self.plan)
        wrong_pair_count["pair_count_per_arm"] = 299
        mutations["partial seed prefix"] = wrong_pair_count

        budget_mode = copy.deepcopy(self.plan)
        budget_mode["fixed_work"]["work_mode"] = "budget"
        budget_mode["fixed_work"]["budget_ms"] = 2_200
        mutations["budget work"] = budget_mode

        reordered_ranking = copy.deepcopy(self.plan)
        reordered_ranking["ranking_rule"][1:3] = reversed(
            reordered_ranking["ranking_rule"][1:3]
        )
        mutations["ranking drift"] = reordered_ranking

        for label, mutation in mutations.items():
            with self.subTest(label=label), self.assertRaises(
                w12.W12OperatorError
            ):
                w12.validate_committed_plan(mutation)


class RankingTests(unittest.TestCase):
    def test_rank_key_applies_every_tie_break_in_declared_order(self) -> None:
        arm_ids = list(enoch_week1.ABLATION_ARMS[:9])
        ordinals = {
            arm_id: ordinal for ordinal, arm_id in enumerate(arm_ids, start=1)
        }
        evidence: dict[str, dict[str, object]] = {}
        for arm_id in arm_ids:
            evidence[arm_id] = {
                "advancement_decision": {"decision": "advance-to-w1.3"},
                "merged_result": {"metrics": _metrics()},
            }

        # A stopped arm remains below every advancing arm despite huge metrics.
        evidence[arm_ids[0]]["advancement_decision"]["decision"] = "stop-and-record"
        evidence[arm_ids[0]]["merged_result"]["metrics"] = _metrics(
            lower=100.0, level=100.0, margin=100.0, win_rate=100.0
        )
        # Lower bound is the first numeric discriminator.
        evidence[arm_ids[1]]["merged_result"]["metrics"] = _metrics(lower=0.4)
        evidence[arm_ids[2]]["merged_result"]["metrics"] = _metrics(lower=0.3)
        # Then level estimate.
        evidence[arm_ids[3]]["merged_result"]["metrics"] = _metrics(
            lower=0.2, level=0.4
        )
        evidence[arm_ids[4]]["merged_result"]["metrics"] = _metrics(
            lower=0.2, level=0.3
        )
        # Then point margin.
        evidence[arm_ids[5]]["merged_result"]["metrics"] = _metrics(
            lower=0.1, level=0.1, margin=2.0
        )
        evidence[arm_ids[6]]["merged_result"]["metrics"] = _metrics(
            lower=0.1, level=0.1, margin=1.0
        )
        # Then win rate; exact ties finally retain canonical ordinal order.
        evidence[arm_ids[7]]["merged_result"]["metrics"] = _metrics(
            lower=0.0, level=0.0, margin=0.0, win_rate=0.2
        )
        evidence[arm_ids[8]]["merged_result"]["metrics"] = _metrics(
            lower=0.0, level=0.0, margin=0.0, win_rate=0.1
        )

        ranked = sorted(
            arm_ids,
            key=lambda arm_id: w12._rank_key(  # noqa: SLF001
                arm_id, evidence[arm_id], ordinals
            ),
        )
        self.assertEqual(
            ranked,
            [
                arm_ids[1],
                arm_ids[2],
                arm_ids[3],
                arm_ids[4],
                arm_ids[5],
                arm_ids[6],
                arm_ids[7],
                arm_ids[8],
                arm_ids[0],
            ],
        )

    def test_rank_key_rejects_nonfinite_metrics(self) -> None:
        arm_id = enoch_week1.ABLATION_ARMS[0]
        for value in (float("nan"), float("inf"), float("-inf")):
            evidence = {
                "advancement_decision": {"decision": "advance-to-w1.3"},
                "merged_result": {"metrics": _metrics(lower=value)},
            }
            with self.subTest(value=value), self.assertRaises(
                w12.W12OperatorError
            ):
                w12._rank_key(arm_id, evidence, {arm_id: 1})  # noqa: SLF001


class RankedTableTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = {
            "protocol_fingerprint": _digest("protocol"),
            "seed_registry_sha256": _digest("seeds"),
        }
        self.fixture = {"fixture": "sealed"}
        self.declaration, self.evidence = _declaration_and_evidence()
        self.fixture_patch = mock.patch.object(
            w12,
            "_fixture_binding",
            return_value={
                "failure_count": 0,
                "fixture_report_fingerprint": _digest("fixture"),
                "record_count": 31,
                "source_files_sha256": _digest("fixture-sources"),
            },
        )
        self.decision_patch = mock.patch.object(
            w12.enoch_week1,
            "validate_w1_3_advancement_decision",
            return_value=_digest("validated-decision"),
        )

    def _build(self) -> dict[str, object]:
        with self.fixture_patch, self.decision_patch:
            return w12.build_ranked_table(
                self.protocol,
                self.declaration,
                self.fixture,
                self.evidence,
            )

    def test_equal_results_rank_by_canonical_arm_ordinal(self) -> None:
        table = self._build()

        self.assertEqual(table["ranking"], list(enoch_week1.ABLATION_ARMS))
        self.assertEqual(
            [record["rank"] for record in table["arm_results"]],
            list(range(1, 16)),
        )
        self.assertEqual(table["summary"]["arm_count"], 15)
        self.assertEqual(table["summary"]["total_pair_count"], 4_500)
        self.assertEqual(
            table["summary"]["aggregate_failure_counters"],
            {name: 0 for name in enoch_week1.FAILURE_COUNTER_NAMES},
        )

    def test_ranked_table_rejects_missing_or_reordered_evidence(self) -> None:
        missing = dict(self.evidence)
        missing.pop(enoch_week1.ABLATION_ARMS[-1])
        reordered = dict(reversed(list(self.evidence.items())))

        for label, evidence in (("missing", missing), ("reordered", reordered)):
            with self.subTest(label=label), self.assertRaisesRegex(
                w12.W12OperatorError, "contain all arms in order"
            ):
                w12.build_ranked_table(
                    self.protocol, self.declaration, self.fixture, evidence
                )

    def test_ranked_table_rejects_substituted_declared_inputs(self) -> None:
        mutations = {
            "comparison": {"comparison_protocol_fingerprint": _digest("other")},
            "launch_configuration": {"arm": "other"},
            "identity_bindings": {"candidate": "other", "control": "none"},
        }
        arm_id = enoch_week1.ABLATION_ARMS[0]
        for field, replacement in mutations.items():
            evidence = copy.deepcopy(self.evidence)
            evidence[arm_id][field] = replacement
            with self.subTest(field=field), self.fixture_patch, self.decision_patch:
                with self.assertRaisesRegex(w12.W12OperatorError, "changed its"):
                    w12.build_ranked_table(
                        self.protocol,
                        self.declaration,
                        self.fixture,
                        evidence,
                    )

    def test_ranked_table_validation_rejects_omissions_and_tampering(self) -> None:
        table = self._build()
        mutations: dict[str, dict[str, object]] = {}

        omitted = copy.deepcopy(table)
        omitted["arm_results"].pop()
        mutations["omitted arm result"] = omitted

        ranking = copy.deepcopy(table)
        ranking["ranking"][0], ranking["ranking"][1] = (
            ranking["ranking"][1],
            ranking["ranking"][0],
        )
        mutations["ranking substitution"] = ranking

        rank = copy.deepcopy(table)
        rank["arm_results"][0]["rank"] = 15
        mutations["rank substitution"] = rank

        summary = copy.deepcopy(table)
        summary["summary"]["total_pair_count"] = 4_499
        mutations["summary substitution"] = summary

        fingerprint = copy.deepcopy(table)
        fingerprint["ranked_independent_ablation_table_fingerprint"] = _digest(
            "forged"
        )
        mutations["fingerprint substitution"] = fingerprint

        for label, mutation in mutations.items():
            with self.subTest(label=label), self.fixture_patch, self.decision_patch:
                with self.assertRaisesRegex(
                    w12.W12OperatorError, "does not reconstruct"
                ):
                    w12.validate_ranked_table(
                        mutation,
                        self.protocol,
                        self.declaration,
                        self.fixture,
                        self.evidence,
                    )


class StateMachineTests(unittest.TestCase):
    def _ledger(self, consumed: list[dict[str, object]]) -> dict[str, object]:
        body = {
            "consumed": consumed,
            "ledger_kind": "test-ledger",
            "manifest_version": 1,
            "protocol_fingerprint": _digest("protocol"),
            "seed_registry_sha256": _digest("seeds"),
        }
        return {
            **body,
            "ledger_fingerprint": enoch_week1.canonical_json_sha256(body),
        }

    def test_historical_ledger_accepts_only_append_only_extensions(self) -> None:
        first = {
            "consumer": "runner:first",
            "index": 0,
            "namespace": "smoke/product/001",
            "seed": 7,
            "sequence": 0,
        }
        snapshot = self._ledger([first])
        second = {
            "consumer": "runner:second",
            "index": 0,
            "namespace": "dev/survivor/bid-ownership",
            "seed": 11,
            "sequence": 1,
        }
        extension = self._ledger([first, second])
        with mock.patch.object(w12.enoch_week1, "validate_seed_ledger"):
            w12._validate_ledger_extension(  # noqa: SLF001
                {}, extension, snapshot
            )

            mutated_first = dict(first)
            mutated_first["consumer"] = "runner:rewritten"
            rewritten = self._ledger([mutated_first, second])
            with self.assertRaises(w12.W12OperatorError):
                w12._validate_ledger_extension(  # noqa: SLF001
                    {}, rewritten, snapshot
                )

    def test_completion_marker_is_not_retired_as_claimed_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = w12.W12Layout(Path(temporary))
            arm = {
                "arm_id": "bid-ownership",
                "ordinal": 1,
                "seed_namespace": "dev/ablation/bid-ownership",
            }
            execution = (
                layout.arm(1, "bid-ownership")
                / "attempts"
                / "attempt-001"
                / "execution"
            )
            execution.mkdir(parents=True)
            (execution / "execution-complete.json").write_text(
                "{}\n", encoding="utf-8"
            )
            ledger = {
                "consumed": [
                    {
                        "consumer": "runner:x:shard-000",
                        "index": 0,
                        "namespace": arm["seed_namespace"],
                        "seed": 1,
                        "sequence": 0,
                    }
                ],
                "ledger_fingerprint": _digest("ledger"),
            }
            w12._retire_if_claimed_incomplete(  # noqa: SLF001
                layout,
                {"protocol_fingerprint": _digest("protocol")},
                ledger,
                arm,
            )
            self.assertFalse(layout.retirement.exists())

    def test_claim_without_completion_retires_and_tombstones(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = w12.W12Layout(Path(temporary))
            arm = {
                "arm_id": "bid-ownership",
                "ordinal": 1,
                "seed_namespace": "dev/ablation/bid-ownership",
            }
            attempt = (
                layout.arm(1, "bid-ownership")
                / "attempts"
                / "attempt-001"
            )
            attempt.mkdir(parents=True)
            ledger = {
                "consumed": [
                    {
                        "consumer": "runner:x:shard-000",
                        "index": 0,
                        "namespace": arm["seed_namespace"],
                        "seed": 1,
                        "sequence": 0,
                    }
                ],
                "ledger_fingerprint": _digest("ledger"),
            }
            with self.assertRaises(w12.W12OperatorError):
                w12._retire_if_claimed_incomplete(  # noqa: SLF001
                    layout,
                    {"protocol_fingerprint": _digest("protocol")},
                    ledger,
                    arm,
                )
            self.assertTrue(layout.retirement.is_file())
            self.assertTrue((attempt / "failure-tombstone.json").is_file())

    def test_out_of_order_claimed_completion_retires(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = w12.W12Layout(Path(temporary))
            arms = []
            for ordinal, arm_id in enumerate(enoch_week1.ABLATION_ARMS, start=1):
                comparison = {
                    "comparison_protocol_fingerprint": _digest(
                        f"comparison:{arm_id}"
                    ),
                    "seed_namespace": f"dev/ablation/{arm_id}",
                    "shards": [
                        {"seed_indices": [0], "shard_id": "shard-000"}
                    ],
                }
                arms.append(
                    {
                        "arm_id": arm_id,
                        "comparison": comparison,
                        "ordinal": ordinal,
                        "seed_namespace": comparison["seed_namespace"],
                    }
                )
            future = arms[1]
            execution = (
                layout.arm(future["ordinal"], future["arm_id"])
                / "attempts"
                / "attempt-001"
                / "execution"
            )
            execution.mkdir(parents=True)
            (execution / "execution-complete.json").write_text(
                "{}\n", encoding="utf-8"
            )
            prefix = future["comparison"]["comparison_protocol_fingerprint"][:16]
            ledger = {
                "consumed": [
                    {
                        "consumer": f"runner:{prefix}:shard-000",
                        "index": 0,
                        "namespace": future["seed_namespace"],
                        "seed": 1,
                        "sequence": 0,
                    }
                ],
                "ledger_fingerprint": _digest("ledger"),
            }
            enoch_week1.atomic_write_json(layout.base.ledger, ledger)
            declaration = {"arms": arms}
            protocol = {"protocol_fingerprint": _digest("protocol")}
            with mock.patch.object(w12.enoch_week1, "validate_seed_ledger"):
                with self.assertRaises(w12.W12OperatorError):
                    w12._scan_resume_frontier(  # noqa: SLF001
                        layout, protocol, declaration
                    )
            self.assertTrue(layout.retirement.is_file())
            self.assertTrue(
                (
                    execution.parent
                    / "failure-tombstone.json"
                ).is_file()
            )

    def test_cached_invalid_completion_retires(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = w12.W12Layout(Path(temporary))
            arm = {
                "arm_id": "bid-ownership",
                "comparison": {"seed_namespace": "dev/ablation/bid-ownership"},
                "ordinal": 1,
                "seed_namespace": "dev/ablation/bid-ownership",
            }
            execution = (
                layout.arm(1, "bid-ownership")
                / "attempts"
                / "attempt-001"
                / "execution"
            )
            execution.mkdir(parents=True)
            (execution / "execution-complete.json").write_text(
                "{}\n", encoding="utf-8"
            )
            ledger = {
                "consumed": [
                    {
                        "consumer": "runner:x:shard-000",
                        "index": 0,
                        "namespace": arm["seed_namespace"],
                        "seed": 1,
                        "sequence": 0,
                    }
                ],
                "ledger_fingerprint": _digest("ledger"),
            }
            enoch_week1.atomic_write_json(layout.base.ledger, ledger)
            state = {"protocol": {"protocol_fingerprint": _digest("protocol")}}
            ledger_patch = mock.patch.object(
                w12.enoch_week1, "validate_seed_ledger"
            )
            completion_patch = mock.patch.object(
                w12,
                "_validate_completed_arm",
                side_effect=w12.W12OperatorError("corrupt completion"),
            )
            with ledger_patch:
                with completion_patch:
                    with self.assertRaises(w12.W12OperatorError):
                        w12._run_arm(  # noqa: SLF001
                            layout,
                            Path(temporary),
                            state,
                            {},
                            {},
                            {},
                            arm,
                            operator_id="test",
                            base_environment={},
                        )
            self.assertTrue(layout.retirement.is_file())
            self.assertTrue(
                (execution.parent / "failure-tombstone.json").is_file()
            )

    def test_abandoned_preclaim_attempt_is_tombstoned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            arm_root = Path(temporary) / "arm"
            attempt = arm_root / "attempts" / "attempt-001"
            attempt.mkdir(parents=True)
            arm = {
                "arm_id": "bid-ownership",
                "seed_namespace": "dev/ablation/bid-ownership",
            }
            w12._seal_abandoned_preclaim_attempts(  # noqa: SLF001
                arm_root, arm
            )
            self.assertTrue((attempt / "preclaim-abandoned.json").is_file())

    def test_ranked_table_rejects_any_invalidating_failure(self) -> None:
        protocol = {
            "protocol_fingerprint": _digest("protocol"),
            "seed_registry_sha256": _digest("seeds"),
        }
        fixture = {"fixture": "sealed"}
        declaration, evidence = _declaration_and_evidence()
        evidence[enoch_week1.ABLATION_ARMS[0]]["merged_result"]["metrics"][
            "failure_counters"
        ]["illegal_action"] = 1
        fixture_patch = mock.patch.object(
            w12,
            "_fixture_binding",
            return_value={
                "failure_count": 0,
                "fixture_report_fingerprint": _digest("fixture"),
                "record_count": 31,
                "source_files_sha256": _digest("fixture-sources"),
            },
        )
        decision_patch = mock.patch.object(
            w12.enoch_week1,
            "validate_w1_3_advancement_decision",
            return_value=_digest("decision"),
        )
        with fixture_patch:
            with decision_patch:
                with self.assertRaises(w12.W12OperatorError):
                    w12.build_ranked_table(
                        protocol, declaration, fixture, evidence
                    )


if __name__ == "__main__":
    unittest.main()
