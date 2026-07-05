from __future__ import annotations

import copy
import hashlib
import unittest

from training import (
    enoch_week1,
    enoch_week1_campaign,
    enoch_week1_fixtures,
    enoch_week1_runner,
)


class Week1CampaignLineageTests(unittest.TestCase):
    ARMS = ["bid-ownership", "compound-follow"]

    @staticmethod
    def _hash(label: str) -> str:
        return hashlib.sha256(label.encode("utf-8")).hexdigest()

    @classmethod
    def _fixture_report(cls) -> dict:
        records = []
        for sequence, (scope, fixture_id, package, test_name) in enumerate(
            enoch_week1_fixtures._fixture_cases()
        ):
            command = [
                "cargo",
                "test",
                "--locked",
                "-p",
                package,
                test_name,
                "--",
                "--exact",
                "--test-threads=1",
            ]
            records.append(
                {
                    "command": command,
                    "exit_code": 0,
                    "failed": 0,
                    "fixture_id": fixture_id,
                    "log_path": f"logs/{sequence:03d}-{scope}-{fixture_id}.log",
                    "output_sha256": cls._hash(f"fixture-output:{sequence}"),
                    "package": package,
                    "passed": 1,
                    "scope": scope,
                    "sequence": sequence,
                    "test_name": test_name,
                }
            )
        source_files = [
            {"path": "core/src/bot/search.rs", "sha256": cls._hash("source")}
        ]
        body = {
            "arm_registry_sha256": enoch_week1.ARM_REGISTRY_SHA256,
            "automatic_production_promotion_allowed": False,
            "failure_count": 0,
            "manifest_kind": enoch_week1_fixtures.REPORT_KIND,
            "manifest_version": enoch_week1_fixtures.MANIFEST_VERSION,
            "records": records,
            "records_sha256": enoch_week1.canonical_json_sha256(records),
            "source_files": source_files,
            "source_files_sha256": enoch_week1.canonical_json_sha256(source_files),
        }
        return {
            **body,
            "fixture_report_fingerprint": enoch_week1.canonical_json_sha256(body),
        }

    @classmethod
    def _record(cls, comparison: dict, index: int, level: float = 0.2) -> dict:
        namespace = next(
            entry
            for entry in cls.protocol["seed_registry"]["namespaces"]
            if entry["name"] == comparison["seed_namespace"]
        )
        seed = namespace["seeds"][index]
        return {
            "candidate_completed_worlds": 4,
            "candidate_latency_ms": 10.0,
            "complete": True,
            "control_completed_worlds": 4,
            "control_latency_ms": 10.0,
            "effective_deal_seed": seed,
            "failure_counters": {
                name: 0 for name in enoch_week1.FAILURE_COUNTER_NAMES
            },
            "level_utility_delta": level,
            "orientations_completed": 2,
            "point_margin_delta": level,
            "seed": seed,
            "seed_index": index,
            "style_metrics": {
                name: 0.25 for name in comparison["required_style_metrics"]
            },
            "win_rate_delta": level / 2.0,
        }

    @classmethod
    def _merged(cls, comparison: dict, level: float = 0.2) -> dict:
        shards = []
        for assignment in comparison["shards"]:
            records = [
                cls._record(comparison, index, level)
                for index in assignment["seed_indices"]
            ]
            shards.append(
                enoch_week1.build_shard_result(
                    cls.protocol,
                    comparison,
                    assignment["shard_id"],
                    records,
                    verified_external_evidence_fingerprint=cls._hash(
                        "verified-external-evidence"
                    ),
                )
            )
        return enoch_week1.merge_shard_results(cls.protocol, comparison, shards)

    @classmethod
    def _comparison(
        cls,
        *,
        phase: str,
        comparison_id: str,
        subject_id: str,
        namespace: str,
        pair_count: int,
        launch: dict,
        identities: dict,
        development_rule: dict | None = None,
    ) -> dict:
        return enoch_week1.build_comparison_protocol_manifest(
            cls.protocol,
            phase=phase,
            comparison_id=comparison_id,
            subject_id=subject_id,
            seed_namespace=namespace,
            pair_count=pair_count,
            shard_count=1,
            candidate_fingerprint=enoch_week1.canonical_json_sha256(
                identities["candidate"]
            ),
            control_fingerprint=enoch_week1.canonical_json_sha256(
                identities["control"]
            ),
            evaluator_fingerprint=enoch_week1.canonical_json_sha256(
                identities["evaluator"]
            ),
            environment_fingerprint=cls.environment_fingerprint,
            configuration_fingerprint=enoch_week1.canonical_json_sha256(launch),
            development_rule=development_rule,
        )

    @classmethod
    def _survivor_evidence(cls, arm_id: str, *, survivor_level: float = 0.2) -> dict:
        scenario = (
            enoch_week1_runner.DEVELOPMENT_FRIEND_SCENARIO
            if arm_id == "friend-revelation"
            else "standard"
        )
        launch = enoch_week1_runner.build_launch_configuration(
            candidate_arm_ids=[arm_id],
            worlds=4,
            candidates=4,
            rollout_tricks=2,
            scenario_id=scenario,
            deadline_ms=30_000,
        )
        identities = enoch_week1_runner.build_in_process_identity_bindings(
            cls.evaluator_identity, launch
        )
        rule = cls.development_rule
        ablation = cls._comparison(
            phase="W1.2",
            comparison_id=f"ablation-{arm_id}",
            subject_id=arm_id,
            namespace=f"dev/ablation/{arm_id}",
            pair_count=200,
            launch=launch,
            identities=identities,
            development_rule=rule,
        )
        ablation_merged = cls._merged(ablation)
        advancement = enoch_week1.build_w1_3_advancement_decision(
            cls.protocol,
            ablation,
            ablation_merged,
            fixture_report=cls.fixture_report,
        )
        survivor = cls._comparison(
            phase="W1.3",
            comparison_id=f"survivor-{arm_id}",
            subject_id=arm_id,
            namespace=f"dev/survivor/{arm_id}",
            pair_count=800,
            launch=launch,
            identities=identities,
            development_rule=rule,
        )
        return {
            "ablation": {
                "comparison": ablation,
                "identity_bindings": identities,
                "launch_configuration": launch,
                "merged_result": ablation_merged,
            },
            "advancement_decision": advancement,
            "arm_id": arm_id,
            "fixture_report": cls.fixture_report,
            "survivor_screen": {
                "comparison": survivor,
                "identity_bindings": identities,
                "launch_configuration": launch,
                "merged_result": cls._merged(survivor, survivor_level),
            },
        }

    @classmethod
    def _build_artifact(
        cls,
        *,
        qualification_comparison: dict | None = None,
        qualification_launch: dict | None = None,
        qualification_identities: dict | None = None,
        screen_comparison: dict | None = None,
        screen_launch: dict | None = None,
        screen_identities: dict | None = None,
        survivor_evidence: list[dict] | None = None,
    ) -> dict:
        return enoch_week1_campaign.build_w1_4_campaign_lineage(
            cls.protocol,
            qualification_comparison=qualification_comparison
            or cls.qualification_comparison,
            qualification_launch_configuration=qualification_launch
            or cls.combination_launch,
            qualification_identity_bindings=qualification_identities
            or cls.combination_identities,
            screen_comparison=screen_comparison or cls.screen_comparison,
            screen_launch_configuration=screen_launch or cls.combination_launch,
            screen_identity_bindings=screen_identities or cls.combination_identities,
            survivor_evidence=survivor_evidence or cls.survivors,
        )

    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = enoch_week1.build_protocol(0xCA4A_2026_0703)
        cls.environment_fingerprint = cls._hash("clean-environment")
        cls.evaluator_identity = enoch_week1.build_frozen_evaluator_identity(
            source_sha256=cls._hash("evaluator-source"),
            binary_sha256=cls._hash("evaluator-binary"),
            configuration_sha256=cls._hash("evaluator-configuration"),
        )
        cls.development_rule = enoch_week1.build_development_rule(
            "independent-survivor",
            minimum_level_utility_estimate=0.0,
            minimum_point_margin_estimate=0.0,
            minimum_win_rate_estimate=0.0,
            maximum_candidate_p95_latency_ms=20.0,
            minimum_candidate_completed_worlds_mean=2.0,
        )
        cls.fixture_report = cls._fixture_report()
        cls.survivors = [cls._survivor_evidence(arm) for arm in cls.ARMS]

        cls.combination_launch = enoch_week1_runner.build_launch_configuration(
            candidate_arm_ids=cls.ARMS,
            worlds=4,
            candidates=4,
            rollout_tricks=2,
            deadline_ms=30_000,
        )
        cls.combination_identities = (
            enoch_week1_runner.build_in_process_identity_bindings(
                cls.evaluator_identity, cls.combination_launch
            )
        )
        cls.qualification_comparison = cls._comparison(
            phase="W1.4",
            comparison_id="combination-qualification",
            subject_id="enoch-1-candidate",
            namespace="dev/combination/qualification",
            pair_count=250,
            launch=cls.combination_launch,
            identities=cls.combination_identities,
            development_rule=cls.development_rule,
        )
        cls.screen_comparison = cls._comparison(
            phase="W1.4",
            comparison_id="combination-screen",
            subject_id="enoch-1-candidate",
            namespace="dev/combination/screen",
            pair_count=800,
            launch=cls.combination_launch,
            identities=cls.combination_identities,
            development_rule=cls.development_rule,
        )
        cls.artifact = cls._build_artifact()
        cls.combination_qualification_merged = cls._merged(
            cls.qualification_comparison
        )
        cls.combination_screen_merged = cls._merged(cls.screen_comparison)
        cls.candidate_decision = (
            enoch_week1_campaign.build_w1_4_candidate_decision(
                cls.protocol,
                cls.artifact,
                qualification_merged_result=cls.combination_qualification_merged,
                screen_merged_result=cls.combination_screen_merged,
            )
        )

    def test_valid_campaign_is_self_contained_and_canonical(self) -> None:
        fingerprint = enoch_week1_campaign.validate_w1_4_campaign_lineage(
            self.protocol, self.artifact
        )
        self.assertEqual(
            fingerprint,
            self.artifact["w1_4_campaign_lineage_fingerprint"],
        )
        self.assertEqual(self.artifact["candidate_arm_ids"], self.ARMS)
        self.assertEqual(
            self._build_artifact()["w1_4_campaign_lineage_fingerprint"], fingerprint
        )

    def test_rejects_different_candidate_across_the_two_stages(self) -> None:
        one_arm_launch = enoch_week1_runner.build_launch_configuration(
            candidate_arm_ids=[self.ARMS[0]],
            worlds=4,
            candidates=4,
            rollout_tricks=2,
            deadline_ms=30_000,
        )
        one_arm_identities = enoch_week1_runner.build_in_process_identity_bindings(
            self.evaluator_identity, one_arm_launch
        )
        one_arm_screen = self._comparison(
            phase="W1.4",
            comparison_id="combination-screen-one-arm",
            subject_id="enoch-1-candidate",
            namespace="dev/combination/screen",
            pair_count=800,
            launch=one_arm_launch,
            identities=one_arm_identities,
        )
        with self.assertRaisesRegex(
            enoch_week1.ProtocolError, "identical candidate arm set"
        ):
            self._build_artifact(
                screen_comparison=one_arm_screen,
                screen_launch=one_arm_launch,
                screen_identities=one_arm_identities,
            )

    def test_rejects_non_fixed_work_w1_4_stage(self) -> None:
        budget_launch = enoch_week1_runner.build_launch_configuration(
            candidate_arm_ids=self.ARMS,
            worlds=4,
            candidates=4,
            rollout_tricks=2,
            budget_ms=100,
        )
        budget_identities = enoch_week1_runner.build_in_process_identity_bindings(
            self.evaluator_identity, budget_launch
        )
        budget_qualification = self._comparison(
            phase="W1.4",
            comparison_id="combination-qualification-budget",
            subject_id="enoch-1-candidate",
            namespace="dev/combination/qualification",
            pair_count=250,
            launch=budget_launch,
            identities=budget_identities,
        )
        with self.assertRaisesRegex(enoch_week1.ProtocolError, "fixed-work"):
            self._build_artifact(
                qualification_comparison=budget_qualification,
                qualification_launch=budget_launch,
                qualification_identities=budget_identities,
            )

    def test_rejects_missing_or_unsupported_independent_survivor(self) -> None:
        with self.assertRaisesRegex(enoch_week1.ProtocolError, "exactly match"):
            self._build_artifact(survivor_evidence=self.survivors[:1])

        unsupported = copy.deepcopy(self.survivors)
        unsupported[1] = self._survivor_evidence(
            self.ARMS[1], survivor_level=-0.5
        )
        with self.assertRaisesRegex(
            enoch_week1.ProtocolError, "not an independently supported"
        ):
            self._build_artifact(survivor_evidence=unsupported)

    def test_rejects_reused_stage_or_tampered_nested_evidence(self) -> None:
        with self.assertRaisesRegex(enoch_week1.ProtocolError, "wrong seed namespace"):
            self._build_artifact(screen_comparison=self.qualification_comparison)

        tampered = copy.deepcopy(self.artifact)
        tampered["survivor_evidence"][0]["survivor_screen"]["merged_result"][
            "records"
        ][0]["level_utility_delta"] = 99.0
        body = dict(tampered)
        body.pop("w1_4_campaign_lineage_fingerprint")
        tampered["w1_4_campaign_lineage_fingerprint"] = (
            enoch_week1.canonical_json_sha256(body)
        )
        with self.assertRaisesRegex(
            enoch_week1.ProtocolError, "raw-record hash|statistics"
        ):
            enoch_week1_campaign.validate_w1_4_campaign_lineage(
                self.protocol, tampered
            )

    def test_w1_5_requires_and_cross_binds_this_exact_campaign(self) -> None:
        configuration_fingerprints = {
            entry["namespace"]: self._hash(
                f"qualification-configuration:{entry['namespace']}"
            )
            for entry in enoch_week1.QUALIFICATION_MATRIX
        }
        arguments = {
            "candidate_fingerprint": self.artifact["candidate_fingerprint"],
            "control_fingerprint": self.artifact["control_fingerprint"],
            "evaluator_fingerprint": self.artifact["evaluator_fingerprint"],
            "environment_fingerprint": self.artifact["environment_fingerprint"],
            "configuration_fingerprints": configuration_fingerprints,
            "serving_envelope": {
                "candidate_completed_worlds_mean_min": 2.0,
                "candidate_p50_latency_ms_max": 20.0,
                "candidate_p95_latency_ms_max": 20.0,
            },
            "intended_equal_byte_identical": False,
            "shard_count": 1,
        }
        with self.assertRaisesRegex(
            enoch_week1.ProtocolError, "requires the eligible W1.4"
        ):
            enoch_week1.build_w1_5_qualification_manifest(
                self.protocol, **arguments
            )

        mismatched_arguments = dict(arguments)
        mismatched_arguments["candidate_fingerprint"] = self._hash(
            "different-qualified-candidate"
        )
        with self.assertRaisesRegex(
            enoch_week1.ProtocolError, "does not match the eligible W1.4"
        ):
            enoch_week1.build_w1_5_qualification_manifest(
                self.protocol,
                **mismatched_arguments,
                w1_4_candidate_decision=self.candidate_decision,
            )

        manifest = enoch_week1.build_w1_5_qualification_manifest(
            self.protocol,
            **arguments,
            w1_4_candidate_decision=self.candidate_decision,
        )
        self.assertEqual(
            manifest["w1_4_candidate_decision_fingerprint"],
            self.candidate_decision["w1_4_candidate_decision_fingerprint"],
        )

    def test_candidate_decision_requires_results_and_rejects_regression(self) -> None:
        with self.assertRaisesRegex(
            enoch_week1.ProtocolError, "requires both exact merged results"
        ):
            enoch_week1_campaign.build_w1_4_candidate_decision(
                self.protocol,
                self.artifact,
                qualification_merged_result=None,
                screen_merged_result=self.combination_screen_merged,
            )

        regressed_screen = self._merged(self.screen_comparison, level=-0.2)
        rejected = enoch_week1_campaign.build_w1_4_candidate_decision(
            self.protocol,
            self.artifact,
            qualification_merged_result=self.combination_qualification_merged,
            screen_merged_result=regressed_screen,
        )
        self.assertEqual(rejected["decision"], "reject-candidate")
        self.assertIn(
            "screen:combination-level-utility-not-positive", rejected["reasons"]
        )
        with self.assertRaisesRegex(
            enoch_week1.ProtocolError, "requires an eligible W1.4"
        ):
            enoch_week1.build_w1_5_qualification_manifest(
                self.protocol,
                candidate_fingerprint=rejected["candidate_fingerprint"],
                control_fingerprint=rejected["control_fingerprint"],
                evaluator_fingerprint=rejected["evaluator_fingerprint"],
                environment_fingerprint=rejected["environment_fingerprint"],
                configuration_fingerprints={
                    entry["namespace"]: self._hash(
                        f"rejected-configuration:{entry['namespace']}"
                    )
                    for entry in enoch_week1.QUALIFICATION_MATRIX
                },
                serving_envelope={
                    "candidate_completed_worlds_mean_min": 2.0,
                    "candidate_p50_latency_ms_max": 20.0,
                    "candidate_p95_latency_ms_max": 20.0,
                },
                intended_equal_byte_identical=False,
                w1_4_candidate_decision=rejected,
            )

    def test_candidate_decision_reconstructs_interaction_and_rejects_forgery(self) -> None:
        fingerprint = enoch_week1_campaign.validate_w1_4_candidate_decision(
            self.protocol, self.candidate_decision
        )
        self.assertEqual(
            fingerprint,
            self.candidate_decision["w1_4_candidate_decision_fingerprint"],
        )
        interaction = self.candidate_decision["interaction_diagnostic"]
        self.assertAlmostEqual(
            interaction["screen_minus_individual_sum"],
            interaction["combination_screen_level_utility_estimate"]
            - interaction["individual_survivor_level_utility_sum"],
        )
        forged = copy.deepcopy(
            enoch_week1_campaign.build_w1_4_candidate_decision(
                self.protocol,
                self.artifact,
                qualification_merged_result=self.combination_qualification_merged,
                screen_merged_result=self._merged(self.screen_comparison, level=-0.2),
            )
        )
        forged["decision"] = "eligible-for-qualification"
        body = dict(forged)
        body.pop("w1_4_candidate_decision_fingerprint")
        forged["w1_4_candidate_decision_fingerprint"] = (
            enoch_week1.canonical_json_sha256(body)
        )
        with self.assertRaisesRegex(
            enoch_week1.ProtocolError, "does not reconstruct"
        ):
            enoch_week1_campaign.validate_w1_4_candidate_decision(
                self.protocol, forged
            )


if __name__ == "__main__":
    unittest.main()
