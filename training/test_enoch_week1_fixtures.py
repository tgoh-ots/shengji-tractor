#!/usr/bin/env python3

from __future__ import annotations

import copy
import unittest

from training import enoch_week1
from training.enoch_week1_fixtures import (
    ARM_FIXTURES,
    FixtureError,
    GLOBAL_FIXTURES,
    SOURCE_FILES,
    _assert_registry,
    _summary,
    validate_report,
)


class Week1FixtureTests(unittest.TestCase):
    def test_global_registry_covers_search_determinism_and_evaluator_sources(
        self,
    ) -> None:
        self.assertIn(
            (
                "strict-fixed-work-search-determinism",
                "shengji-core",
                "bot::search::tests::strict_fixed_work_search_repeats_cards_and_work_telemetry",
            ),
            GLOBAL_FIXTURES,
        )
        self.assertIn("core/examples/enoch_control_probe.rs", SOURCE_FILES)
        self.assertIn("core/examples/enoch_eval.rs", SOURCE_FILES)
        self.assertIn("training/enoch_control_probe_reference.patch", SOURCE_FILES)
        self.assertIn(
            (
                "bid-score-hash-order-determinism",
                "shengji-core",
                "bot::heuristics::tests::test_bid_strength_is_bit_stable_across_fresh_count_maps",
            ),
            GLOBAL_FIXTURES,
        )

    def test_fixture_map_exactly_covers_all_arms(self) -> None:
        _assert_registry()
        self.assertEqual(tuple(ARM_FIXTURES), tuple(enoch_week1.ABLATION_ARMS))

    def test_cargo_summary_rejects_missing_or_failed_results(self) -> None:
        self.assertEqual(
            _summary("test result: ok. 1 passed; 0 failed; 0 ignored"), (1, 0)
        )
        self.assertEqual(
            _summary(
                "test result: ok. 1 passed; 0 failed;\n"
                "test result: ok. 0 passed; 0 failed;"
            ),
            (1, 0),
        )
        with self.assertRaises(FixtureError):
            _summary("no tests here")

    def test_report_fingerprint_and_coverage_fail_closed(self) -> None:
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
                "scope": scope,
                "fixture_id": fixture_id,
                "log_path": f"logs/{index:03d}-{scope}-{fixture_id}.log",
                "output_sha256": "2" * 64,
                "package": package,
                "passed": 1,
                "sequence": index,
                "test_name": test_name,
            }
            for index, (scope, fixture_id, package, test_name) in enumerate(cases)
        ]
        body = {
            "arm_registry_sha256": enoch_week1.ARM_REGISTRY_SHA256,
            "automatic_production_promotion_allowed": False,
            "failure_count": 0,
            "manifest_kind": "enoch-week1-fixture-report",
            "manifest_version": 1,
            "records": records,
            "records_sha256": enoch_week1.canonical_json_sha256(records),
            "source_files": [{"path": "core/src/bot/enoch.rs", "sha256": "1" * 64}],
            "source_files_sha256": enoch_week1.canonical_json_sha256(
                [{"path": "core/src/bot/enoch.rs", "sha256": "1" * 64}]
            ),
        }
        report = {
            **body,
            "fixture_report_fingerprint": enoch_week1.canonical_json_sha256(body),
        }
        self.assertEqual(validate_report(report), report["fixture_report_fingerprint"])
        corrupted = copy.deepcopy(report)
        corrupted["failure_count"] = 1
        with self.assertRaises(FixtureError):
            validate_report(corrupted)
        wrong_test = copy.deepcopy(report)
        wrong_test["records"][0]["test_name"] = "invented-test"
        wrong_test["records_sha256"] = enoch_week1.canonical_json_sha256(
            wrong_test["records"]
        )
        body = dict(wrong_test)
        body.pop("fixture_report_fingerprint")
        wrong_test["fixture_report_fingerprint"] = enoch_week1.canonical_json_sha256(
            body
        )
        with self.assertRaisesRegex(FixtureError, "identity or order"):
            validate_report(wrong_test)


if __name__ == "__main__":
    unittest.main()
