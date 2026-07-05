#!/usr/bin/env python3

from __future__ import annotations

import contextlib
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from training import enoch_week1
from training import enoch_week1_operator as operator
from training import enoch_week1_runner


class Week1OperatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = enoch_week1.build_protocol(0x0A11_CE55)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.layout = operator.RunLayout(Path(self.temporary.name) / "run")
        enoch_week1.atomic_write_json(self.layout.protocol, self.protocol)
        enoch_week1.atomic_write_json(
            self.layout.ledger, enoch_week1.new_seed_ledger(self.protocol)
        )
        self.evaluator_identity = {
            "binary_sha256": "1" * 64,
            "configuration_sha256": enoch_week1.canonical_json_sha256(
                enoch_week1.WEEK1_EVALUATOR_CONTRACT
            ),
            "source_sha256": "2" * 64,
        }
        self.control = {
            "evaluator_identity": self.evaluator_identity,
            "search_knobs": {
                "enoch-0": {
                    "budget_ms": 2_200,
                    "max_candidates": 6,
                    "max_worlds": 144,
                    "rollout_tricks": 12,
                }
            },
        }

    def _declarations(
        self, *, available_parallelism: int = 10, maximum_workers: int = 8
    ) -> dict[str, dict[str, object]]:
        return operator._declarations(
            self.protocol,
            self.layout,
            self.control,
            {},
            available_parallelism,
            maximum_workers,
        )

    def test_product_smoke_ramp_is_one_four_eight_without_writing(self) -> None:
        declarations = self._declarations()
        self.assertEqual(
            [
                len(declarations[namespace]["comparison"]["shards"])
                for namespace, _, _, _ in operator.SMOKE_SPECS
            ],
            [1, 4, 8],
        )
        self.assertEqual(
            [
                declarations[namespace]["workers"]
                for namespace, _, _, _ in operator.SMOKE_SPECS
            ],
            [1, 4, 8],
        )
        self.assertFalse((self.layout.w1_1 / "smokes").exists())

    def test_worker_ramp_caps_at_available_parallelism(self) -> None:
        declarations = self._declarations(
            available_parallelism=3, maximum_workers=8
        )
        self.assertEqual(
            [
                declarations[namespace]["workers"]
                for namespace, _, _, _ in operator.SMOKE_SPECS
            ],
            [1, 3, 3],
        )

    def test_search_authority_uses_an_isolated_run_root_cargo_target(self) -> None:
        expected = str(self.layout.authority_target.resolve())
        authority = {"cargo_target_dir": expected}
        with mock.patch.object(
            operator.enoch_week1_preflight,
            "build_deterministic_search_fixture_authority",
            return_value=authority,
        ) as build:
            actual = operator._ensure_authority(
                self.protocol,
                self.layout,
                Path("/private/tmp/source-worktree"),
                123.0,
            )

        self.assertEqual(actual, authority)
        self.assertEqual(
            build.call_args.kwargs["cargo_target_dir"],
            self.layout.authority_target.resolve(),
        )
        self.assertEqual(
            enoch_week1.load_json_object(self.layout.authority),
            authority,
        )

    def test_cached_search_authority_rejects_a_different_cargo_target(self) -> None:
        enoch_week1.atomic_write_json(
            self.layout.authority,
            {"cargo_target_dir": str(self.layout.root / "wrong-target")},
        )
        with mock.patch.object(
            operator.enoch_week1_preflight,
            "validate_deterministic_search_fixture_authority",
        ), self.assertRaisesRegex(
            operator.OperatorError, "different Cargo target"
        ):
            operator._ensure_authority(
                self.protocol,
                self.layout,
                Path("/private/tmp/source-worktree"),
                123.0,
            )

    def test_source_provenance_rejects_workspace_drift(self) -> None:
        control = {
            "evaluator_identity": {"source_sha256": "3" * 64},
            "production_reference": "c813c8a",
        }
        source = {
            "git_commit": "4" * 40,
            "git_tree": "5" * 40,
            "operator_source_path": "training/enoch_week1_operator.py",
            "operator_source_sha256": "6" * 64,
        }
        with mock.patch.object(
            operator.enoch_week1,
            "validate_w1_0_control_manifest",
            return_value="7" * 64,
        ):
            provenance = operator._build_provenance(
                self.protocol, control, source
            )
            self.assertEqual(
                operator._validate_provenance(
                    self.protocol, control, provenance, source
                ),
                provenance["source_provenance_fingerprint"],
            )
            drifted = dict(source)
            drifted["git_commit"] = "8" * 40
            with self.assertRaisesRegex(
                operator.OperatorError, "workspace differs"
            ):
                operator._validate_provenance(
                    self.protocol, control, provenance, drifted
                )

    def test_missing_completion_with_claimed_seed_fails_closed(self) -> None:
        enoch_week1.consume_seed_once(
            self.layout.ledger,
            self.protocol,
            "smoke/product/001",
            0,
            "interrupted-test",
        )
        with self.assertRaisesRegex(operator.OperatorError, "start a fresh protocol"):
            operator._require_unconsumed(
                self.protocol,
                self.layout,
                {"smoke/product/001"},
                "one-pair smoke",
            )

    def test_probe_failure_writes_nothing_and_consumes_no_seed(self) -> None:
        declared = self._declarations()["smoke/product/001"]
        token = object()
        active = False

        @contextlib.contextmanager
        def locked(*_args: object):
            nonlocal active
            active = True
            try:
                yield token
            finally:
                active = False

        def failed_probe(**kwargs: object) -> dict[str, object]:
            self.assertTrue(active)
            self.assertIs(kwargs["campaign_lock_token"], token)
            raise enoch_week1_runner.RunnerError("parallelism mismatch")

        with mock.patch.object(
            operator.enoch_week1_runner,
            "authoritative_campaign_lock",
            side_effect=locked,
        ), mock.patch.object(
            operator.enoch_week1_runner,
            "probe_evaluator_environment_identity",
            side_effect=failed_probe,
        ):
            with self.assertRaisesRegex(
                enoch_week1_runner.RunnerError, "parallelism mismatch"
            ):
                operator._run_smoke(
                    self.protocol,
                    self.layout,
                    declared,
                    operator_id="operator-test",
                    available_parallelism=10,
                    shard_timeout=60.0,
                    base_environment={},
                )

        self.assertFalse(self.layout.smoke("001").exists())
        self.assertEqual(
            enoch_week1.load_json_object(self.layout.ledger)["consumed"], []
        )

    def test_probe_attestation_evidence_and_runs_share_live_token(self) -> None:
        declared = self._declarations()["smoke/product/001"]
        token = object()
        active = False
        events: list[str] = []
        start = datetime(2026, 7, 4, 10, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 7, 4, 10, 0, 1, tzinfo=timezone.utc)
        attested = datetime(2026, 7, 4, 10, 0, 2, tzinfo=timezone.utc)

        @contextlib.contextmanager
        def locked(*_args: object):
            nonlocal active
            active = True
            events.append("lock-enter")
            try:
                yield token
            finally:
                events.append("lock-exit")
                active = False

        def probe(**kwargs: object) -> dict[str, object]:
            self.assertTrue(active)
            self.assertIs(kwargs["campaign_lock_token"], token)
            self.assertFalse(self.layout.smoke("001").exists())
            events.append("probe")
            environment = declared["environment_identity"]
            return {
                "environment": environment,
                "environment_identity_sha256": enoch_week1.canonical_json_sha256(
                    environment
                ),
            }

        def build_evidence(*_args: object, **_kwargs: object) -> dict[str, object]:
            self.assertTrue(active)
            self.assertTrue(
                (self.layout.smoke("001") / "attempt-001" / "machine-attestation.json").is_file()
            )
            events.append("evidence")
            return {"verified": True}

        def run_comparison(**kwargs: object) -> dict[str, object]:
            self.assertTrue(active)
            self.assertIs(kwargs["campaign_lock_token"], token)
            event = "dry-run" if kwargs["dry_run"] else "real-run"
            events.append(event)
            return {"event": event}

        def validate_completed(*_args: object, **_kwargs: object) -> dict[str, object]:
            self.assertTrue(active)
            events.append("validate-complete")
            return {"complete": True}

        with mock.patch.object(
            operator.enoch_week1_runner,
            "authoritative_campaign_lock",
            side_effect=locked,
        ), mock.patch.object(
            operator.enoch_week1_runner,
            "probe_evaluator_environment_identity",
            side_effect=probe,
        ), mock.patch.object(
            operator.enoch_week1_evidence,
            "build_verified_external_evidence",
            side_effect=build_evidence,
        ), mock.patch.object(
            operator.enoch_week1_evidence,
            "validate_machine_contention_attestation",
            return_value="a" * 64,
        ), mock.patch.object(
            operator.enoch_week1_evidence,
            "validate_verified_external_evidence",
            return_value={},
        ), mock.patch.object(
            operator.enoch_week1_runner,
            "run_comparison",
            side_effect=run_comparison,
        ), mock.patch.object(
            operator, "_validate_completed_smoke", side_effect=validate_completed
        ), mock.patch.object(
            operator, "_utc_now", side_effect=[start, attested]
        ), mock.patch.object(
            operator, "_finish_observation", return_value=end
        ):
            result = operator._run_smoke(
                self.protocol,
                self.layout,
                declared,
                operator_id="operator-test",
                available_parallelism=10,
                shard_timeout=60.0,
                base_environment={},
            )

        self.assertEqual(result, {"complete": True})
        self.assertEqual(
            events,
            [
                "lock-enter",
                "probe",
                "evidence",
                "dry-run",
                "real-run",
                "validate-complete",
                "lock-exit",
            ],
        )

    def test_run_w1_1_orders_fixtures_before_first_seed_consuming_stage(self) -> None:
        (self.layout.control_bundle).mkdir(parents=True)
        enoch_week1.atomic_write_json(
            self.layout.control_bundle / "control-manifest.json", {}
        )
        enoch_week1.atomic_write_json(self.layout.provenance, {})
        source_identity = {
            "git_commit": "a" * 40,
            "git_tree": "b" * 40,
            "operator_source_path": "training/enoch_week1_operator.py",
            "operator_source_sha256": "c" * 64,
        }
        stages: list[str] = []
        declarations = {
            namespace: {"namespace": namespace}
            for namespace, _, _, _ in operator.SMOKE_SPECS
        }
        report = {
            "baseline_worker_report_fingerprint": "d" * 64,
            "fixed_worker_configuration": {},
        }
        phase = {"phase_manifest_fingerprint": "e" * 64}

        def authority(*_args: object, **_kwargs: object) -> dict[str, object]:
            stages.append("authority")
            return {}

        def fixtures(*_args: object, **_kwargs: object) -> dict[str, object]:
            stages.append("fixtures")
            return {}

        def preflight(*_args: object, **_kwargs: object) -> dict[str, object]:
            stages.append("preflight")
            return {}

        def smoke(*_args: object, **_kwargs: object) -> dict[str, object]:
            stages.append("smoke")
            return {}

        with mock.patch.object(operator, "_require_safe_run_root"), mock.patch.object(
            operator, "_git_identity", return_value=source_identity
        ), mock.patch.object(
            operator, "_operator_lock", return_value=contextlib.nullcontext()
        ), mock.patch.object(
            operator.enoch_week1_freeze, "verify_bundle", return_value="f" * 64
        ), mock.patch.object(
            operator, "_validate_provenance", return_value="1" * 64
        ), mock.patch.object(operator, "_seal_phase0"), mock.patch.object(
            operator, "_verify_frozen_source"
        ), mock.patch.object(
            operator, "_ensure_authority", side_effect=authority
        ), mock.patch.object(
            operator, "_ensure_fixtures", side_effect=fixtures
        ), mock.patch.object(
            operator, "_ensure_full_preflight", side_effect=preflight
        ), mock.patch.object(
            operator, "_declarations", return_value=declarations
        ), mock.patch.object(
            operator, "_run_smoke", side_effect=smoke
        ), mock.patch.object(
            operator, "_smoke_evidence_from_disk", return_value={}
        ), mock.patch.object(
            operator, "_seal_report_and_phase1", return_value=(report, phase)
        ):
            result = operator.run_w1_1(
                self.layout,
                Path(self.temporary.name),
                operator_id="operator-test",
                attest_no_machine_contention=True,
                available_parallelism=10,
                environment={},
            )

        self.assertEqual(stages[:3], ["authority", "fixtures", "preflight"])
        self.assertEqual(stages.count("smoke"), 3)
        self.assertEqual(result["baseline_worker_report_fingerprint"], "d" * 64)
        self.assertEqual(
            enoch_week1.load_json_object(self.layout.ledger)["consumed"], []
        )


if __name__ == "__main__":
    unittest.main()
