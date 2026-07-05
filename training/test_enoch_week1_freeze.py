#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from training.enoch_week1_freeze import (
    CONTROL_PROBE_ARTIFACT_IDS,
    FreezeError,
    _build_control_probe,
    _build_current_evaluator,
    _bundle_artifact_path,
    _hardware_summary,
    _inject_control_probe,
    _safe_environment,
    _snapshot_sources,
    verify_bundle,
)


class Week1FreezeTests(unittest.TestCase):
    def test_hardware_summary_records_numeric_memory_without_host_identity(self) -> None:
        summary = _hardware_summary()
        fields = dict(item.split("=", 1) for item in summary.split(";"))
        self.assertEqual(
            set(fields),
            {"machine", "processor", "logical_cpus", "memory_bytes"},
        )
        self.assertGreater(int(fields["logical_cpus"]), 0)
        self.assertGreater(int(fields["memory_bytes"]), 0)

    def test_build_environment_drops_every_experiment_knob(self) -> None:
        environment = _safe_environment(
            {
                "PATH": "/usr/bin:/bin",
                "HOME": "/tmp/home",
                "LANG": "en_US.UTF-8",
                "SHENGJI_SEARCH_PUCT": "1",
                "GM_WORLDS": "999",
                "OMNI_PRIOR": "net",
                "GEN_WORKERS": "99",
            }
        )
        self.assertEqual(environment["LANG"], "C")
        self.assertEqual(environment["LC_ALL"], "C")
        self.assertFalse(
            any(
                name.startswith(("SHENGJI_", "GM_", "OMNI_", "GEN_"))
                for name in environment
            )
        )

    def test_build_environment_requires_path(self) -> None:
        with self.assertRaises(FreezeError):
            _safe_environment({"HOME": "/tmp/home"})

    def test_evaluator_source_identity_is_content_deterministic(self) -> None:
        workspace = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_hash, first_records = _snapshot_sources(workspace, root / "first.tar.gz")
            second_hash, second_records = _snapshot_sources(workspace, root / "second.tar.gz")
        self.assertEqual(first_hash, second_hash)
        self.assertEqual(first_records, second_records)
        self.assertTrue(
            any(record["path"] == "core/examples/enoch_eval.rs" for record in first_records)
        )
        self.assertTrue(
            any(
                record["path"] == "core/examples/enoch_control_probe.rs"
                for record in first_records
            )
        )
        self.assertTrue(all(len(record["sha256"]) == 64 for record in first_records))

    def test_probe_source_is_frozen_then_injected_byte_for_byte(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            reference = root / "reference"
            probe = workspace / "core" / "examples" / "enoch_control_probe.rs"
            probe.parent.mkdir(parents=True)
            probe.write_bytes(b"fn main() {}\n")
            reference.mkdir()

            digest = _inject_control_probe(workspace, reference, root / "bundle")

            frozen = root / "bundle" / "source" / "enoch-control-probe.rs"
            injected = reference / "core" / "examples" / "enoch_control_probe.rs"
            self.assertEqual(frozen.read_bytes(), probe.read_bytes())
            self.assertEqual(injected.read_bytes(), probe.read_bytes())
            self.assertEqual(len(digest), 64)

    def test_probe_source_injection_requires_the_current_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FreezeError, "required control probe source"):
                _inject_control_probe(root / "workspace", root / "reference", root / "bundle")

    def test_probe_build_uses_its_own_target_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            target = root / "probe-target"
            with mock.patch("training.enoch_week1_freeze._run") as run:
                binary = _build_control_probe(
                    source,
                    target,
                    {"PATH": "/usr/bin:/bin", "LANG": "C"},
                )
        self.assertEqual(binary, target / "release" / "examples" / "enoch_control_probe")
        command = run.call_args.args[0]
        self.assertEqual(command[-2:], ("--example", "enoch_control_probe"))
        self.assertEqual(run.call_args.kwargs["cwd"], source)
        self.assertEqual(
            run.call_args.kwargs["environment"]["CARGO_TARGET_DIR"], str(target)
        )

    def test_current_evaluator_requires_the_strict_test_to_execute(self) -> None:
        marker = (
            "test bot::search::tests::"
            "strict_search_rejects_a_zero_sample_prior_fallback ... ok"
        )
        workspace = Path("/workspace")
        target = Path("/target")
        with mock.patch(
            "training.enoch_week1_freeze._run", side_effect=["build ok", marker]
        ) as run:
            binary, output = _build_current_evaluator(
                workspace,
                target,
                {"PATH": "/usr/bin:/bin"},
                "a" * 64,
            )
        self.assertEqual(binary, target / "release" / "examples" / "enoch_eval")
        self.assertEqual(output, marker)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args_list[0].args[0][1], "build")
        self.assertEqual(run.call_args_list[1].args[0][1], "test")

        with mock.patch(
            "training.enoch_week1_freeze._run", side_effect=["build ok", "running 0 tests"]
        ):
            with self.assertRaisesRegex(FreezeError, "did not execute"):
                _build_current_evaluator(
                    workspace,
                    target,
                    {"PATH": "/usr/bin:/bin"},
                    "b" * 64,
                )

    def test_probe_artifact_schema_resolves_all_three_frozen_paths(self) -> None:
        bundle = Path("/bundle")
        expected = {
            "binary/enoch-control-probe-current": bundle
            / "bin"
            / "enoch-control-probe-current",
            "binary/enoch-control-probe-reference": bundle
            / "bin"
            / "enoch-control-probe-reference",
            "source/enoch-control-probe": bundle
            / "source"
            / "enoch-control-probe.rs",
        }
        self.assertEqual(set(expected), set(CONTROL_PROBE_ARTIFACT_IDS))
        for artifact_id, path in expected.items():
            self.assertEqual(_bundle_artifact_path(bundle, artifact_id), path)
        with self.assertRaisesRegex(FreezeError, "unknown source artifact"):
            _bundle_artifact_path(bundle, "source/not-known")

    def test_preflight_and_protocol_artifacts_resolve_inside_the_bundle(self) -> None:
        bundle = Path("/bundle")
        self.assertEqual(
            _bundle_artifact_path(bundle, "preflight/expert-model-validation"),
            bundle / "preflight" / "expert-model-validation.json",
        )
        self.assertEqual(
            _bundle_artifact_path(bundle, "preflight/reference-model-contract-tests"),
            bundle / "preflight" / "reference-model-contract-tests.json",
        )
        self.assertEqual(
            _bundle_artifact_path(bundle, "preflight/strict-evaluator-test"),
            bundle / "preflight" / "strict-evaluator-test.json",
        )
        self.assertEqual(
            _bundle_artifact_path(bundle, "protocol/week1-seed-protocol"),
            bundle / "protocol" / "week1-seed-protocol.json",
        )
        with self.assertRaisesRegex(FreezeError, "unknown artifact category"):
            _bundle_artifact_path(bundle, "protocol/not-known")

    def test_bundle_verification_requires_every_probe_artifact(self) -> None:
        with mock.patch(
            "training.enoch_week1_freeze.enoch_week1.load_json_object",
            side_effect=[{}, {"artifact_hashes": {}}],
        ), mock.patch(
            "training.enoch_week1_freeze.enoch_week1.validate_w1_0_control_manifest",
            return_value="f" * 64,
        ):
            with self.assertRaisesRegex(FreezeError, "control probe artifacts are missing"):
                verify_bundle(Path("protocol.json"), Path("bundle"))


if __name__ == "__main__":
    unittest.main()
