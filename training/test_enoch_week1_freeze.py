#!/usr/bin/env python3

from __future__ import annotations

import io
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest import mock

from training.enoch_week1_freeze import (
    CONTROL_PROBE_DETERMINISM_REPETITIONS,
    CONTROL_PROBE_ARTIFACT_IDS,
    CONTROL_PROBE_REGRESSION_SEED,
    REFERENCE_PROBE_PATCH_ARTIFACT_ID,
    REFERENCE_PROBE_PATCH_RELATIVE,
    REFERENCE_PROBE_PREPATCH_PREREQUISITES,
    FreezeError,
    _apply_reference_probe_patch,
    _bind_reference_probe_patch,
    _build_control_probe,
    _build_current_evaluator,
    _bundle_artifact_path,
    _control_probe_determinism_record,
    _extract_archive,
    _hardware_summary,
    _inject_control_probe,
    _reference_probe_patch_paths,
    _safe_environment,
    _snapshot_sources,
    verify_bundle,
)


def _write_reference_patch_prerequisites(bundle: Path) -> None:
    for relative in REFERENCE_PROBE_PREPATCH_PREREQUISITES:
        path = bundle / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"staged:{relative.as_posix()}".encode("utf-8"))


class Week1FreezeTests(unittest.TestCase):
    def test_archive_extraction_supports_python_without_data_filter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "source.tar"
            payload = b"frozen source\n"
            with tarfile.open(archive_path, "w:") as archive:
                member = tarfile.TarInfo("nested/source.txt")
                member.mode = 0o10644
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
            destination = root / "destination"
            with mock.patch(
                "training.enoch_week1_freeze.inspect.signature",
                return_value=mock.Mock(parameters={}),
            ):
                _extract_archive(archive_path, destination)
            extracted = destination / "nested" / "source.txt"
            self.assertEqual(extracted.read_bytes(), payload)
            self.assertEqual(extracted.stat().st_mode & 0o7777, 0o644)

    def test_archive_extraction_rejects_traversal_and_special_members(self) -> None:
        for name, configure in (
            ("traversal", lambda member: setattr(member, "name", "../escape")),
            ("symlink", lambda member: setattr(member, "type", tarfile.SYMTYPE)),
        ):
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    archive_path = root / "source.tar"
                    with tarfile.open(archive_path, "w:") as archive:
                        member = tarfile.TarInfo("entry")
                        configure(member)
                        if member.issym():
                            member.linkname = "target"
                        archive.addfile(member, io.BytesIO(b""))
                    with self.assertRaisesRegex(FreezeError, "unsafe"):
                        _extract_archive(archive_path, root / "destination")
                    self.assertFalse((root / "escape").exists())

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
        self.assertTrue(
            any(
                record["path"] == REFERENCE_PROBE_PATCH_RELATIVE.as_posix()
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

    def test_reference_probe_patch_is_narrow_bound_and_applies_reversibly(self) -> None:
        patch_body = (
            "diff --git a/core/src/bot/heuristics.rs b/core/src/bot/heuristics.rs\n"
            "--- a/core/src/bot/heuristics.rs\n"
            "+++ b/core/src/bot/heuristics.rs\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
            "diff --git a/core/src/bot/policy.rs b/core/src/bot/policy.rs\n"
            "--- a/core/src/bot/policy.rs\n"
            "+++ b/core/src/bot/policy.rs\n"
            "@@ -1 +1 @@\n"
            "-old-policy\n"
            "+new-policy\n"
            "diff --git a/core/src/bot/harness.rs b/core/src/bot/harness.rs\n"
            "--- a/core/src/bot/harness.rs\n"
            "+++ b/core/src/bot/harness.rs\n"
            "@@ -1 +1 @@\n"
            "-old-harness\n"
            "+new-harness\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            patch = workspace / REFERENCE_PROBE_PATCH_RELATIVE
            patch.parent.mkdir(parents=True)
            patch.write_text(patch_body, encoding="utf-8")
            reference = root / "reference"
            heuristics = reference / "core" / "src" / "bot" / "heuristics.rs"
            heuristics.parent.mkdir(parents=True)
            heuristics.write_text("old\n", encoding="utf-8")
            policy = reference / "core" / "src" / "bot" / "policy.rs"
            policy.write_text("old-policy\n", encoding="utf-8")
            harness = reference / "core" / "src" / "bot" / "harness.rs"
            harness.write_text("old-harness\n", encoding="utf-8")

            bundle = root / "bundle"
            with self.assertRaisesRegex(FreezeError, "cannot be bound before"):
                _bind_reference_probe_patch(workspace, bundle)
            _write_reference_patch_prerequisites(bundle)
            frozen, digest = _bind_reference_probe_patch(workspace, bundle)
            receipt = _apply_reference_probe_patch(
                reference,
                frozen,
                {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
            )

            self.assertEqual(
                _reference_probe_patch_paths(frozen),
                {
                    "core/src/bot/harness.rs",
                    "core/src/bot/heuristics.rs",
                    "core/src/bot/policy.rs",
                },
            )
            self.assertEqual(receipt["patch_sha256"], digest)
            self.assertNotEqual(
                receipt["target_files_before_sha256"],
                receipt["target_files_after_sha256"],
            )
            self.assertEqual(heuristics.read_text(encoding="utf-8"), "new\n")
            self.assertEqual(policy.read_text(encoding="utf-8"), "new-policy\n")
            self.assertEqual(harness.read_text(encoding="utf-8"), "new-harness\n")

    def test_reference_probe_patch_rejects_any_second_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            patch = workspace / REFERENCE_PROBE_PATCH_RELATIVE
            patch.parent.mkdir(parents=True)
            patch.write_text(
                "diff --git a/core/src/bot/heuristics.rs b/core/src/bot/heuristics.rs\n"
                "--- a/core/src/bot/heuristics.rs\n"
                "+++ b/core/src/bot/heuristics.rs\n"
                "diff --git a/core/src/bot/search.rs b/core/src/bot/search.rs\n"
                "--- a/core/src/bot/search.rs\n"
                "+++ b/core/src/bot/search.rs\n",
                encoding="utf-8",
            )
            _write_reference_patch_prerequisites(root / "bundle")
            with self.assertRaisesRegex(FreezeError, "must touch exactly"):
                _bind_reference_probe_patch(workspace, root / "bundle")

    def test_audited_reference_probe_patch_applies_to_production_reference(self) -> None:
        workspace = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "reference.tar"
            subprocess.run(
                [
                    "git",
                    "archive",
                    "--format=tar",
                    f"--output={archive}",
                    "c813c8a",
                ],
                cwd=workspace,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            reference = root / "reference"
            _extract_archive(archive, reference)
            _write_reference_patch_prerequisites(root / "bundle")
            frozen, digest = _bind_reference_probe_patch(workspace, root / "bundle")
            receipt = _apply_reference_probe_patch(
                reference,
                frozen,
                _safe_environment(),
            )
        self.assertEqual(receipt["patch_sha256"], digest)
        self.assertNotEqual(
            receipt["target_files_before_sha256"],
            receipt["target_files_after_sha256"],
        )

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

    def test_control_probe_determinism_runs_fresh_processes_and_cross_checks(self) -> None:
        reference = Path("/bundle/bin/enoch-control-probe-reference")
        current = Path("/bundle/bin/enoch-control-probe-current")
        repetitions = 3
        with mock.patch(
            "training.enoch_week1_freeze._run", return_value='{"stable":true}'
        ) as run:
            record = _control_probe_determinism_record(
                reference,
                current,
                {"PATH": "/usr/bin:/bin"},
                repetitions=repetitions,
            )
        self.assertTrue(record["command_succeeded"])
        self.assertEqual(record["seed"], CONTROL_PROBE_REGRESSION_SEED)
        self.assertEqual(record["repetitions_per_binary"], repetitions)
        self.assertEqual(run.call_count, repetitions * 2)
        for call in run.call_args_list:
            self.assertEqual(call.args[0][-1], str(CONTROL_PROBE_REGRESSION_SEED))

        with mock.patch(
            "training.enoch_week1_freeze._run",
            side_effect=["first", "different"],
        ):
            with self.assertRaisesRegex(FreezeError, "not process-deterministic"):
                _control_probe_determinism_record(
                    reference,
                    current,
                    {"PATH": "/usr/bin:/bin"},
                    repetitions=2,
                )

        with mock.patch(
            "training.enoch_week1_freeze._run",
            side_effect=["reference", "reference", "current", "current"],
        ):
            with self.assertRaisesRegex(FreezeError, "diverge"):
                _control_probe_determinism_record(
                    reference,
                    current,
                    {"PATH": "/usr/bin:/bin"},
                    repetitions=2,
                )

        self.assertGreaterEqual(CONTROL_PROBE_DETERMINISM_REPETITIONS, 2)

    def test_probe_artifact_schema_resolves_all_frozen_paths(self) -> None:
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
            REFERENCE_PROBE_PATCH_ARTIFACT_ID: bundle
            / "source"
            / "enoch-control-probe-reference.patch",
            "preflight/control-probe-determinism": bundle
            / "preflight"
            / "control-probe-determinism.json",
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
