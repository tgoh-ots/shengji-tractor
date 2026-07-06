#!/usr/bin/env python3
"""Adversarial unit tests for the additive W1.3 seal recovery."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from training import enoch_week1
from training import enoch_week1_evidence
from training import enoch_week1_runner
from training import enoch_week1_w1_3_operator as w13
from training import enoch_week1_w1_3_seal_recovery as recovery


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class EvidenceReturnSemanticsTests(unittest.TestCase):
    def test_real_external_validator_returns_counter_map_not_fingerprint(self) -> None:
        protocol = {"protocol_fingerprint": _digest("protocol")}
        comparison = {
            "comparison_protocol_fingerprint": _digest("comparison"),
            "environment_fingerprint": _digest("environment"),
        }
        artifacts = [
            {
                "artifact_id": artifact_id,
                "artifact_kind": "test-artifact",
                "file_sha256": _digest(f"file:{artifact_id}"),
                "path": f"/validated/{artifact_id}",
                "semantic_fingerprint": _digest(f"semantic:{artifact_id}"),
            }
            for artifact_id in enoch_week1_evidence.EXPECTED_ARTIFACT_IDS
        ]
        counters = enoch_week1_evidence._counter_records()  # noqa: SLF001
        body = {
            "artifacts": artifacts,
            "automatic_production_promotion_allowed": False,
            "comparison_protocol_fingerprint": comparison[
                "comparison_protocol_fingerprint"
            ],
            "counters": counters,
            "environment_fingerprint": comparison["environment_fingerprint"],
            "manifest_kind": enoch_week1_evidence.EVIDENCE_KIND,
            "manifest_version": enoch_week1_evidence.MANIFEST_VERSION,
            "protocol_fingerprint": protocol["protocol_fingerprint"],
        }
        evidence = {
            **body,
            "verified_external_evidence_fingerprint": (
                enoch_week1.canonical_json_sha256(body)
            ),
        }
        with mock.patch.object(
            enoch_week1_evidence,
            "_validate_artifact_paths",
            return_value=artifacts,
        ):
            observed = enoch_week1_evidence.validate_verified_external_evidence(
                protocol, comparison, evidence
            )
        self.assertEqual(observed, counters)
        self.assertIsInstance(observed, dict)
        self.assertNotEqual(
            observed, evidence["verified_external_evidence_fingerprint"]
        )


class OfflineGuardTests(unittest.TestCase):
    def test_runner_claim_and_w13_execution_entrypoints_are_tripwired(self) -> None:
        targets = (
            (enoch_week1, "consume_seed_batch_once"),
            (enoch_week1, "consume_seed_once"),
            (enoch_week1_runner, "probe_evaluator_environment_identity"),
            (enoch_week1_runner, "run_comparison"),
            (enoch_week1_runner, "_execute_shard"),
            (w13, "declare_w1_3"),
            (w13, "verify_w1_3"),
            (w13, "run_w1_3"),
            (w13, "_run_arm"),
            (w13, "_scan_resume_frontier"),
        )
        originals = {(owner, name): getattr(owner, name) for owner, name in targets}
        with recovery._offline_api_guard():  # noqa: SLF001
            for owner, name in targets:
                with self.subTest(api=f"{owner.__name__}.{name}"), self.assertRaisesRegex(
                    recovery.SealRecoveryError, "offline seal recovery forbids"
                ):
                    getattr(owner, name)()
        for owner, name in targets:
            self.assertIs(getattr(owner, name), originals[(owner, name)])


class ArchiveTests(unittest.TestCase):
    def _malformed_fixture(self) -> tuple[dict[str, object], bytes]:
        body = {"arm_results": [], "manifest_kind": "test-malformed-set"}
        value = {
            **body,
            "supported_change_set_fingerprint": (
                enoch_week1.canonical_json_sha256(body)
            ),
        }
        payload = enoch_week1.canonical_json_bytes(value) + b"\n"
        return value, payload

    def test_archive_preserves_exact_bytes_and_is_resumable(self) -> None:
        value, payload = self._malformed_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            layout = w13.W13Layout(Path(temporary).resolve())
            layout.supported_set.parent.mkdir(parents=True)
            layout.supported_set.write_bytes(payload)
            paths = recovery.RecoveryPaths(layout)
            patches = (
                mock.patch.object(
                    recovery,
                    "MALFORMED_SUPPORTED_SET",
                    value["supported_change_set_fingerprint"],
                ),
                mock.patch.object(
                    recovery,
                    "MALFORMED_SUPPORTED_SET_FILE_SHA256",
                    hashlib.sha256(payload).hexdigest(),
                ),
                mock.patch.object(
                    recovery, "MALFORMED_SUPPORTED_SET_FILE_SIZE", len(payload)
                ),
            )
            with patches[0], patches[1], patches[2]:
                malformed, state, archive_payload = (
                    recovery._ensure_malformed_archive(  # noqa: SLF001
                        layout, paths, require_archive=False
                    )
                )
                self.assertEqual(malformed, value)
                self.assertEqual(state, "malformed")
                self.assertEqual(archive_payload, payload)
                recovery._atomic_archive_bytes(  # noqa: SLF001
                    layout, paths.archive, archive_payload
                )
                recovery._atomic_archive_bytes(  # noqa: SLF001
                    layout, paths.archive, archive_payload
                )
                self.assertEqual(paths.archive.read_bytes(), payload)
                self.assertEqual(layout.supported_set.read_bytes(), payload)
                recovered, resumed_state, resumed_payload = (
                    recovery._ensure_malformed_archive(  # noqa: SLF001
                        layout, paths, require_archive=True
                    )
                )
                self.assertEqual(recovered, value)
                self.assertEqual(resumed_state, "malformed")
                self.assertEqual(resumed_payload, payload)

    def test_existing_archive_mismatch_fails_closed(self) -> None:
        value, payload = self._malformed_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            layout = w13.W13Layout(Path(temporary).resolve())
            layout.supported_set.parent.mkdir(parents=True)
            layout.supported_set.write_bytes(payload)
            paths = recovery.RecoveryPaths(layout)
            paths.archive.parent.mkdir(parents=True)
            paths.archive.write_bytes(b"changed\n")
            with mock.patch.object(
                recovery,
                "MALFORMED_SUPPORTED_SET",
                value["supported_change_set_fingerprint"],
            ), mock.patch.object(
                recovery,
                "MALFORMED_SUPPORTED_SET_FILE_SHA256",
                hashlib.sha256(payload).hexdigest(),
            ), mock.patch.object(
                recovery, "MALFORMED_SUPPORTED_SET_FILE_SIZE", len(payload)
            ), self.assertRaises(recovery.SealRecoveryError):
                recovery._ensure_malformed_archive(  # noqa: SLF001
                    layout, paths, require_archive=False
                )


class OutputPathSafetyTests(unittest.TestCase):
    def _invoke_output_write(
        self,
        layout: w13.W13Layout,
        path: Path,
        kind: str,
    ) -> None:
        if kind == "archive":
            recovery._atomic_archive_bytes(layout, path, b"archive\n")  # noqa: SLF001
        else:
            recovery._write_or_match(  # noqa: SLF001
                layout, path, {"kind": kind}, f"{kind} test artifact"
            )

    def test_recovery_parent_symlink_cannot_escape_run_root(self) -> None:
        for kind in ("archive", "provenance", "manifest"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as outside_temporary:
                layout = w13.W13Layout(Path(temporary).resolve())
                layout.directory.mkdir()
                paths = recovery.RecoveryPaths(layout)
                recovery_directory = layout.root / recovery.RECOVERY_DIRECTORY_RELATIVE
                recovery_directory.symlink_to(
                    Path(outside_temporary).resolve(), target_is_directory=True
                )
                path = {
                    "archive": paths.archive,
                    "provenance": paths.provenance,
                    "manifest": paths.manifest,
                }[kind]
                with self.assertRaisesRegex(
                    recovery.SealRecoveryError,
                    "non-symlink|symlink|not a real directory",
                ):
                    self._invoke_output_write(layout, path, kind)

        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as outside_temporary:
            layout = w13.W13Layout(Path(temporary).resolve())
            layout.directory.symlink_to(
                Path(outside_temporary).resolve(), target_is_directory=True
            )
            with self.assertRaisesRegex(
                recovery.SealRecoveryError, "non-symlink|symlink"
            ):
                self._invoke_output_write(layout, layout.phase, "phase")

    def test_existing_output_leaf_symlink_is_rejected_for_every_recovery_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as outside_temporary:
            layout = w13.W13Layout(Path(temporary).resolve())
            layout.directory.mkdir()
            recovery_directory = layout.root / recovery.RECOVERY_DIRECTORY_RELATIVE
            recovery_directory.mkdir()
            paths = recovery.RecoveryPaths(layout)
            outside = Path(outside_temporary).resolve() / "outside.json"
            outside.write_text("{}\n", encoding="utf-8")
            cases = (
                ("archive", paths.archive),
                ("provenance", paths.provenance),
                ("manifest", paths.manifest),
                ("phase", layout.phase),
            )
            for kind, path in cases:
                with self.subTest(kind=kind):
                    path.symlink_to(outside)
                    try:
                        with self.assertRaisesRegex(
                            recovery.SealRecoveryError, "regular non-symlink"
                        ):
                            self._invoke_output_write(layout, path, kind)
                    finally:
                        path.unlink()
            self.assertEqual(outside.read_text(encoding="utf-8"), "{}\n")

    def test_lexically_outside_output_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as outside_temporary:
            layout = w13.W13Layout(Path(temporary).resolve())
            layout.directory.mkdir()
            outside = Path(outside_temporary).resolve() / "phase.json"
            with self.assertRaisesRegex(recovery.SealRecoveryError, "escapes"):
                recovery._write_or_match(  # noqa: SLF001
                    layout, outside, {"kind": "outside"}, "outside output"
                )

    def test_recovery_directory_creation_fsyncs_w13_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = w13.W13Layout(Path(temporary).resolve())
            layout.directory.mkdir()
            with mock.patch.object(
                recovery.os, "open", return_value=123
            ) as open_mock, mock.patch.object(
                recovery.os, "fsync"
            ) as fsync_mock, mock.patch.object(
                recovery.os, "close"
            ) as close_mock:
                self.assertTrue(
                    recovery._require_recovery_directory(  # noqa: SLF001
                        layout, create=True
                    )
                )
            open_mock.assert_called_once_with(layout.directory, recovery.os.O_RDONLY)
            fsync_mock.assert_called_once_with(123)
            close_mock.assert_called_once_with(123)


class CorrectionTests(unittest.TestCase):
    def _synthetic_context(
        self, root: Path
    ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        counters = {"validated-counter": {"count": 0}}
        evidence: dict[str, object] = {}
        corrected_records = []
        for arm_id in w13.SURVIVOR_ARM_IDS:
            attempt = root / arm_id / "attempt-001"
            attempt.mkdir(parents=True)
            (attempt / "external-evidence.json").write_text("{}\n", encoding="utf-8")
            expected = recovery.EXPECTED_ARM_LINEAGE[arm_id]
            evidence[arm_id] = {
                "arm_id": arm_id,
                "attempt": attempt,
                "comparison": {
                    "comparison_protocol_fingerprint": expected["comparison"]
                },
                "external_evidence": {
                    "counters": counters,
                    "verified_external_evidence_fingerprint": expected["external"],
                },
                "external_evidence_fingerprint": copy.deepcopy(counters),
                "machine_attestation_fingerprint": expected["machine"],
                "merged_result": {
                    "merged_result_fingerprint": expected["merged"]
                },
                "support_decision": {
                    "support_decision_fingerprint": expected["decision"]
                },
            }
            corrected_records.append(
                {
                    "arm_id": arm_id,
                    "external_evidence_fingerprint": expected["external"],
                }
            )
        corrected = {
            "arm_results": corrected_records,
            "marker": "synthetic-corrected",
            "supported_change_set_fingerprint": recovery.CORRECTED_SUPPORTED_SET,
        }
        malformed = copy.deepcopy(corrected)
        for record in malformed["arm_results"]:
            record["external_evidence_fingerprint"] = copy.deepcopy(counters)
        malformed_body = dict(malformed)
        malformed_body.pop("supported_change_set_fingerprint")
        malformed["supported_change_set_fingerprint"] = (
            enoch_week1.canonical_json_sha256(malformed_body)
        )
        context = {
            "arm_evidence": evidence,
            "declaration": {},
            "ledger": {},
            "state": {"fixture": {}, "protocol": {}},
        }
        return context, malformed, corrected

    def test_exact_five_field_correction_reaches_known_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context, malformed, corrected = self._synthetic_context(Path(temporary))
            counters = next(iter(context["arm_evidence"].values()))[
                "external_evidence_fingerprint"
            ]
            with mock.patch.object(
                w13, "validate_supported_change_set"
            ), mock.patch.object(
                w13, "build_supported_change_set", return_value=corrected
            ), mock.patch.object(
                enoch_week1_evidence,
                "validate_verified_external_evidence",
                return_value=counters,
            ):
                normalized, observed, corrections = (
                    recovery._normalize_arm_evidence(context, malformed)  # noqa: SLF001
                )
            self.assertEqual(
                observed["supported_change_set_fingerprint"],
                recovery.CORRECTED_SUPPORTED_SET,
            )
            self.assertEqual(len(corrections), 5)
            for arm_id in w13.SURVIVOR_ARM_IDS:
                self.assertEqual(
                    normalized[arm_id]["external_evidence_fingerprint"],
                    recovery.EXPECTED_ARM_LINEAGE[arm_id]["external"],
                )
            self.assertTrue(
                all(
                    record["malformed_value_equals_validated_counters"] is True
                    for record in corrections
                )
            )

    def test_non_counter_malformed_leaf_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context, malformed, corrected = self._synthetic_context(Path(temporary))
            malformed["arm_results"][0]["external_evidence_fingerprint"] = {
                "different": True
            }
            counters = next(iter(context["arm_evidence"].values()))[
                "external_evidence_fingerprint"
            ]
            with mock.patch.object(
                w13, "validate_supported_change_set"
            ), mock.patch.object(
                w13, "build_supported_change_set", return_value=corrected
            ), mock.patch.object(
                enoch_week1_evidence,
                "validate_verified_external_evidence",
                return_value=counters,
            ), self.assertRaisesRegex(
                recovery.SealRecoveryError, "does not equal validated evidence counters"
            ):
                recovery._normalize_arm_evidence(context, malformed)  # noqa: SLF001


class PhaseTests(unittest.TestCase):
    def _phase(self) -> dict[str, object]:
        return {
            "artifacts": [
                {
                    "artifact_id": "supported-independent-change-set",
                    "sha256": recovery.CORRECTED_SUPPORTED_SET,
                },
                {
                    "artifact_id": "w1.3/malformed-supported-independent-change-set",
                    "sha256": recovery.MALFORMED_SUPPORTED_SET,
                },
                {
                    "artifact_id": "w1.3/seal-recovery-manifest",
                    "sha256": _digest("manifest"),
                },
                {
                    "artifact_id": "w1.3/seal-recovery-provenance",
                    "sha256": _digest("provenance"),
                },
            ],
            "phase_manifest_fingerprint": _digest("phase"),
        }

    def test_recovery_aware_phase_tamper_is_rejected(self) -> None:
        expected = self._phase()
        provenance = {
            "seal_recovery_provenance_fingerprint": _digest("provenance")
        }
        manifest = {"seal_recovery_manifest_fingerprint": _digest("manifest")}
        context = {
            "state": {
                "phase0": {},
                "phase1": {},
                "phase2": {},
                "protocol": {},
            }
        }
        with mock.patch.object(
            recovery, "_build_recovered_phase", return_value=expected
        ), mock.patch.object(
            enoch_week1, "validate_phase_chain", return_value=[_digest("phase")]
        ):
            self.assertEqual(
                recovery._validate_recovered_phase(  # noqa: SLF001
                    expected, context, provenance, manifest, {}, {}
                ),
                _digest("phase"),
            )
            tampered = copy.deepcopy(expected)
            tampered["artifacts"][0]["sha256"] = _digest("forged")
            with self.assertRaisesRegex(
                recovery.SealRecoveryError, "does not reconstruct"
            ):
                recovery._validate_recovered_phase(  # noqa: SLF001
                    tampered, context, provenance, manifest, {}, {}
                )

    def test_recovered_phase_preserves_all_ordinary_declarations(self) -> None:
        ordinary = {
            "artifacts": [
                {"artifact_id": "ordinary-artifact", "sha256": _digest("ordinary")}
            ],
            "declarations": {
                "attempted_arm_ids": list(w13.SURVIVOR_ARM_IDS),
                "complete_arm_count": 5,
                "continuation_source_commit": recovery.ORIGINAL_W13_COMMIT,
            },
        }
        context = {
            "declaration": {},
            "ledger": {},
            "provenance": {},
            "state": {"fixture": {}, "phase2": {}, "protocol": {}},
        }
        provenance = {
            "recovery_git_commit": _digest("commit")[:40],
            "seal_recovery_provenance_fingerprint": _digest("provenance"),
        }
        manifest = {"seal_recovery_manifest_fingerprint": _digest("manifest")}

        def phase_builder(
            _protocol: object,
            _phase: str,
            *,
            artifacts: dict[str, str],
            declarations: dict[str, object],
            parent_phase_manifests: object,
        ) -> dict[str, object]:
            del parent_phase_manifests
            return {"artifacts": artifacts, "declarations": declarations}

        with mock.patch.object(
            w13, "_build_phase3", return_value=ordinary
        ), mock.patch.object(
            enoch_week1, "build_phase_manifest", side_effect=phase_builder
        ):
            observed = recovery._build_recovered_phase(  # noqa: SLF001
                context, provenance, manifest, {}, {}
            )
        self.assertEqual(observed["declarations"]["complete_arm_count"], 5)
        self.assertEqual(
            observed["declarations"]["continuation_source_commit"],
            recovery.ORIGINAL_W13_COMMIT,
        )
        self.assertEqual(
            observed["artifacts"]["ordinary-artifact"], _digest("ordinary")
        )
        self.assertEqual(
            observed["artifacts"]["w1.3/seal-recovery-manifest"],
            _digest("manifest"),
        )


class SourceContractTests(unittest.TestCase):
    def test_exact_two_file_child_contract_and_known_identities(self) -> None:
        self.assertEqual(
            recovery.RECOVERY_COMMIT_PATHS,
            (
                "training/enoch_week1_w1_3_seal_recovery.py",
                "training/test_enoch_week1_w1_3_seal_recovery.py",
            ),
        )
        self.assertEqual(
            recovery.ORIGINAL_W13_COMMIT,
            "171ee31a1528085f2378c8db211ba74ea25b9925",
        )
        self.assertEqual(
            recovery.ORIGINAL_W13_TREE,
            "121be2fdf4f701c2c1a1c83f531028b3fc74b617",
        )
        self.assertEqual(
            recovery.CORRECTED_SUPPORTED_SET,
            "a6b2e8f0b79eb2199b141f68ce6d65a4716fbe6dbae2e2160738c0cd051ce025",
        )
        self.assertEqual(
            recovery.CORRECTED_SUPPORTED_SET_FILE_SHA256,
            "e6021dee34db462c7d2a5b102de09377fe5b0bda360b51ae4595a9ca4ac2a6dd",
        )
        self.assertEqual(recovery.EXPECTED_PROTECTED_FILE_COUNT, 209)
        self.assertEqual(
            recovery.EXPECTED_PROTECTED_FILES_SHA256,
            "be82902f174db0a9184b4eca74e0e7b70110a80aa70bf53b23c3c5bbe29d29b1",
        )

    def test_git_identity_accepts_only_clean_exact_two_file_child(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            subprocess.run(["git", "init", "-q", str(workspace)], check=True)
            subprocess.run(
                ["git", "-C", str(workspace), "config", "user.name", "Recovery Test"],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(workspace),
                    "config",
                    "user.email",
                    "recovery-test@example.invalid",
                ],
                check=True,
            )
            original_operator = workspace / w13.OPERATOR_RELATIVE
            original_operator.parent.mkdir(parents=True)
            original_operator.write_text("# frozen original operator\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(workspace), "add", "."], check=True)
            subprocess.run(
                ["git", "-C", str(workspace), "commit", "-q", "-m", "original"],
                check=True,
            )
            parent = subprocess.check_output(
                ["git", "-C", str(workspace), "rev-parse", "HEAD"], text=True
            ).strip()
            parent_tree = subprocess.check_output(
                ["git", "-C", str(workspace), "rev-parse", "HEAD^{tree}"],
                text=True,
            ).strip()
            for relative in recovery.RECOVERY_COMMIT_PATHS:
                path = workspace / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"# {relative}\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(workspace), "add", "."], check=True)
            subprocess.run(
                ["git", "-C", str(workspace), "commit", "-q", "-m", "recovery"],
                check=True,
            )
            source_sha = hashlib.sha256(original_operator.read_bytes()).hexdigest()
            with mock.patch.object(
                recovery, "ORIGINAL_W13_COMMIT", parent
            ), mock.patch.object(
                recovery, "ORIGINAL_W13_TREE", parent_tree
            ), mock.patch.object(
                recovery, "ORIGINAL_W13_OPERATOR_SHA256", source_sha
            ):
                identity = recovery._recovery_git_identity(  # noqa: SLF001
                    workspace, require_live=True
                )
                self.assertEqual(
                    [record["path"] for record in identity["changed_paths"]],
                    list(recovery.RECOVERY_COMMIT_PATHS),
                )
                (workspace / "untracked.txt").write_text("dirty\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    recovery.SealRecoveryError, "not clean"
                ):
                    recovery._recovery_git_identity(  # noqa: SLF001
                        workspace, require_live=True
                    )


class InventoryAndOrderingTests(unittest.TestCase):
    def test_protected_snapshot_pins_count_and_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = w13.W13Layout(Path(temporary).resolve())
            paths = [
                layout.input,
                layout.provenance,
                layout.declaration,
                layout.final_ledger,
                layout.directory / "arms" / "01-test" / "attempts" / "attempt-001" / "a.json",
                layout.directory / "arms" / "01-test" / "support-decision.json",
            ]
            for index, path in enumerate(paths):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"artifact-{index}\n", encoding="utf-8")
            records = [
                {
                    "path": path.relative_to(layout.root).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in sorted(paths)
            ]
            expected_hash = enoch_week1.canonical_json_sha256(records)
            with mock.patch.object(
                recovery, "EXPECTED_PROTECTED_FILE_COUNT", len(records)
            ), mock.patch.object(
                recovery, "EXPECTED_PROTECTED_FILES_SHA256", expected_hash
            ):
                self.assertEqual(
                    recovery._protected_snapshot(layout),  # noqa: SLF001
                    {"file_count": len(records), "files_sha256": expected_hash},
                )
                paths[-1].write_text("tampered\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    recovery.SealRecoveryError, "inventory hash changed"
                ):
                    recovery._protected_snapshot(layout)  # noqa: SLF001

    def test_recover_write_order_makes_phase_the_last_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            layout = w13.W13Layout(Path(temporary).resolve())
            layout.directory.mkdir()
            paths = recovery.RecoveryPaths(layout)
            protected = {"file_count": 209, "files_sha256": _digest("protected")}
            provenance = {
                "recovery_git_commit": _digest("commit")[:40],
                "seal_recovery_provenance_fingerprint": _digest("provenance"),
            }
            manifest = {"seal_recovery_manifest_fingerprint": _digest("manifest")}
            corrected = {
                "status": "supported-survivors",
                "summary": {"supported_arm_ids": ["bid-ownership"]},
            }
            phase = {"phase_manifest_fingerprint": _digest("phase")}
            prepared = {
                "canonical_state": "malformed",
                "context": {},
                "corrected": corrected,
                "corrections": [],
                "malformed_payload": b"malformed\n",
                "normalized_evidence": {},
                "paths": paths,
                "phase": phase,
                "protected_snapshot": protected,
                "recovery_manifest": manifest,
                "recovery_provenance": provenance,
            }
            writes: list[str] = []

            def write_or_match(
                got_layout: w13.W13Layout,
                path: Path,
                _value: object,
                _label: str,
            ) -> None:
                self.assertEqual(got_layout, layout)
                if path == paths.provenance:
                    writes.append("provenance")
                elif path == paths.manifest:
                    writes.append("manifest")
                elif path == layout.phase:
                    writes.append("phase")

            def load_json(path: Path) -> dict[str, object]:
                if path == paths.provenance:
                    return provenance
                if path == paths.manifest:
                    return manifest
                if path == layout.phase:
                    return phase
                raise AssertionError(f"unexpected JSON read: {path}")

            with mock.patch.object(
                recovery, "_offline_api_guard", return_value=contextlib.nullcontext()
            ), mock.patch.object(
                w13, "verify_sealed_w1_2"
            ), mock.patch.object(
                recovery.base_operator,
                "_operator_lock",
                return_value=contextlib.nullcontext(),
            ), mock.patch.object(
                recovery, "_prepare_recovery", return_value=prepared
            ), mock.patch.object(
                recovery,
                "_atomic_archive_bytes",
                side_effect=lambda *_args: writes.append("archive"),
            ), mock.patch.object(
                recovery, "_write_or_match", side_effect=write_or_match
            ), mock.patch.object(
                recovery, "_load_json", side_effect=load_json
            ), mock.patch.object(
                recovery, "_safe_output_json", side_effect=lambda _layout, path, _label: load_json(path)
            ), mock.patch.object(
                recovery,
                "_install_corrected_supported_set",
                side_effect=lambda *_args: writes.append("corrected"),
            ), mock.patch.object(
                recovery,
                "_sha256_file",
                return_value=recovery.CORRECTED_SUPPORTED_SET_FILE_SHA256,
            ), mock.patch.object(
                recovery, "_safe_output_bytes", return_value=b"corrected\n"
            ), mock.patch.object(
                recovery,
                "_sha256_bytes",
                return_value=recovery.CORRECTED_SUPPORTED_SET_FILE_SHA256,
            ), mock.patch.object(
                recovery, "_protected_snapshot", return_value=protected
            ), mock.patch.object(
                recovery, "_read_bytes", return_value=b"ledger\n"
            ), mock.patch.object(
                recovery, "_validate_malformed_bytes"
            ), mock.patch.object(
                recovery, "validate_recovery_provenance"
            ), mock.patch.object(
                recovery, "validate_recovery_manifest"
            ), mock.patch.object(
                recovery, "_validate_recovered_phase", return_value=_digest("phase")
            ):
                recovery.recover_seal(
                    layout,
                    Path(temporary),
                    Path(temporary) / "sealed-w1.2",
                    environment={},
                )
            self.assertEqual(
                writes,
                ["archive", "provenance", "manifest", "corrected", "phase"],
            )


if __name__ == "__main__":
    unittest.main()
