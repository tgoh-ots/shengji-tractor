#!/usr/bin/env python3
"""Offline-only, additive recovery for the known W1.3 seal type defect.

The five W1.3 comparisons completed under the exact source at ``171ee31``.
That source validated each external-evidence artifact correctly, but assigned the
validator's counter-map return value to a field named
``external_evidence_fingerprint``.  The malformed supported-set was written and
phase construction then failed because a phase artifact hash must be SHA-256.

This tool never evaluates a hand and never claims a seed.  It validates the
entire immutable run with the unchanged W1.3 module, proves the one known
malformation, archives its exact bytes, derives the corrected supported-set from
the already sealed evidence fingerprints, and writes a recovery-bound phase.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Iterator, Mapping, Sequence

try:
    from training import enoch_week1
    from training import enoch_week1_evidence
    from training import enoch_week1_operator as base_operator
    from training import enoch_week1_runner
    from training import enoch_week1_w1_3_operator as w13
except ImportError:  # pragma: no cover - direct-script import path.
    import enoch_week1  # type: ignore[no-redef]
    import enoch_week1_evidence  # type: ignore[no-redef]
    import enoch_week1_operator as base_operator  # type: ignore[no-redef]
    import enoch_week1_runner  # type: ignore[no-redef]
    import enoch_week1_w1_3_operator as w13  # type: ignore[no-redef]


MANIFEST_VERSION = 1
ORIGINAL_W13_COMMIT = "171ee31a1528085f2378c8db211ba74ea25b9925"
ORIGINAL_W13_TREE = "121be2fdf4f701c2c1a1c83f531028b3fc74b617"
ORIGINAL_W13_OPERATOR_SHA256 = (
    "2d7241f5fcb36cdc0c7582ef47027cf578f2b25c46d52292dcd43f5285b7870c"
)
ORIGINAL_W13_PROVENANCE = (
    "d947489a73996558289e0f2815ad3742d97c717ebc97f447a24a56f04a3ee16e"
)
ORIGINAL_W13_DECLARATION = (
    "a9fe49ba40831844cafa8a56c9fe6663c659785385bac968d338486da676ac89"
)
FINAL_W13_LEDGER = (
    "999e43c97bd27daa372df882a0208a71862cd7bc484e8a87605df5363e38c897"
)
MALFORMED_SUPPORTED_SET = (
    "116b56f15c34d7dafe15579716d6adbc4b1a88e3eb0b75407d0a49e1f6ba0ee0"
)
MALFORMED_SUPPORTED_SET_FILE_SHA256 = (
    "79cd17778ff68582b7886ec0f4e382d060175aedd67b94b84ac6462fcd6816f7"
)
MALFORMED_SUPPORTED_SET_FILE_SIZE = 27_782
CORRECTED_SUPPORTED_SET = (
    "a6b2e8f0b79eb2199b141f68ce6d65a4716fbe6dbae2e2160738c0cd051ce025"
)
CORRECTED_SUPPORTED_SET_FILE_SHA256 = (
    "e6021dee34db462c7d2a5b102de09377fe5b0bda360b51ae4595a9ca4ac2a6dd"
)
EXPECTED_PROTECTED_FILE_COUNT = 209
EXPECTED_PROTECTED_FILES_SHA256 = (
    "be82902f174db0a9184b4eca74e0e7b70110a80aa70bf53b23c3c5bbe29d29b1"
)

RECOVERY_SOURCE_RELATIVE = Path("training/enoch_week1_w1_3_seal_recovery.py")
RECOVERY_TEST_RELATIVE = Path("training/test_enoch_week1_w1_3_seal_recovery.py")
RECOVERY_COMMIT_PATHS = (
    RECOVERY_SOURCE_RELATIVE.as_posix(),
    RECOVERY_TEST_RELATIVE.as_posix(),
)
RECOVERY_DIRECTORY_RELATIVE = Path("w1.3/seal-recovery")
MALFORMED_ARCHIVE_RELATIVE = (
    RECOVERY_DIRECTORY_RELATIVE / "malformed-supported-independent-change-set.json"
)
RECOVERY_PROVENANCE_RELATIVE = (
    RECOVERY_DIRECTORY_RELATIVE / "recovery-provenance.json"
)
RECOVERY_MANIFEST_RELATIVE = RECOVERY_DIRECTORY_RELATIVE / "recovery-manifest.json"

RECOVERY_PROVENANCE_KIND = "enoch-week1-w1.3-seal-recovery-provenance"
RECOVERY_MANIFEST_KIND = "enoch-week1-w1.3-seal-recovery-manifest"
RECOVERY_DEFECT_ID = "w1.3-external-evidence-validator-return-type-seal-defect-v1"

EXPECTED_ARM_LINEAGE = {
    "bid-ownership": {
        "comparison": "6883511f01298646cb27fc5f800590990e33e93316ddcd54f302cfba57004f66",
        "decision": "e59d1b721e13bdbab96888172808708c3cc8d29db16c24ea9513fc8259968494",
        "external": "e15cb2cd85f83e2f73048ff6f55caf85b0af968041786d0db4c5b46eb65abe99",
        "machine": "d1dba4d1c63a0be8568b82d46e93fb5339e0b7d0abcd1c250d981b3e44e82b79",
        "merged": "29c1faa118316f7218b0cd326b0eaa4789f93f595352fc7407cf5c5e1f85d362",
    },
    "compound-follow": {
        "comparison": "d58af492c94d1cedbd01535e7ab99d15fad473c66b854db8f4fd20eade2ede84",
        "decision": "f83cc1c92ca3c49c07bcbdfb7c8003244ca5cb51bd8ccb5ed4a4ec56b2aac25c",
        "external": "31689bf29a84bcdb8d3ed3a5d1a9932bac3f96cfa5567e464aff8c694ba8dba8",
        "machine": "d3fd57a36ca808cd15838c569256b7be8b0de3aacab14c1d7367da9e35d77475",
        "merged": "5c5a2bdc4fb65088a6d20f93530ed3fb5112bdafd1a6ee122c6066f8cea014c3",
    },
    "friend-revelation": {
        "comparison": "8e258bc543fac9b7e083fac0cdeef25c4f9607761b11c3c710faff9761ba325f",
        "decision": "786d208aec0328a328ce4c7f45b02298349426fb6b255e4d0bfb2b61944dfd5f",
        "external": "c4040686354a4e52e604033ad48c69300489ca98e8b14bc61f54ae5ad462f349",
        "machine": "c5194515a17fac5d7b82f8c26578ad8c8105f5298a480517aea7a6b6a4c193f6",
        "merged": "831e936477dd864162b6928da7570dc41ec734f40be8d229b3abd4970fda1c36",
    },
    "team-void-boss": {
        "comparison": "222c9d4b6dedaba1874790435f263af4ea731be72d5814a27aded050614a1cc0",
        "decision": "90c7c3c26578dceb3547d52a48b0aa8b4325a93539006bf92c32d75560db43c1",
        "external": "a3c0cca5525cc53ce1c77e4246a45d5c3279c00f0b9476309b544ae1b5ea89b7",
        "machine": "a931ece94d8d1f1bbf2818be7188ecb9e02945f149a7d182ade81f17172d3312",
        "merged": "f6909877186da7cf82b3020a4fdc7e4c55f825d23f2d0b553dff5e61e1d82276",
    },
    "uncertain-legal-throws": {
        "comparison": "b193264d10e2333cb18c8e4b51046a2a3e11d786e1b2596a9d76032818eae69e",
        "decision": "dc81d7596147b26d3e2a18f4d63eb9a156a5bb676110b53c55d720b96b029064",
        "external": "d871f0aa6be5bdb14d057c0fd07c57519cb528ad9be00ba1263c908346dc5648",
        "machine": "0e9356495ba18cd5733af72e58d69cb9735b3782fc93c4a5ac09b5fe7d22d63f",
        "merged": "b9b6db2e646c0c26df1457c3fd618f48ee756be69fc5d0c6043d066bbf4301df",
    },
}

_SHA256_RE = w13._SHA256_RE  # noqa: SLF001 - share the frozen strict syntax.


class SealRecoveryError(RuntimeError):
    """Raised when the exact known recovery cannot be reconstructed."""


@dataclass(frozen=True)
class RecoveryPaths:
    layout: w13.W13Layout

    @property
    def archive(self) -> Path:
        return self.layout.root / MALFORMED_ARCHIVE_RELATIVE

    @property
    def provenance(self) -> Path:
        return self.layout.root / RECOVERY_PROVENANCE_RELATIVE

    @property
    def manifest(self) -> Path:
        return self.layout.root / RECOVERY_MANIFEST_RELATIVE


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _absolute_without_symlink_resolution(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _require_safe_run_layout(layout: w13.W13Layout) -> Path:
    """Require a real run root and real W1.3 directory, with no symlink hop."""

    root = _absolute_without_symlink_resolution(layout.root)
    if root != layout.root or not root.is_dir() or root.is_symlink():
        raise SealRecoveryError("recovery run root must be one absolute real directory")
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise SealRecoveryError(f"could not resolve recovery run root: {exc}") from exc
    if resolved_root != root:
        raise SealRecoveryError("recovery run root path contains a symlink component")
    directory = layout.directory
    if (
        _absolute_without_symlink_resolution(directory) != directory
        or not _lexists(directory)
        or directory.is_symlink()
        or not directory.is_dir()
    ):
        raise SealRecoveryError("W1.3 directory must be one real non-symlink directory")
    if directory.resolve(strict=True).parent != root:
        raise SealRecoveryError("W1.3 directory escapes the recovery run root")
    return root


def _require_contained_output_leaf(
    layout: w13.W13Layout,
    path: Path,
    label: str,
    *,
    must_exist: bool,
) -> bool:
    """Check lexical containment and every real parent/leaf immediately at I/O."""

    root = _require_safe_run_layout(layout)
    candidate = _absolute_without_symlink_resolution(path)
    if candidate != path:
        raise SealRecoveryError(f"{label} path must be absolute and normalized")
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise SealRecoveryError(f"{label} path escapes the recovery run root") from exc
    if not relative.parts:
        raise SealRecoveryError(f"{label} cannot be the recovery run root")
    current = root
    for component in relative.parts[:-1]:
        current = current / component
        if not _lexists(current):
            raise SealRecoveryError(f"{label} parent directory is missing: {current}")
        if current.is_symlink() or not current.is_dir():
            raise SealRecoveryError(
                f"{label} parent component is not a real directory: {current}"
            )
        if current.resolve(strict=True) != current:
            raise SealRecoveryError(f"{label} parent component is a symlink hop: {current}")
    exists = _lexists(candidate)
    if exists and (candidate.is_symlink() or not candidate.is_file()):
        raise SealRecoveryError(f"existing {label} must be a regular non-symlink file")
    if must_exist and not exists:
        raise SealRecoveryError(f"required {label} is missing: {candidate}")
    return exists


def _require_recovery_directory(
    layout: w13.W13Layout, *, create: bool
) -> bool:
    _require_safe_run_layout(layout)
    directory = layout.root / RECOVERY_DIRECTORY_RELATIVE
    parent = directory.parent
    if parent != layout.directory or parent.is_symlink() or not parent.is_dir():
        raise SealRecoveryError("seal-recovery parent is not the real W1.3 directory")
    if _lexists(directory):
        if directory.is_symlink() or not directory.is_dir():
            raise SealRecoveryError(
                "seal-recovery path exists but is not a real non-symlink directory"
            )
        if directory.resolve(strict=True) != directory:
            raise SealRecoveryError("seal-recovery directory contains a symlink hop")
        if create:
            _fsync_directory(parent, "W1.3 directory")
        return True
    if not create:
        return False
    try:
        directory.mkdir(mode=0o755, exist_ok=False)
    except FileExistsError:
        pass
    except OSError as exc:
        raise SealRecoveryError(f"could not create seal-recovery directory: {exc}") from exc
    if directory.is_symlink() or not directory.is_dir():
        raise SealRecoveryError("created seal-recovery path is not a real directory")
    if directory.resolve(strict=True) != directory:
        raise SealRecoveryError("created seal-recovery directory escapes W1.3")
    # Persist the directory entry itself before any child artifact is created.
    _fsync_directory(parent, "W1.3 directory")
    return True


def _fsync_directory(path: Path, label: str) -> None:
    try:
        directory_fd = os.open(path, os.O_RDONLY)
    except OSError as exc:
        raise SealRecoveryError(f"could not open {label} for fsync: {exc}") from exc
    try:
        os.fsync(directory_fd)
    except OSError as exc:
        raise SealRecoveryError(f"could not fsync {label}: {exc}") from exc
    finally:
        os.close(directory_fd)


def _safe_output_json(
    layout: w13.W13Layout, path: Path, label: str
) -> dict[str, Any]:
    _require_contained_output_leaf(layout, path, label, must_exist=True)
    return _load_json(path)


def _safe_output_bytes(layout: w13.W13Layout, path: Path, label: str) -> bytes:
    _require_contained_output_leaf(layout, path, label, must_exist=True)
    return _read_bytes(path)


def _require_coordination_lock_path(layout: w13.W13Layout) -> None:
    """Harden the lock leaf; the lock remains coordination, not scientific data."""

    _require_safe_run_layout(layout)
    path = layout.base.operator_lock
    if _lexists(path) and (path.is_symlink() or not path.is_file()):
        raise SealRecoveryError("operator lock must be a regular non-symlink file")


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise SealRecoveryError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return enoch_week1.load_json_object(path)
    except (OSError, enoch_week1.ProtocolError) as exc:
        raise SealRecoveryError(f"could not load {path}: {exc}") from exc


def _read_bytes(path: Path) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise SealRecoveryError(f"required recovery input is not a regular file: {path}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise SealRecoveryError(f"could not read {path}: {exc}") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(_read_bytes(path))


def _with_fingerprint(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    frozen = dict(body)
    return {**frozen, field: enoch_week1.canonical_json_sha256(frozen)}


def _validate_fingerprint(value: Mapping[str, Any], field: str, label: str) -> str:
    fingerprint = _require_sha256(value.get(field), f"{label} {field}")
    body = dict(value)
    body.pop(field)
    if enoch_week1.canonical_json_sha256(body) != fingerprint:
        raise SealRecoveryError(f"{label} fingerprint mismatch")
    return fingerprint


def _write_or_match(
    layout: w13.W13Layout,
    path: Path,
    value: Mapping[str, Any],
    label: str,
) -> None:
    _require_contained_output_leaf(layout, path, label, must_exist=False)
    try:
        w13._write_or_match(path, value, label)  # noqa: SLF001
    except w13.W13OperatorError as exc:
        raise SealRecoveryError(str(exc)) from exc


def _git_text(workspace: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(workspace), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise SealRecoveryError(
            f"git {' '.join(arguments)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout


def _git_bytes(workspace: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(workspace), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise SealRecoveryError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout


def _recovery_git_identity(
    workspace: Path,
    *,
    recovery_commit: str | None = None,
    require_live: bool,
) -> dict[str, Any]:
    workspace = workspace.expanduser().resolve()
    top = Path(_git_text(workspace, "rev-parse", "--show-toplevel").strip()).resolve()
    if top != workspace:
        raise SealRecoveryError("recovery workspace must be the Git root")
    if require_live:
        status = _git_text(
            workspace, "status", "--porcelain=v1", "--untracked-files=all"
        )
        if status:
            raise SealRecoveryError(
                f"recovery workspace is not clean: {status.splitlines()[0]}"
            )
        head = _git_text(workspace, "rev-parse", "HEAD^{commit}").strip()
        if recovery_commit is not None and head != recovery_commit:
            raise SealRecoveryError("requested recovery commit differs from live HEAD")
    else:
        if recovery_commit is None:
            raise SealRecoveryError("stored recovery provenance has no source commit")
        head = _git_text(
            workspace, "rev-parse", f"{recovery_commit}^{{commit}}"
        ).strip()
    parents = _git_text(workspace, "rev-list", "--parents", "-n", "1", head).split()
    if parents != [head, ORIGINAL_W13_COMMIT]:
        raise SealRecoveryError(
            "recovery source must be one direct child commit of 171ee31"
        )
    raw_changes = _git_text(
        workspace,
        "diff-tree",
        "--no-commit-id",
        "--name-status",
        "-r",
        "--no-renames",
        ORIGINAL_W13_COMMIT,
        head,
    )
    parsed: list[tuple[str, str]] = []
    for line in raw_changes.splitlines():
        fields = line.split("\t")
        if len(fields) != 2:
            raise SealRecoveryError("recovery Git diff is malformed")
        parsed.append((fields[0], fields[1]))
    if parsed != [("A", path) for path in RECOVERY_COMMIT_PATHS]:
        raise SealRecoveryError(
            "recovery commit must add exactly the recovery tool and its tests"
        )
    changed_paths = []
    for status_code, relative in parsed:
        record = _git_text(workspace, "ls-tree", head, "--", relative).strip()
        fields = record.split(None, 3)
        if (
            len(fields) != 4
            or fields[0] != "100644"
            or fields[1] != "blob"
            or fields[3] != relative
        ):
            raise SealRecoveryError(
                f"recovery path is not one exact regular 100644 blob: {relative}"
            )
        blob = _git_bytes(workspace, "show", f"{head}:{relative}")
        if require_live:
            live_path = workspace / relative
            if not live_path.is_file() or live_path.is_symlink():
                raise SealRecoveryError(f"recovery source is not regular: {relative}")
            if _sha256_file(live_path) != _sha256_bytes(blob):
                raise SealRecoveryError(f"live recovery source differs from Git: {relative}")
        changed_paths.append(
            {
                "new_blob": fields[2],
                "old_blob": None,
                "path": relative,
                "sha256": _sha256_bytes(blob),
                "status": status_code,
            }
        )
    original_operator = _git_bytes(
        workspace,
        "show",
        f"{ORIGINAL_W13_COMMIT}:{w13.OPERATOR_RELATIVE.as_posix()}",
    )
    if _sha256_bytes(original_operator) != ORIGINAL_W13_OPERATOR_SHA256:
        raise SealRecoveryError("171ee31 W1.3 operator bytes changed")
    if _git_bytes(
        workspace, "show", f"{head}:{w13.OPERATOR_RELATIVE.as_posix()}"
    ) != original_operator:
        raise SealRecoveryError("recovery commit changed the original W1.3 operator")
    original_tree = _git_text(
        workspace, "rev-parse", f"{ORIGINAL_W13_COMMIT}^{{tree}}"
    ).strip()
    if original_tree != ORIGINAL_W13_TREE:
        raise SealRecoveryError("171ee31 Git tree changed")
    return {
        "base_tree_manifest_sha256": _sha256_bytes(
            _git_bytes(
                workspace,
                "ls-tree",
                "-r",
                "-z",
                "--full-tree",
                ORIGINAL_W13_COMMIT,
            )
        ),
        "changed_paths": changed_paths,
        "recovery_git_commit": head,
        "recovery_git_tree": _git_text(
            workspace, "rev-parse", f"{head}^{{tree}}"
        ).strip(),
        "git_tree_manifest_sha256": _sha256_bytes(
            _git_bytes(workspace, "ls-tree", "-r", "-z", "--full-tree", head)
        ),
        "original_w1_3_git_tree": original_tree,
        "original_w1_3_operator_blob": _git_text(
            workspace,
            "rev-parse",
            f"{ORIGINAL_W13_COMMIT}:{w13.OPERATOR_RELATIVE.as_posix()}",
        ).strip(),
    }


def _runtime_records(workspace: Path, commit: str) -> list[dict[str, str]]:
    modules = (
        (enoch_week1, "training/enoch_week1.py"),
        (enoch_week1_evidence, "training/enoch_week1_evidence.py"),
        (base_operator, "training/enoch_week1_operator.py"),
        (enoch_week1_runner, "training/enoch_week1_runner.py"),
        (w13, w13.OPERATOR_RELATIVE.as_posix()),
    )
    records = []
    for module, relative in modules:
        expected_path = (workspace / relative).resolve()
        module_path = getattr(module, "__file__", None)
        if not isinstance(module_path, str) or Path(module_path).resolve() != expected_path:
            raise SealRecoveryError(f"recovery runtime import is shadowed: {relative}")
        expected_sha = _sha256_bytes(_git_bytes(workspace, "show", f"{commit}:{relative}"))
        if _sha256_file(expected_path) != expected_sha:
            raise SealRecoveryError(f"recovery runtime differs from Git: {relative}")
        records.append({"path": relative, "sha256": expected_sha})
    return records


def build_recovery_provenance(
    workspace: Path,
    *,
    recovery_commit: str | None = None,
    live_source: bool,
) -> dict[str, Any]:
    workspace = workspace.expanduser().resolve()
    identity = _recovery_git_identity(
        workspace,
        recovery_commit=recovery_commit,
        require_live=live_source,
    )
    commit = identity["recovery_git_commit"]
    expected_source_path = (workspace / RECOVERY_SOURCE_RELATIVE).resolve()
    if Path(__file__).resolve() != expected_source_path:
        raise SealRecoveryError("executing recovery tool is outside its Git workspace")
    source_hashes = {
        relative: _sha256_bytes(_git_bytes(workspace, "show", f"{commit}:{relative}"))
        for relative in RECOVERY_COMMIT_PATHS
    }
    if _sha256_file(expected_source_path) != source_hashes[
        RECOVERY_SOURCE_RELATIVE.as_posix()
    ]:
        raise SealRecoveryError("executing recovery tool differs from its Git commit")
    test_path = workspace / RECOVERY_TEST_RELATIVE
    if not test_path.is_file() or test_path.is_symlink():
        raise SealRecoveryError("recovery test source is missing or non-regular")
    if _sha256_file(test_path) != source_hashes[RECOVERY_TEST_RELATIVE.as_posix()]:
        raise SealRecoveryError("recovery tests differ from their Git commit")
    body = {
        "automatic_production_promotion_allowed": False,
        "base_tree_manifest_sha256": identity["base_tree_manifest_sha256"],
        "changed_paths": identity["changed_paths"],
        "corrected_supported_change_set_fingerprint": CORRECTED_SUPPORTED_SET,
        "defect_id": RECOVERY_DEFECT_ID,
        "git_tree_manifest_sha256": identity["git_tree_manifest_sha256"],
        "manifest_kind": RECOVERY_PROVENANCE_KIND,
        "manifest_version": MANIFEST_VERSION,
        "malformed_supported_change_set_file_sha256": (
            MALFORMED_SUPPORTED_SET_FILE_SHA256
        ),
        "malformed_supported_change_set_fingerprint": MALFORMED_SUPPORTED_SET,
        "original_w1_3_campaign_declaration_fingerprint": ORIGINAL_W13_DECLARATION,
        "original_w1_3_continuation_provenance_fingerprint": ORIGINAL_W13_PROVENANCE,
        "original_w1_3_final_ledger_fingerprint": FINAL_W13_LEDGER,
        "original_w1_3_git_commit": ORIGINAL_W13_COMMIT,
        "original_w1_3_git_tree": identity["original_w1_3_git_tree"],
        "original_w1_3_operator_blob": identity["original_w1_3_operator_blob"],
        "original_w1_3_operator_sha256": ORIGINAL_W13_OPERATOR_SHA256,
        "protocol_fingerprint": w13.PROTOCOL_FINGERPRINT,
        "recovery_git_commit": commit,
        "recovery_git_tree": identity["recovery_git_tree"],
        "recovery_source_path": RECOVERY_SOURCE_RELATIVE.as_posix(),
        "recovery_source_sha256": source_hashes[RECOVERY_SOURCE_RELATIVE.as_posix()],
        "recovery_test_path": RECOVERY_TEST_RELATIVE.as_posix(),
        "recovery_test_sha256": source_hashes[RECOVERY_TEST_RELATIVE.as_posix()],
        "runtime_imports": _runtime_records(workspace, commit),
        "seed_claim_or_evaluator_execution_allowed": False,
    }
    return _with_fingerprint(body, "seal_recovery_provenance_fingerprint")


def validate_recovery_provenance(
    artifact: Mapping[str, Any], workspace: Path, *, live_source: bool
) -> str:
    if not isinstance(artifact, Mapping):
        raise SealRecoveryError("recovery provenance must be an object")
    commit = artifact.get("recovery_git_commit")
    if not isinstance(commit, str):
        raise SealRecoveryError("recovery provenance lacks its source commit")
    expected = build_recovery_provenance(
        workspace,
        recovery_commit=commit,
        live_source=live_source,
    )
    if dict(artifact) != expected:
        raise SealRecoveryError("seal recovery provenance does not reconstruct")
    return expected["seal_recovery_provenance_fingerprint"]


def _blocked_api(name: str):
    def blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise SealRecoveryError(f"offline seal recovery forbids {name}")

    return blocked


@contextlib.contextmanager
def _offline_api_guard() -> Iterator[None]:
    """Process-local tripwires for every evaluation and seed-claim entry point."""

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
    originals = []
    for owner, name in targets:
        originals.append((owner, name, getattr(owner, name)))
        setattr(owner, name, _blocked_api(f"{owner.__name__}.{name}"))
    try:
        yield
    finally:
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)


def _attempt_is_exact(attempt: Path, arm_id: str) -> None:
    arm_root = attempt.parents[1]
    attempts = w13._attempts(arm_root)  # noqa: SLF001
    if [path.name for path in attempts] != ["attempt-001"] or attempt.name != "attempt-001":
        raise SealRecoveryError(f"{arm_id} does not have the exact sole attempt-001")
    for marker in (
        "failure-tombstone.json",
        "preclaim-abandoned.json",
        "preclaim-failure.json",
    ):
        if (attempt / marker).exists():
            raise SealRecoveryError(f"{arm_id} has an unexpected failure marker: {marker}")


def _load_validated_context(
    layout: w13.W13Layout,
    workspace: Path,
    *,
    environment: Mapping[str, str],
    require_live_ledger_exact: bool,
) -> dict[str, Any]:
    stored_plan = w13._load_json(layout.input)  # noqa: SLF001
    state = w13._load_parent_state(layout, stored_plan)  # noqa: SLF001
    plan, provenance, declaration = w13._load_and_validate_declaration_state(  # noqa: SLF001
        layout,
        workspace,
        state,
        environment=environment,
        live_source=False,
    )
    if provenance["continuation_provenance_fingerprint"] != ORIGINAL_W13_PROVENANCE:
        raise SealRecoveryError("original W1.3 provenance is not the known artifact")
    if provenance["continuation_git_commit"] != ORIGINAL_W13_COMMIT:
        raise SealRecoveryError("original W1.3 source commit is not 171ee31")
    if declaration["campaign_declaration_fingerprint"] != ORIGINAL_W13_DECLARATION:
        raise SealRecoveryError("original W1.3 declaration is not the known artifact")
    if layout.retirement.exists():
        raise SealRecoveryError("the protocol is retired; seal recovery is forbidden")
    ledger = w13._load_json(layout.final_ledger)  # noqa: SLF001
    w13._expected_final_ledger(  # noqa: SLF001
        state["protocol"], ledger, state["parent_ledger"], declaration
    )
    if (
        ledger["ledger_fingerprint"] != FINAL_W13_LEDGER
        or len(ledger["consumed"]) != w13.FINAL_COUNT
    ):
        raise SealRecoveryError("W1.3 final ledger is not the known 18,711-claim snapshot")
    live_ledger = w13._load_json(layout.base.ledger)  # noqa: SLF001
    if require_live_ledger_exact:
        if live_ledger != ledger or _read_bytes(layout.base.ledger) != _read_bytes(
            layout.final_ledger
        ):
            raise SealRecoveryError(
                "recovery requires the live ledger to byte-match the W1.3 final snapshot"
            )
    else:
        w13._validate_ledger_extension(state["protocol"], live_ledger, ledger)  # noqa: SLF001

    arm_evidence: dict[str, dict[str, Any]] = {}
    for arm_id in w13.SURVIVOR_ARM_IDS:
        arm = w13._arm_by_id(declaration, arm_id)  # noqa: SLF001
        arm_root = layout.arm(arm["sequence"], arm_id)
        attempt = w13._completed_attempt(arm_root)  # noqa: SLF001
        if attempt is None:
            raise SealRecoveryError(f"missing completed W1.3 arm: {arm_id}")
        _attempt_is_exact(attempt, arm_id)
        evidence = w13._validate_completed_arm(  # noqa: SLF001
            layout,
            state["protocol"],
            ledger,
            declaration,
            arm,
            attempt,
            base_environment=environment,
            allow_decision_write=False,
        )
        expected = EXPECTED_ARM_LINEAGE[arm_id]
        observed = {
            "comparison": evidence["comparison"]["comparison_protocol_fingerprint"],
            "decision": evidence["support_decision"]["support_decision_fingerprint"],
            "external": evidence["external_evidence"][
                "verified_external_evidence_fingerprint"
            ],
            "machine": evidence["machine_attestation_fingerprint"],
            "merged": evidence["merged_result"]["merged_result_fingerprint"],
        }
        if observed != expected:
            raise SealRecoveryError(f"{arm_id} differs from the exact known W1.3 evidence")
        arm_evidence[arm_id] = evidence
    return {
        "arm_evidence": arm_evidence,
        "declaration": declaration,
        "ledger": ledger,
        "plan": plan,
        "provenance": provenance,
        "state": state,
    }


def _protected_snapshot(layout: w13.W13Layout) -> dict[str, Any]:
    _require_safe_run_layout(layout)
    paths = [
        layout.input,
        layout.provenance,
        layout.declaration,
        layout.final_ledger,
    ]
    arms_root = layout.directory / "arms"
    if not arms_root.is_dir() or arms_root.is_symlink():
        raise SealRecoveryError("W1.3 arms directory is missing or non-regular")
    paths.extend(path for path in arms_root.rglob("*") if path.is_file())
    records = []
    for path in sorted(set(paths)):
        if path.is_symlink():
            raise SealRecoveryError(f"protected W1.3 artifact is a symlink: {path}")
        records.append(
            {
                "path": path.relative_to(layout.root).as_posix(),
                "sha256": _sha256_file(path),
            }
        )
    files_sha256 = enoch_week1.canonical_json_sha256(records)
    if len(records) != EXPECTED_PROTECTED_FILE_COUNT:
        raise SealRecoveryError(
            "protected W1.3 artifact inventory is not the exact known 209-file state"
        )
    if files_sha256 != EXPECTED_PROTECTED_FILES_SHA256:
        raise SealRecoveryError("protected W1.3 artifact inventory hash changed")
    return {
        "file_count": len(records),
        "files_sha256": files_sha256,
    }


def _atomic_archive_bytes(
    layout: w13.W13Layout, path: Path, payload: bytes
) -> None:
    _require_recovery_directory(layout, create=True)
    if _require_contained_output_leaf(
        layout, path, "malformed supported-set archive", must_exist=False
    ):
        if _safe_output_bytes(
            layout, path, "malformed supported-set archive"
        ) != payload:
            raise SealRecoveryError(f"existing malformed archive changed: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        _require_recovery_directory(layout, create=False)
        _require_contained_output_leaf(
            layout, path, "malformed supported-set archive", must_exist=False
        )
        try:
            os.link(temporary, path)
        except FileExistsError:
            if _safe_output_bytes(
                layout, path, "malformed supported-set archive"
            ) != payload:
                raise SealRecoveryError(f"racing malformed archive changed: {path}")
        _require_recovery_directory(layout, create=False)
        _fsync_directory(path.parent, "seal-recovery directory")
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _validate_malformed_bytes(payload: bytes) -> dict[str, Any]:
    if len(payload) != MALFORMED_SUPPORTED_SET_FILE_SIZE:
        raise SealRecoveryError("malformed supported-set byte size changed")
    if _sha256_bytes(payload) != MALFORMED_SUPPORTED_SET_FILE_SHA256:
        raise SealRecoveryError("malformed supported-set file hash changed")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SealRecoveryError(f"malformed supported-set archive is not JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SealRecoveryError("malformed supported-set archive root is not an object")
    if value.get("supported_change_set_fingerprint") != MALFORMED_SUPPORTED_SET:
        raise SealRecoveryError("malformed supported-set semantic fingerprint changed")
    _validate_fingerprint(
        value,
        "supported_change_set_fingerprint",
        "malformed supported-independent-change-set",
    )
    return value


def _ensure_malformed_archive(
    layout: w13.W13Layout,
    paths: RecoveryPaths,
    *,
    require_archive: bool,
) -> tuple[dict[str, Any], str, bytes]:
    recovery_directory_exists = _require_recovery_directory(layout, create=False)
    archive_exists = (
        _require_contained_output_leaf(
            layout,
            paths.archive,
            "malformed supported-set archive",
            must_exist=False,
        )
        if recovery_directory_exists
        else False
    )
    archive_payload = (
        _safe_output_bytes(layout, paths.archive, "malformed supported-set archive")
        if archive_exists
        else None
    )
    if archive_payload is not None:
        malformed = _validate_malformed_bytes(archive_payload)
    else:
        malformed = None

    canonical = _safe_output_json(
        layout, layout.supported_set, "canonical supported-independent-change-set"
    )
    canonical_fingerprint = canonical.get("supported_change_set_fingerprint")
    if canonical_fingerprint == MALFORMED_SUPPORTED_SET:
        canonical_payload = _safe_output_bytes(
            layout,
            layout.supported_set,
            "canonical supported-independent-change-set",
        )
        canonical_malformed = _validate_malformed_bytes(canonical_payload)
        if malformed is not None and archive_payload != canonical_payload:
            raise SealRecoveryError("malformed archive bytes differ from the canonical defect")
        if malformed is None:
            if require_archive:
                raise SealRecoveryError("malformed supported-set archive is missing")
            malformed = canonical_malformed
        return malformed, "malformed", canonical_payload
    if canonical_fingerprint == CORRECTED_SUPPORTED_SET:
        if malformed is None:
            raise SealRecoveryError("corrected supported-set exists without its archive")
        if _sha256_bytes(
            _safe_output_bytes(
                layout,
                layout.supported_set,
                "canonical supported-independent-change-set",
            )
        ) != CORRECTED_SUPPORTED_SET_FILE_SHA256:
            raise SealRecoveryError("corrected supported-set file hash changed")
        return malformed, "corrected", archive_payload
    raise SealRecoveryError("canonical supported-set is neither the known defect nor correction")


def _normalize_arm_evidence(
    context: Mapping[str, Any], malformed: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    state = context["state"]
    declaration = context["declaration"]
    ledger = context["ledger"]
    original_evidence = context["arm_evidence"]
    w13.validate_supported_change_set(
        malformed,
        state["protocol"],
        declaration,
        state["fixture"],
        ledger,
        original_evidence,
    )
    malformed_by_arm = {
        record["arm_id"]: record for record in malformed["arm_results"]
    }
    if list(malformed_by_arm) != list(w13.SURVIVOR_ARM_IDS):
        raise SealRecoveryError("malformed supported-set arm order changed")

    normalized = copy.deepcopy(original_evidence)
    corrections = []
    for arm_id in w13.SURVIVOR_ARM_IDS:
        evidence = original_evidence[arm_id]
        counters = enoch_week1_evidence.validate_verified_external_evidence(
            state["protocol"], evidence["comparison"], evidence["external_evidence"]
        )
        if not isinstance(counters, dict):
            raise SealRecoveryError("external evidence validator no longer returns counters")
        if evidence["external_evidence_fingerprint"] != counters:
            raise SealRecoveryError(f"{arm_id} no longer exhibits the exact known defect")
        malformed_value = malformed_by_arm[arm_id].get("external_evidence_fingerprint")
        if malformed_value != counters:
            raise SealRecoveryError(
                f"{arm_id} malformed field does not equal validated evidence counters"
            )
        corrected_fingerprint = _require_sha256(
            evidence["external_evidence"].get(
                "verified_external_evidence_fingerprint"
            ),
            f"{arm_id} corrected external evidence fingerprint",
        )
        if corrected_fingerprint != EXPECTED_ARM_LINEAGE[arm_id]["external"]:
            raise SealRecoveryError(f"{arm_id} external evidence digest changed")
        normalized[arm_id]["external_evidence_fingerprint"] = corrected_fingerprint
        corrections.append(
            {
                "arm_id": arm_id,
                "comparison_protocol_fingerprint": EXPECTED_ARM_LINEAGE[arm_id][
                    "comparison"
                ],
                "corrected_external_evidence_fingerprint": corrected_fingerprint,
                "external_evidence_file_sha256": _sha256_file(
                    evidence["attempt"] / "external-evidence.json"
                ),
                "machine_attestation_fingerprint": EXPECTED_ARM_LINEAGE[arm_id][
                    "machine"
                ],
                "malformed_value_equals_validated_counters": True,
                "malformed_value_sha256": enoch_week1.canonical_json_sha256(counters),
                "merged_result_fingerprint": EXPECTED_ARM_LINEAGE[arm_id]["merged"],
                "support_decision_fingerprint": EXPECTED_ARM_LINEAGE[arm_id][
                    "decision"
                ],
            }
        )
    corrected = w13.build_supported_change_set(
        state["protocol"],
        declaration,
        state["fixture"],
        ledger,
        normalized,
    )
    if corrected["supported_change_set_fingerprint"] != CORRECTED_SUPPORTED_SET:
        raise SealRecoveryError("corrected supported-set fingerprint is not the known value")
    w13.validate_supported_change_set(
        corrected,
        state["protocol"],
        declaration,
        state["fixture"],
        ledger,
        normalized,
    )
    reconstructed_malformed = copy.deepcopy(corrected)
    for record in reconstructed_malformed["arm_results"]:
        record["external_evidence_fingerprint"] = malformed_by_arm[
            record["arm_id"]
        ]["external_evidence_fingerprint"]
    body = dict(reconstructed_malformed)
    body.pop("supported_change_set_fingerprint")
    reconstructed_malformed["supported_change_set_fingerprint"] = (
        enoch_week1.canonical_json_sha256(body)
    )
    if reconstructed_malformed != dict(malformed):
        raise SealRecoveryError("known correction changes more than the five malformed fields")
    return normalized, corrected, corrections


def build_recovery_manifest(
    context: Mapping[str, Any],
    recovery_provenance: Mapping[str, Any],
    corrected: Mapping[str, Any],
    corrections: Sequence[Mapping[str, Any]],
    protected_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    corrected_bytes = enoch_week1.canonical_json_bytes(corrected) + b"\n"
    corrected_file_sha256 = _sha256_bytes(corrected_bytes)
    if corrected_file_sha256 != CORRECTED_SUPPORTED_SET_FILE_SHA256:
        raise SealRecoveryError("corrected supported-set canonical file hash changed")
    body = {
        "arm_evidence_corrections": [dict(record) for record in corrections],
        "automatic_production_promotion_allowed": False,
        "corrected_supported_change_set_file_sha256": corrected_file_sha256,
        "corrected_supported_change_set_fingerprint": CORRECTED_SUPPORTED_SET,
        "corrected_supported_change_set_path": (
            context["layout"].supported_set.relative_to(context["layout"].root).as_posix()
        ),
        "defect_id": RECOVERY_DEFECT_ID,
        "evaluation_invocation_count": 0,
        "final_consumed_count": w13.FINAL_COUNT,
        "final_ledger_fingerprint": FINAL_W13_LEDGER,
        "malformed_archive_file_sha256": MALFORMED_SUPPORTED_SET_FILE_SHA256,
        "malformed_archive_path": MALFORMED_ARCHIVE_RELATIVE.as_posix(),
        "malformed_supported_change_set_fingerprint": MALFORMED_SUPPORTED_SET,
        "manifest_kind": RECOVERY_MANIFEST_KIND,
        "manifest_version": MANIFEST_VERSION,
        "original_w1_3_campaign_declaration_fingerprint": ORIGINAL_W13_DECLARATION,
        "original_w1_3_continuation_provenance_fingerprint": ORIGINAL_W13_PROVENANCE,
        "original_w1_3_git_commit": ORIGINAL_W13_COMMIT,
        "protected_w1_3_artifact_snapshot": dict(protected_snapshot),
        "protocol_fingerprint": w13.PROTOCOL_FINGERPRINT,
        "seal_recovery_provenance_fingerprint": recovery_provenance[
            "seal_recovery_provenance_fingerprint"
        ],
        "seed_claim_count": 0,
    }
    return _with_fingerprint(body, "seal_recovery_manifest_fingerprint")


def validate_recovery_manifest(
    artifact: Mapping[str, Any],
    context: Mapping[str, Any],
    recovery_provenance: Mapping[str, Any],
    corrected: Mapping[str, Any],
    corrections: Sequence[Mapping[str, Any]],
    protected_snapshot: Mapping[str, Any],
) -> str:
    expected = build_recovery_manifest(
        context,
        recovery_provenance,
        corrected,
        corrections,
        protected_snapshot,
    )
    if dict(artifact) != expected:
        raise SealRecoveryError("seal recovery manifest does not reconstruct")
    return expected["seal_recovery_manifest_fingerprint"]


def _build_recovered_phase(
    context: Mapping[str, Any],
    recovery_provenance: Mapping[str, Any],
    recovery_manifest: Mapping[str, Any],
    corrected: Mapping[str, Any],
    normalized_evidence: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    state = context["state"]
    declaration = context["declaration"]
    provenance = context["provenance"]
    ledger = context["ledger"]
    ordinary_phase = w13._build_phase3(  # noqa: SLF001
        state["protocol"],
        state["phase2"],
        provenance,
        declaration,
        state["fixture"],
        corrected,
        ledger,
        normalized_evidence,
    )
    artifacts = {
        record["artifact_id"]: record["sha256"]
        for record in ordinary_phase["artifacts"]
    }
    artifacts.update(
        {
            "w1.3/malformed-supported-independent-change-set": (
                MALFORMED_SUPPORTED_SET
            ),
            "w1.3/seal-recovery-manifest": recovery_manifest[
                "seal_recovery_manifest_fingerprint"
            ],
            "w1.3/seal-recovery-provenance": recovery_provenance[
                "seal_recovery_provenance_fingerprint"
            ],
        }
    )
    declarations = dict(ordinary_phase["declarations"])
    declarations.update({
        "recovery_defect_id": RECOVERY_DEFECT_ID,
        "recovery_evaluation_invocation_count": 0,
        "recovery_malformed_supported_change_set_fingerprint": (
            MALFORMED_SUPPORTED_SET
        ),
        "recovery_seed_claim_count": 0,
        "recovery_source_commit": recovery_provenance["recovery_git_commit"],
        "seal_recovery_manifest_fingerprint": recovery_manifest[
            "seal_recovery_manifest_fingerprint"
        ],
        "seal_recovery_provenance_fingerprint": recovery_provenance[
            "seal_recovery_provenance_fingerprint"
        ],
    })
    return enoch_week1.build_phase_manifest(
        state["protocol"],
        "W1.3",
        artifacts=artifacts,
        declarations=declarations,
        parent_phase_manifests=[state["phase2"]],
    )


def _validate_recovered_phase(
    phase: Mapping[str, Any],
    context: Mapping[str, Any],
    recovery_provenance: Mapping[str, Any],
    recovery_manifest: Mapping[str, Any],
    corrected: Mapping[str, Any],
    normalized_evidence: Mapping[str, Mapping[str, Any]],
) -> str:
    expected = _build_recovered_phase(
        context,
        recovery_provenance,
        recovery_manifest,
        corrected,
        normalized_evidence,
    )
    if dict(phase) != expected:
        raise SealRecoveryError("recovery-aware W1.3 phase does not reconstruct")
    required = {
        "supported-independent-change-set": CORRECTED_SUPPORTED_SET,
        "w1.3/malformed-supported-independent-change-set": MALFORMED_SUPPORTED_SET,
        "w1.3/seal-recovery-manifest": recovery_manifest[
            "seal_recovery_manifest_fingerprint"
        ],
        "w1.3/seal-recovery-provenance": recovery_provenance[
            "seal_recovery_provenance_fingerprint"
        ],
    }
    observed = {item["artifact_id"]: item["sha256"] for item in phase["artifacts"]}
    for artifact_id, fingerprint in required.items():
        if observed.get(artifact_id) != fingerprint:
            raise SealRecoveryError(f"recovered phase omits or changes {artifact_id}")
    chain = enoch_week1.validate_phase_chain(
        context["state"]["protocol"],
        [
            context["state"]["phase0"],
            context["state"]["phase1"],
            context["state"]["phase2"],
            expected,
        ],
    )
    return chain[-1]


def _install_corrected_supported_set(
    layout: w13.W13Layout,
    corrected: Mapping[str, Any],
    canonical_state: str,
) -> None:
    _require_contained_output_leaf(
        layout,
        layout.supported_set,
        "canonical supported-independent-change-set",
        must_exist=True,
    )
    if canonical_state == "corrected":
        if (
            _safe_output_json(
                layout,
                layout.supported_set,
                "canonical supported-independent-change-set",
            )
            != dict(corrected)
            or _sha256_bytes(
                _safe_output_bytes(
                    layout,
                    layout.supported_set,
                    "canonical supported-independent-change-set",
                )
            )
            != CORRECTED_SUPPORTED_SET_FILE_SHA256
        ):
            raise SealRecoveryError("existing corrected supported-set changed")
        return
    if canonical_state != "malformed":
        raise SealRecoveryError("unsupported canonical supported-set state")
    _validate_malformed_bytes(
        _safe_output_bytes(
            layout,
            layout.supported_set,
            "canonical supported-independent-change-set",
        )
    )
    _require_contained_output_leaf(
        layout,
        layout.supported_set,
        "canonical supported-independent-change-set",
        must_exist=True,
    )
    try:
        enoch_week1.atomic_write_json(layout.supported_set, corrected, overwrite=True)
    except OSError as exc:
        raise SealRecoveryError(f"could not atomically install corrected set: {exc}") from exc


def _prepare_recovery(
    layout: w13.W13Layout,
    workspace: Path,
    *,
    environment: Mapping[str, str],
    live_source: bool,
    require_live_ledger_exact: bool,
    require_archive: bool,
) -> dict[str, Any]:
    _require_safe_run_layout(layout)
    paths = RecoveryPaths(layout)
    context = _load_validated_context(
        layout,
        workspace,
        environment=environment,
        require_live_ledger_exact=require_live_ledger_exact,
    )
    context["layout"] = layout
    protected = _protected_snapshot(layout)
    malformed, canonical_state, malformed_payload = _ensure_malformed_archive(
        layout, paths, require_archive=require_archive
    )
    normalized, corrected, corrections = _normalize_arm_evidence(context, malformed)
    if live_source:
        recovery_provenance = build_recovery_provenance(
            workspace, live_source=True
        )
    else:
        stored_provenance = _safe_output_json(
            layout, paths.provenance, "seal recovery provenance"
        )
        validate_recovery_provenance(
            stored_provenance, workspace, live_source=False
        )
        recovery_provenance = stored_provenance
    recovery_manifest = build_recovery_manifest(
        context,
        recovery_provenance,
        corrected,
        corrections,
        protected,
    )
    phase = _build_recovered_phase(
        context,
        recovery_provenance,
        recovery_manifest,
        corrected,
        normalized,
    )
    recovery_directory_exists = _require_recovery_directory(layout, create=False)
    has_provenance = (
        _require_contained_output_leaf(
            layout,
            paths.provenance,
            "seal recovery provenance",
            must_exist=False,
        )
        if recovery_directory_exists
        else False
    )
    has_manifest = (
        _require_contained_output_leaf(
            layout,
            paths.manifest,
            "seal recovery manifest",
            must_exist=False,
        )
        if recovery_directory_exists
        else False
    )
    has_archive = (
        _require_contained_output_leaf(
            layout,
            paths.archive,
            "malformed supported-set archive",
            must_exist=False,
        )
        if recovery_directory_exists
        else False
    )
    has_phase = _require_contained_output_leaf(
        layout, layout.phase, "W1.3 phase manifest", must_exist=False
    )
    if has_provenance and _safe_output_json(
        layout, paths.provenance, "seal recovery provenance"
    ) != recovery_provenance:
        raise SealRecoveryError("existing recovery provenance changed")
    if has_manifest and _safe_output_json(
        layout, paths.manifest, "seal recovery manifest"
    ) != recovery_manifest:
        raise SealRecoveryError("existing recovery manifest changed")
    if has_manifest and not has_provenance:
        raise SealRecoveryError("recovery manifest exists without recovery provenance")
    if canonical_state == "corrected" and not (has_provenance and has_manifest):
        raise SealRecoveryError("corrected supported-set exists before recovery metadata")
    if has_phase and not (
        has_archive
        and has_provenance
        and has_manifest
        and canonical_state == "corrected"
    ):
        raise SealRecoveryError("recovered phase exists before its complete recovery prefix")
    if has_phase:
        _validate_recovered_phase(
            _safe_output_json(layout, layout.phase, "W1.3 phase manifest"),
            context,
            recovery_provenance,
            recovery_manifest,
            corrected,
            normalized,
        )
    partial_exists = has_provenance or has_manifest or has_phase
    if (partial_exists or canonical_state == "corrected") and not has_archive:
        raise SealRecoveryError("recovery partial state exists without malformed archive")
    return {
        "canonical_state": canonical_state,
        "context": context,
        "corrected": corrected,
        "corrections": corrections,
        "malformed": malformed,
        "malformed_payload": malformed_payload,
        "normalized_evidence": normalized,
        "paths": paths,
        "phase": phase,
        "protected_snapshot": protected,
        "recovery_manifest": recovery_manifest,
        "recovery_provenance": recovery_provenance,
    }


def recover_seal(
    layout: w13.W13Layout,
    workspace: Path,
    w1_2_workspace: Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Recover only the W1.3 derived seal; never evaluate or claim."""

    workspace = workspace.expanduser().resolve()
    base_environment = dict(os.environ if environment is None else environment)
    _require_safe_run_layout(layout)
    _require_coordination_lock_path(layout)
    with _offline_api_guard():
        w13.verify_sealed_w1_2(layout, w1_2_workspace)
        with base_operator._operator_lock(layout.base.operator_lock):  # noqa: SLF001
            prepared = _prepare_recovery(
                layout,
                workspace,
                environment=base_environment,
                live_source=True,
                require_live_ledger_exact=True,
                require_archive=False,
            )
            paths = prepared["paths"]
            live_ledger_before = _read_bytes(layout.base.ledger)
            _atomic_archive_bytes(
                layout, paths.archive, prepared["malformed_payload"]
            )
            _write_or_match(
                layout,
                paths.provenance,
                prepared["recovery_provenance"],
                "W1.3 seal recovery provenance",
            )
            _write_or_match(
                layout,
                paths.manifest,
                prepared["recovery_manifest"],
                "W1.3 seal recovery manifest",
            )
            _install_corrected_supported_set(
                layout, prepared["corrected"], prepared["canonical_state"]
            )
            if _sha256_bytes(
                _safe_output_bytes(
                    layout,
                    layout.supported_set,
                    "canonical supported-independent-change-set",
                )
            ) != CORRECTED_SUPPORTED_SET_FILE_SHA256:
                raise SealRecoveryError("installed corrected supported-set file hash changed")
            if _protected_snapshot(layout) != prepared["protected_snapshot"]:
                raise SealRecoveryError("recovery changed a protected W1.3 artifact")
            if _read_bytes(layout.base.ledger) != live_ledger_before:
                raise SealRecoveryError("recovery changed the live seed ledger")
            _validate_malformed_bytes(
                _safe_output_bytes(
                    layout, paths.archive, "malformed supported-set archive"
                )
            )
            validate_recovery_provenance(
                _safe_output_json(
                    layout, paths.provenance, "seal recovery provenance"
                ),
                workspace,
                live_source=True,
            )
            validate_recovery_manifest(
                _safe_output_json(layout, paths.manifest, "seal recovery manifest"),
                prepared["context"],
                prepared["recovery_provenance"],
                prepared["corrected"],
                prepared["corrections"],
                prepared["protected_snapshot"],
            )
            _validate_recovered_phase(
                prepared["phase"],
                prepared["context"],
                prepared["recovery_provenance"],
                prepared["recovery_manifest"],
                prepared["corrected"],
                prepared["normalized_evidence"],
            )
            _write_or_match(
                layout,
                layout.phase,
                prepared["phase"],
                "recovery-aware W1.3 phase manifest",
            )
            phase_fingerprint = _validate_recovered_phase(
                _safe_output_json(layout, layout.phase, "W1.3 phase manifest"),
                prepared["context"],
                prepared["recovery_provenance"],
                prepared["recovery_manifest"],
                prepared["corrected"],
                prepared["normalized_evidence"],
            )
            return {
                "corrected_supported_change_set_fingerprint": CORRECTED_SUPPORTED_SET,
                "final_ledger_fingerprint": FINAL_W13_LEDGER,
                "phase_manifest_fingerprint": phase_fingerprint,
                "seal_recovery_manifest_fingerprint": prepared["recovery_manifest"][
                    "seal_recovery_manifest_fingerprint"
                ],
                "seal_recovery_provenance_fingerprint": prepared[
                    "recovery_provenance"
                ]["seal_recovery_provenance_fingerprint"],
                "status": prepared["corrected"]["status"],
                "supported_arm_ids": prepared["corrected"]["summary"][
                    "supported_arm_ids"
                ],
            }


def verify_recovered_seal(
    layout: w13.W13Layout,
    workspace: Path,
    w1_2_workspace: Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Reconstruct the recovery from disk without writing or evaluating."""

    workspace = workspace.expanduser().resolve()
    base_environment = dict(os.environ if environment is None else environment)
    _require_safe_run_layout(layout)
    _require_coordination_lock_path(layout)
    with _offline_api_guard():
        w13.verify_sealed_w1_2(layout, w1_2_workspace)
        with base_operator._operator_lock(layout.base.operator_lock):  # noqa: SLF001
            _require_contained_output_leaf(
                layout,
                layout.phase,
                "recovery-aware W1.3 phase",
                must_exist=True,
            )
            prepared = _prepare_recovery(
                layout,
                workspace,
                environment=base_environment,
                live_source=False,
                require_live_ledger_exact=False,
                require_archive=True,
            )
            if prepared["canonical_state"] != "corrected":
                raise SealRecoveryError("canonical supported-set is not corrected")
            stored_manifest = _safe_output_json(
                layout, prepared["paths"].manifest, "seal recovery manifest"
            )
            manifest_fingerprint = validate_recovery_manifest(
                stored_manifest,
                prepared["context"],
                prepared["recovery_provenance"],
                prepared["corrected"],
                prepared["corrections"],
                prepared["protected_snapshot"],
            )
            phase_fingerprint = _validate_recovered_phase(
                _safe_output_json(layout, layout.phase, "W1.3 phase manifest"),
                prepared["context"],
                prepared["recovery_provenance"],
                stored_manifest,
                prepared["corrected"],
                prepared["normalized_evidence"],
            )
            return {
                "corrected_supported_change_set_fingerprint": CORRECTED_SUPPORTED_SET,
                "final_ledger_fingerprint": FINAL_W13_LEDGER,
                "phase_manifest_fingerprint": phase_fingerprint,
                "seal_recovery_manifest_fingerprint": manifest_fingerprint,
                "seal_recovery_provenance_fingerprint": prepared[
                    "recovery_provenance"
                ]["seal_recovery_provenance_fingerprint"],
                "status": prepared["corrected"]["status"],
                "supported_arm_ids": prepared["corrected"]["summary"][
                    "supported_arm_ids"
                ],
            }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("recover", "derive and seal the known W1.3 metadata-only recovery"),
        ("verify", "offline-verify the recovery-aware W1.3 seal"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--root", required=True, type=Path)
        command.add_argument("--workspace", type=Path, default=Path.cwd())
        command.add_argument("--w1-2-workspace", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    layout = w13.W13Layout(args.root.expanduser().resolve())
    try:
        if args.command == "recover":
            result = recover_seal(layout, args.workspace, args.w1_2_workspace)
        else:
            result = verify_recovered_seal(
                layout, args.workspace, args.w1_2_workspace
            )
    except (
        SealRecoveryError,
        w13.W13OperatorError,
        base_operator.OperatorError,
        enoch_week1.ProtocolError,
        enoch_week1_evidence.EvidenceError,
        enoch_week1_runner.RunnerError,
        FileExistsError,
        OSError,
    ) as exc:
        print(f"W1.3 seal recovery failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
