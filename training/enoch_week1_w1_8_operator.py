#!/usr/bin/env python3
"""Authoritative metadata-only W1.8 retain-control seal and verifier."""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence

try:
    import fcntl
except ImportError:  # pragma: no cover - authoritative operation requires Unix.
    fcntl = None  # type: ignore[assignment]

try:
    from training import enoch_week1
    from training import enoch_week1_operator as base_operator
    from training import enoch_week1_w1_4_operator as w14
except ImportError:  # pragma: no cover - direct-script import path.
    import enoch_week1  # type: ignore[no-redef]
    import enoch_week1_operator as base_operator  # type: ignore[no-redef]
    import enoch_week1_w1_4_operator as w14  # type: ignore[no-redef]


MANIFEST_VERSION = 1
BASE_COMMIT = "6dfbce7cbcf2098500dc185719fc4481edc8aa8e"
PROTOCOL_FINGERPRINT = "a1e48199e6cb153c68f442cac9f28400798b994d154e03d34cb64420e21db2b7"
CONTROL_MANIFEST_FINGERPRINT = (
    "1aeb0c4f7d62d606eb554cf50aa83250a3672dd806587e695981196766e620f2"
)
PERMANENT_ENOCH0_FINGERPRINT = (
    "5243de72c6669e233c1528f42f5de8e4b578165f55b0964dd48c3326df551e62"
)
PARENT_PHASE_FINGERPRINT = (
    "ad353c33adda2a31dfb1a26c63e3802b8584aee0b711d457d8c1fa60dd88a399"
)
PARENT_EXIT_FINGERPRINT = (
    "d3a449cf29a98f14bcfdcdca7bafe1c5ba9bafbbebd56a21f9d4617f11eddf8b"
)
PARENT_DECISION_FINGERPRINT = (
    "cd80e82024d4d97b71f53d680883a5a0f462fa2792b13770355216a07ede95ae"
)
EVALUATED_CANDIDATE_FINGERPRINT = (
    "abebed7aee684612282e37513b37592bf3fe64f96a34f597870fe72cf3bbe706"
)
FINAL_LEDGER_FINGERPRINT = (
    "17d6eecf5a8119946de82853e34e314a1b3a4a4b18404c3209c217fd52c0fbea"
)
FINAL_CONSUMED_COUNT = 19_811
PHASE_CHAIN_SHA256 = (
    "d18a5fa8ed922636dc0c98397d735baa271867c1bfd324c9e99196b21baec29e"
)
TERMINAL_DECISION_FINGERPRINT = (
    "799be65915b7c60a280e03cbcd038e94bfc3af7340763c86dfe2b93070e5bd84"
)
EXPECTED_EVIDENCE_COUNT = 118
PARENT_PRESERVED_INVENTORY = {
    "aggregate_bytes": 2_466_144_135,
    "file_count": 5_103,
    "manifest_sha256": "c3772664722f3fa46ca3b5d1489382c84fd3079a3a13bb006c38ba8923c267e4",
}
SKIPPED_PHASES = ("W1.5", "W1.6", "W1.7")

PHASE_FINGERPRINTS = (
    "6b2b63187981da256476256b673d2f809ed4ede3089790233a249b50fe5d937c",
    "24503b8909df96b6bb1cf47a91da3e830ea623c4ead6db98f9c4881b9043da63",
    "70d8a05c0e17a372e42c80e93d760ebf5d9fcb26c03c5474b06d5e3bbf214264",
    "1059c449de8b4f181e1887f0064f4544ac8286b152094025246e11a3830b3a5e",
    PARENT_PHASE_FINGERPRINT,
)

EXPECTED_W14_RESULT = {
    "campaign_declaration_fingerprint": (
        "2e8b9a6f530c5595b8b6ba32367fbc9fbcb971cb2e919f2fb8d907126db59765"
    ),
    "candidate_decision_fingerprint": PARENT_DECISION_FINGERPRINT,
    "combination_regression_gate_fingerprint": (
        "abef9967133b0c86af263f6550b7b9381abfbdcf6525c50209be5d5fa19c6241"
    ),
    "final_ledger_fingerprint": FINAL_LEDGER_FINGERPRINT,
    "phase_manifest_fingerprint": PARENT_PHASE_FINGERPRINT,
    "selected_candidate_fingerprint": None,
    "status": "combination-regressed",
    "w1_4_exit_fingerprint": PARENT_EXIT_FINGERPRINT,
}

OPERATOR_RELATIVE = Path("training/enoch_week1_w1_8_operator.py")
PLAN_RELATIVE = Path("training/enoch_week1_w1_8_plan.json")
TEST_RELATIVE = Path("training/test_enoch_week1_w1_8_operator.py")
CONTINUATION_PATHS = tuple(
    item.as_posix() for item in (OPERATOR_RELATIVE, PLAN_RELATIVE, TEST_RELATIVE)
)
CRITICAL_BASE_MODULES = tuple(
    sorted(
        {
            *w14.CRITICAL_BASE_MODULES,
            "training/enoch_week1_w1_4_operator.py",
        }
    )
)
RUNTIME_MODULES = (*w14.RUNTIME_MODULES, (w14, "training/enoch_week1_w1_4_operator.py"))

PLAN_KIND = "enoch-week1-w1.8-committed-plan"
PROVENANCE_KIND = "enoch-week1-w1.8-continuation-provenance"
REVIEW_KIND = "enoch-week1-w1.8-human-review-attestation"
STATUS = "week1-complete-retain-enoch-0"
REVIEW_ATTESTATION = (
    "interactive-human-operator-directed-seal-w1.8-retain-enoch-0"
)

EXPECTED_FILES = frozenset(
    {
        "input.json",
        "continuation-provenance.json",
        "freeze-or-no-confirmed-candidate-decision.json",
        "human-review-attestation.json",
        "final-ledger.json",
        "phase-manifest.json",
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_OBJECT_RE = re.compile(r"[0-9a-f]{40,64}")
_OPERATOR_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")


class W18OperatorError(RuntimeError):
    """Raised when W1.8 cannot be sealed or reconstructed exactly."""


@dataclass(frozen=True)
class W18Layout:
    root: Path

    @property
    def base(self) -> base_operator.RunLayout:
        return base_operator.RunLayout(self.root)

    @property
    def parent(self) -> w14.W14Layout:
        return w14.W14Layout(self.root)

    @property
    def directory(self) -> Path:
        return self.root / "w1.8"

    @property
    def input(self) -> Path:
        return self.directory / "input.json"

    @property
    def provenance(self) -> Path:
        return self.directory / "continuation-provenance.json"

    @property
    def terminal(self) -> Path:
        return self.directory / "freeze-or-no-confirmed-candidate-decision.json"

    @property
    def review(self) -> Path:
        return self.directory / "human-review-attestation.json"

    @property
    def final_ledger(self) -> Path:
        return self.directory / "final-ledger.json"

    @property
    def phase(self) -> Path:
        return self.directory / "phase-manifest.json"


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise W18OperatorError(f"{label} must be lowercase SHA-256")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping):
        raise W18OperatorError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise W18OperatorError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _with_fingerprint(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    frozen = dict(body)
    return {**frozen, field: enoch_week1.canonical_json_sha256(frozen)}


def _validate_fingerprint(value: Mapping[str, Any], field: str, label: str) -> str:
    fingerprint = _require_sha256(value.get(field), f"{label} {field}")
    body = dict(value)
    body.pop(field)
    if enoch_week1.canonical_json_sha256(body) != fingerprint:
        raise W18OperatorError(f"{label} fingerprint mismatch")
    return fingerprint


def _normalize_layout(layout: W18Layout) -> W18Layout:
    expanded = layout.root.expanduser()
    absolute = Path(os.path.abspath(os.fspath(expanded)))
    if expanded != absolute:
        raise W18OperatorError("W1.8 run root must be absolute and normalized")
    return W18Layout(absolute)


def _require_real_root(layout: W18Layout) -> Path:
    root = layout.root
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise W18OperatorError("W1.8 run root must be one real absolute directory")
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise W18OperatorError(f"could not resolve W1.8 run root: {exc}") from exc
    if resolved != root:
        raise W18OperatorError("W1.8 run root path contains a symlink component")
    return root


def _require_safe_run_file(layout: W18Layout, path: Path, label: str) -> Path:
    root = _require_real_root(layout)
    absolute = Path(os.path.abspath(os.fspath(path)))
    if absolute != path:
        raise W18OperatorError(f"{label} path must be absolute and normalized")
    try:
        relative = absolute.relative_to(root)
    except ValueError as exc:
        raise W18OperatorError(f"{label} escapes the run root") from exc
    if not relative.parts:
        raise W18OperatorError(f"{label} cannot be the run root")
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink() or not current.is_dir():
            raise W18OperatorError(f"{label} parent is not a real directory: {current}")
        if current.resolve(strict=True) != current:
            raise W18OperatorError(f"{label} parent contains a symlink hop: {current}")
    if absolute.is_symlink() or not absolute.is_file():
        raise W18OperatorError(f"{label} is not a regular non-symlink file")
    if root != absolute.resolve(strict=True) and root not in absolute.resolve(strict=True).parents:
        raise W18OperatorError(f"{label} resolves outside the run root")
    return absolute


def _require_workspace_file(workspace: Path, relative: str | Path, label: str) -> Path:
    workspace = workspace.expanduser().resolve()
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise W18OperatorError(f"{label} path is not workspace-relative")
    if workspace.is_symlink() or not workspace.is_dir():
        raise W18OperatorError("W1.8 workspace is not a real directory")
    path = workspace / relative_path
    current = workspace
    for part in relative_path.parts[:-1]:
        current = current / part
        if current.is_symlink() or not current.is_dir():
            raise W18OperatorError(f"{label} parent is not a real directory: {current}")
        if current.resolve(strict=True) != current:
            raise W18OperatorError(f"{label} parent contains a symlink hop: {current}")
    if path.is_symlink() or not path.is_file():
        raise W18OperatorError(f"{label} is not a regular non-symlink file")
    if path.resolve(strict=True) != path:
        raise W18OperatorError(f"{label} contains a symlink hop")
    return path


def _fsync_directory(path: Path, label: str) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise W18OperatorError(f"could not open {label} for fsync: {exc}") from exc
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise W18OperatorError(f"could not fsync {label}: {exc}") from exc
    finally:
        os.close(descriptor)


@contextlib.contextmanager
def _secure_lock(
    layout: W18Layout,
    path: Path,
    label: str,
    *,
    nonblocking: bool,
):
    if fcntl is None:
        raise W18OperatorError("authoritative W1.8 requires Unix advisory locking")
    root = _require_real_root(layout)
    if path.parent != root:
        raise W18OperatorError(f"{label} lock is not directly inside the run root")
    if _lexists(path) and (path.is_symlink() or not path.is_file()):
        raise W18OperatorError(f"{label} lock must be a regular non-symlink file")
    if not _lexists(path):
        raise W18OperatorError(f"required {label} lock is missing")
    flags = (
        os.O_RDWR
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o644)
    except OSError as exc:
        raise W18OperatorError(f"could not safely open {label} lock: {exc}") from exc
    try:
        opened = os.fstat(descriptor)
        current = os.lstat(path)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size != 0
            or not os.path.samestat(opened, current)
        ):
            raise W18OperatorError(f"{label} lock identity is unsafe")
        operation = fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0)
        try:
            fcntl.flock(descriptor, operation)
        except BlockingIOError as exc:
            raise W18OperatorError(f"another {label} lock holder is active") from exc
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _require_w18_tree(
    layout: W18Layout, *, create: bool, allow_missing: bool = False
) -> None:
    root = _require_real_root(layout)
    directory = layout.directory
    if _lexists(directory):
        if directory.is_symlink() or not directory.is_dir():
            raise W18OperatorError("W1.8 path is not a real directory")
        if directory.resolve(strict=True).parent != root:
            raise W18OperatorError("W1.8 directory escapes the run root")
    elif create:
        try:
            directory.mkdir()
        except OSError as exc:
            raise W18OperatorError(f"could not create W1.8 directory: {exc}") from exc
        _fsync_directory(root, "Week 1 run root")
    elif allow_missing:
        return
    else:
        raise W18OperatorError("W1.8 directory is missing")
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise W18OperatorError(f"W1.8 tree contains a symlink: {path}")
        if not path.is_file():
            raise W18OperatorError(f"W1.8 tree contains a non-file: {path}")
        if path.stat().st_nlink != 1:
            raise W18OperatorError(f"W1.8 tree contains a hard-linked artifact: {path}")
        if path.relative_to(directory).as_posix() not in EXPECTED_FILES:
            raise W18OperatorError(f"W1.8 tree contains an undeclared artifact: {path}")
        if root not in path.resolve(strict=True).parents:
            raise W18OperatorError("W1.8 artifact resolves outside the run root")


def _safe_w18_leaf(layout: W18Layout, path: Path, *, must_exist: bool) -> Path:
    _require_w18_tree(layout, create=not must_exist, allow_missing=not must_exist)
    if path.parent != layout.directory or path.name not in EXPECTED_FILES:
        raise W18OperatorError("W1.8 artifact path is not declared")
    if _lexists(path):
        if path.is_symlink() or not path.is_file():
            raise W18OperatorError(f"W1.8 artifact is not a regular file: {path}")
        if path.stat().st_nlink != 1:
            raise W18OperatorError(f"W1.8 artifact is hard-linked: {path}")
        if path.resolve(strict=True).parent != layout.directory:
            raise W18OperatorError("W1.8 artifact escapes its directory")
    elif must_exist:
        raise W18OperatorError(f"required W1.8 artifact is missing: {path}")
    return path


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise W18OperatorError(f"{label} is not a regular non-symlink file: {path}")
    try:
        return enoch_week1.load_json_object(path)
    except (OSError, enoch_week1.ProtocolError) as exc:
        raise W18OperatorError(f"could not load {label}: {exc}") from exc


def _load_w18_json(layout: W18Layout, path: Path, label: str) -> dict[str, Any]:
    path = _safe_w18_leaf(layout, path, must_exist=True)
    value = _load_json(path, label)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise W18OperatorError(f"could not read {label}: {exc}") from exc
    if payload != enoch_week1.canonical_json_bytes(value) + b"\n":
        raise W18OperatorError(f"{label} is not canonical JSON")
    return value


def _write_or_match(
    layout: W18Layout, path: Path, value: Mapping[str, Any], label: str
) -> None:
    path = _safe_w18_leaf(layout, path, must_exist=False)
    if path.exists():
        if _load_w18_json(layout, path, label) != dict(value):
            raise W18OperatorError(f"existing {label} does not reconstruct")
        return
    try:
        enoch_week1.atomic_write_json(path, value)
    except (OSError, enoch_week1.ProtocolError) as exc:
        raise W18OperatorError(f"could not write {label}: {exc}") from exc
    _safe_w18_leaf(layout, path, must_exist=True)


def _sha256_regular(path: Path, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise W18OperatorError(f"{label} is not a regular non-symlink file")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while block := source.read(1024 * 1024):
                digest.update(block)
    except OSError as exc:
        raise W18OperatorError(f"could not hash {label}: {exc}") from exc
    return digest.hexdigest()


def _operator_id(value: Any) -> str:
    if not isinstance(value, str) or not _OPERATOR_ID_RE.fullmatch(value):
        raise W18OperatorError("operator id must be a stable 1-128 character identifier")
    return value


def _require_executing_operator(workspace: Path, expected_sha256: str) -> None:
    expected_path = _require_workspace_file(
        workspace, OPERATOR_RELATIVE, "W1.8 operator source"
    )
    actual_path = Path(os.path.abspath(__file__))
    if actual_path != expected_path:
        raise W18OperatorError("executing W1.8 operator is outside its workspace")
    if _sha256_regular(actual_path, "executing W1.8 operator") != _require_sha256(
        expected_sha256, "stored W1.8 operator source hash"
    ):
        raise W18OperatorError("executing W1.8 operator differs from stored provenance")


def validate_committed_plan(plan: Mapping[str, Any]) -> str:
    expected_keys = {
        "automatic_production_promotion_allowed",
        "candidate_fingerprint",
        "candidate_status",
        "control_manifest_fingerprint",
        "decision",
        "expected_evidence_fingerprint_count",
        "final_consumed_count",
        "final_ledger_fingerprint",
        "human_review_attestation",
        "human_review_decision",
        "manifest_kind",
        "manifest_version",
        "no_candidate_reason",
        "parent_candidate_decision_fingerprint",
        "parent_evaluated_candidate_fingerprint",
        "parent_phase_chain_sha256",
        "parent_phase_manifest_fingerprint",
        "parent_preserved_aggregate_bytes",
        "parent_preserved_file_count",
        "parent_preserved_manifest_sha256",
        "parent_single_candidate_or_no_survivor_fingerprint",
        "parent_status",
        "parent_w1_5_allowed",
        "permanent_scientific_control_fingerprint",
        "production_promotion_authorized",
        "protocol_fingerprint",
        "selected_candidate_fingerprint",
        "skipped_phases",
        "source_control_contract",
        "stage2_rebaseline_authorized_after_human_review",
        "terminal_decision_fingerprint",
    }
    _require_exact_keys(plan, expected_keys, "committed W1.8 plan")
    expected = {
        "automatic_production_promotion_allowed": False,
        "candidate_fingerprint": EVALUATED_CANDIDATE_FINGERPRINT,
        "candidate_status": "not-confirmed",
        "control_manifest_fingerprint": CONTROL_MANIFEST_FINGERPRINT,
        "decision": "no-confirmed-candidate",
        "expected_evidence_fingerprint_count": EXPECTED_EVIDENCE_COUNT,
        "final_consumed_count": FINAL_CONSUMED_COUNT,
        "final_ledger_fingerprint": FINAL_LEDGER_FINGERPRINT,
        "human_review_attestation": REVIEW_ATTESTATION,
        "human_review_decision": "retain-enoch-0",
        "manifest_kind": PLAN_KIND,
        "manifest_version": MANIFEST_VERSION,
        "no_candidate_reason": "combination-regressed",
        "parent_candidate_decision_fingerprint": PARENT_DECISION_FINGERPRINT,
        "parent_evaluated_candidate_fingerprint": EVALUATED_CANDIDATE_FINGERPRINT,
        "parent_phase_chain_sha256": PHASE_CHAIN_SHA256,
        "parent_phase_manifest_fingerprint": PARENT_PHASE_FINGERPRINT,
        "parent_preserved_aggregate_bytes": PARENT_PRESERVED_INVENTORY[
            "aggregate_bytes"
        ],
        "parent_preserved_file_count": PARENT_PRESERVED_INVENTORY["file_count"],
        "parent_preserved_manifest_sha256": PARENT_PRESERVED_INVENTORY[
            "manifest_sha256"
        ],
        "parent_single_candidate_or_no_survivor_fingerprint": PARENT_EXIT_FINGERPRINT,
        "parent_status": "combination-regressed",
        "parent_w1_5_allowed": False,
        "permanent_scientific_control_fingerprint": PERMANENT_ENOCH0_FINGERPRINT,
        "production_promotion_authorized": False,
        "protocol_fingerprint": PROTOCOL_FINGERPRINT,
        "selected_candidate_fingerprint": None,
        "skipped_phases": list(SKIPPED_PHASES),
        "source_control_contract": w14._source_control_contract(),  # noqa: SLF001
        "stage2_rebaseline_authorized_after_human_review": True,
        "terminal_decision_fingerprint": TERMINAL_DECISION_FINGERPRINT,
    }
    if dict(plan) != expected:
        changed = sorted(key for key in expected if plan.get(key) != expected[key])
        raise W18OperatorError(f"committed W1.8 plan changed fields: {changed}")
    return enoch_week1.canonical_json_sha256(plan)


def load_committed_plan(workspace: Path) -> dict[str, Any]:
    path = _require_workspace_file(workspace, PLAN_RELATIVE, "committed W1.8 plan")
    plan = _load_json(path, "committed W1.8 plan")
    validate_committed_plan(plan)
    return plan


def _git_blob(workspace: Path, revision: str, relative: str) -> str:
    value = w14._git_text(  # noqa: SLF001
        workspace, "rev-parse", f"{revision}:{relative}"
    ).strip()
    if not _GIT_OBJECT_RE.fullmatch(value):
        raise W18OperatorError(f"invalid Git blob for {revision}:{relative}")
    return value


def _continuation_git_identity(
    workspace: Path, *, continuation_commit: str | None = None, require_live: bool
) -> dict[str, Any]:
    workspace = workspace.expanduser().resolve()
    top = Path(
        w14._git_text(workspace, "rev-parse", "--show-toplevel").strip()  # noqa: SLF001
    ).resolve()
    if top != workspace or workspace != w14.TRUSTED_WORKSPACE:
        raise W18OperatorError("W1.8 workspace differs from the committed Git root")
    if require_live:
        status = w14._git_text(  # noqa: SLF001
            workspace, "status", "--porcelain=v1", "--untracked-files=all"
        )
        if status:
            raise W18OperatorError(f"W1.8 workspace is not clean: {status.splitlines()[0]}")
        head = w14._git_text(workspace, "rev-parse", "HEAD^{commit}").strip()  # noqa: SLF001
        if continuation_commit is not None and continuation_commit != head:
            raise W18OperatorError("requested W1.8 commit differs from live HEAD")
    else:
        if continuation_commit is None:
            raise W18OperatorError("stored W1.8 provenance lacks its commit")
        head = w14._git_text(  # noqa: SLF001
            workspace, "rev-parse", f"{continuation_commit}^{{commit}}"
        ).strip()
    parents = w14._git_text(  # noqa: SLF001
        workspace, "rev-list", "--parents", "-n", "1", head
    ).split()
    if parents != [head, BASE_COMMIT]:
        raise W18OperatorError("W1.8 continuation must be one direct child of 6dfbce7")
    raw = w14._git_text(  # noqa: SLF001
        workspace,
        "diff-tree",
        "--no-commit-id",
        "--name-status",
        "-r",
        "--no-renames",
        BASE_COMMIT,
        head,
    )
    parsed = [tuple(line.split("\t")) for line in raw.splitlines()]
    if parsed != [("A", relative) for relative in CONTINUATION_PATHS]:
        raise W18OperatorError("W1.8 continuation must add exactly operator, plan, and tests")
    changes = []
    for status_code, relative in parsed:
        fields = w14._git_text(workspace, "ls-tree", head, "--", relative).strip().split(  # noqa: SLF001
            None, 3
        )
        if (
            len(fields) != 4
            or fields[0] != "100644"
            or fields[1] != "blob"
            or fields[3] != relative
        ):
            raise W18OperatorError(f"W1.8 source is not a regular blob: {relative}")
        blob = w14._git_bytes(workspace, "show", f"{head}:{relative}")  # noqa: SLF001
        digest = hashlib.sha256(blob).hexdigest()
        if require_live:
            live = _require_workspace_file(workspace, relative, f"live source {relative}")
            if _sha256_regular(live, relative) != digest:
                raise W18OperatorError(f"live W1.8 source differs from Git: {relative}")
        changes.append(
            {
                "new_blob": fields[2],
                "old_blob": None,
                "path": relative,
                "sha256": digest,
                "status": status_code,
            }
        )
    critical = []
    for relative in CRITICAL_BASE_MODULES:
        base_blob = _git_blob(workspace, BASE_COMMIT, relative)
        if _git_blob(workspace, head, relative) != base_blob:
            raise W18OperatorError(f"W1.8 changed frozen runtime module: {relative}")
        critical.append({"blob": base_blob, "path": relative})
    return {
        "base_tree_manifest_sha256": hashlib.sha256(
            w14._git_bytes(  # noqa: SLF001
                workspace, "ls-tree", "-r", "-z", "--full-tree", BASE_COMMIT
            )
        ).hexdigest(),
        "changed_paths": changes,
        "continuation_git_commit": head,
        "continuation_git_tree": w14._git_text(  # noqa: SLF001
            workspace, "rev-parse", f"{head}^{{tree}}"
        ).strip(),
        "critical_base_module_blobs": critical,
        "git_tree_manifest_sha256": hashlib.sha256(
            w14._git_bytes(workspace, "ls-tree", "-r", "-z", "--full-tree", head)  # noqa: SLF001
        ).hexdigest(),
    }


def _runtime_import_records(workspace: Path, commit: str) -> list[dict[str, str]]:
    records = []
    for module, relative in RUNTIME_MODULES:
        expected_path = _require_workspace_file(
            workspace, relative, f"W1.8 runtime {relative}"
        )
        module_path = getattr(module, "__file__", None)
        if (
            not isinstance(module_path, str)
            or Path(os.path.abspath(module_path)) != expected_path
        ):
            raise W18OperatorError(f"W1.8 runtime import is shadowed: {relative}")
        live_digest = _sha256_regular(expected_path, relative)
        git_digest = hashlib.sha256(
            w14._git_bytes(workspace, "show", f"{commit}:{relative}")  # noqa: SLF001
        ).hexdigest()
        if live_digest != git_digest:
            raise W18OperatorError(f"live W1.8 runtime differs from Git: {relative}")
        records.append({"path": relative, "sha256": live_digest})
    return records


def _provenance_from_identity(
    workspace: Path, plan: Mapping[str, Any], identity: Mapping[str, Any]
) -> dict[str, Any]:
    commit = identity["continuation_git_commit"]

    def committed_sha(relative: Path) -> str:
        return hashlib.sha256(
            w14._git_bytes(workspace, "show", f"{commit}:{relative.as_posix()}")  # noqa: SLF001
        ).hexdigest()

    body = {
        "automatic_production_promotion_allowed": False,
        "base_git_commit": BASE_COMMIT,
        "base_git_tree": w14._git_text(  # noqa: SLF001
            workspace, "rev-parse", f"{BASE_COMMIT}^{{tree}}"
        ).strip(),
        "base_tree_manifest_sha256": identity["base_tree_manifest_sha256"],
        "changed_paths": identity["changed_paths"],
        "continuation_git_commit": commit,
        "continuation_git_tree": identity["continuation_git_tree"],
        "critical_base_module_blobs": identity["critical_base_module_blobs"],
        "git_tree_manifest_sha256": identity["git_tree_manifest_sha256"],
        "manifest_kind": PROVENANCE_KIND,
        "manifest_version": MANIFEST_VERSION,
        "operator_source_path": OPERATOR_RELATIVE.as_posix(),
        "operator_source_sha256": committed_sha(OPERATOR_RELATIVE),
        "parent_candidate_decision_fingerprint": PARENT_DECISION_FINGERPRINT,
        "parent_final_ledger_fingerprint": FINAL_LEDGER_FINGERPRINT,
        "parent_phase_manifest_fingerprint": PARENT_PHASE_FINGERPRINT,
        "parent_single_candidate_or_no_survivor_fingerprint": PARENT_EXIT_FINGERPRINT,
        "plan_file_sha256": committed_sha(PLAN_RELATIVE),
        "plan_sha256": enoch_week1.canonical_json_sha256(plan),
        "protocol_fingerprint": PROTOCOL_FINGERPRINT,
        "runtime_imports": _runtime_import_records(workspace, commit),
        "source_control_contract_fingerprint": plan["source_control_contract"][
            "source_control_contract_fingerprint"
        ],
        "starting_consumed_count": FINAL_CONSUMED_COUNT,
        "terminal_decision_fingerprint": TERMINAL_DECISION_FINGERPRINT,
        "test_source_path": TEST_RELATIVE.as_posix(),
        "test_source_sha256": committed_sha(TEST_RELATIVE),
    }
    return _with_fingerprint(body, "continuation_provenance_fingerprint")


def build_continuation_provenance(
    workspace: Path, plan: Mapping[str, Any]
) -> dict[str, Any]:
    validate_committed_plan(plan)
    workspace = workspace.expanduser().resolve()
    identity = _continuation_git_identity(workspace, require_live=True)
    artifact = _provenance_from_identity(workspace, plan, identity)
    _require_executing_operator(workspace, artifact["operator_source_sha256"])
    return artifact


def validate_continuation_provenance(
    artifact: Mapping[str, Any],
    workspace: Path,
    plan: Mapping[str, Any],
    *,
    live_source: bool,
) -> str:
    validate_committed_plan(plan)
    if live_source:
        expected = build_continuation_provenance(workspace, plan)
    else:
        _validate_fingerprint(
            artifact,
            "continuation_provenance_fingerprint",
            "stored W1.8 provenance",
        )
        commit = artifact.get("continuation_git_commit")
        if not isinstance(commit, str):
            raise W18OperatorError("stored W1.8 provenance lacks its commit")
        identity = _continuation_git_identity(
            workspace, continuation_commit=commit, require_live=False
        )
        expected = _provenance_from_identity(workspace, plan, identity)
        _require_executing_operator(workspace, artifact.get("operator_source_sha256"))
    if dict(artifact) != expected:
        raise W18OperatorError("W1.8 continuation provenance does not reconstruct")
    return expected["continuation_provenance_fingerprint"]


def _verify_authoritative_w1_4(
    layout: W18Layout, workspace: Path, w1_2_workspace: Path
) -> dict[str, Any]:
    result = w14.verify_w1_4(layout.parent, workspace, w1_2_workspace)
    if result != EXPECTED_W14_RESULT:
        raise W18OperatorError("authoritative W1.4 verifier result changed")
    return result


def _assert_skipped_phases_absent(layout: W18Layout) -> None:
    for phase in SKIPPED_PHASES:
        path = layout.root / phase.lower()
        if _lexists(path):
            raise W18OperatorError(f"{phase} must not be materialized on this branch")


def _load_parent_state(
    layout: W18Layout, verified_w14: Mapping[str, Any]
) -> dict[str, Any]:
    if dict(verified_w14) != EXPECTED_W14_RESULT:
        raise W18OperatorError("W1.8 requires the exact authoritative W1.4 result")
    _assert_skipped_phases_absent(layout)

    def parent_file(relative: str, label: str) -> Path:
        return _require_safe_run_file(layout, layout.root / relative, label)

    def load_parent(relative: str, label: str) -> dict[str, Any]:
        return _load_json(parent_file(relative, label), label)

    protocol = load_parent("protocol.json", "Week 1 protocol")
    if enoch_week1.validate_protocol(protocol) != PROTOCOL_FINGERPRINT:
        raise W18OperatorError("Week 1 protocol changed before W1.8")
    phases = [
        load_parent(f"w1.{index}/phase-manifest.json", f"W1.{index} phase")
        for index in range(5)
    ]
    if tuple(enoch_week1.validate_phase_chain(protocol, phases)) != PHASE_FINGERPRINTS:
        raise W18OperatorError("W1.0-W1.4 phase chain changed")
    if enoch_week1.canonical_json_sha256(phases) != PHASE_CHAIN_SHA256:
        raise W18OperatorError("W1.0-W1.4 phase-chain hash changed")
    control = load_parent(
        "w1.0/control-bundle/control-manifest.json", "W1.0 control manifest"
    )
    if (
        enoch_week1.validate_w1_0_control_manifest(protocol, control)
        != CONTROL_MANIFEST_FINGERPRINT
    ):
        raise W18OperatorError("frozen W1.0 control changed")
    enoch0 = enoch_week1.canonical_json_sha256(
        control["policy_identities"]["enoch-0"]
    )
    if enoch0 != PERMANENT_ENOCH0_FINGERPRINT:
        raise W18OperatorError("permanent Enoch-0 identity changed")
    live_ledger_path = parent_file("seed-ledger.json", "live seed ledger")
    parent_ledger_path = parent_file(
        "w1.4/final-ledger.json", "W1.4 final ledger"
    )
    live_ledger = _load_json(live_ledger_path, "live seed ledger")
    parent_ledger = _load_json(parent_ledger_path, "W1.4 final ledger")
    for label, ledger in (("live", live_ledger), ("W1.4", parent_ledger)):
        if enoch_week1.validate_seed_ledger(protocol, ledger) != FINAL_LEDGER_FINGERPRINT:
            raise W18OperatorError(f"{label} ledger fingerprint changed")
        if len(ledger["consumed"]) != FINAL_CONSUMED_COUNT:
            raise W18OperatorError(f"{label} ledger count changed")
    if live_ledger != parent_ledger:
        raise W18OperatorError("live ledger advanced beyond the W1.4 snapshot")
    try:
        live_ledger_bytes = live_ledger_path.read_bytes()
        parent_ledger_bytes = parent_ledger_path.read_bytes()
    except OSError as exc:
        raise W18OperatorError(f"could not read sealed ledger bytes: {exc}") from exc
    if live_ledger_bytes != parent_ledger_bytes:
        raise W18OperatorError("live and W1.4 final ledger bytes differ")
    canonical_ledger_bytes = enoch_week1.canonical_json_bytes(live_ledger) + b"\n"
    if live_ledger_bytes != canonical_ledger_bytes:
        raise W18OperatorError("sealed final ledger is not canonical JSON")
    exit_artifact = load_parent(
        "w1.4/single-candidate-or-no-survivor.json", "W1.4 exit"
    )
    if (
        exit_artifact.get("w1_4_exit_fingerprint") != PARENT_EXIT_FINGERPRINT
        or exit_artifact.get("candidate_decision_fingerprint")
        != PARENT_DECISION_FINGERPRINT
        or exit_artifact.get("evaluated_candidate_fingerprint")
        != EVALUATED_CANDIDATE_FINGERPRINT
        or exit_artifact.get("selected_candidate_fingerprint") is not None
        or exit_artifact.get("no_candidate_reason") != "combination-regressed"
        or exit_artifact.get("status") != "combination-regressed"
        or exit_artifact.get("w1_5_allowed") is not False
        or exit_artifact.get("automatic_production_promotion_allowed") is not False
        or exit_artifact.get("final_ledger_fingerprint") != FINAL_LEDGER_FINGERPRINT
        or exit_artifact.get("final_consumed_count") != FINAL_CONSUMED_COUNT
    ):
        raise W18OperatorError("W1.4 exit is not the exact no-candidate branch")
    decision = load_parent("w1.4/candidate-decision.json", "W1.4 decision")
    if (
        decision.get("w1_4_candidate_decision_fingerprint")
        != PARENT_DECISION_FINGERPRINT
        or decision.get("candidate_fingerprint") != EVALUATED_CANDIDATE_FINGERPRINT
        or decision.get("decision") != "reject-candidate"
        or decision.get("automatic_production_promotion_allowed") is not False
    ):
        raise W18OperatorError("W1.4 rejected-candidate identity changed")
    return {
        "control": control,
        "decision": decision,
        "enoch0_fingerprint": enoch0,
        "exit": exit_artifact,
        "ledger": live_ledger,
        "phases": phases,
        "protocol": protocol,
    }


def _protected_snapshot(layout: W18Layout) -> dict[str, str]:
    relatives = [
        "protocol.json",
        "seed-ledger.json",
        "w1.0/control-bundle/control-manifest.json",
        *(f"w1.{index}/phase-manifest.json" for index in range(5)),
        "w1.4/candidate-decision.json",
        "w1.4/combination-regression-gate.json",
        "w1.4/final-ledger.json",
        "w1.4/single-candidate-or-no-survivor.json",
    ]
    return {
        relative: _sha256_regular(
            _require_safe_run_file(layout, layout.root / relative, relative),
            relative,
        )
        for relative in relatives
    }


def _full_parent_inventory(layout: W18Layout) -> dict[str, Any]:
    """Hash every preserved W1.0-W1.4 byte, including fixtures and raw shards."""

    root = _require_real_root(layout)
    records: list[dict[str, Any]] = []
    for relative in ("protocol.json", "seed-ledger.json"):
        path = _require_safe_run_file(layout, root / relative, relative)
        records.append(
            {
                "path": relative,
                "sha256": _sha256_regular(path, relative),
                "size": path.stat().st_size,
            }
        )
    for index in range(5):
        directory = root / f"w1.{index}"
        if (
            directory.is_symlink()
            or not directory.is_dir()
            or directory.resolve(strict=True) != directory
        ):
            raise W18OperatorError(f"W1.{index} evidence directory is unsafe")
        for path in sorted(directory.rglob("*")):
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                raise W18OperatorError(f"parent evidence contains a symlink: {relative}")
            if path.is_dir():
                if path.resolve(strict=True) != path:
                    raise W18OperatorError(
                        f"parent evidence directory contains a symlink hop: {relative}"
                    )
                continue
            if not path.is_file():
                raise W18OperatorError(
                    f"parent evidence contains a special file: {relative}"
                )
            safe = _require_safe_run_file(layout, path, relative)
            records.append(
                {
                    "path": relative,
                    "sha256": _sha256_regular(safe, relative),
                    "size": safe.stat().st_size,
                }
            )
    records.sort(key=lambda record: record["path"])
    return {
        "aggregate_bytes": sum(record["size"] for record in records),
        "file_count": len(records),
        "manifest_sha256": enoch_week1.canonical_json_sha256(records),
    }


def _validate_parent_inventory(
    inventory: Mapping[str, Any], plan: Mapping[str, Any]
) -> None:
    expected = {
        "aggregate_bytes": plan["parent_preserved_aggregate_bytes"],
        "file_count": plan["parent_preserved_file_count"],
        "manifest_sha256": plan["parent_preserved_manifest_sha256"],
    }
    if dict(inventory) != expected or expected != PARENT_PRESERVED_INVENTORY:
        raise W18OperatorError("preserved W1.0-W1.4 inventory differs from the plan")


def _w18_snapshot(layout: W18Layout) -> dict[str, Any]:
    _require_w18_tree(layout, create=False)
    directory_stat = layout.directory.stat()
    records = []
    for path in sorted(layout.directory.iterdir()):
        safe = _safe_w18_leaf(layout, path, must_exist=True)
        metadata = safe.stat()
        records.append(
            {
                "mtime_ns": metadata.st_mtime_ns,
                "path": safe.name,
                "sha256": _sha256_regular(safe, safe.name),
                "size": metadata.st_size,
            }
        )
    return {
        "directory_mtime_ns": directory_stat.st_mtime_ns,
        "files": records,
    }


@contextlib.contextmanager
def _offline_api_guard():
    def blocked(name: str):
        def fail(*_args, **_kwargs):
            raise W18OperatorError(f"offline W1.8 verification attempted {name}")

        return fail

    original_atomic = enoch_week1.atomic_write_json
    original_consume = enoch_week1.consume_seed_batch_once
    original_base_write = base_operator._write_or_match  # noqa: SLF001
    original_local_write = globals()["_write_or_match"]
    original_w14_run = w14.run_w1_4
    original_w14_declare = w14.declare_w1_4
    original_w14_stage = w14._run_stage  # noqa: SLF001
    try:
        enoch_week1.atomic_write_json = blocked("atomic JSON write")
        enoch_week1.consume_seed_batch_once = blocked("seed consumption")
        base_operator._write_or_match = blocked("base artifact write")  # type: ignore[method-assign]  # noqa: SLF001
        globals()["_write_or_match"] = blocked("W1.8 artifact write")
        w14.run_w1_4 = blocked("W1.4 execution")
        w14.declare_w1_4 = blocked("W1.4 declaration")
        w14._run_stage = blocked("W1.4 evaluator stage")  # type: ignore[method-assign]  # noqa: SLF001
        yield
    finally:
        enoch_week1.atomic_write_json = original_atomic
        enoch_week1.consume_seed_batch_once = original_consume
        base_operator._write_or_match = original_base_write  # type: ignore[method-assign]  # noqa: SLF001
        globals()["_write_or_match"] = original_local_write
        w14.run_w1_4 = original_w14_run
        w14.declare_w1_4 = original_w14_declare
        w14._run_stage = original_w14_stage  # type: ignore[method-assign]  # noqa: SLF001


def _build_terminal(state: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    terminal = enoch_week1.build_week1_decision_artifact(
        state["protocol"],
        phase_manifests=state["phases"],
        control_manifest=state["control"],
        enoch0_fingerprint=state["enoch0_fingerprint"],
        candidate_fingerprint=state["decision"]["candidate_fingerprint"],
        primary_gate_decision=None,
        confirmation_gate_decision=None,
        prerequisites_complete=True,
        no_candidate_reason="combination-regressed",
    )
    if (
        terminal.get("week1_decision_fingerprint")
        != plan["terminal_decision_fingerprint"]
        or terminal.get("decision") != plan["decision"]
        or terminal.get("candidate_fingerprint") != plan["candidate_fingerprint"]
        or terminal.get("candidate_status") != plan["candidate_status"]
        or terminal.get("no_candidate_reason") != plan["no_candidate_reason"]
        or terminal.get("downstream_primary_fingerprint")
        != plan["permanent_scientific_control_fingerprint"]
        or terminal.get("permanent_scientific_control_fingerprint")
        != plan["permanent_scientific_control_fingerprint"]
        or terminal.get("phase_chain_sha256") != plan["parent_phase_chain_sha256"]
        or len(terminal.get("evidence_fingerprints", []))
        != plan["expected_evidence_fingerprint_count"]
        or terminal.get("production_promotion_authorized") is not False
        or terminal.get("human_operator_review_required") is not True
        or terminal.get("stage2_rebaseline_authorized_after_human_review") is not True
    ):
        raise W18OperatorError("W1.8 terminal decision does not match its committed plan")
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
        if terminal.get(field) is not None:
            raise W18OperatorError(f"skipped W1.5-W1.7 evidence appeared in {field}")
    enoch_week1.validate_week1_decision_artifact(state["protocol"], terminal)
    return terminal


def _build_review(
    state: Mapping[str, Any],
    plan: Mapping[str, Any],
    provenance: Mapping[str, Any],
    terminal: Mapping[str, Any],
    operator_id: str,
) -> dict[str, Any]:
    body = {
        "automatic_production_promotion_allowed": False,
        "candidate_fingerprint": state["decision"]["candidate_fingerprint"],
        "downstream_primary_fingerprint": state["enoch0_fingerprint"],
        "final_consumed_count": FINAL_CONSUMED_COUNT,
        "final_ledger_fingerprint": FINAL_LEDGER_FINGERPRINT,
        "human_instruction_attestation": plan["human_review_attestation"],
        "human_operator_review_completed": True,
        "human_review_decision": plan["human_review_decision"],
        "human_review_source": "interactive-user-instruction",
        "manifest_kind": REVIEW_KIND,
        "manifest_version": MANIFEST_VERSION,
        "no_candidate_reason": "combination-regressed",
        "operator_id": _operator_id(operator_id),
        "parent_candidate_decision_fingerprint": PARENT_DECISION_FINGERPRINT,
        "parent_evaluated_candidate_fingerprint": EVALUATED_CANDIDATE_FINGERPRINT,
        "parent_phase_manifest_fingerprint": PARENT_PHASE_FINGERPRINT,
        "parent_single_candidate_or_no_survivor_fingerprint": PARENT_EXIT_FINGERPRINT,
        "permanent_scientific_control_fingerprint": state["enoch0_fingerprint"],
        "production_promotion_authorized": False,
        "protocol_fingerprint": PROTOCOL_FINGERPRINT,
        "selected_candidate_fingerprint": None,
        "skipped_phases": list(SKIPPED_PHASES),
        "source_provenance_fingerprint": provenance[
            "continuation_provenance_fingerprint"
        ],
        "stage2_rebaseline_authorized": True,
        "terminal_decision_fingerprint": terminal["week1_decision_fingerprint"],
        "terminal_human_operator_review_required": True,
        "w1_5_allowed": False,
    }
    return _with_fingerprint(body, "human_review_fingerprint")


def _validate_review(
    artifact: Mapping[str, Any],
    state: Mapping[str, Any],
    plan: Mapping[str, Any],
    provenance: Mapping[str, Any],
    terminal: Mapping[str, Any],
) -> str:
    _validate_fingerprint(artifact, "human_review_fingerprint", "W1.8 human review")
    operator_id = _operator_id(artifact.get("operator_id"))
    expected = _build_review(state, plan, provenance, terminal, operator_id)
    if dict(artifact) != expected:
        raise W18OperatorError("W1.8 human review does not reconstruct")
    return expected["human_review_fingerprint"]


def _phase_artifacts(
    plan: Mapping[str, Any],
    provenance: Mapping[str, Any],
    terminal: Mapping[str, Any],
    review: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "freeze-or-no-confirmed-candidate-decision": terminal[
            "week1_decision_fingerprint"
        ],
        "w1.4-phase-manifest": PARENT_PHASE_FINGERPRINT,
        "w1.4-single-candidate-or-no-survivor": PARENT_EXIT_FINGERPRINT,
        "w1.8-continuation-provenance": provenance[
            "continuation_provenance_fingerprint"
        ],
        "w1.8-final-ledger": FINAL_LEDGER_FINGERPRINT,
        "w1.8-human-review-attestation": review["human_review_fingerprint"],
        "w1.8-input": enoch_week1.canonical_json_sha256(plan),
    }


def _phase_declarations(
    state: Mapping[str, Any],
    terminal: Mapping[str, Any],
    review: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "automatic_production_promotion_allowed": False,
        "candidate_fingerprint": EVALUATED_CANDIDATE_FINGERPRINT,
        "decision": "no-confirmed-candidate",
        "downstream_primary_fingerprint": state["enoch0_fingerprint"],
        "final_consumed_count": FINAL_CONSUMED_COUNT,
        "final_ledger_fingerprint": FINAL_LEDGER_FINGERPRINT,
        "human_operator_review_completed": True,
        "human_review_fingerprint": review["human_review_fingerprint"],
        "no_candidate_reason": "combination-regressed",
        "permanent_scientific_control_fingerprint": state["enoch0_fingerprint"],
        "production_promotion_authorized": False,
        "selected_candidate_fingerprint": None,
        "skipped_phases": list(SKIPPED_PHASES),
        "stage2_rebaseline_authorized": True,
        "status": STATUS,
        "terminal_decision_fingerprint": terminal["week1_decision_fingerprint"],
        "w1_4_exit_fingerprint": PARENT_EXIT_FINGERPRINT,
        "w1_4_phase_manifest_fingerprint": PARENT_PHASE_FINGERPRINT,
    }


def _build_phase(
    state: Mapping[str, Any],
    plan: Mapping[str, Any],
    provenance: Mapping[str, Any],
    terminal: Mapping[str, Any],
    review: Mapping[str, Any],
) -> dict[str, Any]:
    phase = enoch_week1.build_phase_manifest(
        state["protocol"],
        "W1.8",
        artifacts=_phase_artifacts(plan, provenance, terminal, review),
        declarations=_phase_declarations(state, terminal, review),
        parent_phase_manifests=[state["phases"][-1]],
    )
    enoch_week1.validate_phase_manifest(state["protocol"], phase)
    actual = {item["artifact_id"]: item["sha256"] for item in phase["artifacts"]}
    expected = _phase_artifacts(plan, provenance, terminal, review)
    if (
        actual != expected
        or actual.get(phase["declared_exit_artifact"])
        != terminal["week1_decision_fingerprint"]
        or phase["parent_phases"]
        != [
            {
                "phase": "W1.4",
                "phase_manifest_fingerprint": PARENT_PHASE_FINGERPRINT,
            }
        ]
        or phase["declarations"] != _phase_declarations(state, terminal, review)
        or phase["automatic_production_promotion_allowed"] is not False
    ):
        raise W18OperatorError("W1.8 phase does not bind the exact terminal branch")
    return phase


def _summary(
    terminal: Mapping[str, Any],
    review: Mapping[str, Any],
    phase: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "candidate_fingerprint": terminal["candidate_fingerprint"],
        "decision": terminal["decision"],
        "downstream_primary_fingerprint": terminal["downstream_primary_fingerprint"],
        "final_consumed_count": FINAL_CONSUMED_COUNT,
        "final_ledger_fingerprint": FINAL_LEDGER_FINGERPRINT,
        "human_review_fingerprint": review["human_review_fingerprint"],
        "no_candidate_reason": terminal["no_candidate_reason"],
        "permanent_scientific_control_fingerprint": terminal[
            "permanent_scientific_control_fingerprint"
        ],
        "phase_manifest_fingerprint": phase["phase_manifest_fingerprint"],
        "production_promotion_authorized": False,
        "selected_candidate_fingerprint": None,
        "stage2_rebaseline_authorized": True,
        "status": STATUS,
        "terminal_decision_fingerprint": terminal["week1_decision_fingerprint"],
    }


def _verify_locked(
    layout: W18Layout,
    workspace: Path,
    plan: Mapping[str, Any],
    state: Mapping[str, Any],
) -> dict[str, Any]:
    _require_w18_tree(layout, create=False)
    actual_files = {
        path.relative_to(layout.directory).as_posix()
        for path in layout.directory.iterdir()
        if path.is_file()
    }
    if actual_files != EXPECTED_FILES:
        raise W18OperatorError(
            f"completed W1.8 file set differs: {sorted(actual_files)}"
        )
    if _load_w18_json(layout, layout.input, "W1.8 input") != dict(plan):
        raise W18OperatorError("W1.8 input does not match the committed plan")
    provenance = _load_w18_json(layout, layout.provenance, "W1.8 provenance")
    validate_continuation_provenance(
        provenance, workspace, plan, live_source=False
    )
    terminal = _load_w18_json(layout, layout.terminal, "W1.8 terminal decision")
    expected_terminal = _build_terminal(state, plan)
    if terminal != expected_terminal:
        raise W18OperatorError("W1.8 terminal decision changed")
    review = _load_w18_json(layout, layout.review, "W1.8 human review")
    _validate_review(review, state, plan, provenance, terminal)
    ledger = _load_w18_json(layout, layout.final_ledger, "W1.8 final ledger")
    if ledger != state["ledger"]:
        raise W18OperatorError("W1.8 final ledger differs from the unchanged live ledger")
    live_ledger_path = _require_safe_run_file(
        layout, layout.base.ledger, "live seed ledger"
    )
    try:
        if layout.final_ledger.read_bytes() != live_ledger_path.read_bytes():
            raise W18OperatorError("W1.8 final ledger bytes differ from the live ledger")
    except OSError as exc:
        raise W18OperatorError(f"could not compare W1.8 ledger bytes: {exc}") from exc
    phase = _load_w18_json(layout, layout.phase, "W1.8 phase")
    expected_phase = _build_phase(state, plan, provenance, terminal, review)
    if phase != expected_phase:
        raise W18OperatorError("W1.8 phase changed")
    return _summary(terminal, review, phase)


def seal_w1_8(
    layout: W18Layout,
    workspace: Path,
    w1_2_workspace: Path,
    *,
    operator_id: str,
    attest_human_reviewed_retain_enoch0: bool,
) -> dict[str, Any]:
    if attest_human_reviewed_retain_enoch0 is not True:
        raise W18OperatorError(
            "--attest-human-reviewed-retain-enoch-0 is required to seal W1.8"
        )
    operator_id = _operator_id(operator_id)
    layout = _normalize_layout(layout)
    workspace = workspace.expanduser().resolve()
    w1_2_workspace = w1_2_workspace.expanduser().resolve()
    _require_w18_tree(layout, create=False, allow_missing=True)
    initial_parent_inventory = _full_parent_inventory(layout)
    verified_w14 = _verify_authoritative_w1_4(layout, workspace, w1_2_workspace)
    with _secure_lock(
        layout, layout.base.operator_lock, "operator", nonblocking=True
    ):
        with _secure_lock(
            layout,
            layout.base.ledger.with_name(f"{layout.base.ledger.name}.lock"),
            "seed-ledger",
            nonblocking=False,
        ):
            if _full_parent_inventory(layout) != initial_parent_inventory:
                raise W18OperatorError("parent evidence changed during W1.4 verification")
            plan = load_committed_plan(workspace)
            _validate_parent_inventory(initial_parent_inventory, plan)
            protected_before = _protected_snapshot(layout)
            state = _load_parent_state(layout, verified_w14)
            protected = _protected_snapshot(layout)
            if protected != protected_before:
                raise W18OperatorError("parent evidence changed while loading W1.8 state")
            if _lexists(layout.phase):
                w18_before = _w18_snapshot(layout)
                with _offline_api_guard():
                    result = _verify_locked(layout, workspace, plan, state)
                if _w18_snapshot(layout) != w18_before:
                    raise W18OperatorError("completed W1.8 changed during verification")
                if _protected_snapshot(layout) != protected:
                    raise W18OperatorError("parent evidence changed during W1.8 verification")
                if _full_parent_inventory(layout) != initial_parent_inventory:
                    raise W18OperatorError("preserved Week 1 evidence changed")
                return result
            provenance = build_continuation_provenance(workspace, plan)
            terminal = _build_terminal(state, plan)
            review = _build_review(state, plan, provenance, terminal, operator_id)
            phase = _build_phase(state, plan, provenance, terminal, review)

            def require_unchanged() -> None:
                _assert_skipped_phases_absent(layout)
                if _protected_snapshot(layout) != protected:
                    raise W18OperatorError("parent evidence changed during W1.8 sealing")
                if build_continuation_provenance(workspace, plan) != provenance:
                    raise W18OperatorError("W1.8 source changed during sealing")
                current_path = _require_safe_run_file(
                    layout, layout.base.ledger, "live seed ledger"
                )
                current = _load_json(current_path, "live seed ledger")
                if current != state["ledger"]:
                    raise W18OperatorError("seed ledger changed during W1.8 sealing")

            require_unchanged()
            _require_w18_tree(layout, create=True)
            _write_or_match(layout, layout.input, plan, "W1.8 input")
            require_unchanged()
            _write_or_match(layout, layout.provenance, provenance, "W1.8 provenance")
            require_unchanged()
            _write_or_match(layout, layout.terminal, terminal, "W1.8 terminal decision")
            require_unchanged()
            _write_or_match(layout, layout.review, review, "W1.8 human review")
            require_unchanged()
            _write_or_match(layout, layout.final_ledger, state["ledger"], "W1.8 final ledger")
            require_unchanged()
            if _full_parent_inventory(layout) != initial_parent_inventory:
                raise W18OperatorError("preserved Week 1 evidence changed before phase seal")
            # The phase is deliberately the final write and sole completion marker.
            _write_or_match(layout, layout.phase, phase, "W1.8 phase manifest")
            require_unchanged()
            if _full_parent_inventory(layout) != initial_parent_inventory:
                raise W18OperatorError("preserved Week 1 evidence changed after phase seal")
            result = _verify_locked(layout, workspace, plan, state)
            if _full_parent_inventory(layout) != initial_parent_inventory:
                raise W18OperatorError("preserved Week 1 evidence changed after reconstruction")
            return result


def verify_w1_8(
    layout: W18Layout, workspace: Path, w1_2_workspace: Path
) -> dict[str, Any]:
    """Reconstruct W1.8 without comparisons, seed claims, or artifact writes."""

    layout = _normalize_layout(layout)
    workspace = workspace.expanduser().resolve()
    w1_2_workspace = w1_2_workspace.expanduser().resolve()
    _require_w18_tree(layout, create=False)
    initial_parent_inventory = _full_parent_inventory(layout)
    verified_w14 = _verify_authoritative_w1_4(layout, workspace, w1_2_workspace)
    with _secure_lock(
        layout, layout.base.operator_lock, "operator", nonblocking=True
    ):
        with _secure_lock(
            layout,
            layout.base.ledger.with_name(f"{layout.base.ledger.name}.lock"),
            "seed-ledger",
            nonblocking=False,
        ):
            if _full_parent_inventory(layout) != initial_parent_inventory:
                raise W18OperatorError("parent evidence changed during W1.4 verification")
            plan = load_committed_plan(workspace)
            _validate_parent_inventory(initial_parent_inventory, plan)
            protected_before = _protected_snapshot(layout)
            state = _load_parent_state(layout, verified_w14)
            protected = _protected_snapshot(layout)
            if protected != protected_before:
                raise W18OperatorError("parent evidence changed while loading W1.8 state")
            w18_before = _w18_snapshot(layout)
            with _offline_api_guard():
                result = _verify_locked(layout, workspace, plan, state)
            if _w18_snapshot(layout) != w18_before:
                raise W18OperatorError("W1.8 artifacts changed during offline verification")
            if _protected_snapshot(layout) != protected:
                raise W18OperatorError("parent evidence changed during W1.8 verification")
            if _full_parent_inventory(layout) != initial_parent_inventory:
                raise W18OperatorError("preserved Week 1 evidence changed")
            return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("seal-w1.8", "seal the reviewed retain-Enoch-0 Week 1 decision"),
        ("verify-w1.8", "offline-verify the completed W1.8 seal"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--root", required=True, type=Path)
        command.add_argument("--workspace", required=True, type=Path)
        command.add_argument("--w1-2-workspace", required=True, type=Path)
        if name == "seal-w1.8":
            command.add_argument("--operator-id", required=True)
            command.add_argument(
                "--attest-human-reviewed-retain-enoch-0", action="store_true"
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    layout = W18Layout(args.root)
    try:
        if args.command == "seal-w1.8":
            result = seal_w1_8(
                layout,
                args.workspace,
                args.w1_2_workspace,
                operator_id=args.operator_id,
                attest_human_reviewed_retain_enoch0=(
                    args.attest_human_reviewed_retain_enoch_0
                ),
            )
        else:
            result = verify_w1_8(layout, args.workspace, args.w1_2_workspace)
    except (
        W18OperatorError,
        w14.W14OperatorError,
        base_operator.OperatorError,
        enoch_week1.ProtocolError,
        FileExistsError,
        OSError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"W1.8 operator failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
