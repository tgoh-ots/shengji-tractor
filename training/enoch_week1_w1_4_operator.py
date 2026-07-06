#!/usr/bin/env python3
"""Authoritative, recovery-aware W1.4 combination-campaign operator."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

try:
    from training import (
        enoch_week1,
        enoch_week1_campaign,
        enoch_week1_evidence,
        enoch_week1_fixtures,
        enoch_week1_freeze,
        enoch_week1_operator as base_operator,
        enoch_week1_preflight,
        enoch_week1_runner,
        enoch_week1_w1_2_operator as w12,
        enoch_week1_w1_3_operator as w13,
        enoch_week1_w1_3_seal_recovery as recovery,
    )
except ImportError:  # pragma: no cover - direct script execution.
    import enoch_week1  # type: ignore[no-redef]
    import enoch_week1_campaign  # type: ignore[no-redef]
    import enoch_week1_evidence  # type: ignore[no-redef]
    import enoch_week1_fixtures  # type: ignore[no-redef]
    import enoch_week1_freeze  # type: ignore[no-redef]
    import enoch_week1_operator as base_operator  # type: ignore[no-redef]
    import enoch_week1_preflight  # type: ignore[no-redef]
    import enoch_week1_runner  # type: ignore[no-redef]
    import enoch_week1_w1_2_operator as w12  # type: ignore[no-redef]
    import enoch_week1_w1_3_operator as w13  # type: ignore[no-redef]
    import enoch_week1_w1_3_seal_recovery as recovery  # type: ignore[no-redef]


MANIFEST_VERSION = 1
BASE_COMMIT = "dbc1d1bdd0307e7081dbc41e15ddfe67284257b2"
PROTOCOL_FINGERPRINT = "a1e48199e6cb153c68f442cac9f28400798b994d154e03d34cb64420e21db2b7"
PARENT_PHASE = "1059c449de8b4f181e1887f0064f4544ac8286b152094025246e11a3830b3a5e"
PARENT_SET = "a6b2e8f0b79eb2199b141f68ce6d65a4716fbe6dbae2e2160738c0cd051ce025"
PARENT_LEDGER = "999e43c97bd27daa372df882a0208a71862cd7bc484e8a87605df5363e38c897"
PARENT_DECLARATION = "a9fe49ba40831844cafa8a56c9fe6663c659785385bac968d338486da676ac89"
RECOVERY_MANIFEST = "8cea1459b32e530e7211d69c5f7c0e4cad281eada081a64a9c29c97d6912a11e"
RECOVERY_PROVENANCE = "130bad03a3eff24ca57da15615df820f582b84f81e054e96bb7436bf6c02c267"
MALFORMED_SET = "116b56f15c34d7dafe15579716d6adbc4b1a88e3eb0b75407d0a49e1f6ba0ee0"
FIXTURE_REPORT_FINGERPRINT = "1fd410dca00fdd83631ae893980176113e621f1eebdec90c3b61f4d87dc7bc50"
FULL_PREFLIGHT_FINGERPRINT = "1ec318070aa22631eec52acf398ce2daec1f32b433bb583c628f1e0ab2b643ca"
CONTROL_MANIFEST_FINGERPRINT = "1aeb0c4f7d62d606eb554cf50aa83250a3672dd806587e695981196766e620f2"
ENVIRONMENT_FINGERPRINT = "4f44d61aa9ea44b2d3987da46f0548c1dfa928468d65265c6d9e8273d363ee5f"
PRECLAIM_COUNT = 18_711
FINAL_COUNT = 19_811

TRUSTED_GIT = Path("/usr/bin/git")
TRUSTED_GIT_SHA256 = "12bed4523661307059b879b9b54e77a73176e9d27d27a0e40363271d8f0668ba"
TRUSTED_GIT_VERSION_SHA256 = "0a72a762b37a3800276a1004e2c169aed0610d4a3ba601ba59e70e1c4d9f3200"
TRUSTED_WORKSPACE = Path("/private/tmp/shengji-strongest-bot-plan")
TRUSTED_GIT_POINTER_SHA256 = "f475f0b4d8e03125f684f6eb0f379c4fc3486baaf2ff2c312636b22e3c6e6349"
TRUSTED_GIT_ADMIN = Path(
    "/Users/tgoh/playground/shengji-tractor/.git/worktrees/shengji-strongest-bot-plan"
)
TRUSTED_GIT_COMMON = Path("/Users/tgoh/playground/shengji-tractor/.git")
TRUSTED_GIT_COMMONDIR_SHA256 = "340ddcb67a6204f742cd1e28e5b462622dde7daaa8ee36001897196aacdc6d47"
TRUSTED_GIT_GITDIR_SHA256 = "52d4a937a5bb2f012a7695132e86f37c1ae3d215b2be2bb97c62e9fd3e4cae88"
TRUSTED_GIT_CONFIG_SHA256 = "a9fdf4ae9100dc844cc7db97cab89d75433ad521934afcb16702a906c09ac370"
TRUSTED_GIT_EXCLUDE_SHA256 = "321dffcb77a68a6c079d35da12944866b895b4cc9fb3b1fae72fe9468c7eafeb"
TRUSTED_GIT_CONFIG_LIST_SHA256 = "b2e1955314488f7ca66242640cf940f28088cd3ce1691b02b383e9bbd73225e0"
TRUSTED_SYSROOT = Path(
    "/Users/tgoh/.rustup/toolchains/1.92.0-aarch64-apple-darwin"
)
TRUSTED_CARGO = TRUSTED_SYSROOT / "bin/cargo"
TRUSTED_RUSTC = TRUSTED_SYSROOT / "bin/rustc"
TRUSTED_RUSTDOC = TRUSTED_SYSROOT / "bin/rustdoc"
TRUSTED_CARGO_HOME = Path("/Users/tgoh/.cargo")
TRUSTED_TOOL_PATH = (
    f"{TRUSTED_SYSROOT}/bin:/usr/bin:/bin:/usr/sbin:/sbin"
)
TRUSTED_CARGO_SHA256 = "03e381389f5b7b8e695a744362f3866478f99034b2cf2df6afd4d42cfdab6f67"
TRUSTED_RUSTC_SHA256 = "12cab30aa9890d54445e29149a1e82d18fbe457de12801bd11bbe7e5e7fe33a0"
TRUSTED_RUSTDOC_SHA256 = "6d1914a8d4e53eb08486166f8008336d2620e858218f3332284d400b2d1733f8"
TRUSTED_CARGO_VERSION_SHA256 = "c7483f06ea048347b5e4818766c54d4dc99588d07ad79f1081c6653dce07e595"
TRUSTED_RUSTC_VERSION_SHA256 = "58c12481ab6ca64f223a3d873e8d32b30fdf2c200be485065372adcd9ddf23d8"
TRUSTED_TOOLCHAIN_FILE_COUNT = 119
TRUSTED_TOOLCHAIN_FILES_SHA256 = "56f5cb557f8922418428583fc6644a665b670d0a9077ca5500c0b438fd84fe18"
TRUSTED_TOOLCHAIN_AGGREGATE_BYTES = 525_686_348
TRUSTED_RUST_TOOLCHAIN_TOML_SHA256 = "e70025d1a1297e2230c1431cf1f7565980781de35fe37af82d9f0d73b4576e37"
TRUSTED_W1_0_COMPILER_SHA256 = "b8e68922af35afb118ca9888776e6ab0b8669d7207e52d2ec8c4681464bdce0f"

CANDIDATE_ARM_IDS = (
    "bid-ownership",
    "compound-follow",
    "friend-revelation",
    "uncertain-legal-throws",
)
STOPPED_ARM_ID = "team-void-boss"
STOPPED_ARM_DECISION = "90c7c3c26578dceb3547d52a48b0aa8b4325a93539006bf92c32d75560db43c1"
STAGE_IDS = ("qualification", "screen")

OPERATOR_RELATIVE = Path("training/enoch_week1_w1_4_operator.py")
PLAN_RELATIVE = Path("training/enoch_week1_w1_4_plan.json")
TEST_RELATIVE = Path("training/test_enoch_week1_w1_4_operator.py")
CONTINUATION_PATHS = tuple(
    path.as_posix() for path in (OPERATOR_RELATIVE, PLAN_RELATIVE, TEST_RELATIVE)
)
CRITICAL_BASE_MODULES = (
    "training/enoch_week1.py",
    "training/enoch_week1_campaign.py",
    "training/enoch_week1_evidence.py",
    "training/enoch_week1_fixtures.py",
    "training/enoch_week1_freeze.py",
    "training/enoch_week1_operator.py",
    "training/enoch_week1_preflight.py",
    "training/enoch_week1_runner.py",
    "training/enoch_week1_w1_2_operator.py",
    "training/enoch_week1_w1_3_operator.py",
    "training/enoch_week1_w1_3_seal_recovery.py",
)
RUNTIME_MODULES = (
    (enoch_week1, "training/enoch_week1.py"),
    (enoch_week1_campaign, "training/enoch_week1_campaign.py"),
    (enoch_week1_evidence, "training/enoch_week1_evidence.py"),
    (enoch_week1_fixtures, "training/enoch_week1_fixtures.py"),
    (enoch_week1_freeze, "training/enoch_week1_freeze.py"),
    (base_operator, "training/enoch_week1_operator.py"),
    (enoch_week1_preflight, "training/enoch_week1_preflight.py"),
    (enoch_week1_runner, "training/enoch_week1_runner.py"),
    (w12, "training/enoch_week1_w1_2_operator.py"),
    (w13, "training/enoch_week1_w1_3_operator.py"),
    (recovery, "training/enoch_week1_w1_3_seal_recovery.py"),
)

PLAN_KIND = "enoch-week1-w1.4-committed-plan"
PROVENANCE_KIND = "enoch-week1-w1.4-continuation-provenance"
DECLARATION_KIND = "enoch-week1-w1.4-campaign-declaration"
REGRESSION_KIND = "enoch-week1-w1.4-combination-regression-gate"
EXIT_KIND = "enoch-week1-w1.4-single-candidate-or-no-survivor"
RETIREMENT_KIND = "enoch-week1-w1.4-protocol-retirement"
FAILURE_DISPOSITION = "retire-protocol-on-any-claimed-incomplete-comparison"

EXPECTED_PARENT_RESULT = {
    "corrected_supported_change_set_fingerprint": PARENT_SET,
    "final_ledger_fingerprint": PARENT_LEDGER,
    "phase_manifest_fingerprint": PARENT_PHASE,
    "seal_recovery_manifest_fingerprint": RECOVERY_MANIFEST,
    "seal_recovery_provenance_fingerprint": RECOVERY_PROVENANCE,
    "status": "supported-survivors",
    "supported_arm_ids": list(CANDIDATE_ARM_IDS),
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_OBJECT_RE = re.compile(r"[0-9a-f]{40,64}")
_ATTEMPT_RE = re.compile(r"attempt-\d{3}")
RECOVERY_RUNPY_BOOTSTRAP = (
    "import runpy,sys;"
    "workspace=sys.argv.pop(1);script=sys.argv.pop(1);"
    "sys.path.insert(0,workspace);sys.argv[0]=script;"
    "runpy.run_path(script,run_name='__main__')"
)


class W14OperatorError(RuntimeError):
    """Raised when W1.4 cannot continue without ambiguity."""


class RegressionTestFailure(W14OperatorError):
    """A completed source-level prerequisite command returned a failed verdict."""

    def __init__(self, record: Mapping[str, Any]):
        self.record = dict(record)
        super().__init__(
            f"W1.4 prerequisite command failed: {self.record.get('command_id')}"
        )


class CandidatePrerequisiteFailed(W14OperatorError):
    """Internal control flow for a successfully sealed prerequisite rejection."""

    def __init__(self, result: Mapping[str, Any]):
        self.result = dict(result)
        super().__init__("W1.4 candidate prerequisites failed")


@dataclass(frozen=True)
class W14Layout:
    root: Path

    @property
    def base(self) -> base_operator.RunLayout:
        return base_operator.RunLayout(self.root)

    @property
    def parent(self) -> w13.W13Layout:
        return w13.W13Layout(self.root)

    @property
    def directory(self) -> Path:
        return self.root / "w1.4"

    @property
    def input(self) -> Path:
        return self.directory / "input.json"

    @property
    def provenance(self) -> Path:
        return self.directory / "continuation-provenance.json"

    @property
    def declaration(self) -> Path:
        return self.directory / "declaration-index.json"

    @property
    def lineage(self) -> Path:
        return self.directory / "campaign-lineage.json"

    @property
    def regression(self) -> Path:
        return self.directory / "combination-regression-gate.json"

    @property
    def prerequisite_failure(self) -> Path:
        return self.directory / "candidate-prerequisite-failure.json"

    @property
    def final_ledger(self) -> Path:
        return self.directory / "final-ledger.json"

    @property
    def decision(self) -> Path:
        return self.directory / "candidate-decision.json"

    @property
    def exit(self) -> Path:
        return self.directory / "single-candidate-or-no-survivor.json"

    @property
    def phase(self) -> Path:
        return self.directory / "phase-manifest.json"

    @property
    def retirement(self) -> Path:
        return self.directory / "protocol-retired.json"

    @property
    def regression_attempts(self) -> Path:
        return self.directory / "regression" / "attempts"

    def stage(self, sequence: int, stage_id: str) -> Path:
        return self.directory / "stages" / f"{sequence:02d}-{stage_id}"


def _w1_4_path_context(path: Path) -> tuple[Path, Path] | None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    indices = [index for index, part in enumerate(absolute.parts) if part == "w1.4"]
    if not indices:
        return None
    index = indices[-1]
    root = Path(*absolute.parts[:index])
    if not root.parts:
        root = Path(absolute.anchor)
    return root, absolute


def _require_safe_w1_4_file_path(
    path: Path,
    *,
    create_parents: bool,
    missing_parents_ok: bool = False,
) -> Path:
    context = _w1_4_path_context(path)
    if context is None:
        return path
    root, absolute = context
    if root.is_symlink() or not root.is_dir():
        raise W14OperatorError("W1.4 run root is not a real directory")
    root_resolved = root.resolve(strict=True)
    try:
        relative = absolute.relative_to(root)
    except ValueError as exc:  # pragma: no cover - lexical construction guard.
        raise W14OperatorError("W1.4 artifact path escapes its run root") from exc
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise W14OperatorError(f"W1.4 artifact parent is a symlink: {current}")
        if not current.exists():
            if create_parents:
                try:
                    current.mkdir()
                except FileExistsError:
                    pass
            elif missing_parents_ok:
                return absolute
            else:
                raise W14OperatorError(f"W1.4 artifact parent is missing: {current}")
        if current.is_symlink() or not current.is_dir():
            raise W14OperatorError(
                f"W1.4 artifact parent is not a real directory: {current}"
            )
        resolved = current.resolve(strict=True)
        if root_resolved != resolved and root_resolved not in resolved.parents:
            raise W14OperatorError("W1.4 artifact parent escapes its run root")
    if absolute.is_symlink():
        raise W14OperatorError(f"W1.4 artifact leaf is a symlink: {absolute}")
    return absolute


def _require_safe_w1_4_directory(
    path: Path, *, create: bool, missing_ok: bool = False
) -> Path:
    absolute = _require_safe_w1_4_file_path(
        path, create_parents=create, missing_parents_ok=missing_ok
    )
    if absolute.is_symlink():
        raise W14OperatorError(f"W1.4 directory is a symlink: {absolute}")
    if not absolute.exists():
        if create:
            try:
                absolute.mkdir()
            except FileExistsError:
                pass
        elif missing_ok:
            return absolute
        else:
            raise W14OperatorError(f"W1.4 directory is missing: {absolute}")
    if absolute.is_symlink() or not absolute.is_dir():
        raise W14OperatorError(f"W1.4 path is not a real directory: {absolute}")
    context = _w1_4_path_context(absolute)
    assert context is not None
    root, _ = context
    root_resolved = root.resolve(strict=True)
    resolved = absolute.resolve(strict=True)
    if root_resolved != resolved and root_resolved not in resolved.parents:
        raise W14OperatorError("W1.4 directory escapes its run root")
    return absolute


def _require_safe_w1_4_tree(layout: W14Layout) -> None:
    root = Path(os.path.abspath(os.fspath(layout.root)))
    if root.is_symlink() or not root.is_dir():
        raise W14OperatorError("W1.4 run root is not a real directory")
    directory = Path(os.path.abspath(os.fspath(layout.directory)))
    if directory.is_symlink():
        raise W14OperatorError("W1.4 artifact directory is a symlink")
    if not directory.exists():
        return
    if not directory.is_dir():
        raise W14OperatorError("W1.4 artifact directory is not a real directory")
    root_resolved = root.resolve(strict=True)
    for path in (directory, *directory.rglob("*")):
        if path.is_symlink():
            raise W14OperatorError(f"W1.4 artifact tree contains a symlink: {path}")
        if not path.is_dir() and not path.is_file():
            raise W14OperatorError(f"W1.4 artifact tree contains a special file: {path}")
        resolved = path.resolve(strict=True)
        if root_resolved != resolved and root_resolved not in resolved.parents:
            raise W14OperatorError("W1.4 artifact tree escapes its run root")


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping):
        raise W14OperatorError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise W14OperatorError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise W14OperatorError(f"{label} must be lowercase SHA-256")
    return value


def _require_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise W14OperatorError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise W14OperatorError(f"{label} must be finite")
    return result


def _load_json(path: Path) -> dict[str, Any]:
    path = _require_safe_w1_4_file_path(path, create_parents=False)
    if _w1_4_path_context(path) is not None and not path.is_file():
        raise W14OperatorError(f"W1.4 JSON artifact is not a regular file: {path}")
    try:
        return enoch_week1.load_json_object(path)
    except (OSError, enoch_week1.ProtocolError) as exc:
        raise W14OperatorError(f"could not load {path}: {exc}") from exc


def _sha256_file(path: Path) -> str:
    path = _require_safe_w1_4_file_path(path, create_parents=False)
    if _w1_4_path_context(path) is not None and not path.is_file():
        raise W14OperatorError(f"W1.4 artifact is not a regular file: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while block := source.read(1024 * 1024):
                digest.update(block)
    except OSError as exc:
        raise W14OperatorError(f"could not hash {path}: {exc}") from exc
    return digest.hexdigest()


def _with_fingerprint(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    frozen = dict(body)
    return {**frozen, field: enoch_week1.canonical_json_sha256(frozen)}


def _validate_fingerprint(value: Mapping[str, Any], field: str, label: str) -> str:
    fingerprint = _require_sha256(value.get(field), f"{label} {field}")
    body = dict(value)
    body.pop(field)
    if enoch_week1.canonical_json_sha256(body) != fingerprint:
        raise W14OperatorError(f"{label} fingerprint mismatch")
    return fingerprint


def _write_or_match(path: Path, value: Mapping[str, Any], label: str) -> None:
    path = _require_safe_w1_4_file_path(path, create_parents=True)
    if path.exists() and not path.is_file():
        raise W14OperatorError(f"existing {label} is not a regular file: {path}")
    try:
        base_operator._write_or_match(path, value, label)  # noqa: SLF001
    except base_operator.OperatorError as exc:
        raise W14OperatorError(str(exc)) from exc


def _atomic_write_w1_4_json(path: Path, value: Mapping[str, Any], label: str) -> None:
    path = _require_safe_w1_4_file_path(path, create_parents=True)
    if path.exists():
        raise W14OperatorError(f"refusing to overwrite existing {label}: {path}")
    try:
        enoch_week1.atomic_write_json(path, value)
    except (OSError, enoch_week1.ProtocolError) as exc:
        raise W14OperatorError(f"could not write {label}: {exc}") from exc


def _write_w1_4_bytes(path: Path, value: bytes, label: str) -> None:
    path = _require_safe_w1_4_file_path(path, create_parents=True)
    try:
        with path.open("xb") as destination:
            destination.write(value)
            destination.flush()
            os.fsync(destination.fileno())
    except OSError as exc:
        raise W14OperatorError(f"could not write {label}: {exc}") from exc


def _source_control_environment() -> dict[str, str]:
    return {
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_LITERAL_PATHSPECS": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
        "HOME": "/var/empty",
        "LANG": "C",
        "LC_ALL": "C",
        "PAGER": "cat",
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "TMPDIR": "/private/tmp",
        "XDG_CONFIG_HOME": "/var/empty",
    }


def _source_control_global_options() -> list[str]:
    return [
        "--no-replace-objects",
        "--no-optional-locks",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
        "-c",
        "core.commitGraph=false",
        "-c",
        "core.attributesFile=/dev/null",
        "-c",
        "color.ui=false",
    ]


def _source_control_contract() -> dict[str, Any]:
    absent = [
        {"path": str(path), "state": "absent"}
        for path in (
            TRUSTED_GIT_COMMON / "config.worktree",
            TRUSTED_GIT_ADMIN / "config.worktree",
            TRUSTED_GIT_COMMON / "objects/info/alternates",
            TRUSTED_GIT_COMMON / "objects/info/http-alternates",
            TRUSTED_GIT_COMMON / "info/attributes",
            TRUSTED_GIT_COMMON / "info/grafts",
            TRUSTED_GIT_COMMON / "shallow",
            TRUSTED_GIT_COMMON / "refs/replace",
        )
    ]
    body = {
        "absent_repository_paths_sha256": enoch_week1.canonical_json_sha256(absent),
        "common_config_sha256": TRUSTED_GIT_CONFIG_SHA256,
        "common_git_directory": str(TRUSTED_GIT_COMMON),
        "commondir_sha256": TRUSTED_GIT_COMMONDIR_SHA256,
        "environment_sha256": enoch_week1.canonical_json_sha256(
            _source_control_environment()
        ),
        "git_admin_directory": str(TRUSTED_GIT_ADMIN),
        "gitdir_sha256": TRUSTED_GIT_GITDIR_SHA256,
        "git_path": str(TRUSTED_GIT),
        "git_pointer_sha256": TRUSTED_GIT_POINTER_SHA256,
        "git_sha256": TRUSTED_GIT_SHA256,
        "git_version_sha256": TRUSTED_GIT_VERSION_SHA256,
        "global_options_sha256": enoch_week1.canonical_json_sha256(
            _source_control_global_options()
        ),
        "info_exclude_sha256": TRUSTED_GIT_EXCLUDE_SHA256,
        "local_config_list_sha256": TRUSTED_GIT_CONFIG_LIST_SHA256,
        "workspace_root": str(TRUSTED_WORKSPACE),
    }
    return {
        **body,
        "source_control_contract_fingerprint": enoch_week1.canonical_json_sha256(
            body
        ),
    }


def _validate_trusted_git() -> dict[str, Any]:
    if TRUSTED_GIT.is_symlink() or not TRUSTED_GIT.is_file():
        raise W14OperatorError("trusted Git executable is not a regular file")
    if _sha256_file(TRUSTED_GIT) != TRUSTED_GIT_SHA256:
        raise W14OperatorError("trusted Git executable hash changed")
    completed = subprocess.run(
        [str(TRUSTED_GIT), "--version"],
        env=_source_control_environment(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if (
        completed.returncode != 0
        or completed.stderr
        or hashlib.sha256(completed.stdout).hexdigest()
        != TRUSTED_GIT_VERSION_SHA256
    ):
        raise W14OperatorError("trusted Git version identity changed")
    return _source_control_contract()


def _require_regular_sha256(path: Path, expected: str, label: str) -> None:
    if path.is_symlink() or not path.is_file() or _sha256_file(path) != expected:
        raise W14OperatorError(f"{label} identity changed")


def _require_absent_path(path: Path, label: str) -> None:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise W14OperatorError(f"could not inspect {label}: {path}") from exc
    raise W14OperatorError(f"unexpected {label}: {path}")


def _validate_trusted_git_repository(workspace: Path) -> None:
    workspace = workspace.resolve()
    if workspace != TRUSTED_WORKSPACE:
        raise W14OperatorError("Git workspace differs from the committed contract")
    _require_regular_sha256(
        workspace / ".git", TRUSTED_GIT_POINTER_SHA256, "Git worktree pointer"
    )
    _require_regular_sha256(
        TRUSTED_GIT_ADMIN / "commondir",
        TRUSTED_GIT_COMMONDIR_SHA256,
        "Git commondir pointer",
    )
    _require_regular_sha256(
        TRUSTED_GIT_ADMIN / "gitdir",
        TRUSTED_GIT_GITDIR_SHA256,
        "Git worktree back-pointer",
    )
    _require_regular_sha256(
        TRUSTED_GIT_COMMON / "config",
        TRUSTED_GIT_CONFIG_SHA256,
        "Git common configuration",
    )
    _require_regular_sha256(
        TRUSTED_GIT_COMMON / "info/exclude",
        TRUSTED_GIT_EXCLUDE_SHA256,
        "Git repository excludes",
    )
    for path in (
        TRUSTED_GIT_COMMON / "config.worktree",
        TRUSTED_GIT_ADMIN / "config.worktree",
        TRUSTED_GIT_COMMON / "objects/info/alternates",
        TRUSTED_GIT_COMMON / "objects/info/http-alternates",
        TRUSTED_GIT_COMMON / "info/attributes",
        TRUSTED_GIT_COMMON / "info/grafts",
        TRUSTED_GIT_COMMON / "shallow",
        TRUSTED_GIT_COMMON / "refs/replace",
    ):
        _require_absent_path(path, "Git repository override")
    prefix = [
        str(TRUSTED_GIT),
        *_source_control_global_options(),
        "-C",
        str(workspace),
    ]
    probes = (
        (
            ("rev-parse", "--absolute-git-dir"),
            f"{TRUSTED_GIT_ADMIN}\n".encode("utf-8"),
            "Git absolute directory",
        ),
        (
            ("rev-parse", "--path-format=absolute", "--git-common-dir"),
            f"{TRUSTED_GIT_COMMON}\n".encode("utf-8"),
            "Git common directory",
        ),
        (
            ("for-each-ref", "--format=%(refname)", "refs/replace"),
            b"",
            "Git replace refs",
        ),
    )
    for arguments, expected, label in probes:
        completed = subprocess.run(
            [*prefix, *arguments],
            env=_source_control_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0 or completed.stderr or completed.stdout != expected:
            raise W14OperatorError(f"{label} changed")
    config = subprocess.run(
        [*prefix, "config", "--local", "--no-includes", "--null", "--list"],
        env=_source_control_environment(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if (
        config.returncode != 0
        or config.stderr
        or hashlib.sha256(config.stdout).hexdigest()
        != TRUSTED_GIT_CONFIG_LIST_SHA256
        or b"include.path" in config.stdout.lower()
        or b"includeif." in config.stdout.lower()
    ):
        raise W14OperatorError("Git local configuration semantics changed")
    tracked = subprocess.run(
        [*prefix, "ls-files", "-v", "-z"],
        env=_source_control_environment(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    entries = [entry for entry in tracked.stdout.split(b"\0") if entry]
    if (
        tracked.returncode != 0
        or tracked.stderr
        or not entries
        or any(not entry.startswith(b"H ") for entry in entries)
    ):
        raise W14OperatorError("Git index contains non-default tracked-file flags")


def _cargo_config_records(workspace: Path) -> list[dict[str, str]]:
    workspace = workspace.resolve()
    ancestors = [workspace, workspace.parent, workspace.parent.parent, Path("/")]
    candidates = [
        base / ".cargo" / name
        for base in ancestors
        for name in ("config", "config.toml")
    ] + [TRUSTED_CARGO_HOME / name for name in ("config", "config.toml")]
    records = []
    for path in candidates:
        try:
            os.lstat(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise W14OperatorError(f"could not inspect Cargo configuration: {path}") from exc
        else:
            raise W14OperatorError(f"Cargo configuration is not allowed: {path}")
        records.append({"path": str(path), "state": "absent"})
    return records


def _toolchain_file_records() -> list[dict[str, str]]:
    manifest_names = (
        "manifest-cargo-aarch64-apple-darwin",
        "manifest-rustc-aarch64-apple-darwin",
        "manifest-rust-std-aarch64-apple-darwin",
    )
    paths: list[Path] = []
    seen: set[str] = set()
    for name in manifest_names:
        manifest = TRUSTED_SYSROOT / "lib/rustlib" / name
        if manifest.is_symlink() or not manifest.is_file():
            raise W14OperatorError(f"trusted toolchain manifest is invalid: {name}")
        paths.append(manifest)
        try:
            lines = manifest.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise W14OperatorError(f"could not read toolchain manifest: {name}") from exc
        for line in lines:
            if not line.startswith("file:"):
                continue
            relative = line[5:]
            candidate = Path(relative)
            if (
                not relative
                or candidate.is_absolute()
                or ".." in candidate.parts
                or relative in seen
            ):
                raise W14OperatorError("trusted toolchain manifest path is invalid")
            seen.add(relative)
            path = TRUSTED_SYSROOT / candidate
            if path.is_symlink() or not path.is_file():
                raise W14OperatorError(
                    f"trusted toolchain file is not regular: {relative}"
                )
            resolved = path.resolve(strict=True)
            sysroot = TRUSTED_SYSROOT.resolve(strict=True)
            if sysroot not in resolved.parents:
                raise W14OperatorError("trusted toolchain file escapes its sysroot")
            paths.append(path)
    records = [
        {
            "path": path.relative_to(TRUSTED_SYSROOT).as_posix(),
            "sha256": _sha256_file(path),
        }
        for path in paths
    ]
    return sorted(records, key=lambda item: item["path"])


def _rust_toolchain_contract() -> dict[str, Any]:
    body = {
        "canonical_path": TRUSTED_TOOL_PATH,
        "cargo_config_candidate_count": 10,
        "cargo_config_records_sha256": "14e54f7733c11cdb6da2729967f233a04b834b5642892c372f60c0a261601c2c",
        "cargo_home": str(TRUSTED_CARGO_HOME),
        "cargo_path": str(TRUSTED_CARGO),
        "cargo_sha256": TRUSTED_CARGO_SHA256,
        "cargo_version_verbose_sha256": TRUSTED_CARGO_VERSION_SHA256,
        "rust_toolchain_toml_sha256": TRUSTED_RUST_TOOLCHAIN_TOML_SHA256,
        "rustc_path": str(TRUSTED_RUSTC),
        "rustc_sha256": TRUSTED_RUSTC_SHA256,
        "rustc_version_verbose_sha256": TRUSTED_RUSTC_VERSION_SHA256,
        "rustdoc_path": str(TRUSTED_RUSTDOC),
        "rustdoc_sha256": TRUSTED_RUSTDOC_SHA256,
        "sysroot": str(TRUSTED_SYSROOT),
        "toolchain_file_count": TRUSTED_TOOLCHAIN_FILE_COUNT,
        "toolchain_files_sha256": TRUSTED_TOOLCHAIN_FILES_SHA256,
        "toolchain_aggregate_bytes": TRUSTED_TOOLCHAIN_AGGREGATE_BYTES,
        "w1_0_compiler_sha256": TRUSTED_W1_0_COMPILER_SHA256,
        "workspace_root": "/private/tmp/shengji-strongest-bot-plan",
    }
    return {
        **body,
        "rust_toolchain_contract_fingerprint": enoch_week1.canonical_json_sha256(
            body
        ),
    }


def _pinned_regression_environment(attempt: Path) -> dict[str, str]:
    return {
        "CARGO_HOME": str(TRUSTED_CARGO_HOME),
        "CARGO_INCREMENTAL": "0",
        "CARGO_NET_OFFLINE": "true",
        "CARGO_TARGET_DIR": str((attempt / "cargo-target").resolve()),
        "HOME": "/Users/tgoh",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": TRUSTED_TOOL_PATH,
        "RUSTC": str(TRUSTED_RUSTC),
        "RUSTDOC": str(TRUSTED_RUSTDOC),
        "RUST_BACKTRACE": "1",
        "TMPDIR": "/private/tmp",
    }


def _run_identity_command(command: Sequence[str], environment: Mapping[str, str]) -> bytes:
    completed = subprocess.run(
        list(command),
        env=dict(environment),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0 or completed.stderr:
        raise W14OperatorError(f"toolchain identity command failed: {command[0]}")
    return completed.stdout


def _validate_live_rust_toolchain(
    workspace: Path, control_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    workspace = workspace.resolve()
    contract = _rust_toolchain_contract()
    if str(workspace) != contract["workspace_root"]:
        raise W14OperatorError("W1.4 workspace differs from its toolchain contract")
    for path, expected, label in (
        (TRUSTED_CARGO, TRUSTED_CARGO_SHA256, "Cargo"),
        (TRUSTED_RUSTC, TRUSTED_RUSTC_SHA256, "rustc"),
        (TRUSTED_RUSTDOC, TRUSTED_RUSTDOC_SHA256, "rustdoc"),
    ):
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != expected:
            raise W14OperatorError(f"trusted {label} executable changed")
    if shutil.which("cargo", path=TRUSTED_TOOL_PATH) != str(TRUSTED_CARGO):
        raise W14OperatorError("canonical PATH does not resolve the trusted Cargo")
    environment = _pinned_regression_environment(
        W14Layout(Path("/private/tmp/identity-only")).directory / "attempt-000"
    )
    environment.pop("CARGO_TARGET_DIR")
    cargo_version = _run_identity_command(
        (str(TRUSTED_CARGO), "--version", "--verbose"), environment
    )
    rustc_version = _run_identity_command((str(TRUSTED_RUSTC), "-vV"), environment)
    if hashlib.sha256(cargo_version).hexdigest() != TRUSTED_CARGO_VERSION_SHA256:
        raise W14OperatorError("trusted Cargo version output changed")
    if hashlib.sha256(rustc_version).hexdigest() != TRUSTED_RUSTC_VERSION_SHA256:
        raise W14OperatorError("trusted rustc version output changed")
    toolchain_file = workspace / "rust-toolchain.toml"
    if (
        toolchain_file.is_symlink()
        or not toolchain_file.is_file()
        or _sha256_file(toolchain_file) != TRUSTED_RUST_TOOLCHAIN_TOML_SHA256
    ):
        raise W14OperatorError("rust-toolchain.toml identity changed")
    records = _toolchain_file_records()
    if (
        len(records) != TRUSTED_TOOLCHAIN_FILE_COUNT
        or sum(
            (TRUSTED_SYSROOT / record["path"]).stat().st_size
            for record in records
        )
        != TRUSTED_TOOLCHAIN_AGGREGATE_BYTES
        or enoch_week1.canonical_json_sha256(records)
        != TRUSTED_TOOLCHAIN_FILES_SHA256
    ):
        raise W14OperatorError("trusted Rust toolchain inventory changed")
    configs = _cargo_config_records(workspace)
    if (
        len(configs) != contract["cargo_config_candidate_count"]
        or enoch_week1.canonical_json_sha256(configs)
        != contract["cargo_config_records_sha256"]
    ):
        raise W14OperatorError("Cargo configuration identity changed")
    compiler = control_manifest.get("compiler")
    cargo_short = _run_identity_command((str(TRUSTED_CARGO), "-V"), environment)
    reconstructed_compiler = (
        rustc_version.decode("utf-8").removesuffix("\n")
        + "\n"
        + cargo_short.decode("utf-8").removesuffix("\n")
    )
    if (
        not isinstance(compiler, str)
        or compiler != reconstructed_compiler
        or hashlib.sha256(compiler.encode("utf-8")).hexdigest()
        != TRUSTED_W1_0_COMPILER_SHA256
    ):
        raise W14OperatorError("W1.0 compiler identity changed")
    return contract


def _git_text(workspace: Path, *arguments: str) -> str:
    _validate_trusted_git()
    _validate_trusted_git_repository(workspace)
    completed = subprocess.run(
        [
            str(TRUSTED_GIT),
            *_source_control_global_options(),
            "-C",
            str(workspace),
            *arguments,
        ],
        env=_source_control_environment(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise W14OperatorError(
            f"git {' '.join(arguments)} failed: {completed.stderr.strip()}"
        )
    _validate_trusted_git()
    _validate_trusted_git_repository(workspace)
    return completed.stdout


def _git_bytes(workspace: Path, *arguments: str) -> bytes:
    _validate_trusted_git()
    _validate_trusted_git_repository(workspace)
    completed = subprocess.run(
        [
            str(TRUSTED_GIT),
            *_source_control_global_options(),
            "-C",
            str(workspace),
            *arguments,
        ],
        env=_source_control_environment(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise W14OperatorError(f"git {' '.join(arguments)} failed: {detail}")
    _validate_trusted_git()
    _validate_trusted_git_repository(workspace)
    return completed.stdout


def _git_blob(workspace: Path, revision: str, relative: str) -> str:
    value = _git_text(workspace, "rev-parse", f"{revision}:{relative}").strip()
    if not _GIT_OBJECT_RE.fullmatch(value):
        raise W14OperatorError(f"invalid Git blob for {revision}:{relative}")
    return value


def _canonical_rule() -> dict[str, Any]:
    return {
        "maximum_candidate_p95_latency_ms": 750.0,
        "minimum_candidate_completed_worlds_mean": None,
        "minimum_level_utility_estimate": 0.0,
        "minimum_level_utility_lower_95": -0.05,
        "minimum_point_margin_estimate": 0.0,
        "minimum_win_rate_estimate": -0.02,
        "require_zero_invalidating_failures": True,
        "rule_id": "w1.4-four-change-combination-v1",
        "style_metric_bounds": {},
    }


def _canonical_launch() -> dict[str, Any]:
    return enoch_week1_runner.build_launch_configuration(
        candidate_arm_ids=CANDIDATE_ARM_IDS,
        worlds=24,
        candidates=6,
        rollout_tricks=6,
        scenario_id="standard",
        deadline_ms=30_000,
    )


def validate_committed_plan(plan: Mapping[str, Any]) -> str:
    """Validate every result-independent W1.4 choice."""

    expected_keys = {
        "automatic_production_promotion_allowed",
        "available_parallelism",
        "candidate_arm_ids",
        "candidate_fingerprint",
        "campaign_lineage_fingerprint",
        "candidate_policy_configuration_sha256",
        "control_fingerprint",
        "development_rule",
        "development_rule_sha256",
        "environment_fingerprint",
        "evaluator_fingerprint",
        "failure_disposition",
        "final_consumed_count",
        "fixed_work",
        "launch_configuration_sha256",
        "manifest_kind",
        "manifest_version",
        "parent_phase_manifest_fingerprint",
        "preclaim_consumed_count",
        "preclaim_ledger_fingerprint",
        "prerequisite_failure_consumed_count",
        "protocol_fingerprint",
        "recovered_supported_change_set_fingerprint",
        "required_style_metrics",
        "seal_recovery_manifest_fingerprint",
        "seal_recovery_provenance_fingerprint",
        "shard_count",
        "shard_timeout_seconds",
        "source_control_contract",
        "stages",
        "subject_id",
        "rust_toolchain_contract",
        "worker_count",
    }
    _require_exact_keys(plan, expected_keys, "committed W1.4 plan")
    exact = {
        "automatic_production_promotion_allowed": False,
        "available_parallelism": 10,
        "candidate_arm_ids": list(CANDIDATE_ARM_IDS),
        "candidate_fingerprint": "abebed7aee684612282e37513b37592bf3fe64f96a34f597870fe72cf3bbe706",
        "campaign_lineage_fingerprint": "b81c9709773431adea7e266d18badb2bcfb4f4fb4ce9d79e4b9444b5fee162fd",
        "candidate_policy_configuration_sha256": "cf38d9c97ba9f2f1b8a4802077442bd50c6d77fc2b77a6214081e131fa8d7ce9",
        "control_fingerprint": "90b32110833102879e299cb78c456aa3392023be9691caa501076b0d6aea05ab",
        "development_rule_sha256": "e10248108df8e28020f651a060d3db7da1ede61ffdd91752c9ebd869adfd85f1",
        "environment_fingerprint": ENVIRONMENT_FINGERPRINT,
        "evaluator_fingerprint": "94b676dd9cb0eb614c3774b3404df533376597b5eebd88afb6f72a56a930b868",
        "failure_disposition": FAILURE_DISPOSITION,
        "final_consumed_count": FINAL_COUNT,
        "launch_configuration_sha256": "073727092ea58da28c7a3139ae9cf683d3abf90251f53edad753aeb4289d75ec",
        "manifest_kind": PLAN_KIND,
        "manifest_version": MANIFEST_VERSION,
        "parent_phase_manifest_fingerprint": PARENT_PHASE,
        "preclaim_consumed_count": PRECLAIM_COUNT,
        "preclaim_ledger_fingerprint": PARENT_LEDGER,
        "prerequisite_failure_consumed_count": PRECLAIM_COUNT,
        "protocol_fingerprint": PROTOCOL_FINGERPRINT,
        "recovered_supported_change_set_fingerprint": PARENT_SET,
        "seal_recovery_manifest_fingerprint": RECOVERY_MANIFEST,
        "seal_recovery_provenance_fingerprint": RECOVERY_PROVENANCE,
        "shard_count": 8,
        "shard_timeout_seconds": 3600,
        "source_control_contract": _source_control_contract(),
        "subject_id": "enoch-1-candidate",
        "rust_toolchain_contract": _rust_toolchain_contract(),
        "worker_count": 8,
    }
    for field, expected in exact.items():
        if plan[field] != expected:
            raise W14OperatorError(f"committed W1.4 plan changed {field}")
    if plan["development_rule"] != _canonical_rule():
        raise W14OperatorError("committed W1.4 rule changed")
    if enoch_week1.validate_development_rule(plan["development_rule"]) != plan[
        "development_rule_sha256"
    ]:
        raise W14OperatorError("committed W1.4 rule hash changed")
    launch = _canonical_launch()
    if launch != plan["fixed_work"] | {
        "arm_feature_mapping_sha256": launch["arm_feature_mapping_sha256"],
        "candidate_arm_ids": list(CANDIDATE_ARM_IDS),
        "scenario_bindings_sha256": launch["scenario_bindings_sha256"],
        "scenario_id": "standard",
    }:
        raise W14OperatorError("W1.4 fixed-work launch does not reconstruct")
    if enoch_week1_runner.validate_launch_configuration(launch) != plan[
        "launch_configuration_sha256"
    ]:
        raise W14OperatorError("W1.4 launch hash changed")
    if tuple(plan["required_style_metrics"]) != enoch_week1.WEEK1_STYLE_METRICS:
        raise W14OperatorError("W1.4 changed the complete style schema")
    expected_stages = (
        (
            "qualification",
            "combination-qualification",
            "b8df3bc4f03bece9ac24e86e5f78fa971b00e82f2b079d38db711b9cd8abe8b1",
            "dev/combination/qualification",
            300,
            "874bb3f0de68e620db4827bc0ab9d4e5a2334e628df2d429ad814a14773a55f8",
            [38, 38, 38, 38, 37, 37, 37, 37],
        ),
        (
            "screen",
            "combination-screen",
            "99899173addfae419add144d73f9760fbe9499a67666e285dfae744d91d99f88",
            "dev/combination/screen",
            800,
            "f418d445db0f51a4cad7c99e3da3ec84cebad282b1a6843f8b1479edc7c86a28",
            [100] * 8,
        ),
    )
    stages = plan["stages"]
    if not isinstance(stages, list) or len(stages) != 2:
        raise W14OperatorError("W1.4 plan requires exactly two stages")
    stage_keys = {
        "comparison_id",
        "comparison_protocol_fingerprint",
        "pair_count",
        "seed_namespace",
        "seed_set_sha256",
        "shard_pair_counts",
        "stage_id",
    }
    for stage, expected in zip(stages, expected_stages):
        _require_exact_keys(stage, stage_keys, "W1.4 plan stage")
        observed = (
            stage["stage_id"],
            stage["comparison_id"],
            stage["comparison_protocol_fingerprint"],
            stage["seed_namespace"],
            stage["pair_count"],
            stage["seed_set_sha256"],
            stage["shard_pair_counts"],
        )
        if observed != expected:
            raise W14OperatorError(f"W1.4 {expected[0]} stage changed")
    return enoch_week1.canonical_json_sha256(plan)


def load_committed_plan(workspace: Path) -> dict[str, Any]:
    plan = _load_json(workspace.resolve() / PLAN_RELATIVE)
    validate_committed_plan(plan)
    return plan


def _continuation_git_identity(
    workspace: Path, *, continuation_commit: str | None = None, require_live: bool
) -> dict[str, Any]:
    workspace = workspace.expanduser().resolve()
    top = Path(_git_text(workspace, "rev-parse", "--show-toplevel").strip()).resolve()
    if top != workspace:
        raise W14OperatorError("W1.4 workspace must be its Git root")
    if require_live:
        status = _git_text(
            workspace, "status", "--porcelain=v1", "--untracked-files=all"
        )
        if status:
            raise W14OperatorError(
                f"W1.4 workspace is not clean: {status.splitlines()[0]}"
            )
        head = _git_text(workspace, "rev-parse", "HEAD^{commit}").strip()
        if continuation_commit is not None and continuation_commit != head:
            raise W14OperatorError("requested W1.4 commit differs from live HEAD")
    else:
        if continuation_commit is None:
            raise W14OperatorError("stored W1.4 provenance lacks its commit")
        head = _git_text(
            workspace, "rev-parse", f"{continuation_commit}^{{commit}}"
        ).strip()
    parents = _git_text(workspace, "rev-list", "--parents", "-n", "1", head).split()
    if parents != [head, BASE_COMMIT]:
        raise W14OperatorError(
            "W1.4 continuation must be one direct child of dbc1d1b"
        )
    raw = _git_text(
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
        raise W14OperatorError(
            "W1.4 continuation must add exactly operator, plan, and tests"
        )
    changes = []
    for status_code, relative in parsed:
        fields = _git_text(workspace, "ls-tree", head, "--", relative).strip().split(
            None, 3
        )
        if (
            len(fields) != 4
            or fields[0] != "100644"
            or fields[1] != "blob"
            or fields[3] != relative
        ):
            raise W14OperatorError(f"W1.4 source is not a regular blob: {relative}")
        blob = _git_bytes(workspace, "show", f"{head}:{relative}")
        digest = hashlib.sha256(blob).hexdigest()
        if require_live:
            path = workspace / relative
            if not path.is_file() or path.is_symlink() or _sha256_file(path) != digest:
                raise W14OperatorError(f"live W1.4 source differs from Git: {relative}")
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
            raise W14OperatorError(f"W1.4 changed frozen runtime module: {relative}")
        critical.append({"blob": base_blob, "path": relative})
    return {
        "base_tree_manifest_sha256": hashlib.sha256(
            _git_bytes(workspace, "ls-tree", "-r", "-z", "--full-tree", BASE_COMMIT)
        ).hexdigest(),
        "changed_paths": changes,
        "continuation_git_commit": head,
        "continuation_git_tree": _git_text(
            workspace, "rev-parse", f"{head}^{{tree}}"
        ).strip(),
        "critical_base_module_blobs": critical,
        "git_tree_manifest_sha256": hashlib.sha256(
            _git_bytes(workspace, "ls-tree", "-r", "-z", "--full-tree", head)
        ).hexdigest(),
    }


def _runtime_import_records(workspace: Path, commit: str | None = None) -> list[dict[str, str]]:
    records = []
    for module, relative in RUNTIME_MODULES:
        expected_path = (workspace / relative).resolve()
        module_path = getattr(module, "__file__", None)
        if not isinstance(module_path, str) or Path(module_path).resolve() != expected_path:
            raise W14OperatorError(f"W1.4 runtime import is shadowed: {relative}")
        digest = _sha256_file(expected_path)
        if commit is not None:
            git_digest = hashlib.sha256(
                _git_bytes(workspace, "show", f"{commit}:{relative}")
            ).hexdigest()
            if digest != git_digest:
                raise W14OperatorError(f"live W1.4 runtime differs from Git: {relative}")
        records.append({"path": relative, "sha256": digest})
    return records


def build_continuation_provenance(
    workspace: Path, plan: Mapping[str, Any]
) -> dict[str, Any]:
    validate_committed_plan(plan)
    workspace = workspace.expanduser().resolve()
    if Path(__file__).resolve() != (workspace / OPERATOR_RELATIVE).resolve():
        raise W14OperatorError("executing W1.4 operator is outside its workspace")
    identity = _continuation_git_identity(workspace, require_live=True)
    body = {
        "automatic_production_promotion_allowed": False,
        "base_git_commit": BASE_COMMIT,
        "base_git_tree": _git_text(
            workspace, "rev-parse", f"{BASE_COMMIT}^{{tree}}"
        ).strip(),
        "base_tree_manifest_sha256": identity["base_tree_manifest_sha256"],
        "changed_paths": identity["changed_paths"],
        "continuation_git_commit": identity["continuation_git_commit"],
        "continuation_git_tree": identity["continuation_git_tree"],
        "critical_base_module_blobs": identity["critical_base_module_blobs"],
        "git_tree_manifest_sha256": identity["git_tree_manifest_sha256"],
        "manifest_kind": PROVENANCE_KIND,
        "manifest_version": MANIFEST_VERSION,
        "operator_source_path": OPERATOR_RELATIVE.as_posix(),
        "operator_source_sha256": _sha256_file(workspace / OPERATOR_RELATIVE),
        "parent_final_ledger_fingerprint": PARENT_LEDGER,
        "parent_phase_manifest_fingerprint": PARENT_PHASE,
        "parent_supported_change_set_fingerprint": PARENT_SET,
        "plan_file_sha256": _sha256_file(workspace / PLAN_RELATIVE),
        "plan_sha256": enoch_week1.canonical_json_sha256(plan),
        "protocol_fingerprint": PROTOCOL_FINGERPRINT,
        "recovery_manifest_fingerprint": RECOVERY_MANIFEST,
        "recovery_provenance_fingerprint": RECOVERY_PROVENANCE,
        "runtime_imports": _runtime_import_records(workspace),
        "rust_toolchain_contract_fingerprint": plan["rust_toolchain_contract"][
            "rust_toolchain_contract_fingerprint"
        ],
        "source_control_contract_fingerprint": plan["source_control_contract"][
            "source_control_contract_fingerprint"
        ],
        "starting_consumed_count": PRECLAIM_COUNT,
        "test_source_path": TEST_RELATIVE.as_posix(),
        "test_source_sha256": _sha256_file(workspace / TEST_RELATIVE),
    }
    return _with_fingerprint(body, "continuation_provenance_fingerprint")


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
        if dict(artifact) != expected:
            raise W14OperatorError("W1.4 provenance does not reconstruct")
        return expected["continuation_provenance_fingerprint"]
    _require_exact_keys(
        artifact,
        {
            "automatic_production_promotion_allowed",
            "base_git_commit",
            "base_git_tree",
            "base_tree_manifest_sha256",
            "changed_paths",
            "continuation_git_commit",
            "continuation_git_tree",
            "continuation_provenance_fingerprint",
            "critical_base_module_blobs",
            "git_tree_manifest_sha256",
            "manifest_kind",
            "manifest_version",
            "operator_source_path",
            "operator_source_sha256",
            "parent_final_ledger_fingerprint",
            "parent_phase_manifest_fingerprint",
            "parent_supported_change_set_fingerprint",
            "plan_file_sha256",
            "plan_sha256",
            "protocol_fingerprint",
            "recovery_manifest_fingerprint",
            "recovery_provenance_fingerprint",
            "runtime_imports",
            "rust_toolchain_contract_fingerprint",
            "source_control_contract_fingerprint",
            "starting_consumed_count",
            "test_source_path",
            "test_source_sha256",
        },
        "stored W1.4 provenance",
    )
    fingerprint = _validate_fingerprint(
        artifact, "continuation_provenance_fingerprint", "stored W1.4 provenance"
    )
    commit = artifact.get("continuation_git_commit")
    if not isinstance(commit, str):
        raise W14OperatorError("stored W1.4 provenance has no commit")
    identity = _continuation_git_identity(
        workspace, continuation_commit=commit, require_live=False
    )
    for field in (
        "base_tree_manifest_sha256",
        "changed_paths",
        "continuation_git_commit",
        "continuation_git_tree",
        "critical_base_module_blobs",
        "git_tree_manifest_sha256",
    ):
        if artifact.get(field) != identity[field]:
            raise W14OperatorError(f"stored W1.4 provenance changed {field}")
    exact = {
        "automatic_production_promotion_allowed": False,
        "base_git_commit": BASE_COMMIT,
        "base_git_tree": _git_text(
            workspace, "rev-parse", f"{BASE_COMMIT}^{{tree}}"
        ).strip(),
        "manifest_kind": PROVENANCE_KIND,
        "manifest_version": MANIFEST_VERSION,
        "operator_source_path": OPERATOR_RELATIVE.as_posix(),
        "parent_final_ledger_fingerprint": PARENT_LEDGER,
        "parent_phase_manifest_fingerprint": PARENT_PHASE,
        "parent_supported_change_set_fingerprint": PARENT_SET,
        "plan_sha256": enoch_week1.canonical_json_sha256(plan),
        "protocol_fingerprint": PROTOCOL_FINGERPRINT,
        "recovery_manifest_fingerprint": RECOVERY_MANIFEST,
        "recovery_provenance_fingerprint": RECOVERY_PROVENANCE,
        "rust_toolchain_contract_fingerprint": plan["rust_toolchain_contract"][
            "rust_toolchain_contract_fingerprint"
        ],
        "source_control_contract_fingerprint": plan["source_control_contract"][
            "source_control_contract_fingerprint"
        ],
        "starting_consumed_count": PRECLAIM_COUNT,
        "test_source_path": TEST_RELATIVE.as_posix(),
    }
    for field, expected in exact.items():
        if artifact.get(field) != expected:
            raise W14OperatorError(f"stored W1.4 provenance changed {field}")
    source_hashes = {
        relative: hashlib.sha256(
            _git_bytes(workspace, "show", f"{commit}:{relative}")
        ).hexdigest()
        for relative in CONTINUATION_PATHS
    }
    if artifact.get("operator_source_sha256") != source_hashes[OPERATOR_RELATIVE.as_posix()]:
        raise W14OperatorError("stored W1.4 operator hash changed")
    if artifact.get("plan_file_sha256") != source_hashes[PLAN_RELATIVE.as_posix()]:
        raise W14OperatorError("stored W1.4 plan hash changed")
    if artifact.get("test_source_sha256") != source_hashes[TEST_RELATIVE.as_posix()]:
        raise W14OperatorError("stored W1.4 test hash changed")
    if _sha256_file(Path(__file__).resolve()) != artifact["operator_source_sha256"]:
        raise W14OperatorError("executing W1.4 operator differs from stored source")
    expected_runtime = [
        {
            "path": relative,
            "sha256": hashlib.sha256(
                _git_bytes(workspace, "show", f"{commit}:{relative}")
            ).hexdigest(),
        }
        for _module, relative in RUNTIME_MODULES
    ]
    if artifact.get("runtime_imports") != expected_runtime:
        raise W14OperatorError("stored W1.4 runtime hashes changed")
    if _runtime_import_records(workspace, commit) != expected_runtime:
        raise W14OperatorError("live W1.4 runtime imports changed")
    return fingerprint


def verify_recovered_w1_3(
    layout: W14Layout, workspace: Path, w1_2_workspace: Path
) -> dict[str, Any]:
    """Verify recovery before the caller acquires the shared operator lock."""

    workspace = workspace.expanduser().resolve()
    w1_2_workspace = w1_2_workspace.expanduser().resolve()
    _validate_trusted_git()
    _validate_trusted_git_repository(workspace)
    command = [
        sys.executable,
        "-I",
        "-B",
        "-c",
        RECOVERY_RUNPY_BOOTSTRAP,
        str(workspace),
        str(workspace / "training/enoch_week1_w1_3_seal_recovery.py"),
        "verify",
        "--root",
        str(layout.root.resolve()),
        "--workspace",
        str(workspace),
        "--w1-2-workspace",
        str(w1_2_workspace),
    ]
    completed = subprocess.run(
        command,
        cwd=workspace,
        env=_source_control_environment(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=1200,
    )
    _validate_trusted_git()
    _validate_trusted_git_repository(workspace)
    if completed.returncode != 0 or completed.stderr:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise W14OperatorError(
            f"isolated recovery-aware W1.3 verification failed: {detail}"
        )
    try:
        result = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise W14OperatorError(
            "isolated recovery-aware W1.3 verifier did not emit JSON"
        ) from exc
    if result != EXPECTED_PARENT_RESULT:
        raise W14OperatorError("recovery-aware W1.3 verification result changed")
    return result


def _supported_set(layout: W14Layout) -> dict[str, Any]:
    artifact = _load_json(layout.parent.supported_set)
    if _validate_fingerprint(
        artifact, "supported_change_set_fingerprint", "recovered supported set"
    ) != PARENT_SET:
        raise W14OperatorError("recovered supported set fingerprint changed")
    summary = artifact.get("summary")
    if not isinstance(summary, Mapping):
        raise W14OperatorError("recovered supported set lacks its summary")
    if (
        artifact.get("status") != "supported-survivors"
        or summary.get("supported_arm_ids") != list(CANDIDATE_ARM_IDS)
        or summary.get("stopped_arm_ids") != [STOPPED_ARM_ID]
        or artifact.get("final_ledger_fingerprint") != PARENT_LEDGER
    ):
        raise W14OperatorError("recovered W1.3 exit set changed")
    stopped = [
        item for item in artifact.get("arm_results", []) if item.get("arm_id") == STOPPED_ARM_ID
    ]
    if len(stopped) != 1 or stopped[0].get("support_decision_fingerprint") != STOPPED_ARM_DECISION:
        raise W14OperatorError("team-void-boss stop decision changed")
    return artifact


def _load_parent_state(
    layout: W14Layout,
    workspace: Path,
    *,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    _validate_trusted_git()
    _validate_trusted_git_repository(workspace)
    original_helpers = (
        recovery._git_text,  # noqa: SLF001
        recovery._git_bytes,  # noqa: SLF001
        w13._git_text,  # noqa: SLF001
        w13._git_bytes,  # noqa: SLF001
    )
    recovery._git_text = _git_text  # type: ignore[attr-defined]  # noqa: SLF001
    recovery._git_bytes = _git_bytes  # type: ignore[attr-defined]  # noqa: SLF001
    w13._git_text = _git_text  # type: ignore[attr-defined]  # noqa: SLF001
    w13._git_bytes = _git_bytes  # type: ignore[attr-defined]  # noqa: SLF001
    try:
        context = recovery._load_validated_context(  # noqa: SLF001
            layout.parent,
            workspace,
            environment=environment,
            require_live_ledger_exact=False,
        )
    finally:
        (
            recovery._git_text,  # noqa: SLF001
            recovery._git_bytes,  # noqa: SLF001
            w13._git_text,  # noqa: SLF001
            w13._git_bytes,  # noqa: SLF001
        ) = original_helpers
        _validate_trusted_git()
        _validate_trusted_git_repository(workspace)
    state = context["state"]
    phase3 = _load_json(layout.parent.phase)
    chain = enoch_week1.validate_phase_chain(
        state["protocol"],
        [state["phase0"], state["phase1"], state["phase2"], phase3],
    )
    if chain[-1] != PARENT_PHASE:
        raise W14OperatorError("recovered W1.3 phase changed")
    parent_ledger = _load_json(layout.parent.final_ledger)
    enoch_week1.validate_seed_ledger(state["protocol"], parent_ledger)
    if (
        parent_ledger.get("ledger_fingerprint") != PARENT_LEDGER
        or len(parent_ledger.get("consumed", [])) != PRECLAIM_COUNT
    ):
        raise W14OperatorError("recovered W1.3 ledger changed")
    supported = _supported_set(layout)
    live = _load_json(layout.base.ledger)
    enoch_week1.validate_seed_ledger(state["protocol"], live)
    return {
        **state,
        "arm_evidence": context["arm_evidence"],
        "parent_context": context,
        "parent_ledger": parent_ledger,
        "phase3": phase3,
        "supported_set": supported,
        "live_ledger": live,
    }


def _survivor_evidence(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for arm_id in CANDIDATE_ARM_IDS:
        parent = state["parent_arms"][arm_id]
        survivor = state["arm_evidence"][arm_id]
        if survivor["support_decision"]["decision"] != "advance-to-w1.4":
            raise W14OperatorError(f"{arm_id} is not a recovered W1.4 survivor")
        result.append(
            {
                "ablation": {
                    "comparison": parent["comparison"],
                    "identity_bindings": parent["identity_bindings"],
                    "launch_configuration": parent["launch_configuration"],
                    "merged_result": parent["merged_result"],
                },
                "advancement_decision": parent["advancement_decision"],
                "arm_id": arm_id,
                "fixture_report": state["fixture"],
                "survivor_screen": {
                    "comparison": survivor["comparison"],
                    "identity_bindings": survivor["identity_bindings"],
                    "launch_configuration": survivor["launch_configuration"],
                    "merged_result": survivor["merged_result"],
                },
            }
        )
    return result


def build_campaign_declaration(
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
    provenance: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    environment: Mapping[str, str] | None = None,
    environment_identity_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_committed_plan(plan)
    enoch_week1.validate_protocol(protocol)
    workspace = Path(__file__).resolve().parents[1]
    source_control = _validate_trusted_git()
    _validate_trusted_git_repository(workspace)
    toolchain = _validate_live_rust_toolchain(workspace, state["control"])
    if plan["source_control_contract"] != source_control:
        raise W14OperatorError("W1.4 source-control contract changed")
    if plan["rust_toolchain_contract"] != toolchain:
        raise W14OperatorError("W1.4 Rust toolchain contract changed")
    if (environment is None) == (environment_identity_override is None):
        raise W14OperatorError(
            "choose exactly one live environment or frozen environment identity"
        )
    launch = _canonical_launch()
    identities = enoch_week1_runner.build_in_process_identity_bindings(
        state["control"]["evaluator_identity"], launch
    )
    if environment_identity_override is None:
        child_environment, _ = enoch_week1.sanitized_evaluator_environment(
            environment,
            allowlist=protocol["evaluator_environment_policy"]["allowlist"],
        )
        environment_identity = enoch_week1_runner.build_evaluator_environment_identity(
            identities["evaluator"],
            protocol,
            child_environment,
            available_parallelism=plan["available_parallelism"],
        )
    else:
        environment_identity = dict(environment_identity_override)
    environment_fingerprint = enoch_week1.canonical_json_sha256(environment_identity)
    observed = {
        "candidate_fingerprint": enoch_week1.canonical_json_sha256(
            identities["candidate"]
        ),
        "control_fingerprint": enoch_week1.canonical_json_sha256(identities["control"]),
        "environment_fingerprint": environment_fingerprint,
        "evaluator_fingerprint": enoch_week1.canonical_json_sha256(
            identities["evaluator"]
        ),
        "launch_configuration_sha256": enoch_week1_runner.validate_launch_configuration(
            launch
        ),
    }
    for field, value in observed.items():
        if plan[field] != value:
            raise W14OperatorError(f"W1.4 live identity changed {field}")
    stages = []
    for sequence, stage_plan in enumerate(plan["stages"], start=1):
        comparison = enoch_week1.build_comparison_protocol_manifest(
            protocol,
            phase="W1.4",
            comparison_id=stage_plan["comparison_id"],
            subject_id=plan["subject_id"],
            seed_namespace=stage_plan["seed_namespace"],
            pair_count=stage_plan["pair_count"],
            shard_count=plan["shard_count"],
            candidate_fingerprint=plan["candidate_fingerprint"],
            control_fingerprint=plan["control_fingerprint"],
            evaluator_fingerprint=plan["evaluator_fingerprint"],
            environment_fingerprint=plan["environment_fingerprint"],
            configuration_fingerprint=plan["launch_configuration_sha256"],
            development_rule=plan["development_rule"],
            required_style_metrics=enoch_week1.WEEK1_STYLE_METRICS,
        )
        if comparison["seed_set_sha256"] != stage_plan["seed_set_sha256"]:
            raise W14OperatorError(f"{stage_plan['stage_id']} seed set changed")
        if comparison["comparison_protocol_fingerprint"] != stage_plan[
            "comparison_protocol_fingerprint"
        ]:
            raise W14OperatorError(
                f"{stage_plan['stage_id']} comparison fingerprint changed"
            )
        sizes = [len(item["seed_indices"]) for item in comparison["shards"]]
        if sizes != stage_plan["shard_pair_counts"]:
            raise W14OperatorError(f"{stage_plan['stage_id']} shard partition changed")
        stages.append(
            {
                "comparison": comparison,
                "environment_identity": copy.deepcopy(environment_identity),
                "identity_bindings": copy.deepcopy(identities),
                "launch_configuration": copy.deepcopy(launch),
                "sequence": sequence,
                "stage_id": stage_plan["stage_id"],
            }
        )
    lineage = enoch_week1_campaign.build_w1_4_campaign_lineage(
        protocol,
        qualification_comparison=stages[0]["comparison"],
        qualification_launch_configuration=launch,
        qualification_identity_bindings=identities,
        screen_comparison=stages[1]["comparison"],
        screen_launch_configuration=launch,
        screen_identity_bindings=identities,
        survivor_evidence=_survivor_evidence(state),
    )
    if lineage[enoch_week1_campaign.FINGERPRINT_FIELD] != plan[
        "campaign_lineage_fingerprint"
    ]:
        raise W14OperatorError("W1.4 campaign lineage fingerprint changed")
    body = {
        "arm_registry_sha256": enoch_week1.ARM_REGISTRY_SHA256,
        "automatic_production_promotion_allowed": False,
        "candidate_arm_ids": list(CANDIDATE_ARM_IDS),
        "campaign_lineage": lineage,
        "campaign_lineage_fingerprint": lineage[
            enoch_week1_campaign.FINGERPRINT_FIELD
        ],
        "execution_contract": {
            "available_parallelism": 10,
            "failure_disposition": FAILURE_DISPOSITION,
            "final_consumed_count": FINAL_COUNT,
            "fixed_work": copy.deepcopy(plan["fixed_work"]),
            "regression_gate_required_before_claims": True,
            "require_machine_global_lock": True,
            "required_external_artifact_ids": list(
                enoch_week1_evidence.EXPECTED_ARTIFACT_IDS
            ),
            "required_failure_counters": list(enoch_week1.FAILURE_COUNTER_NAMES),
            "required_style_metrics": list(enoch_week1.WEEK1_STYLE_METRICS),
            "shard_count": 8,
            "stage_order": list(STAGE_IDS),
            "timeout_seconds": 3600,
            "worker_count": 8,
        },
        "manifest_kind": DECLARATION_KIND,
        "manifest_version": MANIFEST_VERSION,
        "operator_source_provenance_fingerprint": provenance[
            "continuation_provenance_fingerprint"
        ],
        "parent_bindings": {
            "final_ledger_fingerprint": PARENT_LEDGER,
            "malformed_supported_change_set_fingerprint": MALFORMED_SET,
            "phase_manifest_fingerprint": PARENT_PHASE,
            "seal_recovery_manifest_fingerprint": RECOVERY_MANIFEST,
            "seal_recovery_provenance_fingerprint": RECOVERY_PROVENANCE,
            "supported_change_set_fingerprint": PARENT_SET,
        },
        "plan_sha256": enoch_week1.canonical_json_sha256(plan),
        "protocol_fingerprint": PROTOCOL_FINGERPRINT,
        "regression_contract": {
            "absolute_pinned_cargo": True,
            "cargo_configuration_forbidden": True,
            "fresh_cargo_target_per_attempt": True,
            "full_fixture_suite": True,
            "full_mechanics_suite": True,
            "model_contract_tests": True,
            "reconstruct_sealed_preflight_and_control": True,
            "strict_evaluator_test": True,
            "toolchain_identity_before_and_after_each_command": True,
            "zero_seed_claims": True,
        },
        "rust_toolchain_contract_fingerprint": toolchain[
            "rust_toolchain_contract_fingerprint"
        ],
        "seed_registry_sha256": protocol["seed_registry_sha256"],
        "source_control_contract_fingerprint": source_control[
            "source_control_contract_fingerprint"
        ],
        "stages": stages,
    }
    return _with_fingerprint(body, "campaign_declaration_fingerprint")


def validate_campaign_declaration(
    declaration: Mapping[str, Any],
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
    provenance: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    environment: Mapping[str, str] | None = None,
    environment_identity_override: Mapping[str, Any] | None = None,
) -> str:
    expected = build_campaign_declaration(
        protocol,
        plan,
        provenance,
        state,
        environment=environment,
        environment_identity_override=environment_identity_override,
    )
    if dict(declaration) != expected:
        raise W14OperatorError("W1.4 campaign declaration does not reconstruct")
    return expected["campaign_declaration_fingerprint"]


def _stage_by_id(declaration: Mapping[str, Any], stage_id: str) -> Mapping[str, Any]:
    matches = [item for item in declaration["stages"] if item.get("stage_id") == stage_id]
    if len(matches) != 1:
        raise W14OperatorError(f"W1.4 declaration does not name {stage_id} once")
    return matches[0]


def _materialize_declaration(
    layout: W14Layout,
    plan: Mapping[str, Any],
    provenance: Mapping[str, Any],
    declaration: Mapping[str, Any],
) -> None:
    _write_or_match(layout.input, plan, "W1.4 committed plan copy")
    _write_or_match(layout.provenance, provenance, "W1.4 continuation provenance")
    _write_or_match(layout.lineage, declaration["campaign_lineage"], "W1.4 lineage")
    for stage in declaration["stages"]:
        root = layout.stage(stage["sequence"], stage["stage_id"]) / "declaration"
        files = {
            "comparison.json": stage["comparison"],
            "development-rule.json": stage["comparison"]["development_rule"],
            "environment-identity.json": stage["environment_identity"],
            "identities.json": stage["identity_bindings"],
            "launch.json": stage["launch_configuration"],
        }
        for name, value in files.items():
            _write_or_match(root / name, value, f"{stage['stage_id']} {name}")
    _write_or_match(layout.declaration, declaration, "W1.4 declaration index")


def _verify_materialized_declaration(
    layout: W14Layout,
    plan: Mapping[str, Any],
    provenance: Mapping[str, Any],
    declaration: Mapping[str, Any],
) -> None:
    if _load_json(layout.input) != dict(plan):
        raise W14OperatorError("materialized W1.4 plan changed")
    if _load_json(layout.provenance) != dict(provenance):
        raise W14OperatorError("materialized W1.4 provenance changed")
    if _load_json(layout.lineage) != declaration["campaign_lineage"]:
        raise W14OperatorError("materialized W1.4 lineage changed")
    for stage in declaration["stages"]:
        root = layout.stage(stage["sequence"], stage["stage_id"]) / "declaration"
        expected = {
            "comparison.json": stage["comparison"],
            "development-rule.json": stage["comparison"]["development_rule"],
            "environment-identity.json": stage["environment_identity"],
            "identities.json": stage["identity_bindings"],
            "launch.json": stage["launch_configuration"],
        }
        for name, value in expected.items():
            if _load_json(root / name) != value:
                raise W14OperatorError(f"materialized {stage['stage_id']} {name} changed")
    if _load_json(layout.declaration) != dict(declaration):
        raise W14OperatorError("materialized W1.4 declaration changed")


def _require_exact_preclaim_ledger(state: Mapping[str, Any]) -> None:
    if state["live_ledger"] != state["parent_ledger"]:
        raise W14OperatorError(
            "W1.4 declaration requires the exact 18,711-claim W1.3 ledger"
        )


def declare_w1_4(
    layout: W14Layout,
    workspace: Path,
    w1_2_workspace: Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    workspace = workspace.expanduser().resolve()
    base_environment = dict(os.environ if environment is None else environment)
    verify_recovered_w1_3(layout, workspace, w1_2_workspace)
    _require_safe_w1_4_tree(layout)
    with base_operator._operator_lock(layout.base.operator_lock):  # noqa: SLF001
        plan = load_committed_plan(workspace)
        state = _load_parent_state(layout, workspace, environment=base_environment)
        _require_exact_preclaim_ledger(state)
        if layout.retirement.exists():
            raise W14OperatorError("this protocol is already retired for W1.4")
        provenance = build_continuation_provenance(workspace, plan)
        declaration = build_campaign_declaration(
            state["protocol"],
            plan,
            provenance,
            state,
            environment=base_environment,
        )
        _materialize_declaration(layout, plan, provenance, declaration)
        return {
            "campaign_declaration_fingerprint": declaration[
                "campaign_declaration_fingerprint"
            ],
            "campaign_lineage_fingerprint": declaration[
                "campaign_lineage_fingerprint"
            ],
            "candidate_arm_ids": list(CANDIDATE_ARM_IDS),
            "continuation_provenance_fingerprint": provenance[
                "continuation_provenance_fingerprint"
            ],
            "planned_pair_count": 1_100,
            "protocol_fingerprint": PROTOCOL_FINGERPRINT,
        }


def _attempt_directories(root: Path) -> list[Path]:
    root = _require_safe_w1_4_directory(root, create=False, missing_ok=True)
    if not root.exists():
        return []
    result = []
    for path in root.iterdir():
        if not path.is_dir() or path.is_symlink() or not _ATTEMPT_RE.fullmatch(path.name):
            raise W14OperatorError(f"malformed W1.4 attempt entry: {path}")
        result.append(path)
    return sorted(result)


def _next_attempt(root: Path) -> Path:
    root = _require_safe_w1_4_directory(root, create=True)
    attempts = _attempt_directories(root)
    ordinal = max((int(path.name[-3:]) for path in attempts), default=0) + 1
    attempt = root / f"attempt-{ordinal:03d}"
    try:
        attempt.mkdir(exist_ok=False)
    except OSError as exc:
        raise W14OperatorError(f"could not create W1.4 attempt: {exc}") from exc
    return _require_safe_w1_4_directory(attempt, create=False)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_seconds(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _finish_observation(started: datetime) -> datetime:
    ended = _utc_now()
    while _utc_seconds(ended) == _utc_seconds(started):
        time.sleep(0.05)
        ended = _utc_now()
    return ended


def _w1_4_claims(ledger: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        item
        for item in ledger["consumed"]
        if item["namespace"].startswith("dev/combination/")
    ]


def _write_regression_failure(
    attempt: Path,
    exc: BaseException,
    *,
    lock_comparison_protocol_fingerprint: str | None = None,
    control_manifest_fingerprint: str | None = None,
    rust_toolchain_contract_fingerprint: str | None = None,
    source_control_contract_fingerprint: str | None = None,
) -> dict[str, Any]:
    logs = []
    logs_root = attempt / "logs"
    if logs_root.is_dir():
        logs = [
            {
                "path": path.relative_to(attempt).as_posix(),
                "sha256": _sha256_file(path),
            }
            for path in sorted(logs_root.rglob("*"))
            if path.is_file() and not path.is_symlink()
        ]
    environment_path = attempt / "regression-environment.json"
    regression_environment = _load_json(environment_path) if environment_path.is_file() else None
    body = {
        "attempt_id": attempt.name,
        "automatic_production_promotion_allowed": False,
        "error": f"{type(exc).__name__}: {exc}",
        "failed_command": (
            copy.deepcopy(exc.record) if isinstance(exc, RegressionTestFailure) else None
        ),
        "failure_class": (
            "candidate-prerequisite-failure"
            if isinstance(exc, RegressionTestFailure)
            else "operational-preclaim-failure"
        ),
        "manifest_kind": "enoch-week1-w1.4-regression-preclaim-failure",
        "manifest_version": MANIFEST_VERSION,
        "preserved_logs": logs,
        "exclusive_campaign_lock_held": lock_comparison_protocol_fingerprint is not None,
        "lock_comparison_protocol_fingerprint": lock_comparison_protocol_fingerprint,
        "control_manifest_fingerprint": control_manifest_fingerprint,
        "regression_environment": regression_environment,
        "regression_environment_sha256": (
            enoch_week1.canonical_json_sha256(regression_environment)
            if regression_environment is not None
            else None
        ),
        "retry_disposition": (
            "terminal-candidate-prerequisite-failure"
            if isinstance(exc, RegressionTestFailure)
            else "new-attempt-allowed-only-while-w1.4-unconsumed"
        ),
        "rust_toolchain_contract_fingerprint": rust_toolchain_contract_fingerprint,
        "seed_claim_count": 0,
        "source_control_contract_fingerprint": source_control_contract_fingerprint,
    }
    artifact = _with_fingerprint(body, "preclaim_failure_fingerprint")
    _write_or_match(
        attempt / "preclaim-failure.json",
        artifact,
        "W1.4 regression preclaim failure",
    )
    return artifact


def _seal_abandoned_regression_attempts(layout: W14Layout) -> None:
    for attempt in _attempt_directories(layout.regression_attempts):
        if any(
            (attempt / name).exists()
            for name in (
                "regression-complete.json",
                "preclaim-failure.json",
                "preclaim-abandoned.json",
            )
        ):
            continue
        logs = []
        logs_root = attempt / "logs"
        if logs_root.is_dir():
            logs = [
                {
                    "path": path.relative_to(attempt).as_posix(),
                    "sha256": _sha256_file(path),
                }
                for path in sorted(logs_root.rglob("*"))
                if path.is_file() and not path.is_symlink()
            ]
        body = {
            "attempt_id": attempt.name,
            "automatic_production_promotion_allowed": False,
            "manifest_kind": "enoch-week1-w1.4-regression-preclaim-abandoned",
            "manifest_version": MANIFEST_VERSION,
            "preserved_logs": logs,
            "reason": "prior-regression-attempt-ended-before-completion",
            "retry_disposition": "new-attempt-allowed-while-w1.4-unconsumed",
            "seed_claim_count": 0,
        }
        _write_or_match(
            attempt / "preclaim-abandoned.json",
            _with_fingerprint(body, "preclaim_abandoned_fingerprint"),
            "W1.4 abandoned regression attempt",
        )


def _semantic_regression_failure_marker(command_id: str, output: bytes) -> str | None:
    text = output.decode("utf-8", errors="replace")
    if command_id == "full-fixtures":
        if "fixture gate failed: fixture failed or matched no test:" in text:
            return "fixture-test-verdict"
        return None
    if command_id in {"full-mechanics", "model-contracts", "strict-evaluator"}:
        if re.search(
            r"(?m)^test result: FAILED\. \d+ passed; [1-9]\d* failed;",
            text,
        ):
            return "cargo-test-failed-verdict"
        return None
    if command_id == "frozen-model-validator":
        if "parity failure " in text:
            return "model-parity-failed-verdict"
        return None
    raise W14OperatorError(f"unknown W1.4 regression command: {command_id}")


def _run_regression_command(
    command_id: str,
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    attempt: Path,
    control_manifest_fingerprint: str,
    rust_toolchain_contract_fingerprint: str,
    timeout_seconds: int = 3600,
) -> dict[str, Any]:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(environment),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=timeout_seconds,
    )
    log = attempt / "logs" / f"{command_id}.log"
    _write_w1_4_bytes(log, completed.stdout, f"{command_id} regression log")
    record = {
        "command": list(command),
        "command_id": command_id,
        "control_manifest_fingerprint": control_manifest_fingerprint,
        "exit_code": completed.returncode,
        "log_path": log.relative_to(attempt).as_posix(),
        "output_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "rust_toolchain_contract_fingerprint": (
            rust_toolchain_contract_fingerprint
        ),
    }
    if completed.returncode != 0:
        marker = _semantic_regression_failure_marker(command_id, completed.stdout)
        if marker is None:
            raise W14OperatorError(
                f"W1.4 regression command ended nonzero without a "
                f"machine-recognizable test verdict: {command_id}"
            )
        record["semantic_failure_marker"] = marker
        raise RegressionTestFailure(record)
    return record


def _expected_regression_commands(
    layout: W14Layout, workspace: Path, attempt: Path
) -> dict[str, list[str]]:
    bundle = layout.base.control_bundle
    return {
        "full-fixtures": [
            sys.executable,
            "-I",
            "-B",
            "-c",
            RECOVERY_RUNPY_BOOTSTRAP,
            str(workspace),
            str(workspace / "training/enoch_week1_fixtures.py"),
            "run",
            "--workspace",
            str(workspace),
            "--output",
            str(attempt / "fixtures"),
        ],
        "full-mechanics": [
            str(TRUSTED_CARGO),
            "test",
            "--locked",
            "-p",
            "shengji-mechanics",
            "--",
            "--test-threads=1",
        ],
        "model-contracts": [
            str(TRUSTED_CARGO),
            "test",
            "--locked",
            "-p",
            "shengji-core",
            "bot::expert::model_path_tests::",
            "--",
            "--test-threads=1",
        ],
        "strict-evaluator": [
            str(TRUSTED_CARGO),
            "test",
            "--locked",
            "-p",
            "shengji-core",
            "bot::search::tests::strict_search_rejects_a_zero_sample_prior_fallback",
            "--",
            "--exact",
            "--test-threads=1",
        ],
        "frozen-model-validator": [
            str(bundle / "bin" / "validate-expert-model"),
            str(bundle / "models" / "expert_model.onnx"),
            str(bundle / "models" / "expert_model.onnx.manifest.json"),
            str(bundle / "models" / "expert_model.onnx.golden.json"),
        ],
    }


def _regression_environment(
    protocol: Mapping[str, Any], base_environment: Mapping[str, str], attempt: Path
) -> dict[str, str]:
    del protocol, base_environment
    return _pinned_regression_environment(attempt)


def _reconstruct_sealed_preflight(
    layout: W14Layout, state: Mapping[str, Any]
) -> dict[str, str]:
    """Re-open W1.0/W1.1 evidence without building or claiming anything."""

    protocol = state["protocol"]
    ledger = _load_json(layout.base.ledger)
    enoch_week1.validate_seed_ledger(protocol, ledger)
    control_fingerprint = enoch_week1_freeze.verify_bundle(
        layout.base.protocol, layout.base.control_bundle
    )
    authority = _load_json(layout.base.authority)
    authority_fingerprint = (
        enoch_week1_preflight.validate_deterministic_search_fixture_authority(
            protocol, layout.base.control_bundle, authority
        )
    )
    base_operator._validate_authority_target(layout.base, authority)  # noqa: SLF001
    historical_fixture = base_operator._validate_fixture_gate(layout.base)  # noqa: SLF001
    preflight = _load_json(layout.base.preflight)
    preflight_fingerprint = enoch_week1_preflight.validate_preflight_artifact(
        protocol,
        ledger,
        layout.base.control_bundle,
        preflight,
        require_full_coverage=True,
    )
    declarations: dict[str, dict[str, Any]] = {}
    for namespace, label, _pairs, _workers in base_operator.SMOKE_SPECS:
        root = layout.base.smoke(label) / "declaration"
        declarations[namespace] = {
            "comparison": _load_json(root / "comparison.json"),
            "identities": _load_json(root / "identities.json"),
            "launch": _load_json(root / "launch.json"),
            "root": root,
        }
    smoke_evidence = base_operator._smoke_evidence_from_disk(  # noqa: SLF001
        protocol, layout.base, declarations
    )
    report = _load_json(layout.base.report)
    report_fingerprint = enoch_week1_preflight.validate_w1_1_baseline_worker_report(
        protocol,
        ledger,
        layout.base.control_bundle,
        preflight,
        smoke_evidence,
        authority,
        report,
    )
    if preflight_fingerprint != FULL_PREFLIGHT_FINGERPRINT:
        raise W14OperatorError("sealed full preflight reconstruction changed")
    return {
        "baseline_worker_report_fingerprint": report_fingerprint,
        "control_manifest_fingerprint": control_fingerprint,
        "deterministic_search_authority_fingerprint": authority_fingerprint,
        "fixture_report_fingerprint": historical_fixture[
            "fixture_report_fingerprint"
        ],
        "full_preflight_fingerprint": preflight_fingerprint,
    }


def _build_regression_gate(
    layout: W14Layout,
    workspace: Path,
    state: Mapping[str, Any],
    provenance: Mapping[str, Any],
    declaration: Mapping[str, Any],
    parent_verification: Mapping[str, Any],
    attempt: Path,
    records: Sequence[Mapping[str, Any]],
    fixture: Mapping[str, Any],
    environment: Mapping[str, str],
) -> dict[str, Any]:
    fixture_path = attempt / "fixtures" / "fixture-report.json"
    try:
        reopened_fixture, fixture_fingerprint, _file_sha, reopened_path = (
            enoch_week1_evidence._validate_fixture_bundle(fixture_path)  # noqa: SLF001
        )
    except enoch_week1_evidence.EvidenceError as exc:
        raise W14OperatorError(f"W1.4 fixture bundle is incomplete: {exc}") from exc
    if reopened_path != fixture_path.resolve() or dict(reopened_fixture) != dict(fixture):
        raise W14OperatorError("W1.4 fixture bundle reopened as different evidence")
    if fixture["failure_count"] != 0:
        raise W14OperatorError("W1.4 regression fixture report contains failures")
    if fixture["source_files_sha256"] != state["fixture"]["source_files_sha256"]:
        raise W14OperatorError("W1.4 regression source differs from frozen evaluator")
    if dict(parent_verification) != EXPECTED_PARENT_RESULT:
        raise W14OperatorError("W1.4 regression parent reconstruction changed")
    preflight_reconstruction = _reconstruct_sealed_preflight(layout, state)
    ledger = _load_json(layout.base.ledger)
    enoch_week1.validate_seed_ledger(state["protocol"], ledger)
    if ledger["ledger_fingerprint"] != PARENT_LEDGER:
        raise W14OperatorError("W1.4 regression gate changed the preclaim ledger")
    body = {
        "automatic_production_promotion_allowed": False,
        "campaign_declaration_fingerprint": declaration[
            "campaign_declaration_fingerprint"
        ],
        "commands": [copy.deepcopy(dict(item)) for item in records],
        "continuation_provenance_fingerprint": provenance[
            "continuation_provenance_fingerprint"
        ],
        "environment_sha256": enoch_week1.canonical_json_sha256(dict(environment)),
        "environment": dict(environment),
        "exclusive_campaign_lock_held": True,
        "fixture_report_fingerprint": fixture_fingerprint,
        "fixture_report_path": (
            attempt / "fixtures" / "fixture-report.json"
        ).relative_to(layout.root).as_posix(),
        "fixture_source_files_sha256": fixture["source_files_sha256"],
        "full_preflight_fingerprint": FULL_PREFLIGHT_FINGERPRINT,
        "manifest_kind": REGRESSION_KIND,
        "manifest_version": MANIFEST_VERSION,
        "parent_recovery_verification": copy.deepcopy(dict(parent_verification)),
        "post_regression_ledger_fingerprint": ledger["ledger_fingerprint"],
        "preclaim_ledger_fingerprint": PARENT_LEDGER,
        "preflight_control_reconstruction": preflight_reconstruction,
        "protocol_fingerprint": PROTOCOL_FINGERPRINT,
        "rust_toolchain_contract_fingerprint": declaration[
            "rust_toolchain_contract_fingerprint"
        ],
        "seed_claim_count": 0,
        "source_commit": provenance["continuation_git_commit"],
        "source_control_contract_fingerprint": declaration[
            "source_control_contract_fingerprint"
        ],
        "lock_comparison_protocol_fingerprint": declaration["stages"][0][
            "comparison"
        ]["comparison_protocol_fingerprint"],
        "status": "pass",
    }
    return _with_fingerprint(body, "combination_regression_gate_fingerprint")


def validate_regression_gate(
    gate: Mapping[str, Any],
    layout: W14Layout,
    state: Mapping[str, Any],
    provenance: Mapping[str, Any],
    declaration: Mapping[str, Any],
) -> str:
    live_toolchain = _validate_live_rust_toolchain(
        Path(__file__).resolve().parents[1], state["control"]
    )
    if live_toolchain["rust_toolchain_contract_fingerprint"] != declaration[
        "rust_toolchain_contract_fingerprint"
    ]:
        raise W14OperatorError("W1.4 live Rust toolchain changed")
    live_control = enoch_week1_freeze.verify_bundle(
        layout.base.protocol, layout.base.control_bundle
    )
    if live_control != CONTROL_MANIFEST_FINGERPRINT:
        raise W14OperatorError("W1.4 frozen control bundle changed")
    fingerprint = _validate_fingerprint(
        gate,
        "combination_regression_gate_fingerprint",
        "W1.4 regression gate",
    )
    exact = {
        "automatic_production_promotion_allowed": False,
        "campaign_declaration_fingerprint": declaration[
            "campaign_declaration_fingerprint"
        ],
        "continuation_provenance_fingerprint": provenance[
            "continuation_provenance_fingerprint"
        ],
        "fixture_source_files_sha256": state["fixture"]["source_files_sha256"],
        "full_preflight_fingerprint": FULL_PREFLIGHT_FINGERPRINT,
        "manifest_kind": REGRESSION_KIND,
        "manifest_version": MANIFEST_VERSION,
        "parent_recovery_verification": EXPECTED_PARENT_RESULT,
        "post_regression_ledger_fingerprint": PARENT_LEDGER,
        "preclaim_ledger_fingerprint": PARENT_LEDGER,
        "preflight_control_reconstruction": _reconstruct_sealed_preflight(
            layout, state
        ),
        "protocol_fingerprint": PROTOCOL_FINGERPRINT,
        "rust_toolchain_contract_fingerprint": declaration[
            "rust_toolchain_contract_fingerprint"
        ],
        "seed_claim_count": 0,
        "source_commit": provenance["continuation_git_commit"],
        "source_control_contract_fingerprint": declaration[
            "source_control_contract_fingerprint"
        ],
        "exclusive_campaign_lock_held": True,
        "lock_comparison_protocol_fingerprint": declaration["stages"][0][
            "comparison"
        ]["comparison_protocol_fingerprint"],
        "status": "pass",
    }
    for field, expected in exact.items():
        if gate.get(field) != expected:
            raise W14OperatorError(f"W1.4 regression gate changed {field}")
    fixture_relative = gate.get("fixture_report_path")
    if not isinstance(fixture_relative, str):
        raise W14OperatorError("W1.4 regression gate lacks fixture path")
    fixture_path = (layout.root / fixture_relative).resolve()
    root = layout.root.resolve()
    if root != fixture_path and root not in fixture_path.parents:
        raise W14OperatorError("W1.4 regression fixture path escapes run root")
    try:
        _fixture, fixture_fingerprint, _file_sha, reopened_path = (
            enoch_week1_evidence._validate_fixture_bundle(fixture_path)  # noqa: SLF001
        )
    except enoch_week1_evidence.EvidenceError as exc:
        raise W14OperatorError(f"W1.4 fixture bundle changed: {exc}") from exc
    if reopened_path != fixture_path or fixture_fingerprint != gate.get(
        "fixture_report_fingerprint"
    ):
        raise W14OperatorError("W1.4 regression fixture fingerprint changed")
    records = gate.get("commands")
    if not isinstance(records, list):
        raise W14OperatorError("W1.4 regression command records are malformed")
    attempt = fixture_path.parent.parent
    regression_environment = gate.get("environment")
    if not isinstance(regression_environment, Mapping):
        raise W14OperatorError("W1.4 regression environment is malformed")
    expected_environment = _pinned_regression_environment(attempt)
    if (
        dict(regression_environment) != expected_environment
        or gate.get("environment_sha256")
        != enoch_week1.canonical_json_sha256(dict(regression_environment))
    ):
        raise W14OperatorError("W1.4 regression environment identity changed")
    expected_commands = _expected_regression_commands(
        layout, Path(__file__).resolve().parents[1], attempt
    )
    expected_ids = list(expected_commands)
    if [record.get("command_id") for record in records if isinstance(record, Mapping)] != expected_ids:
        raise W14OperatorError("W1.4 regression command set is incomplete")
    for record in records:
        _require_exact_keys(
            record,
            {
                "command",
                "command_id",
                "control_manifest_fingerprint",
                "exit_code",
                "log_path",
                "output_sha256",
                "rust_toolchain_contract_fingerprint",
            },
            "W1.4 regression command record",
        )
        if record["exit_code"] != 0:
            raise W14OperatorError("W1.4 regression command did not pass")
        if record["control_manifest_fingerprint"] != CONTROL_MANIFEST_FINGERPRINT:
            raise W14OperatorError("W1.4 regression command control bundle changed")
        if record["rust_toolchain_contract_fingerprint"] != declaration[
            "rust_toolchain_contract_fingerprint"
        ]:
            raise W14OperatorError("W1.4 regression command toolchain changed")
        if record["command"] != expected_commands[record["command_id"]]:
            raise W14OperatorError("W1.4 regression command changed")
        if record["log_path"] != f"logs/{record['command_id']}.log":
            raise W14OperatorError("W1.4 regression log path changed")
        log = (attempt / record["log_path"]).resolve()
        if attempt.resolve() != log and attempt.resolve() not in log.parents:
            raise W14OperatorError("W1.4 regression log escapes its attempt")
        if _sha256_file(log) != record["output_sha256"]:
            raise W14OperatorError("W1.4 regression log hash changed")
        output = log.read_text(encoding="utf-8", errors="replace")
        markers = {
            "full-fixtures": [gate["fixture_report_fingerprint"]],
            "full-mechanics": ["test result: ok. 77 passed; 0 failed"],
            "model-contracts": [
                "model_path_override_round_trips ... ok",
                "manifest_rejects_width_drift_and_untyped_v2_outputs ... ok",
                "embedded_model_has_no_value_output ... ok",
            ],
            "strict-evaluator": [
                "strict_search_rejects_a_zero_sample_prior_fallback ... ok"
            ],
            "frozen-model-validator": ["PASS:"],
        }[record["command_id"]]
        if any(marker not in output for marker in markers):
            raise W14OperatorError(
                f"W1.4 regression log lacks required marker: {record['command_id']}"
            )
    return fingerprint


def _ensure_regression_gate_locked(
    layout: W14Layout,
    workspace: Path,
    state: Mapping[str, Any],
    provenance: Mapping[str, Any],
    declaration: Mapping[str, Any],
    parent_verification: Mapping[str, Any],
    *,
    base_environment: Mapping[str, str],
) -> dict[str, Any]:
    def require_toolchain() -> str:
        live = _validate_live_rust_toolchain(workspace, state["control"])
        fingerprint = live["rust_toolchain_contract_fingerprint"]
        if fingerprint != declaration["rust_toolchain_contract_fingerprint"]:
            raise W14OperatorError("W1.4 live Rust toolchain changed")
        return fingerprint

    def require_live_source() -> str:
        live_plan = load_committed_plan(workspace)
        return validate_continuation_provenance(
            provenance, workspace, live_plan, live_source=True
        )

    def require_control() -> str:
        fingerprint = enoch_week1_freeze.verify_bundle(
            layout.base.protocol, layout.base.control_bundle
        )
        if (
            fingerprint != CONTROL_MANIFEST_FINGERPRINT
            or state["control"].get("control_manifest_fingerprint")
            != CONTROL_MANIFEST_FINGERPRINT
        ):
            raise W14OperatorError("W1.4 frozen control bundle changed")
        return fingerprint

    if layout.regression.exists():
        gate = _load_json(layout.regression)
        validate_regression_gate(gate, layout, state, provenance, declaration)
        return gate
    ledger = _load_json(layout.base.ledger)
    if _w1_4_claims(ledger):
        raise W14OperatorError("claimed W1.4 work exists before its regression gate")
    if ledger.get("ledger_fingerprint") != PARENT_LEDGER:
        raise W14OperatorError("W1.4 regression requires the exact parent ledger")
    completed_attempts = [
        attempt
        for attempt in _attempt_directories(layout.regression_attempts)
        if (attempt / "regression-complete.json").is_file()
    ]
    if len(completed_attempts) > 1:
        raise W14OperatorError("multiple completed W1.4 regression attempts exist")
    if completed_attempts:
        gate = _load_json(completed_attempts[0] / "regression-complete.json")
        validate_regression_gate(gate, layout, state, provenance, declaration)
        _write_or_match(layout.regression, gate, "W1.4 combination regression gate")
        return gate
    for prior in _attempt_directories(layout.regression_attempts):
        failure_path = prior / "preclaim-failure.json"
        if failure_path.is_file():
            failure = _load_json(failure_path)
            if failure.get("retry_disposition") == "terminal-candidate-prerequisite-failure":
                result = _seal_candidate_prerequisite_failure(
                    layout, state, provenance, declaration, failure, ledger
                )
                raise CandidatePrerequisiteFailed(result)
    _seal_abandoned_regression_attempts(layout)
    attempt = _next_attempt(layout.regression_attempts)
    try:
        toolchain_fingerprint = require_toolchain()
        control_fingerprint = require_control()
        environment = _regression_environment(state["protocol"], base_environment, attempt)
        _write_or_match(
            attempt / "regression-environment.json",
            environment,
            "W1.4 regression environment",
        )
        records = []
        commands = _expected_regression_commands(layout, workspace, attempt)

        def run_command(command_id: str) -> dict[str, Any]:
            before = require_toolchain()
            source_before = require_live_source()
            control_before = require_control()
            try:
                record = _run_regression_command(
                    command_id,
                    commands[command_id],
                    cwd=workspace,
                    environment=environment,
                    attempt=attempt,
                    control_manifest_fingerprint=control_before,
                    rust_toolchain_contract_fingerprint=before,
                )
            except BaseException as exc:
                try:
                    after_failure = require_toolchain()
                    source_after_failure = require_live_source()
                    control_after_failure = require_control()
                except BaseException as drift_exc:
                    raise drift_exc from exc
                if after_failure != before:
                    raise W14OperatorError(
                        "W1.4 Rust toolchain changed during a failed command"
                    ) from exc
                if source_after_failure != source_before:
                    raise W14OperatorError(
                        "W1.4 source changed during a failed command"
                    ) from exc
                if control_after_failure != control_before:
                    raise W14OperatorError(
                        "W1.4 control bundle changed during a failed command"
                    ) from exc
                raise
            after = require_toolchain()
            source_after = require_live_source()
            control_after = require_control()
            if after != before:
                raise W14OperatorError("W1.4 Rust toolchain changed during a command")
            if source_after != source_before:
                raise W14OperatorError("W1.4 source changed during a command")
            if control_after != control_before:
                raise W14OperatorError("W1.4 control bundle changed during a command")
            return record

        for command_id in commands:
            records.append(run_command(command_id))
        if require_toolchain() != toolchain_fingerprint:
            raise W14OperatorError("W1.4 Rust toolchain changed during regression")
        if require_control() != control_fingerprint:
            raise W14OperatorError("W1.4 control bundle changed during regression")
        require_live_source()
        fixture = _load_json(attempt / "fixtures" / "fixture-report.json")
        gate = _build_regression_gate(
            layout,
            workspace,
            state,
            provenance,
            declaration,
            parent_verification,
            attempt,
            records,
            fixture,
            environment,
        )
        require_live_source()
        require_toolchain()
        require_control()
        _write_or_match(attempt / "regression-complete.json", gate, "regression completion")
        require_live_source()
        require_toolchain()
        require_control()
        _write_or_match(layout.regression, gate, "W1.4 combination regression gate")
        require_live_source()
        require_toolchain()
        require_control()
        validate_regression_gate(gate, layout, state, provenance, declaration)
        return gate
    except BaseException as exc:
        ledger = _load_json(layout.base.ledger)
        if not _w1_4_claims(ledger):
            failure_exc = exc
            if isinstance(exc, RegressionTestFailure):
                # A failed test is a permanent candidate verdict only when the
                # tested live source still reconstructs the declared clean
                # continuation.  Source drift is operational and retryable.
                try:
                    live_plan = load_committed_plan(workspace)
                    validate_continuation_provenance(
                        provenance, workspace, live_plan, live_source=True
                    )
                    require_toolchain()
                    require_control()
                except BaseException as provenance_exc:
                    failure_exc = provenance_exc
            failure = _write_regression_failure(
                attempt,
                failure_exc,
                lock_comparison_protocol_fingerprint=declaration["stages"][0][
                    "comparison"
                ]["comparison_protocol_fingerprint"],
                control_manifest_fingerprint=CONTROL_MANIFEST_FINGERPRINT,
                rust_toolchain_contract_fingerprint=declaration[
                    "rust_toolchain_contract_fingerprint"
                ],
                source_control_contract_fingerprint=declaration[
                    "source_control_contract_fingerprint"
                ],
            )
            if isinstance(failure_exc, RegressionTestFailure):
                result = _seal_candidate_prerequisite_failure(
                    layout, state, provenance, declaration, failure, ledger
                )
                raise CandidatePrerequisiteFailed(result) from exc
            if failure_exc is not exc:
                raise failure_exc from exc
        raise


def _ensure_regression_gate(
    layout: W14Layout,
    workspace: Path,
    state: Mapping[str, Any],
    provenance: Mapping[str, Any],
    declaration: Mapping[str, Any],
    parent_verification: Mapping[str, Any],
    *,
    base_environment: Mapping[str, str],
) -> dict[str, Any]:
    if layout.regression.exists():
        gate = _load_json(layout.regression)
        validate_regression_gate(gate, layout, state, provenance, declaration)
        return gate
    qualification = _stage_by_id(declaration, "qualification")["comparison"]
    with enoch_week1_runner.authoritative_campaign_lock(
        state["protocol"], qualification
    ) as campaign_lock_token:
        if not campaign_lock_token:
            raise W14OperatorError("regression gate lacks its machine-global lock token")
        return _ensure_regression_gate_locked(
            layout,
            workspace,
            state,
            provenance,
            declaration,
            parent_verification,
            base_environment=base_environment,
        )


def _namespace_claims(
    ledger: Mapping[str, Any], namespace: str
) -> list[Mapping[str, Any]]:
    return [row for row in ledger["consumed"] if row["namespace"] == namespace]


def _expected_claim_consumers(comparison: Mapping[str, Any]) -> dict[int, str]:
    prefix = comparison["comparison_protocol_fingerprint"][:16]
    return {
        index: f"runner:{prefix}:{shard['shard_id']}"
        for shard in comparison["shards"]
        for index in shard["seed_indices"]
    }


def _validate_claims(
    ledger: Mapping[str, Any], comparison: Mapping[str, Any]
) -> None:
    actual = {
        row["index"]: row["consumer"]
        for row in _namespace_claims(ledger, comparison["seed_namespace"])
    }
    if actual != _expected_claim_consumers(comparison):
        raise W14OperatorError(
            f"{comparison['seed_namespace']} claims do not match its completed run"
        )


def _validate_parent_prefix(
    ledger: Mapping[str, Any], parent_ledger: Mapping[str, Any]
) -> None:
    body = {key: value for key, value in ledger.items() if key != "ledger_fingerprint"}
    body["consumed"] = list(ledger["consumed"][:PRECLAIM_COUNT])
    if (
        enoch_week1.canonical_json_sha256(body) != PARENT_LEDGER
        or parent_ledger["ledger_fingerprint"] != PARENT_LEDGER
        or list(ledger["consumed"][:PRECLAIM_COUNT]) != parent_ledger["consumed"]
    ):
        raise W14OperatorError("recovered W1.3 ledger prefix changed")


def _validate_claim_frontier(
    ledger: Mapping[str, Any],
    parent_ledger: Mapping[str, Any],
    declaration: Mapping[str, Any],
    completed_stage_ids: Sequence[str],
) -> None:
    _validate_parent_prefix(ledger, parent_ledger)
    expected: dict[tuple[str, int], str] = {}
    for stage_id in completed_stage_ids:
        comparison = _stage_by_id(declaration, stage_id)["comparison"]
        for index, consumer in _expected_claim_consumers(comparison).items():
            expected[(comparison["seed_namespace"], index)] = consumer
    actual = {
        (row["namespace"], row["index"]): row["consumer"]
        for row in ledger["consumed"][PRECLAIM_COUNT:]
    }
    if actual != expected:
        raise W14OperatorError(
            "W1.4 ledger frontier has future, downstream, missing, or undeclared claims"
        )


def _stage_attempts(layout: W14Layout, stage: Mapping[str, Any]) -> list[Path]:
    return _attempt_directories(
        layout.stage(stage["sequence"], stage["stage_id"]) / "attempts"
    )


def _stage_attempt_entries_tolerant(
    layout: W14Layout, stage: Mapping[str, Any]
) -> list[Path]:
    root = layout.stage(stage["sequence"], stage["stage_id"]) / "attempts"
    if not root.exists() or root.is_symlink() or not root.is_dir():
        return []
    try:
        return sorted(root.iterdir(), key=lambda path: path.name)
    except OSError:
        return []


def _completed_attempt(
    layout: W14Layout, stage: Mapping[str, Any]
) -> Path | None:
    completed = [
        path
        for path in _stage_attempts(layout, stage)
        if (path / "execution" / "execution-complete.json").is_file()
    ]
    if len(completed) > 1:
        raise W14OperatorError(f"multiple completed {stage['stage_id']} attempts exist")
    return completed[0] if completed else None


def _retirement_body(
    protocol: Mapping[str, Any],
    ledger: Mapping[str, Any],
    stage: Mapping[str, Any],
    layout: W14Layout,
    *,
    reason: str,
) -> dict[str, Any]:
    comparison = stage["comparison"]
    claims = [
        {
            "consumer": row["consumer"],
            "index": row["index"],
            "namespace": row["namespace"],
            "seed": row["seed"],
            "sequence": row["sequence"],
        }
        for row in _namespace_claims(ledger, comparison["seed_namespace"])
    ]
    body = {
        "attempt_directories": [
            path.name for path in _stage_attempt_entries_tolerant(layout, stage)
        ],
        "automatic_production_promotion_allowed": False,
        "claims": claims,
        "disposition": FAILURE_DISPOSITION,
        "ledger_fingerprint": ledger["ledger_fingerprint"],
        "manifest_kind": RETIREMENT_KIND,
        "manifest_version": MANIFEST_VERSION,
        "phase": "W1.4",
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "reason": reason,
        "seed_namespace": comparison["seed_namespace"],
        "stage_id": stage["stage_id"],
    }
    return _with_fingerprint(body, "retirement_fingerprint")


def _retire(
    layout: W14Layout,
    protocol: Mapping[str, Any],
    ledger: Mapping[str, Any],
    stage: Mapping[str, Any],
    *,
    attempt: Path | None,
    reason: str,
) -> None:
    artifact = _retirement_body(protocol, ledger, stage, layout, reason=reason)
    _write_or_match(layout.retirement, artifact, "W1.4 protocol retirement")
    attempts_root = layout.stage(stage["sequence"], stage["stage_id"]) / "attempts"
    if attempt is not None:
        try:
            root_resolved = attempts_root.resolve(strict=True)
            run_root = layout.root.resolve(strict=True)
            attempt_resolved = attempt.resolve(strict=True)
            safe_attempt = (
                not attempts_root.is_symlink()
                and attempts_root.is_dir()
                and not attempt.is_symlink()
                and attempt.is_dir()
                and attempt.parent == attempts_root
                and attempt_resolved.parent == root_resolved
                and (run_root == root_resolved or run_root in root_resolved.parents)
            )
        except OSError:
            safe_attempt = False
        if safe_attempt:
            _write_or_match(
                attempt / "failure-tombstone.json",
                artifact,
                f"{stage['stage_id']} failure tombstone",
            )
    raise W14OperatorError(
        f"{stage['comparison']['seed_namespace']} is invalid; protocol retired"
    )


def _retire_if_claimed_incomplete(
    layout: W14Layout,
    protocol: Mapping[str, Any],
    ledger: Mapping[str, Any],
    stage: Mapping[str, Any],
) -> None:
    claims = _namespace_claims(ledger, stage["comparison"]["seed_namespace"])
    if not claims:
        return
    try:
        completed = _completed_attempt(layout, stage)
        attempts = _stage_attempts(layout, stage)
    except (OSError, W14OperatorError):
        _retire(
            layout,
            protocol,
            ledger,
            stage,
            attempt=None,
            reason="malformed-or-multiple-claimed-w1.4-attempt-entries",
        )
    if completed is None:
        _retire(
            layout,
            protocol,
            ledger,
            stage,
            attempt=attempts[-1] if attempts else None,
            reason="consumed-w1.4-namespace-lacks-one-valid-completion",
        )


def _record_stage_preclaim_failure(
    attempt: Path, stage: Mapping[str, Any], exc: BaseException
) -> None:
    body = {
        "automatic_production_promotion_allowed": False,
        "error": f"{type(exc).__name__}: {exc}",
        "manifest_kind": "enoch-week1-w1.4-preclaim-failure",
        "manifest_version": MANIFEST_VERSION,
        "retry_disposition": "new-attempt-allowed-only-while-namespace-unconsumed",
        "seed_namespace": stage["comparison"]["seed_namespace"],
        "stage_id": stage["stage_id"],
    }
    _write_or_match(
        attempt / "preclaim-failure.json",
        _with_fingerprint(body, "preclaim_failure_fingerprint"),
        f"{stage['stage_id']} preclaim failure",
    )


def _seal_abandoned_attempts(layout: W14Layout, stage: Mapping[str, Any]) -> None:
    for attempt in _stage_attempts(layout, stage):
        if (attempt / "execution" / "execution-complete.json").exists():
            continue
        if any(
            (attempt / name).exists()
            for name in ("failure-tombstone.json", "preclaim-failure.json", "preclaim-abandoned.json")
        ):
            continue
        body = {
            "attempt_id": attempt.name,
            "automatic_production_promotion_allowed": False,
            "manifest_kind": "enoch-week1-w1.4-preclaim-abandoned",
            "manifest_version": MANIFEST_VERSION,
            "reason": "prior-attempt-ended-before-claim-or-completion",
            "retry_disposition": "new-attempt-allowed-because-namespace-is-unconsumed",
            "seed_namespace": stage["comparison"]["seed_namespace"],
            "stage_id": stage["stage_id"],
        }
        _write_or_match(
            attempt / "preclaim-abandoned.json",
            _with_fingerprint(body, "preclaim_abandoned_fingerprint"),
            f"{stage['stage_id']} abandoned attempt",
        )


def _nonzero_failure_counters(metrics: Mapping[str, Any]) -> list[str]:
    counters = metrics.get("failure_counters")
    if not isinstance(counters, Mapping) or set(counters) != set(
        enoch_week1.FAILURE_COUNTER_NAMES
    ):
        raise W14OperatorError("W1.4 invalidating-counter schema changed")
    return [name for name in enoch_week1.FAILURE_COUNTER_NAMES if counters[name] != 0]


def _regression_fixture_path(layout: W14Layout, gate: Mapping[str, Any]) -> Path:
    relative = gate.get("fixture_report_path")
    if not isinstance(relative, str):
        raise W14OperatorError("regression gate has no fixture path")
    path = (layout.root / relative).resolve()
    root = layout.root.resolve()
    if root != path and root not in path.parents:
        raise W14OperatorError("regression fixture path escapes run root")
    return path


def _validated_external_fingerprint(
    protocol: Mapping[str, Any],
    comparison: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> str:
    """Validate file-backed evidence and return its scalar semantic identity."""

    try:
        enoch_week1_evidence.validate_verified_external_evidence(
            protocol, comparison, evidence
        )
    except (enoch_week1.ProtocolError, enoch_week1_evidence.EvidenceError) as exc:
        raise W14OperatorError(f"invalid W1.4 external evidence: {exc}") from exc
    return _require_sha256(
        evidence.get("verified_external_evidence_fingerprint"),
        "verified external evidence fingerprint",
    )


def _validate_completed_stage(
    layout: W14Layout,
    state: Mapping[str, Any],
    declaration: Mapping[str, Any],
    gate: Mapping[str, Any],
    stage: Mapping[str, Any],
    attempt: Path,
    *,
    base_environment: Mapping[str, str],
) -> dict[str, Any]:
    protocol = state["protocol"]
    comparison = stage["comparison"]
    launch = stage["launch_configuration"]
    identities = stage["identity_bindings"]
    execution = attempt / "execution"
    evaluator = layout.base.control_bundle / "bin" / "enoch-week1-evaluator"
    enoch_week1_runner.validate_identity_bindings(
        comparison, identities, evaluator, launch
    )
    for name, expected in {
        "comparison.json": comparison,
        "identity-bindings.json": identities,
        "launch-configuration.json": launch,
    }.items():
        if _load_json(execution / name) != expected:
            raise W14OperatorError(f"completed {stage['stage_id']} changed {name}")
    probe = _load_json(attempt / "environment-probe.json")
    try:
        w12._validate_environment_probe(probe, stage["environment_identity"])  # noqa: SLF001
    except w12.W12OperatorError as exc:
        raise W14OperatorError(str(exc)) from exc
    machine = _load_json(attempt / "machine-attestation.json")
    try:
        machine_fingerprint = enoch_week1_evidence.validate_machine_contention_attestation(
            comparison, machine
        )
    except enoch_week1_evidence.EvidenceError as exc:
        raise W14OperatorError(f"invalid W1.4 machine attestation: {exc}") from exc
    contract = declaration["execution_contract"]
    if (
        machine["worker_count"] != contract["worker_count"]
        or machine["available_parallelism"] != contract["available_parallelism"]
        or machine["machine_contention_count"] != 0
        or machine["competing_process_count"] != 0
        or machine["exclusive_campaign_lock_held"] is not True
    ):
        raise W14OperatorError(f"{stage['stage_id']} machine attestation changed")
    evidence = _load_json(attempt / "external-evidence.json")
    # The validator returns counters; this helper deliberately reads the field.
    external_fingerprint = _validated_external_fingerprint(
        protocol, comparison, evidence
    )
    if _load_json(execution / "external-failure-evidence.json") != evidence:
        raise W14OperatorError(f"completed {stage['stage_id']} changed external evidence")
    child_environment, _ = enoch_week1.sanitized_evaluator_environment(
        base_environment,
        allowlist=protocol["evaluator_environment_policy"]["allowlist"],
    )
    shards = []
    raw_hashes = []
    for assignment in comparison["shards"]:
        shard_id = assignment["shard_id"]
        shard = _load_json(execution / f"{shard_id}.result.json")
        enoch_week1.validate_shard_result(protocol, comparison, shard)
        stderr_path = execution / f"{shard_id}.stderr.txt"
        if not stderr_path.is_file() or stderr_path.stat().st_size:
            raise W14OperatorError(f"{stage['stage_id']} {shard_id} stderr is not empty")
        raw_path = execution / f"{shard_id}.raw.json"
        raw = _load_json(raw_path)
        translated = enoch_week1_runner.translate_evaluator_output(
            protocol,
            comparison,
            shard_id,
            raw,
            launch,
            identities,
            child_environment,
            evidence,
            available_parallelism=contract["available_parallelism"],
        )
        if translated != shard:
            raise W14OperatorError(f"{stage['stage_id']} {shard_id} raw output changed")
        raw_hashes.append({"sha256": _sha256_file(raw_path), "shard_id": shard_id})
        shards.append(shard)
    recomputed = enoch_week1.merge_shard_results(protocol, comparison, shards)
    merged = _load_json(execution / "merged-result.json")
    if merged != recomputed:
        raise W14OperatorError(f"{stage['stage_id']} merged result does not reconstruct")
    enoch_week1.validate_merged_result(protocol, comparison, merged)
    nonzero = _nonzero_failure_counters(merged["metrics"])
    if nonzero:
        raise W14OperatorError(
            f"{stage['stage_id']} has invalidating counters: {', '.join(nonzero)}"
        )
    runner_plan = _load_json(attempt / "runner-plan.json")
    runner_probe = runner_plan.get("environment_identity_probe")
    if not isinstance(runner_probe, Mapping):
        raise W14OperatorError("runner dry-run plan lacks an environment probe")
    completion = _load_json(execution / "execution-complete.json")
    completion_probe = completion.get("environment_identity_probe")
    if not isinstance(completion_probe, Mapping):
        raise W14OperatorError("runner completion lacks an environment probe")
    common_validation = {
        "comparison": comparison,
        "launch": launch,
        "evidence": evidence,
        "evaluator": evaluator,
        "protocol_path": layout.base.protocol,
        "workers": contract["worker_count"],
        "available_parallelism": contract["available_parallelism"],
    }
    try:
        w12._validate_runner_record(  # noqa: SLF001
            runner_plan,
            environment_probe=runner_probe,
            dry_run=True,
            **common_validation,
        )
        w12._validate_environment_probe(  # noqa: SLF001
            runner_probe, stage["environment_identity"]
        )
        w12._validate_runner_record(  # noqa: SLF001
            completion,
            environment_probe=completion_probe,
            dry_run=False,
            **common_validation,
        )
        w12._validate_environment_probe(  # noqa: SLF001
            completion_probe, stage["environment_identity"]
        )
    except w12.W12OperatorError as exc:
        raise W14OperatorError(str(exc)) from exc
    if completion.get("merged_result") != str(execution / "merged-result.json"):
        raise W14OperatorError(f"{stage['stage_id']} completion merge path changed")
    expected_shards = {
        item["shard_id"]: str(execution / f"{item['shard_id']}.result.json")
        for item in comparison["shards"]
    }
    if completion.get("shard_results") != expected_shards:
        raise W14OperatorError(f"{stage['stage_id']} completion shard paths changed")
    ledger = _load_json(layout.base.ledger)
    _validate_claims(ledger, comparison)
    validate_regression_gate(gate, layout, state, _load_json(layout.provenance), declaration)
    return {
        "attempt": attempt,
        "environment_probe": probe,
        "external_evidence": evidence,
        "external_evidence_fingerprint": external_fingerprint,
        "machine_attestation": machine,
        "machine_attestation_fingerprint": machine_fingerprint,
        "merged_result": merged,
        "raw_output_sha256s": raw_hashes,
        "runner_execution": completion,
        "shard_results": shards,
        "stage_id": stage["stage_id"],
    }


_INVALID_COMPLETION_EXCEPTIONS = (
    W14OperatorError,
    OSError,
    KeyError,
    OverflowError,
    TypeError,
    ValueError,
    enoch_week1.ProtocolError,
    enoch_week1_evidence.EvidenceError,
    enoch_week1_fixtures.FixtureError,
    enoch_week1_freeze.FreezeError,
    enoch_week1_preflight.PreflightError,
    enoch_week1_runner.RunnerError,
    base_operator.OperatorError,
    recovery.SealRecoveryError,
    w12.W12OperatorError,
)


def _validate_or_retire_completion(
    layout: W14Layout,
    state: Mapping[str, Any],
    declaration: Mapping[str, Any],
    gate: Mapping[str, Any],
    stage: Mapping[str, Any],
    attempt: Path,
    *,
    base_environment: Mapping[str, str],
) -> dict[str, Any]:
    try:
        return _validate_completed_stage(
            layout,
            state,
            declaration,
            gate,
            stage,
            attempt,
            base_environment=base_environment,
        )
    except _INVALID_COMPLETION_EXCEPTIONS:
        ledger = _load_json(layout.base.ledger)
        _retire(
            layout,
            state["protocol"],
            ledger,
            stage,
            attempt=attempt,
            reason="claimed-w1.4-completion-marker-is-invalid",
        )


def _scan_resume_frontier(
    layout: W14Layout,
    state: Mapping[str, Any],
    declaration: Mapping[str, Any],
) -> list[str]:
    protocol = state["protocol"]
    ledger = _load_json(layout.base.ledger)
    enoch_week1.validate_seed_ledger(protocol, ledger)
    completed_prefix = []
    gap_seen = False
    for stage_id in STAGE_IDS:
        stage = _stage_by_id(declaration, stage_id)
        claims = _namespace_claims(ledger, stage["comparison"]["seed_namespace"])
        try:
            marker = _completed_attempt(layout, stage)
        except (OSError, W14OperatorError):
            if claims:
                _retire(
                    layout,
                    protocol,
                    ledger,
                    stage,
                    attempt=None,
                    reason="malformed-or-multiple-claimed-w1.4-attempt-entries",
                )
            raise
        if claims and marker is None:
            _retire_if_claimed_incomplete(layout, protocol, ledger, stage)
        if marker is not None and not claims:
            raise W14OperatorError(f"{stage_id} completion exists without claims")
        if marker is not None:
            try:
                _validate_claims(ledger, stage["comparison"])
            except W14OperatorError:
                _retire(
                    layout,
                    protocol,
                    ledger,
                    stage,
                    attempt=marker,
                    reason="claimed-w1.4-completion-has-invalid-claims",
                )
            if gap_seen:
                _retire(
                    layout,
                    protocol,
                    ledger,
                    stage,
                    attempt=marker,
                    reason="claimed-w1.4-completion-is-out-of-order",
                )
            completed_prefix.append(stage_id)
        else:
            gap_seen = True
    _validate_claim_frontier(
        ledger, state["parent_ledger"], declaration, completed_prefix
    )
    return completed_prefix


def _run_stage(
    layout: W14Layout,
    workspace: Path,
    state: Mapping[str, Any],
    plan: Mapping[str, Any],
    provenance: Mapping[str, Any],
    declaration: Mapping[str, Any],
    gate: Mapping[str, Any],
    stage: Mapping[str, Any],
    *,
    operator_id: str,
    base_environment: Mapping[str, str],
) -> dict[str, Any]:
    protocol = state["protocol"]
    comparison = stage["comparison"]
    ledger = _load_json(layout.base.ledger)
    enoch_week1.validate_seed_ledger(protocol, ledger)
    try:
        completed = _completed_attempt(layout, stage)
    except (OSError, W14OperatorError):
        if _namespace_claims(ledger, comparison["seed_namespace"]):
            _retire_if_claimed_incomplete(layout, protocol, ledger, stage)
        raise
    if completed is not None:
        return _validate_or_retire_completion(
            layout,
            state,
            declaration,
            gate,
            stage,
            completed,
            base_environment=base_environment,
        )
    _retire_if_claimed_incomplete(layout, protocol, ledger, stage)
    _seal_abandoned_attempts(layout, stage)
    validate_continuation_provenance(
        provenance, workspace, plan, live_source=True
    )
    attempt = None
    try:
        with enoch_week1_runner.authoritative_campaign_lock(
            protocol, comparison
        ) as campaign_lock_token:
            ledger = _load_json(layout.base.ledger)
            _retire_if_claimed_incomplete(layout, protocol, ledger, stage)
            attempt = _next_attempt(
                layout.stage(stage["sequence"], stage["stage_id"]) / "attempts"
            )
            started = _utc_now()
            contract = declaration["execution_contract"]
            child_environment, _ = enoch_week1.sanitized_evaluator_environment(
                base_environment,
                allowlist=protocol["evaluator_environment_policy"]["allowlist"],
            )
            evaluator = layout.base.control_bundle / "bin" / "enoch-week1-evaluator"
            probe = enoch_week1_runner.probe_evaluator_environment_identity(
                evaluator=evaluator,
                protocol_path=layout.base.protocol,
                protocol=protocol,
                comparison=comparison,
                launch_configuration=stage["launch_configuration"],
                child_environment=child_environment,
                evaluator_identity=stage["identity_bindings"]["evaluator"],
                available_parallelism=contract["available_parallelism"],
                campaign_lock_token=campaign_lock_token,
                timeout_seconds=contract["timeout_seconds"],
            )
            try:
                w12._validate_environment_probe(  # noqa: SLF001
                    probe, stage["environment_identity"]
                )
            except w12.W12OperatorError as exc:
                raise W14OperatorError(str(exc)) from exc
            _atomic_write_w1_4_json(
                attempt / "environment-probe.json", probe, "W1.4 environment probe"
            )
            ended = _finish_observation(started)
            machine = enoch_week1_evidence.build_machine_contention_attestation(
                comparison,
                operator_id=operator_id,
                observation_started_utc=_utc_seconds(started),
                observation_ended_utc=_utc_seconds(ended),
                attested_at_utc=_utc_seconds(_utc_now()),
                worker_count=contract["worker_count"],
                available_parallelism=contract["available_parallelism"],
            )
            _atomic_write_w1_4_json(
                attempt / "machine-attestation.json",
                machine,
                "W1.4 machine attestation",
            )
            declaration_root = (
                layout.stage(stage["sequence"], stage["stage_id"]) / "declaration"
            )
            evidence = enoch_week1_evidence.build_verified_external_evidence(
                protocol,
                comparison,
                fixture_report_path=_regression_fixture_path(layout, gate),
                source_identity_path=(
                    layout.base.control_bundle
                    / "source"
                    / "week1-evaluator-source-files.json"
                ),
                control_manifest_path=layout.base.control_bundle / "control-manifest.json",
                runner_identities_path=declaration_root / "identities.json",
                model_contract_artifact_paths=w12._model_contract_paths(  # noqa: SLF001
                    layout.parent.parent
                ),
                machine_attestation_path=attempt / "machine-attestation.json",
            )
            _atomic_write_w1_4_json(
                attempt / "external-evidence.json",
                evidence,
                "W1.4 external evidence",
            )
            enoch_week1_evidence.validate_verified_external_evidence(
                protocol, comparison, evidence
            )
            common = {
                "protocol_path": layout.base.protocol,
                "comparison_path": declaration_root / "comparison.json",
                "launch_configuration_path": declaration_root / "launch.json",
                "identities_path": declaration_root / "identities.json",
                "evaluator": evaluator,
                "ledger_path": layout.base.ledger,
                "external_evidence_path": attempt / "external-evidence.json",
                "workers": contract["worker_count"],
                "merge": True,
                "timeout_seconds": contract["timeout_seconds"],
                "base_environment": base_environment,
                "environment_overrides": {},
                "available_parallelism": contract["available_parallelism"],
                "campaign_lock_token": campaign_lock_token,
            }
            runner_plan = enoch_week1_runner.run_comparison(
                **common, output_dir=attempt / "dry-run-unused", dry_run=True
            )
            _atomic_write_w1_4_json(
                attempt / "runner-plan.json", runner_plan, "W1.4 runner plan"
            )
            enoch_week1_runner.run_comparison(
                **common, output_dir=attempt / "execution", dry_run=False
            )
            result = _validate_or_retire_completion(
                layout,
                state,
                declaration,
                gate,
                stage,
                attempt,
                base_environment=base_environment,
            )
            validate_continuation_provenance(
                provenance, workspace, plan, live_source=True
            )
            return result
    except BaseException as exc:
        ledger = _load_json(layout.base.ledger)
        enoch_week1.validate_seed_ledger(protocol, ledger)
        if _namespace_claims(ledger, comparison["seed_namespace"]):
            if layout.retirement.exists():
                raise
            _retire_if_claimed_incomplete(layout, protocol, ledger, stage)
            raise
        if attempt is not None:
            _record_stage_preclaim_failure(attempt, stage, exc)
        raise


def _load_and_validate_declaration_state(
    layout: W14Layout,
    workspace: Path,
    state: Mapping[str, Any],
    *,
    environment: Mapping[str, str],
    live_source: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan = _load_json(layout.input)
    validate_committed_plan(plan)
    provenance = _load_json(layout.provenance)
    validate_continuation_provenance(
        provenance, workspace, plan, live_source=live_source
    )
    declaration = _load_json(layout.declaration)
    environment_identity = _load_json(
        layout.stage(1, "qualification") / "declaration" / "environment-identity.json"
    )
    validate_campaign_declaration(
        declaration,
        state["protocol"],
        plan,
        provenance,
        state,
        environment=environment if live_source else None,
        environment_identity_override=None if live_source else environment_identity,
    )
    _verify_materialized_declaration(layout, plan, provenance, declaration)
    return plan, provenance, declaration


def _expected_final_ledger(
    protocol: Mapping[str, Any],
    ledger: Mapping[str, Any],
    parent_ledger: Mapping[str, Any],
    declaration: Mapping[str, Any],
) -> str:
    enoch_week1.validate_seed_ledger(protocol, ledger)
    _validate_parent_prefix(ledger, parent_ledger)
    if len(ledger["consumed"]) != FINAL_COUNT:
        raise W14OperatorError("W1.4 final ledger must contain exactly 19,811 claims")
    for stage_id in STAGE_IDS:
        _validate_claims(ledger, _stage_by_id(declaration, stage_id)["comparison"])
    if any(
        row["namespace"].startswith("dev/combination/")
        and row["namespace"]
        not in {
            "dev/combination/qualification",
            "dev/combination/screen",
        }
        for row in ledger["consumed"]
    ):
        raise W14OperatorError("W1.4 ledger contains an undeclared combination namespace")
    _validate_claim_frontier(ledger, parent_ledger, declaration, list(STAGE_IDS))
    return ledger["ledger_fingerprint"]


def _validate_ledger_extension(
    protocol: Mapping[str, Any],
    current: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> None:
    enoch_week1.validate_seed_ledger(protocol, current)
    enoch_week1.validate_seed_ledger(protocol, snapshot)
    count = len(snapshot["consumed"])
    body = {key: value for key, value in current.items() if key != "ledger_fingerprint"}
    body["consumed"] = list(current["consumed"][:count])
    if enoch_week1.canonical_json_sha256(body) != snapshot["ledger_fingerprint"]:
        raise W14OperatorError("live ledger is not an append-only W1.4 extension")
    if any(
        not row["namespace"].startswith(("qual/", "locked/"))
        for row in current["consumed"][count:]
    ):
        raise W14OperatorError("live ledger extension contains a non-W1.5+ namespace")


def _validate_post_w1_4_live_ledger(
    protocol: Mapping[str, Any],
    current: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    exit_artifact: Mapping[str, Any],
) -> None:
    if (
        exit_artifact.get("w1_5_allowed") is True
        and exit_artifact.get("status") == "single-candidate"
    ):
        _validate_ledger_extension(protocol, current, snapshot)
        return
    enoch_week1.validate_seed_ledger(protocol, current)
    enoch_week1.validate_seed_ledger(protocol, snapshot)
    if current != snapshot:
        raise W14OperatorError("live ledger advanced after a no-candidate W1.4 exit")


def build_exit_artifact(
    protocol: Mapping[str, Any],
    declaration: Mapping[str, Any],
    regression: Mapping[str, Any],
    decision: Mapping[str, Any],
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    decision_fingerprint = enoch_week1_campaign.validate_w1_4_candidate_decision(
        protocol, decision
    )
    eligible = decision["decision"] == "eligible-for-qualification"
    body = {
        "automatic_production_promotion_allowed": False,
        "campaign_declaration_fingerprint": declaration[
            "campaign_declaration_fingerprint"
        ],
        "campaign_lineage_fingerprint": declaration[
            "campaign_lineage_fingerprint"
        ],
        "candidate_arm_ids": list(CANDIDATE_ARM_IDS),
        "candidate_decision_fingerprint": decision_fingerprint,
        "combination_regression_gate_fingerprint": regression[
            "combination_regression_gate_fingerprint"
        ],
        "evaluated_candidate_fingerprint": decision["candidate_fingerprint"],
        "final_consumed_count": FINAL_COUNT,
        "final_ledger_fingerprint": ledger["ledger_fingerprint"],
        "manifest_kind": EXIT_KIND,
        "manifest_version": MANIFEST_VERSION,
        "no_candidate_reason": None if eligible else "combination-regressed",
        "protocol_fingerprint": PROTOCOL_FINGERPRINT,
        "selected_candidate_fingerprint": (
            decision["candidate_fingerprint"] if eligible else None
        ),
        "status": "single-candidate" if eligible else "combination-regressed",
        "w1_5_allowed": eligible,
        "w1_5_dependency": (
            "requires-this-w1.4-exit-and-phase-not-bare-candidate-decision"
        ),
    }
    return _with_fingerprint(body, "w1_4_exit_fingerprint")


def validate_exit_artifact(
    artifact: Mapping[str, Any],
    protocol: Mapping[str, Any],
    declaration: Mapping[str, Any],
    regression: Mapping[str, Any],
    decision: Mapping[str, Any],
    ledger: Mapping[str, Any],
) -> str:
    expected = build_exit_artifact(
        protocol, declaration, regression, decision, ledger
    )
    if dict(artifact) != expected:
        raise W14OperatorError("W1.4 exit artifact does not reconstruct")
    return expected["w1_4_exit_fingerprint"]


def _phase_artifacts(
    provenance: Mapping[str, Any],
    declaration: Mapping[str, Any],
    regression: Mapping[str, Any],
    decision: Mapping[str, Any],
    exit_artifact: Mapping[str, Any],
    ledger: Mapping[str, Any],
    stage_evidence: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    artifacts = {
        "continuation-provenance": provenance[
            "continuation_provenance_fingerprint"
        ],
        "w1.3-final-ledger": PARENT_LEDGER,
        "w1.3-recovered-supported-change-set": PARENT_SET,
        "w1.3-seal-recovery-manifest": RECOVERY_MANIFEST,
        "w1.3-seal-recovery-provenance": RECOVERY_PROVENANCE,
        "w1.4-campaign-declaration": declaration[
            "campaign_declaration_fingerprint"
        ],
        "w1.4-campaign-lineage": declaration["campaign_lineage_fingerprint"],
        "w1.4-candidate-decision": decision[
            enoch_week1_campaign.CANDIDATE_DECISION_FINGERPRINT_FIELD
        ],
        "w1.4-combination-regression-gate": regression[
            "combination_regression_gate_fingerprint"
        ],
        "w1.4-final-ledger": ledger["ledger_fingerprint"],
        "single-candidate-or-no-survivor": exit_artifact[
            "w1_4_exit_fingerprint"
        ],
    }
    for stage_id, evidence in stage_evidence.items():
        stage = _stage_by_id(declaration, stage_id)
        artifacts[f"w1.4/{stage_id}/comparison"] = stage["comparison"][
            "comparison_protocol_fingerprint"
        ]
        artifacts[f"w1.4/{stage_id}/external-evidence"] = evidence[
            "external_evidence_fingerprint"
        ]
        artifacts[f"w1.4/{stage_id}/merged-result"] = evidence["merged_result"][
            "merged_result_fingerprint"
        ]
    return artifacts


def _build_phase4(
    protocol: Mapping[str, Any],
    phase3: Mapping[str, Any],
    provenance: Mapping[str, Any],
    declaration: Mapping[str, Any],
    regression: Mapping[str, Any],
    decision: Mapping[str, Any],
    exit_artifact: Mapping[str, Any],
    ledger: Mapping[str, Any],
    stage_evidence: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return enoch_week1.build_phase_manifest(
        protocol,
        "W1.4",
        artifacts=_phase_artifacts(
            provenance,
            declaration,
            regression,
            decision,
            exit_artifact,
            ledger,
            stage_evidence,
        ),
        declarations={
            "campaign_declaration_fingerprint": declaration[
                "campaign_declaration_fingerprint"
            ],
            "candidate_arm_ids": list(CANDIDATE_ARM_IDS),
            "candidate_decision_fingerprint": decision[
                enoch_week1_campaign.CANDIDATE_DECISION_FINGERPRINT_FIELD
            ],
            "combination_regression_gate_fingerprint": regression[
                "combination_regression_gate_fingerprint"
            ],
            "final_ledger_fingerprint": ledger["ledger_fingerprint"],
            "no_candidate_reason": exit_artifact["no_candidate_reason"],
            "selected_candidate_fingerprint": exit_artifact[
                "selected_candidate_fingerprint"
            ],
            "status": exit_artifact["status"],
            "total_pair_count": 1_100,
            "w1_3_phase_manifest_fingerprint": PARENT_PHASE,
            "w1_4_exit_fingerprint": exit_artifact["w1_4_exit_fingerprint"],
            "w1_5_dependency": exit_artifact["w1_5_dependency"],
        },
        parent_phase_manifests=[phase3],
    )


def _validate_phase4(
    phase4: Mapping[str, Any],
    state: Mapping[str, Any],
    provenance: Mapping[str, Any],
    declaration: Mapping[str, Any],
    regression: Mapping[str, Any],
    decision: Mapping[str, Any],
    exit_artifact: Mapping[str, Any],
    ledger: Mapping[str, Any],
    stage_evidence: Mapping[str, Mapping[str, Any]],
) -> str:
    expected = _build_phase4(
        state["protocol"],
        state["phase3"],
        provenance,
        declaration,
        regression,
        decision,
        exit_artifact,
        ledger,
        stage_evidence,
    )
    if dict(phase4) != expected:
        raise W14OperatorError("W1.4 phase manifest does not reconstruct")
    if not any(
        item["artifact_id"] == "single-candidate-or-no-survivor"
        and item["sha256"] == exit_artifact["w1_4_exit_fingerprint"]
        for item in phase4["artifacts"]
    ):
        raise W14OperatorError("W1.4 phase omits its declared exit artifact")
    chain = enoch_week1.validate_phase_chain(
        state["protocol"],
        [state["phase0"], state["phase1"], state["phase2"], state["phase3"], expected],
    )
    return chain[-1]


def _build_prerequisite_failure_exit(
    declaration: Mapping[str, Any],
    failure: Mapping[str, Any],
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "automatic_production_promotion_allowed": False,
        "campaign_declaration_fingerprint": declaration[
            "campaign_declaration_fingerprint"
        ],
        "campaign_lineage_fingerprint": declaration[
            "campaign_lineage_fingerprint"
        ],
        "candidate_arm_ids": list(CANDIDATE_ARM_IDS),
        "candidate_decision_fingerprint": None,
        "candidate_prerequisite_failure_fingerprint": failure[
            "preclaim_failure_fingerprint"
        ],
        "combination_regression_gate_fingerprint": None,
        "evaluated_candidate_fingerprint": declaration["campaign_lineage"][
            "candidate_fingerprint"
        ],
        "final_consumed_count": PRECLAIM_COUNT,
        "final_ledger_fingerprint": ledger["ledger_fingerprint"],
        "manifest_kind": EXIT_KIND,
        "manifest_version": MANIFEST_VERSION,
        "no_candidate_reason": "candidate-prerequisites-incomplete",
        "protocol_fingerprint": PROTOCOL_FINGERPRINT,
        "selected_candidate_fingerprint": None,
        "status": "candidate-prerequisites-incomplete",
        "w1_5_allowed": False,
        "w1_5_dependency": (
            "requires-this-w1.4-exit-and-phase-not-bare-candidate-decision"
        ),
    }
    return _with_fingerprint(body, "w1_4_exit_fingerprint")


def _build_prerequisite_failure_phase(
    state: Mapping[str, Any],
    provenance: Mapping[str, Any],
    declaration: Mapping[str, Any],
    failure: Mapping[str, Any],
    exit_artifact: Mapping[str, Any],
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    return enoch_week1.build_phase_manifest(
        state["protocol"],
        "W1.4",
        artifacts={
            "continuation-provenance": provenance[
                "continuation_provenance_fingerprint"
            ],
            "single-candidate-or-no-survivor": exit_artifact[
                "w1_4_exit_fingerprint"
            ],
            "w1.3-final-ledger": PARENT_LEDGER,
            "w1.3-recovered-supported-change-set": PARENT_SET,
            "w1.3-seal-recovery-manifest": RECOVERY_MANIFEST,
            "w1.3-seal-recovery-provenance": RECOVERY_PROVENANCE,
            "w1.4-campaign-declaration": declaration[
                "campaign_declaration_fingerprint"
            ],
            "w1.4-campaign-lineage": declaration["campaign_lineage_fingerprint"],
            "w1.4-candidate-prerequisite-failure": failure[
                "preclaim_failure_fingerprint"
            ],
            "w1.4-final-ledger": ledger["ledger_fingerprint"],
        },
        declarations={
            "campaign_declaration_fingerprint": declaration[
                "campaign_declaration_fingerprint"
            ],
            "candidate_arm_ids": list(CANDIDATE_ARM_IDS),
            "candidate_prerequisite_failure_fingerprint": failure[
                "preclaim_failure_fingerprint"
            ],
            "final_ledger_fingerprint": ledger["ledger_fingerprint"],
            "no_candidate_reason": "candidate-prerequisites-incomplete",
            "selected_candidate_fingerprint": None,
            "status": "candidate-prerequisites-incomplete",
            "total_pair_count": 0,
            "w1_3_phase_manifest_fingerprint": PARENT_PHASE,
            "w1_4_exit_fingerprint": exit_artifact["w1_4_exit_fingerprint"],
            "w1_5_dependency": exit_artifact["w1_5_dependency"],
        },
        parent_phase_manifests=[state["phase3"]],
    )


def _validate_prerequisite_failure_branch(
    layout: W14Layout,
    state: Mapping[str, Any],
    provenance: Mapping[str, Any],
    declaration: Mapping[str, Any],
) -> dict[str, Any]:
    failure = _load_json(layout.prerequisite_failure)
    failure_fingerprint = _validate_fingerprint(
        failure, "preclaim_failure_fingerprint", "W1.4 prerequisite failure"
    )
    if (
        failure.get("failure_class") != "candidate-prerequisite-failure"
        or failure.get("retry_disposition")
        != "terminal-candidate-prerequisite-failure"
        or failure.get("seed_claim_count") != 0
        or not isinstance(failure.get("failed_command"), Mapping)
    ):
        raise W14OperatorError("W1.4 prerequisite failure is not a terminal verdict")
    attempt_id = failure.get("attempt_id")
    if not isinstance(attempt_id, str) or not _ATTEMPT_RE.fullmatch(attempt_id):
        raise W14OperatorError("W1.4 prerequisite failure attempt id changed")
    attempt_failure = _load_json(
        layout.regression_attempts / attempt_id / "preclaim-failure.json"
    )
    if attempt_failure != failure:
        raise W14OperatorError("W1.4 root prerequisite failure changed its attempt")
    attempt = layout.regression_attempts / attempt_id
    failed_command = failure["failed_command"]
    _require_exact_keys(
        failed_command,
        {
            "command",
            "command_id",
            "control_manifest_fingerprint",
            "exit_code",
            "log_path",
            "output_sha256",
            "rust_toolchain_contract_fingerprint",
            "semantic_failure_marker",
        },
        "W1.4 failed prerequisite command",
    )
    command_id = failed_command.get("command_id")
    expected_commands = _expected_regression_commands(
        layout, Path(__file__).resolve().parents[1], attempt
    )
    exit_code = failed_command.get("exit_code")
    if (
        command_id not in expected_commands
        or failed_command.get("command") != expected_commands[command_id]
        or isinstance(exit_code, bool)
        or not isinstance(exit_code, int)
        or exit_code == 0
        or failed_command.get("log_path") != f"logs/{command_id}.log"
        or failed_command.get("control_manifest_fingerprint")
        != CONTROL_MANIFEST_FINGERPRINT
        or failure.get("control_manifest_fingerprint")
        != CONTROL_MANIFEST_FINGERPRINT
        or failed_command.get("rust_toolchain_contract_fingerprint")
        != declaration["rust_toolchain_contract_fingerprint"]
        or failure.get("rust_toolchain_contract_fingerprint")
        != declaration["rust_toolchain_contract_fingerprint"]
        or failure.get("source_control_contract_fingerprint")
        != declaration["source_control_contract_fingerprint"]
        or failure.get("exclusive_campaign_lock_held") is not True
        or failure.get("lock_comparison_protocol_fingerprint")
        != declaration["stages"][0]["comparison"][
            "comparison_protocol_fingerprint"
        ]
    ):
        raise W14OperatorError("W1.4 failed prerequisite command identity changed")
    if (
        enoch_week1_freeze.verify_bundle(
            layout.base.protocol, layout.base.control_bundle
        )
        != CONTROL_MANIFEST_FINGERPRINT
    ):
        raise W14OperatorError("W1.4 failed prerequisite control bundle changed")
    failed_log = attempt / failed_command["log_path"]
    if _sha256_file(failed_log) != failed_command.get("output_sha256"):
        raise W14OperatorError("W1.4 failed prerequisite log changed")
    try:
        failed_output = failed_log.read_bytes()
    except OSError as exc:
        raise W14OperatorError("could not reopen W1.4 prerequisite log") from exc
    semantic_marker = _semantic_regression_failure_marker(command_id, failed_output)
    if (
        semantic_marker is None
        or failed_command.get("semantic_failure_marker") != semantic_marker
    ):
        raise W14OperatorError(
            "W1.4 failed prerequisite lacks a semantic test verdict"
        )
    regression_environment = failure.get("regression_environment")
    if (
        not isinstance(regression_environment, Mapping)
        or dict(regression_environment) != _pinned_regression_environment(attempt)
        or failure.get("regression_environment_sha256")
        != enoch_week1.canonical_json_sha256(dict(regression_environment))
        or regression_environment.get("CARGO_TARGET_DIR")
        != str((attempt / "cargo-target").resolve())
    ):
        raise W14OperatorError("W1.4 failed prerequisite environment changed")
    ledger = _load_json(layout.final_ledger)
    enoch_week1.validate_seed_ledger(state["protocol"], ledger)
    if (
        ledger != state["parent_ledger"]
        or ledger["ledger_fingerprint"] != PARENT_LEDGER
        or len(ledger["consumed"]) != PRECLAIM_COUNT
    ):
        raise W14OperatorError("prerequisite failure branch changed the parent ledger")
    live_ledger = _load_json(layout.base.ledger)
    enoch_week1.validate_seed_ledger(state["protocol"], live_ledger)
    if live_ledger != ledger:
        raise W14OperatorError(
            "live ledger advanced after the terminal prerequisite failure"
        )
    exit_artifact = _load_json(layout.exit)
    expected_exit = _build_prerequisite_failure_exit(
        declaration, failure, ledger
    )
    if exit_artifact != expected_exit:
        raise W14OperatorError("prerequisite failure exit does not reconstruct")
    phase = _load_json(layout.phase)
    expected_phase = _build_prerequisite_failure_phase(
        state, provenance, declaration, failure, exit_artifact, ledger
    )
    if phase != expected_phase:
        raise W14OperatorError("prerequisite failure phase does not reconstruct")
    chain = enoch_week1.validate_phase_chain(
        state["protocol"],
        [state["phase0"], state["phase1"], state["phase2"], state["phase3"], phase],
    )
    return {
        "campaign_declaration_fingerprint": declaration[
            "campaign_declaration_fingerprint"
        ],
        "candidate_decision_fingerprint": None,
        "candidate_prerequisite_failure_fingerprint": failure_fingerprint,
        "combination_regression_gate_fingerprint": None,
        "final_ledger_fingerprint": PARENT_LEDGER,
        "phase_manifest_fingerprint": chain[-1],
        "selected_candidate_fingerprint": None,
        "status": "candidate-prerequisites-incomplete",
        "total_pair_count": 0,
        "w1_4_exit_fingerprint": exit_artifact["w1_4_exit_fingerprint"],
    }


def _seal_candidate_prerequisite_failure(
    layout: W14Layout,
    state: Mapping[str, Any],
    provenance: Mapping[str, Any],
    declaration: Mapping[str, Any],
    failure: Mapping[str, Any],
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    if ledger != state["parent_ledger"] or ledger["ledger_fingerprint"] != PARENT_LEDGER:
        raise W14OperatorError("cannot seal prerequisite failure after a W1.4 claim")
    _write_or_match(
        layout.prerequisite_failure,
        failure,
        "W1.4 candidate prerequisite failure",
    )
    _write_or_match(layout.final_ledger, ledger, "W1.4 prerequisite final ledger")
    exit_artifact = _build_prerequisite_failure_exit(declaration, failure, ledger)
    _write_or_match(
        layout.exit,
        exit_artifact,
        "W1.4 prerequisite single-candidate-or-no-survivor exit",
    )
    phase = _build_prerequisite_failure_phase(
        state, provenance, declaration, failure, exit_artifact, ledger
    )
    # The phase remains the final successful write on this terminal branch.
    _write_or_match(layout.phase, phase, "W1.4 prerequisite failure phase")
    return _validate_prerequisite_failure_branch(
        layout, state, provenance, declaration
    )


def _seal_w1_4(
    layout: W14Layout,
    state: Mapping[str, Any],
    provenance: Mapping[str, Any],
    declaration: Mapping[str, Any],
    regression: Mapping[str, Any],
    stage_evidence: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    live_ledger = _load_json(layout.base.ledger)
    if layout.final_ledger.exists():
        ledger = _load_json(layout.final_ledger)
        _expected_final_ledger(
            state["protocol"], ledger, state["parent_ledger"], declaration
        )
        if not layout.phase.exists() and live_ledger != ledger:
            raise W14OperatorError(
                "live ledger advanced before the unsealed W1.4 phase completed"
            )
    else:
        _expected_final_ledger(
            state["protocol"], live_ledger, state["parent_ledger"], declaration
        )
        ledger = live_ledger
        _atomic_write_w1_4_json(
            layout.final_ledger, ledger, "W1.4 final ledger"
        )
    lineage = declaration["campaign_lineage"]
    expected_decision = enoch_week1_campaign.build_w1_4_candidate_decision(
        state["protocol"],
        lineage,
        qualification_merged_result=stage_evidence["qualification"]["merged_result"],
        screen_merged_result=stage_evidence["screen"]["merged_result"],
    )
    _write_or_match(layout.decision, expected_decision, "W1.4 candidate decision")
    decision = _load_json(layout.decision)
    if decision != expected_decision:
        raise W14OperatorError("stored W1.4 candidate decision changed")
    enoch_week1_campaign.validate_w1_4_candidate_decision(
        state["protocol"], decision
    )
    expected_exit = build_exit_artifact(
        state["protocol"], declaration, regression, decision, ledger
    )
    _write_or_match(
        layout.exit, expected_exit, "W1.4 single-candidate-or-no-survivor exit"
    )
    exit_artifact = _load_json(layout.exit)
    validate_exit_artifact(
        exit_artifact,
        state["protocol"],
        declaration,
        regression,
        decision,
        ledger,
    )
    _validate_post_w1_4_live_ledger(
        state["protocol"], live_ledger, ledger, exit_artifact
    )
    phase4 = _build_phase4(
        state["protocol"],
        state["phase3"],
        provenance,
        declaration,
        regression,
        decision,
        exit_artifact,
        ledger,
        stage_evidence,
    )
    # This is deliberately the final write in the successful W1.4 seal.
    _write_or_match(layout.phase, phase4, "W1.4 phase manifest")
    phase_fingerprint = _validate_phase4(
        _load_json(layout.phase),
        state,
        provenance,
        declaration,
        regression,
        decision,
        exit_artifact,
        ledger,
        stage_evidence,
    )
    return {
        "campaign_declaration_fingerprint": declaration[
            "campaign_declaration_fingerprint"
        ],
        "candidate_decision_fingerprint": decision[
            enoch_week1_campaign.CANDIDATE_DECISION_FINGERPRINT_FIELD
        ],
        "combination_regression_gate_fingerprint": regression[
            "combination_regression_gate_fingerprint"
        ],
        "final_ledger_fingerprint": ledger["ledger_fingerprint"],
        "phase_manifest_fingerprint": phase_fingerprint,
        "selected_candidate_fingerprint": exit_artifact[
            "selected_candidate_fingerprint"
        ],
        "status": exit_artifact["status"],
        "total_pair_count": 1_100,
        "w1_4_exit_fingerprint": exit_artifact["w1_4_exit_fingerprint"],
    }


def run_w1_4(
    layout: W14Layout,
    workspace: Path,
    w1_2_workspace: Path,
    *,
    operator_id: str,
    attest_no_machine_contention: bool,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run or safely resume both predeclared W1.4 comparisons."""

    if attest_no_machine_contention is not True:
        raise W14OperatorError(
            "--attest-no-machine-contention is required for W1.4 execution"
        )
    workspace = workspace.expanduser().resolve()
    base_environment = dict(os.environ if environment is None else environment)
    parent_verification = verify_recovered_w1_3(
        layout, workspace, w1_2_workspace
    )
    _require_safe_w1_4_tree(layout)
    with base_operator._operator_lock(layout.base.operator_lock):  # noqa: SLF001
        state = _load_parent_state(layout, workspace, environment=base_environment)
        plan, provenance, declaration = _load_and_validate_declaration_state(
            layout,
            workspace,
            state,
            environment=base_environment,
            live_source=True,
        )
        if layout.retirement.exists():
            raise W14OperatorError("this protocol is retired for W1.4")
        current = _load_json(layout.base.ledger)
        claimed = _w1_4_claims(current)
        try:
            regression = _ensure_regression_gate(
                layout,
                workspace,
                state,
                provenance,
                declaration,
                parent_verification,
                base_environment=base_environment,
            )
        except CandidatePrerequisiteFailed as exc:
            return exc.result
        except _INVALID_COMPLETION_EXCEPTIONS:
            if claimed:
                stage = next(
                    _stage_by_id(declaration, stage_id)
                    for stage_id in STAGE_IDS
                    if _namespace_claims(
                        current,
                        _stage_by_id(declaration, stage_id)["comparison"][
                            "seed_namespace"
                        ],
                    )
                )
                _retire(
                    layout,
                    state["protocol"],
                    current,
                    stage,
                    # Retirement inventory is deliberately tolerant; a strict
                    # attempt lookup can itself be the invalid state.
                    attempt=None,
                    reason="claimed-w1.4-work-lacks-valid-regression-gate",
                )
            raise
        completed_prefix = _scan_resume_frontier(layout, state, declaration)
        claim_frontier = list(completed_prefix)
        stage_evidence: dict[str, Mapping[str, Any]] = {}
        for stage_id in STAGE_IDS:
            stage = _stage_by_id(declaration, stage_id)
            stage_evidence[stage_id] = _run_stage(
                layout,
                workspace,
                state,
                plan,
                provenance,
                declaration,
                regression,
                stage,
                operator_id=operator_id,
                base_environment=base_environment,
            )
            if stage_id not in claim_frontier:
                claim_frontier.append(stage_id)
            current = _load_json(layout.base.ledger)
            _validate_claim_frontier(
                current, state["parent_ledger"], declaration, claim_frontier
            )
        validate_continuation_provenance(
            provenance, workspace, plan, live_source=True
        )
        return _seal_w1_4(
            layout,
            state,
            provenance,
            declaration,
            regression,
            stage_evidence,
        )


def verify_w1_4(
    layout: W14Layout,
    workspace: Path,
    w1_2_workspace: Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Reconstruct W1.4 from disk without executing or claiming."""

    workspace = workspace.expanduser().resolve()
    base_environment = dict(os.environ if environment is None else environment)
    verify_recovered_w1_3(layout, workspace, w1_2_workspace)
    _require_safe_w1_4_tree(layout)
    with base_operator._operator_lock(layout.base.operator_lock):  # noqa: SLF001
        state = _load_parent_state(layout, workspace, environment=base_environment)
        _plan, provenance, declaration = _load_and_validate_declaration_state(
            layout,
            workspace,
            state,
            environment=base_environment,
            live_source=False,
        )
        if layout.retirement.exists():
            raise W14OperatorError("this protocol is retired for W1.4")
        if layout.prerequisite_failure.exists():
            return _validate_prerequisite_failure_branch(
                layout, state, provenance, declaration
            )
        regression = _load_json(layout.regression)
        validate_regression_gate(regression, layout, state, provenance, declaration)
        ledger = _load_json(layout.final_ledger)
        _expected_final_ledger(
            state["protocol"], ledger, state["parent_ledger"], declaration
        )
        live_ledger = _load_json(layout.base.ledger)
        stage_evidence = {}
        for stage_id in STAGE_IDS:
            stage = _stage_by_id(declaration, stage_id)
            attempt = _completed_attempt(layout, stage)
            if attempt is None:
                raise W14OperatorError(f"missing completed W1.4 {stage_id} stage")
            stage_evidence[stage_id] = _validate_completed_stage(
                layout,
                state,
                declaration,
                regression,
                stage,
                attempt,
                base_environment=base_environment,
            )
        decision = _load_json(layout.decision)
        expected_decision = enoch_week1_campaign.build_w1_4_candidate_decision(
            state["protocol"],
            declaration["campaign_lineage"],
            qualification_merged_result=stage_evidence["qualification"]["merged_result"],
            screen_merged_result=stage_evidence["screen"]["merged_result"],
        )
        if decision != expected_decision:
            raise W14OperatorError("W1.4 candidate decision changed")
        exit_artifact = _load_json(layout.exit)
        validate_exit_artifact(
            exit_artifact,
            state["protocol"],
            declaration,
            regression,
            decision,
            ledger,
        )
        _validate_post_w1_4_live_ledger(
            state["protocol"], live_ledger, ledger, exit_artifact
        )
        phase_fingerprint = _validate_phase4(
            _load_json(layout.phase),
            state,
            provenance,
            declaration,
            regression,
            decision,
            exit_artifact,
            ledger,
            stage_evidence,
        )
        return {
            "campaign_declaration_fingerprint": declaration[
                "campaign_declaration_fingerprint"
            ],
            "candidate_decision_fingerprint": decision[
                enoch_week1_campaign.CANDIDATE_DECISION_FINGERPRINT_FIELD
            ],
            "combination_regression_gate_fingerprint": regression[
                "combination_regression_gate_fingerprint"
            ],
            "final_ledger_fingerprint": ledger["ledger_fingerprint"],
            "phase_manifest_fingerprint": phase_fingerprint,
            "selected_candidate_fingerprint": exit_artifact[
                "selected_candidate_fingerprint"
            ],
            "status": exit_artifact["status"],
            "w1_4_exit_fingerprint": exit_artifact["w1_4_exit_fingerprint"],
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("declare-w1.4", "freeze both W1.4 comparisons before claims"),
        ("run-w1.4", "run or safely resume authoritative W1.4"),
        ("verify-w1.4", "offline-verify completed W1.4"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--root", required=True, type=Path)
        command.add_argument("--workspace", type=Path, default=Path.cwd())
        command.add_argument("--w1-2-workspace", required=True, type=Path)
        if name == "run-w1.4":
            command.add_argument("--operator-id", required=True)
            command.add_argument(
                "--attest-no-machine-contention", action="store_true"
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    layout = W14Layout(args.root.expanduser().resolve())
    try:
        if args.command == "declare-w1.4":
            result = declare_w1_4(layout, args.workspace, args.w1_2_workspace)
        elif args.command == "run-w1.4":
            result = run_w1_4(
                layout,
                args.workspace,
                args.w1_2_workspace,
                operator_id=args.operator_id,
                attest_no_machine_contention=args.attest_no_machine_contention,
            )
        else:
            result = verify_w1_4(layout, args.workspace, args.w1_2_workspace)
    except (
        W14OperatorError,
        recovery.SealRecoveryError,
        w13.W13OperatorError,
        w12.W12OperatorError,
        base_operator.OperatorError,
        enoch_week1.ProtocolError,
        enoch_week1_evidence.EvidenceError,
        enoch_week1_fixtures.FixtureError,
        enoch_week1_freeze.FreezeError,
        enoch_week1_preflight.PreflightError,
        enoch_week1_runner.RunnerError,
        FileExistsError,
        OSError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"W1.4 operator failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
