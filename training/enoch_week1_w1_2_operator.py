#!/usr/bin/env python3
"""Authoritative W1.2 independent-ablation operator.

This orchestration-only continuation leaves every d048 scientific module and
the frozen evaluator byte-identical.  It freezes all fifteen arm comparisons
before the first development seed claim, runs one comparison at a time under
the opaque machine-global campaign lock, and seals a typed ranked table.

Any claimed-but-incomplete arm retires the protocol.  A preclaim failure may be
retried in a new immutable attempt directory.  Offline verification never
launches the evaluator.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

try:
    from training import enoch_week1
    from training import enoch_week1_evidence
    from training import enoch_week1_fixtures
    from training import enoch_week1_freeze
    from training import enoch_week1_operator as base_operator
    from training import enoch_week1_preflight
    from training import enoch_week1_runner
except ImportError:  # pragma: no cover - direct script execution.
    import enoch_week1  # type: ignore[no-redef]
    import enoch_week1_evidence  # type: ignore[no-redef]
    import enoch_week1_fixtures  # type: ignore[no-redef]
    import enoch_week1_freeze  # type: ignore[no-redef]
    import enoch_week1_operator as base_operator  # type: ignore[no-redef]
    import enoch_week1_preflight  # type: ignore[no-redef]
    import enoch_week1_runner  # type: ignore[no-redef]


MANIFEST_VERSION = 1
OPERATOR_RELATIVE = Path("training/enoch_week1_w1_2_operator.py")
PLAN_RELATIVE = Path("training/enoch_week1_w1_2_plan.json")
TEST_RELATIVE = Path("training/test_enoch_week1_w1_2_operator.py")
CONTINUATION_PATHS = (
    OPERATOR_RELATIVE.as_posix(),
    PLAN_RELATIVE.as_posix(),
    TEST_RELATIVE.as_posix(),
)
CRITICAL_BASE_MODULES = (
    "training/enoch_week1.py",
    "training/enoch_week1_evidence.py",
    "training/enoch_week1_fixtures.py",
    "training/enoch_week1_freeze.py",
    "training/enoch_week1_preflight.py",
    "training/enoch_week1_runner.py",
    "training/enoch_week1_campaign.py",
    "training/enoch_week1_operator.py",
)
PLAN_KIND = "enoch-week1-w1.2-committed-plan"
PROVENANCE_KIND = "enoch-week1-w1.2-continuation-provenance"
DECLARATION_KIND = "enoch-week1-w1.2-campaign-declaration"
TABLE_KIND = "enoch-week1-ranked-independent-ablation-table"
RETIREMENT_KIND = "enoch-week1-w1.2-protocol-retirement"
RANKING_RULE = (
    "advancement-decision-descending",
    "level-utility-lower-95-descending",
    "level-utility-estimate-descending",
    "point-margin-estimate-descending",
    "win-rate-estimate-descending",
    "canonical-arm-ordinal-ascending",
)
FAILURE_DISPOSITION = "retire-protocol-on-any-claimed-incomplete-comparison"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_OBJECT_RE = re.compile(r"[0-9a-f]{40,64}")


class W12OperatorError(RuntimeError):
    """Raised when W1.2 cannot proceed without ambiguity."""


@dataclass(frozen=True)
class W12Layout:
    root: Path

    @property
    def base(self) -> base_operator.RunLayout:
        return base_operator.RunLayout(self.root)

    @property
    def directory(self) -> Path:
        return self.root / "w1.2"

    @property
    def provenance(self) -> Path:
        return self.directory / "continuation-provenance.json"

    @property
    def input(self) -> Path:
        return self.directory / "input.json"

    @property
    def declaration(self) -> Path:
        return self.directory / "declaration-index.json"

    @property
    def ranked_table(self) -> Path:
        return self.directory / "ranked-independent-ablation-table.json"

    @property
    def final_ledger(self) -> Path:
        return self.directory / "final-ledger.json"

    @property
    def phase(self) -> Path:
        return self.directory / "phase-manifest.json"

    @property
    def retirement(self) -> Path:
        return self.directory / "protocol-retired.json"

    def arm(self, ordinal: int, arm_id: str) -> Path:
        return self.directory / "arms" / f"{ordinal:02d}-{arm_id}"


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping):
        raise W12OperatorError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise W12OperatorError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise W12OperatorError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise W12OperatorError(f"{label} must be a positive integer")
    return value


def _require_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise W12OperatorError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise W12OperatorError(f"{label} must be finite")
    return number


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while block := source.read(1024 * 1024):
                digest.update(block)
    except OSError as exc:
        raise W12OperatorError(f"could not hash {path}: {exc}") from exc
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return enoch_week1.load_json_object(path)
    except (OSError, enoch_week1.ProtocolError) as exc:
        raise W12OperatorError(f"could not load {path}: {exc}") from exc


def _with_fingerprint(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    frozen = dict(body)
    return {**frozen, field: enoch_week1.canonical_json_sha256(frozen)}


def _write_or_match(path: Path, value: Mapping[str, Any], label: str) -> None:
    try:
        base_operator._write_or_match(path, value, label)  # noqa: SLF001
    except base_operator.OperatorError as exc:
        raise W12OperatorError(str(exc)) from exc


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


def validate_committed_plan(plan: Mapping[str, Any]) -> str:
    """Validate the result-independent plan committed before W1.2 starts."""

    _require_exact_keys(
        plan,
        {
            "arms",
            "automatic_production_promotion_allowed",
            "available_parallelism",
            "baseline_worker_report_fingerprint",
            "failure_disposition",
            "fixed_work",
            "manifest_kind",
            "manifest_version",
            "pair_count_per_arm",
            "parent_phase_manifest_fingerprint",
            "preclaim_consumed_count",
            "preclaim_ledger_fingerprint",
            "protocol_fingerprint",
            "ranking_rule",
            "required_style_metrics",
            "shard_count",
            "shard_timeout_seconds",
            "worker_count",
        },
        "committed W1.2 plan",
    )
    if (
        plan["manifest_kind"] != PLAN_KIND
        or plan["manifest_version"] != MANIFEST_VERSION
        or plan["automatic_production_promotion_allowed"] is not False
    ):
        raise W12OperatorError("unsupported committed W1.2 plan")
    for field in (
        "baseline_worker_report_fingerprint",
        "parent_phase_manifest_fingerprint",
        "preclaim_ledger_fingerprint",
        "protocol_fingerprint",
    ):
        _require_sha256(plan[field], f"plan {field}")
    if plan["pair_count_per_arm"] != 300:
        raise W12OperatorError("the committed W1.2 plan must use all 300 arm seeds")
    if plan["shard_count"] != 8 or plan["worker_count"] != 8:
        raise W12OperatorError("the committed W1.2 plan must retain the W1.1 eight-worker ceiling")
    if plan["available_parallelism"] != 10:
        raise W12OperatorError("the committed W1.2 plan must retain parallelism 10")
    _require_positive_int(plan["shard_timeout_seconds"], "plan shard timeout")
    if plan["preclaim_consumed_count"] != 10_211:
        raise W12OperatorError("the W1.2 preclaim ledger must be the completed W1.1 ledger")
    if plan["failure_disposition"] != FAILURE_DISPOSITION:
        raise W12OperatorError("the W1.2 partial-failure disposition changed")
    if tuple(plan["ranking_rule"]) != RANKING_RULE:
        raise W12OperatorError("the W1.2 ranking rule changed")
    if tuple(plan["required_style_metrics"]) != enoch_week1.WEEK1_STYLE_METRICS:
        raise W12OperatorError("the W1.2 plan must retain the complete style schema")

    fixed = plan["fixed_work"]
    _require_exact_keys(
        fixed,
        {
            "budget_ms",
            "candidates",
            "deadline_ms",
            "rollout_tricks",
            "work_mode",
            "worlds",
        },
        "W1.2 fixed-work plan",
    )
    if fixed["work_mode"] != "fixed-work" or fixed["budget_ms"] is not None:
        raise W12OperatorError("W1.2 plan must use deterministic fixed work")
    for field in ("worlds", "candidates", "rollout_tricks", "deadline_ms"):
        _require_positive_int(fixed[field], f"fixed-work {field}")

    arms = plan["arms"]
    if not isinstance(arms, list) or len(arms) != len(enoch_week1.ABLATION_ARMS):
        raise W12OperatorError("W1.2 plan must declare exactly fifteen arms")
    expected_arm_ids = list(enoch_week1.ABLATION_ARMS)
    actual_arm_ids: list[str] = []
    for arm in arms:
        _require_exact_keys(
            arm,
            {"arm_id", "comparison_id", "development_rule"},
            "W1.2 plan arm",
        )
        arm_id = arm["arm_id"]
        actual_arm_ids.append(arm_id)
        if arm["comparison_id"] != f"ablation-{arm_id}":
            raise W12OperatorError(f"{arm_id} comparison id is not canonical")
        try:
            enoch_week1.validate_development_rule(arm["development_rule"])
        except enoch_week1.ProtocolError as exc:
            raise W12OperatorError(f"invalid {arm_id} development rule: {exc}") from exc
        if arm["development_rule"]["rule_id"] != f"w1.2-{arm_id}-screen-v1":
            raise W12OperatorError(f"{arm_id} development rule id changed")
    if actual_arm_ids != expected_arm_ids:
        raise W12OperatorError("W1.2 plan arms are missing, duplicated, or reordered")
    return enoch_week1.canonical_json_sha256(plan)


def load_committed_plan(workspace: Path) -> dict[str, Any]:
    plan = _load_json(workspace.resolve() / PLAN_RELATIVE)
    validate_committed_plan(plan)
    return plan


def _git_text(workspace: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(workspace), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise W12OperatorError(
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
        raise W12OperatorError(
            f"git {' '.join(arguments)} failed: "
            f"{completed.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return completed.stdout


def _git_blob(workspace: Path, revision: str, path: str) -> str:
    value = _git_text(workspace, "rev-parse", f"{revision}:{path}").strip()
    if not _GIT_OBJECT_RE.fullmatch(value):
        raise W12OperatorError(f"Git blob identity is invalid for {revision}:{path}")
    return value


def _continuation_git_identity(
    workspace: Path, base_commit: str
) -> dict[str, Any]:
    workspace = workspace.resolve()
    top = Path(_git_text(workspace, "rev-parse", "--show-toplevel").strip()).resolve()
    if top != workspace:
        raise W12OperatorError("W1.2 workspace must be the Git root")
    status = _git_text(
        workspace, "status", "--porcelain=v1", "--untracked-files=all"
    )
    if status:
        raise W12OperatorError(
            f"W1.2 workspace is not clean: {status.splitlines()[0]}"
        )
    head = _git_text(workspace, "rev-parse", "HEAD^{commit}").strip()
    tree = _git_text(workspace, "rev-parse", "HEAD^{tree}").strip()
    parents = _git_text(workspace, "rev-list", "--parents", "-n", "1", head).split()
    if parents != [head, base_commit]:
        raise W12OperatorError(
            "W1.2 continuation must be one clean single-parent commit directly after d048"
        )
    raw_changes = _git_text(
        workspace,
        "diff-tree",
        "--no-commit-id",
        "--name-status",
        "-r",
        "--no-renames",
        base_commit,
        head,
    )
    parsed: list[tuple[str, str]] = []
    for line in raw_changes.splitlines():
        fields = line.split("\t")
        if len(fields) != 2:
            raise W12OperatorError("W1.2 continuation Git diff is malformed")
        parsed.append((fields[0], fields[1]))
    if parsed != [("A", path) for path in CONTINUATION_PATHS]:
        raise W12OperatorError(
            "W1.2 continuation must add exactly the committed operator, plan, and tests"
        )
    change_records = []
    for status_code, relative in parsed:
        mode_record = _git_text(workspace, "ls-tree", head, "--", relative).strip()
        fields = mode_record.split(None, 3)
        if len(fields) != 4 or fields[0] != "100644" or fields[1] != "blob":
            raise W12OperatorError(f"W1.2 continuation path is not a regular blob: {relative}")
        path = workspace / relative
        if not path.is_file() or path.is_symlink():
            raise W12OperatorError(f"W1.2 continuation path is not a regular file: {relative}")
        change_records.append(
            {
                "new_blob": fields[2],
                "old_blob": None,
                "path": relative,
                "sha256": _sha256_file(path),
                "status": status_code,
            }
        )
    critical_blobs = []
    for relative in CRITICAL_BASE_MODULES:
        base_blob = _git_blob(workspace, base_commit, relative)
        current_blob = _git_blob(workspace, head, relative)
        if current_blob != base_blob:
            raise W12OperatorError(f"frozen d048 module changed: {relative}")
        critical_blobs.append({"blob": base_blob, "path": relative})
    return {
        "base_tree_manifest_sha256": hashlib.sha256(
            _git_bytes(workspace, "ls-tree", "-r", "-z", "--full-tree", base_commit)
        ).hexdigest(),
        "changed_paths": change_records,
        "critical_base_module_blobs": critical_blobs,
        "git_commit": head,
        "git_tree": tree,
        "git_tree_manifest_sha256": hashlib.sha256(
            _git_bytes(workspace, "ls-tree", "-r", "-z", "--full-tree", head)
        ).hexdigest(),
    }


def _development_claims(ledger: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        row
        for row in ledger["consumed"]
        if row["namespace"].startswith("dev/ablation/")
    ]


def _fixture_binding(fixture: Mapping[str, Any]) -> dict[str, Any]:
    try:
        fingerprint = enoch_week1_fixtures.validate_report(fixture)
    except enoch_week1_fixtures.FixtureError as exc:
        raise W12OperatorError(f"invalid fixture report: {exc}") from exc
    if fixture["failure_count"] != 0:
        raise W12OperatorError("W1.2 requires a zero-failure fixture gate")
    return {
        "failure_count": 0,
        "fixture_report_fingerprint": fingerprint,
        "record_count": len(fixture["records"]),
        "source_files_sha256": fixture["source_files_sha256"],
    }


def build_continuation_provenance(
    layout: W12Layout,
    workspace: Path,
    plan: Mapping[str, Any],
    *,
    control: Mapping[str, Any],
    report: Mapping[str, Any],
    phase1: Mapping[str, Any],
    fixture: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the only three-file post-freeze continuation accepted by W1.2."""

    validate_committed_plan(plan)
    protocol = _load_json(layout.base.protocol)
    provenance = _load_json(layout.base.provenance)
    base_operator._validate_provenance(protocol, control, provenance)  # noqa: SLF001
    if phase1["phase_manifest_fingerprint"] != plan[
        "parent_phase_manifest_fingerprint"
    ]:
        raise W12OperatorError("W1.2 plan names a different W1.1 parent")
    if report["baseline_worker_report_fingerprint"] != plan[
        "baseline_worker_report_fingerprint"
    ]:
        raise W12OperatorError("W1.2 plan names a different baseline worker report")
    if protocol["protocol_fingerprint"] != plan["protocol_fingerprint"]:
        raise W12OperatorError("W1.2 plan names a different protocol")
    if Path(__file__).resolve() != (workspace / OPERATOR_RELATIVE).resolve():
        raise W12OperatorError("executing W1.2 operator is not the committed workspace file")
    actual_base_tree = _git_text(
        workspace, "rev-parse", f"{provenance['git_commit']}^{{tree}}"
    ).strip()
    if actual_base_tree != provenance["git_tree"]:
        raise W12OperatorError("frozen base commit does not reconstruct its recorded tree")
    git_identity = _continuation_git_identity(workspace, provenance["git_commit"])
    if _sha256_file(workspace / "training/enoch_week1_operator.py") != provenance[
        "operator_source_sha256"
    ]:
        raise W12OperatorError("the d048 operator source changed")
    frozen_sources = json.loads(
        (
            layout.base.control_bundle
            / "source"
            / "week1-evaluator-source-files.json"
        ).read_text(encoding="utf-8")
    )
    runtime_modules = (
        (enoch_week1, "training/enoch_week1.py"),
        (enoch_week1_evidence, "training/enoch_week1_evidence.py"),
        (enoch_week1_fixtures, "training/enoch_week1_fixtures.py"),
        (enoch_week1_freeze, "training/enoch_week1_freeze.py"),
        (base_operator, "training/enoch_week1_operator.py"),
        (enoch_week1_preflight, "training/enoch_week1_preflight.py"),
        (enoch_week1_runner, "training/enoch_week1_runner.py"),
    )
    runtime_imports = []
    for module, relative in runtime_modules:
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str) or Path(module_file).resolve() != (
            workspace / relative
        ).resolve():
            raise W12OperatorError(f"runtime import is shadowed: {relative}")
        runtime_imports.append(
            {
                "path": relative,
                "sha256": _sha256_file(workspace / relative),
            }
        )
    body = {
        "automatic_production_promotion_allowed": False,
        "base_git_commit": provenance["git_commit"],
        "base_git_tree": provenance["git_tree"],
        "base_operator_source_sha256": provenance["operator_source_sha256"],
        "base_source_provenance_fingerprint": provenance[
            "source_provenance_fingerprint"
        ],
        "baseline_worker_report_fingerprint": report[
            "baseline_worker_report_fingerprint"
        ],
        "changed_paths": git_identity["changed_paths"],
        "continuation_git_commit": git_identity["git_commit"],
        "continuation_git_tree": git_identity["git_tree"],
        "control_manifest_fingerprint": control["control_manifest_fingerprint"],
        "critical_base_module_blobs": git_identity["critical_base_module_blobs"],
        "evaluator_binary_sha256": control["evaluator_identity"]["binary_sha256"],
        "evaluator_source_records_sha256": enoch_week1.canonical_json_sha256(
            frozen_sources
        ),
        "evaluator_source_sha256": control["evaluator_identity"]["source_sha256"],
        "fixture_report_fingerprint": fixture["fixture_report_fingerprint"],
        "fixture_source_files_sha256": fixture["source_files_sha256"],
        "git_tree_manifest_sha256": git_identity["git_tree_manifest_sha256"],
        "base_tree_manifest_sha256": git_identity["base_tree_manifest_sha256"],
        "manifest_kind": PROVENANCE_KIND,
        "manifest_version": MANIFEST_VERSION,
        "operator_source_path": OPERATOR_RELATIVE.as_posix(),
        "operator_source_sha256": _sha256_file(workspace / OPERATOR_RELATIVE),
        "parent_phase_manifest_fingerprint": phase1[
            "phase_manifest_fingerprint"
        ],
        "plan_file_sha256": _sha256_file(workspace / PLAN_RELATIVE),
        "plan_sha256": enoch_week1.canonical_json_sha256(plan),
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "runtime_imports": runtime_imports,
        "starting_consumed_count": plan["preclaim_consumed_count"],
        "starting_ledger_fingerprint": plan["preclaim_ledger_fingerprint"],
        "test_source_path": TEST_RELATIVE.as_posix(),
        "test_source_sha256": _sha256_file(workspace / TEST_RELATIVE),
    }
    return _with_fingerprint(body, "continuation_provenance_fingerprint")


def validate_continuation_provenance(
    artifact: Mapping[str, Any],
    layout: W12Layout,
    workspace: Path,
    plan: Mapping[str, Any],
    *,
    control: Mapping[str, Any],
    report: Mapping[str, Any],
    phase1: Mapping[str, Any],
    fixture: Mapping[str, Any],
) -> str:
    expected = build_continuation_provenance(
        layout,
        workspace,
        plan,
        control=control,
        report=report,
        phase1=phase1,
        fixture=fixture,
    )
    if dict(artifact) != expected:
        raise W12OperatorError("W1.2 continuation provenance does not reconstruct")
    return expected["continuation_provenance_fingerprint"]


def _continuation_git_identity_at_commit(
    workspace: Path, base_commit: str, continuation_commit: str
) -> dict[str, Any]:
    workspace = workspace.resolve()
    top = Path(_git_text(workspace, "rev-parse", "--show-toplevel").strip()).resolve()
    if top != workspace:
        raise W12OperatorError("W1.2 verification workspace must be the Git root")
    head = _git_text(
        workspace, "rev-parse", f"{continuation_commit}^{{commit}}"
    ).strip()
    parents = _git_text(workspace, "rev-list", "--parents", "-n", "1", head).split()
    if parents != [head, base_commit]:
        raise W12OperatorError("stored W1.2 continuation is not a direct child of d048")
    tree = _git_text(workspace, "rev-parse", f"{head}^{{tree}}").strip()
    raw_changes = _git_text(
        workspace,
        "diff-tree",
        "--no-commit-id",
        "--name-status",
        "-r",
        "--no-renames",
        base_commit,
        head,
    )
    parsed = [tuple(line.split("\t")) for line in raw_changes.splitlines()]
    if parsed != [("A", path) for path in CONTINUATION_PATHS]:
        raise W12OperatorError("stored W1.2 continuation diff changed")
    change_records = []
    for status_code, relative in parsed:
        mode_record = _git_text(workspace, "ls-tree", head, "--", relative).strip()
        fields = mode_record.split(None, 3)
        if len(fields) != 4 or fields[0] != "100644" or fields[1] != "blob":
            raise W12OperatorError(f"stored continuation path is not a regular blob: {relative}")
        blob_bytes = _git_bytes(workspace, "show", f"{head}:{relative}")
        change_records.append(
            {
                "new_blob": fields[2],
                "old_blob": None,
                "path": relative,
                "sha256": hashlib.sha256(blob_bytes).hexdigest(),
                "status": status_code,
            }
        )
    critical_blobs = []
    for relative in CRITICAL_BASE_MODULES:
        base_blob = _git_blob(workspace, base_commit, relative)
        current_blob = _git_blob(workspace, head, relative)
        if base_blob != current_blob:
            raise W12OperatorError(f"stored continuation changed d048 module: {relative}")
        critical_blobs.append({"blob": base_blob, "path": relative})
    return {
        "base_tree_manifest_sha256": hashlib.sha256(
            _git_bytes(workspace, "ls-tree", "-r", "-z", "--full-tree", base_commit)
        ).hexdigest(),
        "changed_paths": change_records,
        "critical_base_module_blobs": critical_blobs,
        "git_commit": head,
        "git_tree": tree,
        "git_tree_manifest_sha256": hashlib.sha256(
            _git_bytes(workspace, "ls-tree", "-r", "-z", "--full-tree", head)
        ).hexdigest(),
    }


def validate_stored_continuation_provenance(
    artifact: Mapping[str, Any],
    layout: W12Layout,
    workspace: Path,
    plan: Mapping[str, Any],
    *,
    control: Mapping[str, Any],
    report: Mapping[str, Any],
    phase1: Mapping[str, Any],
    fixture: Mapping[str, Any],
) -> str:
    """Validate stored provenance by Git object, independent of current HEAD."""

    expected_keys = {
        "automatic_production_promotion_allowed",
        "base_git_commit",
        "base_git_tree",
        "base_operator_source_sha256",
        "base_source_provenance_fingerprint",
        "baseline_worker_report_fingerprint",
        "base_tree_manifest_sha256",
        "changed_paths",
        "continuation_git_commit",
        "continuation_git_tree",
        "continuation_provenance_fingerprint",
        "control_manifest_fingerprint",
        "critical_base_module_blobs",
        "evaluator_binary_sha256",
        "evaluator_source_records_sha256",
        "evaluator_source_sha256",
        "fixture_report_fingerprint",
        "fixture_source_files_sha256",
        "git_tree_manifest_sha256",
        "manifest_kind",
        "manifest_version",
        "operator_source_path",
        "operator_source_sha256",
        "parent_phase_manifest_fingerprint",
        "plan_file_sha256",
        "plan_sha256",
        "protocol_fingerprint",
        "runtime_imports",
        "starting_consumed_count",
        "starting_ledger_fingerprint",
        "test_source_path",
        "test_source_sha256",
    }
    _require_exact_keys(artifact, expected_keys, "stored W1.2 provenance")
    body = dict(artifact)
    fingerprint = body.pop("continuation_provenance_fingerprint")
    if fingerprint != enoch_week1.canonical_json_sha256(body):
        raise W12OperatorError("stored W1.2 provenance fingerprint mismatch")
    if (
        artifact["manifest_kind"] != PROVENANCE_KIND
        or artifact["manifest_version"] != MANIFEST_VERSION
        or artifact["automatic_production_promotion_allowed"] is not False
    ):
        raise W12OperatorError("unsupported stored W1.2 provenance")
    base = _load_json(layout.base.provenance)
    base_operator._validate_provenance(  # noqa: SLF001
        _load_json(layout.base.protocol), control, base
    )
    bindings = {
        "base_git_commit": base["git_commit"],
        "base_git_tree": base["git_tree"],
        "base_operator_source_sha256": base["operator_source_sha256"],
        "base_source_provenance_fingerprint": base[
            "source_provenance_fingerprint"
        ],
        "baseline_worker_report_fingerprint": report[
            "baseline_worker_report_fingerprint"
        ],
        "control_manifest_fingerprint": control["control_manifest_fingerprint"],
        "evaluator_binary_sha256": control["evaluator_identity"]["binary_sha256"],
        "evaluator_source_sha256": control["evaluator_identity"]["source_sha256"],
        "fixture_report_fingerprint": fixture["fixture_report_fingerprint"],
        "fixture_source_files_sha256": fixture["source_files_sha256"],
        "operator_source_path": OPERATOR_RELATIVE.as_posix(),
        "parent_phase_manifest_fingerprint": phase1[
            "phase_manifest_fingerprint"
        ],
        "plan_sha256": enoch_week1.canonical_json_sha256(plan),
        "protocol_fingerprint": plan["protocol_fingerprint"],
        "starting_consumed_count": plan["preclaim_consumed_count"],
        "starting_ledger_fingerprint": plan["preclaim_ledger_fingerprint"],
        "test_source_path": TEST_RELATIVE.as_posix(),
    }
    for field, expected in bindings.items():
        if artifact[field] != expected:
            raise W12OperatorError(f"stored W1.2 provenance {field} mismatch")
    actual_base_tree = _git_text(
        workspace, "rev-parse", f"{base['git_commit']}^{{tree}}"
    ).strip()
    if actual_base_tree != base["git_tree"]:
        raise W12OperatorError("stored d048 base tree mismatch")
    git_identity = _continuation_git_identity_at_commit(
        workspace, base["git_commit"], artifact["continuation_git_commit"]
    )
    git_bindings = {
        "base_tree_manifest_sha256": git_identity["base_tree_manifest_sha256"],
        "changed_paths": git_identity["changed_paths"],
        "continuation_git_commit": git_identity["git_commit"],
        "continuation_git_tree": git_identity["git_tree"],
        "critical_base_module_blobs": git_identity["critical_base_module_blobs"],
        "git_tree_manifest_sha256": git_identity["git_tree_manifest_sha256"],
    }
    for field, expected in git_bindings.items():
        if artifact[field] != expected:
            raise W12OperatorError(f"stored W1.2 Git binding changed: {field}")
    changed = {record["path"]: record for record in git_identity["changed_paths"]}
    if artifact["operator_source_sha256"] != changed[
        OPERATOR_RELATIVE.as_posix()
    ]["sha256"]:
        raise W12OperatorError("stored W1.2 operator hash changed")
    if artifact["plan_file_sha256"] != changed[PLAN_RELATIVE.as_posix()]["sha256"]:
        raise W12OperatorError("stored W1.2 plan file hash changed")
    if artifact["test_source_sha256"] != changed[TEST_RELATIVE.as_posix()]["sha256"]:
        raise W12OperatorError("stored W1.2 test hash changed")
    plan_bytes = _git_bytes(
        workspace,
        "show",
        f"{artifact['continuation_git_commit']}:{PLAN_RELATIVE.as_posix()}",
    )
    try:
        committed_plan = json.loads(plan_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise W12OperatorError(f"committed W1.2 plan is unreadable: {exc}") from exc
    if committed_plan != dict(plan):
        raise W12OperatorError("materialized W1.2 plan differs from its Git object")
    frozen_sources = json.loads(
        (
            layout.base.control_bundle
            / "source"
            / "week1-evaluator-source-files.json"
        ).read_text(encoding="utf-8")
    )
    if artifact["evaluator_source_records_sha256"] != (
        enoch_week1.canonical_json_sha256(frozen_sources)
    ):
        raise W12OperatorError("stored evaluator source record hash changed")
    runtime_imports = [
        {
            "path": relative,
            "sha256": hashlib.sha256(
                _git_bytes(
                    workspace,
                    "show",
                    f"{base['git_commit']}:{relative}",
                )
            ).hexdigest(),
        }
        for relative in (
            "training/enoch_week1.py",
            "training/enoch_week1_evidence.py",
            "training/enoch_week1_fixtures.py",
            "training/enoch_week1_freeze.py",
            "training/enoch_week1_operator.py",
            "training/enoch_week1_preflight.py",
            "training/enoch_week1_runner.py",
        )
    ]
    if artifact["runtime_imports"] != runtime_imports:
        raise W12OperatorError("stored W1.2 runtime import hashes changed")
    if Path(__file__).resolve() != (workspace / OPERATOR_RELATIVE).resolve():
        raise W12OperatorError("offline verifier is not the stored W1.2 operator")
    if _sha256_file(Path(__file__).resolve()) != artifact["operator_source_sha256"]:
        raise W12OperatorError("offline W1.2 operator bytes changed")
    live_modules = (
        (enoch_week1, "training/enoch_week1.py"),
        (enoch_week1_evidence, "training/enoch_week1_evidence.py"),
        (enoch_week1_fixtures, "training/enoch_week1_fixtures.py"),
        (enoch_week1_freeze, "training/enoch_week1_freeze.py"),
        (base_operator, "training/enoch_week1_operator.py"),
        (enoch_week1_preflight, "training/enoch_week1_preflight.py"),
        (enoch_week1_runner, "training/enoch_week1_runner.py"),
    )
    expected_runtime = {record["path"]: record["sha256"] for record in runtime_imports}
    for module, relative in live_modules:
        module_path = getattr(module, "__file__", None)
        if not isinstance(module_path, str) or Path(module_path).resolve() != (
            workspace / relative
        ).resolve():
            raise W12OperatorError(f"offline runtime import is shadowed: {relative}")
        if _sha256_file(Path(module_path)) != expected_runtime[relative]:
            raise W12OperatorError(f"offline runtime import changed: {relative}")
    return fingerprint


def _root_bindings(
    plan: Mapping[str, Any],
    control: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "baseline_worker_report_fingerprint": report[
            "baseline_worker_report_fingerprint"
        ],
        "control_manifest_fingerprint": control["control_manifest_fingerprint"],
        "preclaim_consumed_count": plan["preclaim_consumed_count"],
        "preclaim_ledger_fingerprint": plan["preclaim_ledger_fingerprint"],
        "reference_enoch0_fingerprint": report["reference_enoch0_fingerprint"],
        "runtime_control_fingerprint": report[
            "runtime_evaluation_control_fingerprint"
        ],
        "runtime_evaluator_fingerprint": report["runtime_evaluator_fingerprint"],
    }


def build_campaign_declaration(
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    control: Mapping[str, Any],
    report: Mapping[str, Any],
    phase1: Mapping[str, Any],
    fixture: Mapping[str, Any],
    environment: Mapping[str, str] | None = None,
    environment_identity_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build all fifteen result-independent arm declarations at once."""

    validate_committed_plan(plan)
    enoch_week1.validate_protocol(protocol)
    enoch_week1.validate_w1_0_control_manifest(protocol, control)
    enoch_week1.validate_phase_manifest(protocol, phase1)
    fixture_binding = _fixture_binding(fixture)
    provenance_fingerprint = _require_sha256(
        provenance.get("continuation_provenance_fingerprint"),
        "continuation provenance fingerprint",
    )
    if (environment is None) == (environment_identity_override is None):
        raise W12OperatorError(
            "choose exactly one live environment or frozen environment identity"
        )
    if environment_identity_override is None:
        child_environment, _ = enoch_week1.sanitized_evaluator_environment(
            environment,
            allowlist=protocol["evaluator_environment_policy"]["allowlist"],
        )
        environment_identity = (
            enoch_week1_runner.build_evaluator_environment_identity(
                control["evaluator_identity"],
                protocol,
                child_environment,
                available_parallelism=plan["available_parallelism"],
            )
        )
    else:
        environment_identity = dict(environment_identity_override)
    environment_fingerprint = enoch_week1.canonical_json_sha256(
        environment_identity
    )
    expected_environment = report["fixed_worker_configuration"][
        "environment_fingerprint"
    ]
    if environment_fingerprint != expected_environment:
        raise W12OperatorError("W1.2 environment differs from the W1.1-selected environment")
    if report["fixed_worker_configuration"]["maximum_parallel_workers"] != plan[
        "worker_count"
    ]:
        raise W12OperatorError("W1.2 worker count differs from the W1.1 selection")

    arms = []
    for ordinal, (arm_id, plan_arm, registry_arm) in enumerate(
        zip(enoch_week1.ABLATION_ARMS, plan["arms"], enoch_week1.ARM_REGISTRY),
        start=1,
    ):
        if plan_arm["arm_id"] != arm_id or registry_arm["arm_id"] != arm_id:
            raise W12OperatorError("W1.2 arm registries disagree")
        scenario = (
            enoch_week1_runner.DEVELOPMENT_FRIEND_SCENARIO
            if arm_id == "friend-revelation"
            else "standard"
        )
        fixed = plan["fixed_work"]
        launch = enoch_week1_runner.build_launch_configuration(
            candidate_arm_ids=[arm_id],
            worlds=fixed["worlds"],
            candidates=fixed["candidates"],
            rollout_tricks=fixed["rollout_tricks"],
            scenario_id=scenario,
            deadline_ms=fixed["deadline_ms"],
        )
        launch_fingerprint = enoch_week1_runner.validate_launch_configuration(launch)
        identities = enoch_week1_runner.build_in_process_identity_bindings(
            control["evaluator_identity"], launch
        )
        identity_fingerprints = {
            name: enoch_week1.canonical_json_sha256(identity)
            for name, identity in identities.items()
        }
        comparison = enoch_week1.build_comparison_protocol_manifest(
            protocol,
            phase="W1.2",
            comparison_id=plan_arm["comparison_id"],
            subject_id=arm_id,
            seed_namespace=registry_arm["ablation_seed_namespace"],
            pair_count=plan["pair_count_per_arm"],
            shard_count=plan["shard_count"],
            candidate_fingerprint=identity_fingerprints["candidate"],
            control_fingerprint=identity_fingerprints["control"],
            evaluator_fingerprint=identity_fingerprints["evaluator"],
            environment_fingerprint=environment_fingerprint,
            configuration_fingerprint=launch_fingerprint,
            development_rule=plan_arm["development_rule"],
            required_style_metrics=enoch_week1.WEEK1_STYLE_METRICS,
        )
        arms.append(
            {
                "arm_id": arm_id,
                "comparison": comparison,
                "development_rule": plan_arm["development_rule"],
                "development_rule_sha256": enoch_week1.validate_development_rule(
                    plan_arm["development_rule"]
                ),
                "fixture_prerequisites": registry_arm["fixture_prerequisites"],
                "identity_bindings": identities,
                "launch_configuration": launch,
                "launch_configuration_sha256": launch_fingerprint,
                "ordinal": ordinal,
                "seed_namespace": registry_arm["ablation_seed_namespace"],
            }
        )
    shard_pair_counts = [
        len(item["seed_indices"]) for item in arms[0]["comparison"]["shards"]
    ]
    body = {
        "arm_registry_sha256": enoch_week1.ARM_REGISTRY_SHA256,
        "arms": arms,
        "automatic_production_promotion_allowed": False,
        "execution_contract": {
            "available_parallelism": plan["available_parallelism"],
            "environment_fingerprint": environment_fingerprint,
            "environment_identity": environment_identity,
            "execution_order": list(enoch_week1.ABLATION_ARMS),
            "failure_disposition": plan["failure_disposition"],
            "fixed_work": plan["fixed_work"],
            "pair_count": plan["pair_count_per_arm"],
            "ranking_rule": plan["ranking_rule"],
            "require_machine_global_lock": True,
            "required_external_artifact_ids": list(
                enoch_week1_evidence.EXPECTED_ARTIFACT_IDS
            ),
            "required_failure_counters": list(enoch_week1.FAILURE_COUNTER_NAMES),
            "required_style_metrics": list(enoch_week1.WEEK1_STYLE_METRICS),
            "shard_count": plan["shard_count"],
            "shard_pair_counts": shard_pair_counts,
            "timeout_seconds": plan["shard_timeout_seconds"],
            "worker_count": plan["worker_count"],
        },
        "fixture_binding": fixture_binding,
        "manifest_kind": DECLARATION_KIND,
        "manifest_version": MANIFEST_VERSION,
        "operator_source_provenance_fingerprint": provenance_fingerprint,
        "parent_phase": {
            "phase": "W1.1",
            "phase_manifest_fingerprint": phase1["phase_manifest_fingerprint"],
        },
        "plan_sha256": enoch_week1.canonical_json_sha256(plan),
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "root_bindings": _root_bindings(plan, control, report),
        "seed_registry_sha256": protocol["seed_registry_sha256"],
    }
    return _with_fingerprint(body, "campaign_declaration_fingerprint")


def validate_campaign_declaration(
    declaration: Mapping[str, Any],
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    control: Mapping[str, Any],
    report: Mapping[str, Any],
    phase1: Mapping[str, Any],
    fixture: Mapping[str, Any],
    environment: Mapping[str, str] | None = None,
    environment_identity_override: Mapping[str, Any] | None = None,
) -> str:
    expected = build_campaign_declaration(
        protocol,
        plan,
        provenance,
        control=control,
        report=report,
        phase1=phase1,
        fixture=fixture,
        environment=environment,
        environment_identity_override=environment_identity_override,
    )
    if dict(declaration) != expected:
        raise W12OperatorError("W1.2 campaign declaration does not reconstruct")
    return expected["campaign_declaration_fingerprint"]


def _arm_by_id(declaration: Mapping[str, Any], arm_id: str) -> Mapping[str, Any]:
    for arm in declaration["arms"]:
        if arm["arm_id"] == arm_id:
            return arm
    raise W12OperatorError(f"campaign declaration omits arm {arm_id}")


def _materialize_declaration(
    layout: W12Layout,
    plan: Mapping[str, Any],
    provenance: Mapping[str, Any],
    declaration: Mapping[str, Any],
) -> None:
    _write_or_match(layout.input, plan, "W1.2 committed plan copy")
    _write_or_match(layout.provenance, provenance, "W1.2 continuation provenance")
    environment_identity = declaration["execution_contract"]["environment_identity"]
    for arm in declaration["arms"]:
        root = layout.arm(arm["ordinal"], arm["arm_id"]) / "declaration"
        _write_or_match(
            root / "development-rule.json",
            arm["development_rule"],
            f"{arm['arm_id']} development rule",
        )
        _write_or_match(
            root / "launch.json",
            arm["launch_configuration"],
            f"{arm['arm_id']} launch configuration",
        )
        _write_or_match(
            root / "identities.json",
            arm["identity_bindings"],
            f"{arm['arm_id']} identities",
        )
        _write_or_match(
            root / "environment-identity.json",
            environment_identity,
            f"{arm['arm_id']} environment identity",
        )
        _write_or_match(
            root / "comparison.json",
            arm["comparison"],
            f"{arm['arm_id']} comparison",
        )
    # The index is the commit marker.  It is written only after all 15 arm
    # declarations exist and match.
    _write_or_match(layout.declaration, declaration, "W1.2 declaration index")


def _verify_materialized_declaration(
    layout: W12Layout,
    plan: Mapping[str, Any],
    provenance: Mapping[str, Any],
    declaration: Mapping[str, Any],
) -> None:
    if _load_json(layout.input) != dict(plan):
        raise W12OperatorError("materialized W1.2 plan changed")
    if _load_json(layout.provenance) != dict(provenance):
        raise W12OperatorError("materialized W1.2 provenance changed")
    if _load_json(layout.declaration) != dict(declaration):
        raise W12OperatorError("materialized W1.2 declaration index changed")
    environment_identity = declaration["execution_contract"]["environment_identity"]
    for arm in declaration["arms"]:
        root = layout.arm(arm["ordinal"], arm["arm_id"]) / "declaration"
        expected_files = {
            "comparison.json": arm["comparison"],
            "development-rule.json": arm["development_rule"],
            "environment-identity.json": environment_identity,
            "identities.json": arm["identity_bindings"],
            "launch.json": arm["launch_configuration"],
        }
        for name, expected in expected_files.items():
            if _load_json(root / name) != expected:
                raise W12OperatorError(
                    f"materialized {arm['arm_id']} declaration changed: {name}"
                )


def _verify_w1_1_first(layout: W12Layout) -> dict[str, str]:
    try:
        return base_operator.verify_complete(layout.base)
    except (
        base_operator.OperatorError,
        enoch_week1.ProtocolError,
        enoch_week1_evidence.EvidenceError,
        enoch_week1_fixtures.FixtureError,
        enoch_week1_freeze.FreezeError,
        enoch_week1_preflight.PreflightError,
        enoch_week1_runner.RunnerError,
        OSError,
    ) as exc:
        raise W12OperatorError(f"completed W1.0/W1.1 verification failed: {exc}") from exc


def _load_base_state(
    layout: W12Layout, verified: Mapping[str, str]
) -> dict[str, Any]:
    protocol = _load_json(layout.base.protocol)
    ledger = _load_json(layout.base.ledger)
    enoch_week1.validate_protocol(protocol)
    enoch_week1.validate_seed_ledger(protocol, ledger)
    control = _load_json(layout.base.control_bundle / "control-manifest.json")
    enoch_week1.validate_w1_0_control_manifest(protocol, control)
    report = _load_json(layout.base.report)
    phase0 = _load_json(layout.base.phase0)
    phase1 = _load_json(layout.base.phase1)
    enoch_week1.validate_phase_chain(protocol, [phase0, phase1])
    fixture = base_operator._validate_fixture_gate(layout.base)  # noqa: SLF001
    expected = {
        "control_manifest_fingerprint": control["control_manifest_fingerprint"],
        "fixture_report_fingerprint": fixture["fixture_report_fingerprint"],
        "phase_manifest_fingerprint": phase1["phase_manifest_fingerprint"],
        "report_fingerprint": report["baseline_worker_report_fingerprint"],
    }
    for field, value in expected.items():
        if verified[field] != value:
            raise W12OperatorError(f"W1.1 state changed after verification: {field}")
    return {
        "control": control,
        "fixture": fixture,
        "ledger": ledger,
        "phase0": phase0,
        "phase1": phase1,
        "protocol": protocol,
        "report": report,
    }


def _require_preclaim_ledger(
    ledger: Mapping[str, Any], plan: Mapping[str, Any]
) -> None:
    if ledger["ledger_fingerprint"] != plan["preclaim_ledger_fingerprint"]:
        raise W12OperatorError("W1.2 declaration requires the exact completed W1.1 ledger")
    if len(ledger["consumed"]) != plan["preclaim_consumed_count"]:
        raise W12OperatorError("W1.2 declaration preclaim count changed")
    if _development_claims(ledger):
        raise W12OperatorError("W1.2 declaration cannot follow a development seed claim")


def declare_w1_2(
    layout: W12Layout,
    workspace: Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Freeze every W1.2 comparison before any development seed is observed."""

    workspace = workspace.expanduser().resolve()
    base_environment = dict(os.environ if environment is None else environment)
    verified = _verify_w1_1_first(layout)
    with base_operator._operator_lock(layout.base.operator_lock):  # noqa: SLF001
        state = _load_base_state(layout, verified)
        plan = load_committed_plan(workspace)
        _require_preclaim_ledger(state["ledger"], plan)
        if layout.retirement.exists():
            raise W12OperatorError("this protocol is already retired for W1.2")
        provenance = build_continuation_provenance(
            layout,
            workspace,
            plan,
            control=state["control"],
            report=state["report"],
            phase1=state["phase1"],
            fixture=state["fixture"],
        )
        declaration = build_campaign_declaration(
            state["protocol"],
            plan,
            provenance,
            control=state["control"],
            report=state["report"],
            phase1=state["phase1"],
            fixture=state["fixture"],
            environment=base_environment,
        )
        _materialize_declaration(layout, plan, provenance, declaration)
        return {
            "campaign_declaration_fingerprint": declaration[
                "campaign_declaration_fingerprint"
            ],
            "continuation_provenance_fingerprint": provenance[
                "continuation_provenance_fingerprint"
            ],
            "pair_count_per_arm": plan["pair_count_per_arm"],
            "planned_pair_count": plan["pair_count_per_arm"]
            * len(enoch_week1.ABLATION_ARMS),
            "protocol_fingerprint": state["protocol"]["protocol_fingerprint"],
        }


def _expected_claim_consumers(comparison: Mapping[str, Any]) -> dict[int, str]:
    prefix = comparison["comparison_protocol_fingerprint"][:16]
    expected: dict[int, str] = {}
    for assignment in comparison["shards"]:
        consumer = f"runner:{prefix}:{assignment['shard_id']}"
        for index in assignment["seed_indices"]:
            expected[index] = consumer
    return expected


def _namespace_claims(
    ledger: Mapping[str, Any], namespace: str
) -> list[Mapping[str, Any]]:
    return [row for row in ledger["consumed"] if row["namespace"] == namespace]


def _validate_claims(
    ledger: Mapping[str, Any], comparison: Mapping[str, Any]
) -> None:
    actual = {
        row["index"]: row["consumer"]
        for row in _namespace_claims(ledger, comparison["seed_namespace"])
    }
    if actual != _expected_claim_consumers(comparison):
        raise W12OperatorError(
            f"{comparison['seed_namespace']} claims do not match the completed run"
        )


def _validate_claim_frontier(
    ledger: Mapping[str, Any],
    declaration: Mapping[str, Any],
    completed_arm_ids: Sequence[str],
) -> None:
    _validate_preclaim_prefix(ledger, declaration)
    prefix_count = declaration["root_bindings"]["preclaim_consumed_count"]
    expected: dict[tuple[str, int], str] = {}
    for arm_id in completed_arm_ids:
        comparison = _arm_by_id(declaration, arm_id)["comparison"]
        for index, consumer in _expected_claim_consumers(comparison).items():
            expected[(comparison["seed_namespace"], index)] = consumer
    actual = {
        (row["namespace"], row["index"]): row["consumer"]
        for row in ledger["consumed"][prefix_count:]
    }
    if actual != expected:
        raise W12OperatorError(
            "ledger frontier contains future, downstream, missing, or undeclared claims"
        )


def _attempts(arm_root: Path) -> list[Path]:
    attempts_root = arm_root / "attempts"
    if not attempts_root.exists():
        return []
    attempts = []
    for path in attempts_root.iterdir():
        if not path.is_dir() or not re.fullmatch(r"attempt-\d{3}", path.name):
            raise W12OperatorError(f"malformed W1.2 attempt entry: {path}")
        attempts.append(path)
    return sorted(attempts)


def _next_attempt(arm_root: Path) -> Path:
    attempts = _attempts(arm_root)
    ordinal = max((int(path.name[-3:]) for path in attempts), default=0) + 1
    path = arm_root / "attempts" / f"attempt-{ordinal:03d}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _completed_attempt(arm_root: Path) -> Path | None:
    completed = [
        path
        for path in _attempts(arm_root)
        if (path / "execution" / "execution-complete.json").is_file()
    ]
    if len(completed) > 1:
        raise W12OperatorError(f"multiple completed attempts exist under {arm_root}")
    return completed[0] if completed else None


def _validate_environment_probe(
    artifact: Mapping[str, Any], expected_environment: Mapping[str, Any]
) -> str:
    _require_exact_keys(
        artifact,
        {"environment", "environment_identity_sha256"},
        "environment probe",
    )
    fingerprint = enoch_week1.canonical_json_sha256(expected_environment)
    if (
        artifact["environment"] != dict(expected_environment)
        or artifact["environment_identity_sha256"] != fingerprint
    ):
        raise W12OperatorError("environment probe differs from the declaration")
    return fingerprint


def _model_contract_paths(layout: W12Layout) -> dict[str, Path]:
    return {
        artifact_id: layout.base.control_bundle
        / "preflight"
        / f"{artifact_id.rsplit('/', 1)[-1]}.json"
        for artifact_id in enoch_week1_evidence.MODEL_CONTRACT_ARTIFACTS
    }


def _validate_runner_record(
    record: Mapping[str, Any],
    *,
    comparison: Mapping[str, Any],
    launch: Mapping[str, Any],
    evidence: Mapping[str, Any],
    environment_probe: Mapping[str, Any],
    evaluator: Path,
    protocol_path: Path,
    workers: int,
    available_parallelism: int,
    dry_run: bool,
) -> None:
    commands = [
        enoch_week1_runner.build_shard_command(
            evaluator,
            protocol_path,
            comparison,
            launch,
            assignment["shard_id"],
        )
        for assignment in comparison["shards"]
    ]
    expected_common = {
        "arm_feature_mapping_sha256": enoch_week1_runner.ARM_TO_RUST_FEATURE_SHA256,
        "available_parallelism": available_parallelism,
        "commands": commands,
        "comparison_protocol_fingerprint": comparison[
            "comparison_protocol_fingerprint"
        ],
        "dry_run": dry_run,
        "external_failure_evidence_fingerprint": evidence[
            "verified_external_evidence_fingerprint"
        ],
        "launch_configuration_sha256": enoch_week1_runner.validate_launch_configuration(
            launch
        ),
        "evaluator_environment_identity_sha256": comparison[
            "environment_fingerprint"
        ],
        "environment_identity_probe": environment_probe,
        "manifest_kind": "enoch-week1-runner-execution",
        "manifest_version": enoch_week1_runner.RUNNER_MANIFEST_VERSION,
        "qualification_scenario_bindings_sha256": (
            enoch_week1_runner.QUALIFICATION_SCENARIO_BINDINGS_SHA256
        ),
        "rust_style_metrics_sha256": enoch_week1_runner.RUST_STYLE_METRICS_SHA256,
        "worker_limit": min(workers, len(commands)),
    }
    if dry_run:
        if dict(record) != expected_common:
            raise W12OperatorError("saved runner dry-run plan does not reconstruct")
        return
    for field, expected in expected_common.items():
        if record.get(field) != expected:
            raise W12OperatorError(f"runner completion changed {field}")
    if set(record) != {*expected_common, "merged_result", "shard_results"}:
        raise W12OperatorError("runner completion schema changed")


def _validate_completed_arm(
    layout: W12Layout,
    protocol: Mapping[str, Any],
    ledger: Mapping[str, Any],
    declaration: Mapping[str, Any],
    arm: Mapping[str, Any],
    attempt: Path,
    *,
    base_environment: Mapping[str, str],
    allow_decision_write: bool = True,
) -> dict[str, Any]:
    comparison = arm["comparison"]
    launch = arm["launch_configuration"]
    identities = arm["identity_bindings"]
    environment_identity = declaration["execution_contract"]["environment_identity"]
    arm_root = layout.arm(arm["ordinal"], arm["arm_id"])
    declaration_root = arm_root / "declaration"
    execution = attempt / "execution"
    expected_execution_files = {
        "comparison.json": comparison,
        "identity-bindings.json": identities,
        "launch-configuration.json": launch,
    }
    for name, expected in expected_execution_files.items():
        if _load_json(execution / name) != expected:
            raise W12OperatorError(f"completed {arm['arm_id']} changed {name}")
    probe = _load_json(attempt / "environment-probe.json")
    _validate_environment_probe(probe, environment_identity)
    machine = _load_json(attempt / "machine-attestation.json")
    try:
        machine_fingerprint = (
            enoch_week1_evidence.validate_machine_contention_attestation(
                comparison, machine
            )
        )
    except enoch_week1_evidence.EvidenceError as exc:
        raise W12OperatorError(f"invalid {arm['arm_id']} machine attestation: {exc}") from exc
    contract = declaration["execution_contract"]
    if (
        machine["worker_count"] != contract["worker_count"]
        or machine["available_parallelism"] != contract["available_parallelism"]
    ):
        raise W12OperatorError(f"{arm['arm_id']} machine declaration changed")
    evidence = _load_json(attempt / "external-evidence.json")
    try:
        enoch_week1_evidence.validate_verified_external_evidence(
            protocol, comparison, evidence
        )
    except (enoch_week1_evidence.EvidenceError, enoch_week1.ProtocolError) as exc:
        raise W12OperatorError(f"invalid {arm['arm_id']} external evidence: {exc}") from exc
    if _load_json(execution / "external-failure-evidence.json") != evidence:
        raise W12OperatorError(f"completed {arm['arm_id']} changed external evidence")

    shards = []
    raw_output_sha256s: list[dict[str, str]] = []
    child_environment, _ = enoch_week1.sanitized_evaluator_environment(
        base_environment,
        allowlist=protocol["evaluator_environment_policy"]["allowlist"],
    )
    for assignment in comparison["shards"]:
        shard_id = assignment["shard_id"]
        shard = _load_json(execution / f"{shard_id}.result.json")
        enoch_week1.validate_shard_result(protocol, comparison, shard)
        stderr_path = execution / f"{shard_id}.stderr.txt"
        if not stderr_path.is_file() or stderr_path.stat().st_size != 0:
            raise W12OperatorError(f"{arm['arm_id']} {shard_id} stderr is not empty")
        raw_path = execution / f"{shard_id}.raw.json"
        if not raw_path.is_file():
            raise W12OperatorError(f"{arm['arm_id']} {shard_id} raw output is missing")
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
            raise W12OperatorError(
                f"{arm['arm_id']} {shard_id} does not reconstruct from raw output"
            )
        raw_output_sha256s.append(
            {"sha256": _sha256_file(raw_path), "shard_id": shard_id}
        )
        shards.append(shard)
    recomputed = enoch_week1.merge_shard_results(protocol, comparison, shards)
    merged = _load_json(execution / "merged-result.json")
    if merged != recomputed:
        raise W12OperatorError(f"{arm['arm_id']} merged result does not reconstruct")
    enoch_week1.validate_merged_result(protocol, comparison, merged)
    for name, count in merged["metrics"]["failure_counters"].items():
        if count != 0:
            raise W12OperatorError(
                f"{arm['arm_id']} has invalidating failure counter {name}={count}"
            )

    runner_plan = _load_json(attempt / "runner-plan.json")
    runner_probe = runner_plan.get("environment_identity_probe")
    if not isinstance(runner_probe, Mapping):
        raise W12OperatorError("runner dry-run plan lacks an environment probe")
    _validate_runner_record(
        runner_plan,
        comparison=comparison,
        launch=launch,
        evidence=evidence,
        environment_probe=runner_probe,
        evaluator=layout.base.control_bundle / "bin" / "enoch-week1-evaluator",
        protocol_path=layout.base.protocol,
        workers=contract["worker_count"],
        available_parallelism=contract["available_parallelism"],
        dry_run=True,
    )
    _validate_environment_probe(
        runner_probe, environment_identity
    )
    completion = _load_json(execution / "execution-complete.json")
    completion_probe = completion.get("environment_identity_probe")
    if not isinstance(completion_probe, Mapping):
        raise W12OperatorError("runner completion lacks an environment probe")
    _validate_runner_record(
        completion,
        comparison=comparison,
        launch=launch,
        evidence=evidence,
        environment_probe=completion_probe,
        evaluator=layout.base.control_bundle / "bin" / "enoch-week1-evaluator",
        protocol_path=layout.base.protocol,
        workers=contract["worker_count"],
        available_parallelism=contract["available_parallelism"],
        dry_run=False,
    )
    _validate_environment_probe(
        completion_probe, environment_identity
    )
    merged_result_path = completion.get("merged_result")
    if not isinstance(merged_result_path, str):
        raise W12OperatorError("runner completion merged-result path is invalid")
    if Path(merged_result_path).resolve() != (
        execution / "merged-result.json"
    ).resolve():
        raise W12OperatorError(f"{arm['arm_id']} completion points at another merge")
    expected_shard_paths = {
        assignment["shard_id"]: str(
            execution / f"{assignment['shard_id']}.result.json"
        )
        for assignment in comparison["shards"]
    }
    if completion.get("shard_results") != expected_shard_paths:
        raise W12OperatorError(f"{arm['arm_id']} completion shard paths changed")
    _validate_claims(ledger, comparison)

    expected_decision = enoch_week1.build_w1_3_advancement_decision(
        protocol,
        comparison,
        merged,
        fixture_report=_load_json(layout.base.fixtures / "fixture-report.json"),
    )
    decision_path = arm_root / "advancement-decision.json"
    if decision_path.exists():
        if _load_json(decision_path) != expected_decision:
            raise W12OperatorError(
                f"{arm['arm_id']} advancement decision does not reconstruct"
            )
    elif allow_decision_write:
        enoch_week1.atomic_write_json(decision_path, expected_decision)
    else:
        raise W12OperatorError(f"{arm['arm_id']} advancement decision is missing")
    enoch_week1.validate_w1_3_advancement_decision(
        protocol,
        comparison,
        merged,
        _load_json(layout.base.fixtures / "fixture-report.json"),
        expected_decision,
    )
    return {
        "advancement_decision": expected_decision,
        "arm_id": arm["arm_id"],
        "attempt": attempt,
        "comparison": comparison,
        "environment_probe": probe,
        "external_evidence": evidence,
        "external_evidence_fingerprint": evidence[
            "verified_external_evidence_fingerprint"
        ],
        "identity_bindings": identities,
        "launch_configuration": launch,
        "machine_attestation": machine,
        "machine_attestation_fingerprint": machine_fingerprint,
        "merged_result": merged,
        "runner_execution": completion,
        "raw_output_sha256s": raw_output_sha256s,
        "shard_results": shards,
    }


def _retirement_body(
    protocol: Mapping[str, Any],
    ledger: Mapping[str, Any],
    arm: Mapping[str, Any],
    arm_root: Path,
    *,
    reason: str = "consumed-w1.2-namespace-lacks-one-valid-completion",
) -> dict[str, Any]:
    claims = [
        {
            "consumer": row["consumer"],
            "index": row["index"],
            "namespace": row["namespace"],
            "seed": row["seed"],
            "sequence": row["sequence"],
        }
        for row in _namespace_claims(ledger, arm["seed_namespace"])
    ]
    body = {
        "arm_id": arm["arm_id"],
        "attempt_directories": [path.name for path in _attempts(arm_root)],
        "automatic_production_promotion_allowed": False,
        "claims": claims,
        "disposition": FAILURE_DISPOSITION,
        "ledger_fingerprint": ledger["ledger_fingerprint"],
        "manifest_kind": RETIREMENT_KIND,
        "manifest_version": MANIFEST_VERSION,
        "phase": "W1.2",
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "reason": reason,
        "seed_namespace": arm["seed_namespace"],
    }
    return _with_fingerprint(body, "retirement_fingerprint")


def _retire_if_claimed_incomplete(
    layout: W12Layout,
    protocol: Mapping[str, Any],
    ledger: Mapping[str, Any],
    arm: Mapping[str, Any],
) -> None:
    arm_root = layout.arm(arm["ordinal"], arm["arm_id"])
    if not _namespace_claims(ledger, arm["seed_namespace"]):
        return
    if _completed_attempt(arm_root) is not None:
        # A completion marker may need decision reconstruction or another
        # validation pass.  Never misclassify it as claimed-incomplete work.
        return
    retirement = _retirement_body(protocol, ledger, arm, arm_root)
    _write_or_match(layout.retirement, retirement, "W1.2 protocol retirement")
    attempts = _attempts(arm_root)
    if attempts:
        _write_or_match(
            attempts[-1] / "failure-tombstone.json",
            retirement,
            f"{arm['arm_id']} failure tombstone",
        )
    raise W12OperatorError(
        f"{arm['seed_namespace']} has claimed but incomplete work; protocol retired"
    )


def _retire_invalid_completion(
    layout: W12Layout,
    protocol: Mapping[str, Any],
    ledger: Mapping[str, Any],
    arm: Mapping[str, Any],
    attempt: Path,
    *,
    reason: str = "claimed-w1.2-completion-marker-is-invalid",
) -> None:
    arm_root = layout.arm(arm["ordinal"], arm["arm_id"])
    retirement = _retirement_body(
        protocol,
        ledger,
        arm,
        arm_root,
        reason=reason,
    )
    _write_or_match(layout.retirement, retirement, "W1.2 protocol retirement")
    _write_or_match(
        attempt / "failure-tombstone.json",
        retirement,
        f"{arm['arm_id']} invalid-completion tombstone",
    )
    raise W12OperatorError(
        f"{arm['seed_namespace']} has an invalid claimed completion; protocol retired"
    )


def _scan_resume_frontier(
    layout: W12Layout,
    protocol: Mapping[str, Any],
    declaration: Mapping[str, Any],
) -> list[str]:
    ledger = _load_json(layout.base.ledger)
    enoch_week1.validate_seed_ledger(protocol, ledger)
    completed_prefix: list[str] = []
    gap_seen = False
    for arm_id in enoch_week1.ABLATION_ARMS:
        arm = _arm_by_id(declaration, arm_id)
        arm_root = layout.arm(arm["ordinal"], arm_id)
        marker = _completed_attempt(arm_root)
        claims = _namespace_claims(ledger, arm["seed_namespace"])
        if claims and marker is None:
            _retire_if_claimed_incomplete(layout, protocol, ledger, arm)
        if marker is not None and not claims:
            raise W12OperatorError(f"{arm_id} has a completion marker without claims")
        if marker is not None:
            try:
                _validate_claims(ledger, arm["comparison"])
            except W12OperatorError:
                _retire_invalid_completion(
                    layout, protocol, ledger, arm, marker
                )
            if gap_seen:
                _retire_invalid_completion(
                    layout,
                    protocol,
                    ledger,
                    arm,
                    marker,
                    reason="claimed-w1.2-completion-is-out-of-order",
                )
            completed_prefix.append(arm_id)
        else:
            gap_seen = True
    _validate_claim_frontier(ledger, declaration, completed_prefix)
    return completed_prefix


def _record_preclaim_failure(attempt: Path, arm: Mapping[str, Any], exc: BaseException) -> None:
    body = {
        "arm_id": arm["arm_id"],
        "automatic_production_promotion_allowed": False,
        "error": f"{type(exc).__name__}: {exc}",
        "manifest_kind": "enoch-week1-w1.2-preclaim-failure",
        "manifest_version": MANIFEST_VERSION,
        "retry_disposition": "new-attempt-allowed-only-while-namespace-unconsumed",
        "seed_namespace": arm["seed_namespace"],
    }
    artifact = _with_fingerprint(body, "preclaim_failure_fingerprint")
    path = attempt / "preclaim-failure.json"
    if path.exists():
        if _load_json(path) != artifact:
            raise W12OperatorError(
                f"existing preclaim failure does not reconstruct: {path}"
            )
        return
    enoch_week1.atomic_write_json(path, artifact)


def _seal_abandoned_preclaim_attempts(
    arm_root: Path, arm: Mapping[str, Any]
) -> None:
    for attempt in _attempts(arm_root):
        if (attempt / "execution" / "execution-complete.json").exists():
            continue
        if any(
            (attempt / name).exists()
            for name in (
                "failure-tombstone.json",
                "preclaim-abandoned.json",
                "preclaim-failure.json",
            )
        ):
            continue
        body = {
            "arm_id": arm["arm_id"],
            "attempt_id": attempt.name,
            "automatic_production_promotion_allowed": False,
            "manifest_kind": "enoch-week1-w1.2-preclaim-abandoned",
            "manifest_version": MANIFEST_VERSION,
            "reason": "prior-attempt-ended-without-claim-or-completion-marker",
            "retry_disposition": "new-attempt-allowed-because-namespace-is-unconsumed",
            "seed_namespace": arm["seed_namespace"],
        }
        enoch_week1.atomic_write_json(
            attempt / "preclaim-abandoned.json",
            _with_fingerprint(body, "preclaim_abandoned_fingerprint"),
        )


def _run_arm(
    layout: W12Layout,
    workspace: Path,
    state: Mapping[str, Any],
    plan: Mapping[str, Any],
    provenance: Mapping[str, Any],
    declaration: Mapping[str, Any],
    arm: Mapping[str, Any],
    *,
    operator_id: str,
    base_environment: Mapping[str, str],
) -> dict[str, Any]:
    protocol = state["protocol"]
    comparison = arm["comparison"]
    arm_root = layout.arm(arm["ordinal"], arm["arm_id"])
    completed = _completed_attempt(arm_root)
    ledger = _load_json(layout.base.ledger)
    enoch_week1.validate_seed_ledger(protocol, ledger)
    if completed is not None:
        try:
            return _validate_completed_arm(
                layout,
                protocol,
                ledger,
                declaration,
                arm,
                completed,
                base_environment=base_environment,
            )
        except OSError:
            # A decision-file I/O failure is recoverable because the execution
            # core and claims remain untouched; a later invocation reconstructs it.
            raise
        except (
            W12OperatorError,
            KeyError,
            TypeError,
            enoch_week1.ProtocolError,
            enoch_week1_evidence.EvidenceError,
            enoch_week1_runner.RunnerError,
        ):
            _retire_invalid_completion(
                layout, protocol, ledger, arm, completed
            )
    _retire_if_claimed_incomplete(layout, protocol, ledger, arm)
    _seal_abandoned_preclaim_attempts(arm_root, arm)

    validate_continuation_provenance(
        provenance,
        layout,
        workspace,
        plan,
        control=state["control"],
        report=state["report"],
        phase1=state["phase1"],
        fixture=state["fixture"],
    )
    attempt: Path | None = None
    try:
        with enoch_week1_runner.authoritative_campaign_lock(
            protocol, comparison
        ) as campaign_lock_token:
            ledger = _load_json(layout.base.ledger)
            enoch_week1.validate_seed_ledger(protocol, ledger)
            _retire_if_claimed_incomplete(layout, protocol, ledger, arm)
            attempt = _next_attempt(arm_root)
            started = _utc_now()
            contract = declaration["execution_contract"]
            probe = enoch_week1_runner.probe_evaluator_environment_identity(
                evaluator=layout.base.control_bundle / "bin" / "enoch-week1-evaluator",
                protocol_path=layout.base.protocol,
                protocol=protocol,
                comparison=comparison,
                launch_configuration=arm["launch_configuration"],
                child_environment=enoch_week1.sanitized_evaluator_environment(
                    base_environment,
                    allowlist=protocol["evaluator_environment_policy"]["allowlist"],
                )[0],
                evaluator_identity=arm["identity_bindings"]["evaluator"],
                available_parallelism=contract["available_parallelism"],
                campaign_lock_token=campaign_lock_token,
                timeout_seconds=contract["timeout_seconds"],
            )
            _validate_environment_probe(probe, contract["environment_identity"])
            enoch_week1.atomic_write_json(attempt / "environment-probe.json", probe)
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
            enoch_week1.atomic_write_json(
                attempt / "machine-attestation.json", machine
            )
            declaration_root = arm_root / "declaration"
            evidence = enoch_week1_evidence.build_verified_external_evidence(
                protocol,
                comparison,
                fixture_report_path=layout.base.fixtures / "fixture-report.json",
                source_identity_path=(
                    layout.base.control_bundle
                    / "source"
                    / "week1-evaluator-source-files.json"
                ),
                control_manifest_path=(
                    layout.base.control_bundle / "control-manifest.json"
                ),
                runner_identities_path=declaration_root / "identities.json",
                model_contract_artifact_paths=_model_contract_paths(layout),
                machine_attestation_path=attempt / "machine-attestation.json",
            )
            enoch_week1.atomic_write_json(
                attempt / "external-evidence.json", evidence
            )
            enoch_week1_evidence.validate_machine_contention_attestation(
                comparison, machine
            )
            enoch_week1_evidence.validate_verified_external_evidence(
                protocol, comparison, evidence
            )
            common = {
                "protocol_path": layout.base.protocol,
                "comparison_path": declaration_root / "comparison.json",
                "launch_configuration_path": declaration_root / "launch.json",
                "identities_path": declaration_root / "identities.json",
                "evaluator": (
                    layout.base.control_bundle / "bin" / "enoch-week1-evaluator"
                ),
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
                **common,
                output_dir=attempt / "dry-run-unused",
                dry_run=True,
            )
            enoch_week1.atomic_write_json(attempt / "runner-plan.json", runner_plan)
            enoch_week1_runner.run_comparison(
                **common,
                output_dir=attempt / "execution",
                dry_run=False,
            )
            ledger = _load_json(layout.base.ledger)
            try:
                result = _validate_completed_arm(
                    layout,
                    protocol,
                    ledger,
                    declaration,
                    arm,
                    attempt,
                    base_environment=base_environment,
                )
            except OSError:
                raise
            except (
                W12OperatorError,
                KeyError,
                TypeError,
                enoch_week1.ProtocolError,
                enoch_week1_evidence.EvidenceError,
                enoch_week1_runner.RunnerError,
            ):
                _retire_invalid_completion(
                    layout, protocol, ledger, arm, attempt
                )
            validate_continuation_provenance(
                provenance,
                layout,
                workspace,
                plan,
                control=state["control"],
                report=state["report"],
                phase1=state["phase1"],
                fixture=state["fixture"],
            )
            return result
    except BaseException as exc:
        ledger = _load_json(layout.base.ledger)
        enoch_week1.validate_seed_ledger(protocol, ledger)
        claims = _namespace_claims(ledger, arm["seed_namespace"])
        if claims:
            if layout.retirement.exists():
                raise
            _retire_if_claimed_incomplete(layout, protocol, ledger, arm)
            # A valid marker followed by a recoverable decision/provenance I/O
            # failure is neither preclaim nor safe to rerun.  Preserve it and
            # re-raise without writing a contradictory retry tombstone.
            raise
        if attempt is not None:
            _record_preclaim_failure(attempt, arm, exc)
        raise


def _rank_key(
    arm_id: str,
    evidence: Mapping[str, Any],
    ordinals: Mapping[str, int],
) -> tuple[Any, ...]:
    decision = evidence["advancement_decision"]["decision"]
    metrics = evidence["merged_result"]["metrics"]
    return (
        0 if decision == "advance-to-w1.3" else 1,
        -_require_finite(
            metrics["level_utility"]["paired_bootstrap_lower_95"],
            f"{arm_id} level utility lower bound",
        ),
        -_require_finite(
            metrics["level_utility"]["estimate"],
            f"{arm_id} level utility estimate",
        ),
        -_require_finite(
            metrics["point_margin"]["estimate"],
            f"{arm_id} point margin estimate",
        ),
        -_require_finite(
            metrics["win_rate"]["estimate"],
            f"{arm_id} win rate estimate",
        ),
        ordinals[arm_id],
    )


def build_ranked_table(
    protocol: Mapping[str, Any],
    declaration: Mapping[str, Any],
    fixture: Mapping[str, Any],
    arm_evidence: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the typed complete W1.2 exit artifact from all arm evidence."""

    expected_ids = list(enoch_week1.ABLATION_ARMS)
    if list(arm_evidence) != expected_ids:
        raise W12OperatorError("ranked table evidence must contain all arms in order")
    fixture_binding = _fixture_binding(fixture)
    ordinals = {arm_id: index for index, arm_id in enumerate(expected_ids, start=1)}
    ranking = sorted(
        expected_ids,
        key=lambda arm_id: _rank_key(arm_id, arm_evidence[arm_id], ordinals),
    )
    ranks = {arm_id: index for index, arm_id in enumerate(ranking, start=1)}
    aggregate_failures = {name: 0 for name in enoch_week1.FAILURE_COUNTER_NAMES}
    records = []
    for arm_id in expected_ids:
        evidence = arm_evidence[arm_id]
        declared = _arm_by_id(declaration, arm_id)
        if evidence["comparison"] != declared["comparison"]:
            raise W12OperatorError(f"{arm_id} result changed its comparison")
        if evidence["launch_configuration"] != declared["launch_configuration"]:
            raise W12OperatorError(f"{arm_id} result changed its launch")
        if evidence["identity_bindings"] != declared["identity_bindings"]:
            raise W12OperatorError(f"{arm_id} result changed its identities")
        decision = evidence["advancement_decision"]
        enoch_week1.validate_w1_3_advancement_decision(
            protocol,
            declared["comparison"],
            evidence["merged_result"],
            fixture,
            decision,
        )
        for name, count in evidence["merged_result"]["metrics"][
            "failure_counters"
        ].items():
            if count != 0:
                raise W12OperatorError(
                    f"{arm_id} has invalidating failure counter {name}={count}"
                )
            aggregate_failures[name] += count
        records.append(
            {
                "advancement_decision": decision["decision"],
                "advancement_decision_fingerprint": decision[
                    "advancement_decision_fingerprint"
                ],
                "arm_id": arm_id,
                "comparison_protocol_fingerprint": declared["comparison"][
                    "comparison_protocol_fingerprint"
                ],
                "environment_probe_sha256": enoch_week1.canonical_json_sha256(
                    evidence["environment_probe"]
                ),
                "machine_attestation_fingerprint": evidence[
                    "machine_attestation_fingerprint"
                ],
                "merged_result_fingerprint": evidence["merged_result"][
                    "merged_result_fingerprint"
                ],
                "metrics": evidence["merged_result"]["metrics"],
                "ordinal": ordinals[arm_id],
                "rank": ranks[arm_id],
                "raw_output_sha256s": evidence["raw_output_sha256s"],
                "runner_execution_sha256": enoch_week1.canonical_json_sha256(
                    evidence["runner_execution"]
                ),
                "shard_result_fingerprints": [
                    shard["shard_result_fingerprint"]
                    for shard in evidence["shard_results"]
                ],
                "verified_external_evidence_fingerprint": evidence[
                    "external_evidence_fingerprint"
                ],
            }
        )
    advancing = [
        arm_id
        for arm_id in expected_ids
        if arm_evidence[arm_id]["advancement_decision"]["decision"]
        == "advance-to-w1.3"
    ]
    advancing_set = set(advancing)
    stopped = [arm_id for arm_id in expected_ids if arm_id not in advancing_set]
    body = {
        "arm_registry_sha256": enoch_week1.ARM_REGISTRY_SHA256,
        "arm_results": records,
        "automatic_production_promotion_allowed": False,
        "campaign_declaration_fingerprint": declaration[
            "campaign_declaration_fingerprint"
        ],
        "fixture_report_fingerprint": fixture_binding[
            "fixture_report_fingerprint"
        ],
        "manifest_kind": TABLE_KIND,
        "manifest_version": MANIFEST_VERSION,
        "operator_source_provenance_fingerprint": declaration[
            "operator_source_provenance_fingerprint"
        ],
        "parent_phase_manifest_fingerprint": declaration["parent_phase"][
            "phase_manifest_fingerprint"
        ],
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "ranking": ranking,
        "ranking_rule": declaration["execution_contract"]["ranking_rule"],
        "seed_registry_sha256": protocol["seed_registry_sha256"],
        "summary": {
            "advance_count": len(advancing),
            "advancing_arm_ids": advancing,
            "aggregate_failure_counters": aggregate_failures,
            "arm_count": len(expected_ids),
            "pair_count_per_arm": declaration["execution_contract"]["pair_count"],
            "stop_count": len(stopped),
            "stopped_arm_ids": stopped,
            "total_pair_count": declaration["execution_contract"]["pair_count"]
            * len(expected_ids),
        },
    }
    return _with_fingerprint(body, "ranked_independent_ablation_table_fingerprint")


def validate_ranked_table(
    table: Mapping[str, Any],
    protocol: Mapping[str, Any],
    declaration: Mapping[str, Any],
    fixture: Mapping[str, Any],
    arm_evidence: Mapping[str, Mapping[str, Any]],
) -> str:
    expected = build_ranked_table(
        protocol, declaration, fixture, arm_evidence
    )
    if dict(table) != expected:
        raise W12OperatorError("W1.2 ranked table does not reconstruct")
    return expected["ranked_independent_ablation_table_fingerprint"]


def _validate_preclaim_prefix(
    ledger: Mapping[str, Any],
    declaration: Mapping[str, Any],
) -> None:
    prefix_count = declaration["root_bindings"]["preclaim_consumed_count"]
    prefix_body = {
        key: value
        for key, value in ledger.items()
        if key != "ledger_fingerprint"
    }
    prefix_body["consumed"] = list(ledger["consumed"][:prefix_count])
    if enoch_week1.canonical_json_sha256(prefix_body) != declaration[
        "root_bindings"
    ]["preclaim_ledger_fingerprint"]:
        raise W12OperatorError("the completed W1.1 ledger prefix changed")


def _expected_final_ledger(
    protocol: Mapping[str, Any],
    ledger: Mapping[str, Any],
    declaration: Mapping[str, Any],
) -> None:
    enoch_week1.validate_seed_ledger(protocol, ledger)
    _validate_preclaim_prefix(ledger, declaration)
    prefix_count = declaration["root_bindings"]["preclaim_consumed_count"]
    expected_count = prefix_count + sum(
        arm["comparison"]["pair_count"] for arm in declaration["arms"]
    )
    if len(ledger["consumed"]) != expected_count:
        raise W12OperatorError("final W1.2 ledger claim count is incomplete or excessive")
    for arm in declaration["arms"]:
        _validate_claims(ledger, arm["comparison"])
    development_namespaces = {
        arm["seed_namespace"] for arm in declaration["arms"]
    }
    if any(
        row["namespace"].startswith("dev/ablation/")
        and row["namespace"] not in development_namespaces
        for row in ledger["consumed"]
    ):
        raise W12OperatorError("ledger contains an undeclared W1.2 namespace")


def _validate_ledger_extension(
    protocol: Mapping[str, Any],
    current: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> None:
    enoch_week1.validate_seed_ledger(protocol, current)
    enoch_week1.validate_seed_ledger(protocol, snapshot)
    prefix_count = len(snapshot["consumed"])
    prefix_body = {
        key: value
        for key, value in current.items()
        if key != "ledger_fingerprint"
    }
    prefix_body["consumed"] = list(current["consumed"][:prefix_count])
    if enoch_week1.canonical_json_sha256(prefix_body) != snapshot[
        "ledger_fingerprint"
    ]:
        raise W12OperatorError("live ledger is not an append-only extension of W1.2")


def _phase_artifacts(
    provenance: Mapping[str, Any],
    declaration: Mapping[str, Any],
    fixture: Mapping[str, Any],
    table: Mapping[str, Any],
    ledger: Mapping[str, Any],
    arm_evidence: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    artifacts = {
        "continuation-provenance": provenance[
            "continuation_provenance_fingerprint"
        ],
        "fixture-gate": fixture["fixture_report_fingerprint"],
        "ranked-independent-ablation-table": table[
            "ranked_independent_ablation_table_fingerprint"
        ],
        "w1.2-campaign-declaration": declaration[
            "campaign_declaration_fingerprint"
        ],
        "w1.2-final-ledger": ledger["ledger_fingerprint"],
    }
    for arm_id, evidence in arm_evidence.items():
        artifacts[f"w1.2/{arm_id}/advancement-decision"] = evidence[
            "advancement_decision"
        ]["advancement_decision_fingerprint"]
        artifacts[f"w1.2/{arm_id}/comparison"] = evidence["comparison"][
            "comparison_protocol_fingerprint"
        ]
        artifacts[f"w1.2/{arm_id}/external-evidence"] = evidence[
            "external_evidence_fingerprint"
        ]
        artifacts[f"w1.2/{arm_id}/merged-result"] = evidence["merged_result"][
            "merged_result_fingerprint"
        ]
    return artifacts


def _build_phase2(
    protocol: Mapping[str, Any],
    phase1: Mapping[str, Any],
    provenance: Mapping[str, Any],
    declaration: Mapping[str, Any],
    fixture: Mapping[str, Any],
    table: Mapping[str, Any],
    ledger: Mapping[str, Any],
    arm_evidence: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return enoch_week1.build_phase_manifest(
        protocol,
        "W1.2",
        artifacts=_phase_artifacts(
            provenance, declaration, fixture, table, ledger, arm_evidence
        ),
        declarations={
            "advancing_arm_ids": table["summary"]["advancing_arm_ids"],
            "campaign_declaration_fingerprint": declaration[
                "campaign_declaration_fingerprint"
            ],
            "complete_arm_count": len(arm_evidence),
            "continuation_source_commit": provenance["continuation_git_commit"],
            "final_ledger_fingerprint": ledger["ledger_fingerprint"],
            "pair_count_per_arm": declaration["execution_contract"]["pair_count"],
            "ranked_independent_ablation_table_fingerprint": table[
                "ranked_independent_ablation_table_fingerprint"
            ],
            "total_pair_count": table["summary"]["total_pair_count"],
        },
        parent_phase_manifests=[phase1],
    )


def _validate_phase2(
    phase2: Mapping[str, Any],
    protocol: Mapping[str, Any],
    phase0: Mapping[str, Any],
    phase1: Mapping[str, Any],
    provenance: Mapping[str, Any],
    declaration: Mapping[str, Any],
    fixture: Mapping[str, Any],
    table: Mapping[str, Any],
    ledger: Mapping[str, Any],
    arm_evidence: Mapping[str, Mapping[str, Any]],
) -> str:
    expected = _build_phase2(
        protocol,
        phase1,
        provenance,
        declaration,
        fixture,
        table,
        ledger,
        arm_evidence,
    )
    if dict(phase2) != expected:
        raise W12OperatorError("W1.2 phase manifest does not reconstruct")
    if not any(
        record["artifact_id"] == "ranked-independent-ablation-table"
        and record["sha256"]
        == table["ranked_independent_ablation_table_fingerprint"]
        for record in phase2["artifacts"]
    ):
        raise W12OperatorError("W1.2 phase omits its declared ranked-table exit artifact")
    enoch_week1.validate_phase_chain(protocol, [phase0, phase1, expected])
    return expected["phase_manifest_fingerprint"]


def _load_and_validate_declaration_state(
    layout: W12Layout,
    workspace: Path,
    state: Mapping[str, Any],
    *,
    environment: Mapping[str, str],
    live_source: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    stored_plan = _load_json(layout.input)
    validate_committed_plan(stored_plan)
    if live_source:
        plan = load_committed_plan(workspace)
        if stored_plan != plan:
            raise W12OperatorError(
                "W1.2 materialized plan differs from its committed source"
            )
    else:
        plan = stored_plan
    provenance = _load_json(layout.provenance)
    provenance_validator = (
        validate_continuation_provenance
        if live_source
        else validate_stored_continuation_provenance
    )
    provenance_validator(
        provenance,
        layout,
        workspace,
        plan,
        control=state["control"],
        report=state["report"],
        phase1=state["phase1"],
        fixture=state["fixture"],
    )
    declaration = _load_json(layout.declaration)
    declaration_arguments: dict[str, Any] = (
        {"environment": environment}
        if live_source
        else {
            "environment_identity_override": declaration["execution_contract"][
                "environment_identity"
            ]
        }
    )
    validate_campaign_declaration(
        declaration,
        state["protocol"],
        plan,
        provenance,
        control=state["control"],
        report=state["report"],
        phase1=state["phase1"],
        fixture=state["fixture"],
        **declaration_arguments,
    )
    _verify_materialized_declaration(layout, plan, provenance, declaration)
    current_ledger = _load_json(layout.base.ledger)
    enoch_week1.validate_seed_ledger(state["protocol"], current_ledger)
    _validate_preclaim_prefix(current_ledger, declaration)
    return plan, provenance, declaration


def run_w1_2(
    layout: W12Layout,
    workspace: Path,
    *,
    operator_id: str,
    attest_no_machine_contention: bool,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run or safely resume all fifteen predeclared W1.2 comparisons."""

    if attest_no_machine_contention is not True:
        raise W12OperatorError(
            "--attest-no-machine-contention is required for W1.2 execution"
        )
    workspace = workspace.expanduser().resolve()
    base_environment = dict(os.environ if environment is None else environment)
    verified = _verify_w1_1_first(layout)
    with base_operator._operator_lock(layout.base.operator_lock):  # noqa: SLF001
        state = _load_base_state(layout, verified)
        plan, provenance, declaration = _load_and_validate_declaration_state(
            layout,
            workspace,
            state,
            environment=base_environment,
            live_source=True,
        )
        if layout.retirement.exists():
            raise W12OperatorError("this protocol is retired for W1.2")
        declared_completed_prefix = _scan_resume_frontier(
            layout, state["protocol"], declaration
        )
        claim_frontier = list(declared_completed_prefix)
        arm_evidence: dict[str, dict[str, Any]] = {}
        for arm_id in enoch_week1.ABLATION_ARMS:
            arm = _arm_by_id(declaration, arm_id)
            arm_evidence[arm_id] = _run_arm(
                layout,
                workspace,
                state,
                plan,
                provenance,
                declaration,
                arm,
                operator_id=operator_id,
                base_environment=base_environment,
            )
            if arm_id not in claim_frontier:
                claim_frontier.append(arm_id)
            current_ledger = _load_json(layout.base.ledger)
            enoch_week1.validate_seed_ledger(state["protocol"], current_ledger)
            _validate_claim_frontier(
                current_ledger, declaration, claim_frontier
            )
        validate_continuation_provenance(
            provenance,
            layout,
            workspace,
            plan,
            control=state["control"],
            report=state["report"],
            phase1=state["phase1"],
            fixture=state["fixture"],
        )
        live_ledger = _load_json(layout.base.ledger)
        if layout.final_ledger.exists():
            ledger = _load_json(layout.final_ledger)
            _expected_final_ledger(state["protocol"], ledger, declaration)
            if layout.phase.exists():
                _validate_ledger_extension(state["protocol"], live_ledger, ledger)
            elif live_ledger != ledger:
                raise W12OperatorError(
                    "live ledger advanced before the unsealed W1.2 phase completed"
                )
        else:
            _expected_final_ledger(state["protocol"], live_ledger, declaration)
            ledger = live_ledger
            enoch_week1.atomic_write_json(layout.final_ledger, ledger)
        table = build_ranked_table(
            state["protocol"], declaration, state["fixture"], arm_evidence
        )
        _write_or_match(layout.ranked_table, table, "W1.2 ranked ablation table")
        phase2 = _build_phase2(
            state["protocol"],
            state["phase1"],
            provenance,
            declaration,
            state["fixture"],
            table,
            ledger,
            arm_evidence,
        )
        _write_or_match(layout.phase, phase2, "W1.2 phase manifest")
        _validate_phase2(
            phase2,
            state["protocol"],
            state["phase0"],
            state["phase1"],
            provenance,
            declaration,
            state["fixture"],
            table,
            ledger,
            arm_evidence,
        )
        return {
            "advancing_arm_ids": table["summary"]["advancing_arm_ids"],
            "final_ledger_fingerprint": ledger["ledger_fingerprint"],
            "phase_manifest_fingerprint": phase2["phase_manifest_fingerprint"],
            "ranked_independent_ablation_table_fingerprint": table[
                "ranked_independent_ablation_table_fingerprint"
            ],
            "total_pair_count": table["summary"]["total_pair_count"],
        }


def verify_w1_2(
    layout: W12Layout,
    workspace: Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Reconstruct W1.2 from disk without launching the evaluator."""

    workspace = workspace.expanduser().resolve()
    base_environment = dict(os.environ if environment is None else environment)
    verified = _verify_w1_1_first(layout)
    with base_operator._operator_lock(layout.base.operator_lock):  # noqa: SLF001
        state = _load_base_state(layout, verified)
        plan, provenance, declaration = _load_and_validate_declaration_state(
            layout,
            workspace,
            state,
            environment=base_environment,
            live_source=False,
        )
        del plan
        if layout.retirement.exists():
            raise W12OperatorError("this protocol is retired for W1.2")
        ledger = _load_json(layout.final_ledger)
        _expected_final_ledger(state["protocol"], ledger, declaration)
        live_ledger = _load_json(layout.base.ledger)
        _validate_ledger_extension(state["protocol"], live_ledger, ledger)
        arm_evidence: dict[str, dict[str, Any]] = {}
        for arm_id in enoch_week1.ABLATION_ARMS:
            arm = _arm_by_id(declaration, arm_id)
            attempt = _completed_attempt(
                layout.arm(arm["ordinal"], arm["arm_id"])
            )
            if attempt is None:
                raise W12OperatorError(f"missing completed W1.2 arm: {arm_id}")
            arm_evidence[arm_id] = _validate_completed_arm(
                layout,
                state["protocol"],
                ledger,
                declaration,
                arm,
                attempt,
                base_environment=base_environment,
                allow_decision_write=False,
            )
        table = _load_json(layout.ranked_table)
        table_fingerprint = validate_ranked_table(
            table,
            state["protocol"],
            declaration,
            state["fixture"],
            arm_evidence,
        )
        phase2 = _load_json(layout.phase)
        phase_fingerprint = _validate_phase2(
            phase2,
            state["protocol"],
            state["phase0"],
            state["phase1"],
            provenance,
            declaration,
            state["fixture"],
            table,
            ledger,
            arm_evidence,
        )
        return {
            "advancing_arm_ids": table["summary"]["advancing_arm_ids"],
            "campaign_declaration_fingerprint": declaration[
                "campaign_declaration_fingerprint"
            ],
            "final_ledger_fingerprint": ledger["ledger_fingerprint"],
            "phase_manifest_fingerprint": phase_fingerprint,
            "ranked_independent_ablation_table_fingerprint": table_fingerprint,
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("declare-w1.2", "freeze all W1.2 comparisons before any seed claim"),
        ("verify-w1.2", "offline-verify the completed W1.2 package"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--root", required=True, type=Path)
        command.add_argument("--workspace", type=Path, default=Path.cwd())
    run = subparsers.add_parser(
        "run-w1.2", help="run/resume every declared W1.2 arm and seal the phase"
    )
    run.add_argument("--root", required=True, type=Path)
    run.add_argument("--workspace", type=Path, default=Path.cwd())
    run.add_argument("--operator-id", required=True)
    run.add_argument(
        "--attest-no-machine-contention",
        action="store_true",
        help="attest that no competing workload is active during each arm",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    layout = W12Layout(args.root.expanduser().resolve())
    try:
        if args.command == "declare-w1.2":
            result = declare_w1_2(layout, args.workspace)
        elif args.command == "run-w1.2":
            result = run_w1_2(
                layout,
                args.workspace,
                operator_id=args.operator_id,
                attest_no_machine_contention=args.attest_no_machine_contention,
            )
        else:
            result = verify_w1_2(layout, args.workspace)
    except (
        W12OperatorError,
        base_operator.OperatorError,
        enoch_week1.ProtocolError,
        enoch_week1_evidence.EvidenceError,
        enoch_week1_fixtures.FixtureError,
        enoch_week1_freeze.FreezeError,
        enoch_week1_runner.RunnerError,
        FileExistsError,
        OSError,
    ) as exc:
        print(f"W1.2 operator failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
