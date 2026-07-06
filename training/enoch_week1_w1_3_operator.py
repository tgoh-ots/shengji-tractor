#!/usr/bin/env python3
"""Authoritative W1.3 independent-survivor operator.

This is an orchestration-only continuation.  It verifies the sealed W1.2
package with the exact historical d8f source in an explicitly supplied clean
worktree, freezes every W1.3 comparison before the first survivor seed claim,
runs the five comparisons serially under the machine-global lock, and seals a
typed supported-independent-change-set.

Claimed work is never rerun.  An unclaimed preflight failure may use a new
immutable attempt directory; any claimed incomplete or malformed completion
retires the protocol.  Offline verification never launches the evaluator.
"""

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
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

try:
    from training import enoch_week1
    from training import enoch_week1_campaign
    from training import enoch_week1_evidence
    from training import enoch_week1_fixtures
    from training import enoch_week1_freeze
    from training import enoch_week1_operator as base_operator
    from training import enoch_week1_preflight
    from training import enoch_week1_runner
    from training import enoch_week1_w1_2_operator as w12_operator
except ImportError:  # pragma: no cover - direct-script import path.
    import enoch_week1  # type: ignore[no-redef]
    import enoch_week1_campaign  # type: ignore[no-redef]
    import enoch_week1_evidence  # type: ignore[no-redef]
    import enoch_week1_fixtures  # type: ignore[no-redef]
    import enoch_week1_freeze  # type: ignore[no-redef]
    import enoch_week1_operator as base_operator  # type: ignore[no-redef]
    import enoch_week1_preflight  # type: ignore[no-redef]
    import enoch_week1_runner  # type: ignore[no-redef]
    import enoch_week1_w1_2_operator as w12_operator  # type: ignore[no-redef]


MANIFEST_VERSION = 1
BASE_COMMIT = "bfaaa6ec5336b38f7c52a21adb3dca9c841b54cf"
SEALED_W12_COMMIT = "d8f144e330d6f02229e8930d89e80f9376bc6749"
SEALED_W12_PARENT = "d048b7979726042a33caf9b8d0decf62f5894399"
SEALED_W12_PROVENANCE = "d84719bae7e1bfbd7292bb1200cd369841d6c2dd117726ba7629783c78b2757d"
SEALED_W12_DECLARATION = "16c1a9d28d490beb624d313ed7aab0b177a2cbc458a8a07ff884b83599b29bc1"
SEALED_W12_TABLE = "71b68e51cd46c4c6cb9a14b0893df046ef3beb7deba325e9e350a6c2b7ef5f06"
SEALED_W12_LEDGER = "2500ae1fbc5e1f03dc04b62ac92564be014f565443b1e6b0e337234cc2e598d8"
SEALED_W12_PHASE = "70d8a05c0e17a372e42c80e93d760ebf5d9fcb26c03c5474b06d5e3bbf214264"
SEALED_W11_PHASE = "24503b8909df96b6bb1cf47a91da3e830ea623c4ead6db98f9c4881b9043da63"
PROTOCOL_FINGERPRINT = "a1e48199e6cb153c68f442cac9f28400798b994d154e03d34cb64420e21db2b7"
WORKER_REPORT_FINGERPRINT = "79d47b95f65c4c9b5377c5a0894abecd4d02069eb8c965176eb1f69587b8dc6f"
FIXTURE_REPORT_FINGERPRINT = "1fd410dca00fdd83631ae893980176113e621f1eebdec90c3b61f4d87dc7bc50"
ENVIRONMENT_FINGERPRINT = "4f44d61aa9ea44b2d3987da46f0548c1dfa928468d65265c6d9e8273d363ee5f"
PRECLAIM_COUNT = 14_711
FINAL_COUNT = 18_711

SURVIVOR_ARM_IDS = (
    "bid-ownership",
    "compound-follow",
    "friend-revelation",
    "team-void-boss",
    "uncertain-legal-throws",
)

OPERATOR_RELATIVE = Path("training/enoch_week1_w1_3_operator.py")
PLAN_RELATIVE = Path("training/enoch_week1_w1_3_plan.json")
TEST_RELATIVE = Path("training/test_enoch_week1_w1_3_operator.py")
CONTINUATION_PATHS = (
    OPERATOR_RELATIVE.as_posix(),
    PLAN_RELATIVE.as_posix(),
    TEST_RELATIVE.as_posix(),
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
    "training/enoch_week1_w1_2_plan.json",
    "training/test_enoch_week1_w1_2_operator.py",
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
    (w12_operator, "training/enoch_week1_w1_2_operator.py"),
)

PLAN_KIND = "enoch-week1-w1.3-committed-plan"
PROVENANCE_KIND = "enoch-week1-w1.3-continuation-provenance"
DECLARATION_KIND = "enoch-week1-w1.3-campaign-declaration"
SUPPORT_DECISION_KIND = "enoch-week1-w1.4-support-decision"
SUPPORTED_SET_KIND = "enoch-week1-supported-independent-change-set"
RETIREMENT_KIND = "enoch-week1-w1.3-protocol-retirement"
FAILURE_DISPOSITION = "retire-protocol-on-any-claimed-incomplete-comparison"
RANKING_RULE = (
    "advancement-decision-descending",
    "level-utility-lower-95-descending",
    "level-utility-estimate-descending",
    "point-margin-estimate-descending",
    "win-rate-estimate-descending",
    "canonical-arm-ordinal-ascending",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_OBJECT_RE = re.compile(r"[0-9a-f]{40,64}")
_ATTEMPT_RE = re.compile(r"attempt-\d{3}")

EXPECTED_W12_VERIFY_RESULT = {
    "advancing_arm_ids": list(SURVIVOR_ARM_IDS),
    "campaign_declaration_fingerprint": SEALED_W12_DECLARATION,
    "final_ledger_fingerprint": SEALED_W12_LEDGER,
    "phase_manifest_fingerprint": SEALED_W12_PHASE,
    "ranked_independent_ablation_table_fingerprint": SEALED_W12_TABLE,
}

# ``-I`` intentionally removes both the script directory and ambient
# PYTHONPATH.  Reintroduce only the already validated, exact d8 Git root before
# executing the historical operator; keep argv identical to a normal script.
W12_RUNPY_BOOTSTRAP = (
    "import runpy,sys;"
    "workspace=sys.argv.pop(1);script=sys.argv.pop(1);"
    "sys.path.insert(0,workspace);sys.argv[0]=script;"
    "runpy.run_path(script,run_name='__main__')"
)

EXPECTED_PARENT_ARM_FINGERPRINTS = {
    "bid-ownership": {
        "candidate": "fbbdb51143285e9621103ee1a17438c71f5dc90cd7f9256f9edfa9c107d6be7a",
        "configuration": "52bdcea0c2e5971cd975adb1a3ebd6377ae27a1cb9627e158f8a67390a6b3de9",
        "decision": "e890bf027a92d87c5d9093ba7a06c9611bdcf79792db45e97c2e30e1d9d14355",
        "comparison": "987622aba8130f5e20e81c94288b0eb141ed62c9d189ad4ed3a4364541aa1b4e",
        "merged": "0baa7dc8923e4d8ec3152ad135e08f6b1abba6fda1a1750bca7f8d118270db44",
        "ordinal": 1,
        "rule": "617643f57d2cc8bf99e742d8d5bf31a60f99ec3dd16c2ac482b7799d02fb44bd",
        "scenario": "standard",
        "seed_set": "85f0a9dc0e2791ee8ab5dae66ebad5d8fc4f604f980abf17658059f560cd5f6b",
    },
    "compound-follow": {
        "candidate": "721a10eb7b9f9556f15c06c03123e5c1a8fcc2e5d68dd4fb590e05334c188d7f",
        "configuration": "29240c877821da5adb8d46969c0235498da523430f7a50db46dc142d389f589b",
        "decision": "948faf6bfa208bc2fd7b073070ee9a917e38b7b4817601df4775f1b9e0f1192c",
        "comparison": "168b0636398ed71ac2c0ce5f3515069d2c48fe60bad6f5fad8b44bb8e26398b8",
        "merged": "6b4d71e1b182c2da6f8c5e951fb922442cb66076cb2788c4defd02df1eac97aa",
        "ordinal": 2,
        "rule": "6a104b28ae361b28a2fee6f3dd6ae29b9f01be154ace6482ef5e9389163cf720",
        "scenario": "standard",
        "seed_set": "04748bb6cbb43d50f179dd0388b8c32975b2faa3b2a66ca7ef85965eec820620",
    },
    "friend-revelation": {
        "candidate": "cda699f467ed1d2186134d2eb622a21fbfdc6e41c15b57f4d209e23cdcbef3ce",
        "configuration": "01cf1b47505942b35e8ec65a2882290442e1b136b32380dd2669414f9d72012f",
        "decision": "ed0092c6e123af8c268077d65fcf16e32234659f2e09a63d5e50924ab0fa1188",
        "comparison": "75c40961e1686c685a4d0e8e8af0602548f6c71692e2e37ccdfe140f509a34e6",
        "merged": "1f81bc4bf85e38989e8cb7380dfbc45893bf5427abb12877c9c121a3b86bdabf",
        "ordinal": 4,
        "rule": "a4e5175ceca062f256a43bb574005dff3774db1ab565a801d8620561b415bd0f",
        "scenario": "development-finding-friends",
        "seed_set": "e068bcfc8cb0de3c70a89231514f94f719e7b11ebad762c21d76d2c9d99a1a83",
    },
    "team-void-boss": {
        "candidate": "5c88079b7e1b00424cb02cfdb1e772ef7bb6597a26ea3d7820409b604e3014c1",
        "configuration": "0df0baa65701c57ca1b76c799d1da46289838dbef373f398f91117decc371fcb",
        "decision": "dcc56d4017ce8c71d59208abf61d9beb6a9bf2b619e3423a21a92ef104b73467",
        "comparison": "37900d282a2edb964a713263097b0c26d5b6355778d172dfb1651650d3e28a11",
        "merged": "71aa60c1918ba506b5642c4f8a1304e3ca4393baa921cb0c9208c6f93e2587eb",
        "ordinal": 10,
        "rule": "6ca1bb977e6a9ba256f5b455ef9b9c81b408cf52a2afe3f5755f255b12ebb5f0",
        "scenario": "standard",
        "seed_set": "fa254099d0544e2fbb69ad49d6846088be5c7ff43e9fdcc9890244bee218aa04",
    },
    "uncertain-legal-throws": {
        "candidate": "e9bccfe21d38dc08b56dc68becd00c32c93a3ff2f0d02d0aee8a8a784adc54aa",
        "configuration": "c4099259e8349d709973091d74227fc2ac1866ed4431a9b147b2afe6a2157580",
        "decision": "bf1b2fad2ec355d6eee95172ad6413e971d454a169941f6337d81d5d19e049fe",
        "comparison": "f3eb01610b360d4664993afb1e0a532fdead9f87ca7a8c43c83803bec6dfb8f0",
        "merged": "659a79e1634b919ebcc91f9b56d97cd3fc06e34a8d530d7bfe2c103005b7d25c",
        "ordinal": 15,
        "rule": "3eda063526ee5058343d9341797823ccb181bc131f891794ad44765d17b6571b",
        "scenario": "standard",
        "seed_set": "0300e35328b94a6d7b33cabe52a04182f53bfe3fff26028ed2b2e0adc2f6bac6",
    },
}


class W13OperatorError(RuntimeError):
    """Raised when W1.3 cannot proceed without ambiguity."""


@dataclass(frozen=True)
class W13Layout:
    root: Path

    @property
    def base(self) -> base_operator.RunLayout:
        return base_operator.RunLayout(self.root)

    @property
    def parent(self) -> w12_operator.W12Layout:
        return w12_operator.W12Layout(self.root)

    @property
    def directory(self) -> Path:
        return self.root / "w1.3"

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
    def supported_set(self) -> Path:
        return self.directory / "supported-independent-change-set.json"

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
        raise W13OperatorError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise W13OperatorError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise W13OperatorError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise W13OperatorError(f"{label} must be a positive integer")
    return value


def _require_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise W13OperatorError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise W13OperatorError(f"{label} must be finite")
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return enoch_week1.load_json_object(path)
    except (OSError, enoch_week1.ProtocolError) as exc:
        raise W13OperatorError(f"could not load {path}: {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    except OSError as exc:
        raise W13OperatorError(f"could not hash {path}: {exc}") from exc
    return digest.hexdigest()


def _with_fingerprint(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    frozen = dict(body)
    return {**frozen, field: enoch_week1.canonical_json_sha256(frozen)}


def _validate_fingerprint(value: Mapping[str, Any], field: str, label: str) -> str:
    fingerprint = _require_sha256(value.get(field), f"{label} {field}")
    body = dict(value)
    body.pop(field)
    if enoch_week1.canonical_json_sha256(body) != fingerprint:
        raise W13OperatorError(f"{label} fingerprint mismatch")
    return fingerprint


def _write_or_match(path: Path, value: Mapping[str, Any], label: str) -> None:
    try:
        base_operator._write_or_match(path, value, label)  # noqa: SLF001
    except base_operator.OperatorError as exc:
        raise W13OperatorError(str(exc)) from exc


def _git_text(workspace: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(workspace), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise W13OperatorError(
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
        raise W13OperatorError(
            f"git {' '.join(arguments)} failed: "
            f"{completed.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return completed.stdout


def _git_blob(workspace: Path, revision: str, relative: str) -> str:
    value = _git_text(workspace, "rev-parse", f"{revision}:{relative}").strip()
    if not _GIT_OBJECT_RE.fullmatch(value):
        raise W13OperatorError(f"invalid Git blob for {revision}:{relative}")
    return value


def _validate_rule_shape(rule: Mapping[str, Any], arm_id: str) -> str:
    expected = {
        "maximum_candidate_p95_latency_ms": 750.0,
        "minimum_candidate_completed_worlds_mean": None,
        "minimum_level_utility_estimate": 0.0,
        "minimum_level_utility_lower_95": -0.05,
        "minimum_point_margin_estimate": 0.0,
        "minimum_win_rate_estimate": -0.02,
        "require_zero_invalidating_failures": True,
        "rule_id": f"w1.2-{arm_id}-screen-v1",
        "style_metric_bounds": {},
    }
    if dict(rule) != expected:
        raise W13OperatorError(f"{arm_id} changed its sealed W1.2 development rule")
    return enoch_week1.validate_development_rule(rule)


def validate_committed_plan(plan: Mapping[str, Any]) -> str:
    """Validate every result-independent W1.3 choice."""

    _require_exact_keys(
        plan,
        {
            "arms",
            "automatic_production_promotion_allowed",
            "available_parallelism",
            "baseline_worker_report_fingerprint",
            "failure_disposition",
            "final_consumed_count",
            "fixture_report_fingerprint",
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
            "w1_2_campaign_declaration_fingerprint",
            "w1_2_continuation_provenance_fingerprint",
            "w1_2_ranked_table_fingerprint",
            "w1_2_source_commit",
            "worker_count",
        },
        "committed W1.3 plan",
    )
    if (
        plan["manifest_kind"] != PLAN_KIND
        or plan["manifest_version"] != MANIFEST_VERSION
        or plan["automatic_production_promotion_allowed"] is not False
    ):
        raise W13OperatorError("unsupported committed W1.3 plan")
    exact = {
        "available_parallelism": 10,
        "baseline_worker_report_fingerprint": WORKER_REPORT_FINGERPRINT,
        "failure_disposition": FAILURE_DISPOSITION,
        "final_consumed_count": FINAL_COUNT,
        "fixture_report_fingerprint": FIXTURE_REPORT_FINGERPRINT,
        "pair_count_per_arm": 800,
        "parent_phase_manifest_fingerprint": SEALED_W12_PHASE,
        "preclaim_consumed_count": PRECLAIM_COUNT,
        "preclaim_ledger_fingerprint": SEALED_W12_LEDGER,
        "protocol_fingerprint": PROTOCOL_FINGERPRINT,
        "shard_count": 8,
        "shard_timeout_seconds": 3600,
        "w1_2_campaign_declaration_fingerprint": SEALED_W12_DECLARATION,
        "w1_2_continuation_provenance_fingerprint": SEALED_W12_PROVENANCE,
        "w1_2_ranked_table_fingerprint": SEALED_W12_TABLE,
        "w1_2_source_commit": SEALED_W12_COMMIT,
        "worker_count": 8,
    }
    for field, expected in exact.items():
        if plan[field] != expected:
            raise W13OperatorError(f"committed W1.3 plan changed {field}")
    if tuple(plan["required_style_metrics"]) != enoch_week1.WEEK1_STYLE_METRICS:
        raise W13OperatorError("W1.3 plan changed the frozen style schema")
    fixed = plan["fixed_work"]
    if fixed != {
        "budget_ms": None,
        "candidates": 6,
        "deadline_ms": 30_000,
        "rollout_tricks": 6,
        "work_mode": "fixed-work",
        "worlds": 24,
    }:
        raise W13OperatorError("W1.3 plan changed the 24/6/6 fixed-work contract")
    arms = plan["arms"]
    if not isinstance(arms, list) or len(arms) != len(SURVIVOR_ARM_IDS):
        raise W13OperatorError("W1.3 plan must contain exactly five survivors")
    actual_ids = []
    for arm in arms:
        _require_exact_keys(
            arm,
            {
                "arm_id",
                "candidate_fingerprint",
                "comparison_id",
                "configuration_fingerprint",
                "development_rule",
                "development_rule_sha256",
                "ordinal",
                "scenario_id",
                "seed_namespace",
                "seed_set_sha256",
                "w1_2_advancement_decision_fingerprint",
                "w1_2_comparison_protocol_fingerprint",
                "w1_2_merged_result_fingerprint",
            },
            "W1.3 plan arm",
        )
        arm_id = arm["arm_id"]
        actual_ids.append(arm_id)
        if arm_id not in EXPECTED_PARENT_ARM_FINGERPRINTS:
            raise W13OperatorError(f"unknown W1.3 survivor: {arm_id!r}")
        if arm["comparison_id"] != f"survivor-{arm_id}":
            raise W13OperatorError(f"{arm_id} comparison id is not canonical")
        expected = EXPECTED_PARENT_ARM_FINGERPRINTS[arm_id]
        bindings = {
            "candidate_fingerprint": expected["candidate"],
            "configuration_fingerprint": expected["configuration"],
            "development_rule_sha256": expected["rule"],
            "ordinal": expected["ordinal"],
            "scenario_id": expected["scenario"],
            "seed_namespace": f"dev/survivor/{arm_id}",
            "seed_set_sha256": expected["seed_set"],
            "w1_2_advancement_decision_fingerprint": expected["decision"],
            "w1_2_comparison_protocol_fingerprint": expected["comparison"],
            "w1_2_merged_result_fingerprint": expected["merged"],
        }
        for field, expected_value in bindings.items():
            if arm[field] != expected_value:
                raise W13OperatorError(f"{arm_id} changed sealed parent binding {field}")
        if _validate_rule_shape(arm["development_rule"], arm_id) != arm[
            "development_rule_sha256"
        ]:
            raise W13OperatorError(f"{arm_id} development-rule hash changed")
    if actual_ids != list(SURVIVOR_ARM_IDS):
        raise W13OperatorError("W1.3 survivor list is missing, duplicated, or reordered")
    if tuple(plan["ranking_rule"]) != RANKING_RULE:
        raise W13OperatorError("W1.3 ranking rule changed")
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
        raise W13OperatorError("W1.3 workspace must be the Git root")
    if require_live:
        status = _git_text(
            workspace, "status", "--porcelain=v1", "--untracked-files=all"
        )
        if status:
            raise W13OperatorError(
                f"W1.3 workspace is not clean: {status.splitlines()[0]}"
            )
        head = _git_text(workspace, "rev-parse", "HEAD^{commit}").strip()
        if continuation_commit is not None and continuation_commit != head:
            raise W13OperatorError("requested W1.3 commit differs from live HEAD")
    else:
        if continuation_commit is None:
            raise W13OperatorError("stored W1.3 provenance has no continuation commit")
        head = _git_text(
            workspace, "rev-parse", f"{continuation_commit}^{{commit}}"
        ).strip()
    parents = _git_text(workspace, "rev-list", "--parents", "-n", "1", head).split()
    if parents != [head, BASE_COMMIT]:
        raise W13OperatorError(
            "W1.3 continuation must be one clean single-parent commit directly after bfaaa6e"
        )
    tree = _git_text(workspace, "rev-parse", f"{head}^{{tree}}").strip()
    raw_changes = _git_text(
        workspace,
        "diff-tree",
        "--no-commit-id",
        "--name-status",
        "-r",
        "--no-renames",
        BASE_COMMIT,
        head,
    )
    parsed = []
    for line in raw_changes.splitlines():
        fields = line.split("\t")
        if len(fields) != 2:
            raise W13OperatorError("W1.3 continuation Git diff is malformed")
        parsed.append((fields[0], fields[1]))
    if parsed != [("A", relative) for relative in CONTINUATION_PATHS]:
        raise W13OperatorError(
            "W1.3 continuation must add exactly the committed operator, plan, and tests"
        )
    changes = []
    for status_code, relative in parsed:
        record = _git_text(workspace, "ls-tree", head, "--", relative).strip()
        fields = record.split(None, 3)
        if len(fields) != 4 or fields[0] != "100644" or fields[1] != "blob":
            raise W13OperatorError(f"W1.3 continuation path is not a regular 100644 blob: {relative}")
        blob_bytes = _git_bytes(workspace, "show", f"{head}:{relative}")
        if require_live:
            live_path = workspace / relative
            if not live_path.is_file() or live_path.is_symlink():
                raise W13OperatorError(f"W1.3 continuation path is not a regular file: {relative}")
            if _sha256_file(live_path) != hashlib.sha256(blob_bytes).hexdigest():
                raise W13OperatorError(f"live W1.3 continuation differs from Git: {relative}")
        changes.append(
            {
                "new_blob": fields[2],
                "old_blob": None,
                "path": relative,
                "sha256": hashlib.sha256(blob_bytes).hexdigest(),
                "status": status_code,
            }
        )
    critical = []
    for relative in CRITICAL_BASE_MODULES:
        base_blob = _git_blob(workspace, BASE_COMMIT, relative)
        current_blob = _git_blob(workspace, head, relative)
        if base_blob != current_blob:
            raise W13OperatorError(f"W1.3 continuation changed frozen module: {relative}")
        critical.append({"blob": base_blob, "path": relative})
    return {
        "base_tree_manifest_sha256": hashlib.sha256(
            _git_bytes(workspace, "ls-tree", "-r", "-z", "--full-tree", BASE_COMMIT)
        ).hexdigest(),
        "changed_paths": changes,
        "continuation_git_commit": head,
        "continuation_git_tree": tree,
        "critical_base_module_blobs": critical,
        "git_tree_manifest_sha256": hashlib.sha256(
            _git_bytes(workspace, "ls-tree", "-r", "-z", "--full-tree", head)
        ).hexdigest(),
    }


def _runtime_import_records(workspace: Path) -> list[dict[str, str]]:
    workspace = workspace.resolve()
    records = []
    for module, relative in RUNTIME_MODULES:
        module_path = getattr(module, "__file__", None)
        expected = (workspace / relative).resolve()
        if not isinstance(module_path, str) or Path(module_path).resolve() != expected:
            raise W13OperatorError(f"runtime import is shadowed: {relative}")
        records.append({"path": relative, "sha256": _sha256_file(expected)})
    return records


def build_continuation_provenance(
    layout: W13Layout, workspace: Path, plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind the only permitted three-file W1.3 continuation."""

    validate_committed_plan(plan)
    workspace = workspace.expanduser().resolve()
    if Path(__file__).resolve() != (workspace / OPERATOR_RELATIVE).resolve():
        raise W13OperatorError("executing W1.3 operator is not the committed workspace file")
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
        "parent_phase_manifest_fingerprint": SEALED_W12_PHASE,
        "plan_file_sha256": _sha256_file(workspace / PLAN_RELATIVE),
        "plan_sha256": enoch_week1.canonical_json_sha256(plan),
        "protocol_fingerprint": PROTOCOL_FINGERPRINT,
        "runtime_imports": _runtime_import_records(workspace),
        "sealed_w1_2_campaign_declaration_fingerprint": SEALED_W12_DECLARATION,
        "sealed_w1_2_continuation_provenance_fingerprint": SEALED_W12_PROVENANCE,
        "sealed_w1_2_final_ledger_fingerprint": SEALED_W12_LEDGER,
        "sealed_w1_2_ranked_table_fingerprint": SEALED_W12_TABLE,
        "sealed_w1_2_source_commit": SEALED_W12_COMMIT,
        "starting_consumed_count": PRECLAIM_COUNT,
        "test_source_path": TEST_RELATIVE.as_posix(),
        "test_source_sha256": _sha256_file(workspace / TEST_RELATIVE),
    }
    return _with_fingerprint(body, "continuation_provenance_fingerprint")


def validate_continuation_provenance(
    artifact: Mapping[str, Any],
    layout: W13Layout,
    workspace: Path,
    plan: Mapping[str, Any],
    *,
    live_source: bool,
) -> str:
    validate_committed_plan(plan)
    if live_source:
        expected = build_continuation_provenance(layout, workspace, plan)
        if dict(artifact) != expected:
            raise W13OperatorError("W1.3 continuation provenance does not reconstruct")
        return expected["continuation_provenance_fingerprint"]
    expected_keys = {
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
        "parent_phase_manifest_fingerprint",
        "plan_file_sha256",
        "plan_sha256",
        "protocol_fingerprint",
        "runtime_imports",
        "sealed_w1_2_campaign_declaration_fingerprint",
        "sealed_w1_2_continuation_provenance_fingerprint",
        "sealed_w1_2_final_ledger_fingerprint",
        "sealed_w1_2_ranked_table_fingerprint",
        "sealed_w1_2_source_commit",
        "starting_consumed_count",
        "test_source_path",
        "test_source_sha256",
    }
    _require_exact_keys(artifact, expected_keys, "stored W1.3 provenance")
    fingerprint = _validate_fingerprint(
        artifact, "continuation_provenance_fingerprint", "stored W1.3 provenance"
    )
    exact = {
        "automatic_production_promotion_allowed": False,
        "base_git_commit": BASE_COMMIT,
        "manifest_kind": PROVENANCE_KIND,
        "manifest_version": MANIFEST_VERSION,
        "operator_source_path": OPERATOR_RELATIVE.as_posix(),
        "parent_phase_manifest_fingerprint": SEALED_W12_PHASE,
        "plan_sha256": enoch_week1.canonical_json_sha256(plan),
        "protocol_fingerprint": PROTOCOL_FINGERPRINT,
        "sealed_w1_2_campaign_declaration_fingerprint": SEALED_W12_DECLARATION,
        "sealed_w1_2_continuation_provenance_fingerprint": SEALED_W12_PROVENANCE,
        "sealed_w1_2_final_ledger_fingerprint": SEALED_W12_LEDGER,
        "sealed_w1_2_ranked_table_fingerprint": SEALED_W12_TABLE,
        "sealed_w1_2_source_commit": SEALED_W12_COMMIT,
        "starting_consumed_count": PRECLAIM_COUNT,
        "test_source_path": TEST_RELATIVE.as_posix(),
    }
    for field, expected in exact.items():
        if artifact[field] != expected:
            raise W13OperatorError(f"stored W1.3 provenance changed {field}")
    expected_base_tree = _git_text(
        workspace, "rev-parse", f"{BASE_COMMIT}^{{tree}}"
    ).strip()
    if artifact["base_git_tree"] != expected_base_tree:
        raise W13OperatorError("stored W1.3 base Git tree changed")
    identity = _continuation_git_identity(
        workspace,
        continuation_commit=artifact["continuation_git_commit"],
        require_live=False,
    )
    for field in (
        "base_tree_manifest_sha256",
        "changed_paths",
        "continuation_git_commit",
        "continuation_git_tree",
        "critical_base_module_blobs",
        "git_tree_manifest_sha256",
    ):
        if artifact[field] != identity[field]:
            raise W13OperatorError(f"stored W1.3 Git identity changed {field}")
    commit = artifact["continuation_git_commit"]
    source_hashes = {
        relative: hashlib.sha256(
            _git_bytes(workspace, "show", f"{commit}:{relative}")
        ).hexdigest()
        for relative in CONTINUATION_PATHS
    }
    if artifact["operator_source_sha256"] != source_hashes[OPERATOR_RELATIVE.as_posix()]:
        raise W13OperatorError("stored W1.3 operator source hash changed")
    if artifact["plan_file_sha256"] != source_hashes[PLAN_RELATIVE.as_posix()]:
        raise W13OperatorError("stored W1.3 plan file hash changed")
    if artifact["test_source_sha256"] != source_hashes[TEST_RELATIVE.as_posix()]:
        raise W13OperatorError("stored W1.3 test source hash changed")
    expected_operator_path = (workspace.resolve() / OPERATOR_RELATIVE).resolve()
    if Path(__file__).resolve() != expected_operator_path:
        raise W13OperatorError("executing W1.3 operator is outside the verification workspace")
    if _sha256_file(Path(__file__).resolve()) != artifact["operator_source_sha256"]:
        raise W13OperatorError("executing W1.3 operator differs from stored source")
    expected_runtime = [
        {
            "path": relative,
            "sha256": hashlib.sha256(
                _git_bytes(workspace, "show", f"{commit}:{relative}")
            ).hexdigest(),
        }
        for _module, relative in RUNTIME_MODULES
    ]
    if artifact["runtime_imports"] != expected_runtime:
        raise W13OperatorError("stored W1.3 runtime hashes differ from its Git commit")
    actual_runtime = _runtime_import_records(workspace)
    if actual_runtime != expected_runtime:
        raise W13OperatorError("live W1.3 runtime imports differ from stored Git bytes")
    return fingerprint


def _validate_w1_2_workspace(
    workspace: Path, stored_provenance: Mapping[str, Any]
) -> Path:
    workspace = workspace.expanduser().resolve()
    top = Path(_git_text(workspace, "rev-parse", "--show-toplevel").strip()).resolve()
    if top != workspace:
        raise W13OperatorError("--w1-2-workspace must resolve to its Git root")
    status = _git_text(
        workspace, "status", "--porcelain=v1", "--untracked-files=all"
    )
    if status:
        raise W13OperatorError(
            f"sealed W1.2 workspace is not clean: {status.splitlines()[0]}"
        )
    head = _git_text(workspace, "rev-parse", "HEAD^{commit}").strip()
    if head != SEALED_W12_COMMIT:
        raise W13OperatorError("--w1-2-workspace is not the sealed d8f commit")
    parents = _git_text(workspace, "rev-list", "--parents", "-n", "1", head).split()
    if parents != [SEALED_W12_COMMIT, SEALED_W12_PARENT]:
        raise W13OperatorError("sealed W1.2 commit has an unexpected parent")
    if stored_provenance.get("continuation_provenance_fingerprint") != SEALED_W12_PROVENANCE:
        raise W13OperatorError("stored W1.2 provenance fingerprint changed")
    _validate_fingerprint(
        stored_provenance,
        "continuation_provenance_fingerprint",
        "sealed W1.2 provenance",
    )
    expected_paths = {
        record["path"]: record
        for record in stored_provenance.get("changed_paths", [])
        if isinstance(record, Mapping)
    }
    if set(expected_paths) != {
        "training/enoch_week1_w1_2_operator.py",
        "training/enoch_week1_w1_2_plan.json",
        "training/test_enoch_week1_w1_2_operator.py",
    }:
        raise W13OperatorError("stored W1.2 provenance changed its three-file source set")
    for relative, expected in expected_paths.items():
        path = workspace / relative
        if not path.is_file() or path.is_symlink():
            raise W13OperatorError(f"sealed W1.2 source is not a regular file: {relative}")
        tree = _git_text(workspace, "ls-tree", head, "--", relative).strip().split(None, 3)
        if len(tree) != 4 or tree[0] != "100644" or tree[1] != "blob":
            raise W13OperatorError(f"sealed W1.2 source has wrong mode: {relative}")
        if tree[2] != expected.get("new_blob"):
            raise W13OperatorError(f"sealed W1.2 source blob changed: {relative}")
        digest = _sha256_file(path)
        if digest != expected.get("sha256"):
            raise W13OperatorError(f"sealed W1.2 source SHA-256 changed: {relative}")
        if digest != hashlib.sha256(
            _git_bytes(workspace, "show", f"{head}:{relative}")
        ).hexdigest():
            raise W13OperatorError(f"sealed W1.2 worktree differs from Git: {relative}")
    return workspace


def verify_sealed_w1_2(
    layout: W13Layout, w1_2_workspace: Path, *, timeout_seconds: int = 600
) -> dict[str, Any]:
    """Run the historical W1.2 verifier in an isolated d8 subprocess."""

    stored = _load_json(layout.parent.provenance)
    workspace = _validate_w1_2_workspace(w1_2_workspace, stored)
    operator = workspace / "training/enoch_week1_w1_2_operator.py"
    command = [
        sys.executable,
        "-I",
        "-B",
        "-c",
        W12_RUNPY_BOOTSTRAP,
        str(workspace),
        str(operator),
        "verify-w1.2",
        "--root",
        str(layout.root.resolve()),
        "--workspace",
        str(workspace),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=workspace,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise W13OperatorError(f"isolated W1.2 verification failed: {exc}") from exc
    _validate_w1_2_workspace(workspace, stored)
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise W13OperatorError(
            f"isolated W1.2 verifier exited {completed.returncode}: {detail}"
        )
    if completed.stderr:
        raise W13OperatorError("isolated W1.2 verifier emitted stderr")
    if len(completed.stdout) > 1024 * 1024:
        raise W13OperatorError("isolated W1.2 verifier output is too large")
    try:
        result = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise W13OperatorError(
            f"isolated W1.2 verifier did not emit one JSON object: {exc}"
        ) from exc
    if result != EXPECTED_W12_VERIFY_RESULT:
        raise W13OperatorError("isolated W1.2 verification result changed")
    return result


def _w12_arm_root(layout: W13Layout, arm_id: str) -> Path:
    ordinal = EXPECTED_PARENT_ARM_FINGERPRINTS[arm_id]["ordinal"]
    return layout.parent.directory / "arms" / f"{ordinal:02d}-{arm_id}"


def _sole_completed_attempt(arm_root: Path, label: str) -> Path:
    attempts_root = arm_root / "attempts"
    if not attempts_root.is_dir():
        raise W13OperatorError(f"{label} has no attempt directory")
    completed = []
    for path in attempts_root.iterdir():
        if not path.is_dir() or not _ATTEMPT_RE.fullmatch(path.name):
            raise W13OperatorError(f"malformed {label} attempt entry: {path}")
        if (path / "execution" / "execution-complete.json").is_file():
            completed.append(path)
    if len(completed) != 1:
        raise W13OperatorError(f"{label} must have exactly one completed attempt")
    return completed[0]


def _load_parent_arm(
    layout: W13Layout,
    protocol: Mapping[str, Any],
    fixture: Mapping[str, Any],
    table: Mapping[str, Any],
    plan_arm: Mapping[str, Any],
) -> dict[str, Any]:
    arm_id = plan_arm["arm_id"]
    root = _w12_arm_root(layout, arm_id)
    declaration = root / "declaration"
    attempt = _sole_completed_attempt(root, f"sealed W1.2 {arm_id}")
    comparison = _load_json(declaration / "comparison.json")
    launch = _load_json(declaration / "launch.json")
    identities = _load_json(declaration / "identities.json")
    environment_identity = _load_json(declaration / "environment-identity.json")
    rule = _load_json(declaration / "development-rule.json")
    merged = _load_json(attempt / "execution" / "merged-result.json")
    decision = _load_json(root / "advancement-decision.json")

    comparison_fingerprint = enoch_week1.validate_comparison_protocol_manifest(
        protocol, comparison
    )
    if comparison["phase"] != "W1.2" or comparison["subject_id"] != arm_id:
        raise W13OperatorError(f"sealed W1.2 {arm_id} comparison identity changed")
    launch_fingerprint = enoch_week1_runner.validate_launch_configuration(launch)
    if launch_fingerprint != comparison["configuration_fingerprint"]:
        raise W13OperatorError(f"sealed W1.2 {arm_id} launch changed")
    expected_identities = enoch_week1_runner.build_in_process_identity_bindings(
        identities["evaluator"], launch
    )
    if identities != expected_identities:
        raise W13OperatorError(f"sealed W1.2 {arm_id} identities changed")
    for name in ("candidate", "control", "evaluator"):
        if enoch_week1.canonical_json_sha256(identities[name]) != comparison[
            f"{name}_fingerprint"
        ]:
            raise W13OperatorError(f"sealed W1.2 {arm_id} {name} identity changed")
    if enoch_week1.canonical_json_sha256(environment_identity) != ENVIRONMENT_FINGERPRINT:
        raise W13OperatorError(f"sealed W1.2 {arm_id} environment changed")
    if comparison["environment_fingerprint"] != ENVIRONMENT_FINGERPRINT:
        raise W13OperatorError(f"sealed W1.2 {arm_id} environment binding changed")
    if rule != comparison["development_rule"] or rule != plan_arm["development_rule"]:
        raise W13OperatorError(f"sealed W1.2 {arm_id} development rule changed")
    merged_fingerprint = enoch_week1.validate_merged_result(protocol, comparison, merged)
    decision_fingerprint = enoch_week1.validate_w1_3_advancement_decision(
        protocol, comparison, merged, fixture, decision
    )
    if decision["decision"] != "advance-to-w1.3":
        raise W13OperatorError(f"sealed W1.2 {arm_id} did not advance")
    expected = EXPECTED_PARENT_ARM_FINGERPRINTS[arm_id]
    observed = {
        "candidate": comparison["candidate_fingerprint"],
        "configuration": launch_fingerprint,
        "decision": decision_fingerprint,
        "comparison": comparison_fingerprint,
        "merged": merged_fingerprint,
        "rule": enoch_week1.canonical_json_sha256(rule),
    }
    for field, value in observed.items():
        if value != expected[field]:
            raise W13OperatorError(f"sealed W1.2 {arm_id} changed {field}")
    records = [
        record for record in table["arm_results"] if record.get("arm_id") == arm_id
    ]
    if len(records) != 1:
        raise W13OperatorError(f"ranked W1.2 table does not name {arm_id} exactly once")
    record = records[0]
    table_bindings = {
        "advancement_decision_fingerprint": decision_fingerprint,
        "comparison_protocol_fingerprint": comparison_fingerprint,
        "merged_result_fingerprint": merged_fingerprint,
    }
    for field, value in table_bindings.items():
        if record.get(field) != value:
            raise W13OperatorError(f"ranked W1.2 table changed {arm_id} {field}")
    return {
        "advancement_decision": decision,
        "arm_id": arm_id,
        "comparison": comparison,
        "environment_identity": environment_identity,
        "identity_bindings": identities,
        "launch_configuration": launch,
        "merged_result": merged,
        "rule": rule,
    }


def _load_parent_state(
    layout: W13Layout, plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Re-read every sealed W1.2 input after acquiring the run lock."""

    validate_committed_plan(plan)
    protocol = _load_json(layout.base.protocol)
    if enoch_week1.validate_protocol(protocol) != PROTOCOL_FINGERPRINT:
        raise W13OperatorError("authoritative protocol changed")
    control = _load_json(layout.base.control_bundle / "control-manifest.json")
    enoch_week1.validate_w1_0_control_manifest(protocol, control)
    report = _load_json(layout.base.report)
    if report.get("baseline_worker_report_fingerprint") != WORKER_REPORT_FINGERPRINT:
        raise W13OperatorError("W1.1 baseline/worker report changed")
    fixed = report.get("fixed_worker_configuration")
    if not isinstance(fixed, Mapping) or (
        fixed.get("maximum_parallel_workers") != 8
        or fixed.get("shard_count") != 8
        or fixed.get("environment_fingerprint") != ENVIRONMENT_FINGERPRINT
    ):
        raise W13OperatorError("W1.1 fixed worker configuration changed")
    fixture = base_operator._validate_fixture_gate(layout.base)  # noqa: SLF001
    if fixture["fixture_report_fingerprint"] != FIXTURE_REPORT_FINGERPRINT:
        raise W13OperatorError("sealed fixture report changed")
    phase0 = _load_json(layout.base.phase0)
    phase1 = _load_json(layout.base.phase1)
    phase2 = _load_json(layout.parent.phase)
    phase_fingerprints = enoch_week1.validate_phase_chain(
        protocol, [phase0, phase1, phase2]
    )
    if phase_fingerprints[1] != SEALED_W11_PHASE or phase_fingerprints[2] != SEALED_W12_PHASE:
        raise W13OperatorError("sealed W1.1/W1.2 phase chain changed")
    parent_provenance = _load_json(layout.parent.provenance)
    if _validate_fingerprint(
        parent_provenance,
        "continuation_provenance_fingerprint",
        "sealed W1.2 provenance",
    ) != SEALED_W12_PROVENANCE:
        raise W13OperatorError("sealed W1.2 continuation provenance changed")
    parent_declaration = _load_json(layout.parent.declaration)
    if _validate_fingerprint(
        parent_declaration,
        "campaign_declaration_fingerprint",
        "sealed W1.2 declaration",
    ) != SEALED_W12_DECLARATION:
        raise W13OperatorError("sealed W1.2 campaign declaration changed")
    table = _load_json(layout.parent.ranked_table)
    if _validate_fingerprint(
        table,
        "ranked_independent_ablation_table_fingerprint",
        "sealed W1.2 ranked table",
    ) != SEALED_W12_TABLE:
        raise W13OperatorError("sealed W1.2 ranked table changed")
    if table.get("summary", {}).get("advancing_arm_ids") != list(SURVIVOR_ARM_IDS):
        raise W13OperatorError("sealed W1.2 survivor set changed")
    parent_ledger = _load_json(layout.parent.final_ledger)
    enoch_week1.validate_seed_ledger(protocol, parent_ledger)
    if (
        parent_ledger["ledger_fingerprint"] != SEALED_W12_LEDGER
        or len(parent_ledger["consumed"]) != PRECLAIM_COUNT
    ):
        raise W13OperatorError("sealed W1.2 final ledger changed")
    live_ledger = _load_json(layout.base.ledger)
    enoch_week1.validate_seed_ledger(protocol, live_ledger)
    parent_arms = {
        plan_arm["arm_id"]: _load_parent_arm(
            layout, protocol, fixture, table, plan_arm
        )
        for plan_arm in plan["arms"]
    }
    environment_identities = {
        enoch_week1.canonical_json_sha256(arm["environment_identity"])
        for arm in parent_arms.values()
    }
    if environment_identities != {ENVIRONMENT_FINGERPRINT}:
        raise W13OperatorError("W1.2 survivors do not share the sealed environment")
    return {
        "control": control,
        "fixture": fixture,
        "live_ledger": live_ledger,
        "parent_arms": parent_arms,
        "parent_declaration": parent_declaration,
        "parent_ledger": parent_ledger,
        "parent_provenance": parent_provenance,
        "phase0": phase0,
        "phase1": phase1,
        "phase2": phase2,
        "protocol": protocol,
        "report": report,
        "table": table,
    }


def _root_bindings(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "baseline_worker_report_fingerprint": WORKER_REPORT_FINGERPRINT,
        "control_manifest_fingerprint": state["control"]["control_manifest_fingerprint"],
        "fixture_report_fingerprint": FIXTURE_REPORT_FINGERPRINT,
        "preclaim_consumed_count": PRECLAIM_COUNT,
        "preclaim_ledger_fingerprint": SEALED_W12_LEDGER,
        "runtime_control_fingerprint": state["report"][
            "runtime_evaluation_control_fingerprint"
        ],
        "runtime_evaluator_fingerprint": state["report"][
            "runtime_evaluator_fingerprint"
        ],
        "w1_2_campaign_declaration_fingerprint": SEALED_W12_DECLARATION,
        "w1_2_continuation_provenance_fingerprint": SEALED_W12_PROVENANCE,
        "w1_2_final_ledger_fingerprint": SEALED_W12_LEDGER,
        "w1_2_ranked_table_fingerprint": SEALED_W12_TABLE,
    }


def build_campaign_declaration(
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
    provenance: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    environment: Mapping[str, str] | None = None,
    environment_identity_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build all five result-independent W1.3 comparisons at once."""

    validate_committed_plan(plan)
    enoch_week1.validate_protocol(protocol)
    if (environment is None) == (environment_identity_override is None):
        raise W13OperatorError(
            "choose exactly one live environment or frozen environment identity"
        )
    if environment_identity_override is None:
        child_environment, _ = enoch_week1.sanitized_evaluator_environment(
            environment,
            allowlist=protocol["evaluator_environment_policy"]["allowlist"],
        )
        environment_identity = enoch_week1_runner.build_evaluator_environment_identity(
            state["control"]["evaluator_identity"],
            protocol,
            child_environment,
            available_parallelism=plan["available_parallelism"],
        )
    else:
        environment_identity = dict(environment_identity_override)
    environment_fingerprint = enoch_week1.canonical_json_sha256(environment_identity)
    if environment_fingerprint != ENVIRONMENT_FINGERPRINT:
        raise W13OperatorError("W1.3 environment differs from sealed W1.2")

    arms = []
    for sequence, plan_arm in enumerate(plan["arms"], start=1):
        arm_id = plan_arm["arm_id"]
        parent = state["parent_arms"][arm_id]
        if parent["environment_identity"] != environment_identity:
            raise W13OperatorError(f"{arm_id} changed its sealed environment identity")
        launch = copy.deepcopy(parent["launch_configuration"])
        identities = copy.deepcopy(parent["identity_bindings"])
        rule = copy.deepcopy(parent["rule"])
        if enoch_week1_runner.validate_launch_configuration(launch) != plan_arm[
            "configuration_fingerprint"
        ]:
            raise W13OperatorError(f"{arm_id} changed its sealed launch")
        if launch["scenario_id"] != plan_arm["scenario_id"]:
            raise W13OperatorError(f"{arm_id} scenario changed")
        comparison = enoch_week1.build_comparison_protocol_manifest(
            protocol,
            phase="W1.3",
            comparison_id=plan_arm["comparison_id"],
            subject_id=arm_id,
            seed_namespace=plan_arm["seed_namespace"],
            pair_count=plan["pair_count_per_arm"],
            shard_count=plan["shard_count"],
            candidate_fingerprint=enoch_week1.canonical_json_sha256(
                identities["candidate"]
            ),
            control_fingerprint=enoch_week1.canonical_json_sha256(
                identities["control"]
            ),
            evaluator_fingerprint=enoch_week1.canonical_json_sha256(
                identities["evaluator"]
            ),
            environment_fingerprint=environment_fingerprint,
            configuration_fingerprint=plan_arm["configuration_fingerprint"],
            development_rule=rule,
            required_style_metrics=enoch_week1.WEEK1_STYLE_METRICS,
        )
        if comparison["seed_set_sha256"] != plan_arm["seed_set_sha256"]:
            raise W13OperatorError(f"{arm_id} 800-seed set changed")
        shard_sizes = [len(shard["seed_indices"]) for shard in comparison["shards"]]
        if shard_sizes != [100] * 8:
            raise W13OperatorError(f"{arm_id} is not partitioned into eight 100-pair shards")
        arms.append(
            {
                "arm_id": arm_id,
                "comparison": comparison,
                "development_rule": rule,
                "development_rule_sha256": plan_arm["development_rule_sha256"],
                "environment_identity": copy.deepcopy(environment_identity),
                "identity_bindings": identities,
                "launch_configuration": launch,
                "launch_configuration_sha256": plan_arm["configuration_fingerprint"],
                "ordinal": plan_arm["ordinal"],
                "parent_evidence": {
                    "advancement_decision_fingerprint": plan_arm[
                        "w1_2_advancement_decision_fingerprint"
                    ],
                    "comparison_protocol_fingerprint": plan_arm[
                        "w1_2_comparison_protocol_fingerprint"
                    ],
                    "merged_result_fingerprint": plan_arm[
                        "w1_2_merged_result_fingerprint"
                    ],
                },
                "scenario_id": plan_arm["scenario_id"],
                "seed_namespace": plan_arm["seed_namespace"],
                "sequence": sequence,
            }
        )
    body = {
        "arm_registry_sha256": enoch_week1.ARM_REGISTRY_SHA256,
        "arms": arms,
        "automatic_production_promotion_allowed": False,
        "execution_contract": {
            "available_parallelism": 10,
            "execution_order": list(SURVIVOR_ARM_IDS),
            "failure_disposition": FAILURE_DISPOSITION,
            "final_consumed_count": FINAL_COUNT,
            "fixed_work": copy.deepcopy(plan["fixed_work"]),
            "pair_count": 800,
            "ranking_rule": list(RANKING_RULE),
            "require_machine_global_lock": True,
            "required_external_artifact_ids": list(
                enoch_week1_evidence.EXPECTED_ARTIFACT_IDS
            ),
            "required_failure_counters": list(enoch_week1.FAILURE_COUNTER_NAMES),
            "required_style_metrics": list(enoch_week1.WEEK1_STYLE_METRICS),
            "shard_count": 8,
            "shard_pair_counts": [100] * 8,
            "timeout_seconds": 3600,
            "worker_count": 8,
        },
        "fixture_binding": {
            "failure_count": 0,
            "fixture_report_fingerprint": FIXTURE_REPORT_FINGERPRINT,
            "record_count": len(state["fixture"]["records"]),
            "source_files_sha256": state["fixture"]["source_files_sha256"],
        },
        "manifest_kind": DECLARATION_KIND,
        "manifest_version": MANIFEST_VERSION,
        "operator_source_provenance_fingerprint": provenance[
            "continuation_provenance_fingerprint"
        ],
        "parent_phase": {
            "phase": "W1.2",
            "phase_manifest_fingerprint": SEALED_W12_PHASE,
        },
        "plan_sha256": enoch_week1.canonical_json_sha256(plan),
        "protocol_fingerprint": PROTOCOL_FINGERPRINT,
        "root_bindings": _root_bindings(state),
        "seed_registry_sha256": protocol["seed_registry_sha256"],
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
        raise W13OperatorError("W1.3 campaign declaration does not reconstruct")
    return expected["campaign_declaration_fingerprint"]


def _arm_by_id(declaration: Mapping[str, Any], arm_id: str) -> Mapping[str, Any]:
    matches = [arm for arm in declaration["arms"] if arm.get("arm_id") == arm_id]
    if len(matches) != 1:
        raise W13OperatorError(f"W1.3 declaration does not name {arm_id} exactly once")
    return matches[0]


def _materialize_declaration(
    layout: W13Layout,
    plan: Mapping[str, Any],
    provenance: Mapping[str, Any],
    declaration: Mapping[str, Any],
) -> None:
    _write_or_match(layout.input, plan, "W1.3 committed plan copy")
    _write_or_match(layout.provenance, provenance, "W1.3 continuation provenance")
    for arm in declaration["arms"]:
        root = layout.arm(arm["sequence"], arm["arm_id"]) / "declaration"
        artifacts = {
            "comparison.json": arm["comparison"],
            "development-rule.json": arm["development_rule"],
            "environment-identity.json": arm["environment_identity"],
            "identities.json": arm["identity_bindings"],
            "launch.json": arm["launch_configuration"],
            "parent-evidence.json": arm["parent_evidence"],
        }
        for name, value in artifacts.items():
            _write_or_match(root / name, value, f"{arm['arm_id']} {name}")
    # Commit marker: write only after all five declaration directories match.
    _write_or_match(layout.declaration, declaration, "W1.3 declaration index")


def _verify_materialized_declaration(
    layout: W13Layout,
    plan: Mapping[str, Any],
    provenance: Mapping[str, Any],
    declaration: Mapping[str, Any],
) -> None:
    if _load_json(layout.input) != dict(plan):
        raise W13OperatorError("materialized W1.3 plan changed")
    if _load_json(layout.provenance) != dict(provenance):
        raise W13OperatorError("materialized W1.3 provenance changed")
    for arm in declaration["arms"]:
        root = layout.arm(arm["sequence"], arm["arm_id"]) / "declaration"
        expected = {
            "comparison.json": arm["comparison"],
            "development-rule.json": arm["development_rule"],
            "environment-identity.json": arm["environment_identity"],
            "identities.json": arm["identity_bindings"],
            "launch.json": arm["launch_configuration"],
            "parent-evidence.json": arm["parent_evidence"],
        }
        for name, value in expected.items():
            if _load_json(root / name) != value:
                raise W13OperatorError(f"materialized {arm['arm_id']} {name} changed")
    if _load_json(layout.declaration) != dict(declaration):
        raise W13OperatorError("materialized W1.3 declaration index changed")


def _require_exact_preclaim_ledger(state: Mapping[str, Any]) -> None:
    if state["live_ledger"] != state["parent_ledger"]:
        raise W13OperatorError(
            "W1.3 declaration requires the byte-semantic 14,711-claim W1.2 ledger"
        )


def declare_w1_3(
    layout: W13Layout,
    workspace: Path,
    w1_2_workspace: Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Freeze all five W1.3 comparisons before any survivor seed claim."""

    workspace = workspace.expanduser().resolve()
    base_environment = dict(os.environ if environment is None else environment)
    verify_sealed_w1_2(layout, w1_2_workspace)
    with base_operator._operator_lock(layout.base.operator_lock):  # noqa: SLF001
        plan = load_committed_plan(workspace)
        state = _load_parent_state(layout, plan)
        _require_exact_preclaim_ledger(state)
        if layout.retirement.exists():
            raise W13OperatorError("this protocol is already retired for W1.3")
        provenance = build_continuation_provenance(layout, workspace, plan)
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
            "continuation_provenance_fingerprint": provenance[
                "continuation_provenance_fingerprint"
            ],
            "pair_count_per_arm": 800,
            "planned_pair_count": 4_000,
            "protocol_fingerprint": PROTOCOL_FINGERPRINT,
            "survivor_arm_ids": list(SURVIVOR_ARM_IDS),
        }


def _nonzero_failure_counters(metrics: Mapping[str, Any]) -> list[str]:
    counters = metrics.get("failure_counters")
    if not isinstance(counters, Mapping) or set(counters) != set(
        enoch_week1.FAILURE_COUNTER_NAMES
    ):
        raise W13OperatorError("W1.3 metrics changed the invalidating-counter schema")
    return [name for name in enoch_week1.FAILURE_COUNTER_NAMES if counters[name] != 0]


def _support_decision_body(
    protocol: Mapping[str, Any],
    arm: Mapping[str, Any],
    merged: Mapping[str, Any],
    fixture: Mapping[str, Any],
) -> dict[str, Any]:
    comparison = arm["comparison"]
    comparison_fingerprint = enoch_week1.validate_comparison_protocol_manifest(
        protocol, comparison
    )
    if comparison["phase"] != "W1.3":
        raise W13OperatorError("W1.4 support decision requires a W1.3 result")
    merged_fingerprint = enoch_week1.validate_merged_result(
        protocol, comparison, merged
    )
    try:
        fixture_fingerprint = enoch_week1_fixtures.validate_report(fixture)
    except enoch_week1_fixtures.FixtureError as exc:
        raise W13OperatorError(f"invalid fixture report for W1.3 decision: {exc}") from exc
    if fixture_fingerprint != FIXTURE_REPORT_FINGERPRINT or fixture["failure_count"] != 0:
        raise W13OperatorError("W1.3 decision requires the sealed zero-failure fixture gate")
    rule = arm["development_rule"]
    if rule != comparison["development_rule"]:
        raise W13OperatorError("W1.3 comparison changed its carried development rule")
    if _nonzero_failure_counters(merged["metrics"]):
        raise W13OperatorError("nonzero invalidating counter cannot be a valid W1.3 completion")
    failures = enoch_week1._development_rule_failures(  # noqa: SLF001
        rule, merged["metrics"]
    )
    parent = arm["parent_evidence"]
    return {
        "advancement_decision_fingerprint": parent[
            "advancement_decision_fingerprint"
        ],
        "arm_id": arm["arm_id"],
        "automatic_production_promotion_allowed": False,
        "decision": "advance-to-w1.4" if not failures else "stop-and-record",
        "development_rule_sha256": arm["development_rule_sha256"],
        "fixture_report_fingerprint": fixture_fingerprint,
        "manifest_kind": SUPPORT_DECISION_KIND,
        "manifest_version": MANIFEST_VERSION,
        "metrics": copy.deepcopy(merged["metrics"]),
        "protocol_fingerprint": PROTOCOL_FINGERPRINT,
        "reasons": failures or ["carried-development-rule-met-and-correctness-clean"],
        "w1_2_comparison_protocol_fingerprint": parent[
            "comparison_protocol_fingerprint"
        ],
        "w1_2_merged_result_fingerprint": parent["merged_result_fingerprint"],
        "w1_3_comparison_protocol_fingerprint": comparison_fingerprint,
        "w1_3_merged_result_fingerprint": merged_fingerprint,
    }


def build_support_decision(
    protocol: Mapping[str, Any],
    arm: Mapping[str, Any],
    merged: Mapping[str, Any],
    fixture: Mapping[str, Any],
) -> dict[str, Any]:
    return _with_fingerprint(
        _support_decision_body(protocol, arm, merged, fixture),
        "support_decision_fingerprint",
    )


def validate_support_decision(
    decision: Mapping[str, Any],
    protocol: Mapping[str, Any],
    arm: Mapping[str, Any],
    merged: Mapping[str, Any],
    fixture: Mapping[str, Any],
) -> str:
    expected = build_support_decision(protocol, arm, merged, fixture)
    if dict(decision) != expected:
        raise W13OperatorError(f"{arm['arm_id']} support decision does not reconstruct")
    return expected["support_decision_fingerprint"]


def _rank_key(
    result: Mapping[str, Any], ordinals: Mapping[str, int]
) -> tuple[Any, ...]:
    metrics = result["merged_result"]["metrics"]
    return (
        0 if result["support_decision"]["decision"] == "advance-to-w1.4" else 1,
        -_require_finite(
            metrics["level_utility"]["paired_bootstrap_lower_95"],
            "level utility lower bound",
        ),
        -_require_finite(metrics["level_utility"]["estimate"], "level utility"),
        -_require_finite(metrics["point_margin_estimate"], "point margin"),
        -_require_finite(metrics["win_rate_estimate"], "win rate"),
        ordinals[result["arm_id"]],
    )


def _validate_supported_final_ledger(
    protocol: Mapping[str, Any],
    declaration: Mapping[str, Any],
    ledger: Mapping[str, Any],
) -> str:
    enoch_week1.validate_seed_ledger(protocol, ledger)
    if len(ledger["consumed"]) != FINAL_COUNT:
        raise W13OperatorError("supported set requires the exact 18,711-claim ledger")
    prefix_fingerprint = declaration["root_bindings"]["preclaim_ledger_fingerprint"]
    prefix_body = {
        key: value for key, value in ledger.items() if key != "ledger_fingerprint"
    }
    prefix_body["consumed"] = list(ledger["consumed"][:PRECLAIM_COUNT])
    if enoch_week1.canonical_json_sha256(prefix_body) != prefix_fingerprint:
        raise W13OperatorError("supported set final ledger changed its W1.2 prefix")
    for arm_id in SURVIVOR_ARM_IDS:
        _validate_claims(ledger, _arm_by_id(declaration, arm_id)["comparison"])
    expected_namespaces = {
        _arm_by_id(declaration, arm_id)["seed_namespace"]
        for arm_id in SURVIVOR_ARM_IDS
    }
    if any(
        row["namespace"].startswith("dev/survivor/")
        and row["namespace"] not in expected_namespaces
        for row in ledger["consumed"]
    ):
        raise W13OperatorError("supported set ledger contains an undeclared survivor")
    return ledger["ledger_fingerprint"]


def build_supported_change_set(
    protocol: Mapping[str, Any],
    declaration: Mapping[str, Any],
    fixture: Mapping[str, Any],
    ledger: Mapping[str, Any],
    arm_evidence: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Seal all five W1.3 outcomes, including an explicit no-survivor branch."""

    if list(arm_evidence) != list(SURVIVOR_ARM_IDS):
        raise W13OperatorError("supported set requires all five results in canonical order")
    final_ledger_fingerprint = _validate_supported_final_ledger(
        protocol, declaration, ledger
    )
    results = []
    ordinals = {}
    aggregate = {name: 0 for name in enoch_week1.FAILURE_COUNTER_NAMES}
    for arm_id in SURVIVOR_ARM_IDS:
        arm = _arm_by_id(declaration, arm_id)
        evidence = arm_evidence[arm_id]
        if evidence["arm_id"] != arm_id:
            raise W13OperatorError("W1.3 result arm identity changed")
        merged = evidence["merged_result"]
        merged_fingerprint = enoch_week1.validate_merged_result(
            protocol, arm["comparison"], merged
        )
        decision_fingerprint = validate_support_decision(
            evidence["support_decision"], protocol, arm, merged, fixture
        )
        for name, count in merged["metrics"]["failure_counters"].items():
            aggregate[name] += count
        ordinals[arm_id] = arm["ordinal"]
        results.append(
            {
                "arm_id": arm_id,
                "candidate_fingerprint": arm["comparison"]["candidate_fingerprint"],
                "comparison_protocol_fingerprint": arm["comparison"][
                    "comparison_protocol_fingerprint"
                ],
                "decision": evidence["support_decision"]["decision"],
                "environment_fingerprint": arm["comparison"]["environment_fingerprint"],
                "external_evidence_fingerprint": evidence[
                    "external_evidence_fingerprint"
                ],
                "machine_attestation_fingerprint": evidence[
                    "machine_attestation_fingerprint"
                ],
                "merged_result_fingerprint": merged_fingerprint,
                "metrics": copy.deepcopy(merged["metrics"]),
                "ordinal": arm["ordinal"],
                "raw_output_sha256s": copy.deepcopy(evidence["raw_output_sha256s"]),
                "reasons": copy.deepcopy(evidence["support_decision"]["reasons"]),
                "runner_execution_sha256": enoch_week1.canonical_json_sha256(
                    evidence["runner_execution"]
                ),
                "shard_result_fingerprints": [
                    shard["shard_result_fingerprint"]
                    for shard in evidence["shard_results"]
                ],
                "support_decision_fingerprint": decision_fingerprint,
                "w1_2_parent_evidence": copy.deepcopy(arm["parent_evidence"]),
            }
        )
    evidence_by_id = {result["arm_id"]: arm_evidence[result["arm_id"]] for result in results}
    ranking = sorted(
        list(SURVIVOR_ARM_IDS),
        key=lambda arm_id: _rank_key(evidence_by_id[arm_id], ordinals),
    )
    supported = [
        arm_id
        for arm_id in SURVIVOR_ARM_IDS
        if arm_evidence[arm_id]["support_decision"]["decision"] == "advance-to-w1.4"
    ]
    stopped = [arm_id for arm_id in SURVIVOR_ARM_IDS if arm_id not in supported]
    body = {
        "arm_registry_sha256": enoch_week1.ARM_REGISTRY_SHA256,
        "arm_results": results,
        "automatic_production_promotion_allowed": False,
        "campaign_declaration_fingerprint": declaration[
            "campaign_declaration_fingerprint"
        ],
        "fixture_report_fingerprint": FIXTURE_REPORT_FINGERPRINT,
        "final_consumed_count": FINAL_COUNT,
        "final_ledger_fingerprint": final_ledger_fingerprint,
        "manifest_kind": SUPPORTED_SET_KIND,
        "manifest_version": MANIFEST_VERSION,
        "parent_phase_manifest_fingerprint": SEALED_W12_PHASE,
        "protocol_fingerprint": PROTOCOL_FINGERPRINT,
        "ranking": ranking,
        "ranking_rule": list(RANKING_RULE),
        "seed_registry_sha256": protocol["seed_registry_sha256"],
        "status": "supported-survivors" if supported else "no-survivor",
        "summary": {
            "aggregate_failure_counters": aggregate,
            "arm_count": 5,
            "pair_count_per_arm": 800,
            "stop_count": len(stopped),
            "stopped_arm_ids": stopped,
            "supported_arm_count": len(supported),
            "supported_arm_ids": supported,
            "total_pair_count": 4_000,
        },
        "w1_2_final_ledger_fingerprint": SEALED_W12_LEDGER,
        "w1_2_ranked_table_fingerprint": SEALED_W12_TABLE,
    }
    return _with_fingerprint(body, "supported_change_set_fingerprint")


def validate_supported_change_set(
    artifact: Mapping[str, Any],
    protocol: Mapping[str, Any],
    declaration: Mapping[str, Any],
    fixture: Mapping[str, Any],
    ledger: Mapping[str, Any],
    arm_evidence: Mapping[str, Mapping[str, Any]],
) -> str:
    expected = build_supported_change_set(
        protocol, declaration, fixture, ledger, arm_evidence
    )
    if dict(artifact) != expected:
        raise W13OperatorError("supported-independent-change-set does not reconstruct")
    return expected["supported_change_set_fingerprint"]


def _namespace_claims(
    ledger: Mapping[str, Any], namespace: str
) -> list[Mapping[str, Any]]:
    return [row for row in ledger["consumed"] if row["namespace"] == namespace]


def _expected_claim_consumers(comparison: Mapping[str, Any]) -> dict[int, str]:
    prefix = comparison["comparison_protocol_fingerprint"][:16]
    result = {}
    for shard in comparison["shards"]:
        consumer = f"runner:{prefix}:{shard['shard_id']}"
        for index in shard["seed_indices"]:
            result[index] = consumer
    return result


def _validate_claims(
    ledger: Mapping[str, Any], comparison: Mapping[str, Any]
) -> None:
    actual = {
        row["index"]: row["consumer"]
        for row in _namespace_claims(ledger, comparison["seed_namespace"])
    }
    if actual != _expected_claim_consumers(comparison):
        raise W13OperatorError(
            f"{comparison['seed_namespace']} claims do not match the completed run"
        )


def _validate_parent_prefix(
    ledger: Mapping[str, Any], parent_ledger: Mapping[str, Any]
) -> None:
    prefix_body = {key: value for key, value in ledger.items() if key != "ledger_fingerprint"}
    prefix_body["consumed"] = list(ledger["consumed"][:PRECLAIM_COUNT])
    if (
        enoch_week1.canonical_json_sha256(prefix_body) != SEALED_W12_LEDGER
        or parent_ledger["ledger_fingerprint"] != SEALED_W12_LEDGER
        or list(ledger["consumed"][:PRECLAIM_COUNT]) != parent_ledger["consumed"]
    ):
        raise W13OperatorError("the sealed W1.2 ledger prefix changed")


def _validate_claim_frontier(
    ledger: Mapping[str, Any],
    parent_ledger: Mapping[str, Any],
    declaration: Mapping[str, Any],
    completed_arm_ids: Sequence[str],
) -> None:
    _validate_parent_prefix(ledger, parent_ledger)
    expected = {}
    for arm_id in completed_arm_ids:
        comparison = _arm_by_id(declaration, arm_id)["comparison"]
        for index, consumer in _expected_claim_consumers(comparison).items():
            expected[(comparison["seed_namespace"], index)] = consumer
    actual = {
        (row["namespace"], row["index"]): row["consumer"]
        for row in ledger["consumed"][PRECLAIM_COUNT:]
    }
    if actual != expected:
        raise W13OperatorError(
            "ledger frontier contains future, downstream, missing, or undeclared claims"
        )


def _attempts(arm_root: Path) -> list[Path]:
    root = arm_root / "attempts"
    if not root.exists():
        return []
    attempts = []
    for path in root.iterdir():
        if not path.is_dir() or not _ATTEMPT_RE.fullmatch(path.name):
            raise W13OperatorError(f"malformed W1.3 attempt entry: {path}")
        attempts.append(path)
    return sorted(attempts)


def _attempt_entries_tolerant(arm_root: Path) -> list[Path]:
    """List raw attempt entries after retirement is already mandatory.

    Retirement must remain writable even when the attempt directory itself is
    the malformed evidence.  No validation or filtering is performed here.
    """

    root = arm_root / "attempts"
    if not root.exists() or not root.is_dir():
        return []
    return sorted(root.iterdir(), key=lambda path: path.name)


def _last_tombstone_directory(entries: Sequence[Path]) -> Path | None:
    directories = [path for path in entries if path.is_dir() and not path.is_symlink()]
    return directories[-1] if directories else None


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
        raise W13OperatorError(f"multiple completed attempts exist under {arm_root}")
    return completed[0] if completed else None


def _retirement_body(
    protocol: Mapping[str, Any],
    ledger: Mapping[str, Any],
    arm: Mapping[str, Any],
    arm_root: Path,
    *,
    reason: str,
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
        "attempt_directories": [
            path.name for path in _attempt_entries_tolerant(arm_root)
        ],
        "automatic_production_promotion_allowed": False,
        "claims": claims,
        "disposition": FAILURE_DISPOSITION,
        "ledger_fingerprint": ledger["ledger_fingerprint"],
        "manifest_kind": RETIREMENT_KIND,
        "manifest_version": MANIFEST_VERSION,
        "phase": "W1.3",
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "reason": reason,
        "seed_namespace": arm["seed_namespace"],
    }
    return _with_fingerprint(body, "retirement_fingerprint")


def _retire(
    layout: W13Layout,
    protocol: Mapping[str, Any],
    ledger: Mapping[str, Any],
    arm: Mapping[str, Any],
    *,
    attempt: Path | None,
    reason: str,
) -> None:
    arm_root = layout.arm(arm["sequence"], arm["arm_id"])
    retirement = _retirement_body(
        protocol, ledger, arm, arm_root, reason=reason
    )
    _write_or_match(layout.retirement, retirement, "W1.3 protocol retirement")
    if attempt is not None:
        _write_or_match(
            attempt / "failure-tombstone.json",
            retirement,
            f"{arm['arm_id']} failure tombstone",
        )
    raise W13OperatorError(f"{arm['seed_namespace']} is invalid; protocol retired")


def _retire_if_claimed_incomplete(
    layout: W13Layout,
    protocol: Mapping[str, Any],
    ledger: Mapping[str, Any],
    arm: Mapping[str, Any],
) -> None:
    claims = _namespace_claims(ledger, arm["seed_namespace"])
    if not claims:
        return
    arm_root = layout.arm(arm["sequence"], arm["arm_id"])
    try:
        completed = _completed_attempt(arm_root)
    except W13OperatorError:
        attempts = _attempt_entries_tolerant(arm_root)
        _retire(
            layout,
            protocol,
            ledger,
            arm,
            attempt=_last_tombstone_directory(attempts),
            reason="malformed-or-multiple-claimed-w1.3-attempt-entries",
        )
    if completed is None:
        attempts = _attempt_entries_tolerant(arm_root)
        _retire(
            layout,
            protocol,
            ledger,
            arm,
            attempt=_last_tombstone_directory(attempts),
            reason="consumed-w1.3-namespace-lacks-one-valid-completion",
        )


def _record_preclaim_failure(
    attempt: Path, arm: Mapping[str, Any], exc: BaseException
) -> None:
    body = {
        "arm_id": arm["arm_id"],
        "automatic_production_promotion_allowed": False,
        "error": f"{type(exc).__name__}: {exc}",
        "manifest_kind": "enoch-week1-w1.3-preclaim-failure",
        "manifest_version": MANIFEST_VERSION,
        "retry_disposition": "new-attempt-allowed-only-while-namespace-unconsumed",
        "seed_namespace": arm["seed_namespace"],
    }
    artifact = _with_fingerprint(body, "preclaim_failure_fingerprint")
    path = attempt / "preclaim-failure.json"
    if path.exists():
        if _load_json(path) != artifact:
            raise W13OperatorError(f"existing preclaim failure changed: {path}")
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
            "manifest_kind": "enoch-week1-w1.3-preclaim-abandoned",
            "manifest_version": MANIFEST_VERSION,
            "reason": "prior-attempt-ended-without-claim-or-completion-marker",
            "retry_disposition": "new-attempt-allowed-because-namespace-is-unconsumed",
            "seed_namespace": arm["seed_namespace"],
        }
        enoch_week1.atomic_write_json(
            attempt / "preclaim-abandoned.json",
            _with_fingerprint(body, "preclaim_abandoned_fingerprint"),
        )


def _validate_completed_arm(
    layout: W13Layout,
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
    arm_root = layout.arm(arm["sequence"], arm["arm_id"])
    execution = attempt / "execution"
    evaluator = layout.base.control_bundle / "bin" / "enoch-week1-evaluator"
    enoch_week1_runner.validate_identity_bindings(
        comparison, identities, evaluator, launch
    )
    expected_execution = {
        "comparison.json": comparison,
        "identity-bindings.json": identities,
        "launch-configuration.json": launch,
    }
    for name, expected in expected_execution.items():
        if _load_json(execution / name) != expected:
            raise W13OperatorError(f"completed {arm['arm_id']} changed {name}")
    probe = _load_json(attempt / "environment-probe.json")
    try:
        w12_operator._validate_environment_probe(  # noqa: SLF001
            probe, arm["environment_identity"]
        )
    except w12_operator.W12OperatorError as exc:
        raise W13OperatorError(str(exc)) from exc
    machine = _load_json(attempt / "machine-attestation.json")
    try:
        machine_fingerprint = enoch_week1_evidence.validate_machine_contention_attestation(
            comparison, machine
        )
    except enoch_week1_evidence.EvidenceError as exc:
        raise W13OperatorError(f"invalid machine attestation: {exc}") from exc
    contract = declaration["execution_contract"]
    if (
        machine["worker_count"] != contract["worker_count"]
        or machine["available_parallelism"] != contract["available_parallelism"]
        or machine["machine_contention_count"] != 0
        or machine["competing_process_count"] != 0
        or machine["exclusive_campaign_lock_held"] is not True
    ):
        raise W13OperatorError(f"{arm['arm_id']} machine attestation changed")
    evidence = _load_json(attempt / "external-evidence.json")
    try:
        external_fingerprint = enoch_week1_evidence.validate_verified_external_evidence(
            protocol, comparison, evidence
        )
    except (enoch_week1.ProtocolError, enoch_week1_evidence.EvidenceError) as exc:
        raise W13OperatorError(f"invalid external evidence: {exc}") from exc
    if _load_json(execution / "external-failure-evidence.json") != evidence:
        raise W13OperatorError(f"completed {arm['arm_id']} changed external evidence")
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
        if not stderr_path.is_file() or stderr_path.stat().st_size != 0:
            raise W13OperatorError(f"{arm['arm_id']} {shard_id} stderr is not empty")
        raw_path = execution / f"{shard_id}.raw.json"
        if not raw_path.is_file():
            raise W13OperatorError(f"{arm['arm_id']} {shard_id} raw output is missing")
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
            raise W13OperatorError(
                f"{arm['arm_id']} {shard_id} does not reconstruct from raw output"
            )
        raw_hashes.append({"sha256": _sha256_file(raw_path), "shard_id": shard_id})
        shards.append(shard)
    recomputed = enoch_week1.merge_shard_results(protocol, comparison, shards)
    merged = _load_json(execution / "merged-result.json")
    if merged != recomputed:
        raise W13OperatorError(f"{arm['arm_id']} merged result does not reconstruct")
    enoch_week1.validate_merged_result(protocol, comparison, merged)
    nonzero = _nonzero_failure_counters(merged["metrics"])
    if nonzero:
        raise W13OperatorError(
            f"{arm['arm_id']} has invalidating counters: {', '.join(nonzero)}"
        )
    runner_plan = _load_json(attempt / "runner-plan.json")
    runner_probe = runner_plan.get("environment_identity_probe")
    if not isinstance(runner_probe, Mapping):
        raise W13OperatorError("runner dry-run plan lacks an environment probe")
    try:
        w12_operator._validate_runner_record(  # noqa: SLF001
            runner_plan,
            comparison=comparison,
            launch=launch,
            evidence=evidence,
            environment_probe=runner_probe,
            evaluator=evaluator,
            protocol_path=layout.base.protocol,
            workers=contract["worker_count"],
            available_parallelism=contract["available_parallelism"],
            dry_run=True,
        )
        w12_operator._validate_environment_probe(  # noqa: SLF001
            runner_probe, arm["environment_identity"]
        )
    except w12_operator.W12OperatorError as exc:
        raise W13OperatorError(str(exc)) from exc
    completion = _load_json(execution / "execution-complete.json")
    completion_probe = completion.get("environment_identity_probe")
    if not isinstance(completion_probe, Mapping):
        raise W13OperatorError("runner completion lacks an environment probe")
    try:
        w12_operator._validate_runner_record(  # noqa: SLF001
            completion,
            comparison=comparison,
            launch=launch,
            evidence=evidence,
            environment_probe=completion_probe,
            evaluator=evaluator,
            protocol_path=layout.base.protocol,
            workers=contract["worker_count"],
            available_parallelism=contract["available_parallelism"],
            dry_run=False,
        )
        w12_operator._validate_environment_probe(  # noqa: SLF001
            completion_probe, arm["environment_identity"]
        )
    except w12_operator.W12OperatorError as exc:
        raise W13OperatorError(str(exc)) from exc
    merged_path = completion.get("merged_result")
    if not isinstance(merged_path, str) or Path(merged_path).resolve() != (
        execution / "merged-result.json"
    ).resolve():
        raise W13OperatorError(f"{arm['arm_id']} completion points at another merge")
    expected_shards = {
        item["shard_id"]: str(execution / f"{item['shard_id']}.result.json")
        for item in comparison["shards"]
    }
    if completion.get("shard_results") != expected_shards:
        raise W13OperatorError(f"{arm['arm_id']} completion shard paths changed")
    _validate_claims(ledger, comparison)
    expected_decision = build_support_decision(
        protocol, arm, merged, _load_json(layout.base.fixtures / "fixture-report.json")
    )
    decision_path = arm_root / "support-decision.json"
    if decision_path.exists():
        if _load_json(decision_path) != expected_decision:
            raise W13OperatorError(f"{arm['arm_id']} support decision changed")
    elif allow_decision_write:
        enoch_week1.atomic_write_json(decision_path, expected_decision)
    else:
        raise W13OperatorError(f"{arm['arm_id']} support decision is missing")
    validate_support_decision(
        expected_decision,
        protocol,
        arm,
        merged,
        _load_json(layout.base.fixtures / "fixture-report.json"),
    )
    return {
        "arm_id": arm["arm_id"],
        "attempt": attempt,
        "comparison": comparison,
        "environment_probe": probe,
        "external_evidence": evidence,
        "external_evidence_fingerprint": external_fingerprint,
        "identity_bindings": identities,
        "launch_configuration": launch,
        "machine_attestation": machine,
        "machine_attestation_fingerprint": machine_fingerprint,
        "merged_result": merged,
        "raw_output_sha256s": raw_hashes,
        "runner_execution": completion,
        "shard_results": shards,
        "support_decision": expected_decision,
    }


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


def _scan_resume_frontier(
    layout: W13Layout,
    protocol: Mapping[str, Any],
    parent_ledger: Mapping[str, Any],
    declaration: Mapping[str, Any],
) -> list[str]:
    ledger = _load_json(layout.base.ledger)
    enoch_week1.validate_seed_ledger(protocol, ledger)
    completed_prefix = []
    gap_seen = False
    for arm_id in SURVIVOR_ARM_IDS:
        arm = _arm_by_id(declaration, arm_id)
        arm_root = layout.arm(arm["sequence"], arm_id)
        try:
            marker = _completed_attempt(arm_root)
        except W13OperatorError:
            if _namespace_claims(ledger, arm["seed_namespace"]):
                attempts = _attempt_entries_tolerant(arm_root)
                _retire(
                    layout,
                    protocol,
                    ledger,
                    arm,
                    attempt=_last_tombstone_directory(attempts),
                    reason="malformed-or-multiple-claimed-w1.3-attempt-entries",
                )
            raise
        claims = _namespace_claims(ledger, arm["seed_namespace"])
        if claims and marker is None:
            _retire_if_claimed_incomplete(layout, protocol, ledger, arm)
        if marker is not None and not claims:
            raise W13OperatorError(f"{arm_id} has a completion marker without claims")
        if marker is not None:
            try:
                _validate_claims(ledger, arm["comparison"])
            except W13OperatorError:
                _retire(
                    layout,
                    protocol,
                    ledger,
                    arm,
                    attempt=marker,
                    reason="claimed-w1.3-completion-has-invalid-claims",
                )
            if gap_seen:
                _retire(
                    layout,
                    protocol,
                    ledger,
                    arm,
                    attempt=marker,
                    reason="claimed-w1.3-completion-is-out-of-order",
                )
            completed_prefix.append(arm_id)
        else:
            gap_seen = True
    _validate_claim_frontier(
        ledger, parent_ledger, declaration, completed_prefix
    )
    return completed_prefix


_INVALID_COMPLETION_EXCEPTIONS = (
    W13OperatorError,
    KeyError,
    OverflowError,
    TypeError,
    ValueError,
    enoch_week1.ProtocolError,
    enoch_week1_evidence.EvidenceError,
    enoch_week1_runner.RunnerError,
    w12_operator.W12OperatorError,
)


def _validate_or_retire_completion(
    layout: W13Layout,
    protocol: Mapping[str, Any],
    ledger: Mapping[str, Any],
    declaration: Mapping[str, Any],
    arm: Mapping[str, Any],
    attempt: Path,
    *,
    base_environment: Mapping[str, str],
) -> dict[str, Any]:
    """Use one retirement boundary for cached and just-finished completions."""

    try:
        return _validate_completed_arm(
            layout,
            protocol,
            ledger,
            declaration,
            arm,
            attempt,
            base_environment=base_environment,
        )
    except OSError:
        # Post-completion decision/seal I/O remains reconstructable.
        raise
    except _INVALID_COMPLETION_EXCEPTIONS:
        _retire(
            layout,
            protocol,
            ledger,
            arm,
            attempt=attempt,
            reason="claimed-w1.3-completion-marker-is-invalid",
        )


def _run_arm(
    layout: W13Layout,
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
    arm_root = layout.arm(arm["sequence"], arm["arm_id"])
    ledger = _load_json(layout.base.ledger)
    enoch_week1.validate_seed_ledger(protocol, ledger)
    try:
        completed = _completed_attempt(arm_root)
    except W13OperatorError:
        if _namespace_claims(ledger, arm["seed_namespace"]):
            attempts = _attempt_entries_tolerant(arm_root)
            _retire(
                layout,
                protocol,
                ledger,
                arm,
                attempt=_last_tombstone_directory(attempts),
                reason="malformed-or-multiple-claimed-w1.3-attempt-entries",
            )
        raise
    if completed is not None:
        return _validate_or_retire_completion(
            layout,
            protocol,
            ledger,
            declaration,
            arm,
            completed,
            base_environment=base_environment,
        )
    _retire_if_claimed_incomplete(layout, protocol, ledger, arm)
    _seal_abandoned_preclaim_attempts(arm_root, arm)
    validate_continuation_provenance(
        provenance, layout, workspace, plan, live_source=True
    )
    attempt = None
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
            child_environment = enoch_week1.sanitized_evaluator_environment(
                base_environment,
                allowlist=protocol["evaluator_environment_policy"]["allowlist"],
            )[0]
            evaluator = layout.base.control_bundle / "bin" / "enoch-week1-evaluator"
            probe = enoch_week1_runner.probe_evaluator_environment_identity(
                evaluator=evaluator,
                protocol_path=layout.base.protocol,
                protocol=protocol,
                comparison=comparison,
                launch_configuration=arm["launch_configuration"],
                child_environment=child_environment,
                evaluator_identity=arm["identity_bindings"]["evaluator"],
                available_parallelism=contract["available_parallelism"],
                campaign_lock_token=campaign_lock_token,
                timeout_seconds=contract["timeout_seconds"],
            )
            try:
                w12_operator._validate_environment_probe(  # noqa: SLF001
                    probe, arm["environment_identity"]
                )
            except w12_operator.W12OperatorError as exc:
                raise W13OperatorError(str(exc)) from exc
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
                model_contract_artifact_paths=w12_operator._model_contract_paths(  # noqa: SLF001
                    layout.parent
                ),
                machine_attestation_path=attempt / "machine-attestation.json",
            )
            enoch_week1.atomic_write_json(attempt / "external-evidence.json", evidence)
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
            enoch_week1.atomic_write_json(attempt / "runner-plan.json", runner_plan)
            enoch_week1_runner.run_comparison(
                **common, output_dir=attempt / "execution", dry_run=False
            )
            ledger = _load_json(layout.base.ledger)
            result = _validate_or_retire_completion(
                layout,
                protocol,
                ledger,
                declaration,
                arm,
                attempt,
                base_environment=base_environment,
            )
            validate_continuation_provenance(
                provenance, layout, workspace, plan, live_source=True
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
            # A valid completion marker followed by recoverable seal I/O is not
            # preclaim and must never trigger a rerun or contradictory tombstone.
            raise
        if attempt is not None:
            _record_preclaim_failure(attempt, arm, exc)
        raise


def _load_and_validate_declaration_state(
    layout: W13Layout,
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
            raise W13OperatorError("materialized W1.3 plan differs from committed source")
    else:
        plan = stored_plan
    provenance = _load_json(layout.provenance)
    validate_continuation_provenance(
        provenance, layout, workspace, plan, live_source=live_source
    )
    declaration = _load_json(layout.declaration)
    arguments = (
        {"environment": environment}
        if live_source
        else {
            "environment_identity_override": declaration["arms"][0][
                "environment_identity"
            ]
        }
    )
    validate_campaign_declaration(
        declaration,
        state["protocol"],
        plan,
        provenance,
        state,
        **arguments,
    )
    _verify_materialized_declaration(layout, plan, provenance, declaration)
    live_ledger = _load_json(layout.base.ledger)
    enoch_week1.validate_seed_ledger(state["protocol"], live_ledger)
    _validate_parent_prefix(live_ledger, state["parent_ledger"])
    return plan, provenance, declaration


def _expected_final_ledger(
    protocol: Mapping[str, Any],
    ledger: Mapping[str, Any],
    parent_ledger: Mapping[str, Any],
    declaration: Mapping[str, Any],
) -> None:
    enoch_week1.validate_seed_ledger(protocol, ledger)
    _validate_parent_prefix(ledger, parent_ledger)
    if len(ledger["consumed"]) != FINAL_COUNT:
        raise W13OperatorError("final W1.3 ledger must contain exactly 18,711 claims")
    for arm_id in SURVIVOR_ARM_IDS:
        _validate_claims(ledger, _arm_by_id(declaration, arm_id)["comparison"])
    expected_namespaces = {
        _arm_by_id(declaration, arm_id)["seed_namespace"]
        for arm_id in SURVIVOR_ARM_IDS
    }
    if any(
        row["namespace"].startswith("dev/survivor/")
        and row["namespace"] not in expected_namespaces
        for row in ledger["consumed"]
    ):
        raise W13OperatorError("ledger contains an undeclared W1.3 survivor namespace")
    _validate_claim_frontier(
        ledger, parent_ledger, declaration, list(SURVIVOR_ARM_IDS)
    )


def _validate_ledger_extension(
    protocol: Mapping[str, Any],
    current: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> None:
    enoch_week1.validate_seed_ledger(protocol, current)
    enoch_week1.validate_seed_ledger(protocol, snapshot)
    prefix_count = len(snapshot["consumed"])
    prefix_body = {
        key: value for key, value in current.items() if key != "ledger_fingerprint"
    }
    prefix_body["consumed"] = list(current["consumed"][:prefix_count])
    if enoch_week1.canonical_json_sha256(prefix_body) != snapshot[
        "ledger_fingerprint"
    ]:
        raise W13OperatorError("live ledger is not an append-only extension of W1.3")
    allowed_downstream_prefixes = ("dev/combination/", "qual/", "locked/")
    if any(
        not row["namespace"].startswith(allowed_downstream_prefixes)
        for row in current["consumed"][prefix_count:]
    ):
        raise W13OperatorError(
            "live ledger extension contains a non-W1.4+ namespace"
        )


def _phase_artifacts(
    provenance: Mapping[str, Any],
    declaration: Mapping[str, Any],
    fixture: Mapping[str, Any],
    supported_set: Mapping[str, Any],
    ledger: Mapping[str, Any],
    arm_evidence: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    artifacts = {
        "continuation-provenance": provenance[
            "continuation_provenance_fingerprint"
        ],
        "fixture-gate": fixture["fixture_report_fingerprint"],
        "supported-independent-change-set": supported_set[
            "supported_change_set_fingerprint"
        ],
        "w1.2-final-ledger": SEALED_W12_LEDGER,
        "w1.2-ranked-independent-ablation-table": SEALED_W12_TABLE,
        "w1.3-campaign-declaration": declaration[
            "campaign_declaration_fingerprint"
        ],
        "w1.3-final-ledger": ledger["ledger_fingerprint"],
    }
    for arm_id, evidence in arm_evidence.items():
        artifacts[f"w1.3/{arm_id}/comparison"] = evidence["comparison"][
            "comparison_protocol_fingerprint"
        ]
        artifacts[f"w1.3/{arm_id}/external-evidence"] = evidence[
            "external_evidence_fingerprint"
        ]
        artifacts[f"w1.3/{arm_id}/merged-result"] = evidence["merged_result"][
            "merged_result_fingerprint"
        ]
        artifacts[f"w1.3/{arm_id}/support-decision"] = evidence[
            "support_decision"
        ]["support_decision_fingerprint"]
    return artifacts


def _build_phase3(
    protocol: Mapping[str, Any],
    phase2: Mapping[str, Any],
    provenance: Mapping[str, Any],
    declaration: Mapping[str, Any],
    fixture: Mapping[str, Any],
    supported_set: Mapping[str, Any],
    ledger: Mapping[str, Any],
    arm_evidence: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    summary = supported_set["summary"]
    return enoch_week1.build_phase_manifest(
        protocol,
        "W1.3",
        artifacts=_phase_artifacts(
            provenance,
            declaration,
            fixture,
            supported_set,
            ledger,
            arm_evidence,
        ),
        declarations={
            "attempted_arm_ids": list(SURVIVOR_ARM_IDS),
            "campaign_declaration_fingerprint": declaration[
                "campaign_declaration_fingerprint"
            ],
            "complete_arm_count": 5,
            "continuation_source_commit": provenance["continuation_git_commit"],
            "final_ledger_fingerprint": ledger["ledger_fingerprint"],
            "pair_count_per_arm": 800,
            "status": supported_set["status"],
            "stopped_arm_ids": summary["stopped_arm_ids"],
            "supported_arm_ids": summary["supported_arm_ids"],
            "supported_change_set_fingerprint": supported_set[
                "supported_change_set_fingerprint"
            ],
            "total_pair_count": 4_000,
            "w1_2_final_ledger_fingerprint": SEALED_W12_LEDGER,
            "w1_2_ranked_table_fingerprint": SEALED_W12_TABLE,
        },
        parent_phase_manifests=[phase2],
    )


def _validate_phase3(
    phase3: Mapping[str, Any],
    state: Mapping[str, Any],
    provenance: Mapping[str, Any],
    declaration: Mapping[str, Any],
    supported_set: Mapping[str, Any],
    ledger: Mapping[str, Any],
    arm_evidence: Mapping[str, Mapping[str, Any]],
) -> str:
    expected = _build_phase3(
        state["protocol"],
        state["phase2"],
        provenance,
        declaration,
        state["fixture"],
        supported_set,
        ledger,
        arm_evidence,
    )
    if dict(phase3) != expected:
        raise W13OperatorError("W1.3 phase manifest does not reconstruct")
    if not any(
        artifact["artifact_id"] == "supported-independent-change-set"
        and artifact["sha256"] == supported_set["supported_change_set_fingerprint"]
        for artifact in phase3["artifacts"]
    ):
        raise W13OperatorError("W1.3 phase omits its declared exit artifact")
    fingerprints = enoch_week1.validate_phase_chain(
        state["protocol"],
        [state["phase0"], state["phase1"], state["phase2"], expected],
    )
    return fingerprints[-1]


def run_w1_3(
    layout: W13Layout,
    workspace: Path,
    w1_2_workspace: Path,
    *,
    operator_id: str,
    attest_no_machine_contention: bool,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run or safely resume all five W1.3 survivor comparisons."""

    if attest_no_machine_contention is not True:
        raise W13OperatorError(
            "--attest-no-machine-contention is required for W1.3 execution"
        )
    workspace = workspace.expanduser().resolve()
    base_environment = dict(os.environ if environment is None else environment)
    verify_sealed_w1_2(layout, w1_2_workspace)
    with base_operator._operator_lock(layout.base.operator_lock):  # noqa: SLF001
        stored_plan = _load_json(layout.input)
        state = _load_parent_state(layout, stored_plan)
        plan, provenance, declaration = _load_and_validate_declaration_state(
            layout,
            workspace,
            state,
            environment=base_environment,
            live_source=True,
        )
        if layout.retirement.exists():
            raise W13OperatorError("this protocol is retired for W1.3")
        completed_prefix = _scan_resume_frontier(
            layout, state["protocol"], state["parent_ledger"], declaration
        )
        claim_frontier = list(completed_prefix)
        arm_evidence = {}
        for arm_id in SURVIVOR_ARM_IDS:
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
            current = _load_json(layout.base.ledger)
            enoch_week1.validate_seed_ledger(state["protocol"], current)
            _validate_claim_frontier(
                current,
                state["parent_ledger"],
                declaration,
                claim_frontier,
            )
        validate_continuation_provenance(
            provenance, layout, workspace, plan, live_source=True
        )
        live_ledger = _load_json(layout.base.ledger)
        if layout.final_ledger.exists():
            ledger = _load_json(layout.final_ledger)
            _expected_final_ledger(
                state["protocol"], ledger, state["parent_ledger"], declaration
            )
            if layout.phase.exists():
                _validate_ledger_extension(state["protocol"], live_ledger, ledger)
            elif live_ledger != ledger:
                raise W13OperatorError(
                    "live ledger advanced before the unsealed W1.3 phase completed"
                )
        else:
            _expected_final_ledger(
                state["protocol"], live_ledger, state["parent_ledger"], declaration
            )
            ledger = live_ledger
            enoch_week1.atomic_write_json(layout.final_ledger, ledger)
        supported_set = build_supported_change_set(
            state["protocol"], declaration, state["fixture"], ledger, arm_evidence
        )
        _write_or_match(
            layout.supported_set,
            supported_set,
            "W1.3 supported-independent-change-set",
        )
        phase3 = _build_phase3(
            state["protocol"],
            state["phase2"],
            provenance,
            declaration,
            state["fixture"],
            supported_set,
            ledger,
            arm_evidence,
        )
        _write_or_match(layout.phase, phase3, "W1.3 phase manifest")
        _validate_phase3(
            phase3,
            state,
            provenance,
            declaration,
            supported_set,
            ledger,
            arm_evidence,
        )
        return {
            "campaign_declaration_fingerprint": declaration[
                "campaign_declaration_fingerprint"
            ],
            "final_ledger_fingerprint": ledger["ledger_fingerprint"],
            "phase_manifest_fingerprint": phase3["phase_manifest_fingerprint"],
            "status": supported_set["status"],
            "supported_arm_ids": supported_set["summary"]["supported_arm_ids"],
            "supported_change_set_fingerprint": supported_set[
                "supported_change_set_fingerprint"
            ],
            "total_pair_count": 4_000,
        }


def verify_w1_3(
    layout: W13Layout,
    workspace: Path,
    w1_2_workspace: Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Reconstruct W1.3 entirely from disk without launching the evaluator."""

    workspace = workspace.expanduser().resolve()
    base_environment = dict(os.environ if environment is None else environment)
    verify_sealed_w1_2(layout, w1_2_workspace)
    with base_operator._operator_lock(layout.base.operator_lock):  # noqa: SLF001
        stored_plan = _load_json(layout.input)
        state = _load_parent_state(layout, stored_plan)
        plan, provenance, declaration = _load_and_validate_declaration_state(
            layout,
            workspace,
            state,
            environment=base_environment,
            live_source=False,
        )
        del plan
        if layout.retirement.exists():
            raise W13OperatorError("this protocol is retired for W1.3")
        ledger = _load_json(layout.final_ledger)
        _expected_final_ledger(
            state["protocol"], ledger, state["parent_ledger"], declaration
        )
        live_ledger = _load_json(layout.base.ledger)
        _validate_ledger_extension(state["protocol"], live_ledger, ledger)
        arm_evidence = {}
        for arm_id in SURVIVOR_ARM_IDS:
            arm = _arm_by_id(declaration, arm_id)
            attempt = _completed_attempt(
                layout.arm(arm["sequence"], arm["arm_id"])
            )
            if attempt is None:
                raise W13OperatorError(f"missing completed W1.3 arm: {arm_id}")
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
        supported_set = _load_json(layout.supported_set)
        set_fingerprint = validate_supported_change_set(
            supported_set,
            state["protocol"],
            declaration,
            state["fixture"],
            ledger,
            arm_evidence,
        )
        phase3 = _load_json(layout.phase)
        phase_fingerprint = _validate_phase3(
            phase3,
            state,
            provenance,
            declaration,
            supported_set,
            ledger,
            arm_evidence,
        )
        return {
            "campaign_declaration_fingerprint": declaration[
                "campaign_declaration_fingerprint"
            ],
            "final_ledger_fingerprint": ledger["ledger_fingerprint"],
            "phase_manifest_fingerprint": phase_fingerprint,
            "status": supported_set["status"],
            "supported_arm_ids": supported_set["summary"]["supported_arm_ids"],
            "supported_change_set_fingerprint": set_fingerprint,
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("declare-w1.3", "freeze all W1.3 comparisons before any seed claim"),
        ("verify-w1.3", "offline-verify the completed W1.3 package"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--root", required=True, type=Path)
        command.add_argument("--workspace", type=Path, default=Path.cwd())
        command.add_argument("--w1-2-workspace", required=True, type=Path)
    run = subparsers.add_parser(
        "run-w1.3", help="run/resume all five W1.3 arms and seal the phase"
    )
    run.add_argument("--root", required=True, type=Path)
    run.add_argument("--workspace", type=Path, default=Path.cwd())
    run.add_argument("--w1-2-workspace", required=True, type=Path)
    run.add_argument("--operator-id", required=True)
    run.add_argument(
        "--attest-no-machine-contention",
        action="store_true",
        help="attest that no competing workload is active during each arm",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    layout = W13Layout(args.root.expanduser().resolve())
    try:
        if args.command == "declare-w1.3":
            result = declare_w1_3(
                layout, args.workspace, args.w1_2_workspace
            )
        elif args.command == "run-w1.3":
            result = run_w1_3(
                layout,
                args.workspace,
                args.w1_2_workspace,
                operator_id=args.operator_id,
                attest_no_machine_contention=args.attest_no_machine_contention,
            )
        else:
            result = verify_w1_3(
                layout, args.workspace, args.w1_2_workspace
            )
    except (
        W13OperatorError,
        base_operator.OperatorError,
        enoch_week1.ProtocolError,
        enoch_week1_evidence.EvidenceError,
        enoch_week1_fixtures.FixtureError,
        enoch_week1_freeze.FreezeError,
        enoch_week1_runner.RunnerError,
        w12_operator.W12OperatorError,
        FileExistsError,
        OSError,
    ) as exc:
        print(f"W1.3 operator failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
