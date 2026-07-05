#!/usr/bin/env python3
"""Run and seal the deterministic W1.2 correctness/feature fixture gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

try:
    from training import enoch_week1
except ImportError:  # Direct script execution.
    import enoch_week1  # type: ignore[no-redef]


MANIFEST_VERSION = 1
REPORT_KIND = "enoch-week1-fixture-report"
_RESULT_RE = re.compile(r"test result: (ok|FAILED)\. (\d+) passed; (\d+) failed;")
SOURCE_ROOTS = ("core/src", "mechanics/src")
SOURCE_FILES = (
    "Cargo.lock",
    "Cargo.toml",
    "core/Cargo.toml",
    "core/examples/enoch_control_probe.rs",
    "core/examples/enoch_eval.rs",
    "mechanics/Cargo.toml",
    "training/enoch_control_probe_reference.patch",
)

GLOBAL_FIXTURES: tuple[tuple[str, str, str], ...] = (
    (
        "bid-score-hash-order-determinism",
        "shengji-core",
        "bot::heuristics::tests::test_bid_strength_is_bit_stable_across_fresh_count_maps",
    ),
    (
        "physical-copy-conservation",
        "shengji-core",
        "bot::tests::test_determinizer_full_memory_conserves_played_cards",
    ),
    (
        "identity-hidden-copy-conservation",
        "shengji-core",
        "bot::determinize::sampler_calibration_tests::identity_hidden_completed_cards_are_conserved_out_of_live_hands",
    ),
    (
        "authoritative-format-alignment",
        "shengji-core",
        "game_state::play_phase::observation_regression_tests::legacy_history_without_formats_stays_index_aligned_after_new_trick",
    ),
    (
        "authoritative-format-redaction",
        "shengji-core",
        "game_state::play_phase::observation_regression_tests::hidden_play_observation_redacts_every_identity_and_format_path",
    ),
    (
        "exhaustive-small-leads",
        "shengji-core",
        "bot::heuristics::tests::exhaustive_small_leads_match_mechanics_across_rule_variants",
    ),
    (
        "exhaustive-small-follows",
        "shengji-core",
        "bot::heuristics::tests::exhaustive_small_follows_match_mechanics_across_rule_variants",
    ),
    (
        "honesty-boundary",
        "shengji-core",
        "bot::tests::test_observed_state_reveals_real_cards_only_for_omniscient",
    ),
    (
        "strict-fixed-work-search-determinism",
        "shengji-core",
        "bot::search::tests::strict_fixed_work_search_repeats_cards_and_work_telemetry",
    ),
)

ARM_FIXTURES: Mapping[str, tuple[str, ...]] = {
    "bid-ownership": (
        "bot::determinize::sampler_calibration_tests::bid_ownership_is_an_isolated_current_holding_constraint",
    ),
    "compound-follow": (
        "bot::determinize::sampler_calibration_tests::compound_follow_rejects_a_sampled_pair_contradicting_public_singles",
    ),
    "failed-throw-better-player": (
        "bot::determinize::sampler_calibration_tests::failed_throw_witness_must_be_able_to_halt_the_attempted_unit",
        "bot::determinize::sampler_calibration_tests::true_world_survives_reverse_validation_of_an_ambiguous_hinted_throw",
    ),
    "friend-revelation": (
        "bot::search::tests::every_hidden_world_ablation_routes_through_the_feature_aware_sampler",
        "bot::determinize::sampler_calibration_tests::friend_revelation_accepts_pending_and_no_double_join_occurrences",
    ),
    "terminal-level-utility": (
        "bot::search::tests::terminal_level_rollout_is_narrowly_gated_to_near_turnover_attackers",
        "bot::search::tests::terminal_scoring_balances_ruff_shape_against_threshold_points",
    ),
    "kitty-burial": (
        "bot::heuristics::tests::test_enoch_kitty_protects_and_buries_no_points_on_weak_hand",
        "bot::heuristics::tests::test_enoch_kitty_completes_void_and_buries_points_on_strong_hand",
    ),
    "late-ruff-shape": (
        "bot::heuristics::tests::test_post_play_ruff_shape_preserves_mechanics_structure",
        "bot::search::tests::terminal_scoring_balances_ruff_shape_against_threshold_points",
    ),
    "contextual-empty-trick": (
        "bot::heuristics::tests::test_contextual_empty_trick_prices_shed_against_trump_spend",
    ),
    "relative-live-suit": (
        "bot::heuristics::tests::test_live_suit_control_uses_relative_share",
    ),
    "team-void-boss": (
        "bot::heuristics::tests::test_team_void_only_discounts_a_boss_when_the_void_opponent_can_ruff",
    ),
    "teammate-entry-return": (
        "bot::heuristics::tests::test_entry_return_only_rewards_returning_a_teammates_entry_suit",
    ),
    "low-trump-handoff": (
        "bot::heuristics::tests::test_handoff_protection_keeps_the_only_final_pair_shape",
    ),
    "structural-family-coverage": (
        "bot::search::tests::root_family_reservation_survives_global_score_pruning",
        "bot::heuristics::tests::test_low_cap_reserves_follow_bomb_family",
    ),
    "progressive-admission": (
        "bot::search::tests::progressive_work_budget_admits_actions_beyond_initial_top_k",
        "bot::search::tests::strict_telemetry_proves_progressive_and_control_have_equal_fixed_work",
    ),
    "uncertain-legal-throws": (
        "bot::search::tests::perfect_information_restores_an_actual_safe_throw_hidden_from_honest_ranking",
    ),
}


class FixtureError(RuntimeError):
    pass


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _source_records(workspace: Path) -> list[dict[str, str]]:
    paths: set[Path] = set()
    for relative in SOURCE_ROOTS:
        root = workspace / relative
        if not root.is_dir():
            raise FixtureError(f"fixture source root is missing: {relative}")
        paths.update(path for path in root.rglob("*") if path.is_file())
    for relative in SOURCE_FILES:
        path = workspace / relative
        if not path.is_file():
            raise FixtureError(f"fixture source file is missing: {relative}")
        paths.add(path)
    return [
        {
            "path": path.relative_to(workspace).as_posix(),
            "sha256": _sha256_file(path),
        }
        for path in sorted(paths, key=lambda path: path.relative_to(workspace).as_posix())
    ]


def _summary(output: str) -> tuple[int, int]:
    results = _RESULT_RE.findall(output)
    if not results:
        raise FixtureError("cargo test output had no machine-recognizable result")
    passed = sum(int(result[1]) for result in results)
    failed = sum(int(result[2]) for result in results)
    return passed, failed


def _safe_environment(environment: Mapping[str, str] | None = None) -> dict[str, str]:
    cleaned, _ = enoch_week1.sanitized_evaluator_environment(environment)
    if not cleaned.get("PATH"):
        raise FixtureError("PATH is required")
    cleaned.update({"LANG": "C", "LC_ALL": "C", "RUST_BACKTRACE": "1"})
    return cleaned


def _run_one(
    workspace: Path,
    package: str,
    test_name: str,
    environment: Mapping[str, str],
) -> tuple[dict[str, Any], bytes]:
    command = [
        "cargo",
        "test",
        "--locked",
        "-p",
        package,
        test_name,
        "--",
        "--exact",
        "--test-threads=1",
    ]
    completed = subprocess.run(
        command,
        cwd=workspace,
        env=dict(environment),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    passed, failed = _summary(completed.stdout.decode("utf-8", errors="replace"))
    record = {
        "command": command,
        "exit_code": completed.returncode,
        "failed": failed,
        "output_sha256": _sha256_bytes(completed.stdout),
        "package": package,
        "passed": passed,
        "test_name": test_name,
    }
    if completed.returncode != 0 or failed != 0 or passed < 1:
        raise FixtureError(f"fixture failed or matched no test: {test_name}")
    return record, completed.stdout


def _assert_registry() -> None:
    if tuple(ARM_FIXTURES) != tuple(enoch_week1.ABLATION_ARMS):
        raise FixtureError("fixture map does not exactly cover the canonical arm registry")
    if any(not tests for tests in ARM_FIXTURES.values()):
        raise FixtureError("every arm needs at least one deterministic fixture")


def _fixture_cases() -> list[tuple[str, str, str, str]]:
    cases = [
        ("global", fixture_id, package, test_name)
        for fixture_id, package, test_name in GLOBAL_FIXTURES
    ]
    cases.extend(
        (arm, test_name.rsplit("::", 1)[-1], "shengji-core", test_name)
        for arm, tests in ARM_FIXTURES.items()
        for test_name in tests
    )
    return cases


def run_fixtures(workspace: Path, output: Path) -> dict[str, Any]:
    _assert_registry()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite fixture artifact directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    environment = _safe_environment()
    source_records = _source_records(workspace)
    records: list[dict[str, Any]] = []
    try:
        cases = _fixture_cases()
        for sequence, (scope, fixture_id, package, test_name) in enumerate(cases):
            record, log = _run_one(workspace, package, test_name, environment)
            log_name = f"{sequence:03d}-{scope}-{fixture_id}.log"
            (temporary / "logs").mkdir(exist_ok=True)
            (temporary / "logs" / log_name).write_bytes(log)
            records.append(
                {
                    **record,
                    "fixture_id": fixture_id,
                    "log_path": f"logs/{log_name}",
                    "scope": scope,
                    "sequence": sequence,
                }
            )
        if _source_records(workspace) != source_records:
            raise FixtureError("fixture source changed while the gate was running")
        body = {
            "arm_registry_sha256": enoch_week1.ARM_REGISTRY_SHA256,
            "automatic_production_promotion_allowed": False,
            "failure_count": 0,
            "manifest_kind": REPORT_KIND,
            "manifest_version": MANIFEST_VERSION,
            "records": records,
            "records_sha256": enoch_week1.canonical_json_sha256(records),
            "source_files": source_records,
            "source_files_sha256": enoch_week1.canonical_json_sha256(source_records),
        }
        report = {
            **body,
            "fixture_report_fingerprint": enoch_week1.canonical_json_sha256(body),
        }
        enoch_week1.atomic_write_json(temporary / "fixture-report.json", report)
        os.replace(temporary, output)
        return report
    except BaseException:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_report(report: Mapping[str, Any]) -> str:
    expected = {
        "arm_registry_sha256",
        "automatic_production_promotion_allowed",
        "failure_count",
        "fixture_report_fingerprint",
        "manifest_kind",
        "manifest_version",
        "records",
        "records_sha256",
        "source_files",
        "source_files_sha256",
    }
    if set(report) != expected:
        raise FixtureError("fixture report fields differ from the frozen schema")
    if report["manifest_kind"] != REPORT_KIND or report["manifest_version"] != 1:
        raise FixtureError("unsupported fixture report")
    if report["automatic_production_promotion_allowed"] is not False:
        raise FixtureError("fixture report cannot authorize promotion")
    if report["arm_registry_sha256"] != enoch_week1.ARM_REGISTRY_SHA256:
        raise FixtureError("fixture report arm registry mismatch")
    if report["failure_count"] != 0:
        raise FixtureError("fixture report contains failures")
    if report["records_sha256"] != enoch_week1.canonical_json_sha256(report["records"]):
        raise FixtureError("fixture record hash mismatch")
    records = report["records"]
    cases = _fixture_cases()
    if not isinstance(records, list) or len(records) != len(cases):
        raise FixtureError("fixture record set is incomplete")
    record_keys = {
        "command",
        "exit_code",
        "failed",
        "fixture_id",
        "log_path",
        "output_sha256",
        "package",
        "passed",
        "scope",
        "sequence",
        "test_name",
    }
    for sequence, (record, (scope, fixture_id, package, test_name)) in enumerate(
        zip(records, cases)
    ):
        if not isinstance(record, Mapping) or set(record) != record_keys:
            raise FixtureError("fixture record schema changed")
        expected_command = [
            "cargo",
            "test",
            "--locked",
            "-p",
            package,
            test_name,
            "--",
            "--exact",
            "--test-threads=1",
        ]
        expected_log_path = f"logs/{sequence:03d}-{scope}-{fixture_id}.log"
        if (
            record["sequence"] != sequence
            or record["scope"] != scope
            or record["fixture_id"] != fixture_id
            or record["package"] != package
            or record["test_name"] != test_name
            or record["command"] != expected_command
            or record["log_path"] != expected_log_path
        ):
            raise FixtureError("fixture record identity or order changed")
        if (
            record["exit_code"] != 0
            or record["failed"] != 0
            or isinstance(record["passed"], bool)
            or not isinstance(record["passed"], int)
            or record["passed"] < 1
        ):
            raise FixtureError("fixture record does not prove a passing exact test")
        digest = record["output_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise FixtureError("fixture output digest is not lowercase SHA-256")
    source_files = report["source_files"]
    if not isinstance(source_files, list) or not source_files:
        raise FixtureError("fixture source identity is empty")
    if report["source_files_sha256"] != enoch_week1.canonical_json_sha256(source_files):
        raise FixtureError("fixture source identity hash mismatch")
    previous_path: str | None = None
    for record in source_files:
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise FixtureError("fixture source record schema changed")
        path = record["path"]
        digest = record["sha256"]
        if not isinstance(path, str) or not path or (previous_path is not None and path <= previous_path):
            raise FixtureError("fixture source paths must be sorted and unique")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise FixtureError("fixture source digest is not lowercase SHA-256")
        previous_path = path
    body = dict(report)
    fingerprint = body.pop("fixture_report_fingerprint")
    if fingerprint != enoch_week1.canonical_json_sha256(body):
        raise FixtureError("fixture report fingerprint mismatch")
    _assert_registry()
    return fingerprint


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"))
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            report = run_fixtures(args.workspace.resolve(), args.output.resolve())
            print(report["fixture_report_fingerprint"])
        else:
            report = enoch_week1.load_json_object(
                args.output.resolve() / "fixture-report.json"
            )
            print(validate_report(report))
    except (FixtureError, enoch_week1.ProtocolError, FileExistsError, OSError) as error:
        print(f"fixture gate failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
