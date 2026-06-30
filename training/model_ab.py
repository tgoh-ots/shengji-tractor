#!/usr/bin/env python3
"""Create and validate fail-closed matched-deal model A/B artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
from pathlib import Path


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_u64(value: str | int, context: str = "seed") -> int:
    if isinstance(value, bool):
        raise ValueError(f"{context} must be an unsigned 64-bit integer")
    text = str(value).strip()
    base = 16 if text.lower().startswith("0x") else 10
    try:
        parsed = int(text, base)
    except ValueError as error:
        raise ValueError(f"{context} must be decimal or 0x-prefixed hexadecimal") from error
    if not 0 <= parsed <= (1 << 64) - 1:
        raise ValueError(f"{context} is outside unsigned 64-bit range")
    return parsed


def _load_json(path: str | os.PathLike[str]) -> dict:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _comparison(candidate: dict, baseline: dict, name: str) -> dict:
    left = candidate[name]
    right = baseline[name]
    if len(left) != len(right):
        raise ValueError(f"{name}: arm lengths differ")
    if not left:
        raise ValueError(f"{name}: arms contain no paired observations")
    delta = [a - b for a, b in zip(left, right)]
    rng = random.Random(0xAB51)
    means = [
        statistics.fmean(delta[rng.randrange(len(delta))] for _ in delta)
        for _ in range(5000)
    ]
    means.sort()
    return {
        # Retain the original field for downstream readers, while the method
        # and gate spell out that this can be a prior candidate rather than the
        # embedded model.
        "candidate_minus_embedded": statistics.fmean(delta),
        "paired_bootstrap95": [means[125], means[4874]],
        "per_deck_delta": delta,
    }


def build_comparison(
    *,
    model: str,
    manifest: str,
    golden: str,
    outdir: str,
    pairs: int,
    seed: str | int,
    budget_ms: int,
    baseline_model: str = "",
    baseline_manifest: str = "",
    baseline_golden: str = "",
    minimum_level_delta: float | None = None,
) -> dict:
    canonical_seed = parse_u64(seed, "A/B seed")
    baseline_result_path = os.path.join(outdir, "embedded.json")
    candidate_result_path = os.path.join(outdir, "candidate.json")
    baseline = _load_json(baseline_result_path)
    candidate = _load_json(candidate_result_path)
    for name, result in (("baseline", baseline), ("candidate", candidate)):
        if result.get("complete_pairs") != pairs:
            raise ValueError(f"{name} arm did not complete all {pairs} pairs")
        if result.get("failed_hands") not in (None, 0):
            raise ValueError(f"{name} arm reports failed hands")
        if result.get("seed") != canonical_seed:
            raise ValueError(
                f"{name} arm seed {result.get('seed')!r} does not match {canonical_seed}"
            )

    payload = {
        "manifest_version": 1,
        "method": "two-process matched-deal candidate-minus-baseline control outcomes",
        "pairs": pairs,
        "seed": canonical_seed,
        "seed_input": str(seed),
        "budget_ms": budget_ms,
        "candidate_model_sha256": sha256_file(model),
        "candidate_manifest_sha256": sha256_file(manifest),
        "golden_sha256": sha256_file(golden),
        "baseline_kind": "candidate-model" if baseline_model else "embedded-model",
        "baseline_model_sha256": sha256_file(baseline_model) if baseline_model else None,
        "baseline_manifest_sha256": (
            sha256_file(baseline_manifest) if baseline_model else None
        ),
        "baseline_golden_sha256": sha256_file(baseline_golden) if baseline_model else None,
        "embedded_result_sha256": sha256_file(baseline_result_path),
        "candidate_result_sha256": sha256_file(candidate_result_path),
        "winrate": _comparison(candidate, baseline, "per_deck_winrate"),
        "point_margin": _comparison(candidate, baseline, "per_deck_margin"),
        "level_utility": _comparison(candidate, baseline, "per_deck_level_utility"),
    }
    # A promotion decision must survive sampling uncertainty, not merely have a
    # favorable point estimate. The paired bootstrap lower bound is conservative
    # for the declared non-inferiority/superiority threshold.
    level_lower95 = payload["level_utility"]["paired_bootstrap95"][0]
    passed = minimum_level_delta is None or level_lower95 >= minimum_level_delta
    payload["promotion_gate"] = {
        "metric": "candidate_minus_baseline_level_utility_bootstrap95_lower",
        "observed_lower95": level_lower95,
        "minimum_level_delta": minimum_level_delta,
        "passed": passed,
    }
    return payload


def _atomic_json(path: str | os.PathLike[str], payload: dict) -> None:
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def write_comparison(outdir: str, payload: dict) -> bool:
    success = os.path.join(outdir, "comparison.json")
    failure = os.path.join(outdir, "comparison.failed.json")
    if payload["promotion_gate"]["passed"] is not True:
        Path(success).unlink(missing_ok=True)
        _atomic_json(failure, payload)
        return False
    Path(failure).unlink(missing_ok=True)
    _atomic_json(success, payload)
    return True


def comparison_is_reusable(
    *,
    comparison_path: str,
    model: str,
    manifest: str,
    golden: str,
    pairs: int,
    seed: str | int,
    budget_ms: int,
    minimum_level_delta: float | None,
    baseline_model: str = "",
    baseline_manifest: str = "",
    baseline_golden: str = "",
) -> bool:
    try:
        result = _load_json(comparison_path)
        if result.get("manifest_version") != 1:
            return False
        expected = {
            "candidate_model_sha256": sha256_file(model),
            "candidate_manifest_sha256": sha256_file(manifest),
            "golden_sha256": sha256_file(golden),
        }
        if any(result.get(name) != value for name, value in expected.items()):
            return False
        if baseline_model:
            if result.get("baseline_kind") != "candidate-model":
                return False
            baseline_expected = {
                "baseline_model_sha256": sha256_file(baseline_model),
                "baseline_manifest_sha256": sha256_file(baseline_manifest),
                "baseline_golden_sha256": sha256_file(baseline_golden),
            }
            if any(result.get(name) != value for name, value in baseline_expected.items()):
                return False
        elif result.get("baseline_kind") != "embedded-model" or any(
            result.get(name) is not None
            for name in (
                "baseline_model_sha256",
                "baseline_manifest_sha256",
                "baseline_golden_sha256",
            )
        ):
            return False
        gate = result.get("promotion_gate")
        if not isinstance(gate, dict) or gate.get("passed") is not True:
            return False
        if gate.get("metric") != "candidate_minus_baseline_level_utility_bootstrap95_lower":
            return False
        interval = result.get("level_utility", {}).get("paired_bootstrap95")
        if not isinstance(interval, list) or len(interval) != 2:
            return False
        if gate.get("observed_lower95") != interval[0]:
            return False
        if gate.get("minimum_level_delta") != minimum_level_delta:
            return False
        if result.get("pairs") != pairs or result.get("budget_ms") != budget_ms:
            return False
        if result.get("seed") != parse_u64(seed, "A/B seed"):
            return False
        outdir = os.path.dirname(os.path.abspath(comparison_path))
        for field, filename in (
            ("embedded_result_sha256", "embedded.json"),
            ("candidate_result_sha256", "candidate.json"),
        ):
            if result.get(field) != sha256_file(os.path.join(outdir, filename)):
                return False
    except (OSError, ValueError, json.JSONDecodeError, KeyError, TypeError):
        return False
    return True


def _common_artifact_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--golden", required=True)
    parser.add_argument("--pairs", type=int, required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--budget-ms", type=int, required=True)
    parser.add_argument("--minimum-level-delta", type=float)
    parser.add_argument("--baseline-model", default="")
    parser.add_argument("--baseline-manifest", default="")
    parser.add_argument("--baseline-golden", default="")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    compare = subparsers.add_parser("compare")
    _common_artifact_arguments(compare)
    compare.add_argument("--outdir", required=True)
    validate = subparsers.add_parser("validate-resume")
    _common_artifact_arguments(validate)
    validate.add_argument("--comparison", required=True)
    args = parser.parse_args()
    common = {
        "model": args.model,
        "manifest": args.manifest,
        "golden": args.golden,
        "pairs": args.pairs,
        "seed": args.seed,
        "budget_ms": args.budget_ms,
        "minimum_level_delta": args.minimum_level_delta,
        "baseline_model": args.baseline_model,
        "baseline_manifest": args.baseline_manifest,
        "baseline_golden": args.baseline_golden,
    }
    if args.command == "compare":
        payload = build_comparison(outdir=args.outdir, **common)
        print(
            json.dumps(
                {key: payload[key] for key in ("winrate", "point_margin", "level_utility")},
                indent=2,
            )
        )
        if not write_comparison(args.outdir, payload):
            raise SystemExit(
                "candidate level delta fails minimum "
                f"{args.minimum_level_delta}"
            )
    elif not comparison_is_reusable(comparison_path=args.comparison, **common):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
