#!/usr/bin/env python3
"""Small, testable resume-contract checks used by training shell drivers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expert_model_is_reusable(
    *,
    model: str,
    dataset: str,
    epochs: int,
    policy_weight: float,
    value_weight: float,
    q_weight: float,
    auxiliary_weight: float,
    policy_target: str,
    early_stop_metric: str,
) -> bool:
    manifest_path = f"{model}.manifest.json"
    golden_path = f"{model}.golden.json"
    dataset_manifest_path = f"{dataset}.manifest.json"
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        expected = {
            "model_sha256": sha256_file(model),
            "golden_sha256": sha256_file(golden_path),
            "dataset_sha256": sha256_file(dataset),
            "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        }
        if any(manifest.get(name) != value for name, value in expected.items()):
            return False
        training = manifest.get("training", {})
        training_expected = {
            "seed": 0,
            "epochs": epochs,
            "policy_weight": policy_weight,
            "value_weight": value_weight,
            "q_weight": q_weight,
            "auxiliary_weight": auxiliary_weight,
            "policy_target": policy_target,
            "early_stop_metric": early_stop_metric,
        }
        if any(training.get(name) != value for name, value in training_expected.items()):
            return False
    except (OSError, ValueError, json.JSONDecodeError, TypeError):
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    model = subparsers.add_parser("verify-model-resume")
    model.add_argument("--model", required=True)
    model.add_argument("--dataset", required=True)
    model.add_argument("--epochs", type=int, required=True)
    model.add_argument("--policy-weight", type=float, required=True)
    model.add_argument("--value-weight", type=float, required=True)
    model.add_argument("--q-weight", type=float, required=True)
    model.add_argument("--auxiliary-weight", type=float, required=True)
    model.add_argument("--policy-target", required=True)
    model.add_argument("--early-stop-metric", required=True)
    args = parser.parse_args()
    if args.command == "verify-model-resume" and not expert_model_is_reusable(
        model=args.model,
        dataset=args.dataset,
        epochs=args.epochs,
        policy_weight=args.policy_weight,
        value_weight=args.value_weight,
        q_weight=args.q_weight,
        auxiliary_weight=args.auxiliary_weight,
        policy_target=args.policy_target,
        early_stop_metric=args.early_stop_metric,
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
