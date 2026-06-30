#!/usr/bin/env python3
"""Train an experimental bid or kitty listwise action ranker.

Input is a replayed/exported candidate CSV, not raw client events.  Every row is
one legal action candidate and every decision has one selected label.  The
required sidecar attests that features came from the acting player's honest
view and that all candidates/actions were checked by the mechanics engine.

The exported manifest exactly matches ``core::bot::phase``.  Artifacts are
always marked ``experimental_candidate``; this script never enables them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn as nn

    _NN_MODULE = nn.Module
    _HAS_TORCH = True
except ModuleNotFoundError:
    torch = None
    nn = None
    _NN_MODULE = object
    _HAS_TORCH = False


FEATURE_DIM = 20
FEATURE_SCHEMA_VERSION = 1
DATASET_SCHEMA_VERSION = 1
CONTRACTS = {
    "bid": "honest_bid_action_ranker",
    "kitty": "honest_kitty_card_ranker",
}
TRAINING_DOMAINS = {
    "bid": "four_player_tractor_two_full_standard_decks_deal_complete_heuristic_v1",
    "kitty": "four_player_tractor_two_full_standard_decks_initial_exchange_heuristic_v1",
}
LOGIT_SEMANTICS = "relative_listwise_rank_only"
FEATURE_NAMES = {
    "bid": [
        "hand_size",
        "deal_fraction",
        "bid_count",
        "bid_is_joker",
        "bid_is_big_joker",
        "candidate_is_no_trump",
        "trump_fraction",
        "trump_points",
        "hand_points",
        "pair_count",
        "trump_pair_count",
        "has_trump_tractor",
        "heuristic_strength",
        "joker_count",
        "has_current_bid",
        "current_bid_count",
        "same_suit_as_current_bid",
        "deal_complete",
        "kitty_size",
        "player_count",
    ],
    "kitty": [
        "card_points",
        "card_is_trump",
        "card_is_joker",
        "card_strength",
        "card_copies",
        "card_is_paired",
        "effective_suit_fraction",
        "would_void_effective_suit",
        "pool_trump_fraction",
        "pool_points",
        "kitty_fraction",
        "heuristic_selected",
        "card_is_ace",
        "card_is_king",
        "card_is_level",
        "card_is_trump_suit",
        "pool_size",
        "effective_suit_remaining_fraction",
        "is_pool_suit_boss",
        "bias",
    ],
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Group:
    game_id: str
    family_id: str
    group_id: str
    x: np.ndarray
    target: int


@dataclass
class Dataset:
    phase: str
    path: str
    groups: list[Group]
    feature_names: list[str]
    sidecar: dict


def load_dataset(path, phase, *, allow_unsafe_no_sidecar=False):
    if phase not in CONTRACTS:
        raise SystemExit(f"phase must be one of {sorted(CONTRACTS)}")
    expected_features = FEATURE_NAMES[phase]
    sidecar_path = f"{path}.manifest.json"
    sidecar = None
    if os.path.isfile(sidecar_path):
        try:
            with open(sidecar_path) as handle:
                sidecar = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(f"invalid phase dataset manifest: {error}") from error
    elif not allow_unsafe_no_sidecar:
        raise SystemExit(
            f"phase dataset requires {sidecar_path}; --allow-unsafe-no-sidecar is research-only"
        )
    if sidecar is not None:
        expected = {
            "manifest_version": 1,
            "dataset_schema_version": DATASET_SCHEMA_VERSION,
            "phase": phase,
            "contract": CONTRACTS[phase],
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_dim": FEATURE_DIM,
            "feature_names": expected_features,
            "logit_semantics": LOGIT_SEMANTICS,
            "training_domain": TRAINING_DOMAINS[phase],
        }
        for name, value in expected.items():
            if sidecar.get(name) != value:
                raise SystemExit(f"phase dataset manifest {name} must be {value!r}")
        if sidecar.get("content_sha256") != sha256_file(path):
            raise SystemExit("phase dataset manifest content_sha256 mismatch")
        verification = sidecar.get("verification")
        required = (
            "honest_observations",
            "legal_candidates",
            "selected_actions_legal",
            "complete_trajectory_ids",
        )
        if not isinstance(verification, dict) or not all(
            verification.get(name) is True for name in required
        ):
            raise SystemExit("phase dataset verification attestations are incomplete")
        if not verification.get("exporter"):
            raise SystemExit("phase dataset verification exporter is required")

    rows_by_group = defaultdict(list)
    game_families = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit("phase dataset is empty or has no header")
        required_columns = {
            "schema_version",
            "game_id",
            "group",
            "candidate_id",
            "label",
            *expected_features,
        }
        missing = sorted(required_columns - set(reader.fieldnames))
        if missing:
            raise SystemExit(f"phase dataset missing columns: {', '.join(missing)}")
        for line, row in enumerate(reader, start=2):
            context = f"row {line}"
            if int(row["schema_version"]) != DATASET_SCHEMA_VERSION:
                raise SystemExit(f"{context}: unsupported phase dataset schema")
            game_id = row["game_id"].strip()
            group_id = row["group"].strip()
            family_id = (row.get("trajectory_family_id") or game_id).strip()
            if not game_id or not group_id or not family_id:
                raise SystemExit(f"{context}: blank game/group/family id")
            if game_families.setdefault(game_id, family_id) != family_id:
                raise SystemExit(f"{context}: game maps to multiple trajectory families")
            try:
                features = np.asarray([float(row[name]) for name in expected_features], np.float32)
            except ValueError as error:
                raise SystemExit(f"{context}: invalid feature") from error
            if not np.all(np.isfinite(features)):
                raise SystemExit(f"{context}: feature contains NaN/Inf")
            candidate_id = int(row["candidate_id"])
            label = int(row["label"])
            if candidate_id < 0 or label not in (0, 1):
                raise SystemExit(f"{context}: invalid candidate/label")
            rows_by_group[group_id].append(
                (candidate_id, label, game_id, family_id, features)
            )

    groups = []
    for group_id, rows in rows_by_group.items():
        rows.sort(key=lambda row: row[0])
        if len(rows) < 2 or [row[0] for row in rows] != list(range(len(rows))):
            raise SystemExit(f"group {group_id}: candidates must be contiguous and non-degenerate")
        if sum(row[1] for row in rows) != 1:
            raise SystemExit(f"group {group_id}: label must be one-hot")
        if len({row[2] for row in rows}) != 1 or len({row[3] for row in rows}) != 1:
            raise SystemExit(f"group {group_id}: candidates span trajectories")
        groups.append(
            Group(
                game_id=rows[0][2],
                family_id=rows[0][3],
                group_id=group_id,
                x=np.stack([row[4] for row in rows]),
                target=next(index for index, row in enumerate(rows) if row[1] == 1),
            )
        )
    if not groups:
        raise SystemExit("phase dataset has no valid decisions")
    return Dataset(
        phase=phase,
        path=path,
        groups=groups,
        feature_names=expected_features,
        sidecar=sidecar or {"unsafe_no_sidecar": True},
    )


def split_by_family(groups, validation_fraction, seed):
    families = sorted({group.family_id for group in groups})
    if len(families) < 2:
        raise SystemExit("phase training requires at least two trajectory families")
    rng = random.Random(seed)
    rng.shuffle(families)
    count = max(1, min(len(families) - 1, round(len(families) * validation_fraction)))
    validation_families = set(families[:count])
    training = [group for group in groups if group.family_id not in validation_families]
    validation = [group for group in groups if group.family_id in validation_families]
    return training, validation, sorted(set(families) - validation_families), sorted(
        validation_families
    )


class Ranker(_NN_MODULE):
    def __init__(self, hidden=64, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, features):
        return self.layers(features)


def batches(groups, batch_size, rng):
    order = list(range(len(groups)))
    rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        selected = [groups[index] for index in order[start : start + batch_size]]
        width = max(group.x.shape[0] for group in selected)
        x = np.zeros((len(selected), width, FEATURE_DIM), dtype=np.float32)
        mask = np.zeros((len(selected), width), dtype=bool)
        targets = np.zeros(len(selected), dtype=np.int64)
        for index, group in enumerate(selected):
            count = group.x.shape[0]
            x[index, :count] = group.x
            mask[index, :count] = True
            targets[index] = group.target
        yield x, mask, targets


def evaluate(model, groups, batch_size):
    model.eval()
    correct = 0
    loss_sum = 0.0
    decisions = 0
    rng = random.Random(0)
    with torch.no_grad():
        for x, mask, targets in batches(groups, batch_size, rng):
            tensor = torch.from_numpy(x)
            logits = model(tensor.reshape(-1, FEATURE_DIM)).reshape(x.shape[:2])
            logits = logits.masked_fill(~torch.from_numpy(mask), -1e9)
            target = torch.from_numpy(targets)
            loss_sum += torch.nn.functional.cross_entropy(
                logits, target, reduction="sum"
            ).item()
            correct += (torch.argmax(logits, dim=1) == target).sum().item()
            decisions += len(targets)
    return {
        "decisions": decisions,
        "top1": correct / max(1, decisions),
        "nll": loss_sum / max(1, decisions),
    }


def atomic_json(path, value):
    temporary = f"{path}.tmp"
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def train(args):
    if not _HAS_TORCH:
        raise SystemExit("PyTorch is required to train/export a phase model")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset = load_dataset(
        args.data, args.phase, allow_unsafe_no_sidecar=args.allow_unsafe_no_sidecar
    )
    training, validation, training_families, validation_families = split_by_family(
        dataset.groups, args.validation_fraction, args.seed
    )
    model = Ranker(args.hidden, args.dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    rng = random.Random(args.seed)
    best_state = None
    best_nll = float("inf")
    best_metrics = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, mask, targets in batches(training, args.batch_size, rng):
            tensor = torch.from_numpy(x)
            logits = model(tensor.reshape(-1, FEATURE_DIM)).reshape(x.shape[:2])
            logits = logits.masked_fill(~torch.from_numpy(mask), -1e9)
            loss = torch.nn.functional.cross_entropy(logits, torch.from_numpy(targets))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        metrics = evaluate(model, validation, args.batch_size)
        print(
            f"epoch {epoch:03d} validation top1={metrics['top1']:.2%} nll={metrics['nll']:.5f}"
        )
        if metrics["nll"] < best_nll:
            best_nll = metrics["nll"]
            best_metrics = metrics
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    dummy = torch.zeros((1, FEATURE_DIM), dtype=torch.float32)
    export_kwargs = dict(
        input_names=["features"],
        output_names=["action_logit"],
        dynamic_axes={"features": {0: "N"}, "action_logit": {0: "N"}},
        opset_version=13,
    )
    try:
        torch.onnx.export(model, dummy, args.out, dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, args.out, **export_kwargs)

    rng_np = np.random.default_rng(0xB1D if args.phase == "bid" else 0x5177)
    golden_x = rng_np.uniform(-1, 1, (11, FEATURE_DIM)).astype(np.float32)
    with torch.no_grad():
        golden_y = model(torch.from_numpy(golden_x)).numpy().reshape(-1)
    golden_path = args.golden_out or f"{args.out}.golden.json"
    atomic_json(
        golden_path,
        {
            "manifest_version": 1,
            "phase": args.phase,
            "feature_dim": FEATURE_DIM,
            "inputs": golden_x.tolist(),
            "action_logits": golden_y.tolist(),
            "atol": 2e-5,
            "rtol": 2e-5,
        },
    )
    unsafe = bool(dataset.sidecar.get("unsafe_no_sidecar"))
    manifest = {
        "manifest_version": 1,
        "contract": CONTRACTS[args.phase],
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_dim": FEATURE_DIM,
        "feature_names": dataset.feature_names,
        "inputs": ["features"],
        "outputs": ["action_logit"],
        "output_semantics": ["policy_logit"],
        "logit_semantics": LOGIT_SEMANTICS,
        "training_domain": TRAINING_DOMAINS[args.phase],
        "model_sha256": sha256_file(args.out),
        "golden_path": os.path.basename(golden_path),
        "serving_status": (
            "non_servable_research" if unsafe else "experimental_candidate"
        ),
        "research_only": True,
        "automatic_production_promotion_allowed": False,
        "unsafe_training_data": unsafe,
        "dataset_sha256": sha256_file(args.data),
        "dataset_manifest_declared_content_sha256": dataset.sidecar.get(
            "content_sha256"
        ),
        "dataset_manifest_sha256": (
            sha256_file(f"{args.data}.manifest.json")
            if os.path.isfile(f"{args.data}.manifest.json")
            else None
        ),
        "golden_sha256": sha256_file(golden_path),
        "training": {
            "seed": args.seed,
            "epochs": args.epochs,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
        },
        "split": {
            "leakage_unit": "trajectory_family_id",
            "training_families": len(training_families),
            "validation_families": len(validation_families),
            "training_family_sha256": hashlib.sha256(
                "\n".join(training_families).encode()
            ).hexdigest(),
            "validation_family_sha256": hashlib.sha256(
                "\n".join(validation_families).encode()
            ).hexdigest(),
        },
        "validation_metrics": best_metrics,
    }
    manifest_path = args.manifest_out or f"{args.out}.manifest.json"
    atomic_json(manifest_path, manifest)
    print(f"wrote experimental {args.phase} model={args.out} manifest={manifest_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=sorted(CONTRACTS))
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest-out")
    parser.add_argument("--golden-out")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument(
        "--allow-unsafe-no-sidecar",
        action="store_true",
        help="research only; exported manifest records unsafe_training_data=true",
    )
    args = parser.parse_args()
    if args.epochs <= 0 or args.hidden < 4 or args.batch_size <= 0:
        raise SystemExit("epochs/batch-size must be positive and hidden >=4")
    if not 0 < args.validation_fraction < 1:
        raise SystemExit("validation-fraction must be in (0,1)")
    train(args)


if __name__ == "__main__":
    main()
