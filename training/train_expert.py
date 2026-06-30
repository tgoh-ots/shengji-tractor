#!/usr/bin/env python3
"""Train the Expert policy baseline plus explicit state-V and action-Q heads.

The loader accepts both legacy CSVs (`group,f*,label[,value]`) and schema-v3
datasets from `gen_training_data` (`game_id`, metadata, `v_target`, sparse
`q_target`). The contracts are intentionally separate:

* `score(o,a)` retains the current listwise teacher/behaviour policy baseline;
* `state_value(o)` is trained once per decision from the behaviour return and
  receives a candidate-masked input, so it cannot masquerade as Q;
* `action_q(o,a)` is trained only where a counterfactual candidate return exists.

Train/validation partitioning is by whole game_id. A model and its companion
`MODEL.onnx.manifest.json` are written together; Rust inference validates the
feature schema, output count, shapes, and finiteness before using the model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
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


@dataclass
class DecisionGroup:
    game_id: str
    group_id: str
    x: np.ndarray
    teacher_y: int
    behaviour_y: int
    v_target: float | None
    q_target: np.ndarray
    v_bucket: int | None
    q_bucket: np.ndarray
    v_win: float | None
    q_win: np.ndarray
    v_kitty: float | None
    q_kitty: np.ndarray


@dataclass
class Dataset:
    groups: list[DecisionGroup]
    feature_names: list[str]
    dataset_schema_version: int
    feature_schema_version: int
    game_ids_are_trajectories: bool
    bucket_values: list[int]
    path: str

    @property
    def feature_dim(self):
        return len(self.feature_names)


def _feature_sort_key(name):
    try:
        return int(name[1:])
    except (TypeError, ValueError):
        return 10**9


def _optional_float(value):
    if value is None or value.strip() == "":
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        return None
    return parsed


def load_dataset(path):
    """Load legacy or v2 CSV into validated per-decision groups."""
    rows_by_group = defaultdict(list)
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit("Training CSV is empty or has no header.")
        feature_names = sorted(
            [name for name in reader.fieldnames if name.startswith("f") and name[1:].isdigit()],
            key=_feature_sort_key,
        )
        if not feature_names:
            raise SystemExit("Training CSV has no f0..fN feature columns.")
        expected = [f"f{i}" for i in range(len(feature_names))]
        if feature_names != expected:
            raise SystemExit(
                f"Feature columns must be contiguous f0..fN; got {feature_names[:4]}...{feature_names[-4:]}"
            )
        if "group" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise SystemExit("Training CSV must contain group and label columns.")

        dataset_schema = 1
        game_ids_are_trajectories = "game_id" in reader.fieldnames
        for row_index, row in enumerate(reader, start=2):
            if not row:
                continue
            group_id = row["group"]
            if not group_id:
                raise SystemExit(f"row {row_index}: blank group id")
            if row.get("schema_version"):
                dataset_schema = max(dataset_schema, int(row["schema_version"]))
            if not row.get("game_id"):
                game_ids_are_trajectories = False
            game_id = row.get("game_id") or f"legacy-group-{group_id}"
            try:
                features = np.asarray(
                    [float(row[name]) for name in feature_names], dtype=np.float32
                )
            except (TypeError, ValueError) as error:
                raise SystemExit(f"row {row_index}: invalid feature: {error}") from error
            if not np.all(np.isfinite(features)):
                raise SystemExit(f"row {row_index}: feature contains NaN/Inf")
            value = row.get("v_target")
            if value is None:
                value = row.get("value")
            rows_by_group[group_id].append(
                {
                    "game_id": game_id,
                    "candidate_id": int(row.get("candidate_id") or len(rows_by_group[group_id])),
                    "features": features,
                    "label": int(row["label"]),
                    "behaviour_label": int(row.get("behaviour_label") or row["label"]),
                    "v_target": _optional_float(value),
                    "q_target": _optional_float(row.get("q_target")),
                    "v_bucket": _optional_float(row.get("v_score_bucket")),
                    "q_bucket": _optional_float(row.get("q_score_bucket")),
                    "v_win": _optional_float(row.get("v_win_target")),
                    "q_win": _optional_float(row.get("q_win_target")),
                    "v_kitty": _optional_float(row.get("v_kitty_target")),
                    "q_kitty": _optional_float(row.get("q_kitty_target")),
                }
            )

    groups = []
    dropped = 0
    for group_id, rows in rows_by_group.items():
        rows.sort(key=lambda row: row["candidate_id"])
        if len(rows) < 2 or sum(row["label"] for row in rows) != 1:
            dropped += 1
            continue
        if sum(row["behaviour_label"] for row in rows) != 1:
            raise SystemExit(f"group {group_id}: expected exactly one behaviour_label")
        game_ids = {row["game_id"] for row in rows}
        if len(game_ids) != 1:
            raise SystemExit(f"group {group_id}: candidates span multiple games")
        values = [row["v_target"] for row in rows if row["v_target"] is not None]
        if values and max(values) - min(values) > 1e-5:
            raise SystemExit(f"group {group_id}: state v_target differs across candidates")
        q_target = np.asarray(
            [np.nan if row["q_target"] is None else row["q_target"] for row in rows],
            dtype=np.float32,
        )
        def vector(name):
            return np.asarray(
                [np.nan if row[name] is None else row[name] for row in rows],
                dtype=np.float32,
            )
        def constant(name):
            values = [row[name] for row in rows if row[name] is not None]
            if values and max(values) - min(values) > 1e-5:
                raise SystemExit(f"group {group_id}: state {name} differs across candidates")
            return values[0] if values else None
        v_bucket = constant("v_bucket")
        groups.append(
            DecisionGroup(
                game_id=next(iter(game_ids)),
                group_id=group_id,
                x=np.stack([row["features"] for row in rows]),
                teacher_y=next(i for i, row in enumerate(rows) if row["label"] == 1),
                behaviour_y=next(
                    i for i, row in enumerate(rows) if row["behaviour_label"] == 1
                ),
                v_target=values[0] if values else None,
                q_target=q_target,
                v_bucket=int(v_bucket) if v_bucket is not None else None,
                q_bucket=vector("q_bucket"),
                v_win=constant("v_win"),
                q_win=vector("q_win"),
                v_kitty=constant("v_kitty"),
                q_kitty=vector("q_kitty"),
            )
        )
    if dropped:
        print(f"Dropped {dropped} malformed/degenerate groups.")
    if len(feature_names) == 36:
        feature_schema = 1
    elif len(feature_names) == 49:
        feature_schema = 2
    else:
        raise SystemExit(
            f"Unsupported feature width {len(feature_names)}; runtime contracts support exactly "
            "schema-v1/dim36 or schema-v2/dim49."
        )
    bucket_values = sorted(
        {
            int(value)
            for group in groups
            for value in (
                ([group.v_bucket] if group.v_bucket is not None else [])
                + [v for v in group.q_bucket if np.isfinite(v)]
            )
        }
    )
    bucket_to_index = {value: index for index, value in enumerate(bucket_values)}
    for group in groups:
        if group.v_bucket is not None:
            group.v_bucket = bucket_to_index[group.v_bucket]
        group.q_bucket = np.asarray(
            [
                np.nan if not np.isfinite(value) else bucket_to_index[int(value)]
                for value in group.q_bucket
            ],
            dtype=np.float32,
        )
    return Dataset(
        groups,
        feature_names,
        dataset_schema,
        feature_schema,
        game_ids_are_trajectories,
        bucket_values,
        path,
    )


def split_by_game(groups, val_frac, seed):
    """Partition whole trajectories; no game may appear in both sets."""
    game_ids = sorted({group.game_id for group in groups})
    if len(game_ids) < 2:
        raise SystemExit(
            "Need at least two game_id values for a leakage-free train/val split. "
            "Generate >=2 schema-v3 games (legacy CSVs cannot recover hand IDs)."
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(game_ids)
    n_val = max(1, int(round(len(game_ids) * val_frac)))
    n_val = min(n_val, len(game_ids) - 1)
    val_games = set(game_ids[:n_val])
    train = [group for group in groups if group.game_id not in val_games]
    val = [group for group in groups if group.game_id in val_games]
    assert {g.game_id for g in train}.isdisjoint({g.game_id for g in val})
    return train, val, sorted(set(game_ids) - val_games), sorted(val_games)


def analyze_dataset(dataset, granularities=(2, 1, 0)):
    groups = dataset.groups
    n_rows = sum(group.x.shape[0] for group in groups)
    games = {group.game_id for group in groups}
    q_rows = sum(np.isfinite(group.q_target).sum() for group in groups)
    q_pairs = sum(np.isfinite(group.q_target).sum() >= 2 for group in groups)
    print("\n=== dataset / target diagnostics ===")
    print(
        f"schema={dataset.dataset_schema_version} feature_schema={dataset.feature_schema_version} "
        f"dim={dataset.feature_dim}; {len(games)} games, {len(groups)} decisions, "
        f"{n_rows} rows, {q_rows} Q rows, {q_pairs} Q-comparable decisions"
    )
    print(f"random policy top-1 baseline ≈ {len(groups) / max(1, n_rows):.1%}")
    for decimals in granularities:
        positives = defaultdict(int)
        totals = defaultdict(int)
        collided = 0
        for group in groups:
            keys = [tuple(np.round(row, decimals).tolist()) for row in group.x]
            for index, key in enumerate(keys):
                totals[key] += 1
                positives[key] += int(index == group.teacher_y)
            chosen = keys[group.teacher_y]
            collided += any(i != group.teacher_y and key == chosen for i, key in enumerate(keys))
        majority_errors = sum(
            total - max(positives[key], total - positives[key])
            for key, total in totals.items()
        )
        print(
            f"round={decimals}dp: row Bayes-error floor "
            f"{majority_errors / max(1, n_rows):.1%}; exact-within-decision collision "
            f"{collided / max(1, len(groups)):.1%}"
        )


def state_feature_mask(feature_dim):
    """Zero every action-dependent coordinate before state-V inference."""
    action_dependent = set(range(0, 9)) | {16, 26, 34}
    if feature_dim >= 48:
        action_dependent |= {42, 43, 44}
    if feature_dim >= 49:
        action_dependent.add(48)
    mask = np.ones(feature_dim, dtype=np.float32)
    for index in action_dependent:
        if index < feature_dim:
            mask[index] = 0.0
    return mask


class CandidateScorer(_NN_MODULE):
    """Policy/Q share action features; state-V has a candidate-masked trunk."""

    def __init__(self, in_dim, hidden=128, dropout=0.1, num_buckets=1):
        super().__init__()
        self.register_buffer(
            "state_mask", torch.from_numpy(state_feature_mask(in_dim)).reshape(1, in_dim)
        )
        self.action_trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        self.state_trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden // 2, 1)
        self.state_value_head = nn.Linear(hidden // 2, 1)
        self.action_q_head = nn.Linear(hidden // 2, 1)
        self.state_bucket_head = nn.Linear(hidden // 2, num_buckets)
        self.action_bucket_head = nn.Linear(hidden // 2, num_buckets)
        self.state_win_head = nn.Linear(hidden // 2, 1)
        self.action_win_head = nn.Linear(hidden // 2, 1)
        self.state_kitty_head = nn.Linear(hidden // 2, 1)
        self.action_kitty_head = nn.Linear(hidden // 2, 1)

    def forward(self, x):
        action_z = self.action_trunk(x)
        state_z = self.state_trunk(x * self.state_mask)
        return (
            self.policy_head(action_z),
            torch.tanh(self.state_value_head(state_z)),
            torch.tanh(self.action_q_head(action_z)),
            self.state_bucket_head(state_z),
            self.action_bucket_head(action_z),
            self.state_win_head(state_z),
            self.action_win_head(action_z),
            self.state_kitty_head(state_z),
            self.action_kitty_head(action_z),
        )


class PolicyOnly(_NN_MODULE):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[0]


class PolicyAndValue(_NN_MODULE):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        policy, state_value = self.model(x)[:2]
        return policy, state_value


class PolicyValueQ(_NN_MODULE):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[:3]


def padded_batches(groups, batch_groups, device, policy_target):
    order = np.random.permutation(len(groups))
    for start in range(0, len(groups), batch_groups):
        batch = [groups[index] for index in order[start : start + batch_groups]]
        k = max(group.x.shape[0] for group in batch)
        b = len(batch)
        d = batch[0].x.shape[1]
        features = np.zeros((b, k, d), dtype=np.float32)
        mask = np.zeros((b, k), dtype=np.float32)
        target = np.zeros(b, dtype=np.int64)
        state_value = np.full(b, np.nan, dtype=np.float32)
        action_q = np.full((b, k), np.nan, dtype=np.float32)
        state_bucket = np.full(b, -1, dtype=np.int64)
        action_bucket = np.full((b, k), -1, dtype=np.int64)
        state_win = np.full(b, np.nan, dtype=np.float32)
        action_win = np.full((b, k), np.nan, dtype=np.float32)
        state_kitty = np.full(b, np.nan, dtype=np.float32)
        action_kitty = np.full((b, k), np.nan, dtype=np.float32)
        for i, group in enumerate(batch):
            n = group.x.shape[0]
            features[i, :n] = group.x
            mask[i, :n] = 1.0
            target[i] = group.teacher_y if policy_target == "teacher" else group.behaviour_y
            if group.v_target is not None:
                state_value[i] = group.v_target
            action_q[i, :n] = group.q_target
            if group.v_bucket is not None:
                state_bucket[i] = group.v_bucket
            finite_bucket = np.isfinite(group.q_bucket)
            action_bucket[i, :n][finite_bucket] = group.q_bucket[finite_bucket].astype(np.int64)
            if group.v_win is not None:
                state_win[i] = group.v_win
            action_win[i, :n] = group.q_win
            if group.v_kitty is not None:
                state_kitty[i] = group.v_kitty
            action_kitty[i, :n] = group.q_kitty
        yield tuple(
            torch.from_numpy(value).to(device)
            for value in (
                features,
                target,
                mask,
                state_value,
                action_q,
                state_bucket,
                action_bucket,
                state_win,
                action_win,
                state_kitty,
                action_kitty,
            )
        )


def loss_for_batch(
    model,
    features,
    target,
    mask,
    state_target,
    q_target,
    state_bucket_target,
    q_bucket_target,
    state_win_target,
    q_win_target,
    state_kitty_target,
    q_kitty_target,
    policy_weight,
    value_weight,
    q_weight,
    auxiliary_weight,
):
    b, k, d = features.shape
    (
        policy,
        state_value,
        action_q,
        state_bucket,
        action_bucket,
        state_win,
        action_win,
        state_kitty,
        action_kitty,
    ) = model(features.reshape(b * k, d))
    policy = policy.reshape(b, k)
    state_value = state_value.reshape(b, k)
    action_q = action_q.reshape(b, k)
    num_buckets = state_bucket.shape[-1]
    state_bucket = state_bucket.reshape(b, k, num_buckets)
    action_bucket = action_bucket.reshape(b, k, num_buckets)
    state_win = state_win.reshape(b, k)
    action_win = action_win.reshape(b, k)
    state_kitty = state_kitty.reshape(b, k)
    action_kitty = action_kitty.reshape(b, k)
    neg_inf = torch.finfo(policy.dtype).min
    logits = torch.where(mask > 0, policy, torch.full_like(policy, neg_inf))
    policy_loss = -torch.log_softmax(logits, dim=1)[
        torch.arange(b, device=logits.device), target
    ].mean()

    state_mask = torch.isfinite(state_target)
    if bool(state_mask.any()):
        # Candidate-dependent coordinates are masked inside the model, so every
        # row has the same V(o); train once per decision rather than weighting
        # states with more legal candidates more heavily.
        value_loss = torch.nn.functional.smooth_l1_loss(
            state_value[:, 0][state_mask], state_target[state_mask]
        )
    else:
        value_loss = policy_loss * 0.0

    q_mask = torch.isfinite(q_target) & (mask > 0)
    if bool(q_mask.any()):
        q_loss = torch.nn.functional.smooth_l1_loss(action_q[q_mask], q_target[q_mask])
    else:
        q_loss = policy_loss * 0.0

    auxiliary_parts = []
    state_bucket_mask = state_bucket_target >= 0
    if bool(state_bucket_mask.any()):
        auxiliary_parts.append(
            torch.nn.functional.cross_entropy(
                state_bucket[:, 0][state_bucket_mask],
                state_bucket_target[state_bucket_mask],
            )
        )
    q_bucket_mask = (q_bucket_target >= 0) & (mask > 0)
    if bool(q_bucket_mask.any()):
        auxiliary_parts.append(
            torch.nn.functional.cross_entropy(
                action_bucket[q_bucket_mask], q_bucket_target[q_bucket_mask]
            )
        )
    for prediction, target_values in (
        (state_win[:, 0], state_win_target),
        (state_kitty[:, 0], state_kitty_target),
    ):
        valid = torch.isfinite(target_values)
        if bool(valid.any()):
            auxiliary_parts.append(
                torch.nn.functional.binary_cross_entropy_with_logits(
                    prediction[valid], target_values[valid]
                )
            )
    for prediction, target_values in (
        (action_win, q_win_target),
        (action_kitty, q_kitty_target),
    ):
        valid = torch.isfinite(target_values) & (mask > 0)
        if bool(valid.any()):
            auxiliary_parts.append(
                torch.nn.functional.binary_cross_entropy_with_logits(
                    prediction[valid], target_values[valid]
                )
            )
    auxiliary_loss = (
        torch.stack(auxiliary_parts).mean() if auxiliary_parts else policy_loss * 0.0
    )
    total = (
        policy_weight * policy_loss
        + value_weight * value_loss
        + q_weight * q_loss
        + auxiliary_weight * auxiliary_loss
    )
    return total, policy_loss, value_loss, q_loss, auxiliary_loss


def evaluate(model, groups, device, policy_target, chunk=2048):
    model.eval()
    correct = 0
    v_sq = 0.0
    v_count = 0
    q_sq = 0.0
    q_count = 0
    q_rank_correct = 0
    q_rank_count = 0
    bucket_correct = 0
    bucket_count = 0
    win_probabilities, win_targets = [], []
    kitty_probabilities, kitty_targets = [], []
    with torch.no_grad():
        for start in range(0, len(groups), chunk):
            for (
                features,
                target,
                mask,
                state_target,
                q_target,
                state_bucket_target,
                q_bucket_target,
                state_win_target,
                q_win_target,
                state_kitty_target,
                q_kitty_target,
            ) in padded_batches(
                groups[start : start + chunk], chunk, device, policy_target
            ):
                b, k, d = features.shape
                outputs = model(features.reshape(b * k, d))
                policy, state_value, action_q = outputs[:3]
                state_bucket, action_bucket = outputs[3:5]
                state_win, action_win, state_kitty, action_kitty = outputs[5:]
                policy = policy.reshape(b, k)
                state_value = state_value.reshape(b, k)
                action_q = action_q.reshape(b, k)
                bucket_dim = state_bucket.shape[-1]
                state_bucket = state_bucket.reshape(b, k, bucket_dim)
                action_bucket = action_bucket.reshape(b, k, bucket_dim)
                state_win = state_win.reshape(b, k)
                action_win = action_win.reshape(b, k)
                state_kitty = state_kitty.reshape(b, k)
                action_kitty = action_kitty.reshape(b, k)
                neg_inf = torch.finfo(policy.dtype).min
                logits = torch.where(mask > 0, policy, torch.full_like(policy, neg_inf))
                correct += int((torch.argmax(logits, dim=1) == target).sum().item())

                v_mask = torch.isfinite(state_target)
                if bool(v_mask.any()):
                    diff = state_value[:, 0][v_mask] - state_target[v_mask]
                    v_sq += float((diff * diff).sum().item())
                    v_count += int(v_mask.sum().item())

                q_mask = torch.isfinite(q_target) & (mask > 0)
                if bool(q_mask.any()):
                    diff = action_q[q_mask] - q_target[q_mask]
                    q_sq += float((diff * diff).sum().item())
                    q_count += int(q_mask.sum().item())
                for i in range(b):
                    valid = torch.nonzero(q_mask[i], as_tuple=False).flatten()
                    if valid.numel() >= 2:
                        predicted = valid[torch.argmax(action_q[i, valid])]
                        actual = valid[torch.argmax(q_target[i, valid])]
                        q_rank_correct += int(predicted == actual)
                        q_rank_count += 1
                state_bucket_mask = state_bucket_target >= 0
                if bool(state_bucket_mask.any()):
                    bucket_correct += int(
                        (
                            torch.argmax(state_bucket[:, 0], dim=1)[state_bucket_mask]
                            == state_bucket_target[state_bucket_mask]
                        )
                        .sum()
                        .item()
                    )
                    bucket_count += int(state_bucket_mask.sum().item())
                q_bucket_mask = (q_bucket_target >= 0) & (mask > 0)
                if bool(q_bucket_mask.any()):
                    bucket_correct += int(
                        (
                            torch.argmax(action_bucket, dim=2)[q_bucket_mask]
                            == q_bucket_target[q_bucket_mask]
                        )
                        .sum()
                        .item()
                    )
                    bucket_count += int(q_bucket_mask.sum().item())
                for prediction, target_values, probabilities, targets in (
                    (state_win[:, 0], state_win_target, win_probabilities, win_targets),
                    (action_win, q_win_target, win_probabilities, win_targets),
                    (
                        state_kitty[:, 0],
                        state_kitty_target,
                        kitty_probabilities,
                        kitty_targets,
                    ),
                    (action_kitty, q_kitty_target, kitty_probabilities, kitty_targets),
                ):
                    valid = torch.isfinite(target_values)
                    if target_values.ndim == 2:
                        valid &= mask > 0
                    if bool(valid.any()):
                        probabilities.extend(
                            torch.sigmoid(prediction[valid]).cpu().tolist()
                        )
                        targets.extend(target_values[valid].cpu().tolist())
    model.train()
    def brier_and_ece(probabilities, targets):
        if not targets:
            return float("nan"), float("nan")
        p = np.asarray(probabilities)
        y = np.asarray(targets)
        brier = float(np.mean((p - y) ** 2))
        ece = 0.0
        for low in np.linspace(0.0, 0.9, 10):
            selected = (p >= low) & (p < low + 0.1)
            if selected.any():
                ece += selected.mean() * abs(p[selected].mean() - y[selected].mean())
        return brier, float(ece)
    win_brier, win_ece = brier_and_ece(win_probabilities, win_targets)
    kitty_brier, kitty_ece = brier_and_ece(kitty_probabilities, kitty_targets)
    return {
        "policy_top1": correct / max(1, len(groups)),
        "value_rmse": math.sqrt(v_sq / v_count) if v_count else float("nan"),
        "q_rmse": math.sqrt(q_sq / q_count) if q_count else float("nan"),
        "q_rank": q_rank_correct / q_rank_count if q_rank_count else float("nan"),
        "q_rows": q_count,
        "q_groups": q_rank_count,
        "bucket_accuracy": bucket_correct / bucket_count if bucket_count else float("nan"),
        "bucket_rows": bucket_count,
        "win_brier": win_brier,
        "win_ece10": win_ece,
        "kitty_brier": kitty_brier,
        "kitty_ece10": kitty_ece,
    }


def metric_value(metrics, name):
    if name == "policy":
        return metrics["policy_top1"]
    if name == "value":
        value = metrics["value_rmse"]
        return -value if math.isfinite(value) else -float("inf")
    if name == "q":
        value = metrics["q_rmse"]
        return -value if math.isfinite(value) else -float("inf")
    raise ValueError(name)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_onnx(model, out_path, feature_dim, with_value, with_q):
    model.eval()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    dummy = torch.zeros((1, feature_dim), dtype=torch.float32)
    if with_q:
        export_model = PolicyValueQ(model)
        outputs = ["score", "state_value", "action_q"]
    elif with_value:
        export_model = PolicyAndValue(model)
        outputs = ["score", "state_value"]
    else:
        export_model = PolicyOnly(model)
        outputs = ["score"]
    dynamic_axes = {"x": {0: "N"}, **{name: {0: "N"} for name in outputs}}
    kwargs = dict(
        input_names=["x"],
        output_names=outputs,
        dynamic_axes=dynamic_axes,
        opset_version=13,
    )
    try:
        torch.onnx.export(export_model, dummy, out_path, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(export_model, dummy, out_path, **kwargs)
    return outputs


def write_golden(model, path, feature_dim, outputs):
    """Write deterministic PyTorch outputs for Rust/tract numerical parity."""
    rng = np.random.default_rng(0x51A7E)
    inputs = rng.uniform(0.0, 1.0, size=(13, feature_dim)).astype(np.float32)
    if feature_dim > 27:
        inputs[:, 27] = 1.0
    if feature_dim > 37:
        inputs[:, 37] = rng.uniform(-1.0, 1.0, size=inputs.shape[0])
    model.eval()
    with torch.no_grad():
        predicted = model(torch.from_numpy(inputs))
    expected = {
        name: predicted[index].detach().cpu().numpy().reshape(-1).tolist()
        for index, name in enumerate(outputs)
    }
    payload = {
        "manifest_version": 1,
        "feature_dim": feature_dim,
        "atol": 2e-5,
        "rtol": 2e-5,
        "inputs": inputs.tolist(),
        "outputs": expected,
    }
    temporary = f"{path}.tmp"
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def write_model_manifest(path, golden_path, dataset, outputs, args, split, final_metrics):
    manifest_path = args.manifest_out or f"{path}.manifest.json"
    value_semantic = (
        "normalized_point_margin"
        if dataset.feature_schema_version == 1
        else "normalized_level_utility"
    )
    output_semantics = ["policy_logit"] + [value_semantic for _ in outputs[1:]]
    manifest = {
        "manifest_version": 1,
        "feature_schema_version": dataset.feature_schema_version,
        "feature_dim": dataset.feature_dim,
        "outputs": outputs,
        "output_semantics": output_semantics,
        "contract": "policy+state_v+action_q" if len(outputs) == 3 else "+".join(outputs),
        "model_sha256": sha256(path),
        "golden_path": os.path.basename(golden_path),
        "golden_sha256": sha256(golden_path),
        "dataset_sha256": sha256(dataset.path),
        "dataset_schema_version": dataset.dataset_schema_version,
        "training": {
            "seed": args.seed,
            "epochs": args.epochs,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "policy_target": args.policy_target,
            "policy_weight": args.policy_weight,
            "value_weight": args.value_weight,
            "q_weight": args.q_weight,
            "auxiliary_weight": args.auxiliary_weight,
            "early_stop_metric": args.early_stop_metric,
        },
        "split": {
            "train_games": len(split[0]),
            "validation_games": len(split[1]),
            "train_game_id_sha256": hashlib.sha256("\n".join(split[0]).encode()).hexdigest(),
            "validation_game_id_sha256": hashlib.sha256("\n".join(split[1]).encode()).hexdigest(),
            "game_ids_are_trajectories": dataset.game_ids_are_trajectories,
        },
        "validation_metrics": final_metrics,
        "offline_auxiliary_heads": {
            "exported_to_onnx": False,
            "score_bucket_values": dataset.bucket_values,
            "targets": [
                "state_score_bucket",
                "action_score_bucket",
                "state_team_win",
                "action_team_win",
                "state_final_trick_kitty_win",
                "action_final_trick_kitty_win",
            ],
            "calibration_metrics": ["win_brier", "win_ece10", "kitty_brier", "kitty_ece10"],
        },
    }
    if dataset.feature_schema_version == 2:
        manifest["level_utility"] = {
            "formula": "actor_team_sign * (1 + levels_awarded_to_winner) / 5",
            "normalizer": 5.0,
            "clamp": [-1.0, 1.0],
        }
    else:
        manifest["point_value"] = {
            "semantic": "normalized terminal point margin",
            "normalizer": 200.0,
        }
    temporary = f"{manifest_path}.tmp"
    with open(temporary, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, manifest_path)
    return manifest_path


def finite_metrics(metrics):
    return {
        key: (value if isinstance(value, int) or math.isfinite(value) else None)
        for key, value in metrics.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "data.csv"))
    parser.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(__file__), "..", "core", "src", "bot", "expert_model.onnx"
        ),
    )
    parser.add_argument("--manifest-out")
    parser.add_argument("--golden-out")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-groups", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--q-weight", type=float, default=1.0)
    parser.add_argument("--auxiliary-weight", type=float, default=0.25)
    parser.add_argument("--policy-target", choices=["teacher", "behaviour"], default="teacher")
    parser.add_argument("--early-stop-metric", choices=["policy", "value", "q"], default="policy")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument(
        "--allow-legacy-group-split",
        action="store_true",
        help="permit decision-level pseudo-game splitting for old CSVs (leakage-prone)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data):
        sys.exit(f"No training data at {args.data}; run gen_training_data first.")
    dataset = load_dataset(args.data)
    if not dataset.groups:
        sys.exit("No usable training groups found.")
    analyze_dataset(dataset)
    if args.analyze:
        return
    if not _HAS_TORCH:
        sys.exit("Training requires torch + onnx; install training/requirements.txt")
    if not dataset.game_ids_are_trajectories and not args.allow_legacy_group_split:
        sys.exit(
            "This CSV has no trustworthy game_id trajectory key. Refusing a leakage-prone "
            "decision split; regenerate schema-v3 data or explicitly pass "
            "--allow-legacy-group-split for a historical reproduction."
        )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train, val, train_games, val_games = split_by_game(
        dataset.groups, args.val_frac, args.seed
    )
    has_v = any(group.v_target is not None for group in dataset.groups)
    has_q = any(np.isfinite(group.q_target).any() for group in dataset.groups)
    if dataset.feature_schema_version == 1 and has_q:
        sys.exit("schema-v1/dim36 models do not support action-Q semantics")
    value_weight = args.value_weight if has_v else 0.0
    q_weight = args.q_weight if has_q else 0.0
    if args.early_stop_metric == "q" and q_weight == 0:
        sys.exit("--early-stop-metric q requires nonblank q_target rows")
    if args.early_stop_metric == "value" and value_weight == 0:
        sys.exit("--early-stop-metric value requires v_target/value rows")

    avg_candidates = sum(group.x.shape[0] for group in dataset.groups) / len(dataset.groups)
    print(
        f"Loaded {len(dataset.groups)} decisions from {len(train_games)+len(val_games)} games: "
        f"{len(train)} train/{len(val)} val decisions, avg {avg_candidates:.1f} candidates."
    )
    print(
        f"Objectives: policy={args.policy_weight:g}({args.policy_target}) "
        f"state-V={value_weight:g} action-Q={q_weight:g} "
        f"offline-outcome-aux={args.auxiliary_weight:g}; split is game-disjoint."
    )

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Training on {device}.")
    model = CandidateScorer(
        dataset.feature_dim,
        args.hidden,
        args.dropout,
        max(1, len(dataset.bucket_values)),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_metric = -float("inf")
    best_state = None
    stale = 0
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in padded_batches(train, args.batch_groups, device, args.policy_target):
            optimizer.zero_grad()
            total, *_parts = loss_for_batch(
                model,
                *batch,
                args.policy_weight,
                value_weight,
                q_weight,
                args.auxiliary_weight,
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0)
            optimizer.step()
            losses.append(float(total.item()))
        scheduler.step()
        metrics = evaluate(model, val, device, args.policy_target)
        monitor = metric_value(metrics, args.early_stop_metric)
        if monitor > best_metric + 1e-4:
            best_metric = monitor
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(
                f"epoch {epoch:3d} loss {np.mean(losses):.4f} "
                f"policy {metrics['policy_top1']:.1%} "
                f"V-rmse {metrics['value_rmse']:.3f} "
                f"Q-rmse {metrics['q_rmse']:.3f} Q-rank {metrics['q_rank']:.1%} "
                f"bucket {metrics['bucket_accuracy']:.1%} win-Brier {metrics['win_brier']:.3f} "
                f"(best {best_metric:.4f})"
            )
        if stale >= args.patience:
            print(f"Early stop at epoch {epoch} after {args.patience} stale epochs.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    final_metrics = finite_metrics(evaluate(model, val, device, args.policy_target))
    print(f"Best validation metrics: {json.dumps(final_metrics, sort_keys=True)}")

    temporary_model = f"{args.out}.tmp"
    outputs = export_onnx(
        model.cpu(), temporary_model, dataset.feature_dim, value_weight > 0, q_weight > 0
    )
    os.replace(temporary_model, args.out)
    golden_path = args.golden_out or f"{args.out}.golden.json"
    write_golden(model.cpu(), golden_path, dataset.feature_dim, outputs)
    manifest_path = write_model_manifest(
        args.out,
        golden_path,
        dataset,
        outputs,
        args,
        (train_games, val_games),
        final_metrics,
    )
    print(f"Exported outputs={outputs} to {os.path.abspath(args.out)}")
    print(f"Wrote PyTorch parity vectors to {os.path.abspath(golden_path)}")
    print(f"Wrote model contract manifest to {os.path.abspath(manifest_path)}")


if __name__ == "__main__":
    main()
