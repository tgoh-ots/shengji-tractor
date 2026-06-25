#!/usr/bin/env python3
"""Train the Expert tier's candidate-scoring MLP and export it to ONNX.

This is a behavioral-cloning / distillation trainer. The data (produced by the
Rust `gen_training_data` example) contains, for each PLAY-phase decision, one row
per legal candidate move:

    group, f0..f{D-1}, label

where `group` ties together the candidates of a single decision and `label == 1`
on exactly the candidate the strong (Omniscient, perfect-information) TEACHER
picked. The features (`f*`) are HONEST-only — derived from the redacted view — so
the net learns to imitate perfect-info choices from honest observation.

Model: a small MLP that maps one candidate's feature vector to a scalar logit.
Training objective: a softmax CROSS-ENTROPY over the candidates within each
group, i.e. the teacher's candidate should receive the highest logit. (This is a
listwise / "learning-to-rank" loss; padding makes variable-length groups
batchable.)

Export: ONNX with input `x : [N, D]` (a batch of N candidate feature vectors) and
output `[N, 1]` logits, so the Rust `tract-onnx` inference path can score a whole
candidate set in one call. Uses opset 13 (well within tract's supported range).

Usage:
    python train_expert.py \
        --data data.csv \
        --out ../core/src/bot/expert_model.onnx \
        --epochs 60

Requires: torch, onnx, numpy (see requirements.txt).
"""

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

# The feature dimension MUST match `bot::expert::FEATURE_DIM` in the Rust crate.
FEATURE_DIM = 28


def load_groups(path):
    """Load the CSV into per-decision groups.

    Returns a list of (X, y) where X is [k, FEATURE_DIM] float32 and y is the
    index of the teacher-chosen candidate within the group (exactly one label==1).
    Groups without a positive label, or with <2 candidates, are dropped.
    """
    by_group_feats = defaultdict(list)
    by_group_labels = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        # Sanity check the header width.
        expected = 1 + FEATURE_DIM + 1
        if len(header) != expected:
            raise SystemExit(
                f"CSV header has {len(header)} cols, expected {expected} "
                f"(group + {FEATURE_DIM} features + label). Did FEATURE_DIM change?"
            )
        for parts in reader:
            if not parts:
                continue
            g = int(parts[0])
            feats = [float(x) for x in parts[1 : 1 + FEATURE_DIM]]
            label = int(parts[1 + FEATURE_DIM])
            by_group_feats[g].append(feats)
            by_group_labels[g].append(label)

    groups = []
    for g, feats in by_group_feats.items():
        labels = by_group_labels[g]
        if len(feats) < 2:
            continue
        if sum(labels) != 1:
            # Keep the data clean: exactly one teacher choice per decision.
            continue
        y = labels.index(1)
        groups.append((np.asarray(feats, dtype=np.float32), y))
    return groups


class CandidateScorer(nn.Module):
    """Small MLP scoring ONE candidate feature vector -> scalar logit.

    Kept intentionally tiny (two hidden layers) so it is fast to train on
    CPU/MPS and cheap to run per-candidate in tract at serving time.
    """

    def __init__(self, in_dim=FEATURE_DIM, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        # x: [N, in_dim] -> [N, 1]
        return self.net(x)


def make_padded_batches(groups, batch_groups, device):
    """Yield padded batches of groups for listwise training.

    Each batch is (feats [B, K, D], target [B], mask [B, K]) where K is the max
    group size in the batch and mask marks the real (non-padding) candidates.
    """
    order = np.random.permutation(len(groups))
    for start in range(0, len(groups), batch_groups):
        idxs = order[start : start + batch_groups]
        batch = [groups[i] for i in idxs]
        k = max(g[0].shape[0] for g in batch)
        b = len(batch)
        feats = np.zeros((b, k, FEATURE_DIM), dtype=np.float32)
        mask = np.zeros((b, k), dtype=np.float32)
        target = np.zeros((b,), dtype=np.int64)
        for i, (X, y) in enumerate(batch):
            n = X.shape[0]
            feats[i, :n] = X
            mask[i, :n] = 1.0
            target[i] = y
        yield (
            torch.from_numpy(feats).to(device),
            torch.from_numpy(target).to(device),
            torch.from_numpy(mask).to(device),
        )


def listwise_loss(model, feats, target, mask):
    """Softmax cross-entropy over each group's candidates (the teacher's pick
    should get the highest logit). Padding candidates are masked out with -inf so
    they never carry probability."""
    b, k, d = feats.shape
    logits = model(feats.reshape(b * k, d)).reshape(b, k)
    neg_inf = torch.finfo(logits.dtype).min
    logits = torch.where(mask > 0, logits, torch.full_like(logits, neg_inf))
    log_probs = torch.log_softmax(logits, dim=1)
    chosen = log_probs[torch.arange(b, device=logits.device), target]
    return -chosen.mean()


def evaluate(model, groups, device):
    """Top-1 accuracy: fraction of decisions where the model's argmax candidate
    is the teacher's pick. This is the headline distillation metric."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in groups:
            t = torch.from_numpy(X).to(device)
            scores = model(t).reshape(-1)
            pred = int(torch.argmax(scores).item())
            if pred == y:
                correct += 1
    model.train()
    return correct / max(1, len(groups))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "data.csv"))
    ap.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(__file__), "..", "core", "src", "bot", "expert_model.onnx"
        ),
    )
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-groups", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists(args.data):
        sys.exit(
            f"No training data at {args.data}. Generate it first with:\n"
            f"  cargo run --release --example gen_training_data"
        )

    groups = load_groups(args.data)
    if not groups:
        sys.exit("No usable training groups found in the data.")

    # Train/val split by group.
    perm = np.random.permutation(len(groups))
    n_val = max(1, int(len(groups) * args.val_frac))
    val_idx = set(perm[:n_val].tolist())
    train_groups = [g for i, g in enumerate(groups) if i not in val_idx]
    val_groups = [g for i, g in enumerate(groups) if i in val_idx]

    avg_cands = sum(g[0].shape[0] for g in groups) / len(groups)
    print(
        f"Loaded {len(groups)} decisions "
        f"({len(train_groups)} train / {len(val_groups)} val), "
        f"avg {avg_cands:.1f} candidates/decision."
    )
    # A random baseline picks the right candidate ~1/avg_cands of the time.
    print(f"Random-guess top-1 baseline ≈ {1.0 / avg_cands:.1%}")

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Training on {device}.")

    model = CandidateScorer(hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_val = -1.0
    best_state = None
    for epoch in range(args.epochs):
        total = 0.0
        nb = 0
        for feats, target, mask in make_padded_batches(
            train_groups, args.batch_groups, device
        ):
            opt.zero_grad()
            loss = listwise_loss(model, feats, target, mask)
            loss.backward()
            opt.step()
            total += float(loss.item())
            nb += 1
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            tr_acc = evaluate(model, train_groups, device)
            val_acc = evaluate(model, val_groups, device)
            print(
                f"epoch {epoch:3d}  loss {total / max(1, nb):.4f}  "
                f"train top-1 {tr_acc:.1%}  val top-1 {val_acc:.1%}"
            )
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    final_val = evaluate(model, val_groups, device)
    final_train = evaluate(model, train_groups, device)
    print(f"Best val top-1 {final_val:.1%} (train {final_train:.1%}).")

    # Export to ONNX (input [N, D] -> output [N, 1]) with a dynamic batch dim.
    export_onnx(model.cpu(), args.out)
    print(f"Exported ONNX model to {os.path.abspath(args.out)}")


def export_onnx(model, out_path):
    model.eval()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    dummy = torch.zeros((1, FEATURE_DIM), dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["x"],
        output_names=["score"],
        dynamic_axes={"x": {0: "N"}, "score": {0: "N"}},
        opset_version=13,
    )


if __name__ == "__main__":
    main()
