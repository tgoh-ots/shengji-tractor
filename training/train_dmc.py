#!/usr/bin/env python3
"""Train the Oracle tier's Deep Monte-Carlo (DMC) action-value net → ONNX.

This is the DouZero-style trainer: a *search-free* self-play agent. Instead of
cloning a teacher (the Expert path), Oracle learns a state-action value

    Q(s, a) ≈ E[ normalized oriented terminal margin | take candidate a in state s ]

by regressing on REALIZED Monte-Carlo returns from self-play. The behavior policy
that generated the data is epsilon-greedy on the *current* Q-net (bootstrapping
itself up over DMC iterations); the data generator is `core/examples/gen_dmc_data.rs`.

Crucially, the per-`(state, candidate)` feature encoding is the SAME
`bot::expert::candidate_features` (FEATURE_DIM=36) used by the Expert net, so the
trained model is a **1-output ONNX** (`x:[N,36] -> score:[N,1]`) that the existing
`tract-onnx` inference path scores directly: at serve time Oracle just takes the
`argmax_a Q(s,a)` over the legal candidates — NO determinized search at all.

Data CSV layout (one row per *taken* decision; produced by gen_dmc_data):
    f0,f1,...,f35,ret
where `ret` is the realized oriented terminal margin / expert::VALUE_NORM, in
[-1, 1] (so a `tanh` head fits). There is no `group`/`label` — DMC regresses the
taken action's value, and the net generalizes to score all legal actions at
inference.

Usage:
    python train_dmc.py --data dmc_data.csv --out ../core/src/bot/oracle_model.onnx \
        --epochs 60

Requires: torch, onnx, numpy (see requirements.txt). Trains on MPS/CPU in minutes.
"""

import argparse
import math
import os
import sys

import numpy as np

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except ModuleNotFoundError:
    torch = None
    nn = None
    _HAS_TORCH = False

# MUST match bot::expert::FEATURE_DIM (and train_expert.py).
FEATURE_DIM = 36


def load_rows(path):
    """Load the DMC CSV into (X [N, FEATURE_DIM] float32, y [N] float32).

    Accepts a header `f0..f{D-1},ret`. Tolerates an optional leading `group`
    column (ignored) for convenience, by detecting the width.
    """
    raw = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    if raw.ndim == 1:
        raw = raw[None, :]
    ncol = raw.shape[1]
    if ncol == FEATURE_DIM + 1:  # f0..fD-1, ret
        X = raw[:, :FEATURE_DIM]
        y = raw[:, FEATURE_DIM]
    elif ncol == FEATURE_DIM + 2:  # group, f0..fD-1, ret
        X = raw[:, 1 : 1 + FEATURE_DIM]
        y = raw[:, 1 + FEATURE_DIM]
    else:
        raise SystemExit(
            f"CSV has {ncol} cols, expected {FEATURE_DIM + 1} (f0..f{FEATURE_DIM - 1},ret) "
            f"or {FEATURE_DIM + 2} (group,...). Did FEATURE_DIM change?"
        )
    return X.astype(np.float32), y.astype(np.float32)


class QNet(_HAS_TORCH and nn.Module or object):
    """Q(s,a) over one candidate feature vector. Shared trunk identical in shape
    to the Expert scorer (36 -> hidden -> hidden -> hidden//2) then a single
    `tanh` value output, so the exported graph is a drop-in 1-output model for the
    existing `tract-onnx` inference path (argmax over `score`)."""

    def __init__(self, in_dim=FEATURE_DIM, hidden=128, dropout=0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden // 2, 1)

    def forward(self, x):
        return torch.tanh(self.head(self.trunk(x)))  # [N, 1] in [-1, 1]


def export_onnx(model, out_path):
    """Export Q to a 1-output ONNX (`x:[N,D] -> score:[N,1]`), opset 13, legacy
    exporter — byte-compatible with the Expert inference path so Oracle's net
    loads via SHENGJI_EXPERT_MODEL_PATH and is served by choose_play_expert."""
    model.eval()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    dummy = torch.zeros((1, FEATURE_DIM), dtype=torch.float32)
    kwargs = dict(
        input_names=["x"],
        output_names=["score"],
        dynamic_axes={"x": {0: "N"}, "score": {0: "N"}},
        opset_version=13,
    )
    try:
        torch.onnx.export(model, dummy, out_path, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, out_path, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(__file__), "..", "core", "src", "bot", "oracle_model.onnx"
        ),
    )
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--init",
        default="",
        help="optional path to a prior QNet state_dict (.pt) to warm-start a DMC iteration",
    )
    args = ap.parse_args()

    if not _HAS_TORCH:
        sys.exit("Training requires PyTorch. pip install -r requirements.txt")
    if not os.path.exists(args.data):
        sys.exit(f"No DMC data at {args.data}. Generate with gen_dmc_data first.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    X, y = load_rows(args.data)
    n = X.shape[0]
    print(f"Loaded {n} DMC (s,a,return) rows; return mean {y.mean():.3f} std {y.std():.3f}.")

    perm = np.random.permutation(n)
    n_val = max(1, int(n * args.val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Training on {device}.")
    Xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y).to(device)

    model = QNet(hidden=args.hidden, dropout=args.dropout).to(device)
    if args.init and os.path.exists(args.init):
        model.load_state_dict(torch.load(args.init, map_location=device))
        print(f"Warm-started from {args.init}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.MSELoss()

    tr = torch.from_numpy(tr_idx).to(device)
    va = torch.from_numpy(val_idx).to(device)

    def val_rmse():
        model.eval()
        with torch.no_grad():
            pred = model(Xt[va]).squeeze(1)
            return math.sqrt(float(((pred - yt[va]) ** 2).mean().item()))

    best, best_state, since = float("inf"), None, 0
    for epoch in range(args.epochs):
        model.train()
        order = tr[torch.randperm(tr.shape[0], device=device)]
        total, nb = 0.0, 0
        for s in range(0, order.shape[0], args.batch):
            idx = order[s : s + args.batch]
            opt.zero_grad()
            loss = lossf(model(Xt[idx]).squeeze(1), yt[idx])
            loss.backward()
            opt.step()
            total += float(loss.item())
            nb += 1
        sched.step()
        vr = val_rmse()
        if vr < best - 1e-4:
            best, since = vr, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(
                f"epoch {epoch:3d}  train-MSE {total / max(1, nb):.4f}  "
                f"val-RMSE {vr:.4f}  (best {best:.4f}, lr {sched.get_last_lr()[0]:.2e})"
            )
        if since >= args.patience:
            print(f"Early stop at epoch {epoch} (no val improvement for {args.patience}).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best val-RMSE {best:.4f}.")

    # Save a .pt next to the onnx so the next DMC iteration can warm-start.
    pt_path = os.path.splitext(args.out)[0] + ".pt"
    torch.save(model.state_dict(), pt_path)
    export_onnx(model.cpu(), args.out)
    print(f"Exported 1-output Q ONNX -> {os.path.abspath(args.out)} (state_dict -> {pt_path})")


if __name__ == "__main__":
    main()
