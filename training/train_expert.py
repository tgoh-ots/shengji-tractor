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

# torch is only needed for TRAINING/EXPORT, not for the pure-numpy `--analyze`
# data diagnostic. Import it lazily so the aliasing analysis runs with just numpy
# (no heavy torch install). When torch is absent, `CandidateScorer` still DEFINES
# (inheriting `object`); only INSTANTIATING it — the training path — needs torch.
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

# The feature dimension MUST match `bot::expert::FEATURE_DIM` in the Rust crate.
FEATURE_DIM = 36


def load_groups(path):
    """Load the CSV into per-decision groups.

    Returns a list of (X, y, value) where X is [k, FEATURE_DIM] float32, y is the
    index of the teacher-chosen candidate within the group (exactly one label==1),
    and value is the group's VALUE target (the normalized realized terminal margin,
    constant across the group) or None when the CSV has no `value` column
    (policy-only / legacy data). Groups without a positive label, or with <2
    candidates, are dropped.

    Accepts BOTH the policy-only layout (group + features + label) and the
    value-augmented layout (… + value), so legacy CSVs still train the policy head.
    """
    by_group_feats = defaultdict(list)
    by_group_labels = defaultdict(list)
    by_group_value = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        # Accept the policy-only width (…+label) or the value-augmented width
        # (…+label+value); anything else means FEATURE_DIM drifted.
        policy_only = 1 + FEATURE_DIM + 1
        with_value = policy_only + 1
        if len(header) not in (policy_only, with_value):
            raise SystemExit(
                f"CSV header has {len(header)} cols, expected {policy_only} "
                f"(group + {FEATURE_DIM} features + label) or {with_value} "
                f"(… + value). Did FEATURE_DIM change?"
            )
        has_value = len(header) == with_value
        for parts in reader:
            if not parts:
                continue
            g = int(parts[0])
            feats = [float(x) for x in parts[1 : 1 + FEATURE_DIM]]
            label = int(parts[1 + FEATURE_DIM])
            by_group_feats[g].append(feats)
            by_group_labels[g].append(label)
            if has_value:
                # The value is constant within a group; last writer wins (equal).
                by_group_value[g] = float(parts[1 + FEATURE_DIM + 1])

    groups = []
    for g, feats in by_group_feats.items():
        labels = by_group_labels[g]
        if len(feats) < 2:
            continue
        if sum(labels) != 1:
            # Keep the data clean: exactly one teacher choice per decision.
            continue
        y = labels.index(1)
        value = by_group_value.get(g) if has_value else None
        groups.append((np.asarray(feats, dtype=np.float32), y, value))
    return groups


def analyze_aliasing(groups, granularities=(2, 1, 0)):
    """Estimate the BEHAVIORAL-CLONING CEILING: how often the HONEST features
    simply cannot identify the teacher's pick.

    The Expert net is trained to imitate a PERFECT-INFORMATION teacher from
    HONEST-only features. For many positions that is information-theoretically
    impossible: two candidates with identical honest features can carry different
    labels depending on UNSEEN cards. This irreducible "aliasing floor" is a hard
    cap that more epochs / a bigger net / more data cannot cross — so its size
    tells you whether to keep polishing the policy (low floor) or invest elsewhere
    (high floor → value head / kitty, per docs/bot-training-roadmap.md).

    Two complementary metrics, reported at several rounding granularities (coarser
    rounding collides more vectors → a more conservative/looser floor):

      * row-level Bayes error — bucket every candidate row by its rounded feature
        vector; the best any function of these features can do is predict each
        bucket's MAJORITY label, so the majority-vote error (rows whose label !=
        their bucket majority) is a floor on per-candidate error.
      * decision-level unwinnable rate — the fraction of decisions where the
        teacher's chosen candidate shares a rounded feature key with a REJECTED
        candidate in the SAME decision: an UNWINNABLE decision for any
        honest-feature scorer (the listwise top-1 task can never separate them).
    """
    from collections import defaultdict

    n_rows = sum(g[0].shape[0] for g in groups)
    n_groups = len(groups)
    print("\n=== label-aliasing analysis (behavioral-cloning ceiling) ===")
    avg_c = n_rows / max(1, n_groups)
    print(f"{n_groups} decisions, {n_rows} candidate rows.")
    print(f"avg {avg_c:.2f} candidates/decision; random-guess top-1 ≈ {1.0 / avg_c:.1%}")

    for dec in granularities:
        buckets_pos = defaultdict(int)
        buckets_tot = defaultdict(int)
        collided = 0
        for X, y, _value in groups:
            keys = [tuple(np.round(X[i], dec).tolist()) for i in range(X.shape[0])]
            for i, k in enumerate(keys):
                buckets_tot[k] += 1
                if i == y:
                    buckets_pos[k] += 1
            chosen = keys[y]
            if any(j != y and keys[j] == chosen for j in range(len(keys))):
                collided += 1
        # Row-level majority-vote (Bayes) error.
        maj_err_rows = sum(
            tot - max(buckets_pos[k], tot - buckets_pos[k])
            for k, tot in buckets_tot.items()
        )
        maj_err = maj_err_rows / max(1, n_rows)
        coll_frac = collided / max(1, n_groups)
        print(
            f"  round={dec}dp: row Bayes-error floor {maj_err:5.1%}  |  "
            f"unwinnable decisions {coll_frac:5.1%}  "
            f"({len(buckets_tot)} distinct feature keys)"
        )
    print(
        "Higher = more of the teacher signal is unidentifiable from honest "
        "features (a ceiling that cloning / bigger nets / more data cannot cross)."
    )


class CandidateScorer(_NN_MODULE):
    """Multi-task MLP over ONE candidate feature vector: a POLICY logit (the
    distillation prior) and a VALUE estimate (the search leaf evaluator).

    A shared 3-hidden-layer trunk (`in -> hidden -> hidden -> hidden//2`) with ReLU
    + dropout, then two linear heads off the same trunk representation:
      * `policy_head`  -> scalar logit  (ranked across a decision's candidates);
      * `value_head`   -> scalar, `tanh`-squashed to [-1, 1] (the normalized
        realized terminal margin oriented for the acting team).
    Still tiny — trains in seconds and runs cheaply in tract at serving time.
    Dropout is identity in eval/export and `tanh` is a standard ONNX op, so the
    exported graph stays within tract's supported op set.

    `forward` returns `(policy [N,1], value [N,1])`. When trained on policy-only
    data the value head is untrained, so `export_onnx(..., with_value=False)`
    exports ONLY the policy output (the legacy single-output contract).
    """

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
        self.policy_head = nn.Linear(hidden // 2, 1)
        self.value_head = nn.Linear(hidden // 2, 1)

    def forward(self, x):
        # x: [N, in_dim] -> (policy [N, 1], value [N, 1])
        z = self.trunk(x)
        policy = self.policy_head(z)
        value = torch.tanh(self.value_head(z))
        return policy, value


class PolicyOnly(_NN_MODULE):
    """Thin wrapper exporting ONLY the policy output (the legacy single-output
    `score` contract), used when training had no value targets so the value head
    is untrained and must not be shipped."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[0]


def _group_value(g):
    """The group's value target as a float (0.0 when absent / policy-only)."""
    v = g[2]
    return 0.0 if v is None else float(v)


def make_padded_batches(groups, batch_groups, device):
    """Yield padded batches of groups for listwise training.

    Each batch is (feats [B, K, D], target [B], mask [B, K], value [B]) where K is
    the max group size in the batch, mask marks the real (non-padding) candidates,
    and value is the group's (normalized) value target.
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
        value = np.zeros((b,), dtype=np.float32)
        for i, (X, y, _v) in enumerate(batch):
            n = X.shape[0]
            feats[i, :n] = X
            mask[i, :n] = 1.0
            target[i] = y
            value[i] = _group_value(batch[i])
        yield (
            torch.from_numpy(feats).to(device),
            torch.from_numpy(target).to(device),
            torch.from_numpy(mask).to(device),
            torch.from_numpy(value).to(device),
        )


def listwise_loss(model, feats, target, mask, value, value_weight):
    """POLICY: softmax cross-entropy over each group's candidates (the teacher's
    pick should get the highest logit); padding masked out with -inf. VALUE (when
    `value_weight > 0`): masked MSE of each real candidate's `tanh` value against
    the group's (broadcast) terminal-margin target. Total = CE + w * MSE."""
    b, k, d = feats.shape
    policy, val = model(feats.reshape(b * k, d))
    logits = policy.reshape(b, k)
    neg_inf = torch.finfo(logits.dtype).min
    logits = torch.where(mask > 0, logits, torch.full_like(logits, neg_inf))
    log_probs = torch.log_softmax(logits, dim=1)
    chosen = log_probs[torch.arange(b, device=logits.device), target]
    ce = -chosen.mean()
    if value_weight <= 0:
        return ce
    val = val.reshape(b, k)
    vt = value.unsqueeze(1).expand(b, k)
    sq = (val - vt) ** 2 * mask
    mse = sq.sum() / mask.sum().clamp(min=1.0)
    return ce + value_weight * mse


def build_eval_tensors(groups, device):
    """Pre-pad all groups into a single (G, Kmax, D) tensor + (G, Kmax) mask +
    (G,) targets + (G,) value targets so evaluation is one vectorized forward pass
    instead of G tiny per-group passes (pathologically slow on MPS). Returns None
    for an empty group list."""
    if not groups:
        return None
    kmax = max(g[0].shape[0] for g in groups)
    g = len(groups)
    feats = np.zeros((g, kmax, FEATURE_DIM), dtype=np.float32)
    mask = np.zeros((g, kmax), dtype=np.float32)
    target = np.zeros((g,), dtype=np.int64)
    value = np.zeros((g,), dtype=np.float32)
    for i, grp in enumerate(groups):
        X, y, _v = grp
        n = X.shape[0]
        feats[i, :n] = X
        mask[i, :n] = 1.0
        target[i] = y
        value[i] = _group_value(grp)
    return (
        torch.from_numpy(feats).to(device),
        torch.from_numpy(mask).to(device),
        torch.from_numpy(target).to(device),
        torch.from_numpy(value).to(device),
    )


def evaluate(model, eval_tensors, device, chunk=4096):
    """Returns (top1, value_rmse). top1 = fraction of decisions whose argmax POLICY
    candidate is the teacher's pick (the headline distillation metric). value_rmse =
    RMSE of the per-candidate `tanh` value vs the group target over real candidates
    (NaN if no value targets). Vectorized + chunked, single host sync at the end."""
    if eval_tensors is None:
        return 0.0, float("nan")
    feats, mask, target, value = eval_tensors
    g, kmax, d = feats.shape
    model.eval()
    correct = 0
    sq_sum = 0.0
    cnt = 0.0
    neg_inf = torch.finfo(feats.dtype).min
    with torch.no_grad():
        for start in range(0, g, chunk):
            fb = feats[start : start + chunk]
            mb = mask[start : start + chunk]
            tb = target[start : start + chunk]
            vb = value[start : start + chunk]
            b = fb.shape[0]
            policy, val = model(fb.reshape(b * kmax, d))
            logits = policy.reshape(b, kmax)
            logits = torch.where(mb > 0, logits, torch.full_like(logits, neg_inf))
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == tb).sum().item())
            val = val.reshape(b, kmax)
            vt = vb.unsqueeze(1).expand(b, kmax)
            sq_sum += float((((val - vt) ** 2) * mb).sum().item())
            cnt += float(mb.sum().item())
    model.train()
    rmse = math.sqrt(sq_sum / cnt) if cnt > 0 else float("nan")
    return correct / max(1, g), rmse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "data.csv"))
    ap.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(__file__), "..", "core", "src", "bot", "expert_model.onnx"
        ),
    )
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-groups", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument(
        "--patience",
        type=int,
        default=25,
        help="early-stop if val top-1 hasn't improved for this many evals",
    )
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--value-weight",
        type=float,
        default=1.0,
        help="weight on the VALUE-head MSE loss (0 disables the value head; it is "
        "auto-0 when the data has no `value` column). The exported model is "
        "2-output (score, value) iff the value head was trained.",
    )
    ap.add_argument(
        "--analyze",
        action="store_true",
        help="report the label-aliasing / behavioral-cloning ceiling and exit "
        "(no training). See analyze_aliasing.",
    )
    args = ap.parse_args()

    np.random.seed(args.seed)

    if not os.path.exists(args.data):
        sys.exit(
            f"No training data at {args.data}. Generate it first with:\n"
            f"  cargo run --release --example gen_training_data"
        )

    groups = load_groups(args.data)
    if not groups:
        sys.exit("No usable training groups found in the data.")

    if args.analyze:
        analyze_aliasing(groups)
        return

    if not _HAS_TORCH:
        sys.exit(
            "Training requires PyTorch. Install the training deps first:\n"
            "  pip install -r requirements.txt\n"
            "(The --analyze data diagnostic runs with just numpy.)"
        )
    # torch is required from here on (seeding, model, training, export).
    torch.manual_seed(args.seed)

    # Train/val split by group.
    perm = np.random.permutation(len(groups))
    n_val = max(1, int(len(groups) * args.val_frac))
    val_idx = set(perm[:n_val].tolist())
    train_groups = [g for i, g in enumerate(groups) if i not in val_idx]
    val_groups = [g for i, g in enumerate(groups) if i in val_idx]

    avg_cands = sum(g[0].shape[0] for g in groups) / len(groups)
    # Train the value head iff the data carries value targets AND the weight > 0.
    has_value = any(g[2] is not None for g in groups)
    value_weight = args.value_weight if has_value else 0.0
    print(
        f"Loaded {len(groups)} decisions "
        f"({len(train_groups)} train / {len(val_groups)} val), "
        f"avg {avg_cands:.1f} candidates/decision."
    )
    print(
        "Value head: "
        + (
            f"ON (weight {value_weight}) — exporting 2-output (score, value)."
            if value_weight > 0
            else "OFF (no `value` column or --value-weight 0) — exporting policy-only."
        )
    )
    # A random baseline picks the right candidate ~1/avg_cands of the time.
    print(f"Random-guess top-1 baseline ≈ {1.0 / avg_cands:.1%}")

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Training on {device}.")

    model = CandidateScorer(hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # Cosine-anneal the learning rate over the full epoch budget for a smooth
    # decay; combined with early stopping this avoids over/under-fitting.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # Pre-pad eval sets ONCE so per-epoch accuracy is a single vectorized pass.
    train_eval = build_eval_tensors(train_groups, device)
    val_eval = build_eval_tensors(val_groups, device)

    best_val = -1.0
    best_state = None
    epochs_since_improve = 0
    stop = False
    for epoch in range(args.epochs):
        total = 0.0
        nb = 0
        for feats, target, mask, value in make_padded_batches(
            train_groups, args.batch_groups, device
        ):
            opt.zero_grad()
            loss = listwise_loss(model, feats, target, mask, value, value_weight)
            loss.backward()
            opt.step()
            total += float(loss.item())
            nb += 1
        sched.step()
        # Evaluate every epoch so early stopping is responsive; printing is
        # throttled to keep the log readable. Early-stop on POLICY top-1 (the
        # head used as the search prior).
        val_acc, val_rmse = evaluate(model, val_eval, device)
        if val_acc > best_val + 1e-4:
            best_val = val_acc
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            tr_acc, _ = evaluate(model, train_eval, device)
            vrmse = f"  val value-RMSE {val_rmse:.3f}" if value_weight > 0 else ""
            print(
                f"epoch {epoch:3d}  loss {total / max(1, nb):.4f}  "
                f"train top-1 {tr_acc:.1%}  val top-1 {val_acc:.1%}{vrmse}  "
                f"(best {best_val:.1%}, lr {sched.get_last_lr()[0]:.2e})"
            )
        if epochs_since_improve >= args.patience:
            print(
                f"Early stop at epoch {epoch} "
                f"(no val improvement for {args.patience} epochs)."
            )
            stop = True
        if stop:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    final_val, final_val_rmse = evaluate(model, val_eval, device)
    final_train, _ = evaluate(model, train_eval, device)
    vmsg = f", val value-RMSE {final_val_rmse:.3f}" if value_weight > 0 else ""
    print(f"Best val top-1 {final_val:.1%} (train {final_train:.1%}){vmsg}.")

    # Export to ONNX with a dynamic batch dim. 2-output (score, value) iff the
    # value head was trained; else policy-only (the legacy single-output contract).
    export_onnx(model.cpu(), args.out, with_value=value_weight > 0)
    print(
        f"Exported {'2-output (score, value)' if value_weight > 0 else 'policy-only'} "
        f"ONNX model to {os.path.abspath(args.out)}"
    )


def export_onnx(model, out_path, with_value=False):
    """Export the scorer to ONNX with a dynamic batch dim (input `x:[N,D]`).

    With `with_value`, exports BOTH outputs: `score [N,1]` (policy logits) and
    `value [N,1]` (`tanh` terminal-margin estimate). Otherwise exports only
    `score` (via the `PolicyOnly` wrapper) — the legacy single-output contract the
    embedded model uses, so an untrained value head is never shipped.

    We force the legacy TorchScript-based exporter (`dynamo=False`): a minimal
    Gemm/Relu/Tanh graph at opset 13 that the pure-Rust `tract-onnx` runtime loads
    without the `onnxscript` package the dynamo exporter pulls in. Dropout is a
    no-op in eval mode, so it doesn't appear in the graph.
    """
    model.eval()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    dummy = torch.zeros((1, FEATURE_DIM), dtype=torch.float32)
    if with_value:
        export_model = model
        output_names = ["score", "value"]
        dynamic_axes = {
            "x": {0: "N"},
            "score": {0: "N"},
            "value": {0: "N"},
        }
    else:
        export_model = PolicyOnly(model)
        output_names = ["score"]
        dynamic_axes = {"x": {0: "N"}, "score": {0: "N"}}
    export_kwargs = dict(
        input_names=["x"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=13,
    )
    try:
        # PyTorch >= 2.x: explicitly select the legacy exporter.
        torch.onnx.export(export_model, dummy, out_path, dynamo=False, **export_kwargs)
    except TypeError:
        # Older PyTorch without a `dynamo` kwarg: the legacy path is the default.
        torch.onnx.export(export_model, dummy, out_path, **export_kwargs)


if __name__ == "__main__":
    main()
