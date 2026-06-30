#!/usr/bin/env python3
"""Train an offline honest card-location belief model with legality masking."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os

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

DATASET_SCHEMA_VERSION = 1
SUPPORTED_FEATURE_SCHEMAS = {(1, 1, 20), (2, 2, 128)}
TARGET_CLASSES = ["next-seat", "opposite-seat", "previous-seat", "kitty"]
SUPPORTED_GAME_CONTRACT = "tractor:4p:2x-standard:kitty8:no-removed"
SUPPORTED_GENERATOR_BEHAVIOUR = "easy-play/expert-bid"
SUPPORTED_BEHAVIOUR_POLICY_DOMAIN = "bidding=expert;exchange=easy;play=easy"
MODEL_CONTRACT = "offline_honest_card_location_belief"
SERVING_STATUS = "experimental_candidate"
LEGALITY_CONTRACT = "mask=1 iff destination has capacity and no public effective-suit void"
TARGET_SEMANTICS = (
    "per-hidden-card destination marginals excluding publicly pinned holdings; rows in a snapshot are correlated physical copies"
)
PROPOSAL_FACTORIZATION = (
    "per-card destination marginals multiplied over physical-copy assignments; approximate joint"
)
GOLDEN_VECTOR_CONTRACT = (
    "synthetic deterministic tensor parity only; state-derived encoder golden pending"
)
ENCODER_SOURCE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "core", "src", "bot", "belief.rs")
)
PUBLIC_HISTORY_CONTRACTS = {
    1: "schema-v1 aggregate public-state features",
    2: "schema-v1 prefix plus ordered tail of 4 public bids and 8 public plays",
}


def belief_feature_names(feature_schema_version):
    names = [f"b{i}" for i in range(20)]
    if feature_schema_version == 1:
        return names
    if feature_schema_version != 2:
        raise SystemExit(f"unsupported belief feature schema {feature_schema_version}")
    names += [
        "seq.completed_tricks",
        "seq.current_trick_occupancy",
        "seq.bid_count",
        "seq.failed_throw_count",
    ]
    bid_names = [
        "present",
        "relative_actor",
        "sequence_position",
        "card_identity",
        "count",
        "epoch",
    ]
    for event in range(4):
        names += [f"seq.bid_{event}.{name}" for name in bid_names]
    play_names = [
        "present",
        "relative_actor",
        "trick_recency",
        "position_in_trick",
        "card_count",
        "point_density",
        "trump_fraction",
        "led_suit_fraction",
        "failed_throw_count",
        "first_effective_suit",
    ]
    for event in range(8):
        names += [f"seq.play_{event}.{name}" for name in play_names]
    assert len(names) == 128
    return names


def belief_encoder_contract(feature_schema_version):
    if feature_schema_version in (1, 2):
        return f"shengji-belief-encoder-v{feature_schema_version}"
    raise SystemExit(f"unsupported belief encoder schema {feature_schema_version}")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_dataset_sidecar(path, allow_missing=False):
    sidecar_path = f"{path}.manifest.json"
    if not os.path.exists(sidecar_path):
        if allow_missing:
            return None
        raise SystemExit(
            f"belief dataset sidecar is required: {sidecar_path}; "
            "use --allow-unsafe-no-sidecar only for non-serving experiments"
        )
    try:
        with open(sidecar_path) as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid belief dataset sidecar: {error}") from error
    if manifest.get("dataset_schema_version") != DATASET_SCHEMA_VERSION:
        raise SystemExit(
            f"belief dataset sidecar schema must be {DATASET_SCHEMA_VERSION}"
        )
    schema_tuple = (
        manifest.get("manifest_version"),
        manifest.get("feature_schema_version"),
        manifest.get("feature_dim"),
    )
    if schema_tuple not in SUPPORTED_FEATURE_SCHEMAS:
        raise SystemExit(
            "belief dataset sidecar (manifest_version, feature_schema_version, feature_dim) "
            "must be (1,1,20) or (2,2,128)"
        )
    expected_features = belief_feature_names(manifest["feature_schema_version"])
    if manifest.get("feature_names") != expected_features:
        raise SystemExit("belief dataset sidecar feature_names order/semantics mismatch")
    if manifest.get("encoder_contract") != belief_encoder_contract(
        manifest["feature_schema_version"]
    ):
        raise SystemExit("belief dataset sidecar encoder contract mismatch")
    declared_encoder_sha256 = manifest.get("encoder_source_sha256")
    if not is_sha256(declared_encoder_sha256):
        raise SystemExit("belief dataset sidecar encoder source hash is malformed")
    if not os.path.exists(ENCODER_SOURCE_PATH):
        raise SystemExit(f"belief encoder source is unavailable: {ENCODER_SOURCE_PATH}")
    if declared_encoder_sha256 != sha256(ENCODER_SOURCE_PATH):
        raise SystemExit("belief dataset was generated by a different encoder source")
    if manifest.get("target_classes") != TARGET_CLASSES:
        raise SystemExit(
            "belief dataset sidecar target_classes order must be "
            + repr(TARGET_CLASSES)
        )
    if manifest.get("target_semantics") != TARGET_SEMANTICS:
        raise SystemExit("belief dataset sidecar target semantics mismatch")
    if manifest.get("publicly_pinned_targets_excluded") is not True:
        raise SystemExit("belief dataset must exclude publicly pinned holding targets")
    if manifest.get("legality_contract") != LEGALITY_CONTRACT:
        raise SystemExit("belief dataset sidecar legality contract mismatch")
    if manifest.get("public_history_contract") != PUBLIC_HISTORY_CONTRACTS[
        manifest["feature_schema_version"]
    ]:
        raise SystemExit("belief dataset sidecar public-history contract mismatch")
    declared_contract = (
        manifest.get("supported_game_contract")
        or manifest.get("game_contract")
        or manifest.get("game_config")
    )
    if declared_contract != SUPPORTED_GAME_CONTRACT:
        raise SystemExit(
            f"unsupported belief game contract {declared_contract!r}; expected "
            f"{SUPPORTED_GAME_CONTRACT!r}"
        )
    if (
        manifest.get("behaviour") != SUPPORTED_GENERATOR_BEHAVIOUR
    ):
        raise SystemExit(
            f"unsupported belief generator behaviour {manifest['behaviour']!r}; "
            f"expected {SUPPORTED_GENERATOR_BEHAVIOUR!r}"
        )
    if manifest.get("behaviour_policy_domain") != SUPPORTED_BEHAVIOUR_POLICY_DOMAIN:
        raise SystemExit(
            "belief dataset sidecar behaviour-policy domain mismatch; expected "
            f"{SUPPORTED_BEHAVIOUR_POLICY_DOMAIN!r}"
        )
    if manifest.get("games_dropped") is not None and (
        not isinstance(manifest["games_dropped"], int)
        or manifest["games_dropped"] < 0
    ):
        raise SystemExit("belief dataset sidecar games_dropped must be nonnegative")
    requested = manifest.get("games_requested")
    completed = manifest.get("games_completed")
    dropped = manifest.get("games_dropped")
    if (
        not isinstance(requested, int)
        or not isinstance(completed, int)
        or not isinstance(dropped, int)
        or min(requested, completed, dropped) < 0
        or completed + dropped != requested
    ):
        raise SystemExit("belief dataset sidecar game counts are inconsistent")
    if not isinstance(manifest.get("rows"), int) or manifest["rows"] < 1:
        raise SystemExit("belief dataset sidecar rows must be a positive integer")
    declared_csv_sha256 = manifest.get("csv_sha256")
    if not is_sha256(declared_csv_sha256):
        raise SystemExit("belief dataset sidecar csv_sha256 must be lowercase SHA-256")
    actual_csv_sha256 = sha256(path)
    if declared_csv_sha256 != actual_csv_sha256:
        raise SystemExit("belief dataset sidecar CSV SHA-256 mismatch")
    return manifest


def load(path, allow_unsafe_no_sidecar=False):
    sidecar = validate_dataset_sidecar(path, allow_missing=allow_unsafe_no_sidecar)
    feature_schema_version = sidecar["feature_schema_version"] if sidecar else 1
    expected_features = belief_feature_names(feature_schema_version)
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        features = [name for name in fields if name in set(expected_features)]
        unexpected_features = [
            name
            for name in fields
            if (name.startswith("seq.") or (name.startswith("b") and name[1:].isdigit()))
            and name not in set(expected_features)
        ]
        if features != expected_features or unexpected_features:
            raise SystemExit("belief CSV feature columns/order do not match declared schema")
        required = {
            "schema_version",
            "feature_schema_version",
            "game_id",
            "target",
            *[f"mask{i}" for i in range(4)],
        }
        if not required.issubset(fields):
            raise SystemExit(f"missing columns: {sorted(required - set(fields))}")
        games, x, masks, targets = [], [], [], []
        for line, row in enumerate(reader, 2):
            try:
                schema_version = int(row["schema_version"])
                row_feature_schema = int(row["feature_schema_version"])
            except (TypeError, ValueError) as error:
                raise SystemExit(f"invalid belief schema version on row {line}") from error
            feature = [float(row[name]) for name in features]
            mask = [float(row[f"mask{i}"]) for i in range(4)]
            target = int(row["target"])
            if (
                schema_version != DATASET_SCHEMA_VERSION
                or row_feature_schema != feature_schema_version
                or not row["game_id"]
                or not np.isfinite(feature).all()
                or target not in range(4)
                or any(value not in (0.0, 1.0) for value in mask)
                or mask[target] != 1.0
            ):
                raise SystemExit(f"invalid belief row {line}")
            games.append(row["game_id"])
            x.append(feature)
            masks.append(mask)
            targets.append(target)
    if not x:
        raise SystemExit("belief dataset is empty")
    if sidecar is not None:
        if sidecar["rows"] != len(x):
            raise SystemExit("belief dataset sidecar row count does not match CSV")
        if sidecar["games_completed"] != len(set(games)):
            raise SystemExit("belief dataset sidecar completed-game count does not match CSV")
    return (
        np.asarray(x, dtype=np.float32),
        np.asarray(masks, dtype=np.float32),
        np.asarray(targets, dtype=np.int64),
        np.asarray(games),
        features,
    )


def write_golden(model, path, feature_names, manifest_version):
    rng = np.random.default_rng(0xBE11EF)
    features = rng.uniform(0.0, 1.0, size=(13, len(feature_names))).astype(
        np.float32
    )
    masks = np.asarray(
        [
            [1, 1, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 1],
            [1, 0, 1, 0],
            [0, 0, 1, 1],
        ]
        * 3,
        dtype=np.float32,
    )[: len(features)]
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(features), torch.from_numpy(masks)).numpy()
    payload = {
        "manifest_version": manifest_version,
        "vector_contract": GOLDEN_VECTOR_CONTRACT,
        "feature_dim": len(feature_names),
        "target_dim": len(TARGET_CLASSES),
        "atol": 5e-4,
        "rtol": 1e-6,
        "features": features.tolist(),
        "legality_mask": masks.tolist(),
        "outputs": {"destination_logits": logits.tolist()},
    }
    temporary = f"{path}.tmp"
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def split_games(games, validation_fraction, seed):
    unique = np.unique(games)
    if len(unique) < 2:
        raise SystemExit("need at least two whole games for belief train/validation")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    count = min(len(unique) - 1, max(1, round(len(unique) * validation_fraction)))
    validation_games = set(unique[:count])
    validation = np.asarray([game in validation_games for game in games])
    return ~validation, validation, sorted(set(unique) - validation_games), sorted(validation_games)


class BeliefNet(_NN_MODULE):
    def __init__(self, width, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 4),
        )

    def forward(self, features, legality_mask):
        logits = self.net(features)
        return torch.where(
            legality_mask > 0,
            logits,
            torch.full_like(logits, -1.0e4),
        )


def metrics(model, x, mask, target):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x), torch.from_numpy(mask))
        probabilities = torch.softmax(logits, dim=1).numpy()
    selected = probabilities[np.arange(len(target)), target]
    accuracy = float(np.mean(np.argmax(probabilities, axis=1) == target))
    nll = float(np.mean(-np.log(np.maximum(selected, 1e-12))))
    one_hot = np.eye(4)[target]
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == target).astype(np.float32)
    ece = 0.0
    for low in np.linspace(0.0, 0.9, 10):
        selected_bin = (confidence >= low) & (confidence < low + 0.1)
        if selected_bin.any():
            ece += float(
                selected_bin.mean()
                * abs(confidence[selected_bin].mean() - correct[selected_bin].mean())
            )
    illegal_mass = float(np.max(np.sum(probabilities * (1.0 - mask), axis=1)))
    return {
        "top1_accuracy": accuracy,
        "nll": nll,
        "multiclass_brier": brier,
        "ece10": ece,
        "max_illegal_probability_mass": illegal_mass,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="training/belief_data.csv")
    parser.add_argument("--out", default="training/belief_model.onnx")
    parser.add_argument("--golden-out")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument(
        "--allow-unsafe-no-sidecar",
        action="store_true",
        help="permit exploratory training without a generator manifest; artifact cannot be served",
    )
    args = parser.parse_args()

    x, mask, target, games, feature_names = load(
        args.data, allow_unsafe_no_sidecar=args.allow_unsafe_no_sidecar
    )
    dataset_sidecar = f"{args.data}.manifest.json"
    has_dataset_sidecar = os.path.exists(dataset_sidecar)
    dataset_manifest = (
        validate_dataset_sidecar(args.data) if has_dataset_sidecar else None
    )
    feature_schema_version = (
        dataset_manifest["feature_schema_version"] if dataset_manifest else 1
    )
    manifest_version = dataset_manifest["manifest_version"] if dataset_manifest else 1
    train, validation, train_games, validation_games = split_games(
        games, args.val_frac, args.seed
    )
    print(
        f"{len(x)} rows, {len(set(games))} games, dim={x.shape[1]}, "
        f"train={train.sum()} validation={validation.sum()}, legal destinations="
        f"{mask.sum(axis=1).mean():.2f}/4"
    )
    if args.analyze:
        return
    if not _HAS_TORCH:
        raise SystemExit("training requires torch + onnx; install training/requirements.txt")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = BeliefNet(x.shape[1], args.hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_indices = np.flatnonzero(train)
    for epoch in range(args.epochs):
        rng.shuffle(train_indices)
        model.train()
        losses = []
        for start in range(0, len(train_indices), args.batch_size):
            idx = train_indices[start : start + args.batch_size]
            logits = model(torch.from_numpy(x[idx]), torch.from_numpy(mask[idx]))
            loss = nn.functional.cross_entropy(logits, torch.from_numpy(target[idx]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            report = metrics(model, x[validation], mask[validation], target[validation])
            print(
                f"epoch {epoch:3d} loss={np.mean(losses):.4f} "
                f"acc={report['top1_accuracy']:.1%} nll={report['nll']:.4f} "
                f"Brier={report['multiclass_brier']:.4f} ECE={report['ece10']:.4f}"
            )
    report = metrics(model, x[validation], mask[validation], target[validation])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    temporary = args.out + ".tmp"
    model.eval()
    torch.onnx.export(
        model,
        (torch.zeros((1, x.shape[1])), torch.ones((1, 4))),
        temporary,
        input_names=["features", "legality_mask"],
        output_names=["destination_logits"],
        dynamic_axes={
            "features": {0: "N"},
            "legality_mask": {0: "N"},
            "destination_logits": {0: "N"},
        },
        opset_version=13,
        dynamo=False,
    )
    os.replace(temporary, args.out)
    golden_path = args.golden_out or f"{args.out}.golden.json"
    write_golden(model, golden_path, feature_names, manifest_version)
    manifest = {
        "manifest_version": manifest_version,
        "contract": MODEL_CONTRACT,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "feature_schema_version": feature_schema_version,
        "feature_dim": len(feature_names),
        "feature_names": feature_names,
        "supported_game_contract": SUPPORTED_GAME_CONTRACT,
        "inputs": ["features", "legality_mask"],
        "outputs": ["destination_logits"],
        "target_classes": TARGET_CLASSES,
        "encoder_contract": (
            dataset_manifest["encoder_contract"] if dataset_manifest else None
        ),
        "encoder_source_sha256": (
            dataset_manifest["encoder_source_sha256"] if dataset_manifest else None
        ),
        "golden_vector_contract": GOLDEN_VECTOR_CONTRACT,
        "training_behaviour_policy_domain": (
            dataset_manifest["behaviour_policy_domain"] if dataset_manifest else None
        ),
        "proposal_factorization": PROPOSAL_FACTORIZATION,
        "hard_legality_mask_value": -1.0e4,
        "model_sha256": sha256(args.out),
        "golden_path": os.path.basename(golden_path),
        "golden_sha256": sha256(golden_path),
        "dataset_sha256": sha256(args.data),
        "dataset_manifest_declared_csv_sha256": (
            dataset_manifest["csv_sha256"] if dataset_manifest else None
        ),
        "dataset_manifest_sha256": (
            sha256(dataset_sidecar) if has_dataset_sidecar else None
        ),
        "split": {
            "train_games": len(train_games),
            "validation_games": len(validation_games),
            "train_game_ids_sha256": hashlib.sha256(
                "\n".join(train_games).encode()
            ).hexdigest(),
            "validation_game_ids_sha256": hashlib.sha256(
                "\n".join(validation_games).encode()
            ).hexdigest(),
        },
        "validation_metrics": report,
        "research_only": True,
        "auto_promotion": False,
        "unsafe": not has_dataset_sidecar,
        "serving_status": (
            SERVING_STATUS if has_dataset_sidecar else "unsafe_missing_dataset_manifest"
        ),
    }
    with open(args.out + ".manifest.json.tmp", "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(args.out + ".manifest.json.tmp", args.out + ".manifest.json")
    print(json.dumps(report, sort_keys=True))
    print(f"wrote {args.out}, {args.out}.manifest.json, and {golden_path}")


if __name__ == "__main__":
    main()
