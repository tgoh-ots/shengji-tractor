#!/usr/bin/env python3
"""Validate and compose schema-v3 Expert datasets.

This module is deliberately conservative.  It only augments the current
four-player/two-standard-deck feature contract by symmetries under which the
49 encoded features are already invariant: rotating every seat and globally
renaming suits.  Landlord/attacker exchange is *not* a symmetry of Shengji's
scoring contract and is rejected.

Offline rows are accepted only from a replay-verifier bundle with a content
hash and explicit legality/honesty/terminal-target attestations.  Human actions
become behaviour-policy labels; unproven counterfactual Q labels are stripped.
The resulting artifact remains research-only until a normal paired promotion
gate is run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


DATASET_SCHEMA_VERSION = 3
FEATURE_SCHEMA_VERSION = 2
FEATURE_DIM = 49
SUPPORTED_CONFIG = "tractor-4p-2deck"

REQUIRED_COLUMNS = {
    "schema_version",
    "run_id",
    "game_id",
    "game_seed",
    "decision_id",
    "candidate_id",
    "group",
    "actor",
    "actor_team",
    "behaviour",
    "rollout_behaviour",
    "config",
    "action",
    "label",
    "behaviour_label",
    "v_target",
    "q_target",
    "q_samples",
}
PROVENANCE_COLUMNS = [
    "trajectory_family_id",
    "source_kind",
    "source_id",
    "symmetry_id",
    "augmentation_parent_game_id",
    "parent_game_seed",
]
Q_COLUMNS = [
    "q_target",
    "q_attacker_points",
    "q_score_bucket",
    "q_win_target",
    "q_kitty_target",
]
OFFLINE_SOURCE_KINDS = {"human-replay", "offline-replay"}
_SAFE_SOURCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: str | os.PathLike[str]) -> dict:
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"{path}: expected a JSON object")
    return value


def _atomic_json(path: str | os.PathLike[str], value: dict) -> None:
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _feature_columns(fieldnames: list[str]) -> list[str]:
    features = sorted(
        (name for name in fieldnames if re.fullmatch(r"f\d+", name)),
        key=lambda name: int(name[1:]),
    )
    expected = [f"f{index}" for index in range(FEATURE_DIM)]
    if features != expected:
        raise SystemExit(
            f"feature contract must be schema {FEATURE_SCHEMA_VERSION}/dim {FEATURE_DIM}; "
            f"found {len(features)} columns"
        )
    return features


def read_header(path: str | os.PathLike[str]) -> list[str]:
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            fieldnames = csv.DictReader(handle).fieldnames
    except OSError as error:
        raise SystemExit(f"cannot read CSV {path}: {error}") from error
    if not fieldnames:
        raise SystemExit(f"{path}: CSV is empty or has no header")
    if len(fieldnames) != len(set(fieldnames)):
        raise SystemExit(f"{path}: duplicate CSV column")
    missing = sorted(REQUIRED_COLUMNS - set(fieldnames))
    if missing:
        raise SystemExit(f"{path}: missing required columns: {', '.join(missing)}")
    _feature_columns(fieldnames)
    return fieldnames


def _finite(value: str, context: str, *, required: bool = True) -> float | None:
    if value is None or value.strip() == "":
        if required:
            raise SystemExit(f"{context}: blank numeric value")
        return None
    try:
        parsed = float(value)
    except ValueError as error:
        raise SystemExit(f"{context}: invalid numeric value {value!r}") from error
    if not math.isfinite(parsed):
        raise SystemExit(f"{context}: NaN/Inf is forbidden")
    return parsed


def _integer(value: str, context: str, *, minimum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise SystemExit(f"{context}: invalid integer {value!r}") from error
    if minimum is not None and parsed < minimum:
        raise SystemExit(f"{context}: expected >= {minimum}, got {parsed}")
    return parsed


def _validate_action(value: str, context: str) -> None:
    if not value:
        raise SystemExit(f"{context}: action is blank")
    ids = [_integer(item, context, minimum=0) for item in value.split(".")]
    if any(card_id > 53 for card_id in ids):
        raise SystemExit(f"{context}: action card id outside FULL_DECK")
    if ids != sorted(ids):
        raise SystemExit(f"{context}: action card ids must be canonical/sorted")


@dataclass
class DatasetStats:
    rows: int
    games: int
    trajectory_families: int
    decisions: int
    q_rows: int
    game_ids: list[str]
    trajectory_family_ids: list[str]


def validate_dataset(path: str | os.PathLike[str]) -> DatasetStats:
    """Fail closed on the schema and per-decision listwise invariants."""

    fieldnames = read_header(path)
    feature_names = _feature_columns(fieldnames)
    groups: dict[str, dict] = {}
    game_families: dict[str, str] = {}
    game_ids: set[str] = set()
    families: set[str] = set()
    q_rows = 0
    rows = 0

    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for line, row in enumerate(reader, start=2):
            context = f"{path}:row {line}"
            if None in row:
                raise SystemExit(f"{context}: row has more fields than the header")
            if _integer(row["schema_version"], context) != DATASET_SCHEMA_VERSION:
                raise SystemExit(
                    f"{context}: expected dataset schema {DATASET_SCHEMA_VERSION}"
                )
            game_id = row["game_id"].strip()
            group_id = row["group"].strip()
            if not game_id or not group_id:
                raise SystemExit(f"{context}: game_id/group cannot be blank")
            family_id = (row.get("trajectory_family_id") or game_id).strip()
            if not family_id:
                raise SystemExit(f"{context}: trajectory_family_id cannot be blank")
            previous_family = game_families.setdefault(game_id, family_id)
            if previous_family != family_id:
                raise SystemExit(f"{context}: one game maps to multiple trajectory families")
            game_ids.add(game_id)
            families.add(family_id)

            if row["config"] != SUPPORTED_CONFIG:
                raise SystemExit(
                    f"{context}: unsupported game config {row['config']!r}; "
                    f"expected {SUPPORTED_CONFIG!r}"
                )
            actor = _integer(row["actor"], context, minimum=0)
            if actor > 3:
                raise SystemExit(f"{context}: actor is not a four-player seat")
            candidate_id = _integer(row["candidate_id"], context, minimum=0)
            label = _integer(row["label"], context)
            behaviour_label = _integer(row["behaviour_label"], context)
            if label not in (0, 1) or behaviour_label not in (0, 1):
                raise SystemExit(f"{context}: labels must be binary")
            _integer(row["decision_id"], context, minimum=0)
            _validate_action(row["action"], context)
            for name in feature_names:
                _finite(row[name], f"{context}:{name}")
            value = _finite(row["v_target"], f"{context}:v_target")
            if value is None or not -1.000001 <= value <= 1.000001:
                raise SystemExit(f"{context}: v_target outside [-1,1]")
            q_value = _finite(row.get("q_target", ""), f"{context}:q_target", required=False)
            q_samples = _integer(row["q_samples"], context, minimum=0)
            if (q_value is None) != (q_samples == 0):
                raise SystemExit(f"{context}: q_target/q_samples disagree")
            if q_value is not None:
                if not -1.000001 <= q_value <= 1.000001:
                    raise SystemExit(f"{context}: q_target outside [-1,1]")
                q_rows += 1

            state = groups.setdefault(
                group_id,
                {
                    "game": game_id,
                    "family": family_id,
                    "candidate_ids": set(),
                    "labels": 0,
                    "behaviour_labels": 0,
                    "v": value,
                },
            )
            if state["game"] != game_id or state["family"] != family_id:
                raise SystemExit(f"{context}: decision spans games/families")
            if candidate_id in state["candidate_ids"]:
                raise SystemExit(f"{context}: duplicate candidate_id within decision")
            state["candidate_ids"].add(candidate_id)
            state["labels"] += label
            state["behaviour_labels"] += behaviour_label
            if abs(state["v"] - value) > 1e-5:
                raise SystemExit(f"{context}: candidate-dependent state V")
            rows += 1

    if rows == 0:
        raise SystemExit(f"{path}: dataset has no rows")
    for group_id, state in groups.items():
        ids = state["candidate_ids"]
        if len(ids) < 2 or ids != set(range(len(ids))):
            raise SystemExit(f"{path}: group {group_id} has non-contiguous/degenerate candidates")
        if state["labels"] != 1 or state["behaviour_labels"] != 1:
            raise SystemExit(f"{path}: group {group_id} is not one-hot listwise data")

    return DatasetStats(
        rows=rows,
        games=len(game_ids),
        trajectory_families=len(families),
        decisions=len(groups),
        q_rows=q_rows,
        game_ids=sorted(game_ids),
        trajectory_family_ids=sorted(families),
    )


@dataclass(frozen=True)
class Transform:
    seat_rotation: int
    suit_shift: int

    @property
    def name(self) -> str:
        if self.seat_rotation == 0 and self.suit_shift == 0:
            return "identity"
        return f"seat{self.seat_rotation}-suit{self.suit_shift}"


def symmetry_transforms(specification: str) -> list[Transform]:
    if "team" in specification.lower():
        raise SystemExit(
            "team/landlord exchange is not a valid symmetry: contract roles and scoring are asymmetric"
        )
    choices = {
        "identity": ([0], [0]),
        "seat": (range(4), [0]),
        "suit-cyclic": ([0], range(4)),
        "seat-suit-cyclic": (range(4), range(4)),
    }
    try:
        seats, suits = choices[specification]
    except KeyError as error:
        raise SystemExit(
            "augmentation must be identity, seat, suit-cyclic, or seat-suit-cyclic"
        ) from error
    return [Transform(seat, suit) for seat in seats for suit in suits]


def transform_action(action: str, suit_shift: int) -> str:
    transformed = []
    for value in action.split("."):
        card_id = int(value)
        if card_id < 52:
            suit, rank = divmod(card_id, 13)
            card_id = ((suit + suit_shift) % 4) * 13 + rank
        transformed.append(card_id)
    return ".".join(str(value) for value in sorted(transformed))


@dataclass
class OfflineBundle:
    source_id: str
    csv_path: str
    manifest_path: str
    manifest: dict
    selected_games: set[str]


def _resolve_path(base: Path, value: str, context: str) -> str:
    if not value:
        raise SystemExit(f"{context}: path cannot be blank")
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return str(path.resolve())


def load_offline_config(path: str | None, generated_games: int) -> tuple[list[OfflineBundle], dict]:
    if not path:
        return [], {"enabled": False, "selected_games": 0}
    config_path = Path(path).resolve()
    config = _load_json(config_path)
    if config.get("manifest_version") != 1:
        raise SystemExit(f"{path}: offline config manifest_version must be 1")
    max_fraction = float(config.get("max_offline_fraction", 0.10))
    if not 0.0 <= max_fraction <= 0.50:
        raise SystemExit(f"{path}: max_offline_fraction must be in [0,0.5]")
    seed = int(config.get("selection_seed", 0))
    entries = config.get("bundles")
    if not isinstance(entries, list) or not entries:
        raise SystemExit(f"{path}: bundles must be a non-empty array")

    bundles: list[OfflineBundle] = []
    all_candidates: list[tuple[str, str, OfflineBundle]] = []
    source_ids: set[str] = set()
    for index, entry in enumerate(entries):
        context = f"{path}:bundles[{index}]"
        if not isinstance(entry, dict):
            raise SystemExit(f"{context}: expected an object")
        csv_path = _resolve_path(config_path.parent, entry.get("csv", ""), context)
        manifest_path = _resolve_path(
            config_path.parent,
            entry.get("manifest", f"{csv_path}.manifest.json"),
            context,
        )
        manifest = _load_json(manifest_path)
        source_id = manifest.get("source_id")
        if not isinstance(source_id, str) or not _SAFE_SOURCE_ID.fullmatch(source_id):
            raise SystemExit(f"{context}: manifest source_id is missing or unsafe")
        if source_id in source_ids:
            raise SystemExit(f"{path}: duplicate source_id {source_id!r}")
        source_ids.add(source_id)
        if manifest.get("manifest_version") != 1:
            raise SystemExit(f"{manifest_path}: manifest_version must be 1")
        if manifest.get("source_kind") not in OFFLINE_SOURCE_KINDS:
            raise SystemExit(f"{manifest_path}: source_kind is not a replay source")
        expected_contract = {
            "dataset_schema_version": DATASET_SCHEMA_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_dim": FEATURE_DIM,
            "game_config": SUPPORTED_CONFIG,
        }
        for name, expected in expected_contract.items():
            if manifest.get(name) != expected:
                raise SystemExit(f"{manifest_path}: {name} must be {expected!r}")
        actual_hash = sha256_file(csv_path)
        if manifest.get("content_sha256") != actual_hash:
            raise SystemExit(f"{manifest_path}: CSV content_sha256 mismatch")
        verification = manifest.get("replay_verification")
        required_attestations = (
            "complete_trajectories",
            "legal_actions",
            "honest_observations",
            "terminal_targets_recomputed",
        )
        if not isinstance(verification, dict) or not all(
            verification.get(name) is True for name in required_attestations
        ):
            raise SystemExit(f"{manifest_path}: replay verification attestations are incomplete")
        if not isinstance(verification.get("verifier"), str) or not verification["verifier"].strip():
            raise SystemExit(f"{manifest_path}: verifier is required")
        if not isinstance(verification.get("raw_replay_sha256"), str) or not _SHA256.fullmatch(
            verification["raw_replay_sha256"]
        ):
            raise SystemExit(f"{manifest_path}: verifier and raw_replay_sha256 are required")
        stats = validate_dataset(csv_path)
        maximum = int(entry.get("max_games", len(stats.game_ids)))
        if maximum < 0:
            raise SystemExit(f"{context}: max_games cannot be negative")
        ranked = sorted(
            stats.game_ids,
            key=lambda game: hashlib.sha256(
                f"{seed}\0{source_id}\0{game}".encode("utf-8")
            ).hexdigest(),
        )[:maximum]
        bundle = OfflineBundle(source_id, csv_path, manifest_path, manifest, set())
        bundles.append(bundle)
        for game in ranked:
            priority = hashlib.sha256(
                f"global\0{seed}\0{source_id}\0{game}".encode("utf-8")
            ).hexdigest()
            all_candidates.append((priority, game, bundle))

    if max_fraction == 1.0:
        global_cap = len(all_candidates)
    elif max_fraction == 0.0:
        global_cap = 0
    else:
        global_cap = math.floor(generated_games * max_fraction / (1.0 - max_fraction))
    for _priority, game, bundle in sorted(all_candidates)[:global_cap]:
        bundle.selected_games.add(game)
    selected = sum(len(bundle.selected_games) for bundle in bundles)
    return bundles, {
        "enabled": True,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "max_offline_fraction": max_fraction,
        "selection_seed": seed,
        "selected_games": selected,
    }


def fingerprint_offline_config(path: str | None) -> dict:
    if not path:
        return {"enabled": False}
    config_path = Path(path).resolve()
    config = _load_json(config_path)
    result = {
        "enabled": True,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "bundles": [],
    }
    for index, entry in enumerate(config.get("bundles", [])):
        if not isinstance(entry, dict):
            raise SystemExit(f"{path}:bundles[{index}] must be an object")
        csv_path = _resolve_path(config_path.parent, entry.get("csv", ""), str(path))
        manifest_path = _resolve_path(
            config_path.parent,
            entry.get("manifest", f"{csv_path}.manifest.json"),
            str(path),
        )
        result["bundles"].append(
            {
                "csv_path": csv_path,
                "csv_sha256": sha256_file(csv_path),
                "manifest_path": manifest_path,
                "manifest_sha256": sha256_file(manifest_path),
            }
        )
    return result


def _namespace_offline(row: dict[str, str], bundle: OfflineBundle) -> dict[str, str]:
    normalized = dict(row)
    prefix = f"offline-{bundle.source_id}"
    original_game = row["game_id"]
    original_group = row["group"]
    original_family = (row.get("trajectory_family_id") or original_game).strip()
    if not original_family:
        raise SystemExit(
            f"{bundle.csv_path}: offline row has a blank parent trajectory family"
        )
    normalized["game_id"] = f"{prefix}::{original_game}"
    normalized["group"] = f"{prefix}::{original_group}"
    normalized["run_id"] = prefix
    # Preserve the replay verifier's parent-family key (with a source
    # namespace). Multiple physical/augmented game IDs from one replay must
    # remain in the same train/validation partition.
    normalized["trajectory_family_id"] = f"{prefix}::{original_family}"
    normalized["source_kind"] = bundle.manifest["source_kind"]
    normalized["source_id"] = bundle.source_id
    # Human/offline actions are useful as behaviour-policy targets.  A replay
    # alone does not establish counterfactual search returns, so Q is blanked.
    normalized["label"] = row["behaviour_label"]
    normalized["behaviour"] = f"offline:{bundle.source_id}"
    normalized["rollout_behaviour"] = "none"
    for name in Q_COLUMNS:
        if name in normalized:
            normalized[name] = ""
    normalized["q_samples"] = "0"
    return normalized


def _apply_transform(row: dict[str, str], transform: Transform) -> dict[str, str]:
    transformed = dict(row)
    parent_game = row["game_id"]
    parent_group = row["group"]
    family = row.get("trajectory_family_id") or parent_game
    transformed["trajectory_family_id"] = family
    transformed["symmetry_id"] = transform.name
    transformed["augmentation_parent_game_id"] = parent_game
    transformed["parent_game_seed"] = row.get("game_seed", "")
    if transform.name != "identity":
        transformed["game_id"] = f"{parent_game}::sym-{transform.name}"
        transformed["group"] = f"{parent_group}::sym-{transform.name}"
        transformed["run_id"] = f"{row['run_id']}::sym-{transform.name}"
        # This is a mathematical rename of the seeded hand, not a hand produced
        # by the original numeric seed.  Do not advertise it as seed-replayable.
        transformed["game_seed"] = ""
    transformed["actor"] = str((int(row["actor"]) + transform.seat_rotation) % 4)
    transformed["action"] = transform_action(row["action"], transform.suit_shift)
    return transformed


def compose_dataset(
    generated_path: str,
    output_path: str,
    augmentation: str = "identity",
    offline_config: str | None = None,
) -> dict:
    generated_path = os.path.abspath(generated_path)
    output_path = os.path.abspath(output_path)
    if generated_path == output_path:
        raise SystemExit("generated input and composed output must be different files")
    generated_stats = validate_dataset(generated_path)
    generated_manifest_path = f"{generated_path}.manifest.json"
    if os.path.isfile(generated_manifest_path):
        generated_manifest_payload = _load_json(generated_manifest_path)
        expected_hash = generated_manifest_payload.get("content_sha256")
        if expected_hash is not None and expected_hash != sha256_file(generated_path):
            raise SystemExit("generated dataset manifest content_sha256 mismatch")
        for name, actual in (
            ("rows", generated_stats.rows),
            ("games", generated_stats.games),
        ):
            declared = generated_manifest_payload.get(name)
            if declared is not None and declared != actual:
                raise SystemExit(f"generated dataset manifest {name} does not match CSV")
    transforms = symmetry_transforms(augmentation)
    bundles, offline_summary = load_offline_config(offline_config, generated_stats.games)

    generated_header = read_header(generated_path)
    output_header = list(generated_header)
    for name in PROVENANCE_COLUMNS:
        if name not in output_header:
            output_header.append(name)
    for bundle in bundles:
        offline_header = read_header(bundle.csv_path)
        unknown = sorted(set(offline_header) - set(output_header))
        if unknown:
            raise SystemExit(
                f"{bundle.csv_path}: offline CSV has unsupported columns: {', '.join(unknown)}"
            )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=os.path.basename(output_path) + ".", suffix=".tmp", dir=os.path.dirname(output_path)
    )
    os.close(descriptor)
    source_rows = {"self-play": 0}
    audited_actions = set()
    audited_rows = 0
    try:
        with open(temporary, "w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, output_header, lineterminator="\n")
            writer.writeheader()
            sources: list[tuple[str, OfflineBundle | None]] = [(generated_path, None)]
            sources.extend((bundle.csv_path, bundle) for bundle in bundles)
            for source_path, bundle in sources:
                with open(source_path, newline="", encoding="utf-8") as source:
                    for row in csv.DictReader(source):
                        if bundle is not None and row["game_id"] not in bundle.selected_games:
                            continue
                        if bundle is None:
                            normalized = dict(row)
                            normalized["trajectory_family_id"] = (
                                row.get("trajectory_family_id") or row["game_id"]
                            )
                            normalized["source_kind"] = row.get("source_kind") or "self-play"
                            normalized["source_id"] = row.get("source_id") or "generated"
                            source_rows["self-play"] += 1
                        else:
                            normalized = _namespace_offline(row, bundle)
                            source_rows.setdefault(bundle.source_id, 0)
                            source_rows[bundle.source_id] += 1

                        # Schema-v2 inputs already quotient absolute seat/suit
                        # identity. Materializing transformed rows would repeat
                        # identical model inputs and silently reweight every
                        # trajectory 4-16x. Audit the metadata bijections but
                        # emit one canonical identity row only.
                        action = normalized["action"]
                        if action not in audited_actions:
                            for shift in {transform.suit_shift for transform in transforms}:
                                renamed = transform_action(action, shift)
                                restored = transform_action(renamed, (-shift) % 4)
                                if restored != action:
                                    raise SystemExit(
                                        f"symmetry audit failed to round-trip action {action!r}"
                                    )
                            audited_actions.add(action)
                        audited_rows += 1
                        canonical = _apply_transform(normalized, Transform(0, 0))
                        writer.writerow({name: canonical.get(name, "") for name in output_header})
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    output_stats = validate_dataset(output_path)
    sources = [
        {
            "source_kind": "self-play",
            "source_id": "generated",
            "path": generated_path,
            "content_sha256": sha256_file(generated_path),
            "manifest_sha256": (
                sha256_file(generated_manifest_path)
                if os.path.isfile(generated_manifest_path)
                else None
            ),
        }
    ]
    for bundle in bundles:
        sources.append(
            {
                "source_kind": bundle.manifest["source_kind"],
                "source_id": bundle.source_id,
                "path": bundle.csv_path,
                "content_sha256": sha256_file(bundle.csv_path),
                "manifest_sha256": sha256_file(bundle.manifest_path),
                "selected_games": len(bundle.selected_games),
                "q_targets_retained": False,
                "policy_target": "verified replay behaviour",
            }
        )
    research_only = bool(bundles)
    manifest = {
        "manifest_version": 1,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_dim": FEATURE_DIM,
        "game_config": SUPPORTED_CONFIG,
        "content_sha256": sha256_file(output_path),
        "rows": output_stats.rows,
        "games": output_stats.games,
        "trajectory_families": output_stats.trajectory_families,
        "decisions": output_stats.decisions,
        "q_rows": output_stats.q_rows,
        "sources": sources,
        "source_rows_before_augmentation": source_rows,
        "augmentation": {
            "specification": augmentation,
            "mode": "audit-only",
            "audited_transforms": [
                asdict(transform) | {"name": transform.name} for transform in transforms
            ],
            "materialized_transforms": ["identity"],
            "audited_rows": audited_rows,
            "audited_unique_actions": len(audited_actions),
            "audit_scope": (
                "action-id suit round-trip and metadata bijections; feature invariance is a "
                "schema design contract, not raw-state re-encoding"
            ),
            "optimization_weight_change": False,
            "feature_contract": "schema-v2 features invariant under whole-table seat rotation and global suit rename",
            "split_contract": "trajectory_family_id remains the parent game id",
            "team_exchange_supported": False,
        },
        "offline": offline_summary,
        "research_only": research_only,
        "automatic_production_promotion_allowed": False,
    }
    _atomic_json(f"{output_path}.manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="validate one schema-v3 CSV")
    validate_parser.add_argument("csv")

    compose_parser = subparsers.add_parser(
        "compose", help="compose generated, replay, and symmetry-derived trajectories"
    )
    compose_parser.add_argument("--generated", required=True)
    compose_parser.add_argument("--out", required=True)
    compose_parser.add_argument(
        "--augmentation",
        default="identity",
        choices=["identity", "seat", "suit-cyclic", "seat-suit-cyclic"],
    )
    compose_parser.add_argument("--offline-config")

    fingerprint_parser = subparsers.add_parser(
        "fingerprint-offline", help="hash an offline config and every referenced bundle"
    )
    fingerprint_parser.add_argument("--config")

    args = parser.parse_args()
    if args.command == "validate":
        stats = validate_dataset(args.csv)
        print(json.dumps(asdict(stats), indent=2, sort_keys=True))
    elif args.command == "compose":
        manifest = compose_dataset(
            args.generated, args.out, args.augmentation, args.offline_config
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
    elif args.command == "fingerprint-offline":
        print(json.dumps(fingerprint_offline_config(args.config), sort_keys=True))


if __name__ == "__main__":
    main()
