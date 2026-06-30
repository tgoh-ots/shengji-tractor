import csv
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import train_expert
import train_belief


def write_dataset(path, width, include_game=True):
    fields = ["group", "candidate_id", "label", "behaviour_label"]
    if include_game:
        fields.insert(0, "game_id")
    fields += [f"f{i}" for i in range(width)]
    fields += [
        "v_target",
        "q_target",
        "v_score_bucket",
        "q_score_bucket",
        "v_win_target",
        "q_win_target",
        "v_kitty_target",
        "q_kitty_target",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for game in range(2):
            for candidate in range(2):
                row = {
                    "group": f"g{game}",
                    "candidate_id": candidate,
                    "label": int(candidate == 0),
                    "behaviour_label": int(candidate == 1),
                    "v_target": 0.2,
                    "q_target": 0.2 if candidate == 0 else -0.2,
                    "v_score_bucket": 2,
                    "q_score_bucket": candidate + 1,
                    "v_win_target": 1,
                    "q_win_target": int(candidate == 0),
                    "v_kitty_target": 0,
                    "q_kitty_target": int(candidate == 1),
                    **{f"f{i}": (i + candidate) / max(1, width) for i in range(width)},
                }
                if include_game:
                    row["game_id"] = f"game-{game}"
                writer.writerow(row)


def write_belief_dataset(
    path,
    feature_schema_version=1,
    schema_version=1,
    sidecar=None,
    feature_names=None,
    row_feature_schema_version=None,
):
    if feature_names is None:
        feature_names = train_belief.belief_feature_names(feature_schema_version)
    fields = [
        "schema_version",
        "feature_schema_version",
        "game_id",
        "snapshot_id",
        "actor",
        "card_id",
        "target",
        *[f"mask{i}" for i in range(4)],
        *feature_names,
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for game in range(2):
            for target in range(4):
                writer.writerow(
                    {
                        "schema_version": schema_version,
                        "feature_schema_version": (
                            feature_schema_version
                            if row_feature_schema_version is None
                            else row_feature_schema_version
                        ),
                        "game_id": f"belief-game-{game}",
                        "snapshot_id": 0,
                        "actor": game,
                        "card_id": target,
                        "target": target,
                        **{f"mask{i}": 1 for i in range(4)},
                        **{
                            name: (i + target) / max(1, len(feature_names))
                            for i, name in enumerate(feature_names)
                        },
                    }
                )
    if sidecar is not None:
        sidecar = dict(sidecar)
        sidecar.setdefault("csv_sha256", train_belief.sha256(path))
        with open(f"{path}.manifest.json", "w") as handle:
            json.dump(sidecar, handle)


def valid_belief_sidecar(feature_schema_version=1):
    feature_names = train_belief.belief_feature_names(feature_schema_version)
    return {
        "manifest_version": feature_schema_version,
        "dataset_schema_version": 1,
        "feature_schema_version": feature_schema_version,
        "feature_dim": len(feature_names),
        "feature_names": feature_names,
        "encoder_contract": train_belief.belief_encoder_contract(feature_schema_version),
        "encoder_source_sha256": train_belief.sha256(
            train_belief.ENCODER_SOURCE_PATH
        ),
        "target_classes": [
            "next-seat",
            "opposite-seat",
            "previous-seat",
            "kitty",
        ],
        "supported_game_contract": "tractor:4p:2x-standard:kitty8:no-removed",
        "behaviour": "easy-play/expert-bid",
        "behaviour_policy_domain": train_belief.SUPPORTED_BEHAVIOUR_POLICY_DOMAIN,
        "target_semantics": train_belief.TARGET_SEMANTICS,
        "publicly_pinned_targets_excluded": True,
        "legality_contract": (
            "mask=1 iff destination has capacity and no public effective-suit void"
        ),
        "public_history_contract": train_belief.PUBLIC_HISTORY_CONTRACTS[
            feature_schema_version
        ],
        "games_requested": 2,
        "games_completed": 2,
        "games_dropped": 0,
        "rows": 8,
    }


class TrainingContractTests(unittest.TestCase):
    def test_schema_v2_auxiliaries_and_game_split(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "data.csv")
            write_dataset(path, 49)
            dataset = train_expert.load_dataset(path)
            self.assertEqual(dataset.feature_schema_version, 2)
            self.assertTrue(dataset.game_ids_are_trajectories)
            self.assertEqual(dataset.bucket_values, [1, 2])
            train, validation, train_ids, validation_ids = train_expert.split_by_game(
                dataset.groups, 0.5, 7
            )
            self.assertTrue(train and validation)
            self.assertTrue(set(train_ids).isdisjoint(validation_ids))
            self.assertTrue(np.isfinite(dataset.groups[0].q_win).all())

    def test_unsupported_width_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "data.csv")
            write_dataset(path, 48)
            with self.assertRaises(SystemExit):
                train_expert.load_dataset(path)

    def test_legacy_groups_are_not_claimed_as_trajectories(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "data.csv")
            write_dataset(path, 36, include_game=False)
            dataset = train_expert.load_dataset(path)
            self.assertEqual(dataset.feature_schema_version, 1)
            self.assertFalse(dataset.game_ids_are_trajectories)

    def test_belief_schema_and_sidecar_contract_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "belief.csv")
            write_belief_dataset(path, sidecar=valid_belief_sidecar())
            x, mask, target, games, features = train_belief.load(path)
            self.assertEqual(features, [f"b{i}" for i in range(20)])
            self.assertEqual(x.shape, (8, 20))
            self.assertEqual(mask.shape, (8, 4))
            self.assertEqual(set(target), {0, 1, 2, 3})
            self.assertEqual(set(games), {"belief-game-0", "belief-game-1"})

    def test_belief_schema_v2_semantic_features_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "belief-v2.csv")
            write_belief_dataset(
                path,
                feature_schema_version=2,
                sidecar=valid_belief_sidecar(2),
            )
            x, mask, target, games, features = train_belief.load(path)
            self.assertEqual(features, train_belief.belief_feature_names(2))
            self.assertEqual(x.shape, (8, 128))
            self.assertEqual(mask.shape, (8, 4))
            self.assertEqual(set(target), {0, 1, 2, 3})
            self.assertEqual(set(games), {"belief-game-0", "belief-game-1"})

    def test_belief_rejects_wrong_row_schema_or_width(self):
        with tempfile.TemporaryDirectory() as directory:
            wrong_schema = os.path.join(directory, "wrong-schema.csv")
            write_belief_dataset(
                wrong_schema,
                schema_version=2,
                sidecar=valid_belief_sidecar(),
            )
            with self.assertRaises(SystemExit):
                train_belief.load(wrong_schema)

            wrong_width = os.path.join(directory, "wrong-width.csv")
            wrong_names = [f"b{i}" for i in range(19)]
            write_belief_dataset(
                wrong_width,
                feature_names=wrong_names,
                sidecar=valid_belief_sidecar(),
            )
            with self.assertRaises(SystemExit):
                train_belief.load(wrong_width)

    def test_belief_rejects_schema_tuple_names_and_row_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            wrong_tuple = valid_belief_sidecar(2)
            wrong_tuple["manifest_version"] = 1
            tuple_path = os.path.join(directory, "wrong-tuple.csv")
            write_belief_dataset(
                tuple_path,
                feature_schema_version=2,
                sidecar=wrong_tuple,
            )
            with self.assertRaises(SystemExit):
                train_belief.load(tuple_path)

            wrong_names = valid_belief_sidecar(2)
            wrong_names["feature_names"] = list(reversed(wrong_names["feature_names"]))
            names_path = os.path.join(directory, "wrong-names.csv")
            write_belief_dataset(
                names_path,
                feature_schema_version=2,
                sidecar=wrong_names,
            )
            with self.assertRaises(SystemExit):
                train_belief.load(names_path)

            row_path = os.path.join(directory, "wrong-row-schema.csv")
            write_belief_dataset(
                row_path,
                sidecar=valid_belief_sidecar(),
                row_feature_schema_version=2,
            )
            with self.assertRaises(SystemExit):
                train_belief.load(row_path)

    def test_belief_rejects_sidecar_target_order_and_game_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            wrong_targets = valid_belief_sidecar()
            wrong_targets["target_classes"] = list(
                reversed(wrong_targets["target_classes"])
            )
            target_path = os.path.join(directory, "wrong-targets.csv")
            write_belief_dataset(target_path, sidecar=wrong_targets)
            with self.assertRaises(SystemExit):
                train_belief.load(target_path)

            wrong_game = valid_belief_sidecar()
            wrong_game["supported_game_contract"] = "finding-friends:6p"
            game_path = os.path.join(directory, "wrong-game.csv")
            write_belief_dataset(game_path, sidecar=wrong_game)
            with self.assertRaises(SystemExit):
                train_belief.load(game_path)

            wrong_behaviour = valid_belief_sidecar()
            wrong_behaviour["behaviour"] = "expert-everywhere"
            behaviour_path = os.path.join(directory, "wrong-behaviour.csv")
            write_belief_dataset(behaviour_path, sidecar=wrong_behaviour)
            with self.assertRaises(SystemExit):
                train_belief.load(behaviour_path)

    def test_belief_requires_sidecar_unless_explicitly_unsafe(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "belief.csv")
            write_belief_dataset(path)
            with self.assertRaises(SystemExit):
                train_belief.load(path)
            x, _mask, _target, _games, features = train_belief.load(
                path, allow_unsafe_no_sidecar=True
            )
            self.assertEqual(x.shape[1], 20)
            self.assertEqual(features, [f"b{i}" for i in range(20)])

            v2_path = os.path.join(directory, "belief-v2.csv")
            write_belief_dataset(v2_path, feature_schema_version=2)
            with self.assertRaises(SystemExit):
                train_belief.load(v2_path, allow_unsafe_no_sidecar=True)

    def test_belief_sidecar_binds_the_exact_csv_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "belief.csv")
            write_belief_dataset(path, sidecar=valid_belief_sidecar())
            with open(path, "a") as handle:
                handle.write("\n")
            with self.assertRaises(SystemExit):
                train_belief.load(path)


if __name__ == "__main__":
    unittest.main()
