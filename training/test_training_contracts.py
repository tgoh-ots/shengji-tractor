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


def write_belief_dataset(path, width=20, schema_version=1, sidecar=None):
    fields = [
        "schema_version",
        "game_id",
        "snapshot_id",
        "actor",
        "card_id",
        "target",
        *[f"mask{i}" for i in range(4)],
        *[f"b{i}" for i in range(width)],
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for game in range(2):
            for target in range(4):
                writer.writerow(
                    {
                        "schema_version": schema_version,
                        "game_id": f"belief-game-{game}",
                        "snapshot_id": 0,
                        "actor": game,
                        "card_id": target,
                        "target": target,
                        **{f"mask{i}": 1 for i in range(4)},
                        **{f"b{i}": (i + target) / 20 for i in range(width)},
                    }
                )
    if sidecar is not None:
        with open(f"{path}.manifest.json", "w") as handle:
            json.dump(sidecar, handle)


def valid_belief_sidecar():
    return {
        "manifest_version": 1,
        "dataset_schema_version": 1,
        "feature_dim": 20,
        "target_classes": [
            "next-seat",
            "opposite-seat",
            "previous-seat",
            "kitty",
        ],
        "supported_game_contract": "tractor:4p:2x-standard:kitty8:no-removed",
        "behaviour": "easy-play/expert-bid",
        "games_dropped": 0,
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

    def test_belief_rejects_wrong_row_schema_or_width(self):
        with tempfile.TemporaryDirectory() as directory:
            wrong_schema = os.path.join(directory, "wrong-schema.csv")
            write_belief_dataset(wrong_schema, schema_version=2)
            with self.assertRaises(SystemExit):
                train_belief.load(wrong_schema)

            wrong_width = os.path.join(directory, "wrong-width.csv")
            write_belief_dataset(wrong_width, width=19)
            with self.assertRaises(SystemExit):
                train_belief.load(wrong_width)

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


if __name__ == "__main__":
    unittest.main()
