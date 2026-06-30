import csv
import hashlib
import json
import os
import re
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
import expert_iteration
import model_ab
import pipeline_contracts
import prepare_expert_data
import train_expert
import train_phase


FEATURES = [f"f{i}" for i in range(49)]
FIELDS = [
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
    *FEATURES,
    "label",
    "behaviour_label",
    "v_target",
    "v_attacker_points",
    "v_score_bucket",
    "v_win_target",
    "v_kitty_target",
    "q_target",
    "q_attacker_points",
    "q_score_bucket",
    "q_win_target",
    "q_kitty_target",
    "q_samples",
]


def file_hash(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def write_schema3(path, games=2, source_prefix="game", family=None):
    fields = list(FIELDS)
    if family is not None:
        fields.append("trajectory_family_id")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        for game in range(games):
            for candidate in range(2):
                writer.writerow(
                    {
                        "schema_version": 3,
                        "run_id": "test-run",
                        "game_id": f"{source_prefix}-{game}",
                        "game_seed": 100 + game,
                        "decision_id": 0,
                        "candidate_id": candidate,
                        "group": f"{source_prefix}-{game}-decision-0",
                        "actor": game % 4,
                        "actor_team": "landlord" if game % 2 == 0 else "attacker",
                        "behaviour": "easy",
                        "rollout_behaviour": "easy",
                        "config": "tractor-4p-2deck",
                        "action": "0" if candidate == 0 else "13",
                        **{
                            name: (index + candidate) / 100
                            for index, name in enumerate(FEATURES)
                        },
                        "label": int(candidate == 0),
                        "behaviour_label": int(candidate == 1),
                        "v_target": 0.2 if game % 2 == 0 else -0.2,
                        "v_attacker_points": 80,
                        "v_score_bucket": 2,
                        "v_win_target": int(game % 2 == 0),
                        "v_kitty_target": 0,
                        "q_target": 0.4 if candidate == 0 else -0.3,
                        "q_attacker_points": 80 + candidate,
                        "q_score_bucket": 2,
                        "q_win_target": int(candidate == 0),
                        "q_kitty_target": int(candidate == 1),
                        "q_samples": 1,
                        **(
                            {"trajectory_family_id": family(game)}
                            if family is not None
                            else {}
                        ),
                    }
                )


def write_offline_manifest(path, csv_path, *, complete=True):
    payload = {
        "manifest_version": 1,
        "dataset_schema_version": 3,
        "feature_schema_version": 2,
        "feature_dim": 49,
        "game_config": "tractor-4p-2deck",
        "source_kind": "human-replay",
        "source_id": "club-night",
        "content_sha256": file_hash(csv_path),
        "replay_verification": {
            "complete_trajectories": complete,
            "legal_actions": True,
            "honest_observations": True,
            "terminal_targets_recomputed": True,
            "verifier": "test-replayer-v1",
            "raw_replay_sha256": "a" * 64,
        },
    }
    with open(path, "w") as handle:
        json.dump(payload, handle)


def example_iteration_config():
    profile = {
        "name": "easy-anchor",
        "slots": 1,
        "behaviour": "easy",
        "behaviour_budget_ms": 5,
        "teacher_budget_ms": 20,
        "q_candidates": 2,
        "q_rollout_behaviour": "easy",
        "q_rollout_budget_ms": 5,
    }
    strong = dict(profile)
    strong.update(
        name="search",
        slots=2,
        behaviour="mix",
        mix_search_fraction=0.7,
        seat_behaviours=["mix", "enoch", "grandmaster", "easy"],
    )
    return {
        "manifest_version": 1,
        "research_only": True,
        "defaults": {"num_shards": 5},
        "rounds": [{"name": "round-one", "league": [profile, strong]}],
    }


class ExpertIterationDataTests(unittest.TestCase):
    def test_phase_feature_names_mirror_runtime_contract(self):
        phase_path = os.path.join(
            os.path.dirname(__file__), "..", "core", "src", "bot", "phase.rs"
        )
        with open(phase_path) as handle:
            source = handle.read()
        for phase, rust_name in (
            ("bid", "BID_FEATURE_NAMES"),
            ("kitty", "KITTY_FEATURE_NAMES"),
        ):
            match = re.search(
                rf"pub const {rust_name}:.*?= \[(.*?)\];", source, re.DOTALL
            )
            self.assertIsNotNone(match)
            runtime_names = re.findall(r'"([^"]+)"', match.group(1))
            self.assertEqual(train_phase.FEATURE_NAMES[phase], runtime_names)

    def test_symmetry_audit_is_deterministic_without_duplicate_training_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            generated = os.path.join(directory, "generated.csv")
            first = os.path.join(directory, "first.csv")
            second = os.path.join(directory, "second.csv")
            write_schema3(generated)

            manifest = prepare_expert_data.compose_dataset(
                generated, first, "seat-suit-cyclic"
            )
            prepare_expert_data.compose_dataset(generated, second, "seat-suit-cyclic")
            self.assertEqual(file_hash(first), file_hash(second))
            self.assertEqual(manifest["games"], 2)
            self.assertEqual(manifest["trajectory_families"], 2)
            self.assertFalse(manifest["research_only"])
            self.assertEqual(manifest["augmentation"]["mode"], "audit-only")
            self.assertEqual(manifest["augmentation"]["materialized_transforms"], ["identity"])
            self.assertEqual(len(manifest["augmentation"]["audited_transforms"]), 16)
            self.assertFalse(manifest["augmentation"]["optimization_weight_change"])

            with open(first, newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(row["symmetry_id"] == "identity" for row in rows))
            renamed = prepare_expert_data._apply_transform(
                rows[0], prepare_expert_data.Transform(1, 1)
            )
            self.assertEqual(renamed["actor"], "1")
            self.assertEqual(renamed["action"], "13")
            self.assertEqual(renamed["game_seed"], "")

            dataset = train_expert.load_dataset(first)
            train, validation, train_families, validation_families = train_expert.split_by_game(
                dataset.groups, 0.5, 17
            )
            self.assertTrue(train and validation)
            self.assertTrue(set(train_families).isdisjoint(validation_families))
            train_physical = {group.game_id for group in train}
            validation_physical = {group.game_id for group in validation}
            self.assertTrue(train_physical.isdisjoint(validation_physical))
            for family in train_families:
                self.assertFalse(
                    any(group.trajectory_family_id == family for group in validation)
                )

    def test_team_exchange_is_explicitly_rejected(self):
        with self.assertRaises(SystemExit):
            prepare_expert_data.symmetry_transforms("team-swap")

    def test_verified_offline_ingestion_is_bounded_and_strips_q(self):
        with tempfile.TemporaryDirectory() as directory:
            generated = os.path.join(directory, "generated.csv")
            offline = os.path.join(directory, "offline.csv")
            sidecar = f"{offline}.manifest.json"
            config = os.path.join(directory, "offline-config.json")
            output = os.path.join(directory, "combined.csv")
            write_schema3(generated, games=4)
            write_schema3(offline, games=4, source_prefix="human")
            write_offline_manifest(sidecar, offline)
            with open(config, "w") as handle:
                json.dump(
                    {
                        "manifest_version": 1,
                        "max_offline_fraction": 0.2,
                        "selection_seed": 3,
                        "bundles": [
                            {"csv": offline, "manifest": sidecar, "max_games": 4}
                        ],
                    },
                    handle,
                )

            manifest = prepare_expert_data.compose_dataset(
                generated, output, offline_config=config
            )
            # Four generated games and a 20%-of-total cap admit one replay game.
            self.assertEqual(manifest["offline"]["selected_games"], 1)
            self.assertEqual(manifest["games"], 5)
            self.assertTrue(manifest["research_only"])
            with open(output, newline="") as handle:
                offline_rows = [
                    row for row in csv.DictReader(handle) if row["source_kind"] == "human-replay"
                ]
            self.assertEqual(len(offline_rows), 2)
            self.assertTrue(all(row["q_target"] == "" for row in offline_rows))
            self.assertTrue(all(row["q_samples"] == "0" for row in offline_rows))
            self.assertTrue(
                all(row["label"] == row["behaviour_label"] for row in offline_rows)
            )
            self.assertTrue(all(row["game_id"].startswith("offline-club-night::") for row in offline_rows))

    def test_offline_parent_family_is_namespaced_but_preserved_for_split(self):
        with tempfile.TemporaryDirectory() as directory:
            generated = os.path.join(directory, "generated.csv")
            offline = os.path.join(directory, "offline.csv")
            sidecar = f"{offline}.manifest.json"
            config = os.path.join(directory, "offline-config.json")
            output = os.path.join(directory, "combined.csv")
            write_schema3(generated, games=4)
            write_schema3(
                offline,
                games=2,
                source_prefix="human-copy",
                family=lambda _game: "physical-replay-7",
            )
            write_offline_manifest(sidecar, offline)
            with open(config, "w") as handle:
                json.dump(
                    {
                        "manifest_version": 1,
                        "max_offline_fraction": 0.5,
                        "selection_seed": 9,
                        "bundles": [{"csv": offline, "manifest": sidecar}],
                    },
                    handle,
                )
            prepare_expert_data.compose_dataset(
                generated, output, offline_config=config
            )
            with open(output, newline="") as handle:
                rows = [
                    row
                    for row in csv.DictReader(handle)
                    if row["source_kind"] == "human-replay"
                ]
            self.assertEqual(
                {row["trajectory_family_id"] for row in rows},
                {"offline-club-night::physical-replay-7"},
            )
            dataset = train_expert.load_dataset(output)
            train, validation, *_ = train_expert.split_by_game(
                dataset.groups, 0.5, 13
            )
            offline_family = "offline-club-night::physical-replay-7"
            self.assertFalse(
                any(group.trajectory_family_id == offline_family for group in train)
                and any(group.trajectory_family_id == offline_family for group in validation)
            )

    def test_offline_bundle_fails_closed_on_attestation_or_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            generated = os.path.join(directory, "generated.csv")
            offline = os.path.join(directory, "offline.csv")
            sidecar = f"{offline}.manifest.json"
            config = os.path.join(directory, "config.json")
            write_schema3(generated)
            write_schema3(offline, source_prefix="human")
            write_offline_manifest(sidecar, offline, complete=False)
            with open(config, "w") as handle:
                json.dump(
                    {
                        "manifest_version": 1,
                        "bundles": [{"csv": offline, "manifest": sidecar}],
                    },
                    handle,
                )
            with self.assertRaises(SystemExit):
                prepare_expert_data.compose_dataset(
                    generated, os.path.join(directory, "out.csv"), offline_config=config
                )

            write_offline_manifest(sidecar, offline)
            with open(offline, "a") as handle:
                handle.write("\n")
            with self.assertRaises(SystemExit):
                prepare_expert_data.compose_dataset(
                    generated, os.path.join(directory, "out2.csv"), offline_config=config
                )

    def test_league_schedule_is_validated_and_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "iteration.json")
            with open(path, "w") as handle:
                json.dump(example_iteration_config(), handle)
            config = expert_iteration.load_config(path)
            schedule = [
                expert_iteration.profile_for_shard(config, 0, index)["name"]
                for index in range(7)
            ]
            self.assertEqual(
                schedule,
                ["easy-anchor", "search", "search", "easy-anchor", "search", "search", "easy-anchor"],
            )
            self.assertEqual(
                expert_iteration.profile_tsv(config, 0, 1).split("\t")[1], "mix"
            )
            self.assertEqual(
                expert_iteration.profile_tsv(config, 0, 1).split("\t")[7],
                "mix,enoch,grandmaster,easy",
            )

    def test_iteration_normalizes_booleans_and_varies_ab_seed_by_round(self):
        config = example_iteration_config()
        config["defaults"].update({"run_ab": True, "ab_seed": "0x5EED"})
        second = dict(config["rounds"][0])
        second["name"] = "round-two"
        config["rounds"].append(second)
        first = expert_iteration._effective_settings(config, config["rounds"][0], 0)
        second_settings = expert_iteration._effective_settings(
            config, config["rounds"][1], 1
        )
        self.assertEqual(expert_iteration._pipeline_env_value("run_ab", True), "1")
        self.assertEqual(expert_iteration._pipeline_env_value("run_ab", False), "0")
        self.assertEqual(first["ab_seed"], 0x5EED)
        self.assertNotEqual(first["ab_seed"], second_settings["ab_seed"])
        self.assertEqual(
            second_settings["ab_seed"],
            (0x5EED + 0x9E3779B97F4A7C15) & ((1 << 64) - 1),
        )

    def test_iteration_sanitizes_inherited_runtime_and_build_overrides(self):
        injected = {
            "SHENGJI_VALUE_WEIGHT": "1",
            "GM_WORLDS": "999",
            "OMNI_WORLDS": "999",
            "RUSTFLAGS": "-C target-cpu=native",
            "CARGO_TARGET_AARCH64_APPLE_DARWIN_RUSTFLAGS": "-C opt-level=0",
            "AB_PAIRS": "1",
            "UNRELATED_TEST_VALUE": "retained",
        }
        with mock.patch.dict(os.environ, injected, clear=False):
            environment, removed = expert_iteration._sanitized_environment()
        for name in set(injected) - {"UNRELATED_TEST_VALUE"}:
            self.assertNotIn(name, environment)
            self.assertIn(name, removed)
        self.assertEqual(environment["UNRELATED_TEST_VALUE"], "retained")

    def test_model_ab_failure_cannot_be_resumed_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for name in ("candidate.onnx", "candidate.manifest.json", "candidate.golden.json"):
                path = os.path.join(directory, name)
                with open(path, "wb") as handle:
                    handle.write(name.encode())
                paths[name] = path
            arm = {
                "complete_pairs": 2,
                "failed_hands": 0,
                "seed": 0x5EED,
                "per_deck_winrate": [0.0, 0.0],
                "per_deck_margin": [0.0, 0.0],
                "per_deck_level_utility": [0.0, 0.0],
            }
            for name in ("embedded.json", "candidate.json"):
                with open(os.path.join(directory, name), "w") as handle:
                    json.dump(arm, handle)
            common = {
                "model": paths["candidate.onnx"],
                "manifest": paths["candidate.manifest.json"],
                "golden": paths["candidate.golden.json"],
                "outdir": directory,
                "pairs": 2,
                "seed": "0x5EED",
                "budget_ms": 5,
            }
            passing = model_ab.build_comparison(
                **common, minimum_level_delta=0.0
            )
            self.assertTrue(model_ab.write_comparison(directory, passing))
            self.assertTrue(os.path.isfile(os.path.join(directory, "comparison.json")))
            failing = model_ab.build_comparison(
                **common, minimum_level_delta=0.1
            )
            self.assertFalse(model_ab.write_comparison(directory, failing))
            success_path = os.path.join(directory, "comparison.json")
            self.assertFalse(os.path.exists(success_path))
            self.assertTrue(
                os.path.isfile(os.path.join(directory, "comparison.failed.json"))
            )
            self.assertFalse(
                model_ab.comparison_is_reusable(
                    comparison_path=success_path,
                    minimum_level_delta=0.1,
                    **{key: value for key, value in common.items() if key != "outdir"},
                )
            )

            # A zero mean with a materially negative lower confidence bound
            # must not pass merely because its point estimate clears -0.1.
            noisy_candidate = dict(arm)
            noisy_candidate["per_deck_level_utility"] = [-1.0, 1.0]
            with open(os.path.join(directory, "candidate.json"), "w") as handle:
                json.dump(noisy_candidate, handle)
            uncertain = model_ab.build_comparison(
                **common, minimum_level_delta=-0.1
            )
            self.assertEqual(
                uncertain["level_utility"]["candidate_minus_embedded"], 0.0
            )
            self.assertLess(
                uncertain["promotion_gate"]["observed_lower95"], -0.1
            )
            self.assertFalse(uncertain["promotion_gate"]["passed"])

    def test_model_resume_is_bound_to_dataset_and_manifest_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            model = os.path.join(directory, "candidate.onnx")
            golden = f"{model}.golden.json"
            dataset = os.path.join(directory, "data.csv")
            dataset_manifest = f"{dataset}.manifest.json"
            for path, content in (
                (model, b"model"),
                (golden, b"golden"),
                (dataset, b"dataset"),
                (dataset_manifest, b"dataset-manifest"),
            ):
                with open(path, "wb") as handle:
                    handle.write(content)
            settings = {
                "epochs": 4,
                "policy_weight": 1.0,
                "value_weight": 1.0,
                "q_weight": 0.5,
                "auxiliary_weight": 0.25,
                "policy_target": "teacher",
                "early_stop_metric": "policy",
            }
            manifest = {
                "model_sha256": file_hash(model),
                "golden_sha256": file_hash(golden),
                "dataset_sha256": file_hash(dataset),
                "dataset_manifest_sha256": file_hash(dataset_manifest),
                "training": {"seed": 0, **settings},
            }
            with open(f"{model}.manifest.json", "w") as handle:
                json.dump(manifest, handle)
            self.assertTrue(
                pipeline_contracts.expert_model_is_reusable(
                    model=model, dataset=dataset, **settings
                )
            )
            with open(dataset, "ab") as handle:
                handle.write(b"-changed")
            self.assertFalse(
                pipeline_contracts.expert_model_is_reusable(
                    model=model, dataset=dataset, **settings
                )
            )

    def test_dataset_manifest_hash_is_enforced_by_trainer(self):
        with tempfile.TemporaryDirectory() as directory:
            generated = os.path.join(directory, "generated.csv")
            output = os.path.join(directory, "composed.csv")
            write_schema3(generated)
            prepare_expert_data.compose_dataset(generated, output)
            with open(output, "a") as handle:
                handle.write("\n")
            with self.assertRaises(SystemExit):
                train_expert.load_dataset(output)

    def test_phase_dataset_requires_honest_legal_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "bid.csv")
            feature_names = train_phase.FEATURE_NAMES["bid"]
            fields = [
                "schema_version",
                "game_id",
                "trajectory_family_id",
                "group",
                "candidate_id",
                "label",
                *feature_names,
            ]
            with open(path, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fields, lineterminator="\n")
                writer.writeheader()
                for game in range(2):
                    for candidate in range(2):
                        writer.writerow(
                            {
                                "schema_version": 1,
                                "game_id": f"phase-game-{game}",
                                "trajectory_family_id": f"phase-game-{game}",
                                "group": f"phase-game-{game}-decision-0",
                                "candidate_id": candidate,
                                "label": int(candidate == game % 2),
                                **{
                                    name: (index + candidate) / 20
                                    for index, name in enumerate(feature_names)
                                },
                            }
                        )
            sidecar = {
                "manifest_version": 1,
                "dataset_schema_version": 1,
                "phase": "bid",
                "contract": "honest_bid_action_ranker",
                "feature_schema_version": 1,
                "feature_dim": 20,
                "feature_names": feature_names,
                "logit_semantics": train_phase.LOGIT_SEMANTICS,
                "training_domain": train_phase.TRAINING_DOMAINS["bid"],
                "content_sha256": file_hash(path),
                "verification": {
                    "honest_observations": True,
                    "legal_candidates": True,
                    "selected_actions_legal": True,
                    "complete_trajectory_ids": True,
                    "exporter": "test-exporter-v1",
                },
            }
            with open(f"{path}.manifest.json", "w") as handle:
                json.dump(sidecar, handle)
            dataset = train_phase.load_dataset(path, "bid")
            self.assertEqual(len(dataset.groups), 2)
            training, validation, train_ids, validation_ids = train_phase.split_by_family(
                dataset.groups, 0.5, 5
            )
            self.assertTrue(training and validation)
            self.assertTrue(set(train_ids).isdisjoint(validation_ids))

            sidecar["verification"]["honest_observations"] = False
            with open(f"{path}.manifest.json", "w") as handle:
                json.dump(sidecar, handle)
            with self.assertRaises(SystemExit):
                train_phase.load_dataset(path, "bid")


if __name__ == "__main__":
    unittest.main()
