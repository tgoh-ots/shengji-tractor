# Expert iteration and offline training experiments

Status: **research tooling, default off**. Nothing in this document promotes a
model or demonstrates a strength gain. Every candidate still has to pass tract
parity, matched-deal A/B, variant checks, latency limits, and a separately
approved production promotion.

## What the loop does

`training/expert_iteration.py` turns the existing schema-v3 value/Q pipeline
into a bounded expert-iteration experiment:

1. Generate honest observations using a configured mixture of Easy, search,
   and Enoch trajectories. The Omniscient search teacher supplies policy labels
   and same-world counterfactual continuations supply sparse Q targets.
2. Train policy, state-V, and action-Q heads and verify PyTorch/ONNX/tract
   numerical parity.
3. Compare the candidate with the prior round on identical deals in separate
   processes. Round zero compares with the embedded model.
4. Load the accepted prior candidate into the next round's search policy, move
   the behaviour distribution toward that policy, and repeat only for the
   explicitly configured number of rounds.

This is search distillation/expert iteration, not unconstrained self-play RL.
The loop never deploys, embeds, or changes a production environment variable.

~~~sh
python3 training/expert_iteration.py plan \
  --config training/expert_iteration.example.json

python3 training/expert_iteration.py run \
  --config training/expert_iteration.example.json \
  --workdir "$HOME/.shengji-expert-iteration"
~~~

The checked example is sized for a real experiment, not a laptop smoke test.
Copy it and explicitly reduce games, shards, epochs, and A/B pairs for a
contract-only test.

## Curriculum and partner/opponent league

Each round contains weighted `league` profiles. `slots` deterministically maps
profiles onto shard indices, so interruption, parallel scheduling, and resume
cannot change which distribution produced a shard. A profile controls:

- trajectory fallback behaviour (`easy`, `expert`, `enoch`, `grandmaster`, or
  an Easy/Expert `mix`);
- behaviour and Omniscient-teacher search budgets;
- number of same-world Q candidates;
- honest continuation policy and continuation budget;
- optional `seat_behaviours`, exactly four entries from those policy families.

Without `seat_behaviours`, the profile's whole-hand `behaviour` remains the
fallback. With it, bidding, exchange, and play use the assigned seat policy;
the generator manifest records all four assignments and each row records the
acting seat's actual policy. Since landlord/team roles arise from the deal and
bidding, rotate heterogeneous assignments across profiles for partner and
opponent coverage.

Every round uses a disjoint deterministic generation-seed range and a distinct
matched-deal A/B seed by default. Decimal and `0x`-prefixed A/B seeds are parsed
to one canonical unsigned-64-bit value and the evaluator's reported seed must
match it. A prior model, its manifest, league configuration, offline inputs,
source tree, trainer, runtime/search environment, compiler/target environment,
and all pipeline settings are content-fingerprinted. The expert-iteration
driver additionally removes inherited `SHENGJI_*`, `GM_*`, `OMNI_*`, pipeline,
and build-target overrides before applying its checked config. Reusing a work
directory after any effective drift fails closed.

Shard success is more than a `.done` file. By default each shard must complete
at least 80% of requested games, emit at least one decision per completed game,
and, when Q generation is enabled, populate Q on at least 1% of rows. The
thresholds are configurable as `min_shard_completion_rate`,
`min_decisions_per_game`, and `min_q_row_fraction`; each shard's measured values
are written into the generated dataset manifest. CSV row/game/decision/Q counts
must also exactly match the generator sidecar. A trained model is reused only
when its manifest hashes match the current composed CSV and CSV manifest, its
own model/golden hashes, and the configured training objectives. A/B reuse
likewise verifies all candidate/baseline artifacts, arm-result hashes, budget,
canonical seed, sample count, and a passed promotion gate. A failed gate is
stored only as `comparison.failed.json`, never as a resumable success marker.

## Deterministic symmetry augmentation

`training/prepare_expert_data.py` can audit:

- `identity`;
- `seat` (four whole-table seat rotations);
- `suit-cyclic` (four global suit renamings);
- `seat-suit-cyclic` (the 16 products of those transformations).

Schema-v2 features encode seats relatively and do not expose absolute suit
identity, so these transformations preserve every feature and target. That also
means materializing them would create 4–16 duplicate model inputs, waste
compute, and change optimization weighting without adding information. The
current modes therefore perform a deterministic **audit only**: action suit
mappings must round-trip, actor mappings remain valid, and the manifest records
every audited transform. Only one canonical identity row is emitted and
`optimization_weight_change=false`.

`trajectory_family_id` remains the split key. Offline IDs are namespaced while
preserving the verifier-provided parent family, so multiple derived game IDs
from one physical replay cannot cross the train/validation boundary. Future sequence/raw-card encoders
that expose absolute identity may materialize transformations, but must retain
the parent family so a validation trajectory cannot leak into training.

Landlord/attacker exchange is deliberately rejected. It is not a symmetry:
contract roles, kitty ownership, score thresholds, and level advancement are
asymmetric. The current compact feature vector already quotients the safe
seat/suit symmetries. Non-identity settings exist to exercise and record that
invariant, not to inflate the dataset.

## Conservative human/offline replay ingestion

Raw websocket/client logs are not accepted as training rows. An upstream replay
exporter must replay the complete hand through the mechanics engine and emit the
same 49-feature candidate table as schema-v3 self-play. Its sidecar must include:

~~~json
{
  "manifest_version": 1,
  "dataset_schema_version": 3,
  "feature_schema_version": 2,
  "feature_dim": 49,
  "game_config": "tractor-4p-2deck",
  "source_kind": "human-replay",
  "source_id": "stable-safe-name",
  "content_sha256": "sha256-of-csv",
  "replay_verification": {
    "complete_trajectories": true,
    "legal_actions": true,
    "honest_observations": true,
    "terminal_targets_recomputed": true,
    "verifier": "name-and-version-of-exporter",
    "raw_replay_sha256": "sha256-of-raw-replay-bundle"
  }
}
~~~

`training/offline_replays.example.json` configures deterministic selection and
a hard fraction cap (10% by default). Ingestion namespaces all IDs, uses the
verified human action as the behaviour/policy label, retains recomputed terminal
V, and **strips every Q target**. A replay shows what happened; it does not prove
the return of actions that were not taken. Offline artifacts are marked
research-only and cannot be automatically promoted.

This is intentionally only the trusted ingestion boundary. Building a replay
exporter for each historical log version remains separate work; weakening the
attestations to ingest unverified logs is not an acceptable shortcut.

## Specialized bid and kitty rankers

`training/train_phase.py` trains a small listwise ranker matching the strict
runtime contracts in `core/src/bot/phase.rs`:

- phase `bid`: contract `honest_bid_action_ranker`, 20 semantic features from
  `hand_size` through `player_count`;
- phase `kitty`: contract `honest_kitty_card_ranker`, features
  `card_points` through `bias`;
- input `features`, output `action_logit`, semantic `policy_logit`.

Both outputs are explicitly `relative_listwise_rank_only`. Their additive zero
and positive scale are unidentifiable, so serving jointly min-max normalizes the
candidate logits before blending ranks. Bid/pass remains entirely on the
existing absolute heuristic threshold; the model can only choose among legal
bid candidates. Kitty logits are normalized over the physical-card candidates,
preserving duplicate multiplicity.

One CSV row is one mechanics-validated candidate. Required identifying columns
are `schema_version,game_id,group,candidate_id,label`; label is one-hot within a
decision. Optional `trajectory_family_id` binds augmentations to one split.
The required dataset sidecar is:

~~~json
{
  "manifest_version": 1,
  "dataset_schema_version": 1,
  "phase": "bid",
  "contract": "honest_bid_action_ranker",
  "feature_schema_version": 1,
  "feature_dim": 20,
  "feature_names": ["hand_size", "...", "player_count"],
  "logit_semantics": "relative_listwise_rank_only",
  "training_domain": "four_player_tractor_two_full_standard_decks_deal_complete_heuristic_v1",
  "content_sha256": "sha256-of-csv",
  "verification": {
    "honest_observations": true,
    "legal_candidates": true,
    "selected_actions_legal": true,
    "complete_trajectory_ids": true,
    "exporter": "name-and-version"
  }
}
~~~

Generate the mechanics-backed imitation corpus, then train:

~~~sh
PHASE_GAMES=200 PHASE_SEED=77 \
PHASE_BID_OUT=/data/bid-candidates.csv \
PHASE_KITTY_OUT=/data/kitty-candidates.csv \
cargo +1.92.0 run --release -p shengji-core \
  --example gen_phase_training_data
~~~

~~~sh
python3 training/train_phase.py --phase bid \
  --data /data/bid-candidates.csv --out /models/bid.onnx

python3 training/train_phase.py --phase kitty \
  --data /data/kitty-candidates.csv --out /models/kitty.onnx
~~~

The model manifest is SHA-bound and always says
`serving_status=experimental_candidate` for verified data. The explicit unsafe
sidecar bypass writes `non_servable_research`, which the runtime rejects.
Runtime also requires the research/safety flags, dataset and golden lineage,
and the exact logit/domain contract. Activation needs an explicit nonzero
phase-model weight:

~~~sh
SHENGJI_BID_MODEL_PATH=/models/bid.onnx \
SHENGJI_BID_MODEL_MANIFEST=/models/bid.onnx.manifest.json \
SHENGJI_KITTY_MODEL_PATH=/models/kitty.onnx \
SHENGJI_KITTY_MODEL_MANIFEST=/models/kitty.onnx.manifest.json \
SHENGJI_PHASE_MODEL_WEIGHT=0.25 your-server-command
~~~

Serving is deliberately narrower than the general game engine: Expert tier,
four-player Tractor, two complete standard decks, no removed cards, fully dealt
bids, and the initial eight-card exchange. Finding Friends, short/special decks,
during-draw bids, exchange overbids, and Enoch/Grandmaster heuristics fall back
exactly to their existing logic. This matches the exporter rather than
extrapolating one artifact into unsupported variants.
Policy imitation is only a baseline:
better phase targets should eventually use contract outcome/level value and
information revealed, evaluated against the existing mechanics-aware heuristic.

## Optional search experiments

The following serving experiments are independently opt-in and have neutral
defaults:

- `SHENGJI_ADAPTIVE_BUDGET=1` with
  `SHENGJI_ADAPTIVE_BUDGET_MIN_FRACTION` lets routine decisions return unused
  wall time while retaining the caller's deadline as a hard ceiling.
- `SHENGJI_SEARCH_STDDEV_PENALTY`, `SHENGJI_SEARCH_CVAR_WEIGHT`, and
  `SHENGJI_SEARCH_CVAR_ALPHA` rank candidates using the empirical lower tail
  across sampled worlds. Zero penalty/weight is the established risk-neutral
  mean.
- `SHENGJI_EXACT_ENDGAME_CARDS` and `SHENGJI_EXACT_ENDGAME_NODES` enable the
  mechanics-validated alpha-beta oracle. Values are hard-capped at 12 cards and
  one million nodes and share the normal wall-clock deadline; an incomplete
  solve is discarded. For an honest bot, “exact” means exact inside one sampled
  perfect-information world; it is not an information-set solution and retains
  strategy-fusion risk.

These knobs need matched wall-time A/B tests. Equal simulation counts are not a
fair comparison when one method changes work per simulation.

## What to measure

For each generation, retain the prior model and report at minimum:

- paired level-utility delta and bootstrap interval versus the prior;
- win-rate and point-margin deltas as secondary outcomes;
- policy top-1, V/Q error and Q ranking only as diagnostics;
- latency, completed-search rate, legality/fallback counts, and variant matrix;
- partner cross-play against frozen old generations and Enoch;
- bid success/contract utility or kitty landlord margin for phase candidates.

The sample config applies its non-inferiority threshold to the lower endpoint of
the paired level-utility bootstrap interval. That is an execution gate, not by
itself evidence of superiority; a strength claim still needs a predeclared
sample size and practical effect threshold.
