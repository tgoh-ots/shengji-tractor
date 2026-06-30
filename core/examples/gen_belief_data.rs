//! Seeded honest-belief dataset: (public observation, card identity) -> hidden
//! relative seat / kitty destination, with hard legality masks.

use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;

use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::Serialize;
use sha2::{Digest, Sha256};
use shengji_core::bot::determinize::Knowledge;
use shengji_core::bot::harness::seeded_draw_phase;
use shengji_core::bot::{belief, policy, BotDifficulty};
use shengji_core::game_state::play_phase::PlayPhase;
use shengji_core::game_state::GameState;
use shengji_core::interactive::Action;
use shengji_mechanics::deck::Deck;
use shengji_mechanics::types::{Card, PlayerID, FULL_DECK};

const DATASET_SCHEMA_VERSION: u32 = 1;
const BEHAVIOUR_POLICY_DOMAIN: &str = "bidding=expert;exchange=easy;play=easy";
const TARGET_SEMANTICS: &str =
    "per-hidden-card destination marginals excluding publicly pinned holdings; rows in a snapshot are correlated physical copies";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FeatureSchema {
    V1,
    V2,
}

impl FeatureSchema {
    fn from_env() -> Self {
        match std::env::var("BELIEF_FEATURE_SCHEMA_VERSION").as_deref() {
            Ok("2") => Self::V2,
            Ok("1") | Err(_) => Self::V1,
            Ok(value) => panic!(
                "BELIEF_FEATURE_SCHEMA_VERSION must be 1 or 2, got {:?}",
                value
            ),
        }
    }

    fn version(self) -> u32 {
        match self {
            Self::V1 => 1,
            Self::V2 => 2,
        }
    }

    fn dimension(self) -> usize {
        match self {
            Self::V1 => belief::FEATURE_DIM,
            Self::V2 => belief::FEATURE_DIM_V2,
        }
    }

    fn feature_names(self) -> Vec<String> {
        belief::belief_feature_names(self.version())
    }
}

#[derive(Serialize)]
struct Manifest {
    manifest_version: u32,
    dataset_schema_version: u32,
    feature_schema_version: u32,
    feature_dim: usize,
    feature_names: Vec<String>,
    encoder_contract: &'static str,
    encoder_source_sha256: String,
    seed: u64,
    games_requested: usize,
    games_completed: usize,
    games_dropped: usize,
    snapshots: usize,
    rows: usize,
    csv_sha256: String,
    snapshot_every: usize,
    behaviour: &'static str,
    behaviour_policy_domain: &'static str,
    behaviour_budget_ms: u64,
    target_classes: [&'static str; 4],
    target_semantics: &'static str,
    publicly_pinned_targets_excluded: bool,
    legality_contract: &'static str,
    supported_game_contract: &'static str,
    public_history_contract: &'static str,
}

struct BeliefRow {
    game_id: String,
    snapshot_id: usize,
    actor: PlayerID,
    card_id: usize,
    target: usize,
    mask: [u8; 4],
    features: Vec<f32>,
}

fn main() {
    let games = env_usize("BELIEF_GAMES", 100);
    let seed = env_u64("BELIEF_SEED", 0x00BE_11EF);
    let every = env_usize("BELIEF_SNAPSHOT_EVERY", 4).max(1);
    let budget = env_u64("BELIEF_BEHAVIOUR_BUDGET_MS", 20).max(1);
    let feature_schema = FeatureSchema::from_env();
    let out =
        std::env::var("BELIEF_OUT").unwrap_or_else(|_| "training/belief_data.csv".to_string());
    let manifest_path =
        std::env::var("BELIEF_MANIFEST").unwrap_or_else(|_| format!("{out}.manifest.json"));
    ensure_parent(&out);
    ensure_parent(&manifest_path);
    let mut writer = BufWriter::new(File::create(&out).expect("create belief CSV"));
    write_header(&mut writer, feature_schema);
    let mut completed = 0;
    let mut snapshots = 0;
    let mut rows = 0;
    for game_index in 0..games {
        let game_seed = derive_game_seed(seed, game_index as u64);
        if let Some(game_rows) = play_game(game_index, game_seed, every, budget, feature_schema) {
            completed += 1;
            snapshots += game_rows
                .last()
                .map(|row| row.snapshot_id + 1)
                .unwrap_or_default();
            rows += game_rows.len();
            for row in game_rows {
                write_row(&mut writer, row, feature_schema);
            }
        }
    }
    writer.flush().expect("flush belief CSV");
    drop(writer);
    let csv_sha256 = sha256_file(&out);
    let manifest = Manifest {
        manifest_version: feature_schema.version(),
        dataset_schema_version: DATASET_SCHEMA_VERSION,
        feature_schema_version: feature_schema.version(),
        feature_dim: feature_schema.dimension(),
        feature_names: feature_schema.feature_names(),
        encoder_contract: belief::belief_encoder_contract(feature_schema.version())
            .expect("supported belief encoder schema"),
        encoder_source_sha256: belief::belief_encoder_source_sha256(),
        seed,
        games_requested: games,
        games_completed: completed,
        games_dropped: games.saturating_sub(completed),
        snapshots,
        rows,
        csv_sha256,
        snapshot_every: every,
        behaviour: "easy-play/expert-bid",
        behaviour_policy_domain: BEHAVIOUR_POLICY_DOMAIN,
        behaviour_budget_ms: budget,
        target_classes: ["next-seat", "opposite-seat", "previous-seat", "kitty"],
        target_semantics: TARGET_SEMANTICS,
        publicly_pinned_targets_excluded: true,
        legality_contract: "mask=1 iff destination has capacity and no public effective-suit void",
        supported_game_contract: "tractor:4p:2x-standard:kitty8:no-removed",
        public_history_contract: match feature_schema {
            FeatureSchema::V1 => "schema-v1 aggregate public-state features",
            FeatureSchema::V2 => {
                "schema-v1 prefix plus ordered tail of 4 public bids and 8 public plays"
            }
        },
    };
    serde_json::to_writer_pretty(
        BufWriter::new(File::create(&manifest_path).expect("create belief manifest")),
        &manifest,
    )
    .expect("write belief manifest");
    let dropped = games.saturating_sub(completed);
    assert!(completed > 0, "belief generator completed no games");
    assert!(
        dropped.saturating_mul(10) <= games.max(1),
        "belief generator silently dropped too many trajectories: {}/{}",
        dropped,
        games
    );
    eprintln!("wrote {rows} belief rows / {snapshots} snapshots / {completed} games to {out}");
}

fn play_game(
    game_index: usize,
    seed: u64,
    every: usize,
    budget: u64,
    feature_schema: FeatureSchema,
) -> Option<Vec<BeliefRow>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let draw = seeded_draw_phase(&[Deck::default(), Deck::default()], &mut rng);
    let seats: Vec<_> = draw.propagated().players().iter().map(|p| p.id).collect();
    let mut state = GameState::Draw(draw);
    let mut decision = 0usize;
    let mut snapshot = 0usize;
    let mut rows = Vec::new();
    for _ in 0..2_000_000 {
        match &mut state {
            GameState::Initialize(_) => return None,
            GameState::Draw(draw) => {
                if !draw.done_drawing() {
                    let player = draw.next_player().ok()?;
                    draw.draw_card(player).ok()?;
                } else if draw.bid_decided() {
                    let responsible = draw.next_player().ok()?;
                    state = GameState::Exchange(draw.advance(responsible).ok()?);
                } else {
                    let mut bid = false;
                    for &seat in &seats {
                        if let Some(candidate) =
                            policy::choose_bid(draw, seat, BotDifficulty::Expert)
                        {
                            if draw.bid(seat, candidate.card, candidate.count) {
                                bid = true;
                                break;
                            }
                        }
                    }
                    if !bid && draw.reveal_card().is_err() {
                        for &seat in &seats {
                            if let Some(candidate) = draw
                                .valid_bids(seat)
                                .ok()?
                                .into_iter()
                                .min_by_key(|candidate| candidate.count)
                            {
                                if draw.bid(seat, candidate.card, candidate.count) {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            GameState::Exchange(exchange) => {
                let landlord = exchange.landlord();
                let view = GameState::Exchange(exchange.clone()).for_player(landlord);
                match policy::select_action_with_search_budget(
                    &view,
                    landlord,
                    BotDifficulty::Easy,
                    budget,
                )
                .ok()
                .flatten()
                {
                    Some(Action::MoveCardToKitty(card)) => {
                        exchange.move_card_to_kitty(landlord, card).ok()?;
                    }
                    Some(Action::MoveCardToHand(card)) => {
                        exchange.move_card_to_hand(landlord, card).ok()?;
                    }
                    Some(Action::SetFriends(friends)) => {
                        exchange.set_friends(landlord, friends).ok()?;
                    }
                    _ => state = GameState::Play(exchange.advance(landlord).ok()?),
                }
            }
            GameState::Play(play) => {
                if play.game_finished() {
                    return Some(rows);
                }
                match play.trick().next_player() {
                    None => {
                        play.finish_trick().ok()?;
                    }
                    Some(actor) => {
                        if decision.is_multiple_of(every) {
                            rows.extend(snapshot_rows(
                                play,
                                actor,
                                game_index,
                                seed,
                                snapshot,
                                feature_schema,
                            )?);
                            snapshot += 1;
                        }
                        decision += 1;
                        let view = GameState::Play(play.clone()).for_player(actor);
                        let cards = match policy::select_action_with_search_budget(
                            &view,
                            actor,
                            BotDifficulty::Easy,
                            budget,
                        )
                        .ok()
                        .flatten()
                        {
                            Some(Action::PlayCards(cards)) => cards,
                            _ => return None,
                        };
                        play.play_cards(actor, &cards).ok()?;
                    }
                }
            }
        }
    }
    None
}

fn snapshot_rows(
    full: &PlayPhase,
    actor: PlayerID,
    game_index: usize,
    game_seed: u64,
    snapshot_id: usize,
    feature_schema: FeatureSchema,
) -> Option<Vec<BeliefRow>> {
    let honest_state = GameState::Play(full.clone()).for_player(actor);
    let honest = match &honest_state {
        GameState::Play(play) => play,
        _ => return None,
    };
    let knowledge = Knowledge::from_play_view(honest, actor);
    let seats: Vec<_> = full.propagated().players().iter().map(|p| p.id).collect();
    if seats.len() != 4 {
        return None;
    }
    let actor_index = seats.iter().position(|seat| *seat == actor)?;
    let mut destinations: Vec<(usize, Card)> = Vec::new();
    for (seat_index, seat) in seats.iter().enumerate() {
        if *seat == actor {
            continue;
        }
        let class = (seat_index + seats.len() - actor_index) % seats.len() - 1;
        let mut publicly_pinned = knowledge
            .known_holding
            .get(seat)
            .cloned()
            .unwrap_or_default();
        for (card, count) in full.hands().get(*seat).ok()? {
            if *card != Card::Unknown {
                let excluded = publicly_pinned.remove(card).unwrap_or_default().min(*count);
                destinations.extend(std::iter::repeat_n((class, *card), *count - excluded));
            }
        }
    }
    let kitty = full.visible_kitty()?;
    let kitty_is_hidden = honest.visible_kitty().is_none();
    if kitty_is_hidden {
        destinations.extend(kitty.iter().copied().map(|card| (3, card)));
    }
    let hidden_total = destinations.len();
    let mut rows = Vec::with_capacity(hidden_total);
    for (target, card) in destinations {
        let card_id = FULL_DECK.iter().position(|known| *known == card)?;
        let (features, encoded_mask) = match feature_schema {
            FeatureSchema::V1 => {
                let (features, mask) =
                    belief::encode_belief_features(honest, actor, &knowledge, card);
                (features.to_vec(), mask)
            }
            FeatureSchema::V2 => {
                let (features, mask) =
                    belief::encode_belief_features_v2(honest, actor, &knowledge, card);
                (features.to_vec(), mask)
            }
        };
        let mask = encoded_mask.map(|value| u8::from(value > 0.5));
        if mask[target] == 0 {
            return None;
        }
        rows.push(BeliefRow {
            game_id: format!("belief-seed-{game_seed}-game-{game_index}"),
            snapshot_id,
            actor,
            card_id,
            target,
            mask,
            features,
        });
    }
    Some(rows)
}

fn write_header(writer: &mut impl Write, feature_schema: FeatureSchema) {
    write!(
        writer,
        "schema_version,feature_schema_version,game_id,snapshot_id,actor,card_id,target"
    )
    .unwrap();
    for index in 0..4 {
        write!(writer, ",mask{index}").unwrap();
    }
    for name in feature_schema.feature_names() {
        write!(writer, ",{name}").unwrap();
    }
    writeln!(writer).unwrap();
}

fn write_row(writer: &mut impl Write, row: BeliefRow, feature_schema: FeatureSchema) {
    write!(
        writer,
        "{},{},{},{},{},{},{}",
        DATASET_SCHEMA_VERSION,
        feature_schema.version(),
        row.game_id,
        row.snapshot_id,
        row.actor.0,
        row.card_id,
        row.target
    )
    .unwrap();
    for value in row.mask {
        write!(writer, ",{value}").unwrap();
    }
    for value in row.features {
        write!(writer, ",{value:.6}").unwrap();
    }
    writeln!(writer).unwrap();
}

fn derive_game_seed(base: u64, index: u64) -> u64 {
    let mut z = base.wrapping_add(index.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn ensure_parent(path: &str) {
    if let Some(parent) = Path::new(path).parent() {
        std::fs::create_dir_all(parent).expect("create output parent");
    }
}

fn sha256_file(path: &str) -> String {
    let mut file = File::open(path).expect("open belief CSV for hashing");
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer).expect("read belief CSV for hashing");
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    format!("{:x}", digest.finalize())
}
