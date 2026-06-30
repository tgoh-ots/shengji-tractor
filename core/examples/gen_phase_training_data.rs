//! Mechanics-backed imitation data for the experimental bid and kitty rankers.
//!
//! Every feature row is encoded from the acting player's redacted state. Labels
//! reproduce the existing deterministic honest heuristic; they validate the
//! training/serving contract but are not evidence that a learned model is
//! stronger than that teacher.
//!
//! Configuration is intentionally small and reproducible by default. Override
//! `PHASE_GAMES`, `PHASE_SEED`, `PHASE_BID_OUT`, or `PHASE_KITTY_OUT` for larger
//! runs or alternate output locations.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::Serialize;
use sha2::{Digest, Sha256};
use shengji_core::bot::harness::{self, PlayBrain};
use shengji_core::bot::heuristics::{self, HeuristicVersion};
use shengji_core::bot::{phase, policy, BotDifficulty};
use shengji_core::game_state::draw_phase::DrawPhase;
use shengji_core::game_state::exchange_phase::ExchangePhase;
use shengji_core::game_state::GameState;
use shengji_core::interactive::Action;
use shengji_mechanics::bidding::Bid;
use shengji_mechanics::deck::Deck;
use shengji_mechanics::types::{Card, PlayerID, Rank, Trump, FULL_DECK};

const DATASET_SCHEMA_VERSION: u32 = 1;
const EXPORTER: &str = "shengji-core/gen_phase_training_data";
const MAX_STEPS: usize = 2_000_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Phase {
    Bid,
    Kitty,
}

impl Phase {
    fn name(self) -> &'static str {
        match self {
            Self::Bid => "bid",
            Self::Kitty => "kitty",
        }
    }

    fn contract(self) -> &'static str {
        match self {
            Self::Bid => "honest_bid_action_ranker",
            Self::Kitty => "honest_kitty_card_ranker",
        }
    }

    fn training_domain(self) -> &'static str {
        match self {
            Self::Bid => "four_player_tractor_two_full_standard_decks_deal_complete_heuristic_v1",
            Self::Kitty => {
                "four_player_tractor_two_full_standard_decks_initial_exchange_heuristic_v1"
            }
        }
    }

    fn feature_names(self) -> Vec<&'static str> {
        match self {
            Self::Bid => phase::BID_FEATURE_NAMES.to_vec(),
            Self::Kitty => phase::KITTY_FEATURE_NAMES.to_vec(),
        }
    }
}

#[derive(Clone)]
struct CandidateRow {
    game_id: String,
    family_id: String,
    group_id: String,
    candidate_id: usize,
    label: u8,
    card_id: usize,
    copy_index: usize,
    features: [f32; 20],
}

#[derive(Default)]
struct GameRows {
    bid: Vec<CandidateRow>,
    kitty: Vec<CandidateRow>,
}

#[derive(Serialize)]
struct Verification {
    honest_observations: bool,
    legal_candidates: bool,
    selected_actions_legal: bool,
    complete_trajectory_ids: bool,
    exporter: &'static str,
}

#[derive(Serialize)]
struct Manifest {
    manifest_version: u32,
    dataset_schema_version: u32,
    phase: &'static str,
    contract: &'static str,
    feature_schema_version: u32,
    feature_dim: usize,
    feature_names: Vec<&'static str>,
    logit_semantics: &'static str,
    training_domain: &'static str,
    content_sha256: String,
    seed: u64,
    games_requested: usize,
    games_completed: usize,
    games_dropped: usize,
    trajectory_families: usize,
    groups: usize,
    rows: usize,
    teacher: &'static str,
    target_semantics: &'static str,
    candidate_identity_contract: &'static str,
    research_note: &'static str,
    verification: Verification,
}

fn main() {
    let games = env_usize("PHASE_GAMES", 4).max(1);
    let seed = env_u64("PHASE_SEED", 0x0202_6063_0517_7B1D);
    let bid_out =
        std::env::var("PHASE_BID_OUT").unwrap_or_else(|_| "training/phase_bid_data.csv".to_owned());
    let kitty_out = std::env::var("PHASE_KITTY_OUT")
        .unwrap_or_else(|_| "training/phase_kitty_data.csv".to_owned());

    let mut bid_rows = Vec::new();
    let mut kitty_rows = Vec::new();
    let mut completed = 0usize;
    for game_index in 0..games {
        let game_seed = derive_game_seed(seed, game_index as u64);
        if let Some(rows) = play_complete_trajectory(game_index, game_seed) {
            completed += 1;
            bid_rows.extend(rows.bid);
            kitty_rows.extend(rows.kitty);
        }
    }
    assert!(completed > 0, "phase generator completed no trajectories");
    let dropped = games - completed;
    assert!(
        dropped.saturating_mul(10) <= games.max(1),
        "phase generator dropped too many trajectories: {}/{}",
        dropped,
        games
    );
    assert!(
        !bid_rows.is_empty(),
        "phase generator produced no bid groups"
    );
    assert!(
        !kitty_rows.is_empty(),
        "phase generator produced no kitty groups"
    );

    write_dataset(Phase::Bid, &bid_out, &bid_rows, seed, games, completed);
    write_dataset(
        Phase::Kitty,
        &kitty_out,
        &kitty_rows,
        seed,
        games,
        completed,
    );
    eprintln!(
        "wrote {} bid rows and {} kitty rows from {completed}/{games} complete trajectories",
        bid_rows.len(),
        kitty_rows.len()
    );
}

fn play_complete_trajectory(game_index: usize, game_seed: u64) -> Option<GameRows> {
    let mut rng = StdRng::seed_from_u64(game_seed);
    let draw = harness::seeded_draw_phase(&[Deck::default(), Deck::default()], &mut rng);
    let seats = draw
        .propagated()
        .players()
        .iter()
        .map(|player| player.id)
        .collect::<Vec<_>>();
    let game_id = format!("phase-game-{game_index:08}-seed-{game_seed:016x}");
    let family_id = format!("phase-family-{game_seed:016x}");
    let mut state = GameState::Draw(draw);
    let mut rows = GameRows::default();
    let mut bid_decision = 0usize;
    let mut kitty_recorded = false;

    for _ in 0..MAX_STEPS {
        match &mut state {
            GameState::Initialize(_) => return None,
            GameState::Draw(draw) => {
                if !draw.done_drawing() {
                    let actor = draw.next_player().ok()?;
                    draw.draw_card(actor).ok()?;
                } else if draw.bid_decided() {
                    let responsible = draw.next_player().ok()?;
                    state = GameState::Exchange(draw.advance(responsible).ok()?);
                } else if !choose_and_apply_bid(
                    draw,
                    &seats,
                    &game_id,
                    &family_id,
                    &mut bid_decision,
                    &mut rows.bid,
                )? && draw.reveal_card().is_err()
                {
                    let fallback = seats.iter().find_map(|&seat| {
                        draw.valid_bids(seat)
                            .ok()?
                            .into_iter()
                            .min_by_key(|bid| (bid.count, bid.card.as_char()))
                    })?;
                    if !draw.bid(fallback.id, fallback.card, fallback.count) {
                        return None;
                    }
                }
            }
            GameState::Exchange(exchange) => {
                let actor = exchange.next_player().ok()?;
                if !kitty_recorded && !exchange.finalized() {
                    rows.kitty
                        .extend(kitty_training_rows(exchange, actor, &game_id, &family_id)?);
                    kitty_recorded = true;
                }
                let view = GameState::Exchange(exchange.clone()).for_player(actor);
                match policy::select_action(&view, actor, BotDifficulty::Easy)
                    .ok()
                    .flatten()?
                {
                    Action::MoveCardToKitty(card) => {
                        exchange.move_card_to_kitty(actor, card).ok()?;
                    }
                    Action::MoveCardToHand(card) => {
                        exchange.move_card_to_hand(actor, card).ok()?;
                    }
                    Action::SetFriends(friends) => {
                        exchange.set_friends(actor, friends).ok()?;
                    }
                    Action::Bid(card, count) if exchange.bid(actor, card, count) => {}
                    Action::PickUpKitty => exchange.pick_up_cards(actor).ok()?,
                    Action::PutDownKitty => exchange.finalize(actor).ok()?,
                    Action::BeginPlay => {
                        state = GameState::Play(exchange.advance(actor).ok()?);
                    }
                    _ => return None,
                }
            }
            GameState::Play(play) => {
                if play.game_finished() {
                    play.current_game_score().ok()?;
                    return Some(rows);
                }
                match play.trick().next_player() {
                    None => {
                        play.finish_trick().ok()?;
                    }
                    Some(actor) => {
                        let cards = harness::play_cards_for(
                            play,
                            actor,
                            &PlayBrain::HeuristicDirect(HeuristicVersion::New),
                        )?;
                        play.play_cards(actor, &cards).ok()?;
                    }
                }
            }
        }
    }
    None
}

fn choose_and_apply_bid(
    draw: &mut DrawPhase,
    seats: &[PlayerID],
    game_id: &str,
    family_id: &str,
    decision: &mut usize,
    rows: &mut Vec<CandidateRow>,
) -> Option<bool> {
    for &actor in seats {
        let honest_state = GameState::Draw(draw.clone()).for_player(actor);
        let honest = match &honest_state {
            GameState::Draw(view) => view,
            _ => return None,
        };
        let Some(selected) = policy::choose_bid(honest, actor, BotDifficulty::Easy) else {
            continue;
        };
        let mut candidates = honest.valid_bids(actor).ok()?;
        candidates.sort_by_key(|bid| (bid.card.as_char(), bid.count, bid.epoch, bid.id.0));
        let selected_index = candidates.iter().position(|bid| *bid == selected)?;
        if candidates.len() >= 2 {
            let group_id = format!("{game_id}:bid:{:06}", *decision);
            let hand = honest
                .hands()
                .get(actor)
                .ok()?
                .iter()
                .flat_map(|(&card, &count)| std::iter::repeat_n(card, count))
                .collect::<Vec<_>>();
            for (candidate_id, &bid) in candidates.iter().enumerate() {
                let trump = candidate_trump(honest, actor, bid)?;
                let strength = heuristics::bid_strength(&hand, trump) - bid.count as f64 * 0.5;
                let features = phase::bid_features(honest, actor, bid, trump, strength);
                if features.iter().any(|value| !value.is_finite()) {
                    return None;
                }
                rows.push(CandidateRow {
                    game_id: game_id.to_owned(),
                    family_id: family_id.to_owned(),
                    group_id: group_id.clone(),
                    candidate_id,
                    label: u8::from(candidate_id == selected_index),
                    card_id: FULL_DECK.iter().position(|card| *card == bid.card)?,
                    copy_index: 0,
                    features,
                });
            }
            *decision += 1;
        }
        return draw
            .bid(selected.id, selected.card, selected.count)
            .then_some(true);
    }
    Some(false)
}

fn candidate_trump(draw: &DrawPhase, actor: PlayerID, bid: Bid) -> Option<Trump> {
    let level_seat = draw.propagated().landlord().unwrap_or(actor);
    let level = draw
        .propagated()
        .players()
        .iter()
        .find(|player| player.id == level_seat)
        .map(|player| player.rank())
        .unwrap_or(Rank::Number(shengji_mechanics::types::Number::Two));
    match bid.card {
        Card::SmallJoker | Card::BigJoker => Some(heuristics::trump_for(level, None)),
        Card::Suited { suit, .. } => Some(heuristics::trump_for(level, Some(suit))),
        Card::Unknown => None,
    }
}

#[derive(Clone, Copy)]
struct PhysicalCard {
    card: Card,
    copy_index: usize,
}

fn canonical_physical_cards(cards: impl IntoIterator<Item = Card>) -> Vec<PhysicalCard> {
    let mut cards = cards.into_iter().collect::<Vec<_>>();
    cards.sort_unstable_by_key(|card| card.as_char());
    let mut seen = HashMap::new();
    cards
        .into_iter()
        .map(|card| {
            let copy_index = seen.entry(card).or_insert(0usize);
            let physical = PhysicalCard {
                card,
                copy_index: *copy_index,
            };
            *copy_index += 1;
            physical
        })
        .collect()
}

fn kitty_training_rows(
    exchange: &ExchangePhase,
    actor: PlayerID,
    game_id: &str,
    family_id: &str,
) -> Option<Vec<CandidateRow>> {
    let honest_state = GameState::Exchange(exchange.clone()).for_player(actor);
    let honest = match &honest_state {
        GameState::Exchange(view) => view,
        _ => return None,
    };
    if honest.exchanger() != actor || honest.finalized() {
        return None;
    }
    let mut pool = honest
        .hands()
        .get(actor)
        .ok()?
        .iter()
        .flat_map(|(&card, &count)| std::iter::repeat_n(card, count))
        .collect::<Vec<_>>();
    pool.extend_from_slice(honest.visible_kitty()?);
    if pool.contains(&Card::Unknown) {
        return None;
    }
    let trump = honest.trump();
    let kitty_size = honest.kitty_size();
    let selected = heuristics::choose_kitty(&pool, trump, kitty_size);
    if selected.len() != kitty_size {
        return None;
    }

    let canonical = canonical_physical_cards(pool.iter().copied());
    let canonical_pool = canonical
        .iter()
        .map(|candidate| candidate.card)
        .collect::<Vec<_>>();
    let mut remaining = canonical;
    let mut remaining_selected = selected.clone();
    let mut rows = Vec::new();
    for (slot, selected_card) in selected.into_iter().enumerate() {
        let selected_index = remaining
            .iter()
            .position(|candidate| candidate.card == selected_card)?;
        let selected_counts = Card::count(remaining_selected.iter().copied());
        let mut marked = HashMap::<Card, usize>::new();
        let group_id = format!("{game_id}:kitty:{slot:03}");
        for (candidate_id, candidate) in remaining.iter().enumerate() {
            let already = marked.entry(candidate.card).or_default();
            let baseline_selected = *already
                < selected_counts
                    .get(&candidate.card)
                    .copied()
                    .unwrap_or_default();
            *already += 1;
            let features = phase::kitty_features(
                &canonical_pool,
                trump,
                kitty_size,
                candidate.card,
                baseline_selected,
            );
            if features.iter().any(|value| !value.is_finite()) {
                return None;
            }
            rows.push(CandidateRow {
                game_id: game_id.to_owned(),
                family_id: family_id.to_owned(),
                group_id: group_id.clone(),
                candidate_id,
                label: u8::from(candidate_id == selected_index),
                card_id: FULL_DECK.iter().position(|card| *card == candidate.card)?,
                copy_index: candidate.copy_index,
                features,
            });
        }
        remaining.remove(selected_index);
        let target_index = remaining_selected
            .iter()
            .position(|card| *card == selected_card)?;
        remaining_selected.remove(target_index);
    }
    Some(rows)
}

fn write_dataset(
    phase: Phase,
    path: &str,
    rows: &[CandidateRow],
    seed: u64,
    requested: usize,
    completed: usize,
) {
    ensure_parent(path);
    let file = File::create(path).expect("create phase dataset");
    let mut writer = BufWriter::new(file);
    write!(
        writer,
        "schema_version,game_id,trajectory_family_id,group,candidate_id,label,card_id,copy_index"
    )
    .unwrap();
    for name in phase.feature_names() {
        write!(writer, ",{name}").unwrap();
    }
    writeln!(writer).unwrap();
    for row in rows {
        write!(
            writer,
            "{},{},{},{},{},{},{},{}",
            DATASET_SCHEMA_VERSION,
            row.game_id,
            row.family_id,
            row.group_id,
            row.candidate_id,
            row.label,
            row.card_id,
            row.copy_index,
        )
        .unwrap();
        for value in row.features {
            write!(writer, ",{value:.8}").unwrap();
        }
        writeln!(writer).unwrap();
    }
    writer.flush().expect("flush phase dataset");
    drop(writer);

    let groups = rows
        .iter()
        .map(|row| row.group_id.as_str())
        .collect::<std::collections::HashSet<_>>()
        .len();
    let manifest = Manifest {
        manifest_version: 1,
        dataset_schema_version: DATASET_SCHEMA_VERSION,
        phase: phase.name(),
        contract: phase.contract(),
        feature_schema_version: phase::FEATURE_SCHEMA_VERSION,
        feature_dim: 20,
        feature_names: phase.feature_names(),
        logit_semantics: "relative_listwise_rank_only",
        training_domain: phase.training_domain(),
        content_sha256: file_sha256(path),
        seed,
        games_requested: requested,
        games_completed: completed,
        games_dropped: requested - completed,
        trajectory_families: completed,
        groups,
        rows: rows.len(),
        teacher: "deterministic honest mechanics heuristic",
        target_semantics: "one selected candidate per listwise decision group",
        candidate_identity_contract: match phase {
            Phase::Bid => "mechanics-valid Bid sorted by card/count/epoch/player",
            Phase::Kitty => {
                "physical card copies sorted by card identity then zero-based copy index"
            }
        },
        research_note: "imitation targets validate plumbing; they are not strength evidence",
        verification: Verification {
            honest_observations: true,
            legal_candidates: true,
            selected_actions_legal: true,
            complete_trajectory_ids: true,
            exporter: EXPORTER,
        },
    };
    let manifest_path = format!("{path}.manifest.json");
    ensure_parent(&manifest_path);
    let mut manifest_writer =
        BufWriter::new(File::create(&manifest_path).expect("create phase dataset manifest"));
    serde_json::to_writer_pretty(&mut manifest_writer, &manifest)
        .expect("write phase dataset manifest");
    writeln!(manifest_writer).expect("finish phase dataset manifest");
}

fn file_sha256(path: &str) -> String {
    let bytes = std::fs::read(path).expect("read phase dataset for SHA-256");
    format!("{:x}", Sha256::digest(bytes))
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
        std::fs::create_dir_all(parent).expect("create phase output parent");
    }
}
