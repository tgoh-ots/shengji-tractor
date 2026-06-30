//! Seeded honest-belief dataset: (public observation, card identity) -> hidden
//! relative seat / kitty destination, with hard legality masks.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::Serialize;
use shengji_core::bot::determinize::Knowledge;
use shengji_core::bot::harness::seeded_draw_phase;
use shengji_core::bot::{belief, policy, BotDifficulty};
use shengji_core::game_state::play_phase::PlayPhase;
use shengji_core::game_state::GameState;
use shengji_core::interactive::Action;
use shengji_mechanics::deck::Deck;
use shengji_mechanics::types::{Card, PlayerID, FULL_DECK};

const FEATURE_DIM: usize = belief::FEATURE_DIM;
const SCHEMA_VERSION: u32 = 1;

#[derive(Serialize)]
struct Manifest {
    manifest_version: u32,
    dataset_schema_version: u32,
    feature_dim: usize,
    seed: u64,
    games_requested: usize,
    games_completed: usize,
    games_dropped: usize,
    snapshots: usize,
    rows: usize,
    snapshot_every: usize,
    behaviour: &'static str,
    behaviour_budget_ms: u64,
    target_classes: [&'static str; 4],
    legality_contract: &'static str,
    supported_game_contract: &'static str,
}

struct BeliefRow {
    game_id: String,
    snapshot_id: usize,
    actor: PlayerID,
    card_id: usize,
    target: usize,
    mask: [u8; 4],
    features: [f32; FEATURE_DIM],
}

fn main() {
    let games = env_usize("BELIEF_GAMES", 100);
    let seed = env_u64("BELIEF_SEED", 0xBE11_EF);
    let every = env_usize("BELIEF_SNAPSHOT_EVERY", 4).max(1);
    let budget = env_u64("BELIEF_BEHAVIOUR_BUDGET_MS", 20).max(1);
    let out =
        std::env::var("BELIEF_OUT").unwrap_or_else(|_| "training/belief_data.csv".to_string());
    let manifest_path =
        std::env::var("BELIEF_MANIFEST").unwrap_or_else(|_| format!("{out}.manifest.json"));
    ensure_parent(&out);
    ensure_parent(&manifest_path);
    let mut writer = BufWriter::new(File::create(&out).expect("create belief CSV"));
    write_header(&mut writer);
    let mut completed = 0;
    let mut snapshots = 0;
    let mut rows = 0;
    for game_index in 0..games {
        let game_seed = derive_game_seed(seed, game_index as u64);
        if let Some(game_rows) = play_game(game_index, game_seed, every, budget) {
            completed += 1;
            snapshots += game_rows
                .last()
                .map(|row| row.snapshot_id + 1)
                .unwrap_or_default();
            rows += game_rows.len();
            for row in game_rows {
                write_row(&mut writer, row);
            }
        }
    }
    writer.flush().expect("flush belief CSV");
    let manifest = Manifest {
        manifest_version: 1,
        dataset_schema_version: SCHEMA_VERSION,
        feature_dim: FEATURE_DIM,
        seed,
        games_requested: games,
        games_completed: completed,
        games_dropped: games.saturating_sub(completed),
        snapshots,
        rows,
        snapshot_every: every,
        behaviour: "easy-play/expert-bid",
        behaviour_budget_ms: budget,
        target_classes: ["next-seat", "opposite-seat", "previous-seat", "kitty"],
        legality_contract: "mask=1 iff destination has capacity and no public effective-suit void",
        supported_game_contract: "tractor:4p:2x-standard:kitty8:no-removed",
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

fn play_game(game_index: usize, seed: u64, every: usize, budget: u64) -> Option<Vec<BeliefRow>> {
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
                        if decision % every == 0 {
                            rows.extend(snapshot_rows(play, actor, game_index, seed, snapshot)?);
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
        for (card, count) in full.hands().get(*seat).ok()? {
            if *card != Card::Unknown {
                destinations.extend(std::iter::repeat_n((class, *card), *count));
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
        let (features, encoded_mask) =
            belief::encode_belief_features(honest, actor, &knowledge, card);
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

fn write_header(writer: &mut impl Write) {
    write!(
        writer,
        "schema_version,game_id,snapshot_id,actor,card_id,target"
    )
    .unwrap();
    for index in 0..4 {
        write!(writer, ",mask{index}").unwrap();
    }
    for index in 0..FEATURE_DIM {
        write!(writer, ",b{index}").unwrap();
    }
    writeln!(writer).unwrap();
}

fn write_row(writer: &mut impl Write, row: BeliefRow) {
    write!(
        writer,
        "{},{},{},{},{},{}",
        SCHEMA_VERSION, row.game_id, row.snapshot_id, row.actor.0, row.card_id, row.target
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
