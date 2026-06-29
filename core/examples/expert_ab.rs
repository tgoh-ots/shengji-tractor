//! Expert-net A/B harness: measures whether the embedded `expert_model.onnx`
//! makes the EXPERT tier stronger, by running the Expert tier (search) against two
//! fixed opponents over the SAME seeds:
//!   * Expert(search) vs Enoch(search)
//!   * Expert(search) vs Easy
//!
//! Run this ONCE with the new net embedded and ONCE with the old net embedded
//! (swap `core/src/bot/expert_model.onnx` and rebuild) using identical args, and
//! compare Expert's win-rate / point margin. Everything else (the seeded deal loop,
//! the tier driver, the honesty boundary) is shared with `tournament` /
//! `enoch_benchmark`.
//!
//! Run with:
//!   cargo run --release --example expert_ab -- [games_per_match] [base_seed]

use std::env;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use shengji_core::bot::{policy, BotDifficulty};
use shengji_core::game_state::draw_phase::DrawPhase;
use shengji_core::game_state::initialize_phase::InitializePhase;
use shengji_core::game_state::GameState;
use shengji_core::interactive::Action;
use shengji_core::settings::GameModeSettings;

use shengji_mechanics::deck::Deck;
use shengji_mechanics::types::{Card, PlayerID};

struct GameOutcome {
    a_won: bool,
    a_point_margin: isize,
}

fn seeded_draw_phase(decks: &[Deck], rng: &mut StdRng) -> DrawPhase {
    let mut deck: Vec<_> = decks.iter().flat_map(|d| d.cards()).collect();
    deck.shuffle(rng);

    let num_players = 4;
    let mut kitty_size = deck.len() % num_players;
    if kitty_size == 0 {
        kitty_size = num_players;
    }
    if kitty_size < 5 {
        kitty_size += num_players;
    }

    let mut init = InitializePhase::new();
    for i in 0..num_players {
        init.add_player(format!("seat{i}")).unwrap();
    }
    init.set_num_decks(Some(decks.len())).unwrap();
    init.set_game_mode(GameModeSettings::Tractor).unwrap();
    let real_seats: Vec<PlayerID> = init.players().iter().map(|p| p.id).collect();
    init.set_landlord(Some(real_seats[0])).unwrap();
    let propagated = (*init).clone();

    let level = Some(propagated.players()[0].rank());
    let hands_deck = deck[0..deck.len() - kitty_size].to_vec();
    let kitty = deck[deck.len() - kitty_size..].to_vec();

    DrawPhase::new(
        propagated,
        0,
        hands_deck,
        kitty,
        decks.len(),
        shengji_core::settings::GameMode::Tractor,
        level,
        decks.to_vec(),
        vec![],
    )
}

fn play_cards_for(
    s: &shengji_core::game_state::play_phase::PlayPhase,
    actor: PlayerID,
    d: BotDifficulty,
) -> Option<Vec<Card>> {
    let view = if matches!(d, BotDifficulty::Omniscient) {
        GameState::Play(s.clone())
    } else {
        GameState::Play(s.clone()).for_player(actor)
    };
    match policy::select_action(&view, actor, d).ok()? {
        Some(Action::PlayCards(c)) => Some(c),
        _ => None,
    }
}

fn play_one_hand(
    a_is_landlord_team: bool,
    a: BotDifficulty,
    b: BotDifficulty,
    rng: &mut StdRng,
) -> Option<GameOutcome> {
    let decks = vec![Deck::default(), Deck::default()];
    let draw = seeded_draw_phase(&decks, rng);
    let seats: Vec<PlayerID> = draw.propagated().players().iter().map(|p| p.id).collect();

    let tier_of = |seat_idx: usize| -> BotDifficulty {
        let is_landlord_team = seat_idx % 2 == 0;
        if is_landlord_team == a_is_landlord_team {
            a
        } else {
            b
        }
    };

    let mut state = GameState::Draw(draw);
    let mut iters = 0usize;
    loop {
        iters += 1;
        if iters > 2_000_000 {
            return None;
        }
        match &mut state {
            GameState::Initialize(_) => return None,
            GameState::Draw(s) => {
                if !s.done_drawing() {
                    let p = s.next_player().ok()?;
                    s.draw_card(p).ok()?;
                } else if s.bid_decided() {
                    let responsible = s.next_player().ok()?;
                    state = GameState::Exchange(s.advance(responsible).ok()?);
                } else {
                    let mut bid = false;
                    for (idx, &seat) in seats.iter().enumerate() {
                        let d = tier_of(idx);
                        if let Some(b) = policy::choose_bid(s, seat, d) {
                            if s.bid(seat, b.card, b.count) {
                                bid = true;
                                break;
                            }
                        }
                    }
                    if !bid && s.reveal_card().is_err() {
                        for &seat in &seats {
                            if let Some(b) =
                                s.valid_bids(seat).ok()?.into_iter().min_by_key(|b| b.count)
                            {
                                if s.bid(seat, b.card, b.count) {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            GameState::Exchange(s) => {
                let landlord = s.landlord();
                let landlord_idx = seats.iter().position(|x| *x == landlord)?;
                let d = tier_of(landlord_idx);
                let view = GameState::Exchange(s.clone()).for_player(landlord);
                match policy::select_action(&view, landlord, d).ok()? {
                    Some(Action::MoveCardToKitty(c)) => s.move_card_to_kitty(landlord, c).ok()?,
                    Some(Action::MoveCardToHand(c)) => s.move_card_to_hand(landlord, c).ok()?,
                    Some(Action::SetFriends(f)) => s.set_friends(landlord, f).ok()?,
                    _ => state = GameState::Play(s.advance(landlord).ok()?),
                }
            }
            GameState::Play(s) => {
                if s.game_finished() {
                    let (non_landlord_points, _) = s.calculate_points();
                    let (_init, landlord_won, _msgs) = s.finish_game().ok()?;

                    let a_is_defender = a_is_landlord_team;
                    let a_won = landlord_won == a_is_defender;
                    let a_point_margin = if a_is_defender {
                        -non_landlord_points
                    } else {
                        non_landlord_points
                    };
                    return Some(GameOutcome {
                        a_won,
                        a_point_margin,
                    });
                }
                match s.trick().next_player() {
                    None => {
                        s.finish_trick().ok()?;
                    }
                    Some(actor) => {
                        let actor_idx = seats.iter().position(|x| *x == actor)?;
                        let cards = play_cards_for(s, actor, tier_of(actor_idx))?;
                        s.play_cards(actor, &cards).ok()?;
                    }
                }
            }
        }
    }
}

fn run_match(a: BotDifficulty, b: BotDifficulty, num_games: usize, base_seed: u64) {
    let start = Instant::now();
    let mut a_wins = 0usize;
    let mut total_margin: isize = 0;
    let mut completed = 0usize;

    for g in 0..num_games {
        let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(g as u64));
        let a_is_landlord_team = g % 2 == 0;
        if let Some(outcome) = play_one_hand(a_is_landlord_team, a, b, &mut rng) {
            completed += 1;
            if outcome.a_won {
                a_wins += 1;
            }
            total_margin += outcome.a_point_margin;
        }
    }

    let wr = if completed > 0 {
        a_wins as f64 / completed as f64 * 100.0
    } else {
        0.0
    };
    println!(
        "=== {} vs {} ({} games) ===",
        a.as_str(),
        b.as_str(),
        completed
    );
    println!(
        "  {} win-rate: {:.2}%  ({} / {})   avg margin {:+.2} pts/game   [{:.1}s]",
        a.as_str(),
        wr,
        a_wins,
        completed,
        total_margin as f64 / completed.max(1) as f64,
        start.elapsed().as_secs_f64()
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let num_games: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(120);
    let base_seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0xAB_EE);
    // Optional 3rd arg selects which match to run ("enoch" | "easy" | "both"),
    // so each match can be run in its own (time-capped) process.
    let which = args.get(3).map(|s| s.as_str()).unwrap_or("both");

    if env::var("SHENGJI_BOT_BUDGET_MS").is_err() {
        env::set_var("SHENGJI_BOT_BUDGET_MS", "80");
    }
    let budget = env::var("SHENGJI_BOT_BUDGET_MS").unwrap_or_default();

    println!("EXPERT NET A/B");
    println!(
        "Games per match: {num_games}  base_seed: {base_seed:#x}  budget_ms: {budget}  match: {which}\n"
    );

    if which == "enoch" || which == "both" {
        run_match(
            BotDifficulty::Expert,
            BotDifficulty::Enoch,
            num_games,
            base_seed,
        );
    }
    if which == "easy" || which == "both" {
        run_match(
            BotDifficulty::Expert,
            BotDifficulty::Easy,
            num_games,
            base_seed,
        );
    }
}
