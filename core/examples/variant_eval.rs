//! Seeded bot-completion smoke/evaluation across representative rule variants.
//!
//! This is primarily a coverage gate: failed hands are reported explicitly and
//! make the process fail.  It also reports win rate, points, and signed level
//! utility so variant support is not inferred from the standard 4p/2-deck game.

use rand::rngs::StdRng;
use rand::SeedableRng;

use shengji_core::bot::harness::{play_one_hand_with_config, HarnessConfig, Seat};
use shengji_core::bot::BotDifficulty;
use shengji_core::settings::GameModeSettings;
use shengji_mechanics::deck::Deck;
use shengji_mechanics::types::{Number, Rank};

fn run_variant(label: &str, config: HarnessConfig, hands: usize, seed: u64) -> usize {
    let seats = vec![Seat::tier(BotDifficulty::Easy); config.num_players];
    let mut completed = 0usize;
    let mut landlord_wins = 0usize;
    let mut points = 0isize;
    let mut landlord_level_utility = 0isize;
    for hand in 0..hands {
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(hand as u64));
        if let Some(result) = play_one_hand_with_config(&seats, &config, &mut rng) {
            completed += 1;
            landlord_wins += usize::from(result.landlord_won);
            points += result.non_landlord_points;
            landlord_level_utility += result.subject_level_utility(true);
        }
    }
    let denom = completed.max(1) as f64;
    println!(
        "{label:24} completed {completed:4}/{hands:<4} landlord-win {:6.1}%  attacker-points {:7.2}  landlord-level {:+.3}",
        landlord_wins as f64 / denom * 100.0,
        points as f64 / denom,
        landlord_level_utility as f64 / denom,
    );
    hands - completed
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let hands = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(20);
    let seed = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(0xA11CE);

    let mut failures = 0usize;
    failures += run_variant("4p / 2-deck Tractor", HarnessConfig::default(), hands, seed);
    failures += run_variant(
        "6p / 3-deck Tractor",
        HarnessConfig {
            num_players: 6,
            decks: vec![Deck::default(), Deck::default(), Deck::default()],
            game_mode: GameModeSettings::Tractor,
            rank: Rank::Number(Number::Seven),
        },
        hands,
        seed,
    );
    failures += run_variant(
        "5p Finding Friends",
        HarnessConfig {
            num_players: 5,
            decks: vec![Deck::default(), Deck::default(), Deck::default()],
            game_mode: GameModeSettings::FindingFriends { num_friends: None },
            rank: Rank::Number(Number::Ace),
        },
        hands,
        seed,
    );
    failures += run_variant(
        "4p short special decks",
        HarnessConfig {
            num_players: 4,
            decks: vec![
                Deck {
                    exclude_small_joker: true,
                    exclude_big_joker: false,
                    min: Number::Five,
                },
                Deck {
                    exclude_small_joker: false,
                    exclude_big_joker: true,
                    min: Number::Five,
                },
            ],
            game_mode: GameModeSettings::Tractor,
            rank: Rank::Number(Number::Ten),
        },
        hands,
        seed,
    );

    if failures > 0 {
        eprintln!("variant evaluation had {failures} failed hands");
        std::process::exit(1);
    }
}
