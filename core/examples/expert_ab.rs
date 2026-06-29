//! Expert-net A/B harness: measures whether the embedded `expert_model.onnx`
//! (or a runtime-overridden net via `SHENGJI_EXPERT_MODEL_PATH`) makes the EXPERT
//! tier stronger, by running the Expert tier (search) against two fixed opponents
//! over the SAME seeds:
//!   * Expert(search) vs Enoch(search)
//!   * Expert(search) vs Easy
//!
//! With the runtime model-path override you can now A/B two nets WITHOUT a rebuild:
//!   SHENGJI_EXPERT_MODEL_PATH=/path/to/candidate.onnx cargo run --release \
//!       --example expert_ab
//! vs an unset run (embedded net). Everything else (the seeded deal loop, the tier
//! driver, the honesty boundary) is shared via `shengji_core::bot::harness`.
//!
//! Run with:
//!   cargo run --release --example expert_ab -- [games_per_match] [base_seed] [which]

use std::env;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;

use shengji_core::bot::harness::{play_one_hand, Seat};
use shengji_core::bot::BotDifficulty;

struct GameOutcome {
    a_won: bool,
    a_point_margin: isize,
}

fn play_one_hand_ab(
    a_is_landlord_team: bool,
    a: BotDifficulty,
    b: BotDifficulty,
    rng: &mut StdRng,
) -> Option<GameOutcome> {
    let tier_of = |idx: usize| -> BotDifficulty {
        let is_landlord_team = idx % 2 == 0;
        if is_landlord_team == a_is_landlord_team {
            a
        } else {
            b
        }
    };
    let seats = [
        Seat::tier(tier_of(0)),
        Seat::tier(tier_of(1)),
        Seat::tier(tier_of(2)),
        Seat::tier(tier_of(3)),
    ];
    let r = play_one_hand(&seats, rng)?;
    let (a_won, a_point_margin) = r.subject_outcome(a_is_landlord_team);
    Some(GameOutcome {
        a_won,
        a_point_margin,
    })
}

fn run_match(a: BotDifficulty, b: BotDifficulty, num_games: usize, base_seed: u64) {
    let start = Instant::now();
    let mut a_wins = 0usize;
    let mut total_margin: isize = 0;
    let mut completed = 0usize;

    for g in 0..num_games {
        let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(g as u64));
        let a_is_landlord_team = g % 2 == 0;
        if let Some(outcome) = play_one_hand_ab(a_is_landlord_team, a, b, &mut rng) {
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
    let which = args.get(3).map(|s| s.as_str()).unwrap_or("both");

    if env::var("SHENGJI_BOT_BUDGET_MS").is_err() {
        env::set_var("SHENGJI_BOT_BUDGET_MS", "80");
    }
    let budget = env::var("SHENGJI_BOT_BUDGET_MS").unwrap_or_default();
    let net = env::var("SHENGJI_EXPERT_MODEL_PATH").unwrap_or_else(|_| "<embedded>".to_string());

    println!("EXPERT NET A/B");
    println!(
        "Games per match: {num_games}  base_seed: {base_seed:#x}  budget_ms: {budget}  match: {which}  net: {net}\n"
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
