//! Headless A/B benchmark for the EASY tier's tuning knobs.
//!
//! This measures the effect of the Easy-tier knob change (softmax temperature
//! and blunder rate) by pitting Easy@NEW against Easy@OLD over many seeded
//! 4-player Tractor hands. Both sides play the SAME (greedy heuristic-direct)
//! candidate scorer with NO search; the ONLY thing that differs is the
//! `(epsilon, temperature)` pair applied on top of the shared heuristic ranking:
//!
//!   * Easy@NEW — ε = 0.06, T = 1.1 (the strengthened, less-noisy Easy)
//!   * Easy@OLD — ε = 0.28, T = 3.5 (the original, very noisy Easy)
//!
//! The deal, the per-hand driver, the per-decision Easy policy, and the honesty
//! boundary are all shared with the other benchmarks via
//! `shengji_core::bot::harness` (`PlayBrain::Easy`). We alternate which
//! partnership is NEW across games to cancel the landlord/dealer positional edge.
//!
//! Run with:
//!   cargo run --release --example easy_ab_benchmark -- [num_games] [base_seed]
//!
//! We expect a MODEST bump for Easy@NEW (~55-60%), NOT a blowout: Easy must stay
//! the weakest, clearly-beatable casual tier.

use std::env;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;

use shengji_core::bot::harness::{play_one_hand, PlayBrain, Seat};
use shengji_core::bot::BotDifficulty;

/// One Easy knob configuration (the only thing that differs between the A/B
/// sides). Mirrors the `epsilon` / `temperature` fields of `policy::Knobs`.
#[derive(Clone, Copy)]
struct EasyKnobs {
    label: &'static str,
    epsilon: f64,
    temperature: f64,
}

/// The NEW (strengthened) Easy: fewer blunders, a cooler softmax. Keep these in
/// sync with `BotDifficulty::Easy` in `core/src/bot/policy.rs`. The defaults can
/// be overridden via `EASY_NEW_EPS` / `EASY_NEW_TEMP` for quick tuning sweeps.
fn new_easy() -> EasyKnobs {
    EasyKnobs {
        label: "Easy@NEW",
        epsilon: env::var("EASY_NEW_EPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.06),
        temperature: env::var("EASY_NEW_TEMP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.1),
    }
}

/// The OLD (pre-change) Easy: very noisy — frequent blunders, hot softmax.
const OLD_EASY: EasyKnobs = EasyKnobs {
    label: "Easy@OLD",
    epsilon: 0.28,
    temperature: 3.5,
};

impl EasyKnobs {
    fn seat(self) -> Seat {
        Seat {
            play: PlayBrain::Easy {
                epsilon: self.epsilon,
                temperature: self.temperature,
            },
            // Bidding + kitty are knob-independent (shared Easy driver), so both
            // sides use the same Easy bid/bury policy and only the play knobs differ.
            bid: BotDifficulty::Easy,
            kitty: BotDifficulty::Easy,
        }
    }
}

/// The outcome of one finished hand, from the NEW partnership's perspective.
struct GameOutcome {
    new_won: bool,
    new_point_margin: isize,
}

/// Drive one seeded hand. `new_is_landlord_team` selects which partnership plays
/// with the NEW Easy knobs; the other uses the OLD knobs.
fn play_one_hand_ab(
    new_is_landlord_team: bool,
    new_knobs: EasyKnobs,
    old_knobs: EasyKnobs,
    rng: &mut StdRng,
) -> Option<GameOutcome> {
    // Seats 0,2 are the landlord (defending) team. The NEW knobs occupy the
    // landlord team iff `new_is_landlord_team`.
    let seat_of = |idx: usize| -> Seat {
        let is_landlord_team = idx.is_multiple_of(2);
        if is_landlord_team == new_is_landlord_team {
            new_knobs.seat()
        } else {
            old_knobs.seat()
        }
    };
    let seats = [seat_of(0), seat_of(1), seat_of(2), seat_of(3)];
    let r = play_one_hand(&seats, rng)?;
    let (new_won, new_point_margin) = r.subject_outcome(new_is_landlord_team);
    Some(GameOutcome {
        new_won,
        new_point_margin,
    })
}

fn run_match(num_games: usize, base_seed: u64) {
    let start = Instant::now();
    let new_easy = new_easy();
    let mut new_wins = 0usize;
    let mut old_wins = 0usize;
    let mut total_margin: isize = 0;
    let mut completed = 0usize;

    for g in 0..num_games {
        let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(g as u64));
        let new_is_landlord_team = g % 2 == 0;
        match play_one_hand_ab(new_is_landlord_team, new_easy, OLD_EASY, &mut rng) {
            Some(outcome) => {
                completed += 1;
                if outcome.new_won {
                    new_wins += 1;
                } else {
                    old_wins += 1;
                }
                total_margin += outcome.new_point_margin;
            }
            None => eprintln!("  game {g}: engine error / skipped"),
        }
    }

    let win_rate = if completed > 0 {
        new_wins as f64 / completed as f64 * 100.0
    } else {
        0.0
    };

    println!();
    println!(
        "=== {} (ε={}, T={}) vs {} (ε={}, T={}) ===",
        new_easy.label,
        new_easy.epsilon,
        new_easy.temperature,
        OLD_EASY.label,
        OLD_EASY.epsilon,
        OLD_EASY.temperature,
    );
    println!("  Games completed:      {completed}");
    println!("  Easy@NEW wins:        {new_wins}");
    println!("  Easy@OLD wins:        {old_wins}");
    println!("  Easy@NEW win-rate:    {win_rate:.2}%");
    println!(
        "  Easy@NEW avg margin:  {:+.2} pts/game",
        total_margin as f64 / completed.max(1) as f64
    );
    println!("  Elapsed: {:.1}s", start.elapsed().as_secs_f64());

    if completed > 0 {
        let n = completed as f64;
        let p = new_wins as f64 / n;
        let se = (0.25 / n).sqrt(); // SE under the null p=0.5
        let z = (p - 0.5) / se;
        println!("  (z vs 50% null = {z:+.2}; |z|>1.96 ≈ p<0.05 two-sided)");
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let num_games: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(400);
    let base_seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0xEA59);

    println!("EASY A/B benchmark: Easy@NEW vs Easy@OLD (knob change only)");
    println!("Games: {num_games}  base_seed: {base_seed:#x}");
    println!(
        "Both sides share the heuristic candidate scorer and the bidding/kitty \
         driver; only the blunder rate (ε) and softmax temperature (T) differ. \
         Sides alternate landlord/attacker each game.\n"
    );

    run_match(num_games, base_seed);
}
