//! Headless A/B benchmark: NEW heuristic-direct play vs LEGACY heuristic-direct
//! play, over many full 4-player Tractor hands with SEEDED, reproducible deals.
//!
//! One partnership (2 seats) plays every PLAY-phase decision with the NEW
//! boss-/partner-aware scorer (`HeuristicVersion::New`), the other partnership
//! with the frozen LEGACY scorer (`HeuristicVersion::Legacy`). Both partnerships
//! share the *same* deterministic bidding (Expert) + kitty (Easy) policy, so ONLY
//! the play-phase scorer differs. We alternate which partnership is NEW across
//! games to cancel the landlord/dealer positional advantage.
//!
//! The deal, the per-hand driver, and the honesty boundary are now shared with
//! every other benchmark via `shengji_core::bot::harness`.
//!
//! Run with:
//!   cargo run --release --example heuristic_benchmark -- [num_games] [base_seed]

use std::env;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;

use shengji_core::bot::harness::{play_one_hand, PlayBrain, Seat};
use shengji_core::bot::heuristics::HeuristicVersion;
use shengji_core::bot::BotDifficulty;

use shengji_mechanics::deck::Deck;
use shengji_mechanics::scoring::{compute_level_deltas, GameScoringParameters};

/// The outcome of one finished hand, from the perspective of the NEW partnership.
struct GameOutcome {
    new_won: bool,
    new_point_margin: isize,
    new_net_levels: isize,
}

/// Seats 0 & 2 are the landlord (defending) team; seats 1 & 3 attack. Seat 0 is
/// always the landlord. `new_is_landlord_team` chooses which partnership uses the
/// NEW heuristic this hand.
fn version_for_seat(seat_idx: usize, new_is_landlord_team: bool) -> HeuristicVersion {
    let is_landlord_team = seat_idx.is_multiple_of(2);
    if is_landlord_team == new_is_landlord_team {
        HeuristicVersion::New
    } else {
        HeuristicVersion::Legacy
    }
}

/// Drive a single seeded hand. Both partnerships bid Expert + bury Easy (so only
/// the play scorer differs); each seat plays greedy heuristic-direct at its
/// assigned version.
fn play_one_hand_ab(new_is_landlord_team: bool, rng: &mut StdRng) -> Option<GameOutcome> {
    let seat_of = |idx: usize| Seat {
        play: PlayBrain::HeuristicDirect(version_for_seat(idx, new_is_landlord_team)),
        bid: BotDifficulty::Expert,
        kitty: BotDifficulty::Easy,
    };
    let seats = [seat_of(0), seat_of(1), seat_of(2), seat_of(3)];
    let r = play_one_hand(&seats, rng)?;

    let (new_won, new_point_margin) = r.subject_outcome(new_is_landlord_team);

    // Net levels: recompute the level deltas the same way the engine does in
    // finish_game (no FindingFriends here, so smaller_landlord_team = false).
    let decks = vec![Deck::default(), Deck::default()];
    let gsp = GameScoringParameters::default();
    let new_net_levels = match compute_level_deltas(&gsp, &decks, r.non_landlord_points, false).ok()
    {
        Some(res) => {
            let landlord_levels = res.landlord_delta as isize;
            let non_landlord_levels = res.non_landlord_delta as isize;
            if new_is_landlord_team {
                landlord_levels - non_landlord_levels
            } else {
                non_landlord_levels - landlord_levels
            }
        }
        None => 0,
    };

    Some(GameOutcome {
        new_won,
        new_point_margin,
        new_net_levels,
    })
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let num_games: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(400);
    let base_seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0xC0FFEE);

    println!("Heuristic A/B benchmark: NEW heuristic-direct vs LEGACY heuristic-direct");
    println!("Games: {num_games}  base_seed: {base_seed:#x}");
    println!(
        "Policy: greedy argmax, NO search. Shared deterministic bidding/kitty; \
         only PLAY-phase scorer differs. Seats alternate NEW partnership across games."
    );
    println!();

    let start = Instant::now();

    let mut new_wins = 0usize;
    let mut legacy_wins = 0usize;
    let mut total_margin: isize = 0;
    let mut total_net_levels: isize = 0;
    let mut completed = 0usize;
    let mut new_as_landlord_games = 0usize;
    let mut new_as_landlord_wins = 0usize;
    let mut new_as_attacker_games = 0usize;
    let mut new_as_attacker_wins = 0usize;

    for g in 0..num_games {
        // Fixed seed sequence: deterministic, no wall-clock seeding.
        let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(g as u64));
        // Alternate which partnership is NEW to cancel the landlord advantage.
        let new_is_landlord_team = g % 2 == 0;

        match play_one_hand_ab(new_is_landlord_team, &mut rng) {
            Some(outcome) => {
                completed += 1;
                if outcome.new_won {
                    new_wins += 1;
                } else {
                    legacy_wins += 1;
                }
                total_margin += outcome.new_point_margin;
                total_net_levels += outcome.new_net_levels;

                if new_is_landlord_team {
                    new_as_landlord_games += 1;
                    if outcome.new_won {
                        new_as_landlord_wins += 1;
                    }
                } else {
                    new_as_attacker_games += 1;
                    if outcome.new_won {
                        new_as_attacker_wins += 1;
                    }
                }
            }
            None => {
                eprintln!("game {g}: engine error / could not complete (skipped)");
            }
        }
    }

    let elapsed = start.elapsed();

    println!("=== Results over {completed} completed games ===");
    let win_rate = if completed > 0 {
        new_wins as f64 / completed as f64 * 100.0
    } else {
        0.0
    };
    println!("NEW wins:    {new_wins}");
    println!("LEGACY wins: {legacy_wins}");
    println!("NEW win-rate:               {win_rate:.2}%");
    println!(
        "NEW avg final point margin: {:+.2} points/game",
        total_margin as f64 / completed.max(1) as f64
    );
    println!(
        "NEW avg net levels gained:  {:+.3} levels/game",
        total_net_levels as f64 / completed.max(1) as f64
    );
    println!();
    println!("Positional split (sanity check that the edge isn't just the dealer):");
    if new_as_landlord_games > 0 {
        println!(
            "  NEW as LANDLORD team: {}/{} = {:.1}%",
            new_as_landlord_wins,
            new_as_landlord_games,
            new_as_landlord_wins as f64 / new_as_landlord_games as f64 * 100.0
        );
    }
    if new_as_attacker_games > 0 {
        println!(
            "  NEW as ATTACKER team: {}/{} = {:.1}%",
            new_as_attacker_wins,
            new_as_attacker_games,
            new_as_attacker_wins as f64 / new_as_attacker_games as f64 * 100.0
        );
    }
    println!();
    println!("Elapsed: {:.1}s", elapsed.as_secs_f64());
    let verdict = if win_rate >= 60.0 {
        "NEW heuristic is CLEARLY stronger (>=60% win-rate)."
    } else if win_rate > 53.0 {
        "NEW heuristic is stronger (win-rate meaningfully above 50%)."
    } else if win_rate >= 47.0 {
        "NO clear difference (win-rate ~50%)."
    } else {
        "NEW heuristic appears WEAKER (win-rate below 50%)."
    };
    println!("Verdict: {verdict}");
}
