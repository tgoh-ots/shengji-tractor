//! Headless benchmark for the ENOCH bot tier.
//!
//! One partnership (2 seats) plays the whole hand as ENOCH — its pair-prioritized
//! trump declaration, point-budgeted kitty burial, and the Enoch-playbook play
//! phase. The other partnership plays as a chosen OPPONENT brain. We alternate
//! which partnership is Enoch across games to cancel the dealer/landlord
//! positional edge, and report Enoch's win-rate (and average point margin).
//!
//! The opponents:
//!   * NewDefault — the new default boss-/partner-aware heuristic, GREEDY (no
//!     search). This is the play scorer every honest tier shares; beating it shows
//!     the Enoch playbook adds value on top of the improved heuristic.
//!   * Expert     — the learned-net prior PLUS the time-boxed determinized search.
//!   * Omniscient — the perfect-information CHEATER (an upper bound).
//!
//! The deal, the per-hand driver, the greedy / search / honesty paths are all
//! shared with the other benchmarks via `shengji_core::bot::harness`. Set
//! `SHENGJI_BOT_BUDGET_MS` to trade strength for speed in the search tiers.
//!
//! Run with:
//!   cargo run --release --example enoch_benchmark -- [num_games] [base_seed]

use std::env;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;

use shengji_core::bot::harness::{play_one_hand, PlayBrain, Seat};
use shengji_core::bot::heuristics::HeuristicVersion;
use shengji_core::bot::BotDifficulty;

/// How a partnership plays a hand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Brain {
    /// Full Enoch playbook, GREEDY play (no search).
    EnochGreedy,
    /// The new default boss-/partner-aware heuristic, GREEDY play (no search).
    NewDefault,
    /// A real difficulty tier driven through `policy::select_action` (search).
    Tier(BotDifficulty),
}

impl Brain {
    fn label(self) -> String {
        match self {
            Brain::EnochGreedy => "Enoch(greedy)".to_string(),
            Brain::NewDefault => "NewDefault(greedy)".to_string(),
            Brain::Tier(d) => format!("{d:?}(search)"),
        }
    }

    /// The [`Seat`] config this brain plays as: its play policy plus the
    /// difficulty-aware bidding + kitty burial it drives the declare phase with.
    fn seat(self) -> Seat {
        match self {
            Brain::EnochGreedy => Seat {
                play: PlayBrain::EnochGreedy,
                bid: BotDifficulty::Enoch,
                kitty: BotDifficulty::Enoch,
            },
            Brain::NewDefault => Seat {
                play: PlayBrain::HeuristicDirect(HeuristicVersion::New),
                bid: BotDifficulty::Expert,
                kitty: BotDifficulty::Expert,
            },
            Brain::Tier(d) => Seat::tier(d),
        }
    }
}

/// The outcome of one finished hand, from Enoch's perspective.
struct GameOutcome {
    enoch_won: bool,
    enoch_point_margin: isize,
}

/// Drive one seeded hand. `enoch_is_landlord_team` selects which partnership is
/// the `enoch` brain; the other plays `opponent`.
fn play_one_hand_ab(
    enoch_is_landlord_team: bool,
    enoch: Brain,
    opponent: Brain,
    rng: &mut StdRng,
) -> Option<GameOutcome> {
    let brain_of = |idx: usize| -> Brain {
        let is_landlord_team = idx.is_multiple_of(2);
        if is_landlord_team == enoch_is_landlord_team {
            enoch
        } else {
            opponent
        }
    };
    let seats = [
        brain_of(0).seat(),
        brain_of(1).seat(),
        brain_of(2).seat(),
        brain_of(3).seat(),
    ];
    let r = play_one_hand(&seats, rng)?;
    let (enoch_won, enoch_point_margin) = r.subject_outcome(enoch_is_landlord_team);
    Some(GameOutcome {
        enoch_won,
        enoch_point_margin,
    })
}

/// Run a full `enoch`-vs-`opponent` match and print the result.
fn run_match(enoch: Brain, opponent: Brain, num_games: usize, base_seed: u64) {
    let start = Instant::now();
    let mut enoch_wins = 0usize;
    let mut opp_wins = 0usize;
    let mut total_margin: isize = 0;
    let mut completed = 0usize;
    let mut enoch_as_landlord_games = 0usize;
    let mut enoch_as_landlord_wins = 0usize;
    let mut enoch_as_attacker_games = 0usize;
    let mut enoch_as_attacker_wins = 0usize;

    for g in 0..num_games {
        let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(g as u64));
        let enoch_is_landlord_team = g % 2 == 0;
        match play_one_hand_ab(enoch_is_landlord_team, enoch, opponent, &mut rng) {
            Some(outcome) => {
                completed += 1;
                if outcome.enoch_won {
                    enoch_wins += 1;
                } else {
                    opp_wins += 1;
                }
                total_margin += outcome.enoch_point_margin;
                if enoch_is_landlord_team {
                    enoch_as_landlord_games += 1;
                    if outcome.enoch_won {
                        enoch_as_landlord_wins += 1;
                    }
                } else {
                    enoch_as_attacker_games += 1;
                    if outcome.enoch_won {
                        enoch_as_attacker_wins += 1;
                    }
                }
            }
            None => eprintln!("  game {g}: engine error / skipped"),
        }
    }

    let win_rate = if completed > 0 {
        enoch_wins as f64 / completed as f64 * 100.0
    } else {
        0.0
    };
    println!(
        "=== {} vs {} ({completed} games) ===",
        enoch.label(),
        opponent.label()
    );
    println!("  Enoch wins:    {enoch_wins}");
    println!("  Opponent wins: {opp_wins}");
    println!("  Enoch win-rate:           {win_rate:.2}%");
    println!(
        "  Enoch avg point margin:   {:+.2} pts/game",
        total_margin as f64 / completed.max(1) as f64
    );
    if enoch_as_landlord_games > 0 {
        println!(
            "    as LANDLORD: {}/{} = {:.1}%",
            enoch_as_landlord_wins,
            enoch_as_landlord_games,
            enoch_as_landlord_wins as f64 / enoch_as_landlord_games as f64 * 100.0
        );
    }
    if enoch_as_attacker_games > 0 {
        println!(
            "    as ATTACKER: {}/{} = {:.1}%",
            enoch_as_attacker_wins,
            enoch_as_attacker_games,
            enoch_as_attacker_wins as f64 / enoch_as_attacker_games as f64 * 100.0
        );
    }
    println!("  Elapsed: {:.1}s", start.elapsed().as_secs_f64());
    println!();
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let num_games: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let base_seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0xE0C);

    println!("ENOCH tier benchmark");
    println!("Games per match: {num_games}  base_seed: {base_seed:#x}");
    println!(
        "Match 1 is greedy-vs-greedy (apples-to-apples on the play scorer); matches \
         2-3 run the REAL Enoch tier (playbook + determinized search) against the \
         other search tiers. Declare/kitty/endgame rules apply deterministically.\n"
    );

    // 1) Enoch-greedy vs the new-default greedy heuristic (the shared play scorer).
    run_match(Brain::EnochGreedy, Brain::NewDefault, num_games, base_seed);
    // 2) The REAL Enoch tier (search) vs Expert (search).
    run_match(
        Brain::Tier(BotDifficulty::Enoch),
        Brain::Tier(BotDifficulty::Expert),
        num_games,
        base_seed,
    );
    // 3) The REAL Enoch tier (search, HONEST) vs the Omniscient cheater.
    run_match(
        Brain::Tier(BotDifficulty::Enoch),
        Brain::Tier(BotDifficulty::Omniscient),
        num_games,
        base_seed,
    );
}
