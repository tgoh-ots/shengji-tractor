//! Full round-robin tournament across the bot TIERS.
//!
//! Plays every unordered pairing of [Easy, Expert, Enoch, Omniscient] (6 pairings)
//! as a seeded head-to-head match. In each match one partnership (2 seats) plays as
//! tier A and the other as tier B; we ALTERNATE which partnership is the
//! landlord/defending team across games to cancel the dealer/positional bias, and
//! we reuse the same seeds across reversed pairings so the matrix is symmetric on
//! the deals. We report a clean WIN-RATE MATRIX (each tier's win-rate vs each other
//! tier) plus average point margins, and the implied ladder ordering.
//!
//! The deal, the per-hand driver, and the honesty boundary are shared with every
//! other benchmark via `shengji_core::bot::harness`. Every tier here is a real
//! difficulty tier driven through `policy::select_action`; the honesty boundary is
//! preserved (only Omniscient sees the unredacted state).
//!
//! Search budget is set via `SHENGJI_BOT_BUDGET_MS` (the example defaults it to
//! 100ms if unset, for speed). NOTE: at a time-bound budget the search tiers are
//! NOT byte-reproducible (the world cap doesn't bind), so win-rates jitter a few
//! points run-to-run.
//!
//! Run with:
//!   cargo run --release --example tournament -- [games_per_pairing] [base_seed]

use std::env;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;

use shengji_core::bot::harness::{play_one_hand, Seat};
use shengji_core::bot::BotDifficulty;

/// The four tiers, in nominal ladder order (weakest -> strongest).
const TIERS: [BotDifficulty; 4] = [
    BotDifficulty::Easy,
    BotDifficulty::Expert,
    BotDifficulty::Enoch,
    BotDifficulty::Omniscient,
];

fn tier_label(d: BotDifficulty) -> &'static str {
    d.as_str()
}

/// The outcome of one finished hand, from tier A's (the "subject") perspective.
struct GameOutcome {
    a_won: bool,
    a_point_margin: isize,
}

/// Drive one seeded hand: tier `a` vs tier `b`. `a_is_landlord_team` selects which
/// partnership plays tier `a`; the other plays tier `b`. The result is always from
/// tier `a`'s perspective.
fn play_one_hand_ab(
    a_is_landlord_team: bool,
    a: BotDifficulty,
    b: BotDifficulty,
    rng: &mut StdRng,
) -> Option<GameOutcome> {
    // Seats 0,2 are the landlord (defending) team; 1,3 attack. Tier `a` occupies
    // the landlord team iff `a_is_landlord_team`.
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

/// Aggregate result of one head-to-head pairing, from tier A's perspective.
struct PairResult {
    a: BotDifficulty,
    b: BotDifficulty,
    completed: usize,
    a_wins: usize,
    a_total_margin: isize,
}

/// Run a full `a`-vs-`b` match over `num_games` seeded hands, alternating which
/// partnership plays tier `a` to cancel positional bias.
fn run_pair(a: BotDifficulty, b: BotDifficulty, num_games: usize, base_seed: u64) -> PairResult {
    let mut a_wins = 0usize;
    let mut a_total_margin: isize = 0;
    let mut completed = 0usize;

    for g in 0..num_games {
        let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(g as u64));
        let a_is_landlord_team = g % 2 == 0;
        if let Some(outcome) = play_one_hand_ab(a_is_landlord_team, a, b, &mut rng) {
            completed += 1;
            if outcome.a_won {
                a_wins += 1;
            }
            a_total_margin += outcome.a_point_margin;
        }
    }

    PairResult {
        a,
        b,
        completed,
        a_wins,
        a_total_margin,
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let games_per_pairing: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(160);
    let base_seed: u64 = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0x70_75_72);
    // Optional 3rd arg: run ONLY a single pairing by its index (0..6) instead of
    // all six.
    let only_pair: Option<usize> = args.get(3).and_then(|s| s.parse().ok());

    // Default the search budget to 100ms for speed if the caller didn't set it.
    if env::var("SHENGJI_BOT_BUDGET_MS").is_err() {
        env::set_var("SHENGJI_BOT_BUDGET_MS", "100");
    }
    let budget = env::var("SHENGJI_BOT_BUDGET_MS").unwrap_or_default();

    println!("ROUND-ROBIN TIER TOURNAMENT");
    println!(
        "Tiers: {}",
        TIERS
            .iter()
            .map(|d| tier_label(*d))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "Games per pairing: {games_per_pairing}  base_seed: {base_seed:#x}  budget_ms: {budget}"
    );
    println!(
        "Each pairing alternates which partnership is the landlord/defending team \
         across games to cancel positional bias. Honesty preserved (only Omniscient \
         sees the unredacted state).\n"
    );

    let n = TIERS.len();
    let mut win_rate = vec![vec![f64::NAN; n]; n];
    let mut margin = vec![vec![f64::NAN; n]; n];
    let mut wins_for = vec![0usize; n];
    let mut games_for = vec![0usize; n];

    let overall_start = Instant::now();

    let pairings: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
        .collect();

    for (pidx, &(i, j)) in pairings.iter().enumerate() {
        if let Some(want) = only_pair {
            if pidx != want {
                continue;
            }
        }
        {
            let a = TIERS[i];
            let b = TIERS[j];
            let start = Instant::now();
            let r = run_pair(a, b, games_per_pairing, base_seed);
            let a_wr = if r.completed > 0 {
                r.a_wins as f64 / r.completed as f64 * 100.0
            } else {
                0.0
            };
            let a_marg = r.a_total_margin as f64 / r.completed.max(1) as f64;
            let b_wins = r.completed - r.a_wins;
            let b_wr = 100.0 - a_wr;
            let b_marg = -a_marg;

            win_rate[i][j] = a_wr;
            win_rate[j][i] = b_wr;
            margin[i][j] = a_marg;
            margin[j][i] = b_marg;

            wins_for[i] += r.a_wins;
            games_for[i] += r.completed;
            wins_for[j] += b_wins;
            games_for[j] += r.completed;

            println!(
                "=== {} vs {} ({} games) ===",
                tier_label(r.a),
                tier_label(r.b),
                r.completed
            );
            println!(
                "  {} win-rate: {:.2}%  ({} wins)   avg margin {:+.2} pts/game",
                tier_label(r.a),
                a_wr,
                r.a_wins,
                a_marg
            );
            println!(
                "  {} win-rate: {:.2}%  ({} wins)   avg margin {:+.2} pts/game",
                tier_label(r.b),
                b_wr,
                b_wins,
                b_marg
            );
            println!("  Elapsed: {:.1}s", start.elapsed().as_secs_f64());
            println!(
                "PAIR_RESULT {pidx} {i} {j} {} {} {}\n",
                r.completed, r.a_wins, r.a_total_margin
            );
        }
    }

    if only_pair.is_some() {
        println!("(single-pairing mode: matrix printed only in full run)");
        return;
    }

    // ---- WIN-RATE MATRIX ----
    println!("================ WIN-RATE MATRIX (row tier's win-% vs column tier) ================");
    print!("{:>12}", "");
    for d in TIERS.iter() {
        print!("{:>12}", tier_label(*d));
    }
    println!("{:>10}", "OVERALL");
    for i in 0..n {
        print!("{:>12}", tier_label(TIERS[i]));
        for j in 0..n {
            if i == j {
                print!("{:>12}", "—");
            } else {
                print!("{:>11.1}%", win_rate[i][j]);
            }
        }
        let overall = if games_for[i] > 0 {
            wins_for[i] as f64 / games_for[i] as f64 * 100.0
        } else {
            0.0
        };
        println!("{:>9.1}%", overall);
    }

    // ---- POINT-MARGIN MATRIX ----
    println!(
        "\n========= AVG POINT-MARGIN MATRIX (row tier's avg pts/game vs column tier) ========="
    );
    print!("{:>12}", "");
    for d in TIERS.iter() {
        print!("{:>12}", tier_label(*d));
    }
    println!();
    for i in 0..n {
        print!("{:>12}", tier_label(TIERS[i]));
        for j in 0..n {
            if i == j {
                print!("{:>12}", "—");
            } else {
                print!("{:>+12.2}", margin[i][j]);
            }
        }
        println!();
    }

    // ---- IMPLIED LADDER (by overall win-rate across all games played) ----
    let mut ranking: Vec<(BotDifficulty, f64)> = (0..n)
        .map(|i| {
            let wr = if games_for[i] > 0 {
                wins_for[i] as f64 / games_for[i] as f64 * 100.0
            } else {
                0.0
            };
            (TIERS[i], wr)
        })
        .collect();
    ranking.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());

    println!("\n================ IMPLIED LADDER (by overall win-rate) ================");
    for (rank, (d, wr)) in ranking.iter().enumerate() {
        println!("  {}. {:<11} {:.1}% overall", rank + 1, tier_label(*d), wr);
    }
    let order = ranking
        .iter()
        .map(|(d, _)| tier_label(*d))
        .collect::<Vec<_>>()
        .join(" > ");
    println!("\n  Implied ordering (strongest -> weakest): {order}");
    println!(
        "\nTotal tournament elapsed: {:.1}s",
        overall_start.elapsed().as_secs_f64()
    );
}
