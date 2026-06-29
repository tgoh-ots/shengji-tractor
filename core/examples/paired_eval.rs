//! Paired-on-mirrored-deck evaluation with confidence intervals — the
//! statistically-sound measurement substrate (see `docs/bot-training-roadmap.md`).
//!
//! Every matchup is played with [`shengji_core::bot::harness::run_paired_ab`],
//! which plays each deck seed in BOTH orientations (A-as-landlord and
//! B-as-landlord on the IDENTICAL deal). Pairing the deal out of the comparison
//! cancels the dominant deal-luck variance, so a given number of hands resolves a
//! much smaller strength difference than the legacy unpaired "alternate the
//! landlord across DIFFERENT seeds" harnesses. We report:
//!
//!   * A's win-rate with a Wilson 95% interval (over individual hands) AND a
//!     paired bootstrap 95% interval (resampling over decks — pairing-aware);
//!   * A's mean paired point margin;
//!   * the 95% minimum detectable effect (MDE) on the win-rate, so "no difference"
//!     is distinguishable from "underpowered".
//!
//! The search-LESS matchups (Easy knobs, NEW-vs-LEGACY heuristic) are fast and
//! stable; the search matchups honor `SHENGJI_BOT_BUDGET_MS` (default 80ms here).
//! With `SHENGJI_EXPERT_MODEL_PATH` set, the Expert matchups use the override net,
//! so you can A/B a freshly-trained net WITHOUT rebuilding.
//!
//! Run with:
//!   cargo run --release --example paired_eval -- [pairs] [base_seed] [which]
//!     which ∈ {all, fast, search}  (default: all)

use std::env;

use shengji_core::bot::harness::{print_paired_ab, run_paired_ab, Contestant, PlayBrain, Seat};
use shengji_core::bot::heuristics::HeuristicVersion;
use shengji_core::bot::BotDifficulty;

fn easy_knobs(label: &str, epsilon: f64, temperature: f64) -> Contestant {
    Contestant::new(
        label,
        Seat {
            play: PlayBrain::Easy {
                epsilon,
                temperature,
            },
            bid: BotDifficulty::Easy,
            kitty: BotDifficulty::Easy,
        },
    )
}

fn heuristic_direct(label: &str, version: HeuristicVersion) -> Contestant {
    Contestant::new(
        label,
        Seat {
            // Bid Expert + bury Easy so only the PLAY scorer differs (matches the
            // legacy `heuristic_benchmark`).
            play: PlayBrain::HeuristicDirect(version),
            bid: BotDifficulty::Expert,
            kitty: BotDifficulty::Easy,
        },
    )
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let pairs: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let base_seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0x5EED);
    let which = args.get(3).map(|s| s.as_str()).unwrap_or("all");

    if env::var("SHENGJI_BOT_BUDGET_MS").is_err() {
        env::set_var("SHENGJI_BOT_BUDGET_MS", "80");
    }

    println!("PAIRED-ON-MIRRORED-DECK EVALUATION");
    println!(
        "pairs: {pairs} decks × 2 orientations = {} hands per matchup   base_seed: {base_seed:#x}",
        pairs * 2
    );
    let net = env::var("SHENGJI_EXPERT_MODEL_PATH").unwrap_or_else(|_| "<embedded>".to_string());
    println!(
        "budget_ms: {}   expert net: {net}\n",
        env::var("SHENGJI_BOT_BUDGET_MS").unwrap_or_default()
    );

    if which == "all" || which == "fast" {
        println!("-- search-LESS matchups (fast, stable) --");
        // The Easy knob change: Easy@NEW should clearly beat Easy@OLD.
        let r = run_paired_ab(
            &easy_knobs("Easy@NEW", 0.06, 1.1),
            &easy_knobs("Easy@OLD", 0.28, 3.5),
            pairs,
            base_seed,
        );
        print_paired_ab(&r);
        // The shared play scorer: NEW boss-/partner-aware vs frozen LEGACY.
        let r = run_paired_ab(
            &heuristic_direct("Heur@NEW", HeuristicVersion::New),
            &heuristic_direct("Heur@LEGACY", HeuristicVersion::Legacy),
            pairs,
            base_seed,
        );
        print_paired_ab(&r);
    }

    if which == "all" || which == "search" {
        println!("\n-- search matchups (honor SHENGJI_BOT_BUDGET_MS; slower) --");
        let r = run_paired_ab(
            &Contestant::tier(BotDifficulty::Expert),
            &Contestant::tier(BotDifficulty::Easy),
            pairs,
            base_seed,
        );
        print_paired_ab(&r);
        let r = run_paired_ab(
            &Contestant::tier(BotDifficulty::Enoch),
            &Contestant::tier(BotDifficulty::Expert),
            pairs,
            base_seed,
        );
        print_paired_ab(&r);
    }
}
