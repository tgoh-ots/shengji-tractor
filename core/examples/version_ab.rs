//! Cross-version A/B against a FIXED yardstick — the measurement the tier-vs-tier
//! benchmarks (`gm_benchmark`, `tournament`) structurally CANNOT be. Those pit two
//! tiers, so a change to the SHARED scorer lands on BOTH sides and a change that
//! helps everyone equally cancels out of the win-rate. Here every subject plays the
//! FROZEN [`HeuristicVersion::Legacy`] scorer, which is byte-identical across code
//! versions, so it is a fixed ruler:
//!
//! * Standalone, this reports how far the current scorer / Enoch playbook sits
//!   ABOVE the frozen baseline.
//! * Run in a master worktree AND your change branch, the DELTA between the two
//!   subject win-rates is the ABSOLUTE effect of your change (same ruler in both).
//!
//! Both subjects are SEARCH-LESS (greedy), so runs are reproducible and
//! CPU-load-independent (no time-boxed search to perturb), and fast enough for a
//! few thousand paired decks. Two subjects cover both change surfaces:
//!
//! * `Heur@New` vs Legacy — the shared `score_lead` / `score_follow` scorer.
//! * `Enoch` vs Legacy — the Enoch playbook (safe throws, kitty valuation, tractor-
//!   first leads, ...) that the plain-heuristic A/B does NOT exercise.
//!
//! All matchups use [`run_paired_ab`]: each deck is played in BOTH orientations
//! (deal luck paired out) and reported with a Wilson + a paired-bootstrap 95% CI
//! and the minimum detectable effect, so "no difference" is distinguished from
//! "underpowered".
//!
//! Protocol to isolate one change's effect:
//! ```text
//!   git worktree add --detach /tmp/sj-old master
//!   cp core/examples/version_ab.rs /tmp/sj-old/core/examples/   # add the tool to OLD
//!   ( cd /tmp/sj-old && cargo run --release --example version_ab -- 1500 0x5EED )   # OLD
//!   cargo run --release --example version_ab -- 1500 0x5EED                         # NEW
//!   # compare each subject's win-rate vs the frozen Legacy ruler; the gap is the delta.
//! ```
//!
//! Run with:
//!   cargo run --release --example version_ab -- [pairs] [base_seed]

use std::env;

use shengji_core::bot::harness::{print_paired_ab, run_paired_ab, Contestant, PlayBrain, Seat};
use shengji_core::bot::heuristics::HeuristicVersion;
use shengji_core::bot::BotDifficulty;

fn print_trace(tag: &str, result: &shengji_core::bot::harness::PairedABResult) {
    assert_eq!(
        result.complete_pairs, result.pairs,
        "trace output requires every mirrored deck pair to complete"
    );
    for (index, ((win, margin), level)) in result
        .per_deck_winrate
        .iter()
        .zip(&result.per_deck_margin)
        .zip(&result.per_deck_level_utility)
        .enumerate()
    {
        println!("TRACE\t{tag}\t{index}\t{win:.1}\t{margin:.6}\t{level:.6}");
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let pairs: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(800);
    let base_seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0x5EED);
    let trace = args.get(3).is_some_and(|arg| arg == "--trace");

    println!("CROSS-VERSION A/B vs the FROZEN Legacy yardstick (search-less, reproducible)");
    println!(
        "pairs: {pairs} decks × 2 orientations = {} hands/matchup   base_seed: {base_seed:#x}\n",
        pairs * 2
    );

    // The fixed reference opponent: the frozen Legacy scorer. Bid Expert + bury Easy
    // so only the PLAY scorer differs (matches `paired_eval`'s heuristic A/B); both
    // are version-independent, so this opponent is identical in every checkout.
    let legacy = Contestant::new(
        "Heur@Legacy",
        Seat {
            play: PlayBrain::HeuristicDirect(HeuristicVersion::Legacy),
            bid: BotDifficulty::Expert,
            kitty: BotDifficulty::Easy,
        },
    );

    // Subject 1 — the current SHARED scorer (waste-on-lost-trick, big-throw point
    // protection, trump-takedown, plus all prior boss/partner logic).
    let new_heur = Contestant::new(
        "Heur@New",
        Seat {
            play: PlayBrain::HeuristicDirect(HeuristicVersion::New),
            bid: BotDifficulty::Expert,
            kitty: BotDifficulty::Easy,
        },
    );

    // Subject 2 — the current Enoch playbook (greedy). This is what exercises the
    // Enoch-only changes (safe set throws + Ace attach, declarer-aware kitty
    // valuation) that the plain-heuristic subject can't reach. Its own bid/kitty are
    // Enoch's (`bid_strength_enoch` / `choose_kitty_enoch`), which were NOT changed,
    // so they stay constant across versions and don't confound the cross-version delta.
    let enoch = Contestant::new(
        "Enoch(greedy)",
        Seat {
            play: PlayBrain::EnochGreedy,
            bid: BotDifficulty::Enoch,
            kitty: BotDifficulty::Enoch,
        },
    );

    let heuristic_result = run_paired_ab(&new_heur, &legacy, pairs, base_seed);
    print_paired_ab(&heuristic_result);
    if trace {
        print_trace("heuristic", &heuristic_result);
    }
    println!();
    let enoch_result = run_paired_ab(&enoch, &legacy, pairs, base_seed);
    print_paired_ab(&enoch_result);
    if trace {
        print_trace("enoch", &enoch_result);
    }
}
