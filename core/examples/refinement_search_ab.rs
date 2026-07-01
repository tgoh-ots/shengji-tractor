//! Fixed-work search benchmark for strategy refinements.
//!
//! The subject uses honest determinized search with a fixed world/candidate/
//! rollout count and a generous deadline, so CPU load does not change its work.
//! The opponent is the frozen `HeuristicVersion::Legacy` ruler. Run this exact
//! example on the baseline commit and the refinement commit with identical args;
//! the change in subject-vs-ruler outcomes measures the combined search-policy
//! effect without moving the control policy.
//!
//! Usage:
//!   cargo run --release --example refinement_search_ab -- [pairs] [seed] [worlds] [mode] [rollout] [candidates]

use std::env;
use std::time::Duration;

use shengji_core::bot::harness::{print_paired_ab, run_paired_ab, Contestant, PlayBrain, Seat};
use shengji_core::bot::heuristics::HeuristicVersion;
use shengji_core::bot::search::{Policy, SearchConfig};
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
    let pairs = args
        .get(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(800);
    let seed = args
        .get(2)
        .and_then(|value| value.parse().ok())
        .unwrap_or(0x5EED);
    let worlds = args
        .get(3)
        .and_then(|value| value.parse().ok())
        .unwrap_or(8);
    let mode = args
        .iter()
        .skip(4)
        .find(|arg| arg.as_str() != "--trace")
        .map(String::as_str)
        .unwrap_or("heuristic");
    let trace = args.iter().any(|arg| arg == "--trace");
    let rollout_tricks = args
        .iter()
        .skip(5)
        .find(|arg| arg.as_str() != "--trace")
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(4);
    let max_candidates = args
        .iter()
        .skip(6)
        .find(|arg| arg.as_str() != "--trace")
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(4);
    let (policy, rollout_policy) = match mode {
        "heuristic" => (Policy::Heuristic, Policy::Heuristic),
        "expert" => (Policy::Net, Policy::Heuristic),
        "enoch" => (Policy::EnochHeuristic, Policy::EnochHeuristic),
        "grandmaster" => (Policy::EnochHeuristic, Policy::Heuristic),
        other => panic!(
            "unknown search mode {:?}; use heuristic, expert, enoch, or grandmaster",
            other
        ),
    };

    let search = Contestant::new(
        format!("Search@{worlds}w/{max_candidates}c"),
        Seat {
            play: PlayBrain::Search(SearchConfig {
                // The world cap, not time, must bind for reproducible comparison.
                // Deliberately generous: the fixed world cap, not wall-clock
                // contention, must bind in a cross-checkout comparison.
                time_budget: Duration::from_secs(30),
                max_candidates,
                max_worlds: worlds,
                rollout_tricks,
                seed,
                policy,
                rollout_policy,
            }),
            bid: BotDifficulty::Expert,
            kitty: BotDifficulty::Easy,
        },
    );
    let legacy = Contestant::new(
        "Heur@Legacy",
        Seat {
            play: PlayBrain::HeuristicDirect(HeuristicVersion::Legacy),
            bid: BotDifficulty::Expert,
            kitty: BotDifficulty::Easy,
        },
    );

    println!(
        "FIXED-WORK REFINEMENT SEARCH A/B: pairs={pairs} seed={seed:#x} worlds={worlds} cands={max_candidates} rollout={rollout_tricks} mode={mode}"
    );
    let result = run_paired_ab(&search, &legacy, pairs, seed);
    print_paired_ab(&result);
    if trace {
        print_trace(&format!("search-{mode}-r{rollout_tricks}"), &result);
    }
}
