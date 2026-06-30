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
//!   cargo run --release --example refinement_search_ab -- [pairs] [seed] [worlds]

use std::env;
use std::time::Duration;

use shengji_core::bot::harness::{print_paired_ab, run_paired_ab, Contestant, PlayBrain, Seat};
use shengji_core::bot::heuristics::HeuristicVersion;
use shengji_core::bot::search::{Policy, SearchConfig};
use shengji_core::bot::BotDifficulty;

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

    let search = Contestant::new(
        format!("Search@{worlds}w"),
        Seat {
            play: PlayBrain::Search(SearchConfig {
                // The world cap, not time, must bind for reproducible comparison.
                time_budget: Duration::from_secs(5),
                max_candidates: 4,
                max_worlds: worlds,
                rollout_tricks: 4,
                seed,
                policy: Policy::Heuristic,
                rollout_policy: Policy::Heuristic,
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
        "FIXED-WORK REFINEMENT SEARCH A/B: pairs={pairs} seed={seed:#x} worlds={worlds} cands=4 rollout=4"
    );
    print_paired_ab(&run_paired_ab(&search, &legacy, pairs, seed));
}
