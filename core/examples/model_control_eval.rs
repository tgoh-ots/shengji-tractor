//! Machine-readable matched-deal Expert-vs-Easy control arm.
//! Run embedded and candidate models in separate processes, then subtract the
//! per-deck outcomes by index to obtain a direct model-effect estimate.

use shengji_core::bot::harness::{run_paired_ab, Contestant};
use shengji_core::bot::BotDifficulty;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let pairs = args
        .get(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(200);
    let seed = args
        .get(2)
        .and_then(|value| value.parse().ok())
        .unwrap_or(0x5EED);
    let result = run_paired_ab(
        &Contestant::tier(BotDifficulty::Expert),
        &Contestant::tier(BotDifficulty::Easy),
        pairs,
        seed,
    );
    println!(
        "{}",
        serde_json::json!({
            "manifest_version": 1,
            "pairs_requested": pairs,
            "seed": seed,
            "complete_pairs": result.complete_pairs,
            "failed_hands": result.failed_hands(),
            "per_deck_winrate": result.per_deck_winrate,
            "per_deck_margin": result.per_deck_margin,
            "per_deck_level_utility": result.per_deck_level_utility,
        })
    );
}
