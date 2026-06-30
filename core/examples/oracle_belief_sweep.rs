//! SEER feasibility test — the oracle-belief upper bound.
//!
//! Biases the Expert determinized search's world-SAMPLING toward the TRUE hidden
//! hands (strength in [0,1]); the search itself stays honest (redacted view only).
//! If even a PERFECT belief (strength=1) doesn't lift Expert's win-rate, a LEARNED
//! belief model (Seer) cannot — so this gates the full Seer build.
//!
//!   cargo run --release --example oracle_belief_sweep -- [pairs] [base_seed]
//!
//! Key comparison: Expert@belief=1.0 vs Expert@belief=0.0 (identical policy, only the
//! world-sampler differs) — a clean isolation of how much realistic worlds are worth.

use std::env;

use shengji_core::bot::harness::{print_paired_ab, run_paired_ab, Contestant, PlayBrain, Seat};
use shengji_core::bot::BotDifficulty;

fn expert_oracle(belief: f64) -> Contestant {
    Contestant::new(
        format!("Expert@belief={belief}"),
        Seat {
            play: PlayBrain::TierOracle {
                difficulty: BotDifficulty::Expert,
                belief,
            },
            bid: BotDifficulty::Expert,
            kitty: BotDifficulty::Expert,
        },
    )
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let pairs: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(150);
    let base_seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0x5EED);
    if env::var("SHENGJI_BOT_BUDGET_MS").is_err() {
        env::set_var("SHENGJI_BOT_BUDGET_MS", "120");
    }
    println!("SEER ORACLE-BELIEF UPPER-BOUND TEST");
    println!(
        "pairs: {pairs} × 2 orientations   seed: {base_seed:#x}   budget_ms: {}\n",
        env::var("SHENGJI_BOT_BUDGET_MS").unwrap_or_default()
    );

    // (1) ISOLATION: perfect / partial belief vs NO belief — identical Expert policy,
    //     only the world-sampler differs. This is the decisive number.
    println!("-- belief vs no-belief (same Expert policy; >50% ⇒ realistic worlds help) --");
    for b in [1.0_f64, 0.5] {
        print_paired_ab(&run_paired_ab(&expert_oracle(b), &expert_oracle(0.0), pairs, base_seed));
    }

    // (2) HEADROOM: does a perfect-belief Expert reach the strong tiers?
    println!("\n-- perfect-belief Expert vs the ladder --");
    print_paired_ab(&run_paired_ab(
        &expert_oracle(1.0),
        &Contestant::tier(BotDifficulty::Enoch),
        pairs,
        base_seed,
    ));
    print_paired_ab(&run_paired_ab(
        &expert_oracle(1.0),
        &Contestant::tier(BotDifficulty::Easy),
        pairs,
        base_seed,
    ));
}
