//! Paired-on-mirrored-deck benchmark for the **Oracle** (Deep Monte-Carlo) tier.
//!
//! Oracle is the search-FREE `argmax_a Q(s,a)` policy served by `PlayBrain::NetGreedy`
//! over the Q-net pointed to by `SHENGJI_EXPERT_MODEL_PATH`. We measure it head-to-head
//! against the existing honest ladder rungs that DON'T use the net (so the single
//! process-global model is only ever the Oracle's): Easy, Enoch, and the Omniscient
//! cheater. Each matchup uses the shared paired harness (Wilson + paired-bootstrap CIs
//! + MDE), playing each deck in both orientations to cancel deal luck.
//!
//! Run with:
//!   SHENGJI_EXPERT_MODEL_PATH=/path/to/oracle.onnx \
//!     cargo run --release --example oracle_eval -- [pairs] [base_seed]
//!
//! `SHENGJI_BOT_BUDGET_MS` only affects the search-based OPPONENTS (Enoch/Omniscient);
//! Oracle itself is search-free, so its serve cost is one net call per decision.

use std::env;

use shengji_core::bot::harness::{print_paired_ab, run_paired_ab, Contestant, PlayBrain, Seat};
use shengji_core::bot::BotDifficulty;

fn oracle() -> Contestant {
    Contestant::new(
        "Oracle(DMC)",
        Seat {
            play: PlayBrain::NetGreedy,
            // Bid + bury with the Expert heuristics (Oracle only learns PLAY).
            bid: BotDifficulty::Expert,
            kitty: BotDifficulty::Expert,
        },
    )
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let pairs: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(200);
    let base_seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0x0AC1E);
    // which ∈ {all, easy, enoch, omni}: filter opponents (easy is search-free/fast).
    let which = args.get(3).map(|s| s.as_str()).unwrap_or("all");
    if env::var("SHENGJI_BOT_BUDGET_MS").is_err() {
        env::set_var("SHENGJI_BOT_BUDGET_MS", "120");
    }

    let net = env::var("SHENGJI_EXPERT_MODEL_PATH").unwrap_or_else(|_| "<embedded>".to_string());
    println!("ORACLE (Deep Monte-Carlo, search-free argmax-Q) PAIRED EVAL");
    println!(
        "pairs: {pairs} × 2 orientations   base_seed: {base_seed:#x}   Q-net: {net}   \
         opponent budget_ms: {}\n",
        env::var("SHENGJI_BOT_BUDGET_MS").unwrap_or_default()
    );

    // vs each honest ladder rung that does NOT consult the net (so the global model
    // is unambiguously the Oracle's). Higher win-rate = stronger.
    let opps: Vec<BotDifficulty> = match which {
        "easy" => vec![BotDifficulty::Easy],
        "enoch" => vec![BotDifficulty::Enoch],
        "omni" => vec![BotDifficulty::Omniscient],
        _ => vec![
            BotDifficulty::Easy,
            BotDifficulty::Enoch,
            BotDifficulty::Omniscient,
        ],
    };
    for opp in opps {
        let r = run_paired_ab(&oracle(), &Contestant::tier(opp), pairs, base_seed);
        print_paired_ab(&r);
    }
}
