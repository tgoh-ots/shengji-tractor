//! Mixed-partner paired evaluation.
//!
//! Homogeneous self-play can reward private conventions that do not transfer to
//! human teammates.  This benchmark keeps deals and role swaps paired while the
//! focal bot, its partner, and the opposing partnership can use different tiers.

use std::env;

use shengji_core::bot::harness::{print_paired_ab, run_paired_crossplay, Contestant};
use shengji_core::bot::BotDifficulty;

fn run(
    focal: BotDifficulty,
    partner: BotDifficulty,
    opponent: BotDifficulty,
    pairs: usize,
    seed: u64,
) {
    let result = run_paired_crossplay(
        &Contestant::tier(focal),
        &Contestant::tier(partner),
        &Contestant::tier(opponent),
        pairs,
        seed,
    );
    print_paired_ab(&result);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let pairs = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(200);
    let seed = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(0xC0A7);
    if env::var("SHENGJI_BOT_BUDGET_MS").is_err() {
        env::set_var("SHENGJI_BOT_BUDGET_MS", "80");
    }

    println!("MIXED-PARTNER CROSS-PLAY — {pairs} paired decks, seed {seed:#x}");
    run(
        BotDifficulty::Expert,
        BotDifficulty::Easy,
        BotDifficulty::Easy,
        pairs,
        seed,
    );
    run(
        BotDifficulty::Expert,
        BotDifficulty::Enoch,
        BotDifficulty::Easy,
        pairs,
        seed,
    );
    run(
        BotDifficulty::Enoch,
        BotDifficulty::Easy,
        BotDifficulty::Expert,
        pairs,
        seed,
    );
    run(
        BotDifficulty::Grandmaster,
        BotDifficulty::Easy,
        BotDifficulty::Expert,
        pairs,
        seed,
    );
}
