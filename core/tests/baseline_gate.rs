//! Committed-baseline NON-INFERIORITY strength gate.
//!
//! These tests pin the load-bearing, *budget-independent* (search-less) strength
//! relationships with the paired-on-mirrored-deck harness + confidence intervals
//! (see `shengji_core::bot::harness` and `docs/bot-eval-baseline.md`). They fail
//! only if a build drops a relationship BELOW its committed floor — the floors sit
//! comfortably under the measured baseline (and under the run-to-run jitter from
//! Rust's per-process `HashMap` iteration order), so a green run is signal, not
//! luck, and a red run means a real regression to the shared scorer / Easy knobs.
//!
//! Why search-LESS matchups here: they are fast (~seconds in debug) and stable, so
//! they make a good fast CI gate. The SEARCH/NET tiers are budget-sensitive and
//! noisy in a debug build, so their gate is the release-only `#[ignore]` test
//! below (and the `paired_eval` / `expert_ab` example harnesses) — exactly the
//! "tight gate on the budget-independent tiers, coarse net on the search tiers"
//! split from `docs/bot-training-roadmap.md`.

use shengji_core::bot::harness::{run_paired_ab, Contestant, PlayBrain, Seat};
use shengji_core::bot::heuristics::HeuristicVersion;
use shengji_core::bot::BotDifficulty;

/// Deck-pairs per matchup (each played in BOTH orientations = 2× hands). 200
/// gives a ~±4pp 95% MDE on win-rate, plenty to resolve the committed margins.
const PAIRS: usize = 200;
const BASE_SEED: u64 = 0x5EED;

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
            play: PlayBrain::HeuristicDirect(version),
            bid: BotDifficulty::Expert,
            kitty: BotDifficulty::Easy,
        },
    )
}

/// BASELINE (measured 2026-06, 200 pairs, search-less): Easy@NEW beats Easy@OLD
/// ~54% (paired-bootstrap95 ≈ [50.5, 58.5]). FLOOR 0.50: the strengthened Easy
/// must still beat the old noisy Easy. A red here means the Easy knobs or the
/// shared heuristic ranking regressed.
#[test]
fn baseline_easy_new_beats_old() {
    let r = run_paired_ab(
        &easy_knobs("Easy@NEW", 0.06, 1.1),
        &easy_knobs("Easy@OLD", 0.28, 3.5),
        PAIRS,
        BASE_SEED,
    );
    let wr = r.paired_win_rate();
    let (lo, _hi) = r.winrate_bootstrap_ci();
    assert!(
        wr > 0.50,
        "Easy@NEW should beat Easy@OLD: paired win-rate {:.3} (bootstrap95 lo {:.3}); \
         floor 0.50. A regression to the Easy knobs / shared heuristic?",
        wr,
        lo,
    );
    // Non-inferiority sanity: the lower CI must not collapse below 0.45.
    assert!(
        lo > 0.45,
        "Easy@NEW vs Easy@OLD bootstrap lower bound {:.3} fell below 0.45",
        lo
    );
}

/// BASELINE (measured 2026-06, 200 pairs, search-less): the NEW boss-/partner-aware
/// play scorer beats the frozen LEGACY scorer ~57% (paired-bootstrap95 ≈
/// [53.5, 60.0]). FLOOR 0.51: NEW must remain at least as strong as LEGACY. A red
/// here means the shared `score_lead`/`score_follow` scorer regressed.
#[test]
fn baseline_new_heuristic_not_worse_than_legacy() {
    let r = run_paired_ab(
        &heuristic_direct("Heur@NEW", HeuristicVersion::New),
        &heuristic_direct("Heur@LEGACY", HeuristicVersion::Legacy),
        PAIRS,
        BASE_SEED,
    );
    let wr = r.paired_win_rate();
    let (lo, _hi) = r.winrate_bootstrap_ci();
    assert!(
        wr > 0.51,
        "NEW heuristic should be >= LEGACY: paired win-rate {:.3} (bootstrap95 lo {:.3}); \
         floor 0.51. A regression to the shared score_lead/score_follow scorer?",
        wr,
        lo,
    );
    assert!(
        lo > 0.45,
        "NEW-vs-LEGACY heuristic bootstrap lower bound {:.3} fell below 0.45",
        lo
    );
}

/// COARSE search/net gate (release-recommended; `#[ignore]` so fast CI stays
/// fast and non-flaky). Expert (net-guided determinized search) should beat Easy.
/// Run with:
///   SHENGJI_BOT_BUDGET_MS=60 cargo test -p shengji-core --release \
///       --test baseline_gate -- --ignored
/// To A/B a candidate net without rebuilding, set `SHENGJI_EXPERT_MODEL_PATH`.
#[test]
#[ignore]
fn baseline_expert_beats_easy_search() {
    if std::env::var("SHENGJI_BOT_BUDGET_MS").is_err() {
        std::env::set_var("SHENGJI_BOT_BUDGET_MS", "60");
    }
    // Fewer pairs since each search hand is far slower; still paired.
    let r = run_paired_ab(
        &Contestant::tier(BotDifficulty::Expert),
        &Contestant::tier(BotDifficulty::Easy),
        80,
        BASE_SEED,
    );
    let wr = r.paired_win_rate();
    let (lo, _hi) = r.winrate_bootstrap_ci();
    // Coarse floor: Expert must not LOSE to Easy. (At a low debug budget the
    // search is starved, so we only assert non-inferiority, not a large edge.)
    assert!(
        wr >= 0.50,
        "Expert(search) should be >= Easy: paired win-rate {:.3} (bootstrap95 lo {:.3})",
        wr,
        lo,
    );
}
