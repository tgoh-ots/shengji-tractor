//! Headless A/B benchmark for the determinized-search BUDGET.
//!
//! This pits the SAME determinized search at the NEW (strengthened)
//! [`SearchConfig`] against the OLD (pre-change) one, over many seeded games, to
//! measure how much the deeper search actually wins. Both sides use the IDENTICAL
//! policy ([`Policy::Heuristic`] root prior + heuristic rollouts) and the identical
//! candidate cap — the ONLY thing that differs is `time_budget`, `max_worlds`, and
//! `rollout_tricks`. So any win-rate edge is attributable purely to deepening the
//! search, which is exactly what we changed for the live Expert/Enoch/
//! Omniscient tiers.
//!
//! Why call `search::search_play` directly instead of `policy::select_action`?
//! The live in-game knobs (`Knobs::for_difficulty` + `search_budget_ms()`) are
//! fixed at compile time / by the `SHENGJI_BOT_BUDGET_MS` env var, so we cannot run
//! BOTH the old and new config in one process through that path. Calling the public
//! `search_play` with an explicit `SearchConfig` per side lets us vary new-vs-old
//! cleanly and deterministically in a single run. The machinery exercised
//! (determinizer → world sampling → heuristic rollouts → static leaf eval) is
//! exactly the live honest search; only the world source differs not at all.
//!
//! Bidding, the kitty exchange, and trick-finishing are identical for both
//! partnerships (driven by the shared `Expert` driver decisions), so they cannot
//! bias the play-phase A/B. We alternate which partnership gets the NEW config
//! across games to cancel the landlord/dealer positional edge.
//!
//! Run with:
//!   cargo run --release --example budget_benchmark -- [num_games] [base_seed]
//!
//! It also prints the per-move search latency for each config so you can confirm
//! the new budget stays within the pacing/turn envelope.

use std::env;
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::SeedableRng;

use shengji_core::bot::harness::seeded_draw_phase;
use shengji_core::bot::policy;
use shengji_core::bot::search::{search_play, Policy, SearchConfig};
use shengji_core::bot::BotDifficulty;
use shengji_core::game_state::play_phase::PlayPhase;
use shengji_core::game_state::GameState;
use shengji_core::interactive::Action;

use shengji_mechanics::deck::Deck;
use shengji_mechanics::types::{Card, PlayerID};

/// The OLD (pre-change) in-game determinized-search config: 1000ms budget, 48
/// worlds, 8 rollout tricks. (The candidate cap and policy are unchanged, so they
/// are shared between both sides.)
const OLD_BUDGET_MS: u64 = 1000;
const OLD_WORLDS: usize = 48;
const OLD_ROLLOUT_TRICKS: usize = 8;

/// The NEW (strengthened) in-game determinized-search config: deeper search now
/// that it runs off-lock and is masked by the ~1200ms visible pacing.
const NEW_BUDGET_MS: u64 = 2200;
const NEW_WORLDS: usize = 144;
const NEW_ROLLOUT_TRICKS: usize = 12;

/// Shared candidate cap (unchanged between old and new).
const MAX_CANDIDATES: usize = 6;

/// A search-config side in the A/B. Carries its own latency accumulator so we can
/// report the typical per-move search time for each config.
#[derive(Clone, Copy)]
struct Side {
    label: &'static str,
    budget_ms: u64,
    worlds: usize,
    rollout_tricks: usize,
}

impl Side {
    fn new() -> Self {
        Side {
            label: "Search@NEW",
            budget_ms: NEW_BUDGET_MS,
            worlds: NEW_WORLDS,
            rollout_tricks: NEW_ROLLOUT_TRICKS,
        }
    }
    fn old() -> Self {
        Side {
            label: "Search@OLD",
            budget_ms: OLD_BUDGET_MS,
            worlds: OLD_WORLDS,
            rollout_tricks: OLD_ROLLOUT_TRICKS,
        }
    }

    fn config(&self, seed: u64) -> SearchConfig {
        SearchConfig {
            time_budget: Duration::from_millis(self.budget_ms),
            max_candidates: MAX_CANDIDATES,
            max_worlds: self.worlds,
            rollout_tricks: self.rollout_tricks,
            seed,
            policy: Policy::Heuristic,
            rollout_policy: Policy::Heuristic,
        }
    }
}

/// Per-move latency accumulator for one side.
#[derive(Default, Clone, Copy)]
struct Latency {
    total: Duration,
    moves: u64,
    max: Duration,
}

impl Latency {
    fn record(&mut self, d: Duration) {
        self.total += d;
        self.moves += 1;
        if d > self.max {
            self.max = d;
        }
    }
    fn avg_ms(&self) -> f64 {
        if self.moves == 0 {
            0.0
        } else {
            self.total.as_secs_f64() * 1000.0 / self.moves as f64
        }
    }
}

/// Pick the play-phase cards for `actor` using the determinized search at `side`'s
/// config, honoring the honesty boundary (only the actor's own redacted view).
/// Falls back to the shared heuristic policy if the search produces nothing (so the
/// game never stalls). Records the per-move search latency into `lat`.
fn play_cards_for(
    s: &PlayPhase,
    actor: PlayerID,
    side: Side,
    seed: u64,
    lat: &mut Latency,
) -> Option<Vec<Card>> {
    let view = GameState::Play(s.clone()).for_player(actor);
    let pp = match &view {
        GameState::Play(pp) => pp,
        _ => return None,
    };
    let config = side.config(seed);
    let start = Instant::now();
    let chosen = search_play(pp, actor, config);
    lat.record(start.elapsed());
    match chosen {
        Some(c) => Some(c),
        // Fallback so a degenerate position never stalls the harness; this is the
        // SAME fallback the live policy uses, and it is config-independent so it
        // can't bias the A/B.
        None => match policy::select_action(&view, actor, BotDifficulty::Expert).ok()? {
            Some(Action::PlayCards(c)) => Some(c),
            _ => None,
        },
    }
}

/// The outcome of one finished hand, from the NEW config partnership's perspective.
struct GameOutcome {
    new_won: bool,
    new_point_margin: isize,
}

/// Drive one seeded hand. `new_is_landlord_team` selects which partnership uses the
/// NEW search config; the other uses the OLD config. Both partnerships bid /
/// exchange identically via the shared `Expert` driver, so only the play-phase search
/// budget differs.
#[allow(clippy::too_many_arguments)]
fn play_one_hand(
    new_is_landlord_team: bool,
    new_side: Side,
    old_side: Side,
    rng: &mut StdRng,
    new_lat: &mut Latency,
    old_lat: &mut Latency,
) -> Option<GameOutcome> {
    let decks = vec![Deck::default(), Deck::default()];
    let draw = seeded_draw_phase(&decks, rng);
    let seats: Vec<PlayerID> = draw.propagated().players().iter().map(|p| p.id).collect();

    // Seats 0,2 are the landlord (defending) team; 1,3 attack. The NEW config
    // occupies the landlord team iff `new_is_landlord_team`.
    let side_of = |seat_idx: usize| -> (Side, bool) {
        let is_landlord_team = seat_idx % 2 == 0;
        let is_new = is_landlord_team == new_is_landlord_team;
        if is_new {
            (new_side, true)
        } else {
            (old_side, false)
        }
    };

    let mut state = GameState::Draw(draw);
    let mut iters = 0usize;
    loop {
        iters += 1;
        if iters > 2_000_000 {
            return None;
        }
        match &mut state {
            GameState::Initialize(_) => return None,
            GameState::Draw(s) => {
                if !s.done_drawing() {
                    let p = s.next_player().ok()?;
                    s.draw_card(p).ok()?;
                } else if s.bid_decided() {
                    let responsible = s.next_player().ok()?;
                    state = GameState::Exchange(s.advance(responsible).ok()?);
                } else {
                    // Both partnerships bid via the same Expert driver so the trump /
                    // landlord is identical regardless of side.
                    let mut bid = false;
                    for &seat in &seats {
                        if let Some(b) = policy::choose_bid(s, seat, BotDifficulty::Expert) {
                            if s.bid(seat, b.card, b.count) {
                                bid = true;
                                break;
                            }
                        }
                    }
                    if !bid && s.reveal_card().is_err() {
                        for &seat in &seats {
                            if let Some(b) =
                                s.valid_bids(seat).ok()?.into_iter().min_by_key(|b| b.count)
                            {
                                if s.bid(seat, b.card, b.count) {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            GameState::Exchange(s) => {
                let landlord = s.landlord();
                let view = GameState::Exchange(s.clone()).for_player(landlord);
                match policy::select_action(&view, landlord, BotDifficulty::Expert).ok()? {
                    Some(Action::MoveCardToKitty(c)) => s.move_card_to_kitty(landlord, c).ok()?,
                    Some(Action::MoveCardToHand(c)) => s.move_card_to_hand(landlord, c).ok()?,
                    Some(Action::SetFriends(f)) => s.set_friends(landlord, f).ok()?,
                    _ => state = GameState::Play(s.advance(landlord).ok()?),
                }
            }
            GameState::Play(s) => {
                if s.game_finished() {
                    let (non_landlord_points, _) = s.calculate_points();
                    let (_init, landlord_won, _msgs) = s.finish_game().ok()?;

                    // NEW is the defending (landlord) team iff new_is_landlord_team.
                    let new_is_defender = new_is_landlord_team;
                    let new_won = landlord_won == new_is_defender;
                    let new_point_margin = if new_is_defender {
                        -non_landlord_points
                    } else {
                        non_landlord_points
                    };
                    return Some(GameOutcome {
                        new_won,
                        new_point_margin,
                    });
                }
                match s.trick().next_player() {
                    None => {
                        s.finish_trick().ok()?;
                    }
                    Some(actor) => {
                        let actor_idx = seats.iter().position(|x| *x == actor)?;
                        let (side, is_new) = side_of(actor_idx);
                        // Deterministic per-decision seed from the observable state
                        // so the search is reproducible (and the two sides get
                        // independent but stable seeds).
                        let hand_size = s
                            .hands()
                            .get(actor)
                            .map(|h| h.values().sum::<usize>())
                            .unwrap_or(0);
                        let seed = (actor.0 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                            ^ (hand_size as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
                            ^ (s.trick().played_cards().len() as u64)
                                .wrapping_mul(0x94D0_49BB_1331_11EB);
                        let lat = if is_new { &mut *new_lat } else { &mut *old_lat };
                        let cards = play_cards_for(s, actor, side, seed, lat)?;
                        s.play_cards(actor, &cards).ok()?;
                    }
                }
            }
        }
    }
}

fn run_match(num_games: usize, base_seed: u64) {
    let new_side = Side::new();
    let old_side = Side::old();
    let start = Instant::now();

    let mut new_wins = 0usize;
    let mut old_wins = 0usize;
    let mut total_margin: isize = 0;
    let mut completed = 0usize;
    let mut new_lat = Latency::default();
    let mut old_lat = Latency::default();

    for g in 0..num_games {
        let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(g as u64));
        let new_is_landlord_team = g % 2 == 0;
        match play_one_hand(
            new_is_landlord_team,
            new_side,
            old_side,
            &mut rng,
            &mut new_lat,
            &mut old_lat,
        ) {
            Some(outcome) => {
                completed += 1;
                if outcome.new_won {
                    new_wins += 1;
                } else {
                    old_wins += 1;
                }
                total_margin += outcome.new_point_margin;
                let elapsed = start.elapsed().as_secs_f64();
                eprintln!(
                    "  game {g}: {} (margin {:+}) | NEW {new_wins}-{old_wins} OLD | {elapsed:.0}s",
                    if outcome.new_won {
                        "NEW won"
                    } else {
                        "OLD won"
                    },
                    outcome.new_point_margin,
                );
            }
            None => eprintln!("  game {g}: engine error / skipped"),
        }
    }

    let win_rate = if completed > 0 {
        new_wins as f64 / completed as f64 * 100.0
    } else {
        0.0
    };

    println!();
    println!(
        "=== {} (NEW: {}ms/{} worlds/{} rollout) vs {} (OLD: {}ms/{} worlds/{} rollout) ===",
        new_side.label,
        NEW_BUDGET_MS,
        NEW_WORLDS,
        NEW_ROLLOUT_TRICKS,
        old_side.label,
        OLD_BUDGET_MS,
        OLD_WORLDS,
        OLD_ROLLOUT_TRICKS,
    );
    println!("  Games completed:      {completed}");
    println!("  NEW wins:             {new_wins}");
    println!("  OLD wins:             {old_wins}");
    println!("  NEW win-rate:         {win_rate:.2}%");
    println!(
        "  NEW avg point margin: {:+.2} pts/game",
        total_margin as f64 / completed.max(1) as f64
    );
    println!(
        "  NEW per-move search:  avg {:.0}ms  max {:.0}ms  ({} moves)",
        new_lat.avg_ms(),
        new_lat.max.as_secs_f64() * 1000.0,
        new_lat.moves,
    );
    println!(
        "  OLD per-move search:  avg {:.0}ms  max {:.0}ms  ({} moves)",
        old_lat.avg_ms(),
        old_lat.max.as_secs_f64() * 1000.0,
        old_lat.moves,
    );
    println!("  Elapsed: {:.1}s", start.elapsed().as_secs_f64());
    println!();

    // Simple two-sided sign-test style readout: how confident are we that NEW > 50%?
    if completed > 0 {
        let n = completed as f64;
        let p = new_wins as f64 / n;
        let se = (0.25 / n).sqrt(); // SE under the null p=0.5
        let z = (p - 0.5) / se;
        println!("  (z vs 50% null = {z:+.2}; |z|>1.96 ≈ p<0.05 two-sided)");
    }
}

fn main() {
    // Ignore any inherited budget override — this benchmark sets its budgets
    // explicitly per side via SearchConfig, so an env var would only confuse.
    env::remove_var("SHENGJI_BOT_BUDGET_MS");

    let args: Vec<String> = env::args().collect();
    let num_games: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(60);
    let base_seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0xB0D);

    println!("BUDGET A/B benchmark: Search@NEW vs Search@OLD determinized search");
    println!("Games: {num_games}  base_seed: {base_seed:#x}");
    println!(
        "Both sides use the heuristic prior + heuristic rollouts and the same \
         candidate cap; only the time budget, max_worlds, and rollout_tricks \
         differ. Sides alternate landlord/attacker each game.\n"
    );

    run_match(num_games, base_seed);
}
