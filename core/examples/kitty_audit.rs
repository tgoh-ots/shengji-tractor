//! Phase-1 KITTY (扣底) audit — does the landlord's burial leave value on the table?
//!
//! The kitty is one of the two decisions that decide 升级 games (the other is the
//! bid), yet the learned net touches it ZERO — burial is a pure heuristic
//! (`heuristics::choose_kitty`, or `choose_kitty_enoch` for Enoch). This audit
//! measures whether that heuristic is leaving points on the table, which directs
//! whether a learned kitty model is worth building (see `docs/bot-training-roadmap.md`).
//!
//! Method (paired, burial-isolating): over seeded hands, at the landlord's
//! burial we FORCE several candidate burials on the SAME deal and play each out
//! with the SAME fixed greedy policy for every seat — so the ONLY thing that
//! varies is the burial, and the landlord-team margin difference is purely the
//! burial's effect. Candidates:
//!   * default — `choose_kitty` (the shared heuristic every non-Enoch tier uses);
//!   * enoch   — `choose_kitty_enoch` (the point-budgeted playbook discipline);
//!   * min-pts — a naive "bury the fewest points, then lowest cards" baseline.
//! We report, per strategy, the avg landlord-team margin and avg points buried,
//! plus how often `default` is the BEST candidate and the avg margin it concedes
//! to the best ("regret"). A large regret ⇒ the kitty heuristic is leaking and a
//! learned kitty model (Phase-2) is worth it; a small regret ⇒ leave it alone.
//!
//! The evaluator policy is greedy heuristic-direct (NO search) so the playout is
//! deterministic given the deal+burial — search noise can't swamp the burial signal.
//!
//! Run with:
//!   cargo run --release --example kitty_audit -- [num_hands] [base_seed]

use std::env;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;

use shengji_core::bot::harness::{play_cards_for, seeded_draw_phase, PlayBrain};
use shengji_core::bot::heuristics::{self, HeuristicVersion};
use shengji_core::bot::{policy, BotDifficulty};
use shengji_core::game_state::exchange_phase::ExchangePhase;
use shengji_core::game_state::GameState;

use shengji_mechanics::deck::Deck;
use shengji_mechanics::types::{Card, PlayerID, Trump};

/// The fixed, deterministic evaluator policy: greedy heuristic-direct, no search,
/// so the only variable across candidate burials is the burial itself.
const PLAY: PlayBrain = PlayBrain::HeuristicDirect(HeuristicVersion::New);

fn points_in(cards: &[Card]) -> i32 {
    cards
        .iter()
        .filter_map(|c| c.points().map(|p| p as i32))
        .sum()
}

/// A naive baseline burial: the `kitty_size` cards with the FEWEST points, then
/// the lowest rank — a "dump the junk" burial to compare the real heuristics to.
fn min_points_burial(pool: &[Card], kitty_size: usize) -> Vec<Card> {
    let mut sorted = pool.to_vec();
    sorted.sort_by_key(|c| {
        let pts = c.points().unwrap_or(0) as i32;
        let rank = c.number().map(|n| n.as_u32() as i32).unwrap_or(0);
        (pts, rank)
    });
    sorted.truncate(kitty_size);
    sorted
}

/// Force the landlord's kitty to exactly `target` (a multiset of `kitty_size`
/// cards from the hand∪kitty pool) by reconciling one card at a time: evict an
/// over-represented kitty card to the hand, then bury a still-missing target card
/// from the hand. Returns false if it cannot converge (shouldn't happen for a
/// target drawn from the pool).
fn force_burial(ex: &mut ExchangePhase, landlord: PlayerID, target: &[Card]) -> bool {
    let target_counts = Card::count(target.iter().copied());
    for _ in 0..2000 {
        let kitty_counts = Card::count(ex.visible_kitty().unwrap_or(&[]).iter().copied());
        // 1) Evict an over-represented kitty card (kitty has more than target wants).
        if let Some((&card, _)) = kitty_counts
            .iter()
            .find(|(c, &have)| have > target_counts.get(c).copied().unwrap_or(0))
        {
            if ex.move_card_to_hand(landlord, card).is_err() {
                return false;
            }
            continue;
        }
        // 2) Bury a still-missing target card that we hold in hand.
        let hand_counts = ex
            .hands()
            .get(landlord)
            .map(|h| h.clone())
            .unwrap_or_default();
        let missing = target_counts.iter().find(|(c, &want)| {
            kitty_counts.get(c).copied().unwrap_or(0) < want
                && hand_counts.get(c).copied().unwrap_or(0) > 0
        });
        match missing {
            Some((&card, _)) => {
                if ex.move_card_to_kitty(landlord, card).is_err() {
                    return false;
                }
            }
            None => break, // reconciled (or stuck — verified below)
        }
    }
    Card::count(ex.visible_kitty().unwrap_or(&[]).iter().copied()) == target_counts
}

/// Play a forked exchange (with its burial already forced) to completion with the
/// fixed greedy policy for every seat, and return the LANDLORD-team margin
/// (`-non_landlord_points`; higher = better burial for the landlord side).
fn playout_landlord_margin(ex: &ExchangePhase, landlord: PlayerID) -> Option<isize> {
    let mut state = GameState::Play(ex.advance(landlord).ok()?);
    let mut iters = 0usize;
    loop {
        iters += 1;
        if iters > 2_000_000 {
            return None;
        }
        let GameState::Play(s) = &mut state else {
            return None;
        };
        if s.game_finished() {
            let (non_landlord_points, _) = s.calculate_points();
            let (_init, _won, _msgs) = s.finish_game().ok()?;
            return Some(-non_landlord_points);
        }
        match s.trick().next_player() {
            None => {
                s.finish_trick().ok()?;
            }
            Some(actor) => {
                let cards = play_cards_for(s, actor, &PLAY)?;
                s.play_cards(actor, &cards).ok()?;
            }
        }
    }
}

/// One candidate burial strategy's result for a hand.
struct Cand {
    label: &'static str,
    burial: Vec<Card>,
    points_buried: i32,
    margin: isize,
}

/// Drive one seeded hand up to the landlord's burial; return the pristine Exchange
/// phase (no burial applied yet), the landlord, and the trump.
fn deal_to_exchange(rng: &mut StdRng) -> Option<(ExchangePhase, PlayerID, Trump)> {
    let decks = vec![Deck::default(), Deck::default()];
    let draw = seeded_draw_phase(&decks, rng);
    let seats: Vec<PlayerID> = draw.propagated().players().iter().map(|p| p.id).collect();
    let mut state = GameState::Draw(draw);
    let mut iters = 0usize;
    loop {
        iters += 1;
        if iters > 2_000_000 {
            return None;
        }
        let GameState::Draw(s) = &mut state else {
            return None;
        };
        if !s.done_drawing() {
            let p = s.next_player().ok()?;
            s.draw_card(p).ok()?;
        } else if s.bid_decided() {
            let responsible = s.next_player().ok()?;
            let ex = s.advance(responsible).ok()?;
            let landlord = ex.landlord();
            let trump = ex.trump();
            return Some((ex, landlord, trump));
        } else {
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
                    if let Some(b) = s.valid_bids(seat).ok()?.into_iter().min_by_key(|b| b.count) {
                        if s.bid(seat, b.card, b.count) {
                            break;
                        }
                    }
                }
            }
        }
    }
}

fn audit_one_hand(rng: &mut StdRng) -> Option<Vec<Cand>> {
    let (ex0, landlord, trump) = deal_to_exchange(rng)?;
    let kitty_size = ex0.kitty_size();
    if kitty_size == 0 {
        return None;
    }
    // The hand∪kitty pool the landlord chooses the burial from (audit sees the
    // full unredacted exchange, so the kitty is visible).
    let hand: Vec<Card> = ex0
        .hands()
        .get(landlord)
        .map(|h| Card::cards(h.iter()).copied().collect())
        .unwrap_or_default();
    let mut pool = hand;
    pool.extend_from_slice(ex0.visible_kitty()?);

    let mut burials: Vec<(&'static str, Vec<Card>)> = vec![
        (
            "default",
            heuristics::choose_kitty(&pool, trump, kitty_size),
        ),
        (
            "enoch",
            heuristics::choose_kitty_enoch(&pool, trump, kitty_size),
        ),
        ("min-pts", min_points_burial(&pool, kitty_size)),
    ];
    // Drop any candidate that didn't produce a full-size burial (defensive).
    burials.retain(|(_, b)| b.len() == kitty_size);

    let mut cands = Vec::new();
    for (label, burial) in burials {
        let mut ex = ex0.clone();
        if !force_burial(&mut ex, landlord, &burial) {
            continue;
        }
        let margin = playout_landlord_margin(&ex, landlord)?;
        cands.push(Cand {
            label,
            points_buried: points_in(&burial),
            burial,
            margin,
        });
    }
    if cands.is_empty() {
        None
    } else {
        Some(cands)
    }
}

fn multiset_eq(a: &[Card], b: &[Card]) -> bool {
    Card::count(a.iter().copied()) == Card::count(b.iter().copied())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let num_hands: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(300);
    let base_seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0x217717);

    println!("KITTY (扣底) AUDIT — burial quality vs alternatives, burial-isolated");
    println!("Hands: {num_hands}  base_seed: {base_seed:#x}");
    println!(
        "Each candidate burial is forced on the SAME deal and played out with a \
         fixed greedy policy; the landlord-team margin delta is purely the burial.\n"
    );

    let labels = ["default", "enoch", "min-pts"];
    let mut sum_margin: std::collections::HashMap<&str, i64> = Default::default();
    let mut sum_points: std::collections::HashMap<&str, i64> = Default::default();
    let mut appears: std::collections::HashMap<&str, usize> = Default::default();
    let mut completed = 0usize;
    let mut default_is_best = 0usize;
    let mut default_regret_sum: i64 = 0; // best_margin - default_margin (>= 0)
    let mut default_ne_enoch = 0usize; // default burial != enoch burial

    let start = Instant::now();
    for h in 0..num_hands {
        let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(h as u64));
        let cands = match audit_one_hand(&mut rng) {
            Some(c) => c,
            None => continue,
        };
        completed += 1;
        for c in &cands {
            *sum_margin.entry(c.label).or_default() += c.margin as i64;
            *sum_points.entry(c.label).or_default() += c.points_buried as i64;
            *appears.entry(c.label).or_default() += 1;
        }
        let best = cands.iter().map(|c| c.margin).max().unwrap();
        if let Some(d) = cands.iter().find(|c| c.label == "default") {
            if d.margin >= best {
                default_is_best += 1;
            }
            default_regret_sum += (best - d.margin) as i64;
        }
        if let (Some(d), Some(e)) = (
            cands.iter().find(|c| c.label == "default"),
            cands.iter().find(|c| c.label == "enoch"),
        ) {
            if !multiset_eq(&d.burial, &e.burial) {
                default_ne_enoch += 1;
            }
        }
    }

    println!("=== Results over {completed} hands ===");
    println!("Per-strategy (higher margin = better burial for the landlord team):");
    for l in labels {
        let n = appears.get(l).copied().unwrap_or(0);
        if n == 0 {
            continue;
        }
        println!(
            "  {:>8}: avg landlord margin {:+6.2}   avg points buried {:5.2}   ({} hands)",
            l,
            sum_margin[l] as f64 / n as f64,
            sum_points[l] as f64 / n as f64,
            n,
        );
    }
    if completed > 0 {
        // The NOISE-ROBUST signal: the gap between `default`'s MEAN margin and the
        // best alternative's MEAN margin. (The per-hand "regret" below is the max
        // over candidates, which is upward-biased by per-run playout noise and is
        // only an UPPER bound over these 3 hand-picked burials — report it, but
        // don't draw the conclusion from it.)
        let mean = |l: &str| {
            sum_margin.get(l).copied().unwrap_or(0) as f64
                / appears.get(l).copied().unwrap_or(1).max(1) as f64
        };
        let default_mean = mean("default");
        let best_alt_mean = ["enoch", "min-pts"]
            .iter()
            .filter(|l| appears.contains_key(**l))
            .map(|l| mean(l))
            .fold(f64::NEG_INFINITY, f64::max);
        let mean_gap = best_alt_mean - default_mean; // > 0 ⇒ an alternative beats default on average

        println!();
        println!(
            "  default vs best-alternative MEAN margin gap: {:+.2} pts/hand \
             (>0 ⇒ an alternative is better on average)",
            mean_gap,
        );
        println!(
            "  default is the per-hand best in {}/{} = {:.1}% of hands; per-hand regret \
             {:+.2} pts/hand (UPPER bound — max over 3 noisy playouts)",
            default_is_best,
            completed,
            default_is_best as f64 / completed as f64 * 100.0,
            default_regret_sum as f64 / completed as f64,
        );
        println!(
            "  default burial differs from enoch's in {}/{} = {:.1}% of hands \
             (see the points-buried column — Enoch's discipline burying POINTS is itself a finding)",
            default_ne_enoch,
            completed,
            default_ne_enoch as f64 / completed as f64 * 100.0,
        );
        let verdict = if mean_gap >= 3.0 {
            "LARGE leak — an alternative beats default on average; a learned kitty model (Phase-2) is worth building."
        } else if mean_gap >= 1.0 {
            "MODEST leak — a kitty model may help; weigh against the effort."
        } else {
            "NO clear leak vs these alternatives — default has the best (or ~tied) MEAN burial; deprioritize a kitty model and instead investigate the per-strategy points-buried column."
        };
        println!("\nVerdict (on the noise-robust MEAN gap): {verdict}");
    }
    println!("\nElapsed: {:.1}s", start.elapsed().as_secs_f64());
}
