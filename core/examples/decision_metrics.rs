//! Decision-level quality metrics — a DENSE signal (one datum per decision, not one
//! bit per hand) that resolves the rare-situation play fixes aggregate win-rate
//! cannot. It drives [`play_one_hand_instrumented`] and, at every PLAY decision,
//! asks BOTH scorer versions what they would play ON THE SAME STATE, then classifies
//! each against the targeted behaviors:
//!
//! * `lost-trick WASTE` — on a trick it cannot get on top of, burned a PREMIUM card
//!   (joker / trump-rank / high trump / side-suit Ace) when a non-premium duck existed (#1/#4)
//! * `lost-trick POINT-LEAK` — on such a trick, shed points when a 0-point duck existed (#3)
//! * `ruff PAIR-FRAGMENTATION` — ruffing a singles lead, broke a trump pair when a
//!   non-pair-breaking winning ruff existed (#7)
//!
//! Crucially it is a COMMON-STATE comparison: the game is advanced by ONE roller
//! policy, and at each decision both `New` and `Legacy` are scored on the IDENTICAL
//! position. So the "lost-trick" denominators are the same for both, and a rate
//! difference is a pure decision-quality difference — not the two policies wandering
//! into different game states (the confound an on-policy comparison suffers). Each
//! rate's denominator is the count of tagged decisions (the statistical power:
//! tens of thousands per few hundred hands). Lower is better for all three.
//!
//! This is an analysis tool, not a bot: each version's move is computed from the
//! actor's REDACTED view (exactly as the bot would), so the honesty boundary holds;
//! the classification then reads that view's own-hand + public trick facts.
//!
//! Run with:
//!   cargo run --release --example decision_metrics -- [roller] [hands] [seed]
//!     roller ∈ {new, legacy, enoch}  (default: new) — whose play ADVANCES the game

use std::collections::HashMap;
use std::env;

use rand::rngs::StdRng;
use rand::SeedableRng;

use shengji_core::bot::harness::{play_one_hand_instrumented, PlayBrain, Seat};
use shengji_core::bot::heuristics::{self, HeuristicVersion};
use shengji_core::bot::BotDifficulty;
use shengji_core::game_state::play_phase::PlayPhase;
use shengji_core::game_state::GameState;
use shengji_mechanics::types::{Card, EffectiveSuit, Number, PlayerID, Trump};

#[derive(Default)]
struct Metrics {
    decisions: usize,
    lost: usize,
    waste: usize,
    point_leak: usize,
    ruff_singles: usize,
    fragment: usize,
}

/// A PREMIUM card — value we should not burn on a lost trick: a joker, the
/// trump-rank card, a high trump-suit card (J/Q/K/A), or a side-suit Ace.
fn is_premium(trump: Trump, c: Card) -> bool {
    match c {
        Card::BigJoker | Card::SmallJoker => true,
        Card::Suited { number, .. } => {
            if trump.effective_suit(c) == EffectiveSuit::Trump {
                trump.number() == Some(number)
                    || matches!(
                        number,
                        Number::Jack | Number::Queen | Number::King | Number::Ace
                    )
            } else {
                number == Number::Ace
            }
        }
        Card::Unknown => false,
    }
}

fn count_cards(cards: &[Card]) -> HashMap<Card, usize> {
    let mut m = HashMap::new();
    for &c in cards {
        *m.entry(c).or_insert(0) += 1;
    }
    m
}

fn points_of(cards: &[Card]) -> usize {
    cards.iter().filter_map(|c| c.points()).sum()
}

fn is_trump(trump: Trump, c: Card) -> bool {
    trump.effective_suit(c) == EffectiveSuit::Trump
}

/// Would playing `cards` put `actor` on top of the trick? Simulate on a clone — the
/// engine is the source of truth for the winner.
fn wins_with(s: &PlayPhase, actor: PlayerID, cards: &[Card]) -> bool {
    if s.can_play_cards(actor, cards).is_err() {
        return false;
    }
    let mut sim = s.clone();
    if sim.play_cards(actor, cards).is_err() {
        return false;
    }
    sim.trick().winner_so_far() == Some(actor)
}

/// Does playing `cards` reduce a trump PAIR `actor` holds to a lone singleton?
fn fragments_trump_pair(trump: Trump, hand: &HashMap<Card, usize>, cards: &[Card]) -> bool {
    for (&card, &pc) in &count_cards(cards) {
        if is_trump(trump, card) {
            let held = hand.get(&card).copied().unwrap_or(0);
            if (held / 2) > (held.saturating_sub(pc) / 2) {
                return true;
            }
        }
    }
    false
}

impl Metrics {
    /// Classify `chosen` (one version's move) on the redacted view `pp`.
    fn observe(&mut self, pp: &PlayPhase, actor: PlayerID, chosen: &[Card]) {
        self.decisions += 1;
        let tf = match pp.trick().trick_format() {
            Some(tf) => tf.clone(),
            None => return, // leading — different playbook, not scored here
        };
        let trump = pp.trick().trump();
        let hand: HashMap<Card, usize> = match pp.hands().get(actor) {
            Ok(h) => h
                .iter()
                .filter(|(c, _)| **c != Card::Unknown)
                .map(|(c, n)| (*c, *n))
                .collect(),
            Err(_) => return,
        };
        let candidates = heuristics::follow_candidates(pp, actor);
        if candidates.is_empty() {
            return;
        }
        let can_get_on_top = candidates.iter().any(|c| wins_with(pp, actor, c));

        if !can_get_on_top {
            self.lost += 1;
            let chosen_premium = chosen.iter().any(|&c| is_premium(trump, c));
            // A TRUE-JUNK duck (no premium card AND no points) was available — so
            // spending a premium card is genuine waste, NOT a forced premium-vs-point
            // trade (which the POINT-LEAK metric scores separately).
            let junk_exists = candidates
                .iter()
                .any(|c| !c.iter().any(|&x| is_premium(trump, x)) && points_of(c) == 0);
            if chosen_premium && junk_exists {
                self.waste += 1;
            }
            if points_of(chosen) > 0 && candidates.iter().any(|c| points_of(c) == 0) {
                self.point_leak += 1;
            }
        } else {
            let led_suit = tf.suit();
            let lead_all_singles = pp
                .trick()
                .played_cards()
                .first()
                .map(|pc| {
                    let c = count_cards(&pc.cards);
                    !c.is_empty() && c.values().all(|&n| n == 1)
                })
                .unwrap_or(false);
            let actor_void_in_led = led_suit != EffectiveSuit::Trump
                && !hand.keys().any(|&c| trump.effective_suit(c) == led_suit);
            let chosen_is_ruff = led_suit != EffectiveSuit::Trump
                && !chosen.is_empty()
                && chosen.iter().all(|&c| is_trump(trump, c));
            if lead_all_singles && actor_void_in_led && chosen_is_ruff {
                self.ruff_singles += 1;
                let nonfrag_win_exists = candidates.iter().any(|c| {
                    c.iter().all(|&x| is_trump(trump, x))
                        && !fragments_trump_pair(trump, &hand, c)
                        && wins_with(pp, actor, c)
                });
                if fragments_trump_pair(trump, &hand, chosen) && nonfrag_win_exists {
                    self.fragment += 1;
                }
            }
        }
    }
}

fn report(label: &str, m: &Metrics) {
    let pct = |num: usize, den: usize| {
        if den == 0 {
            "  n/a".to_string()
        } else {
            format!("{:5.2}%", 100.0 * num as f64 / den as f64)
        }
    };
    println!(
        "  {label:<12}  WASTE {} ({}/{})   POINT-LEAK {} ({}/{})   FRAGMENT {} ({}/{})",
        pct(m.waste, m.lost),
        m.waste,
        m.lost,
        pct(m.point_leak, m.lost),
        m.point_leak,
        m.lost,
        pct(m.fragment, m.ruff_singles),
        m.fragment,
        m.ruff_singles,
    );
}

fn roller_seat(roller: &str) -> Seat {
    match roller {
        "legacy" => Seat {
            play: PlayBrain::HeuristicDirect(HeuristicVersion::Legacy),
            bid: BotDifficulty::Expert,
            kitty: BotDifficulty::Easy,
        },
        "enoch" => Seat::tier(BotDifficulty::Enoch),
        _ => Seat {
            play: PlayBrain::HeuristicDirect(HeuristicVersion::New),
            bid: BotDifficulty::Expert,
            kitty: BotDifficulty::Easy,
        },
    }
}

/// One version's move for `actor`, from the actor's REDACTED view (as a bot would).
fn play_of(pp_full: &PlayPhase, actor: PlayerID, version: HeuristicVersion) -> Option<Vec<Card>> {
    let view = GameState::Play(pp_full.clone()).for_player(actor);
    match &view {
        GameState::Play(pp) => heuristics::choose_play_direct(pp, actor, version),
        _ => None,
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let roller = args.get(1).map(|s| s.as_str()).unwrap_or("new");
    let hands: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(400);
    let base_seed: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0x5EED);

    println!("DECISION-LEVEL METRICS — New vs Legacy on COMMON states (lower is better)");
    println!("roller: {roller}   hands: {hands}   base_seed: {base_seed:#x}\n");

    let seats = [roller_seat(roller); 4];
    let mut new_m = Metrics::default();
    let mut legacy_m = Metrics::default();
    for g in 0..hands {
        let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(g as u64));
        let _ = play_one_hand_instrumented(&seats, &mut rng, &mut |s, actor, _chosen| {
            // Score BOTH versions on the identical state (the actor's redacted view).
            let view = GameState::Play(s.clone()).for_player(actor);
            if let GameState::Play(pp) = &view {
                if let Some(np) = play_of(s, actor, HeuristicVersion::New) {
                    new_m.observe(pp, actor, &np);
                }
                if let Some(lp) = play_of(s, actor, HeuristicVersion::Legacy) {
                    legacy_m.observe(pp, actor, &lp);
                }
            }
        });
    }
    println!("(rate (blunders/eligible-decisions); denominators match across versions)");
    report("Heur@New", &new_m);
    report("Heur@Legacy", &legacy_m);
}
