//! Decision-level quality metrics — a DENSE signal (one datum per decision, not one
//! bit per hand) that resolves the rare-situation play fixes aggregate win-rate
//! cannot. It drives [`play_one_hand_instrumented`] and, at every PLAY decision,
//! asks BOTH scorer versions what they would play ON THE SAME STATE, then classifies
//! each against the targeted behaviors:
//!
//! * `lost-trick WASTE` — on a trick it cannot get on top of, burned a PREMIUM card
//!   (joker / trump-rank / high trump / side-suit Ace) when a non-premium duck existed (#1/#4)
//! * `opponent-winning POINT-LEAK` — while the other team held the trick and no
//!   candidate could take it, shed points when a 0-point duck existed (#3)
//! * `ruff PAIR-FRAGMENTATION` — ruffing a singles lead, broke a trump pair when a
//!   non-pair-breaking winning ruff existed (#7)
//! * `failed TRUMP-SET` — selected an avoidable multi-unit trump throw which the
//!   mechanics engine rejects against the actual (benchmark-only) hidden hands
//! * `naked JOKER` — opened an empty trick with a lone joker when another legal
//!   lead existed
//! * `lost-trick ACE-WASTE` — burned a non-trump Ace on a trick no legal play
//!   could win when an Ace-free duck existed
//! * `ruffable PARTNER-FEED` — fed points under an apparent side-suit boss while a
//!   publicly known-void opponent was still to act and a zero-point follow existed
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

use std::cmp::Ordering;
use std::collections::HashMap;
use std::env;

use rand::rngs::StdRng;
use rand::SeedableRng;

use shengji_core::bot::determinize::Knowledge;
use shengji_core::bot::harness::{play_one_hand_instrumented, PlayBrain, Seat};
use shengji_core::bot::heuristics::{self, HeuristicVersion};
use shengji_core::bot::BotDifficulty;
use shengji_core::game_state::play_phase::PlayPhase;
use shengji_core::game_state::GameState;
use shengji_mechanics::trick::{TractorRequirements, TrickUnit};
use shengji_mechanics::types::{Card, EffectiveSuit, Number, PlayerID, Trump};

#[derive(Default)]
struct Metrics {
    decisions: usize,
    leads: usize,
    lost: usize,
    opponent_lost: usize,
    waste: usize,
    point_leak: usize,
    ruff_singles: usize,
    fragment: usize,
    trump_set_opportunities: usize,
    failed_trump_set: usize,
    naked_joker_opportunities: usize,
    naked_joker: usize,
    ace_waste_opportunities: usize,
    ace_waste: usize,
    ruffable_partner_feed_opportunities: usize,
    ruffable_partner_feed: usize,
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

fn is_nontrump_ace(trump: Trump, c: Card) -> bool {
    matches!(
        c,
        Card::Suited {
            number: Number::Ace,
            ..
        }
    ) && !is_trump(trump, c)
}

fn is_naked_joker(cards: &[Card]) -> bool {
    cards.len() == 1 && cards[0].is_joker()
}

/// A lead spanning multiple trick units in the trump effective suit. This is the
/// mechanically precise shape behind examples such as `low trump + Joker` and
/// `trump pair + Joker`; an actual failed throw is labelled separately by replaying
/// the action through the mechanics engine on the benchmark's full state.
fn is_multi_unit_trump_set(pp: &PlayPhase, cards: &[Card]) -> bool {
    if cards.len() < 2 || !cards.iter().all(|&card| is_trump(pp.trump(), card)) {
        return false;
    }
    !TrickUnit::find_plays(
        pp.trump(),
        // This example drives `HarnessConfig::default()`, whose table uses the
        // mechanics default. `PropagatedState` intentionally keeps this field
        // crate-private, so spell out the matching public default here.
        TractorRequirements::default(),
        cards.iter().copied(),
    )
    .into_iter()
    .any(|units| units.len() == 1)
}

/// Whether the full-information benchmark state proves this proposed throw would
/// be halted. The policy never sees `full`; it is used only as an objective label.
fn throw_fails(full: &PlayPhase, actor: PlayerID, cards: &[Card]) -> bool {
    let mut sim = full.clone();
    if sim.play_cards(actor, cards).is_err() {
        return false;
    }
    sim.trick()
        .played_cards()
        .last()
        .filter(|played| played.id == actor)
        .is_some_and(|played| !played.bad_throw_cards.is_empty())
}

/// Public-card-memory boss test for the card currently on top. This deliberately
/// means "no stronger same-effective-suit copy remains unseen", not "the whole
/// trick is locked": a known-void later opponent may still ruff it, which is the
/// exact risky-feed situation measured below.
fn is_apparent_boss(knowledge: &Knowledge, trump: Trump, card: Card) -> bool {
    let suit = trump.effective_suit(card);
    knowledge.configured_counts.iter().all(|(&other, &copies)| {
        trump.effective_suit(other) != suit
            || trump.compare_effective(card, other) != Ordering::Less
            || knowledge.seen.get(&other).copied().unwrap_or(0) >= copies
    })
}

fn ruffable_partner_boss(pp: &PlayPhase, actor: PlayerID, knowledge: &Knowledge) -> bool {
    let trick = pp.trick();
    let Some(format) = trick.trick_format() else {
        return false;
    };
    // Keep the label exact and interpretable: a single side-suit boss can be
    // overruffed, whereas compound formats require structure-specific reasoning.
    if format.size() != 1 || format.suit() == EffectiveSuit::Trump {
        return false;
    }
    let Some(winner) = trick.winner_so_far() else {
        return false;
    };
    if winner == actor || !heuristics::same_team(pp, actor, winner) {
        return false;
    }
    let Some(winner_card) = trick
        .played_cards()
        .iter()
        .find(|played| played.id == winner)
        .and_then(|played| played.cards.first().copied())
    else {
        return false;
    };
    if !is_apparent_boss(knowledge, pp.trump(), winner_card) {
        return false;
    }

    let mut after_actor = trick.player_queue();
    if after_actor.next() != Some(actor) {
        return false;
    }
    after_actor.any(|player| {
        !heuristics::same_team(pp, actor, player)
            && knowledge
                .voids
                .get(&player)
                .is_some_and(|voids| voids.contains(&format.suit()))
    })
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
    fn observe(&mut self, pp: &PlayPhase, full: &PlayPhase, actor: PlayerID, chosen: &[Card]) {
        self.decisions += 1;
        let tf = match pp.trick().trick_format() {
            Some(tf) => tf.clone(),
            None => {
                self.leads += 1;
                let candidates = heuristics::lead_candidates(pp, actor);
                let trump_set_available = candidates
                    .iter()
                    .any(|candidate| is_multi_unit_trump_set(pp, candidate));
                let atomic_alternative = candidates
                    .iter()
                    .any(|candidate| !is_multi_unit_trump_set(pp, candidate));
                if trump_set_available && atomic_alternative {
                    self.trump_set_opportunities += 1;
                    if is_multi_unit_trump_set(pp, chosen) && throw_fails(full, actor, chosen) {
                        self.failed_trump_set += 1;
                    }
                }

                let naked_joker_available =
                    candidates.iter().any(|candidate| is_naked_joker(candidate));
                let nonjoker_alternative = candidates
                    .iter()
                    .any(|candidate| !candidate.iter().any(|card| card.is_joker()));
                if naked_joker_available && nonjoker_alternative {
                    self.naked_joker_opportunities += 1;
                    if is_naked_joker(chosen) {
                        self.naked_joker += 1;
                    }
                }
                return;
            }
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
            let partner_winning = pp
                .trick()
                .winner_so_far()
                .is_some_and(|winner| heuristics::same_team(pp, actor, winner));
            let ace_candidate_exists = candidates
                .iter()
                .any(|candidate| candidate.iter().any(|&card| is_nontrump_ace(trump, card)));
            let ace_free_duck_exists = candidates.iter().any(|candidate| {
                points_of(candidate) == 0
                    && !candidate.iter().any(|&card| is_nontrump_ace(trump, card))
            });
            if ace_candidate_exists && ace_free_duck_exists {
                self.ace_waste_opportunities += 1;
                if chosen.iter().any(|&card| is_nontrump_ace(trump, card)) {
                    self.ace_waste += 1;
                }
            }
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
            if !partner_winning {
                self.opponent_lost += 1;
                if points_of(chosen) > 0 && candidates.iter().any(|c| points_of(c) == 0) {
                    self.point_leak += 1;
                }
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

        let knowledge = Knowledge::from_play_view(pp, actor);
        if ruffable_partner_boss(pp, actor, &knowledge) {
            let follows_suit = |candidate: &[Card]| {
                candidate
                    .iter()
                    .all(|&card| trump.effective_suit(card) == tf.suit())
            };
            let point_feed_exists = candidates
                .iter()
                .any(|candidate| follows_suit(candidate) && points_of(candidate) > 0);
            let zero_point_follow_exists = candidates
                .iter()
                .any(|candidate| follows_suit(candidate) && points_of(candidate) == 0);
            if point_feed_exists && zero_point_follow_exists {
                self.ruffable_partner_feed_opportunities += 1;
                if follows_suit(chosen) && points_of(chosen) > 0 {
                    self.ruffable_partner_feed += 1;
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
        pct(m.point_leak, m.opponent_lost),
        m.point_leak,
        m.opponent_lost,
        pct(m.fragment, m.ruff_singles),
        m.fragment,
        m.ruff_singles,
    );
    println!(
        "  {label:<12}  TRUMP-SET {} ({}/{})   NAKED-JOKER {} ({}/{})   ACE-WASTE {} ({}/{})   RUFFABLE-FEED {} ({}/{})",
        pct(m.failed_trump_set, m.trump_set_opportunities),
        m.failed_trump_set,
        m.trump_set_opportunities,
        pct(m.naked_joker, m.naked_joker_opportunities),
        m.naked_joker,
        m.naked_joker_opportunities,
        pct(m.ace_waste, m.ace_waste_opportunities),
        m.ace_waste,
        m.ace_waste_opportunities,
        pct(
            m.ruffable_partner_feed,
            m.ruffable_partner_feed_opportunities
        ),
        m.ruffable_partner_feed,
        m.ruffable_partner_feed_opportunities,
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
                    new_m.observe(pp, s, actor, &np);
                }
                if let Some(lp) = play_of(s, actor, HeuristicVersion::Legacy) {
                    legacy_m.observe(pp, s, actor, &lp);
                }
            }
        });
    }
    println!("(rate (blunders/eligible-decisions); denominators match across versions)");
    report("Heur@New", &new_m);
    report("Heur@Legacy", &legacy_m);
}
