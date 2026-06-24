//! Heuristic Shengji policy: the backbone used directly by Easy/Medium and as
//! the rollout / leaf policy inside Hard's determinized search.
//!
//! Everything here is computed from the redacted per-player view only.
//!
//! The core abstraction is a *scoring over legal candidate moves*: we generate
//! a small set of sensible candidate plays (using the engine's legal-move
//! generators), score each with Shengji strategy heuristics, and return them
//! ranked. Callers (the difficulty tiers) then pick from the ranking with
//! tier-specific randomness.

use std::collections::HashMap;

use shengji_mechanics::ordered_card::OrderedCard;
use shengji_mechanics::trick::{TractorRequirements, TrickUnit, UnitLike};
use shengji_mechanics::types::{Card, EffectiveSuit, Number, PlayerID, Rank, Suit, Trump};

use crate::game_state::play_phase::PlayPhase;
use crate::settings::FriendSelection;

/// A scored candidate play. Higher `score` is better.
#[derive(Clone, Debug)]
pub struct ScoredPlay {
    pub cards: Vec<Card>,
    pub score: f64,
}

/// The relative "strength rank" of a card within its effective suit, ignoring
/// suit identity. Bigger means stronger. Used to compare cards quickly.
fn card_strength(trump: Trump, card: Card) -> i32 {
    match card {
        Card::BigJoker => 1000,
        Card::SmallJoker => 999,
        Card::Unknown => -1,
        Card::Suited { number, suit } => {
            // Trump-number cards rank just under the jokers.
            if trump.number() == Some(number) {
                if Some(suit) == trump.suit() {
                    998 // trump-suit trump number
                } else {
                    997 // off-suit trump number
                }
            } else {
                number.as_u32() as i32
            }
        }
    }
}

fn is_trump(trump: Trump, card: Card) -> bool {
    trump.effective_suit(card) == EffectiveSuit::Trump
}

fn is_point(card: Card) -> bool {
    card.points().is_some()
}

/// Whether `me` and `other` are on the same team given the landlord's team.
pub fn same_team(p: &PlayPhase, me: PlayerID, other: PlayerID) -> bool {
    let team = p.landlords_team();
    team.contains(&me) == team.contains(&other)
}

/// Group a hand's cards by effective suit.
fn cards_by_suit(trump: Trump, cards: &[Card]) -> HashMap<EffectiveSuit, Vec<Card>> {
    let mut map: HashMap<EffectiveSuit, Vec<Card>> = HashMap::new();
    for &card in cards {
        map.entry(trump.effective_suit(card))
            .or_default()
            .push(card);
    }
    map
}

/// Generate candidate lead plays (single trick units, never throws), each a
/// legal lead. Mirrors `simulate_play`'s grouping but returns *all* unit
/// candidates rather than just the biggest, so the heuristic can choose.
pub fn lead_candidates(p: &PlayPhase, me: PlayerID) -> Vec<Vec<Card>> {
    let hand = match p.hands().get(me) {
        Ok(h) => h,
        Err(_) => return vec![],
    };
    let cards: Vec<Card> = Card::cards(hand.iter()).copied().collect();
    let trump = p.trick().trump();

    let mut candidates: Vec<Vec<Card>> = vec![];
    for (_, suit_cards) in cards_by_suit(trump, &cards) {
        let results = TrickUnit::find_plays(trump, TractorRequirements::default(), suit_cards);
        for play in results {
            // Each `play` is a full grouping (Units) of this suit. We want each
            // individual unit as a candidate lead.
            for unit in play {
                candidates.push(unit.cards());
            }
        }
    }
    // Deduplicate identical card-sets.
    candidates.sort_by(|a, b| {
        a.len()
            .cmp(&b.len())
            .then_with(|| format!("{a:?}").cmp(&format!("{b:?}")))
    });
    candidates.dedup();

    // Keep only plays the engine accepts as legal. A lead of a single unit is
    // always legal, but be defensive against any edge cases.
    candidates.retain(|c| p.can_play_cards(me, c).is_ok());
    if candidates.is_empty() {
        // Guaranteed-legal fallback: lead a single (lowest) card.
        if let Some(card) = cards.iter().min_by(|a, b| trump.compare(**a, **b)) {
            candidates.push(vec![*card]);
        }
    }
    candidates
}

/// Generate candidate follow plays for the current trick format. Returns a set
/// of distinct legal follows (length-correct). Always includes at least one
/// legal play if the hand is non-empty.
pub fn follow_candidates(p: &PlayPhase, me: PlayerID) -> Vec<Vec<Card>> {
    let hand = match p.hands().get(me) {
        Ok(h) => h.clone(),
        Err(_) => return vec![],
    };
    let trick_format = match p.trick().trick_format() {
        Some(tf) => tf.clone(),
        None => return vec![],
    };
    let trump = trick_format.trump();
    let num_required = trick_format.size();

    let available_cards: Vec<Card> = Card::cards(
        hand.iter()
            .filter(|(c, _)| trump.effective_suit(**c) == trick_format.suit()),
    )
    .copied()
    .collect();

    let mut candidates: Vec<Vec<Card>> = vec![];

    // Format-matching plays (the "correct" structural follows).
    for format in trick_format.decomposition(p.propagated().trick_draw_policy()) {
        let mut playable = UnitLike::check_play(
            OrderedCard::make_map(available_cards.iter().copied(), trump),
            format.iter().cloned(),
            p.propagated().trick_draw_policy(),
        );
        if let Some(u) = playable.next() {
            let matched: Vec<Card> = u
                .into_iter()
                .flat_map(|x| {
                    x.into_iter()
                        .flat_map(|(card, count)| std::iter::repeat_n(card.card, count))
                })
                .collect();
            // Top up if the matched format is shorter than required.
            let play = top_up(
                &matched,
                &available_cards,
                &hand,
                trump,
                trick_format.suit(),
                num_required,
            );
            if play.len() == num_required {
                candidates.push(play);
            }
        }
    }

    // If we have enough in-suit cards, also offer a few "which same-suit cards"
    // variants: play the lowest non-points, and (separately) play points. These
    // give the scorer meaningful choices when discarding within the led suit.
    if available_cards.len() >= num_required {
        let mut low_first = available_cards.clone();
        low_first.sort_by_key(|c| card_strength(trump, *c));
        candidates.push(low_first.iter().take(num_required).copied().collect());

        let mut high_first = available_cards.clone();
        high_first.sort_by_key(|c| std::cmp::Reverse(card_strength(trump, *c)));
        candidates.push(high_first.iter().take(num_required).copied().collect());
    } else {
        // We are short / void in the led suit. We must play all available in-suit
        // cards, then fill with off-suit. Offer variants of the off-suit fill:
        // dump low non-trump non-points, dump points, or trump in.
        let off_suit: Vec<Card> = Card::cards(
            hand.iter()
                .filter(|(c, _)| trump.effective_suit(**c) != trick_format.suit()),
        )
        .copied()
        .collect();
        let need = num_required.saturating_sub(available_cards.len());

        if need > 0 && !off_suit.is_empty() {
            // Variant A: discard the weakest non-trump non-point cards.
            let mut weak = off_suit.clone();
            weak.sort_by_key(|c| fill_discard_key(trump, *c));
            let mut va = available_cards.clone();
            va.extend(weak.iter().take(need).copied());
            if va.len() == num_required {
                candidates.push(va);
            }

            // Variant B: trump in with the lowest trumps (capture attempt).
            let mut trumps: Vec<Card> = off_suit
                .iter()
                .copied()
                .filter(|c| is_trump(trump, *c))
                .collect();
            if trumps.len() >= need {
                trumps.sort_by_key(|c| card_strength(trump, *c));
                let mut vb = available_cards.clone();
                vb.extend(trumps.iter().take(need).copied());
                if vb.len() == num_required {
                    candidates.push(vb);
                }
            }
        } else if need == 0 {
            candidates.push(available_cards.clone());
        }
    }

    // Deduplicate.
    for c in candidates.iter_mut() {
        c.sort_by(|a, b| trump.compare(*a, *b));
    }
    candidates.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
    candidates.dedup();

    // Keep only plays the engine accepts as legal: our heuristic variants can
    // occasionally violate suit-following / tuple-protection rules, and the
    // engine is the source of truth. The deterministic `simple_follow` is a
    // guaranteed-legal fallback if everything else is filtered out.
    candidates.retain(|c| p.can_play_cards(me, c).is_ok());
    if candidates.is_empty() {
        let fallback = simple_follow(
            &available_cards,
            &hand,
            trump,
            trick_format.suit(),
            num_required,
        );
        candidates.push(fallback);
    }
    candidates
}

/// Sort key for choosing off-suit fill/discard cards: prefer to throw away
/// non-trump, non-point, low cards first. Lower key = discarded first.
fn fill_discard_key(trump: Trump, card: Card) -> i32 {
    let mut key = card_strength(trump, card);
    if is_trump(trump, card) {
        key += 2000; // strongly avoid discarding trumps
    }
    if is_point(card) {
        key += 1000; // avoid handing points to opponents
    }
    key
}

fn top_up(
    matched: &[Card],
    available_cards: &[Card],
    hand: &HashMap<Card, usize>,
    trump: Trump,
    suit: EffectiveSuit,
    num_required: usize,
) -> Vec<Card> {
    if matched.len() == num_required {
        return matched.to_vec();
    }
    if num_required >= available_cards.len() {
        // We must play all in-suit cards, plus off-suit fill.
        return simple_follow(available_cards, hand, trump, suit, num_required);
    }
    let mut play = matched.to_vec();
    let mut remaining = available_cards.to_vec();
    for m in matched {
        if let Some(pos) = remaining.iter().position(|c| *c == *m) {
            remaining.remove(pos);
        }
    }
    // Prefer to keep strong in-suit cards; top up with the weakest.
    remaining.sort_by_key(|c| card_strength(trump, *c));
    let needed = num_required - play.len();
    play.extend(remaining.into_iter().take(needed));
    play
}

/// Deterministic always-legal follow (mirrors the dumb policy), used as a
/// fallback so the bot never produces an illegal/empty play.
pub fn simple_follow(
    available_cards: &[Card],
    hand: &HashMap<Card, usize>,
    trump: Trump,
    suit: EffectiveSuit,
    num_required: usize,
) -> Vec<Card> {
    let mut play: Vec<Card> = if available_cards.len() >= num_required {
        available_cards.iter().take(num_required).copied().collect()
    } else {
        available_cards.to_vec()
    };
    let need = num_required.saturating_sub(play.len());
    if need > 0 {
        let mut off: Vec<Card> = Card::cards(
            hand.iter()
                .filter(|(c, _)| trump.effective_suit(**c) != suit),
        )
        .copied()
        .collect();
        // Discard the weakest off-suit non-points first.
        off.sort_by_key(|c| fill_discard_key(trump, *c));
        play.extend(off.into_iter().take(need));
    }
    play
}

/// Score a candidate lead. Encodes leading strategy:
/// * Lead strong winners early (Aces, jokers, tractors) to draw out opponents.
/// * Avoid leading points to opponents.
/// * Conserve trumps; prefer to lead side-suit strength first.
pub fn score_lead(p: &PlayPhase, _me: PlayerID, cards: &[Card]) -> f64 {
    if cards.is_empty() {
        return f64::NEG_INFINITY;
    }
    let trump = p.trick().trump();
    let len = cards.len() as f64;
    let lead = cards[0];
    let trumping = is_trump(trump, lead);

    let max_strength = cards
        .iter()
        .map(|c| card_strength(trump, *c))
        .max()
        .unwrap_or(0) as f64;
    let point_total: i32 = cards
        .iter()
        .filter_map(|c| c.points().map(|x| x as i32))
        .sum();

    let mut score = 0.0;

    // Prefer leading multi-card units (pairs / tractors) — they pressure
    // opponents and are hard to beat.
    score += (len - 1.0) * 6.0;

    // Leading a strong card (Ace, big card) is good: it tends to win the trick
    // and pull out opponents' high cards. Scale strength modestly.
    score += max_strength * 0.05;

    // Strongly reward leading a guaranteed/near-guaranteed winner (Aces of a
    // side suit, big jokers, trump-number cards).
    if max_strength >= 990.0 {
        score += 8.0;
    } else if max_strength >= 13.0 {
        // King/Ace of a side suit.
        score += 4.0;
    }

    // Penalize giving points away in a lead (we don't yet know who wins).
    score -= point_total as f64 * 1.2;

    // Conserve trump: prefer NOT to lead trump early unless it's a strong unit.
    if trumping {
        if len >= 2.0 && max_strength >= 990.0 {
            // Leading a big trump pair (e.g. joker pair) is a powerful play.
            score += 2.0;
        } else {
            score -= 5.0;
        }
    }

    score
}

/// Score a candidate follow. Encodes following strategy relative to the current
/// trick winner:
/// * Contribute points (10/K/5) when our TEAM is currently winning.
/// * Duck / under-play (throw low) when we can't win and our team isn't winning.
/// * Trump in to capture a valuable (point-rich) trick when void.
/// * Don't trump a trick our partner is already winning; don't waste high cards.
pub fn score_follow(p: &PlayPhase, me: PlayerID, cards: &[Card]) -> f64 {
    if cards.is_empty() {
        return f64::NEG_INFINITY;
    }
    let trump = p.trick().trump();
    let trick = p.trick();

    // Who is currently winning, and are they our teammate?
    let current_winner = trick.winner_so_far();
    let team_winning = current_winner.map(|w| same_team(p, me, w)).unwrap_or(false);

    // Is this the last seat to play in the trick? (We then know the outcome.)
    let players_left = trick.player_queue().count();
    let is_last_to_act = players_left <= 1;

    // Points currently committed to the trick (on the table).
    let pot_points: i32 = trick
        .played_cards()
        .iter()
        .flat_map(|pc| pc.cards.iter())
        .filter_map(|c| c.points().map(|x| x as i32))
        .sum();

    let trick_format_suit = trick.trick_format().map(|tf| tf.suit());
    let following_suit = trick_format_suit
        .map(|s| cards.iter().all(|c| trump.effective_suit(*c) == s))
        .unwrap_or(false);

    let my_point_contribution: i32 = cards
        .iter()
        .filter_map(|c| c.points().map(|x| x as i32))
        .sum();
    let max_strength = cards
        .iter()
        .map(|c| card_strength(trump, *c))
        .max()
        .unwrap_or(0);
    let trumping_in = !following_suit && cards.iter().any(|c| is_trump(trump, *c));

    // Estimate whether THIS play can beat the current winner. We can only beat
    // it by following suit with a higher unit, or by trumping in. This is a
    // rough estimate (the engine resolves the real winner): treat a higher
    // top-card in the led suit, or any trump-in over a non-trump winner, as a
    // likely win.
    let winner_is_trump = current_winner
        .and_then(|w| {
            trick
                .played_cards()
                .iter()
                .find(|pc| pc.id == w)
                .and_then(|pc| pc.cards.first().copied())
        })
        .map(|c| is_trump(trump, c))
        .unwrap_or(false);
    let winner_top_strength = current_winner
        .and_then(|w| {
            trick.played_cards().iter().find(|pc| pc.id == w).map(|pc| {
                pc.cards
                    .iter()
                    .map(|c| card_strength(trump, *c))
                    .max()
                    .unwrap_or(0)
            })
        })
        .unwrap_or(0);

    let likely_win = if following_suit {
        max_strength > winner_top_strength && !winner_is_trump
            || (following_suit && current_winner.is_none())
    } else if trumping_in {
        // Trump-in beats a non-trump winner; against a trump winner only if our
        // trump is stronger.
        if winner_is_trump {
            max_strength > winner_top_strength
        } else {
            true
        }
    } else {
        false
    };

    let mut score = 0.0;

    if team_winning {
        // Our team is winning. Feed points if we have them, but don't waste
        // high cards or trumps doing it.
        score += my_point_contribution as f64 * 2.5;
        // Avoid over-trumping our own partner.
        if trumping_in {
            score -= 8.0;
        }
        // Prefer to play low non-point cards otherwise (save strength).
        score -= max_strength as f64 * 0.02;
    } else {
        // Opponents are winning (or trick just led). Decide whether to fight.
        if likely_win {
            // Winning is valuable, especially with points in the pot.
            score += 6.0 + pot_points as f64 * 0.8;
            // If we win, contributing our own points is fine (they come back to
            // our team), but prefer to win cheaply: small penalty for spending
            // more strength than needed.
            score -= (max_strength as f64) * 0.01;
            if trumping_in {
                // Trumping in to capture points is great; bare-trumping an empty
                // pot wastes a trump.
                if pot_points > 0 {
                    score += 4.0;
                } else {
                    score -= 3.0;
                }
                // Use the cheapest trump that still wins.
                score -= max_strength as f64 * 0.03;
            }
        } else {
            // We can't (or shouldn't) win: duck. Throw the weakest cards and
            // NEVER hand over points to the opponents.
            score -= my_point_contribution as f64 * 3.0;
            score -= max_strength as f64 * 0.05;
            // Don't waste trumps when we can't win.
            if trumping_in {
                score -= 6.0;
            }
            // If we're the last to act and can't win, definitely dump trash.
            if is_last_to_act {
                score -= my_point_contribution as f64 * 1.0;
            }
        }
    }

    score
}

/// Rank the legal lead candidates by heuristic score (best first).
pub fn ranked_leads(p: &PlayPhase, me: PlayerID) -> Vec<ScoredPlay> {
    let mut scored: Vec<ScoredPlay> = lead_candidates(p, me)
        .into_iter()
        .map(|cards| {
            let score = score_lead(p, me, &cards);
            ScoredPlay { cards, score }
        })
        .collect();
    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored
}

/// Rank the legal follow candidates by heuristic score (best first).
pub fn ranked_follows(p: &PlayPhase, me: PlayerID) -> Vec<ScoredPlay> {
    let mut scored: Vec<ScoredPlay> = follow_candidates(p, me)
        .into_iter()
        .map(|cards| {
            let score = score_follow(p, me, &cards);
            ScoredPlay { cards, score }
        })
        .collect();
    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored
}

// ===========================================================================
// Bidding
// ===========================================================================

/// Evaluate a hand's trump potential for a given candidate trump, returning a
/// rough "bid strength" score. Encodes: long + paired trump-suit holdings are
/// good, trump-rank cards and jokers are valuable, NT is viable with jokers.
pub fn bid_strength(hand: &[Card], candidate: Trump) -> f64 {
    let mut score = 0.0;
    let counts = Card::count(hand.iter().copied());

    // Count effective-trump cards and their strength.
    let mut trump_count = 0;
    for (&card, &ct) in &counts {
        if candidate.effective_suit(card) == EffectiveSuit::Trump {
            trump_count += ct;
            let s = card_strength(candidate, card);
            score += (s as f64).min(20.0) * 0.3 * ct as f64;
        }
    }
    // A long trump holding is strong.
    score += trump_count as f64 * 1.5;

    // Jokers are independently valuable (and enable NT bids).
    let big = counts.get(&Card::BigJoker).copied().unwrap_or(0);
    let small = counts.get(&Card::SmallJoker).copied().unwrap_or(0);
    score += big as f64 * 3.0 + small as f64 * 2.0;

    // Side-suit Aces/Kings provide control.
    for (&card, &ct) in &counts {
        if candidate.effective_suit(card) != EffectiveSuit::Trump {
            if let Card::Suited { number, .. } = card {
                if number == Number::Ace {
                    score += 1.5 * ct as f64;
                } else if number == Number::King {
                    score += 0.7 * ct as f64;
                }
            }
        }
    }

    score
}

/// Whether the hand is strong enough to justify making the given bid (rather
/// than passing / letting someone else take it). Threshold tuned so we don't
/// overbid a weak hand.
pub fn should_bid(hand: &[Card], candidate: Trump, current_best: f64) -> bool {
    let strength = bid_strength(hand, candidate);
    // Only bid if our hand is meaningfully strong in this trump.
    strength >= 10.0 && strength > current_best + 1.0
}

// ===========================================================================
// Kitty burying (landlord exchange)
// ===========================================================================

/// Choose `kitty_size` cards to bury from the hand. Encodes kitty discipline:
/// NEVER bury point cards (5/10/K) unless unavoidable; prefer to VOID a short
/// side suit (so the landlord can trump it later) and bury low non-point cards.
///
/// Returns the cards to move from hand to kitty.
pub fn choose_kitty(hand: &[Card], trump: Trump, kitty_size: usize) -> Vec<Card> {
    if kitty_size == 0 {
        return vec![];
    }
    // Score each card for "buriability": higher = better to bury.
    // We never want to bury points or strong cards. We DO want to bury cards
    // from short side suits to create voids.
    let by_suit = cards_by_suit(trump, hand);
    let suit_len: HashMap<EffectiveSuit, usize> =
        by_suit.iter().map(|(s, c)| (*s, c.len())).collect();

    let mut scored: Vec<(f64, Card)> = hand
        .iter()
        .map(|&card| {
            let suit = trump.effective_suit(card);
            let strength = card_strength(trump, card);
            let mut bury = 0.0;

            // Strongly avoid burying points.
            if is_point(card) {
                bury -= 100.0;
            }
            // Avoid burying trumps (they win tricks) and very strong cards.
            if suit == EffectiveSuit::Trump {
                bury -= 50.0;
            }
            if strength >= 13 {
                bury -= 20.0; // Aces / Kings / Jokers
            }
            // Prefer to bury low cards.
            bury += (15 - strength.min(15)) as f64;
            // Bonus for burying from a SHORT side suit (creating a void enables
            // trumping that suit later).
            let len = suit_len.get(&suit).copied().unwrap_or(0);
            if suit != EffectiveSuit::Trump {
                // Shorter suits get a bigger "void me" bonus.
                bury += (6.0 - len as f64).max(0.0) * 2.0;
            }
            (bury, card)
        })
        .collect();

    // Sort by buriability descending, breaking ties deterministically by the
    // card's trump-ordering and its char value so the chosen burial set does NOT
    // depend on (nondeterministic) hand HashMap iteration order. This stability
    // is what lets the landlord's kitty reconciliation terminate.
    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| trump.compare(a.1, b.1))
            .then_with(|| a.1.as_char().cmp(&b.1.as_char()))
    });
    scored
        .into_iter()
        .take(kitty_size)
        .map(|(_, c)| c)
        .collect()
}

/// Choose legal friend selections for FindingFriends mode (used only as a
/// fallback; the UI is Tractor-only). Picks side-suit Aces.
pub fn choose_friends(trump: Trump, num_friends: usize) -> Vec<FriendSelection> {
    let mut viable = vec![];
    for suit in &[Suit::Clubs, Suit::Diamonds, Suit::Hearts, Suit::Spades] {
        let c = Card::Suited {
            number: Number::Ace,
            suit: *suit,
        };
        if trump.effective_suit(c) != EffectiveSuit::Trump {
            viable.push(FriendSelection {
                card: c,
                initial_skip: 0,
            });
        }
    }
    viable.into_iter().take(num_friends).collect()
}

/// Convenience: the trump that the level-rank + a candidate suit would create.
pub fn trump_for(level: Rank, suit: Option<Suit>) -> Trump {
    match (level, suit) {
        (Rank::NoTrump, _) => Trump::NoTrump { number: None },
        (Rank::Number(n), Some(s)) => Trump::Standard { suit: s, number: n },
        (Rank::Number(n), None) => Trump::NoTrump { number: Some(n) },
    }
}
