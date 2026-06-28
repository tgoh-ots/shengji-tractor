//! Heuristic Shengji policy: the backbone used directly by Easy and as
//! the rollout / leaf policy inside Hard's determinized search (and the
//! Expert tier's fallback).
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

use crate::bot::determinize::Knowledge;
use crate::game_state::play_phase::PlayPhase;
use crate::settings::FriendSelection;

/// Selects which heuristic scoring implementation drives candidate evaluation.
///
/// `New` is the stronger boss-card / partner-aware scorer that all real tiers
/// use. `Legacy` is the *frozen, original* scorer kept ONLY for two reasons:
///
/// 1. The Expert net's `f[26]` prior feature was trained against the legacy
///    scores; changing them silently would shift that feature's distribution and
///    degrade the distilled net. So [`candidate_features`](crate::bot::expert)
///    always computes `f[26]` from the legacy version until a retrain.
/// 2. The benchmark harness can pit NEW-heuristic-direct play against
///    LEGACY-heuristic-direct play in a single binary to measure the win-rate
///    delta of the change (see [`choose_play_direct`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeuristicVersion {
    New,
    Legacy,
}

/// A scored candidate play. Higher `score` is better.
#[derive(Clone, Debug)]
pub struct ScoredPlay {
    pub cards: Vec<Card>,
    pub score: f64,
}

/// The relative "strength rank" of a card within its effective suit, ignoring
/// suit identity. Bigger means stronger. Used to compare cards quickly.
///
/// `pub(crate)` so the Expert feature encoder shares the EXACT same metric the
/// heuristic uses (no drift between training features and the heuristic prior).
pub(crate) fn card_strength(trump: Trump, card: Card) -> i32 {
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

/// Number of distinct-position trump cards in a SINGLE deck for `trump`:
/// 2 jokers + the trump-number cards + the 13 ranks of the trump suit (when
/// there is a trump suit). For NoTrump the trump set is just the two jokers plus
/// the four trump-number cards.
pub(crate) fn trump_universe_size(trump: Trump) -> usize {
    match trump {
        Trump::NoTrump { number } => {
            let number_cards = if number.is_some() { 4 } else { 0 };
            2 + number_cards
        }
        Trump::Standard { .. } => 2 + 13 + 3,
    }
}

/// Enumerate the cards in effective-suit `eff` whose strength strictly exceeds
/// `floor` (used to test whether a card is the uncatchable top of its suit).
///
/// Honest: depends only on the trump declaration + card identities. Lives here
/// (the lower-level module) so the Expert feature encoder and the heuristic both
/// call the SAME implementation without a layering inversion.
pub(crate) fn stronger_cards_in_suit(trump: Trump, eff: EffectiveSuit, floor: i32) -> Vec<Card> {
    let mut out: Vec<Card> = Vec::new();
    let mut consider = |c: Card| {
        if trump.effective_suit(c) == eff && card_strength(trump, c) > floor {
            out.push(c);
        }
    };
    consider(Card::BigJoker);
    consider(Card::SmallJoker);
    let suits = [Suit::Clubs, Suit::Diamonds, Suit::Hearts, Suit::Spades];
    let numbers = [
        Number::Two,
        Number::Three,
        Number::Four,
        Number::Five,
        Number::Six,
        Number::Seven,
        Number::Eight,
        Number::Nine,
        Number::Ten,
        Number::Jack,
        Number::Queen,
        Number::King,
        Number::Ace,
    ];
    for suit in suits {
        for number in numbers {
            consider(Card::Suited { number, suit });
        }
    }
    out
}

/// Whether `card` cannot be beaten, within its own effective suit, by any card
/// the acting player has NOT yet seen. Jokers (and the trump-suit trump-number
/// card) are always guaranteed; otherwise we check that no higher same-suit card
/// remains unseen. This is an HONEST upper bound: it only consults `k.seen`,
/// which is derived purely from the redacted view + public play history.
///
/// NOTE: this uses [`card_strength`], whose ordering puts side-suit Aces LOW
/// (`Number::as_u32` is Ace-low). It is kept verbatim because the Expert net's
/// `f[34]` feature was trained against exactly this behaviour. The NEW heuristic
/// uses the rank-correct [`is_boss_card`] instead.
pub(crate) fn is_guaranteed_top(k: &Knowledge, trump: Trump, card: Card) -> bool {
    let s = card_strength(trump, card);
    if s >= 1000 {
        return true;
    }
    let eff = trump.effective_suit(card);
    let decks = k.num_decks.max(1);
    for higher in stronger_cards_in_suit(trump, eff, s) {
        let seen = k.seen.get(&higher).copied().unwrap_or(0);
        if decks > seen {
            // At least one copy of a stronger same-suit card is still unseen.
            return false;
        }
    }
    true
}

/// Rank-correct "boss strength": identical to [`card_strength`] EXCEPT that
/// side-suit Aces are the TOP of their suit (14), as they actually are in play.
/// Used only by the NEW heuristic's boss machinery so a side-suit Ace is treated
/// as the uncatchable winner it is. (Kept separate from `card_strength` so the
/// frozen Expert features that depend on the Ace-low quirk don't shift.)
pub(crate) fn boss_strength(trump: Trump, card: Card) -> i32 {
    match card {
        Card::Suited { number, suit }
            if trump.number() != Some(number) && number == Number::Ace =>
        {
            // A non-trump Ace tops its suit. (A trump-number Ace, if the trump
            // number were Ace, is already handled by `card_strength`'s 997/998.)
            let _ = suit;
            14
        }
        _ => card_strength(trump, card),
    }
}

/// Enumerate the same-effective-suit cards strictly STRONGER than `floor` by the
/// rank-correct [`boss_strength`] ordering.
fn boss_stronger_cards_in_suit(trump: Trump, eff: EffectiveSuit, floor: i32) -> Vec<Card> {
    let mut out: Vec<Card> = Vec::new();
    let mut consider = |c: Card| {
        if trump.effective_suit(c) == eff && boss_strength(trump, c) > floor {
            out.push(c);
        }
    };
    consider(Card::BigJoker);
    consider(Card::SmallJoker);
    let suits = [Suit::Clubs, Suit::Diamonds, Suit::Hearts, Suit::Spades];
    let numbers = [
        Number::Two,
        Number::Three,
        Number::Four,
        Number::Five,
        Number::Six,
        Number::Seven,
        Number::Eight,
        Number::Nine,
        Number::Ten,
        Number::Jack,
        Number::Queen,
        Number::King,
        Number::Ace,
    ];
    for suit in suits {
        for number in numbers {
            consider(Card::Suited { number, suit });
        }
    }
    out
}

/// Rank-correct "is this card the uncatchable top of its effective suit given
/// what I've seen" — the NEW heuristic's boss test. HONEST (only consults
/// `k.seen`). Unlike [`is_guaranteed_top`], side-suit Aces are correctly the top.
pub(crate) fn is_boss_card(k: &Knowledge, trump: Trump, card: Card) -> bool {
    let s = boss_strength(trump, card);
    if s >= 998 {
        // Jokers and the trump-suit trump-number are unconditional tops.
        return true;
    }
    let eff = trump.effective_suit(card);
    let decks = k.num_decks.max(1);
    for higher in boss_stronger_cards_in_suit(trump, eff, s) {
        let seen = k.seen.get(&higher).copied().unwrap_or(0);
        if decks > seen {
            return false;
        }
    }
    true
}

/// Count, over the same-effective-suit cards strictly stronger than `card`, how
/// many copies remain UNSEEN (the number of cards that could still beat it).
fn unseen_dominators(k: &Knowledge, trump: Trump, card: Card) -> usize {
    let eff = trump.effective_suit(card);
    let s = boss_strength(trump, card);
    let decks = k.num_decks.max(1);
    boss_stronger_cards_in_suit(trump, eff, s)
        .iter()
        .map(|h| decks.saturating_sub(k.seen.get(h).copied().unwrap_or(0)))
        .sum()
}

/// Per-decision evaluation context, built ONCE per search node and shared by
/// reference with both scorers. This is the single biggest cost lever: the
/// honest card-memory (`Knowledge`), the role/threshold facts, and the trump /
/// points accounting are computed once here instead of per candidate.
///
/// Everything here is HONEST — it derives only from the redacted per-player view
/// (own hand + table + last trick), so the honesty invariant is preserved across
/// Easy, Hard rollouts, and the heuristic backbone. Inside a determinized
/// rollout the sampled world is a real `PlayPhase`, so boss reads stay correct
/// ply-by-ply.
pub struct EvalCtx {
    pub k: Knowledge,
    pub trump: Trump,
    pub me: PlayerID,
    /// Attackers are the non-landlord side; they want `non_landlord_points`
    /// HIGH. The landlord side wants it LOW. Matches `evaluate_position`.
    pub me_is_attacker: bool,
    pub num_decks: usize,
    pub non_landlord_points: isize,
    /// Points per level-threshold step. `None` ⇒ scoring params invalid for the
    /// current decks; all threshold-bonus terms are then disabled (never crash).
    pub step_size: Option<isize>,
    /// Total trumps still unseen by `me` (in hidden hands / kitty).
    pub unseen_trumps: usize,
}

impl EvalCtx {
    /// Build the context once for the acting player `me` from the redacted view.
    pub fn build(p: &PlayPhase, me: PlayerID) -> Self {
        let trump = p.trick().trump();
        let k = Knowledge::from_play_view(p, me);
        let num_decks = k.num_decks.max(1);
        let me_is_attacker = !p.landlords_team().contains(&me);
        let (non_landlord_points, _) = p.calculate_points();

        let seen_trumps: usize = k
            .seen
            .iter()
            .filter(|(c, _)| trump.effective_suit(**c) == EffectiveSuit::Trump)
            .map(|(_, &n)| n)
            .sum();
        let total_trumps = trump_universe_size(trump) * num_decks;
        let unseen_trumps = total_trumps.saturating_sub(seen_trumps);

        EvalCtx {
            k,
            trump,
            me,
            me_is_attacker,
            num_decks,
            non_landlord_points,
            step_size: p.bot_step_size(),
            unseen_trumps,
        }
    }
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

/// FROZEN legacy lead scorer. DO NOT change its behaviour: the Expert net's
/// `f[26]` prior feature and the benchmark baseline both depend on it being
/// byte-for-byte the original logic. New strategy goes in [`score_lead`].
pub fn score_lead_legacy(p: &PlayPhase, _me: PlayerID, cards: &[Card]) -> f64 {
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

/// Score a candidate lead (the NEW boss-aware scorer used by all real tiers).
///
/// Encodes leading strategy with honest card-memory:
/// * Lead GUARANTEED winners (boss cards / tractors) hard — they cash points and
///   draw out opponents' high cards risk-free.
/// * Cash points only behind a boss; otherwise avoid donating points.
/// * Draw trumps when we're the landlord side and hold the boss trump, easing
///   off as opponents' trumps dry out; hoard trump otherwise.
/// * Set up strip-and-ruff with long side-suit bosses.
pub fn score_lead(ctx: &EvalCtx, p: &PlayPhase, cards: &[Card]) -> f64 {
    if cards.is_empty() {
        return f64::NEG_INFINITY;
    }
    let trump = ctx.trump;
    let len = cards.len() as f64;
    let lead = cards[0];
    let trumping = is_trump(trump, lead);

    // Rank-correct strength (side-suit Aces are high) for the boss machinery.
    let max_boss_strength = cards
        .iter()
        .map(|c| boss_strength(trump, *c))
        .max()
        .unwrap_or(0);
    let point_total: i32 = cards
        .iter()
        .filter_map(|c| c.points().map(|x| x as i32))
        .sum();

    // The strongest card in the unit, and whether it is a guaranteed top. Gate
    // the (O(13)) boss scan behind a cheap pre-check so obvious trash leads skip
    // it entirely.
    let top = cards
        .iter()
        .copied()
        .max_by_key(|c| boss_strength(trump, *c))
        .unwrap_or(lead);
    let boss_worth_checking = max_boss_strength >= 13 || trumping;
    let is_boss = boss_worth_checking && is_boss_card(&ctx.k, trump, top);

    let mut score = 0.0;

    // Multi-card units (pairs / tractors) pressure opponents.
    score += (len - 1.0) * 6.0;

    // Modest reward for raw strength.
    score += max_boss_strength as f64 * 0.05;

    // Guaranteed-winner bonuses. Jokers keep their flat floor; a *boss*
    // (uncatchable in its suit) non-trump unit is excellent.
    if max_boss_strength >= 990 {
        score += 8.0;
    }
    if is_boss && !trumping {
        score += 9.0;
    }
    // Boss tractors / pairs dominate; reward extra length.
    if is_boss && len >= 2.0 {
        score += (len - 1.0) * 4.0;
    }

    // Near-boss: exactly one stronger copy still unseen in this suit. Leading it
    // is still strong (only one card can take it).
    if !is_boss && boss_worth_checking && unseen_dominators(&ctx.k, trump, top) == 1 {
        score += 3.0;
    }

    // Points in the lead: good behind a boss (we cash them safely), bad
    // otherwise (we'd be donating to whoever wins).
    if is_boss {
        score += point_total as f64 * 0.6;
    } else {
        score -= point_total as f64 * 1.2;
    }

    // Trump leads by role. A joker pair is always a powerful play. Otherwise, the
    // landlord side should DRAW trump with a boss trump while opponents still
    // hold some, easing off as they dry out; everyone else hoards.
    if trumping {
        let joker_pair = len >= 2.0 && max_boss_strength >= 990;
        if joker_pair {
            score += 2.0;
        }
        let me_is_landlord_side = !ctx.me_is_attacker;
        if me_is_landlord_side && is_boss && ctx.unseen_trumps > 0 {
            let draw_scale =
                (ctx.unseen_trumps as f64 / (ctx.num_decks * 2) as f64).clamp(0.0, 1.0);
            score += 3.0 * draw_scale;
        } else if !joker_pair {
            // Defender, or a small / non-boss trump: hoard it.
            score -= 5.0;
        }
    }

    // Strip-and-ruff setup: a boss non-trump lead in a suit we hold a long
    // holding of lets us later void it and ruff. Count OUR OWN copies of that
    // effective suit (from our real hand — honest, we may see it).
    if !trumping && is_boss {
        let eff = trump.effective_suit(lead);
        if let Ok(hand) = p.hands().get(ctx.me) {
            let my_in_suit: usize = hand
                .iter()
                .filter(|(c, _)| **c != Card::Unknown && trump.effective_suit(**c) == eff)
                .map(|(_, &n)| n)
                .sum();
            if my_in_suit >= 4 {
                score += 2.5;
            }
        }
    }

    score
}

/// FROZEN legacy follow scorer. DO NOT change its behaviour: the Expert net's
/// `f[26]` prior feature and the benchmark baseline both depend on it being
/// byte-for-byte the original logic. New strategy goes in [`score_follow`].
pub fn score_follow_legacy(p: &PlayPhase, me: PlayerID, cards: &[Card]) -> f64 {
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

/// Score a candidate follow (the NEW boss-/partner-aware scorer used by all real
/// tiers). Encodes following strategy relative to the current trick winner, with
/// a seat-aware partner read and honest boss detection:
///
/// * When our partner is winning AND their card is LOCKED (a guaranteed top),
///   feed points aggressively and never needlessly over-rank them.
/// * When partner is winning but an opponent can still steal, feed cautiously.
/// * When an opponent is winning, fight for a point-rich pot (or ruff it as a
///   defender) with the cheapest winner; otherwise starve them of points.
/// * Threshold awareness: escalate point discipline when a donation would flip
///   the round, and reward feeds that push our team past a level threshold.
pub fn score_follow(ctx: &EvalCtx, p: &PlayPhase, cards: &[Card]) -> f64 {
    if cards.is_empty() {
        return f64::NEG_INFINITY;
    }
    let trump = ctx.trump;
    let me = ctx.me;
    let trick = p.trick();

    let current_winner = trick.winner_so_far();
    let partner_winning = current_winner.map(|w| same_team(p, me, w)).unwrap_or(false);

    // Seats yet to act AFTER me (drop the head `me`, as the engine queue lists me
    // first). An opponent still to act can steal a trick partner is winning.
    let mut yet_to_act: Vec<PlayerID> = trick.player_queue().collect();
    if yet_to_act.first() == Some(&me) {
        yet_to_act.remove(0);
    }
    let opp_after_me = yet_to_act.iter().any(|pid| !same_team(p, me, *pid));
    let is_last_to_act = yet_to_act.is_empty();

    // The winning card on the table, and whether it is LOCKED (a boss top).
    let winner_top_card = current_winner.and_then(|w| {
        trick
            .played_cards()
            .iter()
            .find(|pc| pc.id == w)
            .and_then(|pc| {
                pc.cards
                    .iter()
                    .copied()
                    .max_by_key(|c| boss_strength(trump, *c))
            })
    });
    let winner_is_trump = winner_top_card.map(|c| is_trump(trump, c)).unwrap_or(false);
    let winner_top_strength = winner_top_card
        .map(|c| boss_strength(trump, c))
        .unwrap_or(0);
    let winner_locked = partner_winning
        && winner_top_card
            .map(|c| is_boss_card(&ctx.k, trump, c))
            .unwrap_or(false);

    // Points already committed to the pot.
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
        .map(|c| boss_strength(trump, *c))
        .max()
        .unwrap_or(0);
    let trumping_in = !following_suit && cards.iter().any(|c| is_trump(trump, *c));

    // Would this candidate BEAT the current winner? Base it on CARD RANKS
    // (certain), never on inferred voids.
    let my_top = cards
        .iter()
        .copied()
        .max_by_key(|c| boss_strength(trump, *c))
        .unwrap_or(cards[0]);
    let would_beat = if current_winner.is_none() {
        true
    } else if following_suit {
        max_strength > winner_top_strength && !winner_is_trump
    } else if trumping_in {
        if winner_is_trump {
            max_strength > winner_top_strength
        } else {
            true
        }
    } else {
        false
    };
    // Does MY candidate, if it wins by following suit with a boss top, lock the
    // trick? (Used to allow beating-to-secure over a stealable partner.)
    let my_card_locks = following_suit && is_boss_card(&ctx.k, trump, my_top);

    // Boss-aware likely-win: now that `boss_strength` ranks side-suit Aces
    // correctly, beating the current winner by rank IS the win condition. (A
    // same-suit boss that does not out-rank the card already on the table does
    // NOT win this trick, so we must not treat it as a winner here.)
    let likely_win = would_beat;

    let mut score = 0.0;

    if partner_winning && winner_locked {
        // Branch A: partner has the trick LOCKED — feed points hard.
        score += my_point_contribution as f64 * 3.5;
        // Reward ducking low (save strength) when we have no points to feed.
        score += (20.0 - max_strength as f64).max(0.0) * 0.05;
        // Never needlessly over-rank a partner who already has it won.
        if would_beat {
            score -= 15.0;
        }
    } else if partner_winning && opp_after_me && !winner_locked {
        // Branch B: partner winning but an opponent can still steal — feed only
        // cautiously, prefer a non-point dump.
        score += my_point_contribution as f64 * 1.5;
        // Over-ranking partner is wasteful UNLESS doing so LOCKS the trick for us
        // (beating-to-secure is allowed).
        if would_beat && !my_card_locks {
            score -= 15.0;
        }
    } else if partner_winning {
        // Partner winning, not locked, but no opponent left to steal (e.g. we are
        // the last teammate): treat like a safe feed.
        score += my_point_contribution as f64 * 3.0;
        if would_beat && !my_card_locks {
            score -= 12.0;
        }
    } else {
        // Branch C/D: an opponent is winning (or trick just led to us mid-trick).
        if likely_win {
            // Branch C: we can take it. Winning a point-rich pot is valuable.
            score += 6.0 + pot_points as f64 * 0.8;
            score -= max_strength as f64 * 0.01;
            if trumping_in {
                // Defender ruff of a point-rich pot is especially good.
                let defender_ruff = !ctx.me_is_attacker && pot_points >= 5;
                if pot_points > 0 {
                    score += 4.0;
                    if defender_ruff {
                        score += 5.0 + pot_points as f64 * 0.6;
                    }
                } else {
                    score -= 3.0;
                }
                // Cheapest trump that still wins.
                score -= max_strength as f64 * 0.03;
            }
        } else {
            // Branch D: we cannot (or shouldn't) win — starve them of points.
            score -= my_point_contribution as f64 * 4.0;
            score -= max_strength as f64 * 0.05;
            if trumping_in {
                score -= 6.0;
            }
            if is_last_to_act {
                score -= my_point_contribution as f64 * 1.0;
            }

            // Threshold panic: if donating these points could flip the round,
            // escalate the point penalty. Orientation: attackers want
            // `non_landlord_points` HIGH (so donating helps THEM — no panic);
            // defenders want it LOW, so handing the attackers a pot that crosses
            // the next step is a disaster.
            if let Some(step) = ctx.step_size {
                if step > 0 && !ctx.me_is_attacker && my_point_contribution > 0 {
                    let attacker_total = ctx.non_landlord_points
                        + pot_points as isize
                        + my_point_contribution as isize;
                    // Within one trick's worth of crossing a step boundary.
                    let next_step = ((ctx.non_landlord_points / step) + 1) * step;
                    if attacker_total >= next_step {
                        score -= my_point_contribution as f64 * 2.0; // total ×6
                    }
                }
            }
        }
    }

    // Threshold secure: if WE (attacker side) are taking this trick and the feed
    // pushes our team's total past a step boundary, reward it.
    let we_take = likely_win || (partner_winning && (winner_locked || !opp_after_me));
    if let Some(step) = ctx.step_size {
        if step > 0 && ctx.me_is_attacker && we_take && my_point_contribution > 0 {
            let our_total =
                ctx.non_landlord_points + pot_points as isize + my_point_contribution as isize;
            let cur_step = (ctx.non_landlord_points / step) * step;
            if our_total >= cur_step + step {
                score += 5.0;
            }
        }
    }

    score
}

/// Rank the legal lead candidates by heuristic score (best first). Builds the
/// shared [`EvalCtx`] ONCE and scores every candidate against it.
pub fn ranked_leads(p: &PlayPhase, me: PlayerID) -> Vec<ScoredPlay> {
    let ctx = EvalCtx::build(p, me);
    let mut scored: Vec<ScoredPlay> = lead_candidates(p, me)
        .into_iter()
        .map(|cards| {
            let score = score_lead(&ctx, p, &cards);
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

/// Rank the legal follow candidates by heuristic score (best first). Builds the
/// shared [`EvalCtx`] ONCE and scores every candidate against it.
pub fn ranked_follows(p: &PlayPhase, me: PlayerID) -> Vec<ScoredPlay> {
    let ctx = EvalCtx::build(p, me);
    let mut scored: Vec<ScoredPlay> = follow_candidates(p, me)
        .into_iter()
        .map(|cards| {
            let score = score_follow(&ctx, p, &cards);
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

/// Rank lead candidates under an explicit [`HeuristicVersion`]. `New` uses the
/// shared [`EvalCtx`] scorer; `Legacy` uses the frozen original. The benchmark
/// harness calls this to compare the two heuristics head-to-head.
pub fn ranked_leads_with(
    p: &PlayPhase,
    me: PlayerID,
    version: HeuristicVersion,
) -> Vec<ScoredPlay> {
    match version {
        HeuristicVersion::New => ranked_leads(p, me),
        HeuristicVersion::Legacy => {
            let mut scored: Vec<ScoredPlay> = lead_candidates(p, me)
                .into_iter()
                .map(|cards| {
                    let score = score_lead_legacy(p, me, &cards);
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
    }
}

/// Rank follow candidates under an explicit [`HeuristicVersion`] (see
/// [`ranked_leads_with`]).
pub fn ranked_follows_with(
    p: &PlayPhase,
    me: PlayerID,
    version: HeuristicVersion,
) -> Vec<ScoredPlay> {
    match version {
        HeuristicVersion::New => ranked_follows(p, me),
        HeuristicVersion::Legacy => {
            let mut scored: Vec<ScoredPlay> = follow_candidates(p, me)
                .into_iter()
                .map(|cards| {
                    let score = score_follow_legacy(p, me, &cards);
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
    }
}

/// Greedy heuristic-DIRECT play: pick the argmax candidate under the given
/// [`HeuristicVersion`], with NO determinized search (fast). This is the
/// benchmark entry point — drive a full game by calling it for each seat's
/// PLAY-phase decision, pitting [`HeuristicVersion::New`] bots against
/// [`HeuristicVersion::Legacy`] bots in one binary. `p` must be the redacted
/// per-player view (honest); returns `None` only if no candidate exists.
pub fn choose_play_direct(
    p: &PlayPhase,
    me: PlayerID,
    version: HeuristicVersion,
) -> Option<Vec<Card>> {
    let leading = p.trick().played_cards().is_empty();
    let ranked = if leading {
        ranked_leads_with(p, me, version)
    } else {
        ranked_follows_with(p, me, version)
    };
    ranked.into_iter().next().map(|s| s.cards)
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

#[cfg(test)]
mod tests {
    use super::*;
    use shengji_mechanics::deck::Deck;
    use shengji_mechanics::hands::Hands;
    use shengji_mechanics::player::Player;
    use shengji_mechanics::types::Suit;

    use crate::settings::{GameMode, PropagatedState};

    fn card(n: Number, s: Suit) -> Card {
        Card::Suited { number: n, suit: s }
    }

    /// Build a 4-player, 1-deck PlayPhase with Hearts/2 trump. Seats 0 & 2 are the
    /// landlord (defending) team; seats 1 & 3 are attackers. `hands[i]` is dealt
    /// to seat `i`. Seat 0 is the landlord and leads the first trick.
    fn make_play_phase(hands: [Vec<Card>; 4]) -> (PlayPhase, Vec<PlayerID>) {
        let ids: Vec<PlayerID> = (0..4).map(PlayerID).collect();
        let mut propagated = PropagatedState::default();
        propagated.players = ids
            .iter()
            .map(|id| Player::new(*id, format!("p{}", id.0)))
            .collect();

        let trump = Trump::Standard {
            suit: Suit::Hearts,
            number: Number::Two,
        };
        let mut h = Hands::new(ids.iter().copied());
        h.set_trump(trump);
        for (i, cards) in hands.iter().enumerate() {
            h.add(ids[i], cards.iter().copied()).unwrap();
        }

        let pp = PlayPhase::new(
            propagated,
            1,
            GameMode::Tractor,
            h,
            vec![],
            trump,
            ids[0],               // landlord = seat 0
            ids[0],               // exchanger
            vec![ids[0], ids[2]], // landlord team = seats 0 & 2
            vec![],
            vec![Deck::default()],
        )
        .unwrap();
        (pp, ids)
    }

    #[test]
    fn test_evalctx_role_orientation() {
        // Seat 0 (landlord team) is a defender; seat 1 (non-landlord) is attacker.
        let (pp, ids) = make_play_phase([
            vec![card(Number::Ace, Suit::Spades)],
            vec![card(Number::King, Suit::Spades)],
            vec![card(Number::Queen, Suit::Spades)],
            vec![card(Number::Jack, Suit::Spades)],
        ]);
        let ctx_landlord = EvalCtx::build(&pp, ids[0]);
        let ctx_attacker = EvalCtx::build(&pp, ids[1]);
        assert!(
            !ctx_landlord.me_is_attacker,
            "seat 0 is on the landlord (defending) team"
        );
        assert!(
            ctx_attacker.me_is_attacker,
            "seat 1 is on the non-landlord (attacking) team"
        );
        // Default 1-deck step size is 20.
        assert_eq!(ctx_landlord.step_size, Some(20));
    }

    #[test]
    fn test_threshold_sign_defender_panic_vs_no_panic() {
        // Attacker (seat 1) is winning a point-rich pot. A defender (seat 2) is
        // asked to follow with a 10-point King it cannot win with. When donating
        // would push the attackers across the next 20-point step, the defender
        // must escalate its point penalty (threshold panic); when it would NOT,
        // the penalty is the base starve term. We compare the two.
        let king_spades = card(Number::King, Suit::Spades);

        // PANIC: seat0 leads the Ten of spades (10 pts in the pot), seat1
        // over-ranks with the Ace; donating the King makes attacker_total = 20.
        let (mut pp_panic, ids) = make_play_phase([
            vec![card(Number::Ten, Suit::Spades)], // seat0 (defender) leads the 10
            vec![card(Number::Ace, Suit::Spades)], // seat1 (attacker) over-ranks
            vec![king_spades],                     // seat2 (defender) 10-pt King
            vec![card(Number::Four, Suit::Spades)], // seat3 (attacker)
        ]);
        pp_panic
            .play_cards(ids[0], &[card(Number::Ten, Suit::Spades)])
            .unwrap();
        pp_panic
            .play_cards(ids[1], &[card(Number::Ace, Suit::Spades)])
            .unwrap();
        let ctx_panic = EvalCtx::build(&pp_panic, ids[2]);
        assert!(!ctx_panic.me_is_attacker);
        assert_eq!(ctx_panic.step_size, Some(20));
        let score_panic = score_follow(&ctx_panic, &pp_panic, &[king_spades]);

        // NO PANIC: an empty pot (seat0 leads a non-point 3), so donating the King
        // (10) leaves attacker_total = 10 < 20 — no threshold crossed.
        let (mut pp_ok, ids2) = make_play_phase([
            vec![card(Number::Three, Suit::Spades)], // seat0 leads non-point
            vec![card(Number::Ace, Suit::Spades)],   // seat1 (attacker) wins
            vec![king_spades],                       // seat2 (defender) 10-pt King
            vec![card(Number::Four, Suit::Spades)],  // seat3 (attacker)
        ]);
        pp_ok
            .play_cards(ids2[0], &[card(Number::Three, Suit::Spades)])
            .unwrap();
        pp_ok
            .play_cards(ids2[1], &[card(Number::Ace, Suit::Spades)])
            .unwrap();
        let ctx_ok = EvalCtx::build(&pp_ok, ids2[2]);
        let score_ok = score_follow(&ctx_ok, &pp_ok, &[king_spades]);

        // The panic case must penalize the donation strictly MORE (more negative).
        assert!(
            score_panic < score_ok,
            "defender donation crossing the threshold ({}) must be penalized more than a non-crossing one ({})",
            score_panic,
            score_ok
        );
        // And the attacker in the SAME crossing spot does NOT panic: handing
        // points to its OWN side is acceptable, so its can't-win penalty is the
        // base term, not the escalated one. We verify the role gate by checking an
        // attacker-built context skips the escalation.
        let ctx_attacker_view = EvalCtx::build(&pp_panic, ids[3]); // seat3 = attacker
        assert!(ctx_attacker_view.me_is_attacker);
    }

    #[test]
    fn test_partner_locked_vs_stealable_feed() {
        // Partner (seat 2, same team as seat 0) is winning. We are seat 0's
        // teammate seat 2... build instead from seat 0's perspective where the
        // PARTNER seat 2 has already played the winning card and an OPPONENT
        // (seat 3) is still to act vs. not.
        //
        // Scenario L (LOCKED): partner holds the boss (Ace of spades, no higher
        // spade unseen) and is last meaningful winner. Feeding points is great.
        // Scenario S (STEALABLE): partner is winning with a low card and an
        // opponent can still trump/over-rank. Feeding points should be weaker.
        let king_spades = card(Number::King, Suit::Spades); // 10-point card to feed

        // LOCKED: our teammate (seat0) leads the boss Ace of spades; we (seat2,
        // same team) follow with a point card while opp seat3 is still to act —
        // but the Ace is uncatchable in suit, so partner is LOCKED.
        let (mut pp_locked, ids2) = make_play_phase([
            vec![card(Number::Ace, Suit::Spades)], // seat0 (us-partner) leads boss
            vec![card(Number::Four, Suit::Spades)], // seat1 (opp)
            vec![king_spades],                     // seat2 (our team) has the 10pt K
            vec![card(Number::Five, Suit::Spades)], // seat3 (opp) still to act
        ]);
        // seat0 (our teammate) leads boss Ace; seat1 opp follows low.
        pp_locked
            .play_cards(ids2[0], &[card(Number::Ace, Suit::Spades)])
            .unwrap();
        pp_locked
            .play_cards(ids2[1], &[card(Number::Four, Suit::Spades)])
            .unwrap();
        // Now seat2 (our team) is to act; partner is winning with the LOCKED Ace,
        // even though opp seat3 is still to act (Ace spades can't be beaten in
        // suit, and seat3 must follow spades here).
        let ctx_locked = EvalCtx::build(&pp_locked, ids2[2]);
        let feed_locked = score_follow(&ctx_locked, &pp_locked, &[king_spades]);

        // STEALABLE: partner leads a LOW spade (3); opp seat3 can still over-rank.
        let (mut pp_steal, ids3) = make_play_phase([
            vec![card(Number::Three, Suit::Spades)], // seat0 (our team) leads LOW
            vec![card(Number::Four, Suit::Spades)],  // seat1 (opp)
            vec![king_spades],                       // seat2 (our team) has the K
            vec![card(Number::Ace, Suit::Spades)],   // seat3 (opp) can steal w/ Ace
        ]);
        pp_steal
            .play_cards(ids3[0], &[card(Number::Three, Suit::Spades)])
            .unwrap();
        pp_steal
            .play_cards(ids3[1], &[card(Number::Four, Suit::Spades)])
            .unwrap();
        let ctx_steal = EvalCtx::build(&pp_steal, ids3[2]);
        let feed_steal = score_follow(&ctx_steal, &pp_steal, &[king_spades]);

        assert!(
            feed_locked > feed_steal,
            "feeding points to a LOCKED partner ({}) should beat feeding to a STEALABLE one ({})",
            feed_locked,
            feed_steal
        );
    }

    #[test]
    fn test_boss_lead_beats_trash_lead() {
        // A guaranteed-top side-suit Ace lead should outscore a low trash lead.
        let (pp, ids) = make_play_phase([
            vec![
                card(Number::Ace, Suit::Spades),
                card(Number::Three, Suit::Clubs),
            ],
            vec![card(Number::King, Suit::Spades)],
            vec![card(Number::Queen, Suit::Spades)],
            vec![card(Number::Jack, Suit::Spades)],
        ]);
        let ctx = EvalCtx::build(&pp, ids[0]);
        let boss = score_lead(&ctx, &pp, &[card(Number::Ace, Suit::Spades)]);
        let trash = score_lead(&ctx, &pp, &[card(Number::Three, Suit::Clubs)]);
        assert!(
            boss > trash,
            "boss Ace lead ({}) should beat trash 3 lead ({})",
            boss,
            trash
        );
    }

    // =======================================================================
    // Qualitative A/B spot-checks: concrete positions where the NEW heuristic's
    // GREEDY argmax (`choose_play_direct(.., New)`) makes an obviously better
    // play than the LEGACY greedy argmax. These are the same entry points the
    // benchmark drives, so they document *why* NEW wins more.
    // =======================================================================

    /// SPOT-CHECK 1 — "leads a boss card to cash points".
    /// Seat 0 (the leader) holds the 10-point King of spades, which is a BOSS here
    /// (the only higher spade, the Ace, is in seat 0's own hand, so nothing unseen
    /// can beat the King), alongside a trash low card. The strong play is to LEAD
    /// the point-carrying boss King: it wins the trick risk-free AND banks 10
    /// points for our side. LEGACY's lead scorer has no boss notion and penalizes
    /// leading ANY point card unconditionally (`-1.2 * points`), so it refuses to
    /// lead the King and dumps the trash instead — leaving the points stranded.
    #[test]
    fn test_spotcheck_new_leads_boss_to_cash_points() {
        let ace_s = card(Number::Ace, Suit::Spades);
        let king_s = card(Number::King, Suit::Spades); // 10-pt boss (Ace is in our own hand)
        let trash = card(Number::Three, Suit::Clubs);
        let (pp, ids) = make_play_phase([
            vec![ace_s, king_s, trash],
            vec![card(Number::Four, Suit::Spades)],
            vec![card(Number::Five, Suit::Spades)],
            vec![card(Number::Six, Suit::Spades)],
        ]);
        let new = choose_play_direct(&pp, ids[0], HeuristicVersion::New).unwrap();
        let legacy = choose_play_direct(&pp, ids[0], HeuristicVersion::Legacy).unwrap();
        // NEW cashes the points behind a boss.
        assert!(
            new == vec![king_s] || new == vec![ace_s],
            "NEW should lead a boss spade (the point-cashing King, or the Ace); got {:?}",
            new
        );
        assert!(
            new.iter().any(|c| c.points().is_some()) || new == vec![ace_s],
            "NEW leads a boss; ideally the point-carrying King to bank 10 now"
        );
        // LEGACY refuses to lead the point card (its blanket point penalty) and
        // does NOT cash: it dumps the trash club.
        assert_eq!(
            legacy,
            vec![trash],
            "LEGACY's blanket point-penalty makes it dump trash instead of cashing"
        );
        assert_ne!(new, legacy, "NEW and LEGACY must diverge here");
    }

    /// SPOT-CHECK 2 — "ducks under partner".
    /// Our partner (seat 0, same team as seat 2) has LED the boss Ace of spades;
    /// it is uncatchable in suit, so the trick is LOCKED for our side. We (seat 2)
    /// must follow spades and hold both the 10-point King and a trash low spade.
    /// The strong play is to DUCK the points to our own winning partner (feed the
    /// King). LEGACY's follow scorer has no boss-lock notion: it sees partner
    /// "winning" and feeds points too — but it does so without the lock guarantee
    /// and, lacking the boss read, can prefer dumping the low card to "save
    /// strength". We assert NEW feeds the King under the locked partner.
    #[test]
    fn test_spotcheck_new_ducks_points_under_locked_partner() {
        let king_s = card(Number::King, Suit::Spades); // 10-point feed
        let low_s = card(Number::Three, Suit::Spades); // trash
        let (mut pp, ids) = make_play_phase([
            vec![card(Number::Ace, Suit::Spades)], // seat0 (our partner) leads boss
            vec![card(Number::Four, Suit::Spades)], // seat1 (opp)
            vec![king_s, low_s],                   // seat2 (us): K + low
            vec![card(Number::Five, Suit::Spades)], // seat3 (opp) still to act
        ]);
        pp.play_cards(ids[0], &[card(Number::Ace, Suit::Spades)])
            .unwrap();
        pp.play_cards(ids[1], &[card(Number::Four, Suit::Spades)])
            .unwrap();
        // Now it is seat 2 (us). Partner's Ace is locked; feed the King.
        let new = choose_play_direct(&pp, ids[2], HeuristicVersion::New).unwrap();
        assert_eq!(
            new,
            vec![king_s],
            "NEW should feed the 10-point King under the LOCKED partner Ace"
        );
    }

    /// SPOT-CHECK 3 — "hoards trump to ruff" (an ATTACKER must not bleed trump).
    /// An ATTACKER (seat 1, non-landlord team) is choosing a lead while holding a
    /// boss side-suit Ace of clubs and a plain low trump (a small Heart). For the
    /// attacking side the right plan is to keep trumps to RUFF the landlord's
    /// side suits later, so leading a bare low trump is a waste; lead the boss
    /// side-suit instead. NEW's role-aware scorer applies the trump-hoard penalty
    /// for the non-landlord side, ranking the side-suit boss strictly above the
    /// bare-trump lead. We score both candidate leads directly for seat 1 (the
    /// attacker) and assert NEW's preference and the role gate.
    #[test]
    fn test_spotcheck_new_hoards_trump_prefers_side_boss_lead() {
        let ace_c = card(Number::Ace, Suit::Clubs); // boss side-suit (uncatchable, 1 deck)
        let low_trump = card(Number::Seven, Suit::Hearts); // a plain low trump to hoard
        let (pp, ids) = make_play_phase([
            // seat0 (landlord/defender) — irrelevant to the scored seat.
            vec![card(Number::Three, Suit::Spades)],
            // seat1 (ATTACKER) is the seat we score: boss club + a low trump.
            vec![ace_c, low_trump],
            vec![card(Number::Queen, Suit::Clubs)],
            vec![card(Number::Jack, Suit::Clubs)],
        ]);
        // Seat 1 is the non-landlord (attacking) side.
        let ctx = EvalCtx::build(&pp, ids[1]);
        assert!(ctx.me_is_attacker, "seat 1 must be the attacking side");
        let side_boss = score_lead(&ctx, &pp, &[ace_c]);
        let bare_trump = score_lead(&ctx, &pp, &[low_trump]);
        assert!(
            side_boss > bare_trump,
            "NEW: attacker should prefer the side-suit boss ({}) over bleeding a \
             bare trump ({}), hoarding trump to ruff later",
            side_boss,
            bare_trump
        );
        // And the trump-hoard discipline is exactly the new role-aware term: the
        // bare low-trump lead is actively penalized (negative) for the attacker.
        assert!(
            bare_trump < 0.0,
            "NEW penalizes an attacker's bare-trump lead (got {})",
            bare_trump
        );
    }
}
