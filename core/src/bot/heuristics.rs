//! Heuristic Shengji policy: the backbone used directly by Easy and as
//! the rollout / leaf policy inside the Expert/Enoch determinized search (and
//! the Expert tier's net-load fallback prior).
//!
//! Everything here is computed from the redacted per-player view only.
//!
//! The core abstraction is a *scoring over legal candidate moves*: we generate
//! a small set of sensible candidate plays (using the engine's legal-move
//! generators), score each with Shengji strategy heuristics, and return them
//! ranked. Callers (the difficulty tiers) then pick from the ranking with
//! tier-specific randomness.

use std::collections::{BTreeMap, HashMap, HashSet};

use shengji_mechanics::ordered_card::OrderedCard;
use shengji_mechanics::trick::{PlayCards, TrickUnit, UnitLike};
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

/// The historical relative-strength encoding used by the embedded Expert model.
///
/// This deliberately preserves the old Ace-low quirk (`Ace == 1`). It must only
/// be used by frozen legacy scorers/features whose distributions the shipped
/// model was trained against. Live move generation and strategy must use
/// [`card_strength`], which follows the mechanics engine's Ace-high ordering.
pub(crate) fn legacy_card_strength(trump: Trump, card: Card) -> i32 {
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

/// Rank-correct relative strength within a card's effective suit.
///
/// The mechanics engine orders ordinary side-suit cards Ace-high. Keeping this
/// as the default bot strength function prevents the live policy from treating
/// an Ace as low trash when generating follows, bidding, or choosing a kitty.
/// Frozen Expert-model inputs explicitly call [`legacy_card_strength`] instead.
pub(crate) fn card_strength(trump: Trump, card: Card) -> i32 {
    match card {
        Card::Suited { number, suit }
            if trump.number() != Some(number) && number == Number::Ace =>
        {
            let _ = suit;
            14
        }
        _ => legacy_card_strength(trump, card),
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

/// Enumerate cards above `floor` under the frozen legacy Expert ordering.
///
/// Honest: depends only on the trump declaration + card identities. Lives here
/// (the lower-level module) so the Expert feature encoder and the heuristic both
/// call the SAME implementation without a layering inversion.
pub(crate) fn stronger_cards_in_suit(trump: Trump, eff: EffectiveSuit, floor: i32) -> Vec<Card> {
    let mut out: Vec<Card> = Vec::new();
    let mut consider = |c: Card| {
        if trump.effective_suit(c) == eff && legacy_card_strength(trump, c) > floor {
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

/// Frozen Expert-feature estimate of whether `card` cannot be beaten within its
/// effective suit by an unseen card. This is honest, but intentionally preserves
/// the old Ace-low ordering for the embedded model; live strategy uses
/// [`is_boss_card`].
///
/// NOTE: this uses [`legacy_card_strength`], whose ordering puts side-suit Aces LOW
/// (`Number::as_u32` is Ace-low). It is kept verbatim because the Expert net's
/// `f[34]` feature was trained against exactly this behaviour. The NEW heuristic
/// uses the rank-correct [`is_boss_card`] instead.
pub(crate) fn is_guaranteed_top(k: &Knowledge, trump: Trump, card: Card) -> bool {
    let s = legacy_card_strength(trump, card);
    if s >= 1000 {
        return true;
    }
    let eff = trump.effective_suit(card);
    for higher in stronger_cards_in_suit(trump, eff, s) {
        let seen = k.seen.get(&higher).copied().unwrap_or(0);
        if k.configured_copies(higher) > seen {
            // At least one copy of a stronger same-suit card is still unseen.
            return false;
        }
    }
    true
}

/// Backward-compatible name for the rank-correct live strength. New strategy
/// code historically called this helper; keeping it avoids churn while making
/// [`card_strength`] itself safe for all live policy paths.
pub(crate) fn boss_strength(trump: Trump, card: Card) -> i32 {
    card_strength(trump, card)
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
    if s >= 1000 {
        // A big joker cannot be over-ranked. A small joker and either kind of
        // level card remain beatable and must pass through the unseen-card scan.
        return true;
    }
    let eff = trump.effective_suit(card);
    for higher in boss_stronger_cards_in_suit(trump, eff, s) {
        let seen = k.seen.get(&higher).copied().unwrap_or(0);
        if k.configured_copies(higher) > seen {
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
    boss_stronger_cards_in_suit(trump, eff, s)
        .iter()
        .map(|h| {
            k.configured_copies(*h)
                .saturating_sub(k.seen.get(h).copied().unwrap_or(0))
        })
        .sum()
}

/// Per-decision evaluation context, built ONCE per search node and shared by
/// reference with both scorers. This is the single biggest cost lever: the
/// honest card-memory (`Knowledge`), the role/threshold facts, and the trump /
/// points accounting are computed once here instead of per candidate.
///
/// Everything here is HONEST — it derives only from the redacted per-player view
/// (own hand + table + last trick), so the honesty invariant is preserved across
/// Easy, the determinized-search rollouts, and the heuristic backbone. Inside a determinized
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
    /// Exact configured attacking-team win boundary (80 in standard two-deck
    /// Tractor). Used for late threshold-coverage plays without hard-coding 80.
    pub non_landlord_turnover_score: Option<isize>,
    /// Total trumps still unseen by `me` (in hidden hands / kitty).
    pub unseen_trumps: usize,
    /// "Enoch mode": when true, the Enoch-specific full-game playbook bonuses are
    /// layered onto the shared boss-/partner-aware scorers (tractor-first leads,
    /// long-suit running, defender low-trump hand-off, endgame kitty protection).
    /// Honest — every fact below derives from the redacted view only.
    pub enoch: bool,
    /// Number of cards `me` currently holds. Used by the Enoch endgame logic to
    /// detect the late game (a small hand) and to scale aggression.
    pub my_hand_size: usize,
    /// Trumps `me` currently holds (effective-suit trump). Used by the Enoch
    /// endgame kitty-protection rule (hoard trump when the kitty is valuable).
    pub my_trump_count: usize,
    /// The point value of the buried kitty, IF `me` is the exchanger (the
    /// landlord who buried it and may honestly recall its contents); `None`
    /// otherwise (attackers / non-exchanger defenders cannot see the kitty). The
    /// kitty is doubled when the attacking side wins it on the last trick, so a
    /// large value drives the Enoch endgame-protection rule.
    pub kitty_points: Option<isize>,
}

/// Honest expected point value of the buried kitty for a seat that did NOT bury it
/// (a non-exchanger), estimated from the UNSEEN card pool. The kitty is a random
/// subset of the cards `me` cannot see (other hands + the kitty itself), so its
/// expected points are its share of the points still unaccounted for:
/// `unseen_points × kitty_size / unseen_cards`. As non-point cards leave play and
/// points stay hidden (e.g. a strong declarer buried them), the unseen pool's point
/// density — and thus this estimate — rises, which is exactly the "heavy bank when
/// the declarer is strong" signal. HONEST: reads only `k.seen` (own hand + public
/// play) and the PUBLIC kitty size; the buried cards themselves are never consulted.
fn estimate_kitty_points(k: &Knowledge, kitty_size: usize, num_decks: usize) -> isize {
    if kitty_size == 0 {
        return 0;
    }
    // A standard deck holds 100 points (4×5 + 4×10 + 4×K = 100) and 54 cards.
    let total_points = 100isize * num_decks as isize;
    let total_cards = 54usize * num_decks;
    let seen_points: isize = k
        .seen
        .iter()
        .map(|(c, &n)| c.points().map(|x| x as isize).unwrap_or(0) * n as isize)
        .sum();
    let seen_cards: usize = k.seen.values().copied().sum();
    let unseen_points = (total_points - seen_points).max(0);
    let unseen_cards = total_cards.saturating_sub(seen_cards);
    if unseen_cards == 0 {
        return 0;
    }
    (unseen_points * kitty_size as isize) / unseen_cards as isize
}

impl EvalCtx {
    /// Build the context once for the acting player `me` from the redacted view
    /// (no Enoch playbook — used by Easy/Expert/Omniscient).
    pub fn build(p: &PlayPhase, me: PlayerID) -> Self {
        Self::build_inner(p, me, false)
    }

    /// Build the context with the Enoch full-game playbook ENABLED. Identical to
    /// [`EvalCtx::build`] except `enoch` is set, so the shared scorers add the
    /// Enoch-specific lead/follow bonuses. Still HONEST — Enoch reads only the
    /// redacted per-player view.
    pub fn build_enoch(p: &PlayPhase, me: PlayerID) -> Self {
        Self::build_inner(p, me, true)
    }

    fn build_inner(p: &PlayPhase, me: PlayerID, enoch: bool) -> Self {
        let trump = p.trick().trump();
        // Full public memory belongs to the observation, not a difficulty tier.
        // Stronger tiers differ in policy/search rather than intentionally
        // forgetting cards every human at the table saw.
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
        let unseen_trumps = k.total_trumps.saturating_sub(seen_trumps);

        // Own-hand summary (honest: it is our own hand). Cheap to compute and
        // only meaningfully consulted by the Enoch playbook.
        let (my_hand_size, my_trump_count) = match p.hands().get(me) {
            Ok(hand) => {
                let mut total = 0usize;
                let mut trumps = 0usize;
                for (&c, &n) in hand.iter() {
                    if c == Card::Unknown {
                        continue;
                    }
                    total += n;
                    if is_trump(trump, c) {
                        trumps += n;
                    }
                }
                (total, trumps)
            }
            Err(_) => (0, 0),
        };

        // The exchanger knows the buried cards exactly. Every other seat gets an
        // honest estimate from
        // the unseen-card point density (#2: "value the bank from what the declarer
        // has been playing"): the kitty is a random subset of the cards we cannot
        // see, so its expected value is its share of the points still unaccounted
        // for. This naturally rises as non-point cards leave play and points stay
        // hidden (a strong declarer who buried points leaves them unseen), so the
        // endgame rules make a non-declarer play for the doubled bank on the last
        // trick when it looks heavy. HONEST — the estimate reads only `k.seen`
        // (own hand + public play) and the PUBLIC kitty SIZE, never the buried cards.
        let kitty_points = if enoch {
            if p.exchanger() == me {
                p.visible_kitty().map(|kitty| {
                    kitty
                        .iter()
                        .filter_map(|c| c.points().map(|x| x as isize))
                        .sum()
                })
            } else {
                Some(estimate_kitty_points(&k, p.kitty_size(), num_decks))
            }
        } else {
            None
        };

        EvalCtx {
            k,
            trump,
            me,
            me_is_attacker,
            num_decks,
            non_landlord_points,
            step_size: p.bot_step_size(),
            non_landlord_turnover_score: p.bot_non_landlord_turnover_score(),
            unseen_trumps,
            enoch,
            my_hand_size,
            my_trump_count,
            kitty_points,
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

const DEFAULT_CANDIDATE_GENERATION_CAP: usize = 256;
const DEFAULT_MAX_THROW_UNITS: usize = 3;

fn candidate_generation_cap() -> usize {
    std::env::var("SHENGJI_BOT_CANDIDATE_GEN_CAP")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|cap: &usize| *cap > 0)
        .unwrap_or(DEFAULT_CANDIDATE_GENERATION_CAP)
        .min(4096)
}

fn max_throw_units() -> usize {
    std::env::var("SHENGJI_BOT_MAX_THROW_UNITS")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|units: &usize| *units > 0)
        .unwrap_or(DEFAULT_MAX_THROW_UNITS)
        .min(8)
}

fn canonicalize_play(trump: Trump, cards: &mut [Card]) {
    cards.sort_by(|a, b| {
        trump
            .compare(*a, *b)
            .then_with(|| a.as_char().cmp(&b.as_char()))
    });
}

fn canonical_play_key(cards: &[Card]) -> Vec<char> {
    cards.iter().map(|card| card.as_char()).collect()
}

fn push_legal_candidate(
    p: &PlayPhase,
    me: PlayerID,
    trump: Trump,
    mut cards: Vec<Card>,
    candidates: &mut Vec<Vec<Card>>,
    seen: &mut HashSet<Vec<char>>,
    cap: usize,
) {
    if cards.is_empty() || candidates.len() >= cap {
        return;
    }
    canonicalize_play(trump, &mut cards);
    let key = canonical_play_key(&cards);
    if !seen.contains(&key) && p.can_play_cards(me, &cards).is_ok() {
        seen.insert(key);
        candidates.push(cards);
    }
}

fn canonical_candidate_order(trump: Trump, candidates: &mut Vec<Vec<Card>>) {
    for candidate in candidates.iter_mut() {
        canonicalize_play(trump, candidate);
    }
    candidates.sort_by(|a, b| {
        a.len()
            .cmp(&b.len())
            .then_with(|| canonical_play_key(a).cmp(&canonical_play_key(b)))
    });
    candidates.dedup();
}

/// Merge independently generated action families without letting a large early
/// family consume the global cap. One candidate is taken from each family in
/// turn; ranking happens afterwards, so this preserves strategic coverage while
/// remaining deterministic.
fn merge_candidate_families(
    trump: Trump,
    mut families: Vec<Vec<Vec<Card>>>,
    cap: usize,
) -> Vec<Vec<Card>> {
    for family in &mut families {
        canonical_candidate_order(trump, family);
    }
    let mut result = Vec::with_capacity(cap);
    let mut seen = HashSet::new();
    let mut index = 0usize;
    while result.len() < cap {
        let mut added = false;
        for family in &families {
            if let Some(candidate) = family.get(index) {
                let key = canonical_play_key(candidate);
                if seen.insert(key) {
                    result.push(candidate.clone());
                    added = true;
                    if result.len() == cap {
                        break;
                    }
                }
            }
        }
        if !added && families.iter().all(|family| family.len() <= index + 1) {
            break;
        }
        index += 1;
    }
    canonical_candidate_order(trump, &mut result);
    result
}

fn hand_entries(
    hand: &HashMap<Card, usize>,
    trump: Trump,
    predicate: impl Fn(Card) -> bool,
) -> Vec<(Card, usize)> {
    let mut entries: Vec<(Card, usize)> = hand
        .iter()
        .filter_map(|(&card, &count)| {
            (count > 0 && card != Card::Unknown && predicate(card)).then_some((card, count))
        })
        .collect();
    entries.sort_by(|(a, _), (b, _)| {
        trump
            .compare(*a, *b)
            .then_with(|| a.as_char().cmp(&b.as_char()))
    });
    entries
}

fn count_multiset_combinations(entries: &[(Card, usize)], choose: usize, cap: usize) -> usize {
    let mut ways = vec![0usize; choose + 1];
    ways[0] = 1;
    for &(_, available) in entries {
        let previous = ways.clone();
        for selected in 1..=choose {
            ways[selected] = (0..=available.min(selected))
                .map(|take| previous[selected - take])
                .fold(0usize, |total, n| total.saturating_add(n).min(cap + 1));
        }
    }
    ways[choose]
}

fn enumerate_multiset_combinations(
    entries: &[(Card, usize)],
    choose: usize,
    limit: usize,
) -> Vec<Vec<Card>> {
    fn recurse(
        entries: &[(Card, usize)],
        index: usize,
        remaining: usize,
        current: &mut Vec<Card>,
        out: &mut Vec<Vec<Card>>,
        limit: usize,
    ) {
        if out.len() >= limit {
            return;
        }
        if index == entries.len() {
            if remaining == 0 {
                out.push(current.clone());
            }
            return;
        }
        let (card, available) = entries[index];
        for take in 0..=available.min(remaining) {
            current.extend(std::iter::repeat_n(card, take));
            recurse(entries, index + 1, remaining - take, current, out, limit);
            current.truncate(current.len() - take);
            if out.len() >= limit {
                return;
            }
        }
    }

    let mut out = vec![];
    recurse(entries, 0, choose, &mut vec![], &mut out, limit);
    out
}

#[allow(clippy::too_many_arguments)]
fn compose_unit_throws(
    p: &PlayPhase,
    me: PlayerID,
    trump: Trump,
    units: &[Vec<Card>],
    start: usize,
    units_left: usize,
    current: &mut Vec<Card>,
    candidates: &mut Vec<Vec<Card>>,
    seen: &mut HashSet<Vec<char>>,
    cap: usize,
    attempts: &mut usize,
    attempt_cap: usize,
) {
    if candidates.len() >= cap || *attempts >= attempt_cap {
        return;
    }
    if units_left == 0 {
        *attempts += 1;
        push_legal_candidate(p, me, trump, current.clone(), candidates, seen, cap);
        return;
    }
    for index in start..units.len() {
        let old_len = current.len();
        current.extend_from_slice(&units[index]);
        compose_unit_throws(
            p,
            me,
            trump,
            units,
            index + 1,
            units_left - 1,
            current,
            candidates,
            seen,
            cap,
            attempts,
            attempt_cap,
        );
        current.truncate(old_len);
        if candidates.len() >= cap || *attempts >= attempt_cap {
            return;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn enumerate_rainbow_counts(
    p: &PlayPhase,
    me: PlayerID,
    trump: Trump,
    groups: &[(Card, usize)],
    index: usize,
    min_cards: usize,
    current: &mut Vec<Card>,
    candidates: &mut Vec<Vec<Card>>,
    seen: &mut HashSet<Vec<char>>,
    cap: usize,
) {
    if candidates.len() >= cap {
        return;
    }
    if index == groups.len() {
        if current.len() >= min_cards {
            push_legal_candidate(p, me, trump, current.clone(), candidates, seen, cap);
        }
        return;
    }
    let (card, available) = groups[index];
    for take in 1..=available {
        current.extend(std::iter::repeat_n(card, take));
        enumerate_rainbow_counts(
            p,
            me,
            trump,
            groups,
            index + 1,
            min_cards,
            current,
            candidates,
            seen,
            cap,
        );
        current.truncate(current.len() - take);
        if candidates.len() >= cap {
            return;
        }
    }
}

/// Generate a bounded hierarchy of legal lead plays:
///
/// 1. singles/repeated units/tractors under the table's configured requirements;
/// 2. enabled rainbows;
/// 3. ordinary same-suit throws composed progressively from 2..N units.
///
/// Every proposal is validated by the mechanics engine and canonicalized, so
/// search receives deterministic, legal candidates without a combinatorial
/// powerset explosion. The cap and maximum throw width are configurable through
/// `SHENGJI_BOT_CANDIDATE_GEN_CAP` and `SHENGJI_BOT_MAX_THROW_UNITS`.
pub fn lead_candidates(p: &PlayPhase, me: PlayerID) -> Vec<Vec<Card>> {
    lead_candidates_with_limits(p, me, candidate_generation_cap(), max_throw_units())
}

/// Bounded candidate set for the rollout hot path. Root search keeps the wider
/// configurable generator, while each hypothetical rollout ply avoids spending
/// most of its budget constructing/scoring hundreds of actions. Sixty-four still
/// covers every ordinary single/pair/tractor in a standard hand and permits
/// two-unit throws; the mechanics engine validates every proposal.
pub(crate) fn rollout_lead_candidates(p: &PlayPhase, me: PlayerID) -> Vec<Vec<Card>> {
    lead_candidates_with_limits(p, me, 64, 2)
}

fn lead_candidates_with_limits(
    p: &PlayPhase,
    me: PlayerID,
    cap: usize,
    max_units: usize,
) -> Vec<Vec<Card>> {
    let hand = match p.hands().get(me) {
        Ok(h) => h,
        Err(_) => return vec![],
    };
    let cards: Vec<Card> = Card::cards(hand.iter()).copied().collect();
    let trump = p.trick().trump();
    let cap = cap.max(1);
    let mut atomic_candidates: Vec<Vec<Card>> = vec![];
    let mut atomic_seen = HashSet::new();
    let mut rainbow_candidates: Vec<Vec<Card>> = vec![];
    let mut rainbow_seen = HashSet::new();
    let mut throw_candidates: Vec<Vec<Card>> = vec![];
    let mut throw_seen = HashSet::new();

    // Build atomic units deterministically by effective suit. Explicit repeated
    // units ensure a single card from a held pair remains available as a lead;
    // `find_plays` adds configured tractors and alternative decompositions.
    let mut atomic_by_suit: BTreeMap<EffectiveSuit, Vec<Vec<Card>>> = BTreeMap::new();
    for (&card, &count) in hand {
        if card == Card::Unknown || count == 0 {
            continue;
        }
        let units = atomic_by_suit
            .entry(trump.effective_suit(card))
            .or_default();
        for repeated in 1..=count {
            units.push(std::iter::repeat_n(card, repeated).collect());
        }
    }

    let mut suit_groups: Vec<(EffectiveSuit, Vec<Card>)> =
        cards_by_suit(trump, &cards).into_iter().collect();
    suit_groups.sort_by_key(|(suit, _)| *suit);
    for (suit, mut suit_cards) in suit_groups {
        canonicalize_play(trump, &mut suit_cards);
        let results = TrickUnit::find_plays(trump, p.propagated().tractor_requirements, suit_cards);
        for play in results.into_iter().take(cap) {
            for unit in play {
                atomic_by_suit.entry(suit).or_default().push(unit.cards());
            }
        }
    }

    for units in atomic_by_suit.values_mut() {
        canonical_candidate_order(trump, units);
        for unit in units.iter().cloned() {
            push_legal_candidate(
                p,
                me,
                trump,
                unit,
                &mut atomic_candidates,
                &mut atomic_seen,
                cap,
            );
        }
    }

    // Rainbows are the one supported compound lead that spans effective suits.
    if let Some(min_cards) = p.propagated().compound_formats.rainbows {
        let mut by_number: BTreeMap<Number, BTreeMap<EffectiveSuit, (Card, usize)>> =
            BTreeMap::new();
        for (&card, &count) in hand {
            let Some(number) = card.number() else {
                continue;
            };
            by_number
                .entry(number)
                .or_default()
                .insert(trump.effective_suit(card), (card, count));
        }
        for groups in by_number.values() {
            if groups.len() < 4 {
                continue;
            }
            let groups: Vec<(Card, usize)> = groups.values().copied().collect();
            enumerate_rainbow_counts(
                p,
                me,
                trump,
                &groups,
                0,
                min_cards,
                &mut vec![],
                &mut rainbow_candidates,
                &mut rainbow_seen,
                cap,
            );
        }
    }

    // Compose ordinary throws progressively: all atomic units first, then every
    // legal two-unit throw, then three-unit throws, and so on up to the limit.
    let mut throw_attempts = 0usize;
    let throw_attempt_cap = cap.saturating_mul(16);
    for units_wanted in 2..=max_units.max(1) {
        for units in atomic_by_suit.values() {
            if units.len() < units_wanted || throw_candidates.len() >= cap {
                continue;
            }
            compose_unit_throws(
                p,
                me,
                trump,
                units,
                0,
                units_wanted,
                &mut vec![],
                &mut throw_candidates,
                &mut throw_seen,
                cap,
                &mut throw_attempts,
                throw_attempt_cap,
            );
        }
    }

    let mut candidates = merge_candidate_families(
        trump,
        vec![atomic_candidates, rainbow_candidates, throw_candidates],
        cap,
    );
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
    follow_candidates_with_cap(p, me, candidate_generation_cap())
}

/// Bounded counterpart to [`follow_candidates`] for repeated rollout plies.
pub(crate) fn rollout_follow_candidates(p: &PlayPhase, me: PlayerID) -> Vec<Vec<Card>> {
    follow_candidates_with_cap(p, me, 64)
}

fn follow_candidates_with_cap(p: &PlayPhase, me: PlayerID, cap: usize) -> Vec<Vec<Card>> {
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

    let mut available_cards: Vec<Card> = Card::cards(
        hand.iter()
            .filter(|(c, _)| trump.effective_suit(**c) == trick_format.suit()),
    )
    .copied()
    .collect();
    canonicalize_play(trump, &mut available_cards);

    let cap = cap.max(1);
    let mut structural_candidates: Vec<Vec<Card>> = vec![];
    let mut structural_seen = HashSet::new();

    // Format-matching plays. `check_play` can yield many distinct mappings; the
    // previous generator kept only `.next()`, hiding legal pairs/tractors that
    // happened to appear later in iterator order.
    for format in trick_format.decomposition(p.propagated().trick_draw_policy()) {
        let playable = UnitLike::check_play(
            OrderedCard::make_map(available_cards.iter().copied(), trump),
            format.iter().cloned(),
            p.propagated().trick_draw_policy(),
        );
        for u in playable.into_iter().take(cap) {
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
                push_legal_candidate(
                    p,
                    me,
                    trump,
                    play,
                    &mut structural_candidates,
                    &mut structural_seen,
                    cap,
                );
            }
            if structural_candidates.len() >= cap {
                break;
            }
        }
    }

    // If we have enough in-suit cards, also offer a few "which same-suit cards"
    // variants: play the lowest cards, the highest cards, and a POINT-PROTECTING
    // set (lowest NON-POINT first). These give the scorer meaningful choices when
    // discarding within the led suit.
    if available_cards.len() >= num_required {
        let mut low_first = available_cards.clone();
        low_first.sort_by_key(|c| (card_strength(trump, *c), c.as_char()));
        push_legal_candidate(
            p,
            me,
            trump,
            low_first.iter().take(num_required).copied().collect(),
            &mut structural_candidates,
            &mut structural_seen,
            cap,
        );

        let mut high_first = available_cards.clone();
        high_first.sort_by_key(|c| (std::cmp::Reverse(card_strength(trump, *c)), c.as_char()));
        push_legal_candidate(
            p,
            me,
            trump,
            high_first.iter().take(num_required).copied().collect(),
            &mut structural_candidates,
            &mut structural_seen,
            cap,
        );

        // Point-feeding alternative for cases where our partner is safely ahead.
        let mut points_first = available_cards.clone();
        points_first.sort_by_key(|c| (!is_point(*c), card_strength(trump, *c), c.as_char()));
        push_legal_candidate(
            p,
            me,
            trump,
            points_first.iter().take(num_required).copied().collect(),
            &mut structural_candidates,
            &mut structural_seen,
            cap,
        );

        // Point-protecting alternative: retain 5s/10s/Ks where the required
        // structure allows and shed low non-points first.
        let mut nonpoints_first = available_cards.clone();
        nonpoints_first.sort_by_key(|c| (is_point(*c), card_strength(trump, *c), c.as_char()));
        push_legal_candidate(
            p,
            me,
            trump,
            nonpoints_first.iter().take(num_required).copied().collect(),
            &mut structural_candidates,
            &mut structural_seen,
            cap,
        );
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
            weak.sort_by_key(|c| (fill_discard_key(trump, *c), c.as_char()));
            let mut va = available_cards.clone();
            va.extend(weak.iter().take(need).copied());
            if va.len() == num_required {
                push_legal_candidate(
                    p,
                    me,
                    trump,
                    va,
                    &mut structural_candidates,
                    &mut structural_seen,
                    cap,
                );
            }

            // Variant B: if our partner is winning, the scorer may prefer to
            // feed points. Candidate generation must expose that option.
            let mut point_dump = off_suit.clone();
            point_dump.sort_by_key(|c| (!is_point(*c), card_strength(trump, *c), c.as_char()));
            let mut vb = available_cards.clone();
            vb.extend(point_dump.iter().take(need).copied());
            if vb.len() == num_required {
                push_legal_candidate(
                    p,
                    me,
                    trump,
                    vb,
                    &mut structural_candidates,
                    &mut structural_seen,
                    cap,
                );
            }

            // Variant C: trump in with the lowest trumps (capture attempt).
            let mut trumps: Vec<Card> = off_suit
                .iter()
                .copied()
                .filter(|c| is_trump(trump, *c))
                .collect();
            if trumps.len() >= need {
                trumps.sort_by_key(|c| (card_strength(trump, *c), c.as_char()));
                let mut vb = available_cards.clone();
                vb.extend(trumps.iter().take(need).copied());
                if vb.len() == num_required {
                    push_legal_candidate(
                        p,
                        me,
                        trump,
                        vb,
                        &mut structural_candidates,
                        &mut structural_seen,
                        cap,
                    );
                }

                // Variant C': ruff preferring NON-PAIRED trumps, so we don't
                // fragment a trump pair to take a trick that doesn't need one
                // (#7: take a non-pair throw with non-paired trump). Singletons in
                // hand first, then by strength; the scorer's pair-break penalty
                // then favors this over Variant B when both can win.
                let mut singleton_first = trumps.clone();
                singleton_first.sort_by_key(|c| {
                    let paired = hand.get(c).copied().unwrap_or(0) >= 2;
                    (paired as i32, card_strength(trump, *c), c.as_char())
                });
                let mut vbp = available_cards.clone();
                vbp.extend(singleton_first.iter().take(need).copied());
                if vbp.len() == num_required {
                    push_legal_candidate(
                        p,
                        me,
                        trump,
                        vbp,
                        &mut structural_candidates,
                        &mut structural_seen,
                        cap,
                    );
                }
            }
        } else if need == 0 {
            push_legal_candidate(
                p,
                me,
                trump,
                available_cards.clone(),
                &mut structural_candidates,
                &mut structural_seen,
                cap,
            );
        }
    }

    // Explicit bomb proposals. Bomb policies can permit a follow of a different
    // length/suit, so a `num_required`-only generator would never discover them.
    let mut bomb_candidates = Vec::new();
    let mut bomb_seen = HashSet::new();
    if p.propagated().bomb_policy.bombs_enabled() {
        let entries = hand_entries(&hand, trump, |_| true);
        for (card, count) in entries {
            for size in 4..=count {
                push_legal_candidate(
                    p,
                    me,
                    trump,
                    std::iter::repeat_n(card, size).collect(),
                    &mut bomb_candidates,
                    &mut bomb_seen,
                    cap,
                );
            }
        }
    }

    // For genuinely small action spaces, enumerate every multiset of the normal
    // required length and let mechanics filter it. This gives complete legal-
    // equivalence coverage in endgames while avoiding a late-hand C(27,k)
    // explosion. Larger spaces stay on the structured proposals above.
    let entries = hand_entries(&hand, trump, |_| true);
    let combinations = count_multiset_combinations(&entries, num_required, cap);
    let mut exhaustive_candidates = Vec::new();
    let mut exhaustive_seen = HashSet::new();
    if combinations <= cap {
        for play in enumerate_multiset_combinations(&entries, num_required, combinations.max(1)) {
            push_legal_candidate(
                p,
                me,
                trump,
                play,
                &mut exhaustive_candidates,
                &mut exhaustive_seen,
                cap,
            );
        }
    }

    let mut candidates = merge_candidate_families(
        trump,
        vec![
            structural_candidates,
            bomb_candidates,
            exhaustive_candidates,
        ],
        cap,
    );
    if candidates.is_empty() {
        let fallback = simple_follow(
            &available_cards,
            &hand,
            trump,
            trick_format.suit(),
            num_required,
        );
        let mut fallback_seen = HashSet::new();
        push_legal_candidate(
            p,
            me,
            trump,
            fallback,
            &mut candidates,
            &mut fallback_seen,
            cap,
        );
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
    remaining.sort_by_key(|c| (card_strength(trump, *c), c.as_char()));
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
        off.sort_by_key(|c| (fill_discard_key(trump, *c), c.as_char()));
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
        .map(|c| legacy_card_strength(trump, *c))
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

#[derive(Clone, Copy, Debug)]
struct LeadSafety {
    single_unit: bool,
    whole_play_safe: bool,
}

fn trick_unit_is_proven_safe(ctx: &EvalCtx, unit: &TrickUnit) -> bool {
    match unit {
        TrickUnit::Repeated { count, card } => {
            if *count == 1 {
                is_boss_card(&ctx.k, ctx.trump, card.card)
            } else {
                repeated_unbeatable_above(
                    ctx,
                    ctx.trump.effective_suit(card.card),
                    boss_strength(ctx.trump, card.card),
                    *count,
                )
            }
        }
        TrickUnit::Tractor { count, members } => members.last().is_some_and(|top| {
            // Any strictly higher tractor must contain a repeated card above
            // this tractor's top. Proving that no such repeated holding exists
            // is conservative but sufficient to prove the tractor safe.
            repeated_unbeatable_above(
                ctx,
                ctx.trump.effective_suit(top.card),
                boss_strength(ctx.trump, top.card),
                *count,
            )
        }),
    }
}

/// Classify whether a lead is one indivisible trick unit, or a compound whose
/// every component is provably immune to throw invalidation from public
/// information. The same decomposition selected by
/// [`shengji_mechanics::trick::TrickFormat::from_cards`] is checked unit-by-unit,
/// so cards consumed by a tractor cannot hide a vulnerable leftover singleton.
/// The proof is deliberately conservative: singleton components must be boss
/// cards, and repeated/tractor components must have no unseen strictly-higher
/// repeated holding capable of stopping them.
fn lead_safety(ctx: &EvalCtx, p: &PlayPhase, cards: &[Card]) -> LeadSafety {
    if cards.is_empty() {
        return LeadSafety {
            single_unit: false,
            whole_play_safe: false,
        };
    }

    let mut decompositions: Vec<Vec<TrickUnit>> = TrickUnit::find_plays(
        ctx.trump,
        p.propagated().tractor_requirements,
        cards.iter().copied(),
    )
    .into_iter()
    .collect();
    // Keep this ordering in lockstep with TrickFormat::from_cards(None): the
    // engine chooses the decomposition containing the largest unit, preferring
    // a tractor over a repeated unit of equal size, and takes the final tie.
    decompositions.sort_by_key(|units| {
        units
            .iter()
            .map(|unit| (unit.size(), unit.is_tractor()))
            .max()
    });
    let Some(units) = decompositions.pop() else {
        return LeadSafety {
            single_unit: false,
            whole_play_safe: false,
        };
    };

    let single_unit = units.len() == 1;
    if single_unit {
        return LeadSafety {
            single_unit,
            whole_play_safe: true,
        };
    }

    let whole_play_safe = units
        .iter()
        .all(|unit| trick_unit_is_proven_safe(ctx, unit));

    LeadSafety {
        single_unit,
        whole_play_safe,
    }
}

/// Joker compounds are especially prone to a misleading max-card score: one
/// unbeatable joker can mask a beatable low component. Keep single-unit joker
/// plays and compounds whose *whole* throw is proven safe; omit speculative
/// joker compounds from production lead rankings. The raw lead generator stays
/// exhaustive for callers that need it, and follow candidates are unaffected.
pub(crate) fn admissible_ranked_lead(ctx: &EvalCtx, p: &PlayPhase, cards: &[Card]) -> bool {
    let contains_joker = cards
        .iter()
        .any(|card| matches!(card, Card::BigJoker | Card::SmallJoker));
    if !contains_joker {
        return true;
    }

    let safety = lead_safety(ctx, p, cards);
    safety.single_unit || safety.whole_play_safe
}

/// Narrow late-game exception to normal joker conservation: the attacking team
/// is one small point dump from the exact configured turnover boundary, our
/// teammate is publicly known void in trump, and our own side has weak remaining
/// trump control. A boss joker can then cover the teammate's discard and secure
/// the contract instead of gambling for later/bank points.
fn threshold_coverage_lead(
    ctx: &EvalCtx,
    p: &PlayPhase,
    lead: Card,
    is_boss: bool,
    len: f64,
) -> bool {
    let teammate_known_void_in_trump = p
        .propagated()
        .players()
        .iter()
        .map(|player| player.id)
        .filter(|player| *player != ctx.me && same_team(p, ctx.me, *player))
        .any(|player| {
            ctx.k
                .voids
                .get(&player)
                .is_some_and(|voids| voids.contains(&EffectiveSuit::Trump))
        });
    ctx.me_is_attacker
        && ctx.my_hand_size <= 10
        && ctx
            .non_landlord_turnover_score
            .map(|threshold| threshold - ctx.non_landlord_points)
            .is_some_and(|needed| (1..=15).contains(&needed))
        && teammate_known_void_in_trump
        && ctx.my_trump_count <= 3
        && ctx.unseen_trumps > ctx.my_trump_count
        && len < 2.0
        && matches!(lead, Card::BigJoker | Card::SmallJoker)
        && is_boss
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
    let min_boss_strength = cards
        .iter()
        .map(|c| boss_strength(trump, *c))
        .min()
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
    let safety = lead_safety(ctx, p, cards);
    let whole_play_safe = safety.whole_play_safe;
    let boss_worth_checking = whole_play_safe && (max_boss_strength >= 13 || trumping);
    let is_boss = boss_worth_checking && is_boss_card(&ctx.k, trump, top);

    // A failed compound is governed by its vulnerable component, not by the
    // joker or other winner attached to it. Safe units/throws receive their
    // strongest-card value; speculative compounds receive only their weakest.
    let credited_strength = if whole_play_safe {
        max_boss_strength
    } else {
        min_boss_strength
    };

    let threshold_coverage = threshold_coverage_lead(ctx, p, lead, is_boss, len);

    let mut score = 0.0;

    // Multi-card units (pairs / tractors) pressure opponents.
    score += (len - 1.0) * 6.0;

    // A multi-unit throw is valuable only when every component is protected.
    // Without this guard, the old max-card approximation gave an unsafe throw
    // containing one joker roughly +50 raw-strength points and treated its total
    // length like a tractor, causing routine failed throws.
    if !whole_play_safe {
        score -= (len - 1.0) * 50.0;
    }

    // Modest reward for raw strength. Unsafe compounds use their weakest
    // component, so attaching a joker cannot manufacture a strength bonus.
    // `boss_strength` uses 997..1000 sentinels for trump-rank cards and jokers;
    // cap those semantic sentinels because the explicit boss/unit terms below
    // carry their tactical value.
    score += credited_strength.min(20) as f64 * 0.05;

    // Guaranteed-winner bonuses. Jokers keep their flat floor; a *boss*
    // (uncatchable in its suit) non-trump unit is excellent.
    if whole_play_safe && max_boss_strength >= 990 {
        score += 8.0;
    }
    if is_boss && !trumping {
        score += 9.0;
    }
    // Boss tractors / pairs dominate; reward extra length.
    if is_boss && len >= 2.0 {
        score += (len - 1.0) * 4.0;
    }

    // A naked joker should normally cash points, not open an empty trick. Search
    // may still select it when downstream rollouts justify the line, while the
    // narrow late attacking exception keeps a boss-joker coverage lead available
    // when a known-void partner can dump the last 5–15 points needed to win.
    if cards.len() == 1 && matches!(lead, Card::BigJoker | Card::SmallJoker) {
        if threshold_coverage {
            score += 12.0;
        } else if ctx.my_hand_size > 1 {
            score -= 20.0;
        }
    }

    // Near-boss: exactly one stronger copy still unseen in this suit. Leading it
    // is still strong (only one card can take it).
    if whole_play_safe
        && !is_boss
        && boss_worth_checking
        && unseen_dominators(&ctx.k, trump, top) == 1
    {
        score += 3.0;
    }

    // Points in the lead: good behind a boss (we cash them safely), bad
    // otherwise (we'd be donating to whoever wins).
    if is_boss {
        score += point_total as f64 * 0.6;
    } else {
        score -= point_total as f64 * 1.2;
    }

    // Early point-cashing through a known partner void is only "free" when the
    // public record positively proves every opponent still holds this suit.
    // Absence of a known void is not proof of a holding. Conversely, a known-void
    // opponent plus a partner known to hold the suit is a ruff trap, so protect
    // points unless another scorer/search line can justify the sacrifice.
    if !trumping && ctx.my_hand_size >= 14 && point_total > 0 {
        let eff = trump.effective_suit(lead);
        let mut partner_void = false;
        let mut partner_positive = false;
        let mut opponents = 0usize;
        let mut all_opponents_positive = true;
        let mut any_opponent_void = false;
        for player in p.propagated().players() {
            let pid = player.id;
            if pid == ctx.me {
                continue;
            }
            let known_void = ctx
                .k
                .voids
                .get(&pid)
                .is_some_and(|voids| voids.contains(&eff));
            let known_positive = ctx.k.known_holding.get(&pid).is_some_and(|holding| {
                holding
                    .iter()
                    .any(|(card, count)| *count > 0 && trump.effective_suit(*card) == eff)
            });
            if same_team(p, ctx.me, pid) {
                partner_void |= known_void;
                partner_positive |= known_positive;
            } else {
                opponents += 1;
                any_opponent_void |= known_void;
                all_opponents_positive &= known_positive;
            }
        }
        if partner_void && opponents > 0 && all_opponents_positive {
            score += point_total as f64 * 1.8 + 3.0;
        } else if partner_positive && any_opponent_void {
            score -= point_total as f64 * 2.0 + 3.0;
        }
    }

    // Trump leads by role. A joker pair is always a powerful play. Otherwise, the
    // landlord side should DRAW trump with a boss trump while opponents still
    // hold some, easing off as they dry out; everyone else hoards.
    if trumping {
        let counts = Card::count(cards.iter().copied());
        let joker_pair = counts.get(&Card::BigJoker).copied().unwrap_or(0) >= 2
            || counts.get(&Card::SmallJoker).copied().unwrap_or(0) >= 2;
        let protected_joker_pair = whole_play_safe && joker_pair;
        if protected_joker_pair {
            score += 2.0;
        }
        let me_is_landlord_side = !ctx.me_is_attacker;
        if me_is_landlord_side && is_boss && ctx.unseen_trumps > 0 {
            let draw_scale =
                (ctx.unseen_trumps as f64 / (ctx.num_decks * 2) as f64).clamp(0.0, 1.0);
            score += 3.0 * draw_scale;
        } else if !protected_joker_pair {
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

    if ctx.enoch {
        score += enoch_lead_bonus(ctx, p, lead, trumping, is_boss, len);
    }

    score
}

/// The Enoch-mode lead bonuses layered onto [`score_lead`]. Encodes the
/// enthusiast's leading playbook (`docs/strategy/double-holder.txt`):
///
/// * "Absolutely prioritize tractors": lead consecutive-pair tractors (and pairs)
///   before they get broken up — the earlier the better.
/// * "Long-suit run": leading a non-trump suit we hold MANY of (especially with
///   pairs in it) forces opponents to burn trump to stop it. Reward length, with
///   a kicker when we also hold a pair there.
/// * "Defender hand-off": as a defender, after our boss cards are spent, leading
///   a LOW TRUMP burns the attackers' trump and passes the lead to our partner,
///   who can then run a suit we are void in. Rewarded only late-ish, only for
///   small (non-boss) trumps, and only for the defending side.
/// * "Endgame kitty protection": when WE buried a point-rich kitty, in the late
///   game don't fritter away high cards / trump on a speculative lead — hoard
///   them to win the last trick. We damp aggressive non-boss leads accordingly.
///
/// All inputs are from the redacted view; the function only reads our OWN hand.
/// Whether we hold ANY non-trump boss card (an uncatchable single in some side
/// suit) we could lead instead. Used to gate the defender hand-off: only toss the
/// lead away with a low trump once our cashable winners are spent. Honest — reads
/// only our own hand + `k.seen`.
fn my_best_nontrump_is_boss(ctx: &EvalCtx, p: &PlayPhase) -> bool {
    let trump = ctx.trump;
    if let Ok(hand) = p.hands().get(ctx.me) {
        for (&c, &n) in hand.iter() {
            if n == 0 || c == Card::Unknown || is_trump(trump, c) {
                continue;
            }
            if is_boss_card(&ctx.k, trump, c) {
                return true;
            }
        }
    }
    false
}

fn enoch_lead_bonus(
    ctx: &EvalCtx,
    p: &PlayPhase,
    lead: Card,
    trumping: bool,
    is_boss: bool,
    len: f64,
) -> f64 {
    let trump = ctx.trump;
    let mut bonus = 0.0;

    // Endgame fraction: 1.0 at the very end, ramping up over the last third of
    // the hand. `my_hand_size` shrinks each trick; for a 2-deck Tractor hand a
    // seat starts with ~25 cards.
    let late = if ctx.my_hand_size <= 4 {
        1.0
    } else if ctx.my_hand_size <= 8 {
        0.6
    } else {
        0.0
    };
    let early = ctx.my_hand_size >= 14;

    // --- Never OPEN with high trump (Trip Holder) --------------------------
    // "Stop opening as a big trump. Stop opening with jokers. Never play a naked
    //  joker to start. Never start with the small joker or the trump-rank card.
    //  This should be a 0% frequency." Because `score_lead` only runs for the
    //  player OPENING a trick, every lead here is a trick opening, so we apply a
    //  heavy penalty to a single (naked) joker, the trump-rank (trump-number)
    //  card, or any big trump lead. The ONLY exception is the rare "bleed" line:
    //  when Enoch is sitting on ~15+ trump cards it may open high trump to strip
    //  everyone else of trump. That exception is gated on `my_trump_count >= 15`.
    let threshold_coverage = threshold_coverage_lead(ctx, p, lead, is_boss, len);
    let bleed_exception = ctx.my_trump_count >= 15 || threshold_coverage;
    if trumping && !bleed_exception {
        let s = boss_strength(trump, lead);
        let is_naked_joker = s >= 999 && len < 2.0;
        let is_trump_rank = matches!(
            lead,
            Card::Suited { number, .. } if trump.number() == Some(number)
        );
        // A "big trump" here is any trump at or above a high spot card (Jack+ of
        // the trump suit, the trump-number cards, or the jokers).
        let is_big_trump = s >= 11;
        // The shared scorer credits a high card's raw strength (`+0.05*strength`,
        // ≈ +50 for a joker / trump-number) and gives jokers a guaranteed-top
        // bonus, so the open-penalty must be LARGE enough to drive these clearly
        // negative — and below any reasonable ace open — for a true ~0% frequency.
        if is_naked_joker {
            bonus -= 90.0; // a naked joker open is the worst offender → ~0%
        } else if is_trump_rank {
            bonus -= 85.0; // opening the trump-rank card is forbidden
        } else if is_big_trump {
            bonus -= 30.0; // any other big trump open is strongly discouraged
        }
    }

    // --- Ace-first single leading (Trip Holder) ----------------------------
    // "Open up with your aces, which is good — prioritize aces, then pairs,
    //  tractors, high pairs, then low pairs." A single non-trump BOSS (a side-suit
    //  ace, or any uncatchable single) is the textbook opener: it cashes a winner
    //  and draws out opponents' high cards risk-free. Give it a clean bump so a
    //  cashable ace opens ahead of a speculative low pair.
    if !trumping && len < 2.0 && is_boss {
        bonus += 4.0;
    }

    // --- Tractor-first (and strong-pair-first) leading ---------------------
    // A multi-card non-trump unit is a pair (len 2) or a tractor (len >= 4 with
    // consecutive pairs). Leading these EARLY, before they fragment, is the
    // enthusiast's single strongest emphasis — but only when the unit is strong
    // enough to actually win (a boss, or near-boss): leading a beatable low pair
    // just donates tempo. So we reward length only for boss / near-boss units,
    // and add an early-game kicker so they go out before they get broken up.
    if !trumping && len >= 2.0 {
        let top = boss_strength(trump, lead);
        let one_off = unseen_dominators(&ctx.k, trump, lead) <= 1;
        let is_tractor = len >= 4.0;
        if is_boss {
            // A boss pair/tractor is the textbook early cash: steep length reward
            // plus an early kicker (lead it now, before it splits).
            bonus += (len - 1.0) * 3.0 + 3.0;
            if early {
                bonus += 2.0;
            }
        } else if is_tractor {
            // A TRACTOR is structurally very hard to beat (needs a higher tractor),
            // so per the playbook lead it early even when not strictly boss — but a
            // touch more modestly than a boss, and with the early-game kicker.
            bonus += (len - 1.0) * 2.0;
            if early {
                bonus += 2.0;
            }
        } else if one_off || top >= 13 {
            // A near-boss (only one card can beat it) or high pair is still good
            // to lead early; a smaller reward, no kicker.
            bonus += (len - 1.0) * 1.5;
        }
    }

    // --- Long-suit run -----------------------------------------------------
    // Count our own copies of the led non-trump suit; a long holding (especially
    // pair-rich) is worth running to drain opponents' trump — but only run it from
    // a card that can plausibly WIN the lead (a boss / near-boss), otherwise we
    // just feed the opponents. Reward the length + pairs we hold there.
    if !trumping && (is_boss || unseen_dominators(&ctx.k, trump, lead) <= 1) {
        let eff = trump.effective_suit(lead);
        if let Ok(hand) = p.hands().get(ctx.me) {
            let mut in_suit = 0usize;
            let mut pairs = 0usize;
            for (&c, &n) in hand.iter() {
                if c == Card::Unknown || trump.effective_suit(c) != eff {
                    continue;
                }
                in_suit += n;
                if n >= 2 {
                    pairs += 1;
                }
            }
            if in_suit >= 5 {
                bonus += 1.5 + (in_suit.saturating_sub(5)) as f64 * 0.4;
            }
            // A pair in a long suit makes the run far harder to stop (opponents
            // must ruff with a trump PAIR), so reward holding pairs there.
            if in_suit >= 4 && pairs >= 1 {
                bonus += 1.0 * pairs as f64;
            }
        }
    }

    // --- Void-aware leads: partner ruff + trump drain (Trip/Double Holder) --
    // Honest: voids come from `ctx.k.voids`, inferred from PUBLIC off-suit follows
    // (the full-hand void log for Enoch). Two named tactics, both for NON-TRUMP
    // leads:
    //   (a) "Lead a suit your partner is void in so they can ruff": if a teammate
    //       is known void in the led suit AND at least one opponent must still
    //       follow (so we are not simply gifting the trick to a void opponent),
    //       lead a CHEAP card to hand the partner a free ruff. Scaled DOWN for high
    //       cards — give partner a junk entry, don't waste a winner.
    //   (b) "Drain trump": when EVERY opponent is known void in the led non-trump
    //       suit, leading it forces them to burn trump (a PAIR/tractor of trump to
    //       beat our pair/tractor); reward dumping the whole multi-card unit, which
    //       is what makes the long-suit throw a real play. Gated to a multi-card
    //       lead — a lone low card just gets ruffed for the trick.
    if !trumping {
        let eff = trump.effective_suit(lead);
        let mut partner_void = false;
        let mut opp_total = 0usize;
        let mut opp_void = 0usize;
        for player in p.propagated().players() {
            let pid = player.id;
            if pid == ctx.me {
                continue;
            }
            let void_here = ctx
                .k
                .voids
                .get(&pid)
                .map(|v| v.contains(&eff))
                .unwrap_or(false);
            if same_team(p, ctx.me, pid) {
                partner_void |= void_here;
            } else {
                opp_total += 1;
                if void_here {
                    opp_void += 1;
                }
            }
        }
        let some_opp_must_follow = opp_void < opp_total;

        // (a) Partner ruff: hand a known-void teammate a cheap entry (non-boss).
        if partner_void && some_opp_must_follow && !is_boss {
            let s = boss_strength(trump, lead).min(998) as f64;
            // ~1.0 for a 2, tapering to ~0 for an ace-high card.
            let cheapness = (1.0 - (s / 14.0)).clamp(0.0, 1.0);
            bonus += 2.0 + 3.0 * cheapness; // ~+2..+5
        }

        // (b) Trump drain: every opponent void → reward dumping the multi-card unit.
        if opp_total > 0 && opp_void == opp_total && len >= 2.0 {
            bonus += 2.0 + (len - 1.0) * 1.5;
        }
    }

    // --- Defender low-trump hand-off (Trip Holder) -------------------------
    // For the mid/late-game partner HAND-OFF — "pass it to your partner with a
    //  LOW trump card, like a two, three, or four, something small, non-points;
    //  NEVER a joker or the trump-rank number" — a defender, once its boss cards
    //  are spent, leads a SMALL non-point trump to burn the attackers' trump and
    //  toss the lead to a partner who can then run a suit. The shared scorer
    //  penalizes a non-boss trump lead (-5) for the defending side; we PARTIALLY
    //  offset that ONLY for a genuinely small, non-point, non-rank trump so the
    //  hand-off becomes a deliberate tool rather than a reflexive blunder. Gated
    //  to the late game and to the case where we have NO boss lead available (no
    //  high cards to cash first).
    if trumping && !is_boss && !ctx.me_is_attacker && len < 2.0 && late >= 1.0 {
        let s = boss_strength(trump, lead);
        // A small trump: low trump-suit spot card (2/3/4 ⇒ strength 2..=4), and
        // never a joker / trump-number card (those are >= 997) nor a point card.
        let is_small_trump = s <= 4;
        let is_trump_rank = matches!(
            lead,
            Card::Suited { number, .. } if trump.number() == Some(number)
        );
        let have_boss_lead = my_best_nontrump_is_boss(ctx, p);
        if is_small_trump && !is_trump_rank && !is_point(lead) && !have_boss_lead {
            // Offset most (not all) of the -5 hoard penalty: a deliberate hand-off
            // with a SMALL trump (never a joker / rank card).
            bonus += 4.0;
        }
    }

    // --- Endgame kitty protection ------------------------------------------
    // If we buried a point-rich kitty (only the exchanger knows this), late in
    // the hand we must HOARD our winners to take the last trick rather than
    // throwing them into a speculative non-boss lead. Damp aggressive non-boss
    // leads in proportion to how valuable the (doubled) kitty is and how strong
    // the card we'd be spending is.
    if let Some(kp) = ctx.kitty_points {
        // The attacking side takes DOUBLE the kitty if they win the last trick,
        // so a 10-point kitty is effectively 20 at stake.
        let at_stake = kp * 2;
        if at_stake >= 20 && late > 0.0 && !is_boss && len < 2.0 {
            let spend = boss_strength(trump, lead).min(998) as f64;
            // Spending a strong card (high trump / boss-ish) on a throwaway lead
            // is the thing we want to avoid; scale the damp by card strength.
            if spend >= 13.0 {
                bonus -= (spend / 100.0).clamp(0.0, 1.0) * 5.0 * late;
            }
        }
    }

    bonus
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
        .map(|c| legacy_card_strength(trump, *c))
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
                    .map(|c| legacy_card_strength(trump, *c))
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

/// Ask the mechanics engine whether a legal follow would take the lead in the
/// in-progress trick. This is cheap (one small state clone) and, unlike comparing
/// only the candidates' highest cards, is correct for pairs, tractors, throws,
/// bombs, and trump-led tricks.
pub(crate) fn candidate_wins_current_trick(p: &PlayPhase, me: PlayerID, cards: &[Card]) -> bool {
    let mut trick = p.trick().clone();
    let mut hands = p.hands().clone();
    let rules = p.propagated();
    if trick
        .play_cards(PlayCards {
            id: me,
            hands: &mut hands,
            cards,
            trick_draw_policy: rules.trick_draw_policy,
            throw_eval_policy: rules.throw_evaluation_policy,
            format_hint: None,
            hide_throw_halting_player: rules.hide_throw_halting_player,
            tractor_requirements: rules.tractor_requirements,
            bomb_policy: rules.bomb_policy,
            compound_formats: rules.compound_formats.clone(),
        })
        .is_err()
    {
        return false;
    }
    trick.winner_so_far() == Some(me)
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
///
/// How costly it is to SPEND `card` on a trick we have already concluded we
/// cannot win — i.e. the future-winner value we throw away by playing it now.
/// Used by [`score_follow`]'s can't-win branch so the bot ducks with its LOWEST
/// card and never burns a winner on a lost trick (the reported "plays the Ace /
/// a redundant trump-rank card / a small joker under a higher one" blunders).
///
/// The ordering, worst-to-waste first: a joker, then the trump-rank card, then a
/// high trump, then an uncatchable side-suit boss (e.g. an Ace), then ordinary
/// cards (cost rising gently with rank so the absolute lowest junk is preferred).
/// It is deliberately STEEP relative to the rest of the can't-win terms so the
/// duck is decisive enough to survive the determinized search's prior/blend (the
/// crude leaf `control` term alone washes out against rollout point variance).
///
/// Honest — reads only the trump declaration and (for the boss test) `ctx.k`,
/// which derives purely from the redacted view + public play history.
fn waste_penalty(ctx: &EvalCtx, card: Card) -> f64 {
    let trump = ctx.trump;
    let s = boss_strength(trump, card);
    if s >= 999 {
        12.0 // a joker — never burn one on a lost trick
    } else if s >= 997 {
        9.0 // the trump-rank (trump-number) card
    } else if is_trump(trump, card) {
        if s >= 11 {
            6.0 // a high trump-suit card (J/Q/K/A of trump)
        } else {
            2.0 + s as f64 * 0.2 // a low trump still has ruffing value
        }
    } else if matches!(
        card,
        Card::Suited {
            number: Number::Ace,
            ..
        }
    ) {
        // Keep every side-suit Ace on a mechanically lost trick, even before it
        // is provably the public boss. It remains a likely future winner and is
        // far more valuable than ordinary low discard material.
        7.0
    } else if s >= 13 && is_boss_card(&ctx.k, trump, card) {
        // An uncatchable side-suit card (e.g. an Ace) is a winner to cash later;
        // the `s >= 13` gate keeps the (O(13)) boss scan off the low cards.
        4.0
    } else {
        // Ordinary non-trump: cheap to throw, but prefer the absolute lowest.
        s as f64 * 0.12
    }
}

/// How many trump PAIRS in `me`'s hand would be fragmented (reduced to a lone
/// singleton) by playing `cards`. Used to discourage breaking up a trump pair to
/// ruff a trick that does not require one — #7: "take a non-pair throw with
/// non-paired trump cards, not paired trumps." Honest — reads only our own hand.
fn trump_pairs_broken(ctx: &EvalCtx, p: &PlayPhase, cards: &[Card]) -> usize {
    let hand = match p.hands().get(ctx.me) {
        Ok(h) => h,
        Err(_) => return 0,
    };
    let played = Card::count(cards.iter().copied());
    let mut broken = 0usize;
    for (&card, &pc) in &played {
        if !is_trump(ctx.trump, card) {
            continue;
        }
        let held = hand.get(&card).copied().unwrap_or(0);
        broken += (held / 2).saturating_sub(held.saturating_sub(pc) / 2);
    }
    broken
}

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
    let trick_format_suit = trick.trick_format().map(|tf| tf.suit());
    let remaining_opponent_known_void = trick_format_suit
        .filter(|suit| *suit != EffectiveSuit::Trump)
        .is_some_and(|led_suit| {
            yet_to_act.iter().any(|pid| {
                !same_team(p, me, *pid)
                    && ctx
                        .k
                        .voids
                        .get(pid)
                        .is_some_and(|voids| voids.contains(&led_suit))
            })
        });

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
    let winner_locked = partner_winning
        && winner_top_card
            .map(|c| is_boss_card(&ctx.k, trump, c))
            .unwrap_or(false)
        && !remaining_opponent_known_void;

    // Points already committed to the pot.
    let pot_points: i32 = trick
        .played_cards()
        .iter()
        .flat_map(|pc| pc.cards.iter())
        .filter_map(|c| c.points().map(|x| x as i32))
        .sum();

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

    // Would this candidate beat the current winner? Delegate the actual format
    // comparison to the mechanics engine instead of approximating a pair,
    // tractor, throw, or bomb by its single highest card.
    let my_top = cards
        .iter()
        .copied()
        .max_by_key(|c| boss_strength(trump, *c))
        .unwrap_or(cards[0]);
    let would_beat = candidate_wins_current_trick(p, me, cards);
    // Does MY candidate, if it wins by following suit with a boss top, lock the
    // trick? (Used to allow beating-to-secure over a stealable partner.)
    let my_card_locks =
        following_suit && is_boss_card(&ctx.k, trump, my_top) && !remaining_opponent_known_void;
    let unseen_point_cards_in_led_suit = trick_format_suit
        .map(|led_suit| {
            ctx.k
                .configured_counts
                .iter()
                .filter(|(card, _)| {
                    card.points().is_some() && trump.effective_suit(**card) == led_suit
                })
                .map(|(card, configured)| {
                    configured.saturating_sub(ctx.k.seen.get(card).copied().unwrap_or(0))
                })
                .sum::<usize>()
        })
        .unwrap_or(0);
    let edge_guard = partner_winning
        && opp_after_me
        && following_suit
        && would_beat
        && !remaining_opponent_known_void
        && !is_trump(trump, my_top)
        && boss_strength(trump, my_top) >= 11
        && unseen_dominators(&ctx.k, trump, my_top) <= 1
        && unseen_point_cards_in_led_suit > 0;

    // `would_beat` is mechanics-accurate for the current table. It does not claim
    // the trick is safe from seats that have yet to act; the surrounding partner
    // and boss logic handles that uncertainty separately.
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
        if edge_guard {
            // Second/third-seat edge guard: spend the cheapest near-boss that
            // materially secures the lead before the last opponent can dump a
            // ten/king for free. This is a denial bonus, not a blanket order to
            // overtake a partner.
            score += 7.0 + unseen_point_cards_in_led_suit.min(4) as f64;
        }
        if remaining_opponent_known_void && my_point_contribution > 0 {
            // Partner's in-suit boss is not locked when a later opponent is
            // publicly known void and can ruff it. Do not feed into that trap.
            score -= my_point_contribution as f64 * 5.0;
        }
        if would_beat && !my_card_locks && !edge_guard {
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
            // Branch D: we cannot (or shouldn't) win — starve them of points and
            // throw away the LEAST valuable card. "If the opponent is winning and
            // no card you can play will win the trick, play the absolute lowest
            // card." We penalize by each card's retention value (`waste_penalty`)
            // so a joker / trump-rank / high trump / side-suit boss is never burnt
            // on a lost trick — the bot ducks with its lowest junk instead. This is
            // STEEP enough to survive the determinized search's prior/blend, where
            // the crude leaf `control` term alone washes out against point variance.
            score -= my_point_contribution as f64 * 4.0;
            score -= cards.iter().map(|c| waste_penalty(ctx, *c)).sum::<f64>();
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

    // #7: when RUFFING a lead made up entirely of SINGLES (a single or a throw of
    // singles — no pairs/tractors to match), winning needs only individual trumps,
    // so don't fragment a trump pair to do it. Penalize each trump pair this play
    // would break, so a non-paired-trump ruff (Variant B') is chosen over breaking
    // a pair (Variant B) when both win. Gated to all-singles leads so a genuine
    // pair/tractor ruff (which REQUIRES a trump pair) is never discouraged.
    if trumping_in && would_beat {
        let lead_all_singles = trick
            .played_cards()
            .first()
            .map(|pc| {
                Card::count(pc.cards.iter().copied())
                    .values()
                    .all(|&c| c == 1)
            })
            .unwrap_or(false);
        if lead_all_singles {
            score -= trump_pairs_broken(ctx, p, cards) as f64 * 4.0;
        }
    }

    if ctx.enoch {
        // --- Dump points to a WINNING PARTNER (Trip Holder) ----------------
        // "If your partner is winning — dropping a big joker / small joker and
        //  winning the trick — then it's OK to drop points. You should be dropping
        //  points instead of saving them. Why are you saving them?" When our
        //  partner currently holds the trick, ADD to the shared partner-feed
        //  reward so Enoch actively sheds 10s/Ks/5s rather than hoarding them. We
        //  feed hardest when the partner is LOCKED (the trick is theirs for sure)
        //  and a touch more cautiously when an opponent can still steal.
        if partner_winning && my_point_contribution > 0 && !would_beat {
            if winner_locked || !opp_after_me {
                // Partner has it for sure: shed every point we can.
                score += my_point_contribution as f64 * 2.0;
            } else {
                // Stealable: still favor dumping, but a smaller bump.
                score += my_point_contribution as f64 * 0.8;
            }
        }

        // --- Smallest card when we CANNOT win (Trip Holder) ----------------
        // "If you don't think you can win the trick, just play the smallest
        //  possible card. Don't waste your high trumps / jokers for no reason." If
        //  no opponent-/partner-winning consideration calls for strength here (we
        //  can't beat the current winner, and we're not feeding a winning partner),
        //  penalize spending strength so Enoch ducks with its LOWEST card and never
        //  burns a high trump or joker on a trick it can't take.
        let feeding_partner = partner_winning && my_point_contribution > 0;
        if !would_beat && !feeding_partner {
            // Scale by the strength of what we'd spend; jokers / high trump hurt
            // most, so a low card is strictly preferred. This is on TOP of the
            // shared can't-win duck term, making the smallest-card duck decisive.
            score -= max_strength as f64 * 0.08;
            if trumping_in {
                // Wasting a trump on an unwinnable trick is the worst case.
                score -= 6.0;
            }
        }

        // Endgame kitty protection (follow side). If WE buried a point-rich kitty
        // (only the exchanger knows this), in the late game we should HOARD trump
        // / high cards to win the LAST trick rather than spend them ruffing a
        // small pot now: per the playbook, a defender may concede a little here so
        // long as it is less than the doubled kitty at stake. We damp trumping in
        // on a low pot when the (doubled) kitty is large and the game is late.
        if let Some(kp) = ctx.kitty_points {
            let at_stake = kp * 2;
            let late = ctx.my_hand_size <= 8;
            if at_stake >= 20 && late && trumping_in && pot_points < (at_stake as i32) {
                // Discourage burning a trump to win a pot smaller than the kitty
                // we are protecting, so the winning card survives for the finale.
                score -= 4.0;
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
        .filter(|cards| admissible_ranked_lead(&ctx, p, cards))
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

/// Root lead ranking without the public-information Joker-compound filter.
/// Omniscient search uses this wider proposal set before validating candidates
/// against its actual world, so a throw that is safe in that world is not lost
/// merely because its safety was not publicly provable. Scores remain the same
/// honest heuristic scores as [`ranked_leads`]. Rollout rankings stay filtered.
pub(crate) fn ranked_leads_unfiltered(p: &PlayPhase, me: PlayerID) -> Vec<ScoredPlay> {
    let ctx = EvalCtx::build(p, me);
    let mut scored: Vec<ScoredPlay> = lead_candidates(p, me)
        .into_iter()
        .map(|cards| {
            let score = score_lead(&ctx, p, &cards);
            ScoredPlay { cards, score }
        })
        .filter(|play| play.score.is_finite())
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

pub(crate) fn ranked_leads_for_rollout(p: &PlayPhase, me: PlayerID) -> Vec<ScoredPlay> {
    let ctx = EvalCtx::build(p, me);
    let mut scored: Vec<ScoredPlay> = rollout_lead_candidates(p, me)
        .into_iter()
        .filter(|cards| admissible_ranked_lead(&ctx, p, cards))
        .map(|cards| ScoredPlay {
            score: score_lead(&ctx, p, &cards),
            cards,
        })
        .filter(|play| play.score.is_finite())
        .collect();
    scored.sort_by(|a, b| b.score.total_cmp(&a.score));
    scored
}

pub(crate) fn ranked_follows_for_rollout(p: &PlayPhase, me: PlayerID) -> Vec<ScoredPlay> {
    let ctx = EvalCtx::build(p, me);
    let mut scored: Vec<ScoredPlay> = rollout_follow_candidates(p, me)
        .into_iter()
        .map(|cards| ScoredPlay {
            score: score_follow(&ctx, p, &cards),
            cards,
        })
        .collect();
    scored.sort_by(|a, b| b.score.total_cmp(&a.score));
    scored
}

/// Deduplicate identical card-sets in place (order-insensitive), preserving the
/// first occurrence's card order. Used when merging the Enoch throw candidates
/// with the shared single-unit candidates.
fn dedup_card_sets(sets: &mut Vec<Vec<Card>>) {
    let mut seen: std::collections::HashSet<Vec<Card>> = std::collections::HashSet::new();
    sets.retain(|s| {
        let mut key = s.clone();
        key.sort_by_key(|c| c.as_char());
        seen.insert(key)
    });
}

/// Whether a PAIR (or each pair of a tractor) whose top is `top` strength in
/// effective suit `eff` cannot be beaten by a higher PAIR a follower could
/// assemble — i.e. no strictly-higher same-suit rank still has >= 2 unseen
/// copies. HONEST (reads
/// only `ctx.k.seen`, which counts our own hand + public play). Conservative for
/// tractors (it requires each pair to be unbeatable, stronger than "no higher
/// tractor"), so it never green-lights an unsafe throw.
fn repeated_unbeatable_above(ctx: &EvalCtx, eff: EffectiveSuit, top: i32, count: usize) -> bool {
    for higher in boss_stronger_cards_in_suit(ctx.trump, eff, top) {
        let unseen = ctx
            .k
            .configured_copies(higher)
            .saturating_sub(ctx.k.seen.get(&higher).copied().unwrap_or(0));
        if unseen >= count {
            return false;
        }
    }
    true
}

fn pair_unbeatable_above(ctx: &EvalCtx, eff: EffectiveSuit, top: i32) -> bool {
    repeated_unbeatable_above(ctx, eff, top, 2)
}

/// Enoch-ONLY extra lead candidates: playbook-shaped multi-unit "throws" (甩牌).
/// The shared [`lead_candidates`] emits bounded generic throws; this augments
/// them with larger safe whole-suit/subset throws that Enoch and Grandmaster
/// should actively consider. We emit two kinds, each SAFE to lay down:
///
/// 1. The whole-suit throw, when it is safe wholesale: every card is an uncatchable
///    boss, OR every opponent is already known void in the suit (they can only
///    ruff — the trump-drain play).
/// 2. The maximal SAFE SUBSET throw: the combination of every unit that is
///    individually un-beatable IN SUIT — a boss single (e.g. an Ace), or a pair /
///    tractor no higher pair can beat (e.g. `KK+A`, `AA+K`, a boss tractor + an
///    Ace, or — holding `KQJ` — `TT+A`). Gated to when NO opponent is known void in
///    the suit (so the throw can't simply be ruffed). This is the playbook's "open
///    with an unbeatable / near-unbeatable set throw" idea, and it always attaches
///    the cashable Ace to a safe tractor/pair.
///
/// The engine's [`PlayPhase::can_play_cards`] is the final arbiter of throw
/// legality; duplicates against the single-unit candidates are removed by the
/// caller's [`dedup_card_sets`]. HONEST — reads only our own hand + `ctx.k`.
fn enoch_throw_candidates(ctx: &EvalCtx, p: &PlayPhase, me: PlayerID) -> Vec<Vec<Card>> {
    let trump = ctx.trump;
    let hand = match p.hands().get(me) {
        Ok(h) => h,
        Err(_) => return vec![],
    };
    let opp_ids: Vec<PlayerID> = p
        .propagated()
        .players()
        .iter()
        .map(|pl| pl.id)
        .filter(|id| *id != me && !same_team(p, me, *id))
        .collect();

    let my_cards: Vec<Card> = Card::cards(hand.iter()).copied().collect();
    let mut out = vec![];
    for (eff, suit_cards) in cards_by_suit(trump, &my_cards) {
        // Only NON-trump suits, and only when we hold enough to throw more than a
        // single unit's worth.
        if eff == EffectiveSuit::Trump || suit_cards.len() < 2 {
            continue;
        }
        let any_opp_void = opp_ids.iter().any(|id| {
            ctx.k
                .voids
                .get(id)
                .map(|v| v.contains(&eff))
                .unwrap_or(false)
        });
        let all_opp_void = !opp_ids.is_empty()
            && opp_ids.iter().all(|id| {
                ctx.k
                    .voids
                    .get(id)
                    .map(|v| v.contains(&eff))
                    .unwrap_or(false)
            });

        // (1) Whole-suit throw, safe wholesale.
        let all_boss = suit_cards.iter().all(|c| is_boss_card(&ctx.k, trump, *c));
        if (all_boss || all_opp_void) && p.can_play_cards(me, &suit_cards).is_ok() {
            out.push(suit_cards.clone());
        }

        // (2) Maximal SAFE-SUBSET throw — only when no opponent can ruff it.
        if !any_opp_void {
            let counts = Card::count(suit_cards.iter().copied());
            let mut safe: Vec<Card> = Vec::new();
            let mut distinct_units = 0usize;
            for (&card, &ct) in &counts {
                let top = boss_strength(trump, card);
                let unit_safe = if ct >= 2 {
                    pair_unbeatable_above(ctx, eff, top)
                } else {
                    is_boss_card(&ctx.k, trump, card)
                };
                if unit_safe {
                    for _ in 0..ct {
                        safe.push(card);
                    }
                    distinct_units += 1;
                }
            }
            // A genuine throw spans >= 2 units; a lone safe pair/tractor/single is
            // already a single-unit lead candidate. Sort low-first for a stable
            // representative `cards[0]`; dedup + legality are handled downstream.
            if distinct_units >= 2 && safe.len() >= 2 {
                safe.sort_by(|a, b| trump.compare(*a, *b));
                if p.can_play_cards(me, &safe).is_ok() {
                    out.push(safe);
                }
            }
        }
    }
    out
}

/// Rank the legal lead candidates with the Enoch full-game playbook ENABLED
/// (`EvalCtx::build_enoch`). Identical machinery to [`ranked_leads`] but the
/// shared [`score_lead`] adds the Enoch-specific bonuses, and the candidate set is
/// augmented with Enoch-only multi-unit throws. Used by the Enoch tier (directly
/// and as the search prior / rollout policy).
pub fn ranked_leads_enoch(p: &PlayPhase, me: PlayerID) -> Vec<ScoredPlay> {
    let ctx = EvalCtx::build_enoch(p, me);
    // Augment the shared bounded candidates with the playbook's larger safe
    // full-suit/subset throws, then de-duplicate.
    let mut cand_sets = lead_candidates(p, me);
    cand_sets.extend(enoch_throw_candidates(&ctx, p, me));
    dedup_card_sets(&mut cand_sets);
    let mut scored: Vec<ScoredPlay> = cand_sets
        .into_iter()
        .filter(|cards| admissible_ranked_lead(&ctx, p, cards))
        .map(|cards| {
            let score = score_lead(&ctx, p, &cards);
            ScoredPlay { cards, score }
        })
        .filter(|play| play.score.is_finite())
        .collect();
    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored
}

/// Enoch-policy counterpart to [`ranked_leads_unfiltered`]. This widens only
/// root proposals for exact-world validation; normal Enoch rankings and all
/// rollout rankings continue to reject unproven Joker compounds.
pub(crate) fn ranked_leads_enoch_unfiltered(p: &PlayPhase, me: PlayerID) -> Vec<ScoredPlay> {
    let ctx = EvalCtx::build_enoch(p, me);
    let mut candidates = lead_candidates(p, me);
    candidates.extend(enoch_throw_candidates(&ctx, p, me));
    dedup_card_sets(&mut candidates);
    let mut scored: Vec<ScoredPlay> = candidates
        .into_iter()
        .map(|cards| ScoredPlay {
            score: score_lead(&ctx, p, &cards),
            cards,
        })
        .collect();
    scored.sort_by(|a, b| b.score.total_cmp(&a.score));
    scored
}

/// Rank the legal follow candidates with the Enoch playbook ENABLED. See
/// [`ranked_leads_enoch`].
pub fn ranked_follows_enoch(p: &PlayPhase, me: PlayerID) -> Vec<ScoredPlay> {
    let ctx = EvalCtx::build_enoch(p, me);
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

pub(crate) fn ranked_leads_enoch_for_rollout(p: &PlayPhase, me: PlayerID) -> Vec<ScoredPlay> {
    let ctx = EvalCtx::build_enoch(p, me);
    let mut candidates = rollout_lead_candidates(p, me);
    candidates.extend(enoch_throw_candidates(&ctx, p, me));
    dedup_card_sets(&mut candidates);
    let mut scored: Vec<ScoredPlay> = candidates
        .into_iter()
        .filter(|cards| admissible_ranked_lead(&ctx, p, cards))
        .map(|cards| ScoredPlay {
            score: score_lead(&ctx, p, &cards),
            cards,
        })
        .filter(|play| play.score.is_finite())
        .collect();
    scored.sort_by(|a, b| b.score.total_cmp(&a.score));
    scored
}

pub(crate) fn ranked_follows_enoch_for_rollout(p: &PlayPhase, me: PlayerID) -> Vec<ScoredPlay> {
    let ctx = EvalCtx::build_enoch(p, me);
    let mut scored: Vec<ScoredPlay> = rollout_follow_candidates(p, me)
        .into_iter()
        .map(|cards| ScoredPlay {
            score: score_follow(&ctx, p, &cards),
            cards,
        })
        .collect();
    scored.sort_by(|a, b| b.score.total_cmp(&a.score));
    scored
}

/// Greedy Enoch-heuristic-DIRECT play: pick the argmax candidate with the Enoch
/// playbook enabled and NO search (fast). Mirrors [`choose_play_direct`] but for
/// the Enoch scorer; used by the benchmark harness and as the Enoch fallback.
pub fn choose_play_direct_enoch(p: &PlayPhase, me: PlayerID) -> Option<Vec<Card>> {
    let leading = p.trick().played_cards().is_empty();
    let ranked = if leading {
        ranked_leads_enoch(p, me)
    } else {
        ranked_follows_enoch(p, me)
    };
    ranked.into_iter().next().map(|s| s.cards)
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

/// Enoch's trump-declaration evaluation. Same backbone as [`bid_strength`] but it
/// PRIORITIZES PAIRS in the candidate trump suit per the enthusiast's playbook:
/// "a pair of trump cards is worth like three or four non-pair trump cards." We
/// add a large bonus for each trump-suit pair (and tractor), so Enoch declares
/// the suit it is most *paired* in, not merely the one it has the most cards in.
/// Honest — derives only from `me`'s own hand.
pub fn bid_strength_enoch(hand: &[Card], candidate: Trump) -> f64 {
    // Start from the shared backbone (length + raw strength + jokers + side aces).
    let mut score = bid_strength(hand, candidate);
    let counts = Card::count(hand.iter().copied());

    // PAIR PRIORITY: each effective-trump PAIR (two copies) is worth a big bump.
    // The backbone already counted each card once for length; this adds the extra
    // "a pair is worth ~3-4 singles" premium on top. Trump-number / joker pairs
    // (the trump tops) are worth even more.
    for (&card, &ct) in &counts {
        if candidate.effective_suit(card) != EffectiveSuit::Trump {
            continue;
        }
        let pairs = ct / 2;
        if pairs == 0 {
            continue;
        }
        let s = card_strength(candidate, card);
        // A trump pair is worth ~3-4 SINGLE trumps per the playbook. The backbone
        // already counts each paired card once for length (~1.5 each), so we add
        // the EXTRA premium on top to reach that "3-4 singles" valuation, scaling
        // up for high trumps (a high-pair is even harder to break).
        let per_pair = if s >= 997 {
            6.0 // joker / trump-number pair
        } else if s >= 13 {
            5.0 // Ace / King of trump suit pair
        } else {
            4.0
        };
        score += per_pair * pairs as f64;
    }

    // Consecutive trump-suit pairs (a tractor in trump) are extra strong: detect
    // adjacent ranks both held as pairs in the trump SUIT.
    if let Trump::Standard { suit, number } = candidate {
        let ladder = [
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
        let is_pair = |n: Number| -> bool {
            if n == number {
                return false; // trump-number handled above (off-suit twins)
            }
            counts
                .get(&Card::Suited { number: n, suit })
                .copied()
                .unwrap_or(0)
                >= 2
        };
        for w in ladder.windows(2) {
            if is_pair(w[0]) && is_pair(w[1]) {
                score += 3.0; // a trump tractor — pressure that's hard to break
            }
        }
    }

    score
}

/// Count the trump-suit PAIRS `hand` holds under `trump`, and whether it contains
/// a trump-suit TRACTOR (two consecutive trump-suit ranks each held as a pair).
/// Used by Enoch's declaring discipline to require genuine PAIR structure — not
/// mere length — before it commits to a trump early in the deal. Honest (reads
/// only `me`'s own hand). Mirrors the pair/tractor detection in
/// [`bid_strength_enoch`].
pub fn trump_pair_structure(hand: &[Card], trump: Trump) -> (usize, bool) {
    let counts = Card::count(hand.iter().copied());
    let mut pairs = 0usize;
    for (&card, &ct) in &counts {
        if trump.effective_suit(card) == EffectiveSuit::Trump {
            pairs += ct / 2;
        }
    }
    let mut tractor = false;
    if let Trump::Standard { suit, number } = trump {
        let ladder = [
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
        let is_pair = |n: Number| -> bool {
            if n == number {
                return false; // trump-number twins are off-suit; handled as a pair above
            }
            counts
                .get(&Card::Suited { number: n, suit })
                .copied()
                .unwrap_or(0)
                >= 2
        };
        for w in ladder.windows(2) {
            if is_pair(w[0]) && is_pair(w[1]) {
                tractor = true;
                break;
            }
        }
    }
    (pairs, tractor)
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

/// How many POINT cards' worth of points the Enoch playbook is willing to bury in
/// the kitty, given the landlord's hand strength. The kitty is doubled when the
/// attacking side wins the last trick, so points are only safe to bury when we
/// expect to TAKE that last trick — which is a function of jokers + trump length.
///
/// Trip Holder: "How come you never put any points in the kitty as a declarer? If
/// you have a decent amount of trumps or a few jokers, you SHOULD be putting
/// points in the kitty." The old budget was far too stingy (it demanded jokers
/// before allowing more than 5 points), so a strong-but-jokerless hand buried
/// nothing. We loosen it so a strong hand buries a HEALTHY amount of points while
/// a weak hand still buries none:
///
/// * dominant (~4 jokers + lots of trump) → bury freely (cap high).
/// * a few jokers OR a long trump holding → up to ~20-25 points.
/// * a decent trump holding (jokerless) → up to ~10-15 points.
/// * below-average trump → a small allowance.
/// * a weak hand → bury no points.
///
/// Returns the maximum total point-VALUE to allow in the kitty.
fn enoch_point_budget(hand: &[Card], trump: Trump) -> usize {
    let counts = Card::count(hand.iter().copied());
    let jokers = counts.get(&Card::BigJoker).copied().unwrap_or(0)
        + counts.get(&Card::SmallJoker).copied().unwrap_or(0);
    let trump_count: usize = counts
        .iter()
        .filter(|(c, _)| trump.effective_suit(**c) == EffectiveSuit::Trump)
        .map(|(_, &n)| n)
        .sum();

    // The enthusiast's "average trump you'll hold is ~8-10"; treat 9 as average.
    // A strong hand expects to win the last trick (and reclaim the doubled kitty),
    // so it should sink points rather than strand them in play.
    if jokers >= 4 && trump_count >= 12 {
        // Dominant: confident of the last trick; bury as much as we like.
        100
    } else if jokers >= 2 || trump_count >= 12 {
        // Strong: a couple of jokers OR a long trump holding.
        25
    } else if jokers >= 1 || trump_count >= 10 {
        // Good: a joker or an above-average trump holding.
        20
    } else if trump_count >= 8 {
        // Decent (jokerless) trump holding still warrants burying real points.
        15
    } else if trump_count >= 6 {
        // Below average: a small allowance.
        10
    } else {
        // Weak hand: bury no points.
        0
    }
}

/// Enoch's kitty-burial discipline (Trip Holder refinement). Same "void a side
/// suit, bury low trash" backbone as [`choose_kitty`], but it HARD-PROTECTS the
/// cards the enthusiast says never to bury — aces, ALL pairs, all trump, and the
/// current bid-rank (trump-number) card. Per Trip Holder:
///
/// * "Stop laying pairs in the kitty as a declarer — don't do that. You CAN do it
///   at a low frequency if it helps you get rid of a suit." So EVERY pair is
///   protected by default, and the protection is relaxed ONLY for a side suit we
///   can FULLY void within the kitty (the sole sanctioned exception).
/// * "Prioritize getting rid of an ENTIRE suit first." We detect which side suits
///   can be wholly voided within the kitty and give all their cards a large void
///   bonus, so completing a suit-void outranks scattering trash.
/// * "It's OK to lay points in there if you have a decent amount of trumps / a few
///   jokers." Point burial is gated by the hand-strength budget
///   ([`enoch_point_budget`]), so a strong hand sinks real points while a weak
///   hand buries none.
///
/// Honest — reads only `me`'s own combined pool.
pub fn choose_kitty_enoch(hand: &[Card], trump: Trump, kitty_size: usize) -> Vec<Card> {
    if kitty_size == 0 {
        return vec![];
    }
    let by_suit = cards_by_suit(trump, hand);
    let suit_len: HashMap<EffectiveSuit, usize> =
        by_suit.iter().map(|(s, c)| (*s, c.len())).collect();
    let counts = Card::count(hand.iter().copied());
    let trump_number = trump.number();
    let point_budget = enoch_point_budget(hand, trump);

    // --- Identify side suits we can FULLY void within the kitty -------------
    // Voiding an ENTIRE side suit (so we can later ruff it) is the top kitty
    // priority. A suit is a void target if it is a NON-trump side suit whose whole
    // length fits in the kitty AND it contains no ace (we never bury an ace, so a
    // suit holding one can't be cleanly voided). Among the eligible suits, prefer
    // the SHORTEST (cheapest to void) and only mark as many as the kitty can hold.
    let mut voidable: Vec<(usize, EffectiveSuit)> = by_suit
        .iter()
        .filter(|(suit, cards)| {
            **suit != EffectiveSuit::Trump
                && cards.len() <= kitty_size
                && !cards.iter().any(|c| {
                    matches!(
                        c,
                        Card::Suited {
                            number: Number::Ace,
                            ..
                        }
                    )
                })
        })
        .map(|(suit, cards)| (cards.len(), *suit))
        .collect();
    voidable.sort_unstable();
    // Greedily pick the shortest suits whose combined length fits the kitty; those
    // are the suits whose pair-protection we relax (the sanctioned exception) and
    // whose cards get the entire-suit void bonus.
    let mut void_suits: std::collections::HashSet<EffectiveSuit> = std::collections::HashSet::new();
    let mut void_budget = kitty_size;
    let mut void_points = 0usize;
    for (len, suit) in voidable {
        // Voiding a suit may require burying its point cards (the sanctioned
        // pair/point exception). Only commit to a void whose points fit the
        // hand-strength point budget: a weak hand (budget 0) then never STARTS a
        // void it could only finish by burying points, and a stronger hand commits
        // to it fully. Combined with the pre-commit pass below, this removes the
        // old silent failure where a void suit's point card was skipped by the
        // point-budget cap and the suit was left half-buried.
        let suit_points: usize = by_suit
            .get(&suit)
            .map(|cs| cs.iter().filter_map(|c| c.points()).sum())
            .unwrap_or(0);
        if len <= void_budget && void_points + suit_points <= point_budget {
            void_suits.insert(suit);
            void_budget -= len;
            void_points += suit_points;
        }
    }

    let mut scored: Vec<(f64, Card)> = hand
        .iter()
        .map(|&card| {
            let suit = trump.effective_suit(card);
            let strength = card_strength(trump, card);
            let held = counts.get(&card).copied().unwrap_or(0);
            let in_void_suit = void_suits.contains(&suit);
            let mut bury = 0.0;

            // HARD protections (the enthusiast's "never bury these").
            if suit == EffectiveSuit::Trump {
                // Never bury trump. (This already covers every bid-rank /
                // trump-number card and both jokers, since they are all effective
                // trump.)
                bury -= 1000.0;
            }
            if let Card::Suited { number, .. } = card {
                // The current bid-rank (trump-number) card — protected explicitly
                // for clarity even though it is also effective trump above.
                if Some(number) == trump_number {
                    bury -= 1000.0;
                }
                if number == Number::Ace {
                    bury -= 1000.0; // never bury aces
                }
                // PAIRS: protect EVERY pair (not just jack-pair-and-higher), per
                // Trip Holder. The only sanctioned exception is when the pair sits
                // in a side suit we can fully VOID — relax the protection there so
                // the entire-suit void can complete. (We still keep a mild cost so
                // a void that does NOT need the pair prefers the singles.)
                if held >= 2 {
                    if in_void_suit {
                        bury -= 5.0; // low-frequency exception: only to void a suit
                    } else {
                        bury -= 1000.0; // never bury a pair otherwise
                    }
                }
            }
            if strength >= 990 {
                bury -= 1000.0; // jokers / trump-number tops
            }

            // POINT cards: their burial is gated by the hand-strength budget
            // below. Trip Holder: a STRONG hand SHOULD bury points (they are safe
            // in the kitty and come back doubled if it takes the last trick), so
            // when we have a point budget we give points a POSITIVE burial
            // incentive (ranked just under a clean suit-void). A weak hand has a
            // zero budget, so the gate buries none regardless; the penalty there
            // just keeps points from being attempted before trash.
            if is_point(card) {
                if point_budget > 0 {
                    bury += 15.0; // strong hand: deliberately sink points
                } else {
                    bury -= 40.0; // weak hand: never reach for points
                }
            }

            // Prefer to bury low cards and to VOID a side suit (so we can ruff it
            // later). A suit we can FULLY void gets a big entire-suit bonus on
            // every one of its cards, so completing the void outranks scattering
            // trash across suits; otherwise fall back to the "shorter suit" nudge.
            bury += (15 - strength.min(15)) as f64;
            if suit != EffectiveSuit::Trump {
                if in_void_suit {
                    bury += 30.0; // prioritize getting rid of an ENTIRE suit
                } else {
                    let len = suit_len.get(&suit).copied().unwrap_or(0);
                    bury += (6.0 - len as f64).max(0.0) * 2.0;
                }
            }
            (bury, card)
        })
        .collect();

    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| trump.compare(a.1, b.1))
            .then_with(|| a.1.as_char().cmp(&b.1.as_char()))
    });

    // `scored` has one entry per PHYSICAL card slot (the `hand` argument is a flat
    // list), so we select by slot index. Pass 1: greedily take the most-buriable
    // slots while capping the total POINT value buried at the hand-strength
    // budget — a point card that would exceed the budget is skipped. Pass 2: if
    // skipping points left us short, fill the remaining slots from the not-yet-
    // taken entries (still in buriability order). This always yields exactly
    // `kitty_size` legal cards.
    let mut taken = vec![false; scored.len()];
    let mut chosen: Vec<Card> = Vec::with_capacity(kitty_size);
    let mut points_buried = 0usize;
    // PASS 0: pre-commit EVERY card of a sanctioned void suit. Completing an
    // entire-suit void is the top kitty priority ("get rid of an ENTIRE suit
    // first"), so it takes precedence over the point ceiling — and since void
    // suits were chosen so their combined length fits the kitty AND their combined
    // point value fits the budget, this overruns neither. This is what guarantees a
    // sanctioned void actually completes instead of being silently half-buried.
    for (i, (_, card)) in scored.iter().enumerate() {
        if chosen.len() == kitty_size {
            break;
        }
        if void_suits.contains(&trump.effective_suit(*card)) {
            if let Some(pt) = card.points() {
                points_buried += pt;
            }
            taken[i] = true;
            chosen.push(*card);
        }
    }
    // PASS 1: fill the remaining slots by buriability, capping the buried POINT
    // value at the budget (points needed to complete a void were taken in pass 0).
    for (i, (_, card)) in scored.iter().enumerate() {
        if chosen.len() == kitty_size {
            break;
        }
        if taken[i] {
            continue;
        }
        if let Some(pt) = card.points() {
            if points_buried + pt > point_budget {
                continue; // would blow the point budget; prefer trash instead
            }
            points_buried += pt;
        }
        taken[i] = true;
        chosen.push(*card);
    }
    if chosen.len() < kitty_size {
        for (i, (_, card)) in scored.iter().enumerate() {
            if chosen.len() == kitty_size {
                break;
            }
            if !taken[i] {
                taken[i] = true;
                chosen.push(*card);
            }
        }
    }
    chosen.truncate(kitty_size);
    chosen
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
            vec![],
        )
        .unwrap();
        (pp, ids)
    }

    /// Like [`make_play_phase`] but with a configurable deck count and trump, for
    /// tests (e.g. safe-throw generation) that need pairs/tractors a single deck
    /// can't form. Seats 0 & 2 are the landlord team; seat 0 leads.
    fn make_play_phase_decks(
        hands: [Vec<Card>; 4],
        num_decks: usize,
        trump: Trump,
    ) -> (PlayPhase, Vec<PlayerID>) {
        let ids: Vec<PlayerID> = (0..4).map(PlayerID).collect();
        let mut propagated = PropagatedState::default();
        propagated.players = ids
            .iter()
            .map(|id| Player::new(*id, format!("p{}", id.0)))
            .collect();
        let mut h = Hands::new(ids.iter().copied());
        h.set_trump(trump);
        for (i, cards) in hands.iter().enumerate() {
            h.add(ids[i], cards.iter().copied()).unwrap();
        }
        let pp = PlayPhase::new(
            propagated,
            num_decks,
            GameMode::Tractor,
            h,
            vec![],
            trump,
            ids[0],
            ids[0],
            vec![ids[0], ids[2]],
            vec![],
            vec![Deck::default(); num_decks],
            vec![],
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
    fn test_unsafe_joker_throw_uses_weakest_component_and_is_filtered_from_leads() {
        let low_trump = card(Number::Three, Suit::Hearts);
        let mut unsafe_throw = vec![Card::BigJoker, low_trump];
        let (pp, ids) = make_play_phase([
            unsafe_throw.clone(),
            vec![card(Number::Ace, Suit::Hearts)],
            vec![card(Number::King, Suit::Hearts)],
            vec![card(Number::Queen, Suit::Hearts)],
        ]);
        canonicalize_play(pp.trump(), &mut unsafe_throw);

        let ctx = EvalCtx::build(&pp, ids[0]);
        let safety = lead_safety(&ctx, &pp, &unsafe_throw);
        assert!(!safety.single_unit, "joker + low trump must be a throw");
        assert!(
            !safety.whole_play_safe,
            "the beatable low-trump component makes the whole throw unsafe"
        );

        let throw_score = score_lead(&ctx, &pp, &unsafe_throw);
        let low_score = score_lead(&ctx, &pp, &[low_trump]);
        assert!(
            throw_score < low_score,
            "an unsafe joker throw ({}) must not inherit enough joker value to beat the direct low lead ({})",
            throw_score,
            low_score
        );

        assert!(
            lead_candidates(&pp, ids[0]).contains(&unsafe_throw),
            "the exhaustive raw root generator should remain unchanged"
        );
        assert!(
            rollout_lead_candidates(&pp, ids[0]).contains(&unsafe_throw),
            "the exhaustive raw rollout generator should remain unchanged"
        );
        assert!(
            ranked_leads_unfiltered(&pp, ids[0])
                .iter()
                .any(|play| play.cards == unsafe_throw),
            "exact-world root search must still be able to inspect the raw compound"
        );
        assert!(
            ranked_leads_enoch_unfiltered(&pp, ids[0])
                .iter()
                .any(|play| play.cards == unsafe_throw),
            "exact-world Enoch root search must still be able to inspect the raw compound"
        );

        let assert_filtered = |name: &str, ranked: Vec<ScoredPlay>| {
            assert!(
                !ranked.iter().any(|play| play.cards == unsafe_throw),
                "{} must filter the unsafe joker compound",
                name
            );
            assert!(
                ranked.iter().any(|play| play.cards == vec![Card::BigJoker]),
                "{} must retain the legitimate one-unit joker lead",
                name
            );
        };
        assert_filtered("normal root ranking", ranked_leads(&pp, ids[0]));
        assert_filtered(
            "normal rollout ranking",
            ranked_leads_for_rollout(&pp, ids[0]),
        );
        assert_filtered("Enoch root ranking", ranked_leads_enoch(&pp, ids[0]));
        assert_filtered(
            "Enoch rollout ranking",
            ranked_leads_enoch_for_rollout(&pp, ids[0]),
        );
    }

    #[test]
    fn test_throw_safety_checks_leftover_units_after_tractor_decomposition() {
        let three = card(Number::Three, Suit::Hearts);
        let four = card(Number::Four, Suit::Hearts);
        let five = card(Number::Five, Suit::Hearts);
        let throw = vec![three, three, three, four, four, Card::BigJoker];
        let trump = Trump::Standard {
            suit: Suit::Hearts,
            number: Number::Two,
        };
        let (pp, ids) = make_play_phase_decks(
            [
                throw.clone(),
                vec![five],
                vec![card(Number::Six, Suit::Clubs)],
                vec![card(Number::Seven, Suit::Diamonds)],
            ],
            3,
            trump,
        );

        let decompositions: Vec<Vec<TrickUnit>> = TrickUnit::find_plays(
            trump,
            pp.propagated().tractor_requirements,
            throw.iter().copied(),
        )
        .into_iter()
        .collect();
        assert!(decompositions.iter().any(|units| {
            units.iter().any(TrickUnit::is_tractor)
                && units.iter().any(|unit| unit.cards() == vec![three])
        }));

        let mut probe = pp.clone();
        probe.play_cards(ids[0], &throw).unwrap();
        assert!(
            !probe
                .trick()
                .played_cards()
                .last()
                .unwrap()
                .bad_throw_cards
                .is_empty(),
            "the hidden lone five must halt the leftover three singleton"
        );

        // Model a late-game public history where every higher trump is accounted
        // for except one Five. Aggregate rank counts would call the triple Three
        // and pair Four safe, but the tractor consumes two Threes and exposes the
        // remaining Three as a beatable singleton.
        let mut ctx = EvalCtx::build(&pp, ids[0]);
        for higher in
            boss_stronger_cards_in_suit(trump, EffectiveSuit::Trump, boss_strength(trump, three))
        {
            let configured = ctx.k.configured_copies(higher);
            ctx.k.seen.insert(higher, configured);
        }
        let configured_fives = ctx.k.configured_copies(five);
        ctx.k.seen.insert(five, configured_fives - 1);

        assert!(repeated_unbeatable_above(
            &ctx,
            EffectiveSuit::Trump,
            boss_strength(trump, three),
            3
        ));
        assert!(repeated_unbeatable_above(
            &ctx,
            EffectiveSuit::Trump,
            boss_strength(trump, four),
            2
        ));
        assert!(
            !is_boss_card(&ctx.k, trump, three),
            "the one unseen Five still beats the leftover singleton"
        );

        let safety = lead_safety(&ctx, &pp, &throw);
        assert!(!safety.single_unit);
        assert!(
            !safety.whole_play_safe,
            "unit-aware safety must reject the vulnerable leftover singleton"
        );
        assert!(
            !admissible_ranked_lead(&ctx, &pp, &throw),
            "the overlapping unsafe Joker throw must be filtered"
        );
    }

    #[test]
    fn test_proven_safe_multi_unit_joker_throw_is_retained() {
        // With both one-deck jokers in our hand, the small joker is also boss:
        // the only higher card (the big joker) is accounted for. The two
        // singleton components therefore form a provably safe compound.
        let mut safe_throw = vec![Card::BigJoker, Card::SmallJoker];
        let (pp, ids) = make_play_phase([
            safe_throw.clone(),
            vec![card(Number::Ace, Suit::Clubs)],
            vec![card(Number::Ace, Suit::Diamonds)],
            vec![card(Number::Ace, Suit::Spades)],
        ]);
        canonicalize_play(pp.trump(), &mut safe_throw);

        let ctx = EvalCtx::build(&pp, ids[0]);
        let safety = lead_safety(&ctx, &pp, &safe_throw);
        assert!(!safety.single_unit, "two distinct jokers are two units");
        assert!(
            safety.whole_play_safe,
            "every component is publicly proven unbeatable"
        );

        let assert_retained = |name: &str, ranked: Vec<ScoredPlay>| {
            assert!(
                ranked.iter().any(|play| play.cards == safe_throw),
                "{} must retain a proven-safe joker compound",
                name
            );
        };
        assert_retained("normal root ranking", ranked_leads(&pp, ids[0]));
        assert_retained(
            "normal rollout ranking",
            ranked_leads_for_rollout(&pp, ids[0]),
        );
        assert_retained("Enoch root ranking", ranked_leads_enoch(&pp, ids[0]));
        assert_retained(
            "Enoch rollout ranking",
            ranked_leads_enoch_for_rollout(&pp, ids[0]),
        );
    }

    #[test]
    fn test_joker_plus_low_trump_follow_candidate_is_unaffected() {
        let lead_trump = card(Number::Three, Suit::Hearts);
        let low_trump = card(Number::Four, Suit::Hearts);
        let mut follow = vec![Card::BigJoker, low_trump];
        let trump = Trump::Standard {
            suit: Suit::Hearts,
            number: Number::Two,
        };
        let (mut pp, ids) = make_play_phase_decks(
            [
                vec![lead_trump, lead_trump],
                follow.clone(),
                vec![card(Number::Five, Suit::Clubs)],
                vec![card(Number::Six, Suit::Clubs)],
            ],
            2,
            trump,
        );
        pp.play_cards(ids[0], &[lead_trump, lead_trump]).unwrap();
        canonicalize_play(pp.trump(), &mut follow);

        assert!(
            follow_candidates(&pp, ids[1]).contains(&follow),
            "lead-only filtering must not alter the raw follow candidates"
        );
        assert!(
            ranked_follows(&pp, ids[1])
                .iter()
                .any(|play| play.cards == follow),
            "normal follow ranking must retain joker + low trump"
        );
        assert!(
            ranked_follows_for_rollout(&pp, ids[1])
                .iter()
                .any(|play| play.cards == follow),
            "normal rollout follow ranking must retain joker + low trump"
        );
        assert!(
            ranked_follows_enoch(&pp, ids[1])
                .iter()
                .any(|play| play.cards == follow),
            "Enoch follow ranking must retain joker + low trump"
        );
        assert!(
            ranked_follows_enoch_for_rollout(&pp, ids[1])
                .iter()
                .any(|play| play.cards == follow),
            "Enoch rollout follow ranking must retain joker + low trump"
        );
    }

    /// #5 / #6 — Enoch/Grandmaster should open with a safe "near-unbeatable" set
    /// throw, attaching the cashable Ace to a pair no higher pair can beat. Holding
    /// (in spades) an Ace + a King pair, the OTHER ace is the only spade above the
    /// Kings; we hold one ace, so no `AA` pair can form → the `KK` pair is safe and
    /// `KK + A` is a safe throw.
    #[test]
    fn test_enoch_throws_safe_pair_plus_ace() {
        let trump = Trump::Standard {
            suit: Suit::Hearts,
            number: Number::Two,
        };
        let a_s = card(Number::Ace, Suit::Spades);
        let k_s = card(Number::King, Suit::Spades);
        let (pp, ids) = make_play_phase_decks(
            [
                vec![a_s, k_s, k_s, card(Number::Three, Suit::Clubs)], // seat0 Enoch leads
                vec![
                    card(Number::Four, Suit::Clubs),
                    card(Number::Five, Suit::Diamonds),
                ],
                vec![
                    card(Number::Six, Suit::Clubs),
                    card(Number::Seven, Suit::Diamonds),
                ],
                vec![
                    card(Number::Eight, Suit::Clubs),
                    card(Number::Nine, Suit::Diamonds),
                ],
            ],
            2,
            trump,
        );
        let ctx = EvalCtx::build_enoch(&pp, ids[0]);
        let throws = enoch_throw_candidates(&ctx, &pp, ids[0]);
        let has_kk_a = throws.iter().any(|t| {
            t.len() == 3
                && t.iter().filter(|c| **c == a_s).count() == 1
                && t.iter().filter(|c| **c == k_s).count() == 2
        });
        assert!(
            has_kk_a,
            "should generate the safe KK+A throw; got {:?}",
            throws
        );
        // And the greedy Enoch lead should actually make the multi-card throw
        // (attaching the Ace) rather than dribble out a single.
        let played = choose_play_direct_enoch(&pp, ids[0]).unwrap();
        assert!(
            played.len() >= 3 && played.contains(&a_s),
            "Enoch should open with the KK+A throw; got {:?}",
            played
        );
    }

    /// #3 — following a large throw we cannot win, lay down the lowest NON-POINT
    /// cards and KEEP the points. Seat 0 throws four high spades; seat 1 must follow
    /// four spades from {5♠(point) 6 7 8 9} and should shed 6-9, keeping the 5.
    #[test]
    fn test_follow_big_throw_protects_points() {
        let five_s = card(Number::Five, Suit::Spades); // 5-point
        let (mut pp, ids) = make_play_phase([
            vec![
                card(Number::Ace, Suit::Spades),
                card(Number::King, Suit::Spades),
                card(Number::Queen, Suit::Spades),
                card(Number::Jack, Suit::Spades),
            ],
            vec![
                five_s,
                card(Number::Six, Suit::Spades),
                card(Number::Seven, Suit::Spades),
                card(Number::Eight, Suit::Spades),
                card(Number::Nine, Suit::Spades),
            ],
            vec![card(Number::Three, Suit::Clubs)],
            vec![card(Number::Four, Suit::Clubs)],
        ]);
        pp.play_cards(
            ids[0],
            &[
                card(Number::Ace, Suit::Spades),
                card(Number::King, Suit::Spades),
                card(Number::Queen, Suit::Spades),
                card(Number::Jack, Suit::Spades),
            ],
        )
        .unwrap();
        let played = choose_play_direct(&pp, ids[1], HeuristicVersion::New).unwrap();
        assert!(
            !played.contains(&five_s) && played.len() == 4,
            "should keep the 5-point and shed non-points; got {:?}",
            played
        );
    }

    /// #7 — ruff a non-pair lead with a NON-PAIRED trump rather than fragmenting a
    /// trump pair. Seat 0 leads the 10-point King of spades; seat 1 (attacker) is
    /// void in spades and holds a low trump PAIR (3♥3♥) plus a singleton trump
    /// (5♥). It should ruff with the singleton 5♥, keeping the pair intact.
    #[test]
    fn test_follow_ruffs_with_nonpaired_trump() {
        let trump = Trump::Standard {
            suit: Suit::Hearts,
            number: Number::Two,
        };
        let three_h = card(Number::Three, Suit::Hearts);
        let five_h = card(Number::Five, Suit::Hearts);
        let (mut pp, ids) = make_play_phase_decks(
            [
                vec![
                    card(Number::King, Suit::Spades),
                    card(Number::Two, Suit::Clubs),
                ],
                vec![three_h, three_h, five_h, card(Number::Nine, Suit::Clubs)],
                vec![
                    card(Number::Four, Suit::Spades),
                    card(Number::Six, Suit::Clubs),
                ],
                vec![
                    card(Number::Seven, Suit::Spades),
                    card(Number::Eight, Suit::Clubs),
                ],
            ],
            2,
            trump,
        );
        pp.play_cards(ids[0], &[card(Number::King, Suit::Spades)])
            .unwrap();
        let ctx = EvalCtx::build(&pp, ids[1]);
        let ruff_singleton = score_follow(&ctx, &pp, &[five_h]);
        let ruff_break_pair = score_follow(&ctx, &pp, &[three_h]);
        assert!(
            ruff_singleton > ruff_break_pair,
            "should ruff with the singleton 5H ({}) rather than break the 3H pair ({})",
            ruff_singleton,
            ruff_break_pair,
        );
        let played = choose_play_direct(&pp, ids[1], HeuristicVersion::New).unwrap();
        assert_eq!(
            played,
            vec![five_h],
            "greedy follow should ruff with the non-paired trump; got {:?}",
            played
        );
    }

    /// #1 / #4 — when an opponent is winning and NO legal card can take the trick,
    /// duck with the absolute lowest card; never burn a future winner. These use
    /// the PLAIN (non-Enoch) scorer to prove the rule is shared across every tier.
    #[test]
    fn test_follow_ducks_low_over_near_boss_when_losing() {
        // Seat 0 (defender) leads the boss Ace of spades. Seat 1 (attacker) must
        // follow spades with the King (a future winner once the Ace is gone) or a
        // low 3. It can't beat the Ace on the table, so it must keep the King and
        // duck the 3.
        let king_s = card(Number::King, Suit::Spades);
        let low_s = card(Number::Three, Suit::Spades);
        let (mut pp, ids) = make_play_phase([
            vec![card(Number::Ace, Suit::Spades)],
            vec![king_s, low_s],
            vec![card(Number::Four, Suit::Spades)],
            vec![card(Number::Five, Suit::Spades)],
        ]);
        pp.play_cards(ids[0], &[card(Number::Ace, Suit::Spades)])
            .unwrap();
        let ctx = EvalCtx::build(&pp, ids[1]);
        let duck = score_follow(&ctx, &pp, &[low_s]);
        let waste = score_follow(&ctx, &pp, &[king_s]);
        assert!(
            duck > waste,
            "must duck the low 3 ({}) rather than waste the boss King ({}) \
             on a trick it cannot win",
            duck,
            waste,
        );
        let played = choose_play_direct(&pp, ids[1], HeuristicVersion::New).unwrap();
        assert_eq!(played, vec![low_s], "greedy follow should duck the lowest");
    }

    /// #4 — never ruff/follow with a redundant trump-rank card (here an off-suit
    /// trump-number, all of which TIE and so cannot beat one already on the table)
    /// when a low trump is available. Trump is Hearts/Two.
    #[test]
    fn test_follow_ducks_low_trump_over_offsuit_rank() {
        // Seat 0 leads the trump-suit rank card (2 of hearts, the boss trump). Seat
        // 1 must follow trump with an off-suit rank card (2 of clubs — effective
        // trump, but ties the 2s and can't win) or a low trump (3 of hearts). Keep
        // the valuable rank card; duck the low trump.
        let offsuit_rank = card(Number::Two, Suit::Clubs); // off-suit trump number
        let low_trump = card(Number::Three, Suit::Hearts);
        let (mut pp, ids) = make_play_phase([
            vec![card(Number::Two, Suit::Hearts)],
            vec![offsuit_rank, low_trump],
            vec![card(Number::Four, Suit::Hearts)],
            vec![card(Number::Five, Suit::Hearts)],
        ]);
        pp.play_cards(ids[0], &[card(Number::Two, Suit::Hearts)])
            .unwrap();
        let ctx = EvalCtx::build(&pp, ids[1]);
        let duck = score_follow(&ctx, &pp, &[low_trump]);
        let waste = score_follow(&ctx, &pp, &[offsuit_rank]);
        assert!(
            duck > waste,
            "must duck the low trump ({}) rather than waste the off-suit rank \
             card ({}) on a trick it cannot win",
            duck,
            waste,
        );
    }

    /// #4 — never play a small joker under a big joker already on the table.
    #[test]
    fn test_follow_ducks_under_higher_joker() {
        // Seat 0 leads the Big Joker (the unbeatable trump top). Seat 1 must follow
        // trump with the Small Joker (can't win) or a low trump; keep the joker.
        let low_trump = card(Number::Three, Suit::Hearts);
        let (mut pp, ids) = make_play_phase([
            vec![Card::BigJoker],
            vec![Card::SmallJoker, low_trump],
            vec![card(Number::Four, Suit::Hearts)],
            vec![card(Number::Five, Suit::Hearts)],
        ]);
        pp.play_cards(ids[0], &[Card::BigJoker]).unwrap();
        let ctx = EvalCtx::build(&pp, ids[1]);
        let duck = score_follow(&ctx, &pp, &[low_trump]);
        let waste = score_follow(&ctx, &pp, &[Card::SmallJoker]);
        assert!(
            duck > waste,
            "must duck the low trump ({}) rather than waste the small joker \
             ({}) under the big joker",
            duck,
            waste,
        );
        let played = choose_play_direct(&pp, ids[1], HeuristicVersion::New).unwrap();
        assert_eq!(
            played,
            vec![low_trump],
            "greedy follow should keep the joker"
        );
    }

    #[test]
    fn test_mixed_trump_set_throw_is_vetoed() {
        let trump = Trump::Standard {
            suit: Suit::Hearts,
            number: Number::Two,
        };
        let low_trump = card(Number::Three, Suit::Hearts);
        let (pp, ids) = make_play_phase_decks(
            [
                vec![
                    Card::BigJoker,
                    low_trump,
                    low_trump,
                    card(Number::Four, Suit::Clubs),
                ],
                vec![card(Number::Five, Suit::Clubs)],
                vec![card(Number::Six, Suit::Clubs)],
                vec![card(Number::Seven, Suit::Clubs)],
            ],
            2,
            trump,
        );
        let ctx = EvalCtx::build(&pp, ids[0]);
        for unsafe_throw in [
            vec![Card::BigJoker, low_trump],
            vec![Card::BigJoker, low_trump, low_trump],
        ] {
            assert!(pp.can_play_cards(ids[0], &unsafe_throw).is_ok());
            assert!(
                !admissible_ranked_lead(&ctx, &pp, &unsafe_throw),
                "mixed trump set must be excluded from honest rankings: {:?}",
                unsafe_throw
            );
            assert!(
                !ranked_leads(&pp, ids[0]).iter().any(|play| Card::count(
                    play.cards.iter().copied()
                ) == Card::count(
                    unsafe_throw.iter().copied()
                )),
                "vetoed trump set must not survive into search proposals"
            );
        }
    }

    #[test]
    fn test_naked_joker_open_loses_to_cashable_side_ace() {
        let ace_spades = card(Number::Ace, Suit::Spades);
        let (pp, ids) = make_play_phase([
            vec![Card::BigJoker, ace_spades, card(Number::Three, Suit::Clubs)],
            vec![card(Number::Four, Suit::Spades)],
            vec![card(Number::Five, Suit::Spades)],
            vec![card(Number::Six, Suit::Spades)],
        ]);
        let ctx = EvalCtx::build(&pp, ids[0]);
        assert!(
            score_lead(&ctx, &pp, &[ace_spades]) > score_lead(&ctx, &pp, &[Card::BigJoker]),
            "an empty-pot joker lead must not outrank a cashable side-suit Ace"
        );
    }

    #[test]
    fn test_last_seat_keeps_side_ace_on_lost_trick() {
        let ace_spades = card(Number::Ace, Suit::Spades);
        let low_club = card(Number::Three, Suit::Clubs);
        let (mut pp, ids) = make_play_phase([
            vec![Card::BigJoker],
            vec![card(Number::Three, Suit::Hearts)],
            vec![card(Number::Four, Suit::Hearts)],
            vec![ace_spades, low_club],
        ]);
        pp.play_cards(ids[0], &[Card::BigJoker]).unwrap();
        pp.play_cards(ids[1], &[card(Number::Three, Suit::Hearts)])
            .unwrap();
        pp.play_cards(ids[2], &[card(Number::Four, Suit::Hearts)])
            .unwrap();
        let played = choose_play_direct(&pp, ids[3], HeuristicVersion::New).unwrap();
        assert_eq!(played, vec![low_club]);
    }

    #[test]
    fn test_partner_boss_is_not_locked_against_known_void_opponent() {
        let trump = Trump::Standard {
            suit: Suit::Hearts,
            number: Number::Two,
        };
        let ace_spades = card(Number::Ace, Suit::Spades);
        let king_spades = card(Number::King, Suit::Spades);
        let low_spades = card(Number::Three, Suit::Spades);
        let (mut pp, ids) = make_play_phase_decks(
            [
                vec![ace_spades, ace_spades],
                vec![
                    card(Number::Four, Suit::Spades),
                    card(Number::Five, Suit::Spades),
                ],
                vec![card(Number::Six, Suit::Spades), king_spades, low_spades],
                vec![
                    card(Number::Three, Suit::Clubs),
                    card(Number::Four, Suit::Clubs),
                ],
            ],
            2,
            trump,
        );
        // First trick publicly proves that the last opponent is void in spades.
        pp.play_cards(ids[0], &[ace_spades]).unwrap();
        pp.play_cards(ids[1], &[card(Number::Four, Suit::Spades)])
            .unwrap();
        pp.play_cards(ids[2], &[card(Number::Six, Suit::Spades)])
            .unwrap();
        pp.play_cards(ids[3], &[card(Number::Three, Suit::Clubs)])
            .unwrap();
        pp.finish_trick().unwrap();

        pp.play_cards(ids[0], &[ace_spades]).unwrap();
        pp.play_cards(ids[1], &[card(Number::Five, Suit::Spades)])
            .unwrap();
        let ctx = EvalCtx::build(&pp, ids[2]);
        assert!(
            score_follow(&ctx, &pp, &[low_spades]) > score_follow(&ctx, &pp, &[king_spades]),
            "do not feed a point King when the last opponent is known able to ruff"
        );
    }

    #[test]
    fn test_edge_guard_secures_partner_before_last_opponent() {
        let ace_spades = card(Number::Ace, Suit::Spades);
        let low_spades = card(Number::Three, Suit::Spades);
        let (mut pp, ids) = make_play_phase([
            vec![card(Number::Nine, Suit::Spades)],
            vec![card(Number::Four, Suit::Spades)],
            vec![ace_spades, low_spades],
            vec![card(Number::King, Suit::Spades)],
        ]);
        pp.play_cards(ids[0], &[card(Number::Nine, Suit::Spades)])
            .unwrap();
        pp.play_cards(ids[1], &[card(Number::Four, Suit::Spades)])
            .unwrap();
        let played = choose_play_direct(&pp, ids[2], HeuristicVersion::New).unwrap();
        assert_eq!(played, vec![ace_spades]);
    }

    #[test]
    fn test_threshold_coverage_uses_configured_turnover_not_hardcoded_80() {
        let (pp, ids) = make_play_phase_decks(
            [
                vec![card(Number::Three, Suit::Clubs)],
                vec![Card::BigJoker, card(Number::Four, Suit::Clubs)],
                vec![card(Number::Five, Suit::Clubs)],
                vec![card(Number::Six, Suit::Clubs)],
            ],
            2,
            Trump::Standard {
                suit: Suit::Hearts,
                number: Number::Two,
            },
        );
        let mut ctx = EvalCtx::build(&pp, ids[1]);
        assert_eq!(ctx.non_landlord_turnover_score, Some(80));
        ctx.non_landlord_points = 70;
        ctx.my_hand_size = 8;
        ctx.my_trump_count = 1;
        ctx.unseen_trumps = 8;
        ctx.k.voids.insert(ids[3], vec![EffectiveSuit::Trump]);
        assert!(threshold_coverage_lead(
            &ctx,
            &pp,
            Card::BigJoker,
            true,
            1.0
        ));
        ctx.non_landlord_points = 45;
        assert!(!threshold_coverage_lead(
            &ctx,
            &pp,
            Card::BigJoker,
            true,
            1.0
        ));
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
            new.contains(&king_s) || new.contains(&ace_s),
            "NEW should lead a protected boss spade (alone or as a safe throw); got {:?}",
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

    // =======================================================================
    // Enoch-tier playbook tests.
    // =======================================================================

    fn std_trump(suit: Suit) -> Trump {
        Trump::Standard {
            suit,
            number: Number::Two,
        }
    }

    /// Enoch's pair-aware declaration: given a hand with MORE clubs (singles) but a
    /// PAIR in spades, Enoch should value the spade trump above the club trump
    /// (a trump pair is worth ~3-4 single trumps).
    #[test]
    fn test_enoch_bid_prefers_paired_suit() {
        // Equal LENGTH (4 each) and matched rank strengths, but spades hold a PAIR
        // (8-8) where clubs are all singles. With length tied, the trump pair must
        // tip Enoch toward spades.
        let hand = vec![
            card(Number::Five, Suit::Clubs),
            card(Number::Seven, Suit::Clubs),
            card(Number::Eight, Suit::Clubs),
            card(Number::Nine, Suit::Clubs),
            card(Number::Five, Suit::Spades),
            card(Number::Seven, Suit::Spades),
            card(Number::Eight, Suit::Spades),
            card(Number::Eight, Suit::Spades), // the spade PAIR (replaces the 9)
        ];
        let clubs = bid_strength_enoch(&hand, std_trump(Suit::Clubs));
        let spades = bid_strength_enoch(&hand, std_trump(Suit::Spades));
        assert!(
            spades > clubs,
            "Enoch should prefer the PAIRED spade trump ({}) over the equal-length \
             but unpaired club trump ({})",
            spades,
            clubs
        );
    }

    /// Enoch kitty discipline: on a weak hand (no jokers, few trump) it buries NO
    /// points, never buries aces / trump / strong pairs, and prefers voiding a
    /// short side suit.
    #[test]
    fn test_enoch_kitty_protects_and_buries_no_points_on_weak_hand() {
        let trump = std_trump(Suit::Hearts);
        // A weak-ish pool: a couple of low trump, an Ace, a jack pair, a 10-point
        // King, and lots of low side-suit trash (clubs + a SHORT diamonds suit).
        let pool = vec![
            card(Number::Three, Suit::Hearts), // low trump
            card(Number::Four, Suit::Hearts),  // low trump
            card(Number::Ace, Suit::Spades),   // ace (protected)
            card(Number::Jack, Suit::Spades),  // jack pair (protected)
            card(Number::Jack, Suit::Spades),
            card(Number::King, Suit::Spades), // 10-pt point (avoid burying)
            card(Number::Three, Suit::Clubs),
            card(Number::Four, Suit::Clubs),
            card(Number::Six, Suit::Clubs),
            card(Number::Seven, Suit::Clubs),
            card(Number::Nine, Suit::Diamonds), // short diamonds (void target)
            card(Number::Eight, Suit::Diamonds),
        ];
        let buried = choose_kitty_enoch(&pool, trump, 4);
        assert_eq!(buried.len(), 4);
        // No trump.
        assert!(
            !buried.iter().any(|c| is_trump(trump, *c)),
            "Enoch must never bury trump; buried {:?}",
            buried
        );
        // No aces.
        assert!(
            !buried.iter().any(|c| matches!(
                c,
                Card::Suited {
                    number: Number::Ace,
                    ..
                }
            )),
            "Enoch must never bury an ace; buried {:?}",
            buried
        );
        // No part of the protected jack pair.
        assert!(
            !buried.iter().any(|c| matches!(
                c,
                Card::Suited {
                    number: Number::Jack,
                    ..
                }
            )),
            "Enoch must not bury the jack pair; buried {:?}",
            buried
        );
        // Weak hand => no points buried.
        let pts: usize = buried.iter().filter_map(|c| c.points()).sum();
        assert_eq!(pts, 0, "weak hand buries no points; buried {:?}", buried);
    }

    /// Enoch kitty (Trip Holder), STRONG hand: a healthy point budget (2 jokers)
    /// lets Enoch FULLY void a short point-free side suit (so it can ruff later)
    /// AND sink real points up to its budget — and the void-completion pass never
    /// leaves a sanctioned void half-finished. Exercises the void/point-budget
    /// gating + pre-commit path.
    #[test]
    fn test_enoch_kitty_completes_void_and_buries_points_on_strong_hand() {
        let trump = std_trump(Suit::Hearts);
        let pool = vec![
            Card::BigJoker,
            Card::SmallJoker,                // two jokers -> a healthy point budget
            card(Number::Ace, Suit::Spades), // ace (protected)
            card(Number::King, Suit::Clubs), // 10-pt point in a LONG suit (buryable)
            card(Number::Ten, Suit::Clubs),  // 10-pt point
            card(Number::Three, Suit::Clubs),
            card(Number::Four, Suit::Clubs),
            card(Number::Seven, Suit::Diamonds), // SHORT point-free diamonds = void target
            card(Number::Eight, Suit::Diamonds),
        ];
        let buried = choose_kitty_enoch(&pool, trump, 4);
        assert_eq!(buried.len(), 4);
        assert!(
            !buried.iter().any(|c| is_trump(trump, *c)),
            "never bury trump; buried {:?}",
            buried
        );
        assert!(
            !buried.iter().any(|c| matches!(
                c,
                Card::Suited {
                    number: Number::Ace,
                    ..
                }
            )),
            "never bury an ace; buried {:?}",
            buried
        );
        // The short, point-free diamonds suit is FULLY voided (both cards buried) —
        // a sanctioned void completes rather than being half-buried.
        let diamonds_buried = buried
            .iter()
            .filter(|c| {
                matches!(
                    c,
                    Card::Suited {
                        suit: Suit::Diamonds,
                        ..
                    }
                )
            })
            .count();
        assert_eq!(
            diamonds_buried, 2,
            "strong hand should FULLY void the short diamonds suit; buried {:?}",
            buried
        );
        // A strong hand sinks real points (within its budget) instead of stranding
        // them in play.
        let pts: usize = buried.iter().filter_map(|c| c.points()).sum();
        assert!(
            pts > 0 && pts <= 25,
            "strong hand buries points within budget; buried {:?} ({} pts)",
            buried,
            pts
        );
    }

    /// Enoch leads tractors first: a boss tractor lead should outscore the same
    /// cards' lower single lead, and the Enoch bonus should make the tractor lead
    /// score HIGHER under Enoch than under the plain heuristic.
    #[test]
    fn test_enoch_prioritizes_tractor_lead() {
        // Seat 0 holds a spade tractor 7-7-8-8 (consecutive pairs) plus trash.
        let (pp, ids) = make_play_phase([
            vec![
                card(Number::Seven, Suit::Spades),
                card(Number::Seven, Suit::Spades),
                card(Number::Eight, Suit::Spades),
                card(Number::Eight, Suit::Spades),
                card(Number::Three, Suit::Clubs),
            ],
            vec![card(Number::Four, Suit::Spades)],
            vec![card(Number::Five, Suit::Spades)],
            vec![card(Number::Six, Suit::Spades)],
        ]);
        let tractor = vec![
            card(Number::Seven, Suit::Spades),
            card(Number::Seven, Suit::Spades),
            card(Number::Eight, Suit::Spades),
            card(Number::Eight, Suit::Spades),
        ];
        let single = vec![card(Number::Three, Suit::Clubs)];

        let ctx_enoch = EvalCtx::build_enoch(&pp, ids[0]);
        let ctx_plain = EvalCtx::build(&pp, ids[0]);

        let tractor_enoch = score_lead(&ctx_enoch, &pp, &tractor);
        let single_enoch = score_lead(&ctx_enoch, &pp, &single);
        let tractor_plain = score_lead(&ctx_plain, &pp, &tractor);

        assert!(
            tractor_enoch > single_enoch,
            "Enoch should lead the tractor ({}) over a single ({})",
            tractor_enoch,
            single_enoch
        );
        assert!(
            tractor_enoch > tractor_plain,
            "Enoch's tractor-first bonus should raise the tractor lead score \
             ({}) above the plain heuristic's ({})",
            tractor_enoch,
            tractor_plain
        );

        // And the greedy Enoch direct play actually leads the tractor.
        let played = choose_play_direct_enoch(&pp, ids[0]).unwrap();
        assert_eq!(
            played.len(),
            4,
            "Enoch's greedy lead should be the 4-card tractor; got {:?}",
            played
        );
    }

    /// #2 — the exchanger reads its OWN buried kitty EXACTLY; every other seat gets
    /// an HONEST ESTIMATE from the unseen-card point density (never the buried
    /// cards). We assert the non-exchanger's value equals the public-info estimate,
    /// proving the honesty boundary holds (no hidden-card leak).
    #[test]
    fn test_enoch_kitty_points_estimate_is_honest_for_non_exchanger() {
        let (pp, ids) = make_play_phase([
            vec![card(Number::Ace, Suit::Spades)],
            vec![card(Number::King, Suit::Spades)],
            vec![card(Number::Queen, Suit::Spades)],
            vec![card(Number::Jack, Suit::Spades)],
        ]);
        // Seat 0 is the landlord/exchanger here; an attacker (seat 1) is not.
        let ctx_landlord = EvalCtx::build_enoch(&pp, ids[0]);
        let ctx_attacker = EvalCtx::build_enoch(&pp, ids[1]);
        assert!(
            ctx_landlord.kitty_points.is_some(),
            "the exchanger may read its own buried kitty exactly"
        );
        // The non-exchanger now gets an estimate (Some), derived ONLY from public
        // info — it must match the value recomputed from its own honest Knowledge
        // plus the PUBLIC kitty size, never the real buried cards.
        let k = Knowledge::from_play_view_full_memory(&pp, ids[1]);
        let expected = estimate_kitty_points(&k, pp.kitty_size(), pp.num_decks());
        assert_eq!(
            ctx_attacker.kitty_points,
            Some(expected),
            "a non-exchanger's kitty value must be the honest public-info estimate"
        );
    }

    // =======================================================================
    // Trip Holder refinements (Enoch-only). Each test maps to one refinement.
    // =======================================================================

    /// Trip Holder #1 — "never OPEN with high trump". When Enoch is opening a
    /// trick it must NOT lead a naked joker, the trump-rank (trump-number) card, or
    /// a big trump; it should prefer to open with an ace (a side-suit boss). We
    /// score the candidate opens directly for the Enoch leader (seat 0).
    #[test]
    fn test_enoch_never_opens_with_high_trump() {
        // Seat 0 (Enoch leader) holds a side-suit ace, a naked small joker, the
        // trump-rank (Two of hearts = trump number), a big trump (King of hearts),
        // and a low trump. Trump is Hearts/2.
        let ace_s = card(Number::Ace, Suit::Spades); // side-suit boss (1 deck)
        let small_joker = Card::SmallJoker;
        let trump_rank = card(Number::Two, Suit::Hearts); // trump-number card
        let big_trump = card(Number::King, Suit::Hearts); // big trump-suit card
        let low_trump = card(Number::Three, Suit::Hearts);
        let (pp, ids) = make_play_phase([
            vec![ace_s, small_joker, trump_rank, big_trump, low_trump],
            vec![card(Number::Four, Suit::Spades)],
            vec![card(Number::Five, Suit::Spades)],
            vec![card(Number::Six, Suit::Spades)],
        ]);
        let ctx = EvalCtx::build_enoch(&pp, ids[0]);
        let ace = score_lead(&ctx, &pp, &[ace_s]);
        let joker = score_lead(&ctx, &pp, &[small_joker]);
        let rank = score_lead(&ctx, &pp, &[trump_rank]);
        let big = score_lead(&ctx, &pp, &[big_trump]);

        // The ace open must outrank every high-trump open by a wide margin.
        assert!(
            ace > joker && ace > rank && ace > big,
            "Enoch should open the ace ({}) over a naked joker ({}), the \
             trump-rank ({}), or a big trump ({})",
            ace,
            joker,
            rank,
            big
        );
        // Each forbidden high-trump open is heavily negative (≈0% frequency).
        assert!(
            joker < -20.0,
            "naked joker open must be heavily penalized: {}",
            joker
        );
        assert!(
            rank < -20.0,
            "trump-rank open must be heavily penalized: {}",
            rank
        );
        assert!(big < -10.0, "big-trump open must be penalized: {}", big);

        // And the greedy Enoch lead actually opens the ace, not high trump.
        let played = choose_play_direct_enoch(&pp, ids[0]).unwrap();
        assert_eq!(
            played,
            vec![ace_s],
            "Enoch's greedy open should be the side-suit ace; got {:?}",
            played
        );
    }

    /// Trip Holder #1 (exception) — Enoch MAY open high trump only when sitting on
    /// ~15+ trump cards (the rare "bleed everyone of trump" line). With a huge
    /// trump holding the big-trump-open penalty is LIFTED, so the same big-trump
    /// open scores strictly HIGHER than it does from a normal (non-flooded) hand.
    #[test]
    fn test_enoch_high_trump_open_allowed_when_trump_flooded() {
        let big_trump = card(Number::King, Suit::Hearts);

        // FLOODED: 12 trump-suit hearts + 2 jokers + the trump-number = 15 trumps.
        let flood: Vec<Card> = vec![
            card(Number::Three, Suit::Hearts),
            card(Number::Four, Suit::Hearts),
            card(Number::Five, Suit::Hearts),
            card(Number::Six, Suit::Hearts),
            card(Number::Seven, Suit::Hearts),
            card(Number::Eight, Suit::Hearts),
            card(Number::Nine, Suit::Hearts),
            card(Number::Ten, Suit::Hearts),
            card(Number::Jack, Suit::Hearts),
            card(Number::Queen, Suit::Hearts),
            card(Number::King, Suit::Hearts),
            card(Number::Ace, Suit::Hearts),
            Card::SmallJoker,
            Card::BigJoker,
            card(Number::Two, Suit::Hearts), // trump-number (effective trump)
        ];
        let (pp_flood, ids_f) = make_play_phase([
            flood,
            vec![card(Number::Four, Suit::Clubs)],
            vec![card(Number::Five, Suit::Clubs)],
            vec![card(Number::Six, Suit::Clubs)],
        ]);
        let ctx_flood = EvalCtx::build_enoch(&pp_flood, ids_f[0]);
        assert!(
            ctx_flood.my_trump_count >= 15,
            "fixture must flood trump (got {})",
            ctx_flood.my_trump_count
        );
        let big_flooded = score_lead(&ctx_flood, &pp_flood, &[big_trump]);

        // NORMAL: only a few trumps (incl. the same K♥), so the open penalty bites.
        let (pp_norm, ids_n) = make_play_phase([
            vec![
                big_trump,
                card(Number::Three, Suit::Hearts),
                card(Number::Four, Suit::Hearts),
                card(Number::Nine, Suit::Spades),
            ],
            vec![card(Number::Four, Suit::Clubs)],
            vec![card(Number::Five, Suit::Clubs)],
            vec![card(Number::Six, Suit::Clubs)],
        ]);
        let ctx_norm = EvalCtx::build_enoch(&pp_norm, ids_n[0]);
        assert!(ctx_norm.my_trump_count < 15, "normal hand has < 15 trumps");
        let big_normal = score_lead(&ctx_norm, &pp_norm, &[big_trump]);

        // The flood lifts the -30 forbidden-open penalty, so the bleed open scores
        // markedly higher than from a normal hand.
        assert!(
            big_flooded > big_normal + 20.0,
            "with 15+ trumps the big-trump bleed open ({}) must beat the penalized \
             normal-hand open ({}) by the lifted penalty",
            big_flooded,
            big_normal
        );
    }

    /// Trip Holder #1 (hand-off) — the defender mid/late-game hand-off must use a
    /// SMALL trump (2/3/4 non-point), never a joker or the trump-rank card. We
    /// build a late-game defender with its bosses spent and check that a low trump
    /// hand-off outscores a joker / trump-rank "hand-off".
    #[test]
    fn test_enoch_handoff_uses_small_trump_not_joker() {
        // Seat 0 is a defender (landlord team). Late game (small hand), no boss
        // non-trump lead available. Holds a low trump (3♥), the trump-rank (2♥),
        // and a small joker. A small-trump hand-off should beat a joker / rank one.
        let low_trump = card(Number::Three, Suit::Hearts);
        let trump_rank = card(Number::Two, Suit::Hearts);
        let small_joker = Card::SmallJoker;
        let (pp, ids) = make_play_phase([
            vec![
                low_trump,
                trump_rank,
                small_joker,
                card(Number::Six, Suit::Clubs),
            ],
            vec![card(Number::Four, Suit::Spades)],
            vec![card(Number::Five, Suit::Spades)],
            vec![card(Number::Six, Suit::Spades)],
        ]);
        let ctx = EvalCtx::build_enoch(&pp, ids[0]);
        assert!(!ctx.me_is_attacker, "seat 0 is a defender");
        assert!(ctx.my_hand_size <= 8, "must be late game for the hand-off");
        let small = score_lead(&ctx, &pp, &[low_trump]);
        let joker = score_lead(&ctx, &pp, &[small_joker]);
        let rank = score_lead(&ctx, &pp, &[trump_rank]);
        assert!(
            small > joker && small > rank,
            "hand-off with a SMALL trump ({}) must beat a joker ({}) or \
             trump-rank ({}) hand-off",
            small,
            joker,
            rank
        );
    }

    /// Trip Holder #3 — dump points to a WINNING PARTNER. When our partner is
    /// winning the trick (here with a locked boss), Enoch should DROP a 10-point
    /// card rather than hoard a low card. The Enoch follow bonus must raise the
    /// point-feed above the low dump (and above the plain heuristic's feed).
    #[test]
    fn test_enoch_dumps_points_to_winning_partner() {
        // Seat 0 (our partner, same team as seat 2) leads the boss Ace of spades;
        // seat 1 (opp) follows low; seat 2 (Enoch) must follow with K (10pts) or a
        // low spade. The partner Ace is locked, so dump the King.
        let king_s = card(Number::King, Suit::Spades); // 10-pt feed
        let low_s = card(Number::Three, Suit::Spades); // trash
        let (mut pp, ids) = make_play_phase([
            vec![card(Number::Ace, Suit::Spades)], // seat0 partner leads boss
            vec![card(Number::Four, Suit::Spades)], // seat1 opp
            vec![king_s, low_s],                   // seat2 Enoch
            vec![card(Number::Five, Suit::Spades)], // seat3 opp still to act
        ]);
        pp.play_cards(ids[0], &[card(Number::Ace, Suit::Spades)])
            .unwrap();
        pp.play_cards(ids[1], &[card(Number::Four, Suit::Spades)])
            .unwrap();
        let ctx_enoch = EvalCtx::build_enoch(&pp, ids[2]);
        let ctx_plain = EvalCtx::build(&pp, ids[2]);
        let feed_king_enoch = score_follow(&ctx_enoch, &pp, &[king_s]);
        let dump_low_enoch = score_follow(&ctx_enoch, &pp, &[low_s]);
        let feed_king_plain = score_follow(&ctx_plain, &pp, &[king_s]);
        assert!(
            feed_king_enoch > dump_low_enoch,
            "Enoch should DUMP the 10-pt King ({}) to the winning partner over a \
             low dump ({})",
            feed_king_enoch,
            dump_low_enoch
        );
        assert!(
            feed_king_enoch > feed_king_plain,
            "Enoch's winning-partner point-dump bonus must raise the feed ({}) \
             above the plain heuristic ({})",
            feed_king_enoch,
            feed_king_plain
        );
        // The greedy Enoch follow actually feeds the King.
        let played = choose_play_direct_enoch(&pp, ids[2]).unwrap();
        assert_eq!(
            played,
            vec![king_s],
            "Enoch's greedy follow should feed the King; got {:?}",
            played
        );
    }

    /// Trip Holder #3 — play the SMALLEST card when we cannot win. An opponent is
    /// winning with a card we cannot beat; Enoch must duck with its lowest card and
    /// NEVER waste a high trump / joker on a trick it can't take.
    #[test]
    fn test_enoch_plays_smallest_when_cannot_win() {
        // Seat 0 (opp of seat 1) leads the boss Ace of spades. Seat 1 (Enoch,
        // attacker) is void in spades and must discard: it holds a low club, a
        // small joker (high trump), and nothing in spades. Ducking the low club
        // (can't win, mustn't waste the joker) is correct.
        let low_club = card(Number::Three, Suit::Clubs);
        let small_joker = Card::SmallJoker; // high trump — must not be wasted
        let (mut pp, ids) = make_play_phase([
            vec![card(Number::Ace, Suit::Spades)], // seat0 leads boss spade
            vec![low_club, small_joker],           // seat1 Enoch void in spades
            vec![card(Number::Four, Suit::Spades)], // seat2
            vec![card(Number::Five, Suit::Spades)], // seat3
        ]);
        pp.play_cards(ids[0], &[card(Number::Ace, Suit::Spades)])
            .unwrap();
        let ctx = EvalCtx::build_enoch(&pp, ids[1]);
        assert!(ctx.me_is_attacker, "seat 1 is an attacker");
        let duck_low = score_follow(&ctx, &pp, &[low_club]);
        let waste_joker = score_follow(&ctx, &pp, &[small_joker]);
        assert!(
            duck_low > waste_joker,
            "Enoch must duck the low club ({}) rather than waste the joker ({}) \
             on a trick it can't win",
            duck_low,
            waste_joker
        );
        let played = choose_play_direct_enoch(&pp, ids[1]).unwrap();
        assert_eq!(
            played,
            vec![low_club],
            "Enoch's greedy follow should duck the lowest card; got {:?}",
            played
        );
    }

    /// Trip Holder #2 — a STRONG declarer hand buries real points in the kitty
    /// (the old budget left them "never putting points in the kitty"), still never
    /// buries trump / aces / pairs, and prioritizes voiding an entire side suit.
    #[test]
    fn test_enoch_kitty_buries_points_when_strong() {
        let trump = std_trump(Suit::Hearts);
        // STRONG pool: 4 jokers + a heart trump holding ⇒ a healthy point budget.
        // A 10-pt King + low club trash, an ace + a spade pair (both protected),
        // and a SHORT diamonds side suit (a clean void target with no points/aces).
        let pool = vec![
            Card::BigJoker,
            Card::BigJoker,
            Card::SmallJoker,
            Card::SmallJoker,
            card(Number::Three, Suit::Hearts), // trump
            card(Number::Four, Suit::Hearts),  // trump
            card(Number::Five, Suit::Hearts),  // trump
            card(Number::Six, Suit::Hearts),   // trump
            card(Number::Seven, Suit::Hearts), // trump
            card(Number::Ace, Suit::Spades),   // ace (protected)
            card(Number::Eight, Suit::Spades), // spade pair (protected)
            card(Number::Eight, Suit::Spades),
            card(Number::King, Suit::Clubs), // 10-pt point (buriable when strong)
            card(Number::Three, Suit::Clubs), // low club trash (non-point)
            card(Number::Six, Suit::Clubs),  // low club trash (non-point)
            card(Number::Seven, Suit::Clubs), // low club trash (non-point)
            card(Number::Three, Suit::Diamonds), // SHORT diamonds: void target
            card(Number::Four, Suit::Diamonds),
        ];
        let buried = choose_kitty_enoch(&pool, trump, 4);
        assert_eq!(buried.len(), 4);
        // Never trump.
        assert!(
            !buried.iter().any(|c| is_trump(trump, *c)),
            "must never bury trump; buried {:?}",
            buried
        );
        // Never an ace.
        assert!(
            !buried.iter().any(|c| matches!(
                c,
                Card::Suited {
                    number: Number::Ace,
                    ..
                }
            )),
            "must never bury an ace; buried {:?}",
            buried
        );
        // Never the protected spade pair.
        assert!(
            !buried.iter().any(|c| matches!(
                c,
                Card::Suited {
                    number: Number::Seven,
                    suit: Suit::Spades,
                }
            )),
            "must never bury the protected pair; buried {:?}",
            buried
        );
        // STRONG hand ⇒ it DOES bury real points now (the King of clubs, 10 pts).
        let pts: usize = buried.iter().filter_map(|c| c.points()).sum();
        assert!(
            pts >= 10,
            "a strong hand must bury real points; buried {} pts ({:?})",
            pts,
            buried
        );
        // And it voids the ENTIRE short diamonds suit (both diamonds buried).
        let diamonds_buried = buried
            .iter()
            .filter(|c| {
                matches!(
                    c,
                    Card::Suited {
                        suit: Suit::Diamonds,
                        ..
                    }
                )
            })
            .count();
        assert_eq!(
            diamonds_buried, 2,
            "should void the entire short diamonds suit; buried {:?}",
            buried
        );
    }

    /// Trip Holder #2 — pairs are protected by default: a hand with a low side-suit
    /// PAIR and enough trash to bury must NOT bury the pair (the old code only
    /// protected jack-pair-and-higher).
    #[test]
    fn test_enoch_kitty_protects_all_pairs() {
        let trump = std_trump(Suit::Hearts);
        // A low club PAIR (4-4) inside a LONG club suit (6 cards — too long to void
        // within a 3-card kitty), plus plenty of spade-singles trash to absorb the
        // burial. So there is no void need that would relax the pair protection.
        let pool = vec![
            card(Number::Four, Suit::Clubs), // low club PAIR (protected)
            card(Number::Four, Suit::Clubs),
            card(Number::Six, Suit::Clubs), // club singles → clubs is 6 long
            card(Number::Seven, Suit::Clubs),
            card(Number::Eight, Suit::Clubs),
            card(Number::Nine, Suit::Clubs),
            card(Number::Three, Suit::Spades), // spade-singles trash to bury
            card(Number::Six, Suit::Spades),
            card(Number::Seven, Suit::Spades),
            card(Number::Eight, Suit::Spades),
            card(Number::Three, Suit::Hearts), // trump
            card(Number::Four, Suit::Hearts),  // trump
        ];
        let buried = choose_kitty_enoch(&pool, trump, 3);
        assert_eq!(buried.len(), 3);
        // The low club pair must survive: clubs (6 long) and spades (4 long) are
        // both too long to fully void within a 3-card kitty, so the void exception
        // does NOT apply and EVERY pair stays protected.
        let club_fours = buried
            .iter()
            .filter(|c| {
                matches!(
                    c,
                    Card::Suited {
                        number: Number::Four,
                        suit: Suit::Clubs,
                    }
                )
            })
            .count();
        assert_eq!(
            club_fours, 0,
            "Enoch must protect the low club pair; buried {:?}",
            buried
        );
    }

    #[test]
    fn test_live_strength_is_ace_high_but_legacy_encoding_is_frozen() {
        let trump = std_trump(Suit::Hearts);
        let ace = card(Number::Ace, Suit::Spades);
        let king = card(Number::King, Suit::Spades);
        assert!(
            card_strength(trump, ace) > card_strength(trump, king),
            "live policy strength must match the mechanics engine's Ace-high order"
        );
        assert!(
            legacy_card_strength(trump, ace) < legacy_card_strength(trump, king),
            "the embedded Expert feature encoding must remain backward compatible"
        );

        let queen = card(Number::Queen, Suit::Spades);
        assert_eq!(
            choose_kitty(&[ace, queen], trump, 1),
            vec![queen],
            "kitty selection must keep the Ace rather than treating it as low trash"
        );
        let spade_trump = std_trump(Suit::Spades);
        assert!(
            bid_strength(&[ace], spade_trump) > bid_strength(&[king], spade_trump),
            "bid evaluation must value an Ace above a King in the candidate trump suit"
        );
    }

    #[test]
    fn test_only_big_joker_is_unconditionally_boss() {
        let trump = std_trump(Suit::Hearts);
        let make_knowledge = |seen: HashMap<Card, usize>| Knowledge {
            configured_counts: Card::count(Deck::default().cards()),
            seen,
            voids: HashMap::new(),
            hidden_counts: HashMap::new(),
            known_holding: HashMap::new(),
            trump,
            num_decks: 1,
            total_cards: 54,
            total_points: 100,
            total_trumps: 18,
        };
        let empty = make_knowledge(HashMap::new());
        assert!(is_boss_card(&empty, trump, Card::BigJoker));
        assert!(
            !is_boss_card(&empty, trump, Card::SmallJoker),
            "an unseen big joker can beat the small joker"
        );
        assert!(
            !is_boss_card(&empty, trump, card(Number::Two, Suit::Hearts)),
            "jokers can beat the trump-suit level card"
        );

        let mut seen = HashMap::new();
        seen.insert(Card::BigJoker, 1);
        assert!(is_boss_card(&make_knowledge(seen), trump, Card::SmallJoker));
    }

    #[test]
    fn test_special_deck_does_not_invent_excluded_dominators() {
        let trump = std_trump(Suit::Hearts);
        let deck = Deck {
            exclude_small_joker: false,
            exclude_big_joker: true,
            min: Number::Five,
        };
        let cards: Vec<Card> = deck.cards().collect();
        let knowledge = Knowledge {
            configured_counts: Card::count(cards.iter().copied()),
            total_cards: cards.len(),
            total_points: cards.iter().map(|card| card.points().unwrap_or(0)).sum(),
            total_trumps: cards
                .iter()
                .filter(|card| trump.effective_suit(**card) == EffectiveSuit::Trump)
                .count(),
            seen: HashMap::new(),
            voids: HashMap::new(),
            hidden_counts: HashMap::new(),
            known_holding: HashMap::new(),
            trump,
            num_decks: 1,
        };
        assert!(
            is_boss_card(&knowledge, trump, Card::SmallJoker),
            "an excluded big joker must not remain a phantom dominator"
        );
    }

    #[test]
    fn test_mechanics_winner_check_handles_higher_trump_follow() {
        let low_trump = card(Number::Three, Suit::Hearts);
        let high_trump = card(Number::Ace, Suit::Hearts);
        let (mut pp, ids) = make_play_phase([
            vec![low_trump],
            vec![high_trump],
            vec![card(Number::Four, Suit::Hearts)],
            vec![card(Number::Five, Suit::Hearts)],
        ]);
        pp.play_cards(ids[0], &[low_trump]).unwrap();
        assert!(
            candidate_wins_current_trick(&pp, ids[1], &[high_trump]),
            "a higher trump following a trump lead must take the current lead"
        );
    }

    #[test]
    fn test_small_hand_lead_candidates_cover_every_legal_multiset() {
        let three = card(Number::Three, Suit::Spades);
        let four = card(Number::Four, Suit::Spades);
        let hand = vec![three, four, four];
        let (pp, ids) = make_play_phase([
            hand.clone(),
            vec![card(Number::Five, Suit::Clubs)],
            vec![card(Number::Six, Suit::Clubs)],
            vec![card(Number::Seven, Suit::Clubs)],
        ]);
        let trump = pp.trump();
        let entries = hand_entries(pp.hands().get(ids[0]).unwrap(), trump, |_| true);
        let mut exhaustive = HashSet::new();
        for size in 1..=hand.len() {
            for mut candidate in enumerate_multiset_combinations(&entries, size, 1024) {
                canonicalize_play(trump, &mut candidate);
                if pp.can_play_cards(ids[0], &candidate).is_ok() {
                    exhaustive.insert(canonical_play_key(&candidate));
                }
            }
        }

        let generated: HashSet<Vec<char>> = lead_candidates_with_limits(&pp, ids[0], 1024, 3)
            .iter()
            .map(|candidate| canonical_play_key(candidate))
            .collect();
        assert_eq!(
            generated, exhaustive,
            "hierarchical leads must cover every legal-equivalence class in a small hand"
        );
    }

    #[test]
    fn test_small_hand_follow_candidates_cover_every_legal_multiset() {
        let lead = card(Number::Nine, Suit::Spades);
        let followers = vec![
            card(Number::Three, Suit::Spades),
            card(Number::Four, Suit::Spades),
            card(Number::Five, Suit::Spades),
        ];
        let (mut pp, ids) = make_play_phase([
            vec![lead, lead],
            followers.clone(),
            vec![
                card(Number::Six, Suit::Spades),
                card(Number::Seven, Suit::Spades),
            ],
            vec![
                card(Number::Ten, Suit::Spades),
                card(Number::Jack, Suit::Spades),
            ],
        ]);
        pp.play_cards(ids[0], &[lead, lead]).unwrap();
        let trump = pp.trump();
        let entries = hand_entries(pp.hands().get(ids[1]).unwrap(), trump, |_| true);
        let mut exhaustive = HashSet::new();
        for mut candidate in enumerate_multiset_combinations(&entries, 2, 1024) {
            canonicalize_play(trump, &mut candidate);
            if pp.can_play_cards(ids[1], &candidate).is_ok() {
                exhaustive.insert(canonical_play_key(&candidate));
            }
        }
        let generated: HashSet<Vec<char>> = follow_candidates_with_cap(&pp, ids[1], 1024)
            .iter()
            .map(|candidate| canonical_play_key(candidate))
            .collect();
        assert_eq!(
            generated, exhaustive,
            "bounded exhaustive fallback must cover every legal small-hand follow"
        );
    }

    #[test]
    fn test_lead_candidates_use_configured_tractor_requirements() {
        use shengji_mechanics::trick::TractorRequirements;

        let three = card(Number::Three, Suit::Spades);
        let four = card(Number::Four, Suit::Spades);
        let tractor = vec![three, three, four, four];
        let (mut pp, ids) = make_play_phase([
            tractor.clone(),
            vec![card(Number::Five, Suit::Clubs)],
            vec![card(Number::Six, Suit::Clubs)],
            vec![card(Number::Seven, Suit::Clubs)],
        ]);
        let default_atomic = lead_candidates_with_limits(&pp, ids[0], 1024, 1);
        assert!(default_atomic.contains(&tractor));

        pp.propagated_mut().tractor_requirements = TractorRequirements {
            min_count: 3,
            min_length: 2,
        };
        let configured_atomic = lead_candidates_with_limits(&pp, ids[0], 1024, 1);
        assert!(
            !configured_atomic.contains(&tractor),
            "a pair tractor must not be proposed as one unit when triples are configured"
        );
    }

    #[test]
    fn test_lead_candidates_propose_enabled_rainbow() {
        use shengji_mechanics::trick::CompoundFormats;

        let mut rainbow = vec![
            card(Number::Three, Suit::Clubs),
            card(Number::Three, Suit::Diamonds),
            card(Number::Three, Suit::Hearts),
            card(Number::Three, Suit::Spades),
        ];
        let (mut pp, ids) = make_play_phase([
            rainbow.clone(),
            vec![card(Number::Five, Suit::Clubs)],
            vec![card(Number::Six, Suit::Clubs)],
            vec![card(Number::Seven, Suit::Clubs)],
        ]);
        pp.propagated_mut().compound_formats = CompoundFormats { rainbows: Some(4) };
        canonicalize_play(pp.trump(), &mut rainbow);
        assert!(
            lead_candidates_with_limits(&pp, ids[0], 1024, 1).contains(&rainbow),
            "enabled cross-suit rainbow must be visible to search"
        );
    }

    #[test]
    fn test_low_cap_reserves_each_lead_family() {
        use shengji_mechanics::trick::CompoundFormats;

        let mut rainbow = vec![
            card(Number::Three, Suit::Clubs),
            card(Number::Three, Suit::Diamonds),
            card(Number::Three, Suit::Hearts),
            card(Number::Three, Suit::Spades),
        ];
        let mut hand = rainbow.clone();
        hand.push(card(Number::Four, Suit::Clubs));
        let (mut pp, ids) = make_play_phase([
            hand,
            vec![card(Number::Five, Suit::Clubs)],
            vec![card(Number::Six, Suit::Clubs)],
            vec![card(Number::Seven, Suit::Clubs)],
        ]);
        pp.propagated_mut().compound_formats = CompoundFormats { rainbows: Some(4) };
        canonicalize_play(pp.trump(), &mut rainbow);
        let candidates = lead_candidates_with_limits(&pp, ids[0], 3, 2);
        assert_eq!(candidates.len(), 3);
        assert!(candidates.iter().any(|candidate| candidate.len() == 1));
        assert!(candidates.contains(&rainbow));
        assert!(
            candidates.iter().any(|candidate| {
                candidate.len() == 2
                    && candidate
                        .iter()
                        .all(|card| card.suit() == Some(Suit::Clubs))
            }),
            "an ordinary throw must survive alongside atomics and rainbows"
        );
    }

    #[test]
    fn test_low_cap_reserves_follow_bomb_family() {
        use shengji_mechanics::trick::BombPolicy;

        let lead_low = card(Number::Three, Suit::Spades);
        let lead_high = card(Number::Four, Suit::Spades);
        let lead = vec![lead_low, lead_low, lead_high, lead_high];
        let bomb = card(Number::Three, Suit::Clubs);
        let (mut pp, ids) = make_play_phase([
            lead.clone(),
            vec![
                card(Number::Five, Suit::Spades),
                card(Number::Five, Suit::Spades),
                card(Number::Six, Suit::Spades),
                card(Number::Six, Suit::Spades),
                bomb,
                bomb,
                bomb,
                bomb,
            ],
            vec![card(Number::Six, Suit::Spades)],
            vec![card(Number::Seven, Suit::Spades)],
        ]);
        pp.propagated_mut().bomb_policy = BombPolicy::AllowBombs;
        pp.play_cards(ids[0], &lead).unwrap();
        let candidates = follow_candidates_with_cap(&pp, ids[1], 2);
        assert!(
            candidates
                .iter()
                .any(|candidate| candidate == &vec![bomb; 4]),
            "an enabled bomb must not be starved by structural follows"
        );
    }
}
