//! Card / void tracking and a determinizer for imperfect-information search.
//!
//! # Honesty invariant
//!
//! Everything in this module is derived only from the acting player's redacted
//! [`PlayPhase`]. Opponents' cards and a hidden kitty are represented there by
//! [`Card::Unknown`]; this module replaces those placeholders with one sampled
//! world. Sampling accounts for the exact configured deck multiset, every
//! publicly played card, public removed cards, pile capacities, and proven suit
//! voids. It never consults the real hidden hands or kitty.

use std::collections::HashMap;

use rand::seq::{index, SliceRandom};
use rand::Rng;

use shengji_mechanics::hands::Hands;
use shengji_mechanics::trick::PlayedCards;
use shengji_mechanics::types::{Card, EffectiveSuit, PlayerID, Trump};

use crate::game_state::play_phase::{follow_proves_void, PlayPhase};

/// A determinized "world": a complete assignment of every hidden card location
/// consistent with the acting player's information.
pub struct DeterminizedWorld {
    pub play: PlayPhase,
}

/// Information available from a player's redacted view.
pub struct Knowledge {
    /// Every visible physical card. These copies are removed from the configured
    /// deck before hidden locations are filled.
    pub seen: HashMap<Card, usize>,
    /// Hard, publicly established per-seat effective-suit voids.
    pub voids: HashMap<PlayerID, Vec<EffectiveSuit>>,
    /// Number of unknown card placeholders in each opponent hand.
    pub hidden_counts: HashMap<PlayerID, usize>,
    /// Public lower bounds on cards still held by a seat. A failed throw reveals
    /// every rejected card even though those cards return to the thrower's hand.
    pub known_holding: HashMap<PlayerID, HashMap<Card, usize>>,
    pub trump: Trump,
    /// Retained for public-card boss logic. Exact deck composition is used by
    /// the sampler itself; this count remains the compatibility summary used by
    /// existing heuristic features.
    pub num_decks: usize,
    /// Exact configured physical-copy counts, including short/joker-less and
    /// heterogeneous special decks.
    pub configured_counts: HashMap<Card, usize>,
    pub total_cards: usize,
    pub total_points: usize,
    pub total_trumps: usize,
}

fn observe_revealed_holding(
    holdings: &mut HashMap<PlayerID, HashMap<Card, usize>>,
    played: &PlayedCards,
) {
    let holding = holdings.entry(played.id).or_default();
    // Cards actually played leave the hand, including copies revealed by an
    // earlier failed throw.
    for (card, count) in Card::count(played.cards.iter().copied()) {
        if let Some(known) = holding.get_mut(&card) {
            *known = known.saturating_sub(count);
        }
    }
    holding.retain(|_, count| *count > 0);
    // The rejected portion remains held. Repeated revelations are lower bounds,
    // so take the maximum rather than double-counting copies.
    for (card, count) in Card::count(played.bad_throw_cards.iter().copied()) {
        let known = holding.entry(card).or_default();
        *known = (*known).max(count);
    }
}

impl Knowledge {
    /// Derive all honest information from a redacted play view.
    ///
    /// Full public memory is unconditional. It is a property of the observation,
    /// not of the root ranking policy or bot tier: every honest player saw every
    /// completed play and every off-suit follow.
    pub fn from_play_view(p: &PlayPhase, me: PlayerID) -> Self {
        let trump = p.trick().trump();
        let configured_cards = p.configured_cards_for_determinization().unwrap_or_default();
        let configured_counts = Card::count(configured_cards.iter().copied());
        let total_cards = configured_cards.len();
        let total_points = configured_cards
            .iter()
            .map(|card| card.points().unwrap_or(0))
            .sum();
        let total_trumps = configured_cards
            .iter()
            .filter(|card| trump.effective_suit(**card) == EffectiveSuit::Trump)
            .count();
        let hands = p.hands();
        let mut seen: HashMap<Card, usize> = HashMap::new();
        let mut hidden_counts: HashMap<PlayerID, usize> = HashMap::new();

        let note_visible = |seen: &mut HashMap<Card, usize>, card: Card, count: usize| {
            if card != Card::Unknown {
                *seen.entry(card).or_insert(0) += count;
            }
        };

        // Treat only the observer's own held-card identities as visible, even if
        // this function is called on a materialized sampled world during a
        // rollout. Hand sizes are public, so every other seat contributes that
        // many hidden slots. This makes the observation boundary structural and
        // avoids an expensive deep redaction clone on every simulated ply.
        for player in p.propagated().players() {
            let Ok(hand) = hands.get(player.id) else {
                continue;
            };
            if player.id == me {
                for (&card, &count) in hand {
                    note_visible(&mut seen, card, count);
                }
            } else {
                hidden_counts.insert(player.id, hand.values().copied().sum());
            }
        }

        // Current table cards and the complete accumulated public history are
        // distinct locations: current cards enter the history only at trick end.
        for played in p.trick().played_cards() {
            for &card in &played.cards {
                note_visible(&mut seen, card, 1);
            }
        }
        for (&card, &count) in p.played_this_hand() {
            note_visible(&mut seen, card, count);
        }

        // Out-of-hand piles may contain a mixture of known cards and Unknown
        // placeholders. Known copies are removed now; unknown positions become
        // explicit matching slots below.
        let (kitty, removed) = p.piles_for_determinization();
        // The exchanger legitimately saw the kitty. For every other observer,
        // ignore materialized identities even in a sampled world; its size remains
        // part of the hidden-location template. Removed cards are public.
        if p.exchanger() == me {
            for &card in kitty {
                note_visible(&mut seen, card, 1);
            }
        }
        for &card in removed {
            note_visible(&mut seen, card, 1);
        }

        let known_holding = Self::infer_revealed_holdings(p, me);

        Knowledge {
            seen,
            voids: Self::infer_voids(p, me),
            hidden_counts,
            known_holding,
            trump,
            num_decks: p.num_decks(),
            configured_counts,
            total_cards,
            total_points,
            total_trumps,
        }
    }

    pub fn configured_copies(&self, card: Card) -> usize {
        self.configured_counts.get(&card).copied().unwrap_or(0)
    }

    /// Compatibility alias. Full public memory is now universal, so both
    /// constructors intentionally return identical knowledge.
    pub fn from_play_view_full_memory(p: &PlayPhase, me: PlayerID) -> Self {
        Self::from_play_view(p, me)
    }

    fn infer_revealed_holdings(
        p: &PlayPhase,
        me: PlayerID,
    ) -> HashMap<PlayerID, HashMap<Card, usize>> {
        let mut holdings = HashMap::new();
        for trick in p.public_play_history() {
            for played in trick {
                observe_revealed_holding(&mut holdings, played);
            }
        }
        for played in p.trick().played_cards() {
            observe_revealed_holding(&mut holdings, played);
        }
        holdings.remove(&me);
        holdings.retain(|_, cards| !cards.is_empty());
        holdings
    }

    /// Infer hard void constraints from all completed tricks plus the current
    /// in-progress trick. An off-suit card is observed only after the player has
    /// exhausted the led effective suit, so that suit cannot legally be assigned
    /// back to their remaining hand.
    fn infer_voids(p: &PlayPhase, me: PlayerID) -> HashMap<PlayerID, Vec<EffectiveSuit>> {
        let trump = p.trick().trump();
        let mut voids: HashMap<PlayerID, Vec<EffectiveSuit>> = HashMap::new();

        for (&pid, suits) in p.voids_this_hand() {
            if pid == me {
                continue;
            }
            for &suit in suits {
                let entry = voids.entry(pid).or_default();
                if !entry.contains(&suit) {
                    entry.push(suit);
                }
            }
        }

        // The current trick has not yet been folded into `voids_this_hand`.
        let played = p.trick().played_cards();
        if let Some(format) = p.trick().trick_format() {
            let led_suit = format.suit();
            for pc in played.iter().skip(1) {
                if pc.id == me {
                    continue;
                }
                if follow_proves_void(
                    pc,
                    trump,
                    led_suit,
                    format.is_rainbow(),
                    p.propagated().bomb_policy,
                ) {
                    let entry = voids.entry(pc.id).or_default();
                    if !entry.contains(&led_suit) {
                        entry.push(led_suit);
                    }
                }
            }
        }

        // Canonical order makes diagnostics and serialized state stable.
        for suits in voids.values_mut() {
            suits.sort_unstable();
            suits.dedup();
        }
        voids
    }

    /// Exact hidden multiset: configured cards minus every visible physical copy.
    fn hidden_pool(&self, configured_cards: &[Card]) -> Option<Vec<Card>> {
        let mut remaining_seen = self.seen.clone();
        let mut pool = Vec::with_capacity(configured_cards.len());
        for &card in configured_cards {
            match remaining_seen.get_mut(&card) {
                Some(count) if *count > 0 => *count -= 1,
                _ => pool.push(card),
            }
        }
        if remaining_seen.values().any(|&count| count != 0) {
            // The view claims a visible card copy that the configured decks do
            // not contain. Refuse to invent an impossible world.
            return None;
        }
        Some(pool)
    }
}

/// A hidden physical location a proposal model may score. Duplicate capacity
/// slots at one location receive the same model likelihood; exact matching still
/// owns conservation and hand-size constraints.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum HiddenCardLocation {
    Player(PlayerID),
    Kitty,
    Removed,
}

/// Honest context exposed to an optional hidden-location proposal model.
pub struct AssignmentProposalContext<'a> {
    pub observer: PlayerID,
    pub trump: Trump,
    /// The same redacted view supplied to the determinizer. This gives a future
    /// proposal public role/score/rule context without creating another feature
    /// contract or granting access to the true world.
    pub view: &'a PlayPhase,
    pub public_play_history: &'a [Vec<PlayedCards>],
    pub current_trick: &'a [PlayedCards],
    pub voids: &'a HashMap<PlayerID, Vec<EffectiveSuit>>,
    pub hidden_counts: &'a HashMap<PlayerID, usize>,
    pub knowledge: &'a Knowledge,
}

/// Optional proposal over hidden card locations. The unnormalized log likelihood
/// only orders edges already admitted by the hard constraint solver. It can
/// neither introduce an impossible assignment nor remove a legal edge. Returning
/// `0.0` for every input is the neutral/uniform proposal.
pub trait HiddenAssignmentProposal: Send + Sync {
    fn log_weight(
        &self,
        context: &AssignmentProposalContext<'_>,
        card: Card,
        location: HiddenCardLocation,
    ) -> f64;

    /// Batched hook used by learned proposals so one inference scores every
    /// `(card, location)` edge in a sampled world. The default preserves the
    /// simple scalar implementation for heuristic/test proposals.
    fn batch_log_weights(
        &self,
        context: &AssignmentProposalContext<'_>,
        cards: &[Card],
        slots: &[HiddenCardLocation],
    ) -> Vec<Vec<f64>> {
        cards
            .iter()
            .map(|&card| {
                slots
                    .iter()
                    .map(|&location| {
                        if location_accepts(location, card, context.trump, context.voids) {
                            self.log_weight(context, card, location)
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect()
    }
}

fn location_accepts(
    location: HiddenCardLocation,
    card: Card,
    trump: Trump,
    voids: &HashMap<PlayerID, Vec<EffectiveSuit>>,
) -> bool {
    match location {
        HiddenCardLocation::Player(pid) => !voids
            .get(&pid)
            .map(|suits| suits.contains(&trump.effective_suit(card)))
            .unwrap_or(false),
        HiddenCardLocation::Kitty | HiddenCardLocation::Removed => true,
    }
}

/// Random weighted edge order for each card. Exponential-race keys yield a
/// uniform random permutation when all log weights are zero and favor locations
/// with higher proposed likelihood otherwise. Every hard-legal edge remains.
fn proposal_edge_order<R: Rng>(
    cards: &[Card],
    slots: &[HiddenCardLocation],
    context: &AssignmentProposalContext<'_>,
    log_weights: Option<&[Vec<f64>]>,
    rng: &mut R,
) -> Vec<Vec<usize>> {
    if log_weights.is_none() {
        // Fast neutral path used in production today: one uniformly shuffled slot
        // order plus the already shuffled card order. Avoid drawing O(cards ×
        // slots) random numbers per determinization until a proposal is enabled.
        let mut order = (0..slots.len()).collect::<Vec<_>>();
        order.shuffle(rng);
        return vec![order; cards.len()];
    }

    cards
        .iter()
        .enumerate()
        .map(|(card_idx, &card)| {
            let mut edges = slots
                .iter()
                .enumerate()
                .filter(|(_, &location)| {
                    location_accepts(location, card, context.trump, context.voids)
                })
                .map(|(slot_idx, &_location)| {
                    let raw = log_weights
                        .and_then(|weights| weights.get(card_idx))
                        .and_then(|weights| weights.get(slot_idx))
                        .copied()
                        .unwrap_or(0.0);
                    // Invalid model output is neutral. Clamping avoids numerical
                    // overflow and ensures no learned score becomes a hard mask.
                    let log_weight = if raw.is_finite() {
                        raw.clamp(-20.0, 20.0)
                    } else {
                        0.0
                    };
                    let weight = log_weight.exp();
                    let uniform_open = 1.0 - rng.gen::<f64>(); // (0, 1]
                    let priority = -uniform_open.ln() / weight;
                    (slot_idx, priority)
                })
                .collect::<Vec<_>>();
            edges.sort_by(|(a_idx, a), (b_idx, b)| a.total_cmp(b).then_with(|| a_idx.cmp(b_idx)));
            edges.into_iter().map(|(slot_idx, _)| slot_idx).collect()
        })
        .collect()
}

/// One augmenting-path step for exact card-to-location bipartite matching. Pile
/// slots accept any card; player slots reject cards in proven-void suits. Unlike
/// the former greedy fallback, this can rearrange earlier choices and therefore
/// never needs to violate a void merely because the first shuffled deal dead-ended.
fn augment_card(
    card_idx: usize,
    cards: &[Card],
    slots: &[HiddenCardLocation],
    edge_order: &[Vec<usize>],
    trump: Trump,
    voids: &HashMap<PlayerID, Vec<EffectiveSuit>>,
    visited_slots: &mut [bool],
    slot_to_card: &mut [Option<usize>],
) -> bool {
    for &slot_idx in &edge_order[card_idx] {
        let location = slots[slot_idx];
        if visited_slots[slot_idx] || !location_accepts(location, cards[card_idx], trump, voids) {
            continue;
        }
        visited_slots[slot_idx] = true;
        let previous = slot_to_card[slot_idx];
        if previous.is_none()
            || augment_card(
                previous.unwrap(),
                cards,
                slots,
                edge_order,
                trump,
                voids,
                visited_slots,
                slot_to_card,
            )
        {
            slot_to_card[slot_idx] = Some(card_idx);
            return true;
        }
    }
    false
}

/// Exact uniform rejection sampler over labeled card copies and labeled slots.
/// A uniformly shuffled permutation conditioned on every edge being legal is a
/// uniform draw from feasible matchings. Late-hand void constraints often accept
/// quickly; pathological cases fall back to the feasibility matcher + MCMC below.
fn rejection_matching<R: Rng>(
    cards: &[Card],
    slots: &[HiddenCardLocation],
    trump: Trump,
    voids: &HashMap<PlayerID, Vec<EffectiveSuit>>,
    attempts: usize,
    rng: &mut R,
) -> Option<Vec<Option<usize>>> {
    let mut card_indices: Vec<usize> = (0..cards.len()).collect();
    for _ in 0..attempts {
        card_indices.shuffle(rng);
        if slots
            .iter()
            .zip(&card_indices)
            .all(|(&location, &card_idx)| location_accepts(location, cards[card_idx], trump, voids))
        {
            return Some(card_indices.iter().copied().map(Some).collect());
        }
    }
    None
}

fn safe_log_weight(log_weights: Option<&[Vec<f64>]>, card: usize, slot: usize) -> f64 {
    log_weights
        .and_then(|weights| weights.get(card))
        .and_then(|weights| weights.get(slot))
        .copied()
        .filter(|weight| weight.is_finite())
        .unwrap_or(0.0)
        .clamp(-20.0, 20.0)
}

/// Metropolis permutation moves over an already-feasible matching. The neutral
/// target is uniform over legal labeled assignments; with a proposal it targets
/// the product of the proposal's legal edge weights. Multi-slot permutations
/// permit alternating cycles that pair swaps alone cannot traverse. This is a
/// bounded mixing fallback—not a proof of an exact posterior—and is deliberately
/// preceded by exact rejection sampling whenever that succeeds.
fn refine_matching<R: Rng>(
    cards: &[Card],
    slots: &[HiddenCardLocation],
    trump: Trump,
    voids: &HashMap<PlayerID, Vec<EffectiveSuit>>,
    log_weights: Option<&[Vec<f64>]>,
    rng: &mut R,
    slot_to_card: &mut [Option<usize>],
) {
    let n = slots.len();
    if n < 2 {
        return;
    }
    let steps = n.saturating_mul(24).min(50_000);
    let max_width = n.min(8);
    for _ in 0..steps {
        let width = if max_width == 2 {
            2
        } else {
            rng.gen_range(2..=max_width)
        };
        let selected = index::sample(rng, n, width).into_vec();
        let mut proposed_cards: Vec<usize> = selected
            .iter()
            .filter_map(|&slot| slot_to_card[slot])
            .collect();
        if proposed_cards.len() != width {
            continue;
        }
        proposed_cards.shuffle(rng);
        if !selected
            .iter()
            .zip(&proposed_cards)
            .all(|(&slot, &card_idx)| location_accepts(slots[slot], cards[card_idx], trump, voids))
        {
            continue;
        }

        let log_ratio: f64 = selected
            .iter()
            .zip(&proposed_cards)
            .map(|(&slot, &new_card)| {
                let old_card = slot_to_card[slot].expect("matching slot is populated");
                safe_log_weight(log_weights, new_card, slot)
                    - safe_log_weight(log_weights, old_card, slot)
            })
            .sum();
        if log_ratio >= 0.0 || rng.gen::<f64>() < log_ratio.exp() {
            for (&slot, &card_idx) in selected.iter().zip(&proposed_cards) {
                slot_to_card[slot] = Some(card_idx);
            }
        }
    }
}

fn materialize_unknowns(template: &[Card], sampled: Vec<Card>) -> Option<Vec<Card>> {
    let mut sampled = sampled.into_iter();
    let mut result = Vec::with_capacity(template.len());
    for &card in template {
        result.push(if card == Card::Unknown {
            sampled.next()?
        } else {
            card
        });
    }
    if sampled.next().is_some() {
        return None;
    }
    Some(result)
}

/// Sample a complete assignment of all hidden cards to opponent hands, kitty,
/// and (if redacted) removed-card positions.
///
/// Returns `None` when the redacted state and configured deck have no assignment
/// satisfying all capacities and hard void constraints. Callers should then use
/// their policy-prior fallback; impossible evidence is never silently relaxed.
pub fn sample_hidden_hands<R: Rng>(
    view: &PlayPhase,
    me: PlayerID,
    rng: &mut R,
) -> Option<DeterminizedWorld> {
    sample_hidden_hands_with_proposal(view, me, rng, None)
}

/// Sample with an optional belief proposal. Proposal weights change posterior
/// preference only; the same exact matcher enforces all public hard constraints.
pub fn sample_hidden_hands_with_proposal<R: Rng>(
    view: &PlayPhase,
    me: PlayerID,
    rng: &mut R,
    proposal: Option<&dyn HiddenAssignmentProposal>,
) -> Option<DeterminizedWorld> {
    let knowledge = Knowledge::from_play_view(view, me);
    let configured_cards = view.configured_cards_for_determinization()?;
    let (kitty_template, removed_template) = view.piles_for_determinization();
    let kitty_template = kitty_template.to_vec();
    let removed_template = removed_template.to_vec();
    let mut pool = knowledge.hidden_pool(&configured_cards)?;
    let original_hidden_counts = knowledge.hidden_counts.clone();
    let mut residual_hidden_counts = original_hidden_counts.clone();
    let mut fixed_assignment: HashMap<PlayerID, Vec<Card>> = original_hidden_counts
        .keys()
        .copied()
        .map(|pid| (pid, Vec::new()))
        .collect();

    // Reserve failed-throw cards in the seat that publicly revealed them. These
    // are lower bounds only; all remaining copies continue through the sampler.
    let mut revealed: Vec<_> = knowledge.known_holding.iter().collect();
    revealed.sort_by_key(|(pid, _)| pid.0);
    for (&pid, cards) in revealed {
        let residual = residual_hidden_counts.get_mut(&pid)?;
        let mut cards: Vec<_> = cards.iter().collect();
        cards.sort_by_key(|(card, _)| card.as_char());
        for (&card, &count) in cards {
            if count > *residual
                || !location_accepts(
                    HiddenCardLocation::Player(pid),
                    card,
                    knowledge.trump,
                    &knowledge.voids,
                )
            {
                return None;
            }
            for _ in 0..count {
                let index = pool.iter().position(|candidate| *candidate == card)?;
                pool.swap_remove(index);
                fixed_assignment.get_mut(&pid)?.push(card);
            }
            *residual -= count;
        }
    }
    pool.shuffle(rng);

    let mut targets: Vec<(PlayerID, usize)> = residual_hidden_counts
        .iter()
        .map(|(&pid, &count)| (pid, count))
        .collect();
    targets.sort_by_key(|(pid, _)| pid.0);

    let mut slots = Vec::with_capacity(pool.len());
    for &(pid, count) in &targets {
        slots.extend(std::iter::repeat(HiddenCardLocation::Player(pid)).take(count));
    }
    slots.extend(
        std::iter::repeat(HiddenCardLocation::Kitty).take(
            kitty_template
                .iter()
                .filter(|&&c| c == Card::Unknown)
                .count(),
        ),
    );
    slots.extend(
        std::iter::repeat(HiddenCardLocation::Removed).take(
            removed_template
                .iter()
                .filter(|&&c| c == Card::Unknown)
                .count(),
        ),
    );

    // Every configured physical card must have exactly one location. A mismatch
    // catches stale/custom-deck assumptions instead of dropping leftovers.
    if pool.len() != slots.len() {
        return None;
    }
    let proposal_context = AssignmentProposalContext {
        observer: me,
        trump: knowledge.trump,
        view,
        public_play_history: view.public_play_history(),
        current_trick: view.trick().played_cards(),
        voids: &knowledge.voids,
        hidden_counts: &residual_hidden_counts,
        knowledge: &knowledge,
    };
    let proposal_weights = proposal
        .map(|model| model.batch_log_weights(&proposal_context, &pool, &slots))
        .filter(|weights| {
            weights.len() == pool.len() && weights.iter().all(|row| row.len() == slots.len())
        });
    let weighted = proposal_weights.is_some();
    // Most early-hand worlds have no established voids: a shuffled pool zipped
    // to fixed slots is already an exact uniform assignment and avoids cubic
    // augmenting-path work. With voids, try exact conditional rejection first.
    let mut exact_uniform = knowledge.voids.is_empty();
    let mut slot_to_card = if exact_uniform {
        (0..pool.len()).map(Some).collect::<Vec<_>>()
    } else {
        match rejection_matching(
            &pool,
            &slots,
            knowledge.trump,
            &knowledge.voids,
            if pool.len() > 256 { 16 } else { 96 },
            rng,
        ) {
            Some(matching) => {
                exact_uniform = true;
                matching
            }
            None => vec![None; slots.len()],
        }
    };

    if slot_to_card.iter().any(Option::is_none) {
        let edge_order = proposal_edge_order(
            &pool,
            &slots,
            &proposal_context,
            proposal_weights.as_deref(),
            rng,
        );
        for card_idx in 0..pool.len() {
            let mut visited_slots = vec![false; slots.len()];
            if !augment_card(
                card_idx,
                &pool,
                &slots,
                &edge_order,
                knowledge.trump,
                &knowledge.voids,
                &mut visited_slots,
                &mut slot_to_card,
            ) {
                return None;
            }
        }
    }
    if !exact_uniform || weighted {
        refine_matching(
            &pool,
            &slots,
            knowledge.trump,
            &knowledge.voids,
            proposal_weights.as_deref(),
            rng,
            &mut slot_to_card,
        );
    }

    let mut assignment = fixed_assignment;
    let mut sampled_kitty = Vec::new();
    let mut sampled_removed = Vec::new();
    for (&location, card_idx) in slots.iter().zip(slot_to_card.into_iter()) {
        let card = pool[card_idx?];
        match location {
            HiddenCardLocation::Player(pid) => assignment.get_mut(&pid)?.push(card),
            HiddenCardLocation::Kitty => sampled_kitty.push(card),
            HiddenCardLocation::Removed => sampled_removed.push(card),
        }
    }

    for (&pid, &expected) in &original_hidden_counts {
        if assignment.get(&pid).map(Vec::len) != Some(expected) {
            return None;
        }
    }

    // Build complete hands from any visible cards plus the sampled hidden cards.
    let mut hands = Hands::new(view.propagated().players().iter().map(|p| p.id));
    hands.set_trump(knowledge.trump);
    for player in view.propagated().players() {
        let visible = view
            .hands()
            .get(player.id)
            .ok()?
            .iter()
            .filter(|(card, _)| **card != Card::Unknown)
            .flat_map(|(&card, &count)| std::iter::repeat(card).take(count))
            .collect::<Vec<_>>();
        hands.add(player.id, visible).ok()?;
        if let Some(cards) = assignment.remove(&player.id) {
            hands.add(player.id, cards).ok()?;
        }
    }

    let kitty = materialize_unknowns(&kitty_template, sampled_kitty)?;
    let removed = materialize_unknowns(&removed_template, sampled_removed)?;
    let play = view.clone_with_determinized_cards(hands, kitty, removed);
    Some(DeterminizedWorld { play })
}

#[cfg(test)]
mod sampler_calibration_tests {
    use super::{
        observe_revealed_holding, refine_matching, rejection_matching, HiddenCardLocation,
    };
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use shengji_mechanics::trick::PlayedCards;
    use shengji_mechanics::types::{Card, EffectiveSuit, Number, PlayerID, Suit, Trump};
    use std::collections::HashMap;

    #[test]
    fn failed_throw_revelations_follow_the_card_until_it_is_played() {
        let player = PlayerID(7);
        let three = Card::Suited {
            suit: Suit::Clubs,
            number: Number::Three,
        };
        let four = Card::Suited {
            suit: Suit::Clubs,
            number: Number::Four,
        };
        let mut holdings = HashMap::new();
        observe_revealed_holding(
            &mut holdings,
            &PlayedCards {
                id: player,
                cards: vec![],
                bad_throw_cards: vec![three, three, four],
                better_player: None,
            },
        );
        observe_revealed_holding(
            &mut holdings,
            &PlayedCards {
                id: player,
                cards: vec![three, four],
                bad_throw_cards: vec![],
                better_player: None,
            },
        );
        assert_eq!(holdings[&player].get(&three), Some(&1));
        assert!(!holdings[&player].contains_key(&four));
    }

    #[test]
    fn exact_rejection_is_balanced_over_small_feasible_matchings() {
        let cards = vec![
            Card::Suited {
                suit: Suit::Clubs,
                number: Number::Three,
            },
            Card::Suited {
                suit: Suit::Diamonds,
                number: Number::Four,
            },
            Card::Suited {
                suit: Suit::Spades,
                number: Number::Five,
            },
        ];
        let slots = vec![
            HiddenCardLocation::Player(PlayerID(0)),
            HiddenCardLocation::Player(PlayerID(1)),
            HiddenCardLocation::Player(PlayerID(2)),
        ];
        let trump = Trump::NoTrump { number: None };
        let voids = HashMap::from([
            (PlayerID(1), vec![EffectiveSuit::Clubs]),
            (PlayerID(2), vec![EffectiveSuit::Diamonds]),
        ]);
        let mut counts: HashMap<Vec<usize>, usize> = HashMap::new();
        for seed in 0..6_000u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let assignment = rejection_matching(&cards, &slots, trump, &voids, 96, &mut rng)
                .expect("small constrained graph should accept");
            let key: Vec<usize> = assignment.into_iter().map(Option::unwrap).collect();
            *counts.entry(key).or_default() += 1;
        }
        assert_eq!(counts.len(), 3);
        for count in counts.values() {
            assert!(
                (1_750..=2_250).contains(count),
                "uniform target is ~2000 per matching, got {:?}",
                counts
            );
        }
    }

    #[test]
    fn weighted_refinement_tracks_enumerable_two_by_two_target() {
        let cards = vec![
            Card::Suited {
                suit: Suit::Clubs,
                number: Number::Three,
            },
            Card::Suited {
                suit: Suit::Diamonds,
                number: Number::Four,
            },
        ];
        let slots = vec![HiddenCardLocation::Kitty, HiddenCardLocation::Removed];
        // Identity has total log weight 2; swap has 0, so the exact identity
        // probability is exp(2)/(exp(2)+1) ~= 0.881.
        let weights = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let voids = HashMap::new();
        let mut identity = 0usize;
        let trials = 4_000usize;
        for seed in 0..trials as u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut assignment = vec![Some(0), Some(1)];
            refine_matching(
                &cards,
                &slots,
                Trump::NoTrump { number: None },
                &voids,
                Some(&weights),
                &mut rng,
                &mut assignment,
            );
            identity += usize::from(assignment == [Some(0), Some(1)]);
        }
        let observed = identity as f64 / trials as f64;
        let expected = 2.0f64.exp() / (2.0f64.exp() + 1.0);
        assert!(
            (observed - expected).abs() < 0.03,
            "weighted target {:.3}, observed {:.3}",
            expected,
            observed
        );
    }
}
