//! Card / void tracking and a determinizer for imperfect-information search.
//!
//! # Honesty invariant
//!
//! Everything in this module is derived ONLY from the per-player redacted view
//! ([`GameState::for_player`]). In that view every other seat's cards are
//! [`Card::Unknown`] and the kitty is hidden. We NEVER read the real hands. To
//! reason about the hidden cards, [`sample_hidden_hands`] *samples* a plausible
//! assignment of the unseen cards to the other seats, respecting:
//!
//! * each seat's known hand count (number of [`Card::Unknown`]s it holds),
//! * established per-seat suit voids (inferred from public play history),
//! * the declared trump,
//! * cards already played / visible (so we never deal a card that is accounted
//!   for elsewhere).
//!
//! The result is a fully-determined [`PlayPhase`] (built from the real engine
//! constructor) on which rollouts can run using the genuine game APIs.

use std::collections::HashMap;

use rand::seq::SliceRandom;
use rand::Rng;

use shengji_mechanics::hands::Hands;
use std::cell::RefCell;

use shengji_mechanics::types::{Card, EffectiveSuit, PlayerID, Trump, FULL_DECK};

use crate::game_state::play_phase::PlayPhase;

thread_local! {
    /// TEST-ONLY "oracle belief": `(strength, true_hands)`. Set ONLY by the
    /// benchmark harness (`bot::harness`) to feasibility-test a learned belief
    /// model (Seer): with probability `strength`, [`sample_hidden_hands`] pre-places
    /// each hidden seat's TRUE cards, biasing the determinized world toward reality.
    /// It is `None` in all production paths, so the live bot's sampling is byte-
    /// unchanged and the honesty invariant is untouched (the SEARCH still reads only
    /// the redacted view; only the world-SAMPLER peeks, and only under test).
    static ORACLE_BELIEF: RefCell<Option<(f64, std::collections::HashMap<PlayerID, Vec<Card>>)>> =
        const { RefCell::new(None) };
}

/// Install a test-only oracle belief for the current thread (see [`ORACLE_BELIEF`]).
pub fn set_oracle_belief(strength: f64, true_hands: std::collections::HashMap<PlayerID, Vec<Card>>) {
    ORACLE_BELIEF.with(|o| *o.borrow_mut() = Some((strength, true_hands)));
}

/// Clear the test-only oracle belief (restores pure honest sampling).
pub fn clear_oracle_belief() {
    ORACLE_BELIEF.with(|o| *o.borrow_mut() = None);
}

/// A determinized "world": a guess at every seat's full hand, consistent with
/// the acting player's knowledge.
pub struct DeterminizedWorld {
    pub play: PlayPhase,
}

/// Tracks, from the redacted view + public play history, which cards are no
/// longer hidden and which seats are known to be void in a given suit.
pub struct Knowledge {
    /// Cards the acting player can see (own hand + everything on the table +
    /// everything in completed tricks). These are removed from the pool of
    /// hidden cards to be dealt to opponents.
    pub seen: HashMap<Card, usize>,
    /// Per-seat established voids (effective suits the seat is known to lack).
    pub voids: HashMap<PlayerID, Vec<EffectiveSuit>>,
    /// Number of unknown (hidden) cards each seat currently holds.
    pub hidden_counts: HashMap<PlayerID, usize>,
    pub trump: Trump,
    pub num_decks: usize,
}

impl Knowledge {
    /// Derive knowledge from a redacted [`PlayPhase`] view for player `me`.
    ///
    /// This is the DEFAULT (limited-memory) seeding used by Easy / Expert /
    /// Omniscient: `seen` covers the own hand + the current trick + ONLY the last
    /// completed trick (the engine's `last_trick`). Earlier tricks are forgotten.
    pub fn from_play_view(p: &PlayPhase, me: PlayerID) -> Self {
        Self::from_play_view_impl(p, me, false)
    }

    /// Derive knowledge with PERFECT MEMORY of every card played so far this hand
    /// (own hand + current trick + the FULL public play history). Used ONLY by the
    /// Enoch tier so its boss-card / guaranteed-winner detection is EXACT.
    ///
    /// HONEST: the full history ([`PlayPhase::played_this_hand`]) is the public
    /// record of cards every seat watched hit the table — never a hidden hand.
    pub fn from_play_view_full_memory(p: &PlayPhase, me: PlayerID) -> Self {
        Self::from_play_view_impl(p, me, true)
    }

    fn from_play_view_impl(p: &PlayPhase, me: PlayerID, full_memory: bool) -> Self {
        let trump = p.trick().trump();
        let num_decks = p.num_decks();
        let hands = p.hands();

        let mut seen: HashMap<Card, usize> = HashMap::new();
        let mut hidden_counts: HashMap<PlayerID, usize> = HashMap::new();

        // Our own hand is fully visible.
        if let Ok(my_hand) = hands.get(me) {
            for (card, ct) in my_hand {
                if *card != Card::Unknown {
                    *seen.entry(*card).or_insert(0) += *ct;
                }
            }
        }

        // Every other seat shows only Card::Unknown; count how many they hold.
        for player in p.propagated().players() {
            if player.id == me {
                continue;
            }
            let count = hands
                .counts(player.id)
                .map(|h| h.values().sum::<usize>())
                .unwrap_or(0);
            hidden_counts.insert(player.id, count);
        }

        // Cards already played and resting on the table (current trick) are
        // visible to everyone.
        for pc in p.trick().played_cards() {
            for card in &pc.cards {
                if *card != Card::Unknown {
                    *seen.entry(*card).or_insert(0) += 1;
                }
            }
        }

        if full_memory {
            // Enoch: seed from the FULL public play history of completed tricks —
            // every card every seat watched go down this hand, not just the last
            // trick. Honest (these cards are public) and EXACT.
            for (card, ct) in p.played_this_hand() {
                if *card != Card::Unknown {
                    *seen.entry(*card).or_insert(0) += *ct;
                }
            }
        } else if let Some(last) = p.last_trick() {
            // Default (limited memory): only the most recently completed trick
            // (the engine keeps the last trick around).
            for pc in last.played_cards() {
                for card in &pc.cards {
                    if *card != Card::Unknown {
                        *seen.entry(*card).or_insert(0) += 1;
                    }
                }
            }
        }

        let voids = Self::infer_voids(p, me, full_memory);

        Knowledge {
            seen,
            voids,
            hidden_counts,
            trump,
            num_decks,
        }
    }

    /// Infer per-seat voids from the public history. A seat is void in the led
    /// suit of a trick if it followed with cards of a different effective suit
    /// (i.e. it couldn't follow).
    ///
    /// With `full_memory` (the Enoch tier) we seed from the engine's accumulated
    /// FULL-hand void log ([`PlayPhase::voids_this_hand`], every completed trick)
    /// and then add the current in-progress trick. Otherwise (Easy/Expert) we use
    /// only the current trick + the single retained last trick — the limited
    /// history available without the log.
    fn infer_voids(
        p: &PlayPhase,
        me: PlayerID,
        full_memory: bool,
    ) -> HashMap<PlayerID, Vec<EffectiveSuit>> {
        let trump = p.trick().trump();
        let mut voids: HashMap<PlayerID, Vec<EffectiveSuit>> = HashMap::new();

        let mut note_void = |pid: PlayerID, suit: EffectiveSuit| {
            let entry = voids.entry(pid).or_default();
            if !entry.contains(&suit) {
                entry.push(suit);
            }
        };

        // Full memory (Enoch): seed from the accumulated full-hand void log — every
        // completed trick, not just the last. Honest (off-suit follows are public).
        if full_memory {
            for (pid, suits) in p.voids_this_hand() {
                if *pid == me {
                    continue;
                }
                for suit in suits {
                    note_void(*pid, *suit);
                }
            }
        }

        let mut scan = |trick: &shengji_mechanics::trick::Trick| {
            let played = trick.played_cards();
            if played.is_empty() {
                return;
            }
            // The led suit is the effective suit of the first card the leader
            // played.
            let led_suit = match played[0].cards.first() {
                Some(c) => trump.effective_suit(*c),
                None => return,
            };
            for pc in played.iter().skip(1) {
                if pc.id == me {
                    continue;
                }
                // If any played card is off-suit, the seat couldn't fully follow
                // and is therefore void (or short) in the led suit. We treat
                // "played an off-suit card" as a void signal: an honest follower
                // only plays off-suit when it has run out of the led suit.
                let played_off_suit = pc
                    .cards
                    .iter()
                    .any(|c| trump.effective_suit(*c) != led_suit);
                if played_off_suit {
                    note_void(pc.id, led_suit);
                }
            }
        };

        // Always scan the CURRENT (in-progress) trick: it is not yet in the
        // completed-trick void log seeded above.
        scan(p.trick());
        // Limited memory also folds in the single retained last trick; full memory
        // already covered every completed trick (incl. the last) from the log.
        if !full_memory {
            if let Some(last) = p.last_trick() {
                scan(last);
            }
        }

        voids
    }

    /// The multiset of cards that are hidden from the acting player and must be
    /// distributed to the other seats. This is the full deck (num_decks copies)
    /// minus everything the player has seen. The kitty (buried cards) is NOT in
    /// any seat's hand, so we must also account for it: we compute the total
    /// hidden-seat capacity and only deal that many cards, leaving the rest
    /// (which correspond to the kitty / removed cards) undealt.
    fn hidden_pool(&self) -> Vec<Card> {
        let mut pool: Vec<Card> = Vec::new();
        for card in FULL_DECK.iter() {
            let total = self.num_decks;
            let seen = self.seen.get(card).copied().unwrap_or(0);
            for _ in seen..total {
                pool.push(*card);
            }
        }
        pool
    }
}

/// Sample a plausible full assignment of hidden cards to the other seats,
/// producing a fully-determined [`PlayPhase`] for rollouts.
///
/// Returns `None` if a consistent assignment could not be found (e.g. the void
/// constraints are unsatisfiable for the sampled order); callers should fall
/// back to the heuristic in that case.
pub fn sample_hidden_hands<R: Rng>(
    view: &PlayPhase,
    me: PlayerID,
    full_memory: bool,
    rng: &mut R,
) -> Option<DeterminizedWorld> {
    // The Enoch tier samples with PERFECT memory: every card played this hand is
    // excluded from the hidden pool (so already-played cards are never re-dealt to
    // an opponent) and voids are inferred over the full history. Easy/Expert keep
    // the limited last-trick memory so the net's prior matches its training-time
    // world model. Honest either way — only publicly-played cards are excluded.
    let knowledge = if full_memory {
        Knowledge::from_play_view_full_memory(view, me)
    } else {
        Knowledge::from_play_view(view, me)
    };
    let trump = knowledge.trump;

    let mut pool = knowledge.hidden_pool();
    pool.shuffle(rng);

    // Seats to fill (everyone but me), each needing `hidden_counts` cards.
    let mut targets: Vec<(PlayerID, usize)> = knowledge
        .hidden_counts
        .iter()
        .map(|(pid, ct)| (*pid, *ct))
        .filter(|(_, ct)| *ct > 0)
        .collect();
    // Deterministic-ish ordering then shuffle so we don't bias by player id.
    targets.sort_by_key(|(pid, _)| pid.0);
    targets.shuffle(rng);

    let total_needed: usize = targets.iter().map(|(_, ct)| ct).sum();
    if pool.len() < total_needed {
        // Shouldn't happen, but guard against an inconsistent count.
        return None;
    }

    // Greedy constraint-respecting deal: for each card in the (shuffled) pool,
    // place it into a seat that still needs cards and is not void in that card's
    // effective suit. We process seats round-robin-ish by repeatedly trying to
    // satisfy the neediest seats first to avoid dead-ends.
    let mut assignment: HashMap<PlayerID, Vec<Card>> = HashMap::new();
    for (pid, _) in &targets {
        assignment.insert(*pid, Vec::new());
    }
    let mut remaining: HashMap<PlayerID, usize> =
        targets.iter().map(|(pid, ct)| (*pid, *ct)).collect();

    // TEST-ONLY oracle-belief pre-pass (None in production → skipped entirely). With
    // probability `strength`, pre-place each hidden seat's TRUE cards into its hand
    // (removing them from `pool`), so the rest of the deal fills around a world biased
    // toward reality. strength=1 ⇒ the exact true world; strength=0 ⇒ unchanged. This
    // is the cheap upper bound for a learned belief model.
    let oracle = ORACLE_BELIEF.with(|o| o.borrow().clone());
    if let Some((strength, true_hands)) = oracle {
        if strength > 0.0 {
            for (pid, _) in &targets {
                let truth = match true_hands.get(pid) {
                    Some(t) => t,
                    None => continue,
                };
                for &card in truth {
                    if *remaining.get(pid).unwrap_or(&0) == 0 {
                        break;
                    }
                    if !rng.gen_bool(strength.clamp(0.0, 1.0)) {
                        continue;
                    }
                    if let Some(idx) = pool.iter().position(|c| *c == card) {
                        pool.remove(idx);
                        assignment.get_mut(pid).unwrap().push(card);
                        *remaining.get_mut(pid).unwrap() -= 1;
                    }
                }
            }
        }
    }

    let void_of = |pid: PlayerID, suit: EffectiveSuit| -> bool {
        knowledge
            .voids
            .get(&pid)
            .map(|v| v.contains(&suit))
            .unwrap_or(false)
    };

    // First pass: deal each pool card to a legal seat. Track leftovers that
    // couldn't be placed on the first try.
    let mut leftovers: Vec<Card> = Vec::new();
    for card in pool {
        let suit = trump.effective_suit(card);
        // Candidate seats: still need cards and not void in this suit.
        let mut best: Option<PlayerID> = None;
        let mut best_need = 0usize;
        for (pid, _) in &targets {
            let need = *remaining.get(pid).unwrap_or(&0);
            if need == 0 || void_of(*pid, suit) {
                continue;
            }
            if need > best_need {
                best_need = need;
                best = Some(*pid);
            }
        }
        match best {
            Some(pid) => {
                assignment.get_mut(&pid).unwrap().push(card);
                *remaining.get_mut(&pid).unwrap() -= 1;
            }
            None => leftovers.push(card),
        }
    }

    // Second pass: place leftovers into any seat that still needs cards, even if
    // that violates a (soft) void constraint. Voids are heuristic inferences,
    // so relaxing them here keeps the determinization from failing outright
    // while still being correct w.r.t. hand counts and the card multiset.
    for card in leftovers {
        let mut placed = false;
        for (pid, _) in &targets {
            if *remaining.get(pid).unwrap_or(&0) > 0 {
                assignment.get_mut(pid).unwrap().push(card);
                *remaining.get_mut(pid).unwrap() -= 1;
                placed = true;
                break;
            }
        }
        if !placed {
            // All seats full; this card belongs to the kitty/removed pile. Drop.
        }
    }

    // Sanity: every seat must be exactly full.
    for (pid, ct) in &targets {
        if assignment.get(pid).map(|c| c.len()).unwrap_or(0) != *ct {
            return None;
        }
    }

    // Build the determinized Hands: my real hand + sampled opponent hands.
    let mut hands = Hands::new(view.propagated().players().iter().map(|p| p.id));
    hands.set_trump(trump);
    if let Ok(my_hand) = view.hands().get(me) {
        let my_cards: Vec<Card> = Card::cards(my_hand.iter()).copied().collect();
        hands.add(me, my_cards).ok()?;
    }
    for (pid, cards) in assignment {
        hands.add(pid, cards).ok()?;
    }

    let play = view.clone_with_hands(hands);
    Some(DeterminizedWorld { play })
}
