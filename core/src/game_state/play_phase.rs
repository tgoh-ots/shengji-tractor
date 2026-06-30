use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{anyhow, bail, Error};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use shengji_mechanics::bidding::Bid;
use shengji_mechanics::deck::Deck;
use shengji_mechanics::hands::Hands;
use shengji_mechanics::player::Player;
use shengji_mechanics::scoring::{compute_level_deltas, next_threshold_reachable, GameScoreResult};
use shengji_mechanics::trick::{
    BombPolicy, PlayCards, PlayCardsMessage, PlayedCards, Trick, TrickEnded, TrickUnit,
};
use shengji_mechanics::types::{Card, EffectiveSuit, PlayerID, Rank, Trump};

use crate::bot::BotDifficulty;
use crate::message::MessageVariant;
use crate::settings::{
    AdvancementPolicy, BackToTwoSetting, GameMode, KittyPenalty, MultipleJoinPolicy,
    PlayTakebackPolicy, PropagatedState, ThrowPenalty,
};

use crate::game_state::initialize_phase::InitializePhase;

macro_rules! bail_unwrap {
    ($opt:expr) => {
        match $opt {
            Some(v) => v,
            None => return Err(anyhow!("option was none")),
        }
    };
}

/// Whether an observed follow is hard proof that the seat has exhausted the led
/// effective suit. Ordinary off-suit follows prove a void, but two supported rule
/// exceptions do not: rainbows have no single-suit following obligation, and a
/// bomb may legally override suit following (all bombs under `AllowBombs`, or a
/// trump bomb under `AllowBombsSuitFollowing`). Keeping this predicate shared by
/// persisted history and the in-progress-trick determinizer prevents a legal
/// exotic play from becoming an impossible-world constraint.
pub(crate) fn follow_proves_void(
    played: &PlayedCards,
    trump: Trump,
    led_suit: EffectiveSuit,
    rainbow: bool,
    bomb_policy: BombPolicy,
) -> bool {
    if rainbow || played.cards.is_empty() {
        return false;
    }
    let played_off_suit = played
        .cards
        .iter()
        .any(|card| *card != Card::Unknown && trump.effective_suit(*card) != led_suit);
    if !played_off_suit {
        return false;
    }

    let first = played.cards[0];
    let is_bomb = played.cards.len() >= 4 && played.cards.iter().all(|card| *card == first);
    let bomb_bypasses_following = is_bomb
        && match bomb_policy {
            BombPolicy::AllowBombs => true,
            BombPolicy::AllowBombsSuitFollowing => {
                trump.effective_suit(first) == EffectiveSuit::Trump
            }
            BombPolicy::NoBombs => false,
        };
    !bomb_bypasses_following
}

#[cfg(test)]
mod void_evidence_tests {
    use super::follow_proves_void;
    use shengji_mechanics::trick::{BombPolicy, PlayedCards};
    use shengji_mechanics::types::{Card, EffectiveSuit, Number, PlayerID, Suit, Trump};

    fn played(cards: Vec<Card>) -> PlayedCards {
        PlayedCards {
            id: PlayerID(1),
            cards,
            bad_throw_cards: vec![],
            better_player: None,
        }
    }

    fn card(suit: Suit, number: Number) -> Card {
        Card::Suited { suit, number }
    }

    #[test]
    fn exotic_legal_follows_do_not_create_false_voids() {
        let trump = Trump::Standard {
            suit: Suit::Hearts,
            number: Number::Two,
        };
        let off = card(Suit::Clubs, Number::Nine);
        assert!(!follow_proves_void(
            &played(vec![off]),
            trump,
            EffectiveSuit::Spades,
            true,
            BombPolicy::NoBombs,
        ));
        assert!(!follow_proves_void(
            &played(vec![off; 4]),
            trump,
            EffectiveSuit::Spades,
            false,
            BombPolicy::AllowBombs,
        ));
        let trump_bomb = card(Suit::Hearts, Number::Nine);
        assert!(!follow_proves_void(
            &played(vec![trump_bomb; 4]),
            trump,
            EffectiveSuit::Spades,
            false,
            BombPolicy::AllowBombsSuitFollowing,
        ));
    }

    #[test]
    fn ordinary_and_restricted_offsuit_follows_still_prove_void() {
        let trump = Trump::Standard {
            suit: Suit::Hearts,
            number: Number::Two,
        };
        let off = card(Suit::Clubs, Number::Nine);
        assert!(follow_proves_void(
            &played(vec![off]),
            trump,
            EffectiveSuit::Spades,
            false,
            BombPolicy::NoBombs,
        ));
        assert!(follow_proves_void(
            &played(vec![off; 4]),
            trump,
            EffectiveSuit::Spades,
            false,
            BombPolicy::AllowBombsSuitFollowing,
        ));
    }
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, JsonSchema, Eq, PartialEq)]
pub struct PlayerGameFinishedResult {
    pub won_game: bool,
    pub is_defending: bool,
    pub is_landlord: bool,
    pub ranks_up: usize,
    pub confetti: bool,
    pub rank: Rank,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PlayPhase {
    num_decks: usize,
    game_mode: GameMode,
    propagated: PropagatedState,
    hands: Hands,
    points: HashMap<PlayerID, Vec<Card>>,
    penalties: HashMap<PlayerID, usize>,
    kitty: Vec<Card>,
    landlord: PlayerID,
    landlords_team: Vec<PlayerID>,
    exchanger: PlayerID,
    trump: Trump,
    trick: Trick,
    last_trick: Option<Trick>,
    /// Every card that has been played in a COMPLETED trick this hand, as a
    /// multiset (card -> count). This is HONEST/public: every seat watched these
    /// cards hit the table. Used by every honest bot's full-memory world model
    /// and boss-card detection so no tier "forgets" earlier tricks.
    /// `#[serde(default)]` so older serialized states (which lack this field)
    /// still deserialize without breaking wasm state-sync.
    #[serde(default)]
    played_this_hand: HashMap<Card, usize>,
    /// Completed public play history with trick boundaries and seat attribution.
    /// The multiset above remains a compact compatibility/indexing aid; this log
    /// is the belief-model-ready record needed to condition hidden-card proposals
    /// on *who* played *what* and in which trick. `PlayedCards` also retains throw
    /// metadata. The current in-progress trick remains in [`PlayPhase::trick`].
    #[serde(default)]
    public_play_history: Arc<Vec<Vec<PlayedCards>>>,
    /// False only for a mid-hand snapshot written by an older server that had an
    /// aggregate played-card multiset but no attributed history. Belief models can
    /// then degrade conservatively instead of treating missing history as evidence.
    #[serde(default)]
    public_history_complete: bool,
    /// Public declaration history retained into play for belief proposals.
    #[serde(default)]
    public_bids: Vec<Bid>,
    /// Per-seat suit voids established this hand: a seat is recorded void in the
    /// led effective suit of a completed trick when it could not follow (it played
    /// an off-suit card). HONEST/public — off-suit follows are watched by every
    /// seat. Persist this alongside the played-card multiset: dropping it during a
    /// room save/reload used to make an honest bot forget hard constraints learned
    /// earlier in the hand and sample impossible worlds until those voids happened
    /// to be observed again.
    #[serde(default)]
    voids_this_hand: HashMap<PlayerID, Vec<EffectiveSuit>>,
    game_ended_early: bool,
    #[serde(default)]
    removed_cards: Vec<Card>,
    #[serde(default)]
    decks: Vec<Deck>,
    player_requested_reset: Option<PlayerID>,
}

impl PlayPhase {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        propagated: PropagatedState,
        num_decks: usize,
        game_mode: GameMode,
        hands: Hands,
        kitty: Vec<Card>,
        trump: Trump,
        landlord: PlayerID,
        exchanger: PlayerID,
        landlords_team: Vec<PlayerID>,
        removed_cards: Vec<Card>,
        decks: Vec<Deck>,
        public_bids: Vec<Bid>,
    ) -> Result<Self, Error> {
        let landlord_idx = bail_unwrap!(propagated.players.iter().position(|p| p.id == landlord));
        Ok(PlayPhase {
            trick: Trick::new(
                trump,
                (0..propagated.players.len()).map(|offset| {
                    let idx = (landlord_idx + offset) % propagated.players.len();
                    propagated.players[idx].id
                }),
                propagated.bomb_policy,
            ),
            points: propagated
                .players
                .iter()
                .map(|p| (p.id, Vec::new()))
                .collect(),
            penalties: propagated.players.iter().map(|p| (p.id, 0)).collect(),
            num_decks,
            game_mode,
            hands,
            kitty,
            landlord,
            exchanger,
            landlords_team,
            trump,
            propagated,
            removed_cards,
            decks,
            game_ended_early: false,
            last_trick: None,
            played_this_hand: HashMap::new(),
            public_play_history: Arc::new(Vec::new()),
            public_history_complete: true,
            public_bids,
            voids_this_hand: HashMap::new(),
            player_requested_reset: None,
        })
    }

    pub fn add_observer(&mut self, name: String) -> Result<PlayerID, Error> {
        self.propagated.add_observer(name)
    }

    pub fn remove_observer(&mut self, id: PlayerID) -> Result<(), Error> {
        self.propagated.remove_observer(id)
    }

    pub fn next_player(&self) -> Result<PlayerID, Error> {
        Ok(bail_unwrap!(self.trick.next_player()))
    }

    pub fn game_mode(&self) -> &GameMode {
        &self.game_mode
    }

    pub fn landlords_team(&self) -> &[PlayerID] {
        &self.landlords_team
    }

    pub fn trick(&self) -> &Trick {
        &self.trick
    }

    pub fn hands(&self) -> &Hands {
        &self.hands
    }

    /// The most recently completed trick, if any. Full-memory bot code uses the
    /// accumulated public fields below; this remains useful for display and rules.
    pub fn last_trick(&self) -> Option<&Trick> {
        self.last_trick.as_ref()
    }

    /// The full public history of cards played in COMPLETED tricks this hand, as
    /// a multiset (card -> count). HONEST: every seat saw these cards played, so
    /// this leaks nothing and is included unchanged in the redacted per-player
    /// view. Cards still on the table in the *current* trick are NOT here yet
    /// (read [`PlayPhase::trick`] for those). Used by honest bots for exact card
    /// conservation and boss-card reasoning across the whole hand.
    pub fn played_this_hand(&self) -> &HashMap<Card, usize> {
        &self.played_this_hand
    }

    /// Completed plays grouped by trick, in original seating/action order.
    /// This is honest observation history: it contains only cards/actions that
    /// have already occurred, never cards remaining in a hidden hand or kitty.
    pub fn public_play_history(&self) -> &[Vec<PlayedCards>] {
        self.public_play_history.as_slice()
    }

    pub fn public_history_complete(&self) -> bool {
        self.public_history_complete
    }

    pub fn public_bids(&self) -> &[Bid] {
        &self.public_bids
    }

    /// Per-seat suit voids established across ALL completed tricks this hand (a
    /// seat that played off the led suit is void in it). HONEST/public — off-suit
    /// follows are watched by every seat, so this is included unchanged in the
    /// redacted per-player view. Used by every honest determinization to enforce
    /// hard full-history constraints rather than sample impossible hands.
    pub fn voids_this_hand(&self) -> &HashMap<PlayerID, Vec<EffectiveSuit>> {
        &self.voids_this_hand
    }

    /// The number of decks in play.
    pub fn num_decks(&self) -> usize {
        self.num_decks
    }

    /// The buried kitty cards, IF they are visible in this (possibly redacted)
    /// view. The kitty is un-hidden ONLY for the exchanger (the landlord who
    /// buried it) until the end of the game; for every other seat the cards are
    /// [`Card::Unknown`] (see [`PlayPhase::destructively_redact_for_player`]).
    /// Returns `None` when the kitty is hidden so honest callers (e.g. the Enoch
    /// bot's endgame kitty-protection logic) never act on garbage. This preserves
    /// the honesty boundary: only the seat that actually saw the kitty can read
    /// its point value.
    pub fn visible_kitty(&self) -> Option<&[Card]> {
        if self.kitty.contains(&Card::Unknown) {
            None
        } else {
            Some(&self.kitty)
        }
    }

    /// The NUMBER of cards buried in the kitty. This is public information (the
    /// redacted view replaces the kitty cards with [`Card::Unknown`] but keeps the
    /// count), so unlike [`PlayPhase::visible_kitty`] every seat may read it — used
    /// by the bot heuristic to ESTIMATE the kitty's point value from the unseen
    /// card pool without seeing the actual buried cards.
    pub fn kitty_size(&self) -> usize {
        self.kitty.len()
    }

    /// The scoring "step size" (points per level threshold) for this hand, used
    /// by the bot heuristic to reason about how close the attacking team is to
    /// flipping the round. Returns `None` if the configured parameters are
    /// invalid for the current decks (the caller then disables threshold logic
    /// rather than guessing). This reads only public, observable game settings.
    pub fn bot_step_size(&self) -> Option<isize> {
        self.propagated
            .game_scoring_parameters
            .step_size(&self.decks)
            .ok()
            .map(|s| s as isize)
    }

    /// The landlord seat.
    pub fn landlord(&self) -> PlayerID {
        self.landlord
    }

    /// Seat that actually handled and therefore observed the buried kitty. This
    /// can differ from the landlord when kitty theft is enabled.
    pub fn exchanger(&self) -> PlayerID {
        self.exchanger
    }

    /// The trump for this hand.
    pub fn trump(&self) -> Trump {
        self.trump
    }

    /// Exact configured card multiset for this hand. The stored `decks` are the
    /// materialized configuration used when the hand was dealt (including short
    /// or joker-less special decks), rather than an assumed standard deck.
    ///
    /// `pub(crate)` intentionally keeps this as a simulation API. Bot callers must
    /// still receive a per-player redacted [`PlayPhase`]; the central observation
    /// boundary is what prevents access to hidden hands and the hidden kitty.
    pub(crate) fn configured_cards_for_determinization(&self) -> Option<Vec<Card>> {
        let decks = if self.decks.is_empty() {
            // Backwards compatibility for a room serialized before `decks` was
            // persisted on PlayPhase. Re-materialize the public configuration.
            self.propagated.decks().ok()?
        } else {
            self.decks.clone()
        };
        Some(decks.iter().flat_map(Deck::cards).collect())
    }

    /// The (possibly redacted) physical piles that are outside players' hands.
    /// Unknown placeholders communicate only pile size. Removed cards are public
    /// in the game UI today, but treating unknown placeholders generically keeps
    /// the determinizer correct if that visibility rule changes later.
    pub(crate) fn piles_for_determinization(&self) -> (&[Card], &[Card]) {
        (&self.kitty, &self.removed_cards)
    }

    /// Build a fully materialized sampled world for bot rollouts. Hands and both
    /// out-of-hand piles are replaced together, so terminal scoring sees the
    /// sampled buried kitty rather than the redacted `Card::Unknown` placeholders.
    pub(crate) fn clone_with_determinized_cards(
        &self,
        hands: Hands,
        kitty: Vec<Card>,
        removed_cards: Vec<Card>,
    ) -> PlayPhase {
        let mut clone = self.clone();
        clone.hands = hands;
        clone.kitty = kitty;
        clone.removed_cards = removed_cards;
        clone
    }

    pub fn propagated(&self) -> &PropagatedState {
        &self.propagated
    }

    pub fn propagated_mut(&mut self) -> &mut PropagatedState {
        &mut self.propagated
    }

    pub fn can_play_cards(&self, id: PlayerID, cards: &[Card]) -> Result<(), Error> {
        if self.game_ended_early {
            bail!("Game has already ended; cards can't be played");
        }
        Ok(self.trick.can_play_cards(
            id,
            &self.hands,
            cards,
            self.propagated.trick_draw_policy,
            self.propagated.compound_formats.clone(),
        )?)
    }

    pub fn play_cards(
        &mut self,
        id: PlayerID,
        cards: &[Card],
    ) -> Result<Vec<MessageVariant>, Error> {
        self.play_cards_with_hint(id, cards, None)
    }

    pub fn play_cards_with_hint(
        &mut self,
        id: PlayerID,
        cards: &[Card],
        format_hint: Option<&'_ [TrickUnit]>,
    ) -> Result<Vec<MessageVariant>, Error> {
        if self.game_ended_early {
            bail!("Game has already ended; cards can't be played");
        }

        let mut msgs = self.trick.play_cards(PlayCards {
            id,
            hands: &mut self.hands,
            cards,
            trick_draw_policy: self.propagated.trick_draw_policy,
            throw_eval_policy: self.propagated.throw_evaluation_policy,
            format_hint,
            hide_throw_halting_player: self.propagated.hide_throw_halting_player,
            tractor_requirements: self.propagated.tractor_requirements,
            bomb_policy: self.propagated.bomb_policy,
            compound_formats: self.propagated.compound_formats.clone(),
        })?;
        if self.propagated.hide_played_cards {
            for msg in &mut msgs {
                match msg {
                    PlayCardsMessage::PlayedCards { ref mut cards, .. } => {
                        for card in cards {
                            *card = Card::Unknown;
                        }
                    }
                    PlayCardsMessage::ThrowFailed {
                        ref mut original_cards,
                        ..
                    } => {
                        for card in original_cards {
                            *card = Card::Unknown;
                        }
                    }
                }
            }
        }
        Ok(msgs
            .into_iter()
            .map(|p| match p {
                PlayCardsMessage::ThrowFailed {
                    original_cards,
                    better_player,
                } => MessageVariant::ThrowFailed {
                    original_cards,
                    better_player,
                },
                PlayCardsMessage::PlayedCards { cards } => MessageVariant::PlayedCards { cards },
            })
            .collect())
    }

    pub fn take_back_cards(&mut self, id: PlayerID) -> Result<(), Error> {
        if self.game_ended_early {
            bail!("Game has already ended; cards can't be taken back");
        }
        if self.propagated.play_takeback_policy == PlayTakebackPolicy::NoPlayTakeback {
            bail!("Taking back played cards is not allowed")
        }
        Ok(self
            .trick
            .take_back(id, &mut self.hands, self.propagated.throw_evaluation_policy)?)
    }

    pub fn finish_trick(&mut self) -> Result<Vec<MessageVariant>, Error> {
        if self.game_ended_early {
            bail!("Game has already ended; trick can't be finished");
        }
        let TrickEnded {
            winner,
            points: mut new_points,
            largest_trick_unit_size,
            failed_throw_size,
        } = self.trick.complete()?;

        let kitty_multiplier = match self.propagated.kitty_penalty {
            KittyPenalty::Times => 2 * largest_trick_unit_size,
            KittyPenalty::Power => 2usize.pow(largest_trick_unit_size as u32),
        };

        if failed_throw_size > 0 {
            match self.propagated.throw_penalty {
                ThrowPenalty::None => (),
                ThrowPenalty::TenPointsPerAttempt => {
                    if let Some(id) = self.trick.played_cards().first().map(|pc| pc.id) {
                        *self.penalties.entry(id).or_insert(0) += 10;
                    }
                }
            }
        }

        let mut msgs = vec![];
        if let GameMode::FindingFriends {
            ref mut friends, ..
        } = self.game_mode
        {
            for played in self.trick.played_cards() {
                for card in played.cards.iter() {
                    for friend in friends.iter_mut() {
                        if friend.card == *card {
                            if friend.skip == 0 {
                                if friend.player_id.is_none() {
                                    let already_on_the_team =
                                        self.landlords_team.contains(&played.id);

                                    match self.propagated.multiple_join_policy {
                                        MultipleJoinPolicy::Unrestricted if already_on_the_team => {
                                            // double-join!
                                            friend.player_id = Some(played.id);
                                            msgs.push(MessageVariant::JoinedTeam {
                                                player: played.id,
                                                already_joined: true,
                                            });
                                        }
                                        MultipleJoinPolicy::NoDoubleJoin if already_on_the_team => {
                                        }
                                        MultipleJoinPolicy::Unrestricted
                                        | MultipleJoinPolicy::NoDoubleJoin => {
                                            friend.player_id = Some(played.id);
                                            self.landlords_team.push(played.id);
                                            msgs.push(MessageVariant::JoinedTeam {
                                                player: played.id,
                                                already_joined: false,
                                            });
                                        }
                                    }
                                }
                            } else {
                                friend.skip -= 1;
                            }
                        }
                    }
                }
            }
        }
        let points = bail_unwrap!(self.points.get_mut(&winner));
        let kitty_points = self
            .kitty
            .iter()
            .filter(|c| c.points().is_some())
            .copied()
            .collect::<Vec<_>>();

        if self.hands.is_empty() {
            if self.propagated.should_reveal_kitty_at_end_of_game {
                msgs.push(MessageVariant::EndOfGameKittyReveal {
                    cards: self.kitty.clone(),
                });
            }
            for _ in 0..kitty_multiplier {
                new_points.extend(kitty_points.iter().copied());
            }
            let raw_kitty_points = kitty_points.iter().flat_map(|c| c.points()).sum::<usize>();
            if !kitty_points.is_empty() && kitty_multiplier > 0 {
                msgs.push(MessageVariant::PointsInKitty {
                    points: raw_kitty_points,
                    multiplier: kitty_multiplier,
                });
            }
            // The kitty bonus is attached to the winner of the last trick, but
            // only counts toward the attacking team's total when the attacking
            // (non-landlord) team wins it. Report who the kitty went to.
            msgs.push(MessageVariant::KittyScored {
                kitty_points: raw_kitty_points,
                multiplier: kitty_multiplier,
                awarded_to_landlord_team: self.landlords_team.contains(&winner),
                winner,
            });
        }
        let winner_idx = bail_unwrap!(self.propagated.players.iter().position(|p| p.id == winner));
        if !new_points.is_empty() {
            let trump = self.trump;
            let num_points = new_points.iter().flat_map(|c| c.points()).sum::<usize>();
            points.extend(new_points);
            points.sort_by(|a, b| trump.compare(*a, *b));
            msgs.push(MessageVariant::TrickWon {
                winner: self.propagated.players[winner_idx].id,
                points: num_points,
            });
        } else {
            msgs.push(MessageVariant::TrickWon {
                winner: self.propagated.players[winner_idx].id,
                points: 0,
            });
        }
        let new_trick = Trick::new(
            self.trump,
            (0..self.propagated.players.len()).map(|offset| {
                let idx = (winner_idx + offset) % self.propagated.players.len();
                self.propagated.players[idx].id
            }),
            self.propagated.bomb_policy,
        );
        let completed = std::mem::replace(&mut self.trick, new_trick);
        Arc::make_mut(&mut self.public_play_history).push(completed.played_cards().to_vec());
        // Accumulate every card from the just-completed trick into the honest,
        // public full-hand play history. Tally even `Card::Unknown` plays (when
        // `hide_played_cards` is on) consistently; honest Knowledge ignores
        // Unknown entries, so this never leaks.
        for pc in completed.played_cards() {
            for card in &pc.cards {
                *self.played_this_hand.entry(*card).or_insert(0) += 1;
            }
        }
        // Record, for the full hand, any seat that could NOT follow this trick's
        // led suit (it played an off-suit card) — it is void in the led suit.
        // HONEST: off-suit follows are public. This is the same off-suit signal
        // the determinizer's `infer_voids` uses, but accumulated across every
        // completed trick (the engine keeps only `last_trick`).
        if let Some(format) = completed.trick_format() {
            let played = completed.played_cards();
            let led_suit = format.suit();
            for pc in played.iter().skip(1) {
                if follow_proves_void(
                    pc,
                    self.trump,
                    led_suit,
                    format.is_rainbow(),
                    self.propagated.bomb_policy,
                ) {
                    let entry = self.voids_this_hand.entry(pc.id).or_default();
                    if !entry.contains(&led_suit) {
                        entry.push(led_suit);
                    }
                }
            }
        }
        self.last_trick = Some(completed);

        Ok(msgs)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compute_player_level_deltas<'a, 'b: 'a>(
        players: impl Iterator<Item = &'b mut Player>,
        non_landlord_level_bump: usize,
        landlord_level_bump: usize,
        landlords_team: &'a [PlayerID],
        landlord_won: bool,
        landlord: (PlayerID, Rank),
        advancement_policy: AdvancementPolicy,
        max_rank: Rank,
        last_trick: Option<Trick>,
        jack_variation: BackToTwoSetting,
    ) -> Vec<MessageVariant> {
        let mut msgs = vec![];

        let should_go_back_to_two =
            Self::check_jacks_last_trick(last_trick, jack_variation, landlords_team, landlord.1);

        let result = players
            .map(|player| {
                let is_defending = landlords_team.contains(&player.id);
                let bump = if is_defending {
                    landlord_level_bump
                } else {
                    non_landlord_level_bump
                };
                let mut num_advances = 0;
                let mut was_blocked = false;
                if is_defending && should_go_back_to_two {
                    player.reset_rank();
                };
                let initial_rank = player.rank();

                for bump_idx in 0..bump {
                    let must_defend = match (advancement_policy, player.rank()) {
                        (AdvancementPolicy::Unrestricted, r)
                        | (AdvancementPolicy::Unrestricted, r)
                        | (AdvancementPolicy::DefendPoints, r)
                        | (AdvancementPolicy::DefendPoints, r)
                            if r == max_rank
                                || (r.successor() == Some(max_rank)
                                    && max_rank == Rank::NoTrump) =>
                        {
                            true
                        }
                        (AdvancementPolicy::DefendPoints, Rank::Number(n))
                            if n.points().is_some() =>
                        {
                            true
                        }
                        (AdvancementPolicy::FullyUnrestricted, _)
                        | (AdvancementPolicy::Unrestricted, _)
                        | (AdvancementPolicy::DefendPoints, _) => false,
                    };
                    // In order to advance past NoTrump, the landlord must also be defending
                    // NoTrump.
                    let landlord_must_defend = must_defend && player.rank() == Rank::NoTrump;

                    if must_defend
                        && (!is_defending
                            || bump_idx > 0
                            || (landlord_must_defend && landlord.1 != Rank::NoTrump))
                    {
                        was_blocked = true;
                        break;
                    }

                    player.advance(max_rank);
                    num_advances += 1;
                }
                if num_advances > 0 {
                    msgs.push(MessageVariant::RankAdvanced {
                        player: player.id,
                        new_rank: player.rank(),
                    });
                }
                if was_blocked {
                    msgs.push(MessageVariant::AdvancementBlocked {
                        player: player.id,
                        rank: player.rank(),
                    });
                }

                (
                    player.name.to_string(),
                    PlayerGameFinishedResult {
                        won_game: landlord_won == is_defending,
                        is_defending,
                        is_landlord: landlord.0 == player.id,
                        ranks_up: num_advances,
                        confetti: num_advances > 0
                            && landlord_won
                            && is_defending
                            && initial_rank == max_rank,
                        rank: initial_rank,
                    },
                )
            })
            .collect();

        msgs.push(MessageVariant::GameFinished { result });
        msgs
    }

    pub fn check_jacks_last_trick(
        last_trick: Option<Trick>,
        jack_variation: BackToTwoSetting,
        landlords_team: &[PlayerID],
        landlord_rank: Rank,
    ) -> bool {
        if !jack_variation.is_applicable(landlord_rank) {
            return false;
        }

        // Guard every fallible step: this is reachable from an untrusted
        // `EndGameEarly`/`StartNewGame` sequence where no trick has been played
        // yet, and a panic here would take down the shared (single-process)
        // server for every room. The safe default is "the jack rule does not
        // apply".
        let Some(last_trick) = last_trick else {
            return false;
        };
        let Ok(TrickEnded {
            winner: winner_pid, ..
        }) = last_trick.complete()
        else {
            return false;
        };

        // In any jack variation, the rule can only applies if the non-landord team wins the
        // last trick
        if landlords_team.contains(&winner_pid) {
            return false;
        }

        let lt_played_cards = last_trick.played_cards();
        let Some(PlayedCards { cards, .. }) = lt_played_cards.iter().find(|pc| pc.id == winner_pid)
        else {
            return false;
        };

        // In the jack variation, the last trick must be won with a single (trump) jack
        jack_variation.compute(cards)
    }

    pub fn calculate_points(&self) -> (isize, isize) {
        let mut non_landlords_points: isize = self
            .points
            .iter()
            .filter(|(id, _)| !self.landlords_team.contains(id))
            .flat_map(|(_, cards)| cards)
            .flat_map(|c| c.points())
            .sum::<usize>() as isize;

        let observed_points = self
            .points
            .iter()
            .filter(|(id, _)| {
                !self.propagated.hide_landlord_points || !self.landlords_team.contains(id)
            })
            .flat_map(|(_, cards)| cards)
            .flat_map(|c| c.points())
            .sum::<usize>() as isize;

        for (id, penalty) in &self.penalties {
            if *penalty > 0 {
                if self.landlords_team.contains(id) {
                    non_landlords_points += *penalty as isize;
                } else {
                    non_landlords_points -= *penalty as isize;
                }
            }
        }
        (non_landlords_points, observed_points)
    }

    pub fn game_finished(&self) -> bool {
        self.game_ended_early || self.hands.is_empty() && self.trick.played_cards().is_empty()
    }

    pub fn finish_game_early(&mut self) -> Result<MessageVariant, Error> {
        if self.game_finished() {
            bail!("Game has already ended");
        }
        let (non_landlords_points, observed_points) = self.calculate_points();
        let can_end_early = !next_threshold_reachable(
            &self.propagated.game_scoring_parameters,
            &self.decks,
            non_landlords_points,
            observed_points,
        )?;

        if can_end_early {
            self.game_ended_early = true;
            Ok(MessageVariant::GameEndedEarly)
        } else {
            bail!("Game can't be ended early; there are still points in play")
        }
    }

    /// Score the currently accumulated non-landlord points under this hand's
    /// exact configured decks, thresholds, and Finding Friends team-size bonus.
    ///
    /// On a finished hand this is the authoritative terminal result used by
    /// [`PlayPhase::finish_game`], including level deltas and any bonus level. It
    /// is also intentionally usable before the hand ends as a provisional
    /// "if scoring stopped now" result for bot training/evaluation. Callers that
    /// require a terminal target should first check [`PlayPhase::game_finished`].
    pub fn current_game_score(&self) -> Result<GameScoreResult, Error> {
        let (non_landlords_points, _) = self.calculate_points();
        let smaller_landlord_team = match &self.game_mode {
            GameMode::FindingFriends { num_friends, .. } => {
                self.landlords_team.len() < *num_friends + 1
            }
            GameMode::Tractor => false,
        };
        let fallback_decks;
        let decks = if self.decks.is_empty() {
            // Older serialized rooms predate the materialized `decks` field.
            fallback_decks = self.propagated.decks()?;
            fallback_decks.as_slice()
        } else {
            self.decks.as_slice()
        };
        compute_level_deltas(
            &self.propagated.game_scoring_parameters,
            decks,
            non_landlords_points,
            smaller_landlord_team,
        )
    }

    pub fn finish_game(&self) -> Result<(InitializePhase, bool, Vec<MessageVariant>), Error> {
        let mut msgs = vec![];
        if !self.game_finished() {
            bail!("not done playing yet!")
        }

        let (non_landlords_points, _) = self.calculate_points();

        let mut propagated = self.propagated.clone();

        let GameScoreResult {
            non_landlord_delta: non_landlord_level_bump,
            landlord_delta: landlord_level_bump,
            landlord_won,
            landlord_bonus: bonus_level_earned,
        } = self.current_game_score()?;

        msgs.push(MessageVariant::EndOfGameSummary {
            landlord_won,
            non_landlords_points,
        });

        if bonus_level_earned {
            msgs.push(MessageVariant::BonusLevelEarned);
        };

        let landlord_idx = bail_unwrap!(propagated
            .players
            .iter()
            .position(|p| p.id == self.landlord));

        msgs.extend(Self::compute_player_level_deltas(
            propagated.players.iter_mut(),
            non_landlord_level_bump,
            landlord_level_bump,
            &self.landlords_team[..],
            landlord_won,
            (self.landlord, self.propagated.players[landlord_idx].level),
            propagated.advancement_policy,
            *propagated.max_rank,
            self.last_trick.clone(),
            self.propagated.jack_variation,
        ));

        // Flavor: only when the MATCH ends — a team runs the rank ladder past
        // `max_rank`, bumping its metalevel (see `Player::advance`) — does every
        // Enoch bot on the LOSING side post its catchphrase. `compute_player_level_deltas`
        // above already advanced `propagated.players`, so a metalevel that grew
        // relative to the pre-advancement `self.propagated.players` means this
        // hand won/lost the whole game (not just bumped ranks within it).
        let match_ended = self
            .propagated
            .players
            .iter()
            .zip(propagated.players.iter())
            .any(|(before, after)| after.metalevel > before.metalevel);
        msgs.extend(Self::enoch_loser_chat(
            &propagated,
            &self.landlords_team[..],
            landlord_won,
            match_ended,
        ));

        let mut idx = (landlord_idx + 1) % propagated.players.len();
        let (next_landlord, next_landlord_idx) = loop {
            if landlord_won == self.landlords_team.contains(&propagated.players[idx].id) {
                break (propagated.players[idx].id, idx);
            }
            idx = (idx + 1) % propagated.players.len()
        };

        msgs.push(MessageVariant::NewLandlordForNextGame {
            landlord: propagated.players[next_landlord_idx].id,
        });
        propagated.set_landlord(Some(next_landlord))?;
        propagated.num_games_finished += 1;
        msgs.extend(propagated.make_all_observers_into_players()?);

        Ok((
            InitializePhase::from_propagated(propagated),
            landlord_won,
            msgs,
        ))
    }

    /// Flavor catchphrase: when the MATCH is over (`match_ended`), every Enoch
    /// bot on the LOSING side emits a `BotChat` line attributed to that bot. On an
    /// ordinary hand that only bumped ranks (`!match_ended`), nobody speaks — the
    /// catchphrase fires once per lost *game*, not once per lost *hand*.
    ///
    /// "Losing" mirrors [`PlayerGameFinishedResult::won_game`] (which is set to
    /// `landlord_won == is_defending`): a player lost iff
    /// `landlord_won != is_defending`, i.e. they are on the team that did NOT
    /// win / level up. Only Enoch bots speak; humans and
    /// Easy/Expert/Omniscient bots stay silent. Each losing Enoch bot says it
    /// exactly once.
    fn enoch_loser_chat(
        propagated: &PropagatedState,
        landlords_team: &[PlayerID],
        landlord_won: bool,
        match_ended: bool,
    ) -> Vec<MessageVariant> {
        let mut msgs = vec![];
        if !match_ended {
            return msgs;
        }
        for player in &propagated.players {
            let is_defending = landlords_team.contains(&player.id);
            let lost_hand = landlord_won != is_defending;
            if lost_hand && matches!(propagated.is_bot(player.id), Some(BotDifficulty::Enoch)) {
                msgs.push(MessageVariant::BotChat {
                    from: player.name.clone(),
                    text: "fah i need a shot".to_string(),
                });
            }
        }
        msgs
    }

    pub fn request_reset(
        &mut self,
        player: PlayerID,
    ) -> Result<(Option<InitializePhase>, Vec<MessageVariant>), Error> {
        match self.player_requested_reset {
            Some(p) => {
                // ignore duplicate reset requests from same player
                if p == player {
                    return Ok((None, vec![]));
                }

                let (s, m) = self.return_to_initialize()?;
                Ok((Some(s), m))
            }
            None => {
                self.player_requested_reset = Some(player);
                Ok((None, vec![MessageVariant::ResetRequested]))
            }
        }
    }

    pub fn cancel_reset(&mut self) -> Option<MessageVariant> {
        if self.player_requested_reset.is_some() {
            self.player_requested_reset = None;
            return Some(MessageVariant::ResetCanceled);
        }
        None
    }

    /// The player (if any) who has an outstanding, unconfirmed reset request.
    /// A reset only completes once a *second*, distinct player also requests it.
    pub fn player_requested_reset(&self) -> Option<PlayerID> {
        self.player_requested_reset
    }

    fn return_to_initialize(&self) -> Result<(InitializePhase, Vec<MessageVariant>), Error> {
        let mut msgs = vec![MessageVariant::ResettingGame];

        let mut propagated = self.propagated.clone();
        msgs.extend(propagated.make_all_observers_into_players()?);

        Ok((InitializePhase::from_propagated(propagated), msgs))
    }

    pub fn destructively_redact_for_player(&mut self, player: PlayerID) {
        if self.propagated.hide_landlord_points {
            for (k, v) in self.points.iter_mut() {
                if self.landlords_team.contains(k) {
                    v.clear();
                }
            }
        }
        // Don't redact at the end of the game.
        let game_ongoing = !self.game_ended_early
            && (!self.hands.is_empty() || !self.trick.played_cards().is_empty());
        if game_ongoing {
            self.hands.destructively_redact_except_for_player(player);
        }
        if game_ongoing && player != self.exchanger {
            for card in &mut self.kitty {
                *card = Card::Unknown;
            }
        }
    }
}

#[cfg(test)]
mod enoch_chat_tests {
    use shengji_mechanics::types::PlayerID;

    use crate::bot::BotDifficulty;
    use crate::message::MessageVariant;
    use crate::settings::PropagatedState;

    use super::PlayPhase;

    /// Builds a propagated state with one player per `(name, bot)` entry, where
    /// `bot` is the optional difficulty to register that seat as. Returns the
    /// state plus the assigned `PlayerID`s in order.
    fn make_state(seats: &[(&str, Option<BotDifficulty>)]) -> (PropagatedState, Vec<PlayerID>) {
        let mut propagated = PropagatedState::default();
        let mut ids = vec![];
        for (name, bot) in seats {
            let (id, _) = propagated.add_player((*name).to_string()).unwrap();
            if let Some(difficulty) = bot {
                propagated.register_bot(id, *difficulty);
            }
            ids.push(id);
        }
        (propagated, ids)
    }

    fn bot_chat_lines(msgs: &[MessageVariant]) -> Vec<(String, String)> {
        msgs.iter()
            .filter_map(|m| match m {
                MessageVariant::BotChat { from, text } => Some((from.clone(), text.clone())),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn losing_enoch_bot_says_catchphrase() {
        // Seats: 0 = landlord (human, defending), 1 = Enoch bot (attacking).
        let (propagated, ids) =
            make_state(&[("landlord", None), ("enoch", Some(BotDifficulty::Enoch))]);
        let landlords_team = vec![ids[0]];

        // Landlord (defending) won -> the attacking Enoch bot lost the hand.
        let msgs = PlayPhase::enoch_loser_chat(&propagated, &landlords_team, true, true);

        assert_eq!(
            bot_chat_lines(&msgs),
            vec![("enoch".to_string(), "fah i need a shot".to_string())],
        );
    }

    #[test]
    fn winning_enoch_bot_stays_silent() {
        // Seats: 0 = Enoch landlord (defending), 1 = human (attacking).
        let (propagated, ids) =
            make_state(&[("enoch", Some(BotDifficulty::Enoch)), ("human", None)]);
        let landlords_team = vec![ids[0]];

        // Defending team won -> the Enoch bot is on the WINNING side, so silent.
        let msgs = PlayPhase::enoch_loser_chat(&propagated, &landlords_team, true, true);

        assert!(bot_chat_lines(&msgs).is_empty());
    }

    #[test]
    fn losing_non_enoch_bot_stays_silent() {
        // A LOSING Easy/Expert/Omniscient bot (and a losing human) must NOT speak.
        let (propagated, ids) = make_state(&[
            ("landlord", None),
            ("easy", Some(BotDifficulty::Easy)),
            ("expert", Some(BotDifficulty::Expert)),
            ("omni", Some(BotDifficulty::Omniscient)),
        ]);
        let landlords_team = vec![ids[0]];

        // Defending team won -> seats 1..=3 (all non-Enoch) lost, but none speak.
        let msgs = PlayPhase::enoch_loser_chat(&propagated, &landlords_team, true, true);

        assert!(bot_chat_lines(&msgs).is_empty());
    }

    #[test]
    fn multiple_losing_enoch_bots_each_say_it_once() {
        let (propagated, ids) = make_state(&[
            ("landlord", None),
            ("enoch_a", Some(BotDifficulty::Enoch)),
            ("enoch_b", Some(BotDifficulty::Enoch)),
        ]);
        let landlords_team = vec![ids[0]];

        let msgs = PlayPhase::enoch_loser_chat(&propagated, &landlords_team, true, true);

        assert_eq!(
            bot_chat_lines(&msgs),
            vec![
                ("enoch_a".to_string(), "fah i need a shot".to_string()),
                ("enoch_b".to_string(), "fah i need a shot".to_string()),
            ],
        );
    }

    #[test]
    fn losing_enoch_bot_silent_when_match_not_over() {
        // A losing Enoch bot must stay SILENT on an ordinary hand loss; the
        // catchphrase only fires when the whole match ends. Seats: 0 = landlord
        // (human, defending), 1 = Enoch (attacking, lost the hand).
        let (propagated, ids) =
            make_state(&[("landlord", None), ("enoch", Some(BotDifficulty::Enoch))]);
        let landlords_team = vec![ids[0]];

        // landlord_won = true (Enoch lost the hand) but match_ended = false.
        let msgs = PlayPhase::enoch_loser_chat(&propagated, &landlords_team, true, false);

        assert!(bot_chat_lines(&msgs).is_empty());
    }
}
