use anyhow::{anyhow, bail, Error};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use shengji_mechanics::bidding::Bid;
use shengji_mechanics::deck::Deck;
use shengji_mechanics::hands::Hands;
use shengji_mechanics::types::{Card, PlayerID, Rank, Trump};

use crate::message::MessageVariant;
use crate::settings::{FirstLandlordSelectionPolicy, GameMode, KittyBidPolicy, PropagatedState};

use crate::game_state::exchange_phase::ExchangePhase;
use crate::game_state::initialize_phase::InitializePhase;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DrawPhase {
    num_decks: usize,
    game_mode: GameMode,
    deck: Vec<Card>,
    propagated: PropagatedState,
    hands: Hands,
    bids: Vec<Bid>,
    #[serde(default)]
    autobid: Option<Bid>,
    position: usize,
    kitty: Vec<Card>,
    #[serde(default)]
    revealed_cards: usize,
    level: Option<Rank>,
    #[serde(default)]
    removed_cards: Vec<Card>,
    #[serde(default)]
    decks: Vec<Deck>,
    player_requested_reset: Option<PlayerID>,
    /// The seats that have explicitly marked themselves "done bidding" (clicked
    /// the temporary "Done bidding" button). The bidding window stays open until
    /// EVERY human (non-bot) seat is in this set; a NEW bid clears it so everyone
    /// re-confirms in response. Bots are never added here — they count as
    /// implicitly done (see [`DrawPhase::all_humans_done_bidding`]). Defaulted so
    /// older serialized states (without the field) deserialize cleanly.
    #[serde(default)]
    done_bidding: Vec<PlayerID>,
}

impl DrawPhase {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        propagated: PropagatedState,
        position: usize,
        deck: Vec<Card>,
        kitty: Vec<Card>,
        num_decks: usize,
        game_mode: GameMode,
        level: Option<Rank>,
        decks: Vec<Deck>,
        removed_cards: Vec<Card>,
    ) -> Self {
        DrawPhase {
            hands: Hands::new(propagated.players.iter().map(|p| p.id)),
            deck,
            kitty,
            propagated,
            position,
            num_decks,
            decks,
            game_mode,
            level,
            removed_cards,
            bids: Vec::new(),
            revealed_cards: 0,
            autobid: None,
            player_requested_reset: None,
            done_bidding: Vec::new(),
        }
    }

    pub fn propagated(&self) -> &PropagatedState {
        &self.propagated
    }

    pub fn propagated_mut(&mut self) -> &mut PropagatedState {
        &mut self.propagated
    }

    /// The (possibly redacted) hands drawn so far. Used by the bot policy to
    /// evaluate bidding strength from the acting player's own hand.
    pub fn hands(&self) -> &Hands {
        &self.hands
    }

    pub fn removed_cards(&self) -> &[Card] {
        &self.removed_cards
    }

    pub fn deck(&self) -> &[Card] {
        &self.deck
    }

    pub fn kitty(&self) -> &[Card] {
        &self.kitty
    }

    /// Exact number of physical cards participating in this hand (excluding
    /// publicly removed cards), independent of standard/special deck shape.
    pub fn cards_in_play(&self) -> usize {
        self.deck.len()
            + self.kitty.len()
            + self
                .propagated
                .players
                .iter()
                .filter_map(|player| self.hands.get(player.id).ok())
                .map(|hand| hand.values().sum::<usize>())
                .sum::<usize>()
    }

    #[cfg(test)]
    pub fn deck_mut(&mut self) -> &mut Vec<Card> {
        &mut self.deck
    }

    #[cfg(test)]
    pub fn position_mut(&mut self) -> &mut usize {
        &mut self.position
    }

    #[cfg(test)]
    pub fn kitty_mut(&mut self) -> &mut Vec<Card> {
        &mut self.kitty
    }

    pub fn add_observer(&mut self, name: String) -> Result<PlayerID, Error> {
        self.propagated.add_observer(name)
    }

    pub fn remove_observer(&mut self, id: PlayerID) -> Result<(), Error> {
        self.propagated.remove_observer(id)
    }

    pub fn next_player(&self) -> Result<PlayerID, Error> {
        if self.deck.is_empty() {
            let (first_bid, winning_bid) = Bid::first_and_winner(&self.bids, self.autobid)?;
            let landlord = self.propagated.landlord.unwrap_or(
                match self.propagated.first_landlord_selection_policy {
                    FirstLandlordSelectionPolicy::ByWinningBid => winning_bid.id,
                    FirstLandlordSelectionPolicy::ByFirstBid => first_bid.id,
                },
            );

            Ok(landlord)
        } else {
            Ok(self.propagated.players[self.position].id)
        }
    }

    pub fn draw_card(&mut self, id: PlayerID) -> Result<(), Error> {
        if id != self.propagated.players[self.position].id {
            bail!("not your turn!");
        }
        if let Some(next_card) = self.deck.pop() {
            self.hands.add(id, Some(next_card))?;
            self.position = (self.position + 1) % self.propagated.players.len();
            Ok(())
        } else {
            bail!("no cards left in deck")
        }
    }

    pub fn reveal_card(&mut self) -> Result<MessageVariant, Error> {
        if !self.deck.is_empty() {
            bail!("can't reveal card until deck is fully drawn")
        }
        if !self.bids.is_empty() {
            bail!("can't reveal card if at least one bid has been made")
        }
        let id = self
            .propagated
            .landlord
            .ok_or_else(|| anyhow!("can't reveal card if landlord hasn't been selected yet"))?;

        let landlord_level = self
            .propagated
            .players
            .iter()
            .find(|p| p.id == id)
            .ok_or_else(|| anyhow!("Couldn't find landlord level?"))?
            .rank();

        if landlord_level == Rank::NoTrump {
            bail!("can't reveal card if the level is no trump!");
        }

        if self.revealed_cards >= self.kitty.len() || self.autobid.is_some() {
            bail!("can't reveal any more cards")
        }

        let level = self
            .propagated
            .players
            .iter()
            .find(|p| p.id == id)
            .map(|p| p.rank())
            .ok_or_else(|| anyhow!("can't find landlord level?"))?;

        let card = self.kitty[self.revealed_cards];

        match self.propagated.kitty_bid_policy {
            KittyBidPolicy::FirstCard => {
                self.autobid = Some(Bid {
                    count: 1,
                    id,
                    card,
                    epoch: 0,
                });
            }
            KittyBidPolicy::FirstCardOfLevelOrHighest
                if card.is_joker() || card.number().map(Rank::Number) == Some(level) =>
            {
                self.autobid = Some(Bid {
                    count: 1,
                    id,
                    card,
                    epoch: 0,
                });
            }
            KittyBidPolicy::FirstCardOfLevelOrHighest
                if self.revealed_cards >= self.kitty.len() - 1 =>
            {
                let mut sorted_kitty = self.kitty.clone();
                sorted_kitty.sort_by(|a, b| {
                    Trump::NoTrump {
                        number: match level {
                            Rank::Number(n) => Some(n),
                            Rank::NoTrump => None,
                        },
                    }
                    .compare(*a, *b)
                });
                if let Some(highest_card) = sorted_kitty.last() {
                    self.autobid = Some(Bid {
                        count: 1,
                        id,
                        card: *highest_card,
                        epoch: 0,
                    });
                }
            }
            _ => (),
        }
        self.revealed_cards += 1;

        Ok(MessageVariant::RevealedCardFromKitty)
    }

    pub fn bid(&mut self, id: PlayerID, card: Card, count: usize) -> bool {
        if self.revealed_cards > 0 {
            return false;
        }
        let bid_made = Bid::bid(
            id,
            card,
            count,
            &mut self.bids,
            self.autobid,
            &self.hands,
            &self.propagated.players,
            self.propagated.landlord,
            self.propagated.bid_policy,
            self.propagated.bid_reinforcement_policy,
            self.propagated.joker_bid_policy,
            self.num_decks,
            0,
        );
        // A NEW bid re-opens the bidding window: clear everyone's "done bidding"
        // flag so every human can respond to (and then re-confirm against) the new
        // standing bid. We only clear when a bid was actually accepted.
        if bid_made {
            self.done_bidding.clear();
        }
        bid_made
    }

    /// Toggle whether the given player has marked themselves "done bidding". This
    /// is the explicit replacement for the old time-based counter-bid grace: the
    /// bidding window stays open until every HUMAN seat has marked done. Idempotent
    /// — marking an already-done player done (or clearing an already-cleared one)
    /// is a no-op.
    pub fn set_done_bidding(&mut self, id: PlayerID, ready: bool) {
        if ready {
            if !self.done_bidding.contains(&id) {
                self.done_bidding.push(id);
            }
        } else {
            self.done_bidding.retain(|p| *p != id);
        }
    }

    /// Whether the given player has marked themselves "done bidding".
    pub fn is_done_bidding(&self, id: PlayerID) -> bool {
        self.done_bidding.contains(&id)
    }

    /// The seat that currently holds the standing (winning) bid, if a bid has been
    /// decided AND the deck has been fully drawn (so the landlord is about to be
    /// finalized). This is exactly the seat that [`DrawPhase::next_player`] returns
    /// once the deck is empty: the player who will pick up the kitty.
    ///
    /// This seat is *never* shown a "Done bidding" button in the UI (they finalize
    /// by picking up the kitty), so requiring them to mark "done" would deadlock if
    /// they are a human. We therefore treat the standing winner as IMPLICITLY done
    /// (see [`DrawPhase::all_humans_done_bidding`]). Recomputed from the live bids,
    /// so when someone outbids, the *new* winner becomes the excluded seat and the
    /// previous winner re-joins the "must mark done" set.
    fn standing_winner(&self) -> Option<PlayerID> {
        if !self.deck.is_empty() || !self.bid_decided() {
            return None;
        }
        // Mirror `next_player`'s landlord selection so the excluded seat is exactly
        // the one that will pick up the kitty.
        let (first_bid, winning_bid) = Bid::first_and_winner(&self.bids, self.autobid).ok()?;
        Some(self.propagated.landlord.unwrap_or(
            match self.propagated.first_landlord_selection_policy {
                FirstLandlordSelectionPolicy::ByWinningBid => winning_bid.id,
                FirstLandlordSelectionPolicy::ByFirstBid => first_bid.id,
            },
        ))
    }

    /// Whether EVERY human (non-bot) seat that still needs to confirm has marked
    /// itself "done bidding". Bots count as implicitly done, so an all-bot table is
    /// trivially `true` (no deadlock). The current standing (winning) bidder is ALSO
    /// implicitly done: that seat finalizes by picking up the kitty and is never
    /// shown a "Done bidding" button, so requiring it would deadlock whenever the
    /// winner is a human. This is the gate that replaces the old timed counter-bid
    /// grace: the landlord may be finalized only once this returns `true`.
    ///
    /// A human with NO legal bid right now (`valid_bids` is empty) is ALSO treated as
    /// implicitly done. Such a seat physically cannot respond to the standing bid, so
    /// requiring them to click "Done bidding" would deadlock the table: e.g. when an
    /// Enoch bot FLIPS the declaration to a higher bid (which clears everyone's "done"
    /// flag, re-opening the window) but the human cannot beat it, the human is
    /// stranded with no way to act AND no longer marked done — the production "bidding
    /// freeze". Counting a no-valid-bids human as implicitly done lets finalization
    /// proceed. (A human who DOES hold a legal bid is unaffected: they must still
    /// click "Done bidding" to release the bots — the normal counter-bid window.)
    pub fn all_humans_done_bidding(&self) -> bool {
        let standing_winner = self.standing_winner();
        self.propagated
            .players
            .iter()
            .filter(|p| self.propagated.is_bot(p.id).is_none())
            // The standing winner finalizes via the kitty pickup, not the "Done
            // bidding" button, so never require them to mark done.
            .filter(|p| Some(p.id) != standing_winner)
            .all(|p| {
                // Explicitly marked done, OR cannot legally respond (no valid bid) so
                // there is nothing to wait on — treat as implicitly done.
                self.done_bidding.contains(&p.id)
                    || self.valid_bids(p.id).map(|b| b.is_empty()).unwrap_or(true)
            })
    }

    pub fn take_back_bid(&mut self, id: PlayerID) -> Result<(), Error> {
        Bid::take_back_bid(id, self.propagated.bid_takeback_policy, &mut self.bids, 0)?;
        // Taking back a bid changes the standing bid, so re-open the window: every
        // human re-confirms "done" against the new state.
        self.done_bidding.clear();
        Ok(())
    }

    /// The legal bids the given player could make right now. Used by the bot
    /// policy to make a minimal (dumb-but-legal) bid so that an all-bot table can
    /// make progress when no landlord has been pre-selected. Returns an empty
    /// vector if the player cannot bid (e.g. cards have already been revealed).
    pub fn valid_bids(&self, id: PlayerID) -> Result<Vec<Bid>, Error> {
        if self.revealed_cards > 0 {
            return Ok(vec![]);
        }
        Bid::valid_bids(
            id,
            &self.bids,
            &self.hands,
            &self.propagated.players,
            self.propagated.landlord,
            0,
            self.propagated.bid_policy,
            self.propagated.bid_reinforcement_policy,
            self.propagated.joker_bid_policy,
            self.num_decks,
        )
    }

    pub fn done_drawing(&self) -> bool {
        self.deck.is_empty()
    }

    /// The number of cards that have been revealed from the bottom of the deck so
    /// far. Together with an auto-bid this is used by the bot driver to decide
    /// whether the reveal-bottom step has already happened.
    pub fn revealed_cards(&self) -> usize {
        self.revealed_cards
    }

    /// Whether the winning bid has already been determined (either through an
    /// auto-bid from revealing the bottom, or a regular bid).
    pub fn bid_decided(&self) -> bool {
        self.autobid.is_some() || !self.bids.is_empty()
    }

    /// The current standing (winning) bid, if any. Public so the bot driver can
    /// let an Enoch bot evaluate whether to "flip" (counter-bid) the standing
    /// declaration with a trump suit its own hand is stronger in.
    pub fn winning_bid(&self) -> Option<Bid> {
        Bid::first_and_winner(&self.bids, self.autobid)
            .ok()
            .map(|(_, winner)| winner)
    }

    pub fn advance(&self, id: PlayerID) -> Result<ExchangePhase, Error> {
        if !self.deck.is_empty() {
            bail!("deck has cards remaining")
        }

        let (landlord, landlord_level) = {
            let landlord = match self.propagated.landlord {
                Some(landlord) => landlord,
                None => {
                    let (first_bid, winning_bid) = Bid::first_and_winner(&self.bids, self.autobid)?;
                    match self.propagated.first_landlord_selection_policy {
                        FirstLandlordSelectionPolicy::ByWinningBid => winning_bid.id,
                        FirstLandlordSelectionPolicy::ByFirstBid => first_bid.id,
                    }
                }
            };

            if id != landlord {
                bail!("only the leader can advance the game");
            }
            let landlord_level = self
                .propagated
                .players
                .iter()
                .find(|p| p.id == landlord)
                .ok_or_else(|| anyhow!("Couldn't find landlord level?"))?
                .rank();
            (landlord, landlord_level)
        };
        let trump = match landlord_level {
            Rank::NoTrump => Trump::NoTrump { number: None },
            Rank::Number(landlord_level) => {
                // Note: this is not repeated in all cases above, but it is
                // repeated in some. It's OK because the bid calculation is
                // fast.
                let (_, winning_bid) = Bid::first_and_winner(&self.bids, self.autobid)?;
                match winning_bid.card {
                    Card::Unknown => bail!("can't bid with unknown cards!"),
                    Card::SmallJoker | Card::BigJoker => Trump::NoTrump {
                        number: Some(landlord_level),
                    },
                    Card::Suited { suit, .. } => Trump::Standard {
                        suit,
                        number: landlord_level,
                    },
                }
            }
        };
        let mut hands = self.hands.clone();
        hands.set_trump(trump);
        Ok(ExchangePhase::new(
            self.propagated.clone(),
            self.num_decks,
            self.game_mode.clone(),
            self.kitty.clone(),
            landlord,
            hands,
            trump,
            self.bids.clone(),
            self.autobid,
            self.removed_cards.clone(),
            self.decks.clone(),
        ))
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
        self.hands.destructively_redact_except_for_player(player);
        for card in &mut self.kitty[self.revealed_cards..] {
            *card = Card::Unknown;
        }
        for card in &mut self.deck {
            *card = Card::Unknown;
        }
    }
}
