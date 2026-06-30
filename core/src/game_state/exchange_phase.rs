use std::collections::HashSet;

use anyhow::{anyhow, bail, Error};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use shengji_mechanics::bidding::Bid;
use shengji_mechanics::deck::Deck;
use shengji_mechanics::hands::Hands;
use shengji_mechanics::types::{Card, Number, PlayerID, Rank, Trump, FULL_DECK};

use crate::message::MessageVariant;
use crate::settings::{
    Friend, FriendSelection, FriendSelectionPolicy, GameMode, KittyTheftPolicy, PropagatedState,
};

use crate::game_state::{initialize_phase::InitializePhase, play_phase::PlayPhase};

macro_rules! bail_unwrap {
    ($opt:expr) => {
        match $opt {
            Some(v) => v,
            None => return Err(anyhow!("option was none")),
        }
    };
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ExchangePhase {
    propagated: PropagatedState,
    num_decks: usize,
    game_mode: GameMode,
    hands: Hands,
    kitty: Vec<Card>,
    kitty_size: usize,
    landlord: PlayerID,
    trump: Trump,
    exchanger: PlayerID,
    #[serde(default)]
    finalized: bool,
    #[serde(default)]
    epoch: usize,
    #[serde(default)]
    bids: Vec<Bid>,
    #[serde(default)]
    autobid: Option<Bid>,
    #[serde(default)]
    removed_cards: Vec<Card>,
    #[serde(default)]
    decks: Vec<Deck>,
    /// Humans that explicitly passed during the current finalized kitty-theft
    /// bidding window. A successful bid or a new exchange epoch clears this so
    /// everyone can respond to the changed standing bid.
    #[serde(default)]
    done_bidding: Vec<PlayerID>,
    player_requested_reset: Option<PlayerID>,
}

impl ExchangePhase {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        propagated: PropagatedState,
        num_decks: usize,
        game_mode: GameMode,
        kitty: Vec<Card>,
        landlord: PlayerID,
        hands: Hands,
        trump: Trump,
        bids: Vec<Bid>,
        autobid: Option<Bid>,
        removed_cards: Vec<Card>,
        decks: Vec<Deck>,
    ) -> Self {
        ExchangePhase {
            kitty_size: kitty.len(),
            num_decks,
            game_mode,
            kitty,
            propagated,
            landlord,
            exchanger: landlord,
            hands,
            trump,
            bids,
            autobid,
            removed_cards,
            decks,
            done_bidding: Vec::new(),
            finalized: false,
            epoch: 1,
            player_requested_reset: None,
        }
    }

    pub fn add_observer(&mut self, name: String) -> Result<PlayerID, Error> {
        self.propagated.add_observer(name)
    }

    pub fn remove_observer(&mut self, id: PlayerID) -> Result<(), Error> {
        self.propagated.remove_observer(id)
    }

    pub fn move_card_to_kitty(&mut self, id: PlayerID, card: Card) -> Result<(), Error> {
        if self.exchanger != id {
            bail!("not the exchanger")
        }
        if self.finalized {
            bail!("cards already finalized")
        }
        self.hands.remove(self.exchanger, Some(card))?;
        self.kitty.push(card);
        Ok(())
    }

    pub fn move_card_to_hand(&mut self, id: PlayerID, card: Card) -> Result<(), Error> {
        if self.exchanger != id {
            bail!("not the exchanger")
        }
        if self.finalized {
            bail!("cards already finalized")
        }
        if let Some(index) = self.kitty.iter().position(|c| *c == card) {
            self.kitty.swap_remove(index);
            self.hands.add(self.exchanger, Some(card))?;
            Ok(())
        } else {
            bail!("card not in the kitty")
        }
    }

    pub fn num_friends(&self) -> usize {
        match self.game_mode {
            GameMode::FindingFriends { num_friends, .. } => num_friends,
            GameMode::Tractor => 0,
        }
    }

    /// Whether the landlord has already supplied the complete friend list.
    pub fn friends_selected(&self) -> bool {
        match &self.game_mode {
            GameMode::FindingFriends {
                num_friends,
                friends,
            } => friends.len() == *num_friends,
            GameMode::Tractor => true,
        }
    }

    fn available_friend_copies(&self, card: Card) -> usize {
        let configured = if self.decks.is_empty() {
            self.num_decks
        } else {
            self.decks
                .iter()
                .filter(|deck| deck.includes_card(card))
                .count()
        };
        configured.saturating_sub(
            self.removed_cards
                .iter()
                .filter(|removed| **removed == card)
                .count(),
        )
    }

    fn validate_friend_selection(&self, friend: &FriendSelection) -> Result<(), Error> {
        if FriendSelectionPolicy::TrumpsIncluded != self.propagated.friend_selection_policy {
            if friend.card.is_joker() || friend.card.number() == self.trump.number() {
                if let Some(n) = self.trump.number() {
                    bail!("you can't pick a joker or a {} as your friend", n.as_str())
                } else {
                    bail!("you can't pick a joker as your friend")
                }
            }
            if self.trump.suit().is_some() && friend.card.suit() == self.trump.suit() {
                bail!("you can't pick a trump suit as your friend")
            }
        }
        if friend.initial_skip >= self.available_friend_copies(friend.card) {
            bail!("need to pick a card that exists!")
        }

        if let FriendSelectionPolicy::HighestCardNotAllowed =
            self.propagated.friend_selection_policy
        {
            match (self.trump.number(), friend.card.number()) {
                (Some(Number::Ace), Some(Number::King)) | (_, Some(Number::Ace)) => {
                    bail!("you can't pick the highest card as your friend")
                }
                _ => (),
            }
        }

        if let FriendSelectionPolicy::PointCardNotAllowed = self.propagated.friend_selection_policy
        {
            let landlord_level = self
                .propagated
                .players
                .iter()
                .find(|p| p.id == self.landlord)
                .ok_or_else(|| anyhow!("Couldn't find landlord level?"))?
                .rank();
            match (landlord_level, friend.card.points(), friend.card.number()) {
                (Rank::Number(Number::Ace), _, Some(Number::King)) => (),
                (_, Some(_), _) => bail!("you can't pick a point card as your friend"),
                (_, _, _) => (),
            }
        }
        Ok(())
    }

    /// All individual friend declarations accepted by the current rules.
    pub fn valid_friend_selections(&self) -> Vec<FriendSelection> {
        FULL_DECK
            .iter()
            .copied()
            .flat_map(|card| {
                let copies = self.available_friend_copies(card);
                (0..copies).map(move |initial_skip| FriendSelection { card, initial_skip })
            })
            .filter(|friend| self.validate_friend_selection(friend).is_ok())
            .collect()
    }

    pub fn set_friends(
        &mut self,
        id: PlayerID,
        iter: impl IntoIterator<Item = FriendSelection>,
    ) -> Result<(), Error> {
        if self.landlord != id {
            bail!("not the landlord")
        }
        let num_friends = match self.game_mode {
            GameMode::FindingFriends { num_friends, .. } => num_friends,
            GameMode::Tractor => return Err(anyhow!("not playing finding friends")),
        };
        let friend_set = iter.into_iter().collect::<HashSet<_>>();
        if num_friends != friend_set.len() {
            bail!("incorrect number of friends")
        }
        // Validate the complete proposal before clearing the existing list. This
        // also lets bot code enumerate choices from the exact same rules.
        for friend in &friend_set {
            self.validate_friend_selection(friend)?;
        }

        if let GameMode::FindingFriends {
            ref mut friends, ..
        } = self.game_mode
        {
            friends.clear();
            friends.extend(friend_set.iter().map(|friend| Friend {
                card: friend.card,
                initial_skip: friend.initial_skip,
                skip: friend.initial_skip,
                player_id: None,
            }));
            Ok(())
        } else {
            bail!("not playing finding friends")
        }
    }

    pub fn finalize(&mut self, id: PlayerID) -> Result<(), Error> {
        if id != self.exchanger {
            bail!("only the exchanger can finalize their cards")
        }
        if self.finalized {
            bail!("Already finalized")
        }
        if self.kitty.len() != self.kitty_size {
            bail!("incorrect number of cards in the bottom")
        }
        self.finalized = true;
        self.done_bidding.clear();
        Ok(())
    }

    pub fn pick_up_cards(&mut self, id: PlayerID) -> Result<(), Error> {
        if !self.finalized {
            bail!("Current exchanger is still exchanging cards!")
        }
        if self.autobid.is_some() {
            bail!("Bid was automatically determined; no overbidding allowed")
        }
        if self.bids.last().map(|b| b.epoch) != Some(self.epoch) {
            bail!("No bids have been made since the last player finished exchanging cards")
        }
        let (_, winning_bid) = Bid::first_and_winner(&self.bids, self.autobid)?;
        if id != winning_bid.id {
            bail!("Only the winner of the bid can pick up the cards")
        }
        self.trump = match winning_bid.card {
            Card::Unknown => bail!("can't bid with unknown cards!"),
            Card::SmallJoker | Card::BigJoker => Trump::NoTrump {
                number: self.trump.number(),
            },
            Card::Suited { suit, .. } => Trump::Standard {
                suit,
                number: self
                    .trump
                    .number()
                    .ok_or_else(|| anyhow!("suited bid requires a numbered trump level"))?,
            },
        };
        self.finalized = false;
        self.epoch += 1;
        self.exchanger = winning_bid.id;
        self.done_bidding.clear();

        Ok(())
    }

    pub fn bid(&mut self, id: PlayerID, card: Card, count: usize) -> bool {
        if !self.finalized || self.autobid.is_some() {
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
            self.epoch,
        );
        if bid_made {
            self.done_bidding.clear();
        }
        bid_made
    }

    pub fn take_back_bid(&mut self, id: PlayerID) -> Result<(), Error> {
        if !self.finalized {
            bail!("Can't take back bid until exchanger is done swapping cards")
        }
        if self.autobid.is_some() {
            bail!("Can't take back bid if the winning bid was automatic")
        }
        Bid::take_back_bid(
            id,
            self.propagated.bid_takeback_policy,
            &mut self.bids,
            self.epoch,
        )?;
        self.done_bidding.clear();
        Ok(())
    }

    pub fn set_done_bidding(&mut self, id: PlayerID, ready: bool) {
        if ready {
            if !self.done_bidding.contains(&id) {
                self.done_bidding.push(id);
            }
        } else {
            self.done_bidding.retain(|player| *player != id);
        }
    }

    pub fn is_done_bidding(&self, id: PlayerID) -> bool {
        self.done_bidding.contains(&id)
    }

    /// Every human other than the seat responsible for resolving this window
    /// must explicitly pass. Bots are implicit; the standing winner resolves by
    /// picking up, while a no-bid landlord resolves by beginning play.
    pub fn all_humans_done_bidding(&self) -> bool {
        let resolver = self
            .current_epoch_winning_bid()
            .map(|bid| bid.id)
            .unwrap_or(self.landlord);
        self.propagated
            .players
            .iter()
            .filter(|player| self.propagated.is_bot(player.id).is_none())
            .filter(|player| player.id != resolver)
            .all(|player| self.done_bidding.contains(&player.id))
    }

    pub fn landlord(&self) -> PlayerID {
        self.landlord
    }

    pub fn exchanger(&self) -> PlayerID {
        self.exchanger
    }

    pub fn finalized(&self) -> bool {
        self.finalized
    }

    pub fn kitty_theft_enabled(&self) -> bool {
        self.propagated.kitty_theft_policy == KittyTheftPolicy::AllowKittyTheft
            && self.autobid.is_none()
    }

    /// Legal overbids for the current exchange epoch. Empty while the exchanger
    /// is still arranging the kitty or when an auto-bid disables theft.
    pub fn valid_bids(&self, id: PlayerID) -> Result<Vec<Bid>, Error> {
        if !self.finalized || self.autobid.is_some() {
            return Ok(vec![]);
        }
        Bid::valid_bids(
            id,
            &self.bids,
            &self.hands,
            &self.propagated.players,
            self.propagated.landlord,
            self.epoch,
            self.propagated.bid_policy,
            self.propagated.bid_reinforcement_policy,
            self.propagated.joker_bid_policy,
            self.num_decks,
        )
    }

    /// The winning bid made since the most recent exchanger finalized, if any.
    pub fn current_epoch_winning_bid(&self) -> Option<Bid> {
        if self.bids.last().map(|b| b.epoch) != Some(self.epoch) {
            return None;
        }
        Bid::first_and_winner(&self.bids, self.autobid)
            .ok()
            .map(|(_, winning)| winning)
    }

    /// The number of cards that belong in the kitty (buried pile).
    pub fn kitty_size(&self) -> usize {
        self.kitty_size
    }

    /// The kitty cards, if they are visible in this (possibly redacted) view.
    /// In a redacted view the kitty is only un-hidden for the current exchanger
    /// before finalization; otherwise the cards are [`Card::Unknown`]. Returns
    /// `None` when the kitty is hidden so callers don't act on garbage.
    pub fn visible_kitty(&self) -> Option<&[Card]> {
        if self.kitty.contains(&Card::Unknown) {
            None
        } else {
            Some(&self.kitty)
        }
    }

    pub fn hands(&self) -> &Hands {
        &self.hands
    }

    pub fn trump(&self) -> Trump {
        self.trump
    }

    pub fn propagated(&self) -> &PropagatedState {
        &self.propagated
    }

    pub fn propagated_mut(&mut self) -> &mut PropagatedState {
        &mut self.propagated
    }

    pub fn next_player(&self) -> Result<PlayerID, Error> {
        if self.propagated.kitty_theft_policy == KittyTheftPolicy::AllowKittyTheft
            && self.autobid.is_none()
            && !self.finalized
        {
            Ok(self.exchanger)
        } else {
            Ok(self.landlord)
        }
    }

    pub fn advance(&self, id: PlayerID) -> Result<PlayPhase, Error> {
        if id != self.landlord {
            bail!("only the leader can advance the game")
        }
        if self.kitty.len() != self.kitty_size {
            bail!("incorrect number of cards in the bottom")
        }
        if let GameMode::FindingFriends {
            num_friends,
            ref friends,
        } = self.game_mode
        {
            if friends.len() != num_friends {
                bail!("need to pick friends")
            }
        }

        if self.propagated.kitty_theft_policy == KittyTheftPolicy::AllowKittyTheft
            && self.autobid.is_none()
            && !self.finalized
        {
            bail!("must give other players a chance to over-bid and swap cards")
        }

        let landlord_position = bail_unwrap!(self
            .propagated
            .players
            .iter()
            .position(|p| p.id == self.landlord));
        let landlords_team = match self.game_mode {
            GameMode::Tractor => self
                .propagated
                .players
                .iter()
                .enumerate()
                .flat_map(|(idx, p)| {
                    if idx % 2 == landlord_position % 2 {
                        Some(p.id)
                    } else {
                        None
                    }
                })
                .collect(),
            GameMode::FindingFriends { .. } => vec![self.landlord],
        };

        PlayPhase::new(
            self.propagated.clone(),
            self.num_decks,
            self.game_mode.clone(),
            self.hands.clone(),
            self.kitty.clone(),
            self.trump,
            self.landlord,
            self.exchanger,
            landlords_team,
            self.removed_cards.clone(),
            self.decks.clone(),
            self.bids.clone(),
        )
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
        if player != self.exchanger || self.finalized {
            for card in &mut self.kitty {
                *card = Card::Unknown;
            }
        }
        if player != self.landlord {
            if let GameMode::FindingFriends {
                ref mut friends, ..
            } = self.game_mode
            {
                friends.clear();
            }
        }
    }
}
