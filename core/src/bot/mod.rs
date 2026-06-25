use anyhow::Error;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use slog::{error, Logger};

use shengji_mechanics::types::PlayerID;

use crate::game_state::GameState;
use crate::interactive::{Action, BroadcastMessage, InteractiveGame};

pub mod determinize;
pub mod expert;
pub mod heuristics;
pub mod policy;
pub mod search;

#[cfg(test)]
mod tests;

/// The difficulty of a bot player. The four tiers form a strength ladder
/// `Easy < Hard <= Expert < Omniscient`:
///
/// * `Easy` — the bare heuristic backbone played noisily (frequent blunders, hot
///   softmax, no card memory or search). Feels like a casual human.
/// * `Hard` — the same heuristic PLUS a time-boxed determinized (ISMCTS-style)
///   search over sampled worlds. Honest.
/// * `Expert` — a learned neural net (a small MLP trained by behavioral cloning /
///   distillation of the Omniscient teacher's choices) scores each legal
///   candidate from HONEST features only. It approximates perfect-info play from
///   the honest observation. If the model fails to load/run it falls back to the
///   `Hard` heuristic, so Expert is never illegal/None. Honest.
/// * `Omniscient` — a DELIBERATE, clearly-labeled, opt-in CHEATING tier that
///   plays with PERFECT INFORMATION (it is allowed to see every opponent's
///   hand). It exists for testing and for an "impossible" practice opponent; it
///   must be chosen explicitly in the lobby and is surfaced with a cheater badge
///   in the UI.
///
/// The three honest tiers (`Easy`/`Hard`/`Expert`) never receive anything but
/// their own redacted, per-player view — see [`observed_state`], which is the
/// single, centralized place where the perfect-information bypass is gated.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
pub enum BotDifficulty {
    Easy,
    #[default]
    Hard,
    /// Learned-net tier: a small MLP distilled from the Omniscient teacher,
    /// scoring legal candidates from HONEST features only. Falls back to the
    /// `Hard` heuristic if the model can't run. See [`crate::bot::expert`].
    Expert,
    /// CHEATER tier: sees all opponents' hands and plays with perfect
    /// information. The ONLY difficulty for which [`observed_state`] returns the
    /// true, unredacted state.
    Omniscient,
}

impl BotDifficulty {
    pub fn as_str(self) -> &'static str {
        match self {
            BotDifficulty::Easy => "Easy",
            BotDifficulty::Hard => "Hard",
            BotDifficulty::Expert => "Expert",
            BotDifficulty::Omniscient => "Omniscient",
        }
    }

    /// Whether this tier is allowed to cheat by observing the full
    /// (unredacted) game state. ONLY `Omniscient` may. This is the single
    /// predicate that the honesty bypass in [`observed_state`] consults; the
    /// honest tiers can never flip it.
    fn sees_perfect_information(self) -> bool {
        matches!(self, BotDifficulty::Omniscient)
    }
}

/// Safety cap on the number of bot moves that can be applied in a single call to
/// [`advance_bots`]; this prevents an unexpected state from spinning forever.
const MAX_BOT_ITERATIONS: usize = 5000;

/// Drive any bot players that need to act, applying their moves through the same
/// validated [`InteractiveGame::interact`] API a human uses.
///
/// The honesty invariant is preserved by computing each HONEST bot's move from
/// the per-player redacted view via [`observed_state`]: the policy never sees the
/// unredacted game state, so an Easy/Hard/Expert bot cannot observe information a
/// human in its seat couldn't. The ONLY exception is the explicitly opt-in
/// `Omniscient` CHEATER tier, for which [`observed_state`] (the single,
/// centralized perfect-information bypass) returns the true full state.
///
/// This loops while the next actor (or a required phase transition such as
/// drawing, finishing a full trick, or the draw -> exchange reveal) belongs to a
/// bot, and stops as soon as the next actor is a human or no further progress is
/// possible. It deliberately does NOT auto-start a new game; that remains a human
/// lobby choice.
pub fn advance_bots(
    game: &mut InteractiveGame,
    logger: &Logger,
) -> Result<Vec<(BroadcastMessage, String)>, Error> {
    let mut out = vec![];

    for _ in 0..MAX_BOT_ITERATIONS {
        // Determine the bot (if any) that should take the next action, along with
        // the action it should take. We only ever read the redacted view for the
        // acting bot.
        let next = match next_bot_action(game)? {
            Some(next) => next,
            None => return Ok(out),
        };

        let (bot_id, action) = next;
        let msgs = game.interact(action, bot_id, logger)?;
        out.extend(msgs);
    }

    error!(
        logger,
        "advance_bots hit the iteration cap; aborting to avoid an infinite loop"
    );
    Ok(out)
}

/// Compute the next `(bot_id, action)` pair to apply, or `None` if no bot needs
/// to act / no progress can be made. This is the place where we decide *whose*
/// turn it is and which phase transition (if any) is required, but the actual
/// move is always selected by the policy from the redacted view.
fn next_bot_action(game: &mut InteractiveGame) -> Result<Option<(PlayerID, Action)>, Error> {
    let state = game.dump_state()?;

    match &state {
        GameState::Initialize(_) => Ok(None),
        GameState::Draw(p) => {
            if !p.done_drawing() {
                // The player whose turn it is to draw must be a bot for us to act.
                let next_player = p.next_player()?;
                match bot_for(&state, next_player) {
                    Some(difficulty) => {
                        let view = observed_state(game, next_player, difficulty)?;
                        Ok(policy::select_action(&view, next_player, difficulty)?
                            .map(|action| (next_player, action)))
                    }
                    None => Ok(None),
                }
            } else if p.bid_decided() {
                // A bid exists (whether a real bid or an auto-bid from revealing the
                // bottom). The winning bidder advances into the exchange phase. We
                // only do this automatically if that player is a bot; otherwise we
                // stop and let the human pick up the kitty.
                let responsible = p.next_player()?;
                match bot_for(&state, responsible) {
                    Some(_) => Ok(Some((responsible, Action::PickUpKitty))),
                    None => Ok(None),
                }
            } else if let Some(landlord) = state.propagated().landlord {
                // No bid yet, but a landlord has been pre-selected: mirror
                // simulate_play's no-bid path by revealing the bottom (auto-bid) if
                // the landlord is a bot.
                match bot_for(&state, landlord) {
                    Some(_) => Ok(Some((landlord, Action::RevealCard))),
                    None => Ok(None),
                }
            } else {
                // No bid and no landlord yet. Let bots bid by strength: each bot
                // evaluates its hand and bids only if it has a genuinely strong
                // trump holding (so we don't overbid weak hands). To guarantee
                // the table always makes progress, if NO bot wants a strategic
                // bid we fall back to the lowest-count legal bid from the first
                // able bot. If no bot can bid at all, stop and let humans act.
                for player in state.propagated().players() {
                    if bot_for(&state, player.id).is_some() {
                        if let Some(bid) = policy::choose_bid(p, player.id) {
                            return Ok(Some((player.id, Action::Bid(bid.card, bid.count))));
                        }
                    }
                }
                // Fallback so an all-bot table never deadlocks: minimal legal bid.
                for player in state.propagated().players() {
                    if bot_for(&state, player.id).is_some() {
                        if let Some(bid) =
                            p.valid_bids(player.id)?.into_iter().min_by_key(|b| b.count)
                        {
                            return Ok(Some((player.id, Action::Bid(bid.card, bid.count))));
                        }
                    }
                }
                Ok(None)
            }
        }
        GameState::Exchange(p) => {
            let next_player = p.next_player()?;
            match bot_for(&state, next_player) {
                Some(difficulty) => {
                    let view = observed_state(game, next_player, difficulty)?;
                    Ok(policy::select_action(&view, next_player, difficulty)?
                        .map(|action| (next_player, action)))
                }
                None => Ok(None),
            }
        }
        GameState::Play(p) => {
            if p.game_finished() {
                // The hand is over; we do not auto-start a new game.
                return Ok(None);
            }
            match p.trick().next_player() {
                None => {
                    // The trick is complete (the play queue is empty) and ready to
                    // be finished. Finishing isn't tied to a single "next player"
                    // (the actor id is unused by finish_trick), but the winner leads
                    // the next trick, so use the winner to decide whether a bot
                    // should auto-finish.
                    let next_leader = match p.trick().complete() {
                        Ok(ended) => ended.winner,
                        // If we can't determine the winner yet, don't act.
                        Err(_) => return Ok(None),
                    };
                    match bot_for(&state, next_leader) {
                        Some(_) => Ok(Some((next_leader, Action::EndTrick))),
                        None => Ok(None),
                    }
                }
                Some(next_player) => match bot_for(&state, next_player) {
                    Some(difficulty) => {
                        let view = observed_state(game, next_player, difficulty)?;
                        Ok(policy::select_action(&view, next_player, difficulty)?
                            .map(|action| (next_player, action)))
                    }
                    None => Ok(None),
                },
            }
        }
    }
}

/// Compute the game state a bot is allowed to observe before choosing a move.
///
/// # The sole, intentional perfect-information (honesty) bypass
///
/// This is the ONE place in the codebase where a bot may be handed the true,
/// unredacted game state instead of its own redacted, per-player view. The
/// decision is gated entirely on [`BotDifficulty::sees_perfect_information`]:
///
/// * For the HONEST tiers (`Easy`/`Hard`/`Expert`) we return
///   [`InteractiveGame::dump_state_for_player`], i.e. `GameState::for_player`,
///   in which every OTHER seat's cards are [`Card::Unknown`](shengji_mechanics::types::Card::Unknown)
///   and the kitty is hidden. These tiers therefore structurally cannot read
///   information a human in their seat couldn't — the cheat-boundary tests
///   (`test_bot_view_hides_other_seats_cards`) assert this and must keep
///   passing. The honest tiers NEVER reach the `dump_state()` branch below.
///
/// * For the CHEATER tier (`Omniscient`) ONLY, we return
///   [`InteractiveGame::dump_state`], the TRUE full state, including every
///   opponent's real cards. This is a deliberate, opt-in cheat used to build an
///   "impossible" practice/test opponent that plays with perfect information.
///
/// Centralizing the bypass here (rather than at the three call sites in
/// [`next_bot_action`]) means there is exactly one branch to audit, and adding a
/// future honest tier cannot accidentally leak hidden cards: it would have to
/// opt in via `sees_perfect_information`, which only `Omniscient` does.
fn observed_state(
    game: &InteractiveGame,
    player: PlayerID,
    difficulty: BotDifficulty,
) -> Result<GameState, Error> {
    if difficulty.sees_perfect_information() {
        // CHEAT (Omniscient only): the real, unredacted state with every
        // opponent's hand visible. This is the single intentional honesty
        // bypass; honest tiers never take this branch.
        game.dump_state()
    } else {
        // HONEST (Easy/Hard/Expert): the redacted per-player view; opponents'
        // cards are Card::Unknown and the kitty is hidden.
        game.dump_state_for_player(player)
    }
}

/// Look up the bot difficulty for the given player id, if it is a registered bot.
fn bot_for(state: &GameState, id: PlayerID) -> Option<BotDifficulty> {
    state.propagated().is_bot(id)
}
