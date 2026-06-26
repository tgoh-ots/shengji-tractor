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

/// Why an [`advance_bots`] run stopped early to give a human a beat to register
/// what a bot just did. Only ever produced when the caller opts into deferral
/// (`defer_bot_trick_finish = true`); a synchronous (`false`) run never sets it.
///
/// The two variants carry different human-visible pause lengths (the caller maps
/// them to its own configurable durations):
///
/// * [`BotPause::TrickClear`] — a bot is about to finish (clear) a now-complete,
///   bot-won 4-card trick. The loop stops *before* applying the winning bot's
///   [`Action::EndTrick`], leaving the full trick on the table so a human can see
///   it. This is the LONGER pause; the caller resumes via
///   [`finish_deferred_bot_trick`].
/// * [`BotPause::Action`] — a bot just took a single meaningful move (a play, a
///   bid, a reveal, a kitty/landlord/exchange decision). Unlike `TrickClear`, the
///   action has ALREADY been applied; the loop stops *after* it so the new state
///   is published and a SHORT pause lets the human register the one move before
///   the next bot acts. The caller resumes by simply calling `advance_bots`
///   (with deferral) again — [`finish_deferred_bot_trick`] handles that too.
///
/// Bot draws (`Action::DrawCard`) deliberately produce NO pause: the draw phase
/// deals ~25 cards per bot, so per-draw pauses would take minutes. Draws are
/// applied in a burst and paced by the human's own draw cadence; a single
/// `Action` pause at the post-draw bid/kitty/landlord decision is enough.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BotPause {
    /// A bot-won complete trick is about to be cleared; stop before clearing it
    /// (longer, trick-visible pause).
    TrickClear,
    /// A bot just applied a single meaningful move; stop after it (short,
    /// per-action pause).
    Action,
}

/// The outcome of an [`advance_bots`] run, bundling the broadcasts produced with
/// whether (and why) the loop stopped early so a human can follow along.
///
/// When `pause` is `Some(..)`, the run stopped early at a human-visible beat (see
/// [`BotPause`]); the caller publishes the just-produced state, waits the
/// appropriate beat WITHOUT holding the game lock, and then resumes via
/// [`finish_deferred_bot_trick`]. When `None`, the loop ran to a normal stopping
/// point (human's turn, game over, or no progress) exactly as before.
///
/// [`deferred_bot_trick_finish`](AdvanceResult::deferred_bot_trick_finish) is a
/// convenience predicate kept for the trick-clear-specific call sites/tests: it
/// is `true` iff `pause == Some(BotPause::TrickClear)`.
#[derive(Debug, Default)]
pub struct AdvanceResult {
    /// Broadcast messages produced by the bot moves applied during this run.
    pub messages: Vec<(BroadcastMessage, String)>,
    /// `Some(..)` iff the run stopped early at a human-visible beat (and the
    /// caller opted into deferral); the variant selects the pause length.
    pub pause: Option<BotPause>,
}

impl AdvanceResult {
    /// `true` iff the run stopped specifically because a bot won a now-complete
    /// trick and the caller requested deferral of the finish (the LONG,
    /// trick-visible pause). Convenience predicate over [`AdvanceResult::pause`].
    pub fn deferred_bot_trick_finish(&self) -> bool {
        matches!(self.pause, Some(BotPause::TrickClear))
    }

    /// `true` iff the run stopped after applying a single meaningful bot move so
    /// the human can register it (the SHORT, per-action pause). Convenience
    /// predicate over [`AdvanceResult::pause`].
    pub fn paused_after_bot_action(&self) -> bool {
        matches!(self.pause, Some(BotPause::Action))
    }
}

/// Whether a bot [`Action`], when applied by [`advance_bots`] in deferral mode,
/// is a single "meaningful" move that should be followed by a SHORT human-visible
/// pause so a human can register what happened. Returns `false` for the actions
/// that must stay un-paced:
///
/// * `DrawCard` — the deal is ~25 draws/bot; per-draw pauses would make the draw
///   phase minutes long. Draws are bursted and paced by the human's draw cadence
///   instead (a single pause at the post-draw decision is enough).
/// * `EndTrick` — handled separately by the LONGER trick-clear deferral, which
///   stops *before* the action rather than after it.
/// * `ResetGame` and any lobby/no-op action — kept instant for responsiveness.
fn is_paceable_bot_action(action: &Action) -> bool {
    matches!(
        action,
        Action::Bid(..)
            | Action::RevealCard
            | Action::PickUpKitty
            | Action::PutDownKitty
            | Action::MoveCardToKitty(..)
            | Action::MoveCardToHand(..)
            | Action::SetFriends(..)
            | Action::BeginPlay
            | Action::PlayCards(..)
            | Action::PlayCardsWithHint(..)
    )
}

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
///
/// # `defer_bot_trick_finish`
///
/// When `false`, the loop is fully synchronous and unchanged: every bot move
/// (including a bot finishing a completed trick) is applied immediately, so a
/// whole sequence of bot turns runs within this single call. ALL tests, the
/// self-play/ladder/dealing harnesses, and the e2e driver use `false` so a whole
/// hand can be driven without any timing.
///
/// When `true` (the production handler), the loop instead STOPS at the first
/// human-visible beat so a human can follow the game one bot move at a time. Two
/// kinds of beat exist (see [`BotPause`]):
///
/// * **Trick-clear** ([`BotPause::TrickClear`]): the moment it would apply a
///   winning bot's `EndTrick` for a now-complete trick, it stops WITHOUT applying
///   it, leaving the full 4-card trick on the table. This is the LONGER pause.
/// * **Per-action** ([`BotPause::Action`]): after applying a SINGLE meaningful
///   bot move (a play, a bid, a reveal, a kitty/landlord/exchange decision — see
///   [`is_paceable_bot_action`]), it stops so the one move can be published and a
///   SHORT pause lets the human register it before the next bot acts.
///
/// Bot **draws** are exempt: they are applied in a burst with no pause (the deal
/// is ~25 draws/bot; per-draw pauses would take minutes), paced instead by the
/// human's own draw cadence. A reset confirmation is also applied without a
/// pause so the table returns to the lobby promptly.
///
/// In every deferred case the caller publishes the just-produced state, waits the
/// appropriate beat WITHOUT holding the game lock, and resumes via
/// [`finish_deferred_bot_trick`] (which handles both pause kinds). A HUMAN's turn
/// is unaffected in both modes — `next_bot_action` returns `None` there and we
/// wait for the human exactly as before.
pub fn advance_bots(
    game: &mut InteractiveGame,
    logger: &Logger,
    defer_bot_trick_finish: bool,
) -> Result<AdvanceResult, Error> {
    let mut out = vec![];

    for _ in 0..MAX_BOT_ITERATIONS {
        // Determine the bot (if any) that should take the next action, along with
        // the action it should take. We only ever read the redacted view for the
        // acting bot.
        let next = match next_bot_action(game)? {
            Some(next) => next,
            None => {
                return Ok(AdvanceResult {
                    messages: out,
                    pause: None,
                })
            }
        };

        let (bot_id, action) = next;

        // Trick-clear deferral: if the next bot action is finishing a trick the
        // bot won, and the caller opted into deferral, stop WITHOUT applying it so
        // the completed trick stays on the table for a (longer) human-visible
        // beat. The caller resumes via `finish_deferred_bot_trick` after the
        // delay. (`next_bot_action` only ever yields `Action::EndTrick` for a
        // complete, bot-won trick — see its `GameState::Play` arm.)
        if defer_bot_trick_finish && matches!(action, Action::EndTrick) {
            return Ok(AdvanceResult {
                messages: out,
                pause: Some(BotPause::TrickClear),
            });
        }

        // Per-action pacing: when deferring, a single meaningful bot move (play,
        // bid, kitty/landlord/exchange decision) is APPLIED and then we stop so it
        // can be published and a SHORT pause lets the human register it before the
        // next bot acts. Draws and the reset confirmation are NOT paced — they
        // fall through and keep bursting in this same call.
        let pace_after = defer_bot_trick_finish && is_paceable_bot_action(&action);

        let msgs = game.interact(action, bot_id, logger)?;
        out.extend(msgs);

        if pace_after {
            return Ok(AdvanceResult {
                messages: out,
                pause: Some(BotPause::Action),
            });
        }
    }

    error!(
        logger,
        "advance_bots hit the iteration cap; aborting to avoid an infinite loop"
    );
    Ok(AdvanceResult {
        messages: out,
        pause: None,
    })
}

/// Resume after a deferred bot beat (see `advance_bots` with
/// `defer_bot_trick_finish = true`). This is the lock-held, idempotent,
/// re-checking second half of every deferral — both the trick-clear pause
/// ([`BotPause::TrickClear`]) and the per-action pause ([`BotPause::Action`]).
/// The caller invokes it after its (lock-free) delay has elapsed.
///
/// It is deliberately SAFE to call even if the world changed during the delay
/// (a human pressed "Finish trick", a takeback happened, a reset occurred, the
/// game ended, ...). We re-derive what the bot driver wants to do RIGHT NOW:
///
/// * If the very next action is still a bot finishing a trick it won, we apply
///   that single `EndTrick` and then continue driving the bots (again with
///   deferral, so the next beat — trick-clear or per-action — pauses again).
/// * Otherwise (the per-action resume, or the world changed) we simply fall
///   through to a normal deferred `advance_bots`, which applies the next single
///   meaningful bot move and stops, or makes whatever progress is valid, applying
///   no stale `EndTrick`.
///
/// Either way there is no double-finish, no double-apply, and no reliance on the
/// pre-delay snapshot.
pub fn finish_deferred_bot_trick(
    game: &mut InteractiveGame,
    logger: &Logger,
) -> Result<AdvanceResult, Error> {
    // The trick-clear pause stops *before* the `EndTrick`, so to make forward
    // progress we must explicitly apply that one pending bot `EndTrick` (if it is
    // still pending and still bot-won) and only then continue deferring. The
    // per-action pause already applied its move before stopping, so there is
    // nothing to re-apply for it — the fall-through deferred `advance_bots` picks
    // up the next move on its own.
    if let Some((bot_id, action)) = next_bot_action(game)? {
        if matches!(action, Action::EndTrick) {
            let msgs = game.interact(action, bot_id, logger)?;
            let mut result = advance_bots(game, logger, true)?;
            let mut combined = msgs;
            combined.extend(result.messages);
            result.messages = combined;
            return Ok(result);
        }
    }
    // The deferred trick finish is no longer applicable (per-action resume, or the
    // world changed during the delay). Drive the bots normally (still deferring);
    // no stale EndTrick is applied.
    advance_bots(game, logger, true)
}

/// Compute the next `(bot_id, action)` pair to apply, or `None` if no bot needs
/// to act / no progress can be made. This is the place where we decide *whose*
/// turn it is and which phase transition (if any) is required, but the actual
/// move is always selected by the policy from the redacted view.
fn next_bot_action(game: &mut InteractiveGame) -> Result<Option<(PlayerID, Action)>, Error> {
    let state = game.dump_state()?;

    // A reset is a two-player confirmation vote: the first `Action::ResetGame`
    // only records the requester and stays in-phase, and the reset completes
    // only when a SECOND, distinct player also requests it. Bots never request a
    // reset on their own, so in a human+bots room a human's request would hang
    // forever ("Waiting for confirmation..."). If a request is pending, have an
    // eligible bot (any bot seat that is NOT the requester) CONFIRM it. This is
    // strictly a confirmation of an already-pending request — a bot never
    // spontaneously starts a reset — and it takes priority over normal play so
    // the table returns to the lobby promptly. After it completes the game is
    // back in Initialize, where this function returns `None` (no auto-start).
    if let Some(requester) = state.player_requested_reset() {
        if let Some(confirmer) = state
            .propagated()
            .players()
            .iter()
            .map(|p| p.id)
            .find(|id| *id != requester && bot_for(&state, *id).is_some())
        {
            return Ok(Some((confirmer, Action::ResetGame)));
        }
    }

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
                // No bot wanted a strategic bid. The minimal legal-bid FALLBACK
                // below exists ONLY to keep a pure all-bot table from deadlocking
                // when the deck is drawn but nobody has bid. We must NOT fire it
                // when a HUMAN is seated: doing so robs the human of their bidding
                // turn (a bot would force a weak bid and immediately resolve the
                // landlord, racing the deal into play before the human can bid).
                // With a human present we instead PARK here (return None) so the
                // human can bid, reveal the bottom, or pass via the UI. (A
                // pre-selected landlord or a human bid takes a different branch
                // above; this branch is reached only when the human can still act.)
                let any_human_seat = state
                    .propagated()
                    .players()
                    .iter()
                    .any(|pl| bot_for(&state, pl.id).is_none());
                if any_human_seat {
                    return Ok(None);
                }
                // All-bot table: fall back to the lowest-count legal bid from the
                // first able bot so the table never deadlocks.
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
