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

/// A single planned bot step, produced by [`plan_next_bot_action`] from a
/// (cheap-to-clone) snapshot of the game WITHOUT mutating it. This is the
/// off-lock half of the non-blocking bot driver: the (potentially expensive)
/// move selection — the determinized search for a Hard/Expert play — is done
/// here, off the game lock and off the async worker, so the lock is only briefly
/// re-acquired to APPLY the precomputed `action` (see
/// [`apply_planned_bot_action`]).
///
/// `pause` carries the SAME deferral disposition `advance_bots(.., true)` would
/// have produced for this step (see [`AdvanceResult::pause`] / [`BotPause`]):
///
/// * `None` — a burst step (a draw, or a reset confirmation). Apply it and
///   immediately plan the next step with no human-visible pause.
/// * `Some(BotPause::TrickClear)` — the next move is a bot finishing a trick it
///   won. The `action` is the `EndTrick`, but it must NOT be applied until the
///   trick-clear pause has elapsed (the full 4-card trick stays on the table).
/// * `Some(BotPause::Action)` — a single meaningful move (play/bid/kitty/etc.).
///   Apply it, publish, then pause briefly before planning the next step.
#[derive(Debug, Clone)]
pub struct BotStep {
    /// The bot seat that should act.
    pub bot_id: PlayerID,
    /// The concrete, already-selected action to apply.
    pub action: Action,
    /// The deferral disposition for this step (see the struct docs).
    pub pause: Option<BotPause>,
}

/// Cheaply classify whose turn it is and WHAT KIND of bot action is expected
/// next, WITHOUT invoking the (expensive) move-selection policy. This is the
/// re-validation key used by [`apply_planned_bot_action`]: a step planned off the
/// lock is only applied if, under the lock, the world still expects the SAME bot
/// to take the SAME kind of action. If the world moved on during the off-lock
/// compute (a human finished the trick, a takeback/reset happened, the game
/// ended, ...) the responsibility no longer matches and the stale step is
/// dropped — the driver re-plans on the true new state.
///
/// It mirrors the turn/phase decisions of [`next_bot_action`] EXACTLY, but stops
/// short of selecting the concrete move (no `policy::select_action`,
/// `choose_bid`, or search), so it is cheap enough to run under the lock.
fn expected_bot_responsibility(
    game: &InteractiveGame,
    respect_human_bid_window: bool,
) -> Result<Option<(PlayerID, BotResponsibility)>, Error> {
    let state = game.dump_state()?;

    // Reset confirmation takes priority, exactly as in `next_bot_action`.
    if let Some(requester) = state.player_requested_reset() {
        if let Some(confirmer) = state
            .propagated()
            .players()
            .iter()
            .map(|p| p.id)
            .find(|id| *id != requester && bot_for(&state, *id).is_some())
        {
            return Ok(Some((confirmer, BotResponsibility::ResetGame)));
        }
    }

    match &state {
        GameState::Initialize(_) => Ok(None),
        GameState::Draw(p) => {
            if !p.done_drawing() {
                let next_player = p.next_player()?;
                match bot_for(&state, next_player) {
                    Some(_) => Ok(Some((next_player, BotResponsibility::Select))),
                    None => Ok(None),
                }
            } else if p.bid_decided() {
                let responsible = p.next_player()?;
                match bot_for(&state, responsible) {
                    None => Ok(None),
                    Some(_) => {
                        let awaiting_human_done =
                            respect_human_bid_window && !p.all_humans_done_bidding();
                        if awaiting_human_done {
                            Ok(None)
                        } else {
                            Ok(Some((responsible, BotResponsibility::PickUpKitty)))
                        }
                    }
                }
            } else if let Some(landlord) = state.propagated().landlord {
                match bot_for(&state, landlord) {
                    Some(_) => Ok(Some((landlord, BotResponsibility::RevealCard))),
                    None => Ok(None),
                }
            } else {
                // No bid and no landlord yet: a bot may bid. We can't know which
                // bot (or whether any) wants to bid without the policy, so we
                // classify this as a generic "a bot should act" keyed on the seat
                // the planner will pick. To keep the re-check cheap AND precise we
                // re-derive the SAME first-able-bot the planner uses below; but
                // because the bid choice itself needs the policy, the planner
                // resolves the concrete seat. For the re-check we accept ANY bot
                // bid here (the bid window logic is unchanged), so we report the
                // first seated bot as the responsible seat.
                for player in state.propagated().players() {
                    if bot_for(&state, player.id).is_some() {
                        return Ok(Some((player.id, BotResponsibility::Bid)));
                    }
                }
                Ok(None)
            }
        }
        GameState::Exchange(p) => {
            let next_player = p.next_player()?;
            match bot_for(&state, next_player) {
                Some(_) => Ok(Some((next_player, BotResponsibility::Select))),
                None => Ok(None),
            }
        }
        GameState::Play(p) => {
            if p.game_finished() {
                return Ok(None);
            }
            match p.trick().next_player() {
                None => {
                    let next_leader = match p.trick().complete() {
                        Ok(ended) => ended.winner,
                        Err(_) => return Ok(None),
                    };
                    match bot_for(&state, next_leader) {
                        Some(_) => Ok(Some((next_leader, BotResponsibility::EndTrick))),
                        None => Ok(None),
                    }
                }
                Some(next_player) => match bot_for(&state, next_player) {
                    Some(_) => Ok(Some((next_player, BotResponsibility::Select))),
                    None => Ok(None),
                },
            }
        }
    }
}

/// The KIND of move a bot seat is responsible for next, independent of the
/// concrete [`Action`] the policy will choose. Used purely as a cheap
/// re-validation key (see [`expected_bot_responsibility`]); it deliberately does
/// NOT distinguish among the many concrete `Select` actions (draw, play,
/// exchange step, ...), since `interact` re-validates the concrete move's
/// legality anyway and the actor+phase match is what guards against applying a
/// stale step after the world changed.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum BotResponsibility {
    /// A reset confirmation by an eligible bot.
    ResetGame,
    /// The standing-bid bot finalizes the landlord by picking up the kitty.
    PickUpKitty,
    /// The pre-selected-landlord bot reveals the bottom (auto-bid).
    RevealCard,
    /// A bot may place a bid (no bid/landlord yet).
    Bid,
    /// The winning bot finishes a completed, bot-won trick.
    EndTrick,
    /// A policy-selected move (draw, exchange step, or play). The concrete action
    /// is chosen by the policy; `interact` re-validates its legality.
    Select,
}

impl BotResponsibility {
    /// Whether a concrete `action` is consistent with this responsibility kind,
    /// used to confirm a precomputed step still matches the world under the lock.
    fn matches_action(self, action: &Action) -> bool {
        match self {
            BotResponsibility::ResetGame => matches!(action, Action::ResetGame),
            BotResponsibility::PickUpKitty => matches!(action, Action::PickUpKitty),
            BotResponsibility::RevealCard => matches!(action, Action::RevealCard),
            BotResponsibility::Bid => matches!(action, Action::Bid(..)),
            BotResponsibility::EndTrick => matches!(action, Action::EndTrick),
            // A policy-selected move: any of the per-phase select actions. We do
            // not over-constrain here; `interact` enforces concrete legality.
            BotResponsibility::Select => matches!(
                action,
                Action::DrawCard
                    | Action::RevealCard
                    | Action::PickUpKitty
                    | Action::PutDownKitty
                    | Action::MoveCardToKitty(..)
                    | Action::MoveCardToHand(..)
                    | Action::SetFriends(..)
                    | Action::BeginPlay
                    | Action::PlayCards(..)
                    | Action::PlayCardsWithHint(..)
                    | Action::Bid(..)
            ),
        }
    }
}

/// Plan the next bot step from a read-only snapshot, performing the (possibly
/// expensive) move selection — the determinized search for a Hard/Expert play —
/// WITHOUT holding any lock and WITHOUT mutating the game. This is the off-thread
/// half of the non-blocking bot driver: callers run it on a cloned snapshot via
/// `tokio::task::spawn_blocking`, then briefly re-acquire the game lock to apply
/// the returned [`BotStep`] with [`apply_planned_bot_action`].
///
/// The deferral disposition (`pause`) mirrors `advance_bots(.., true)` for ONE
/// step:
///
/// * a bot draw or a reset confirmation → `pause: None` (a burst step);
/// * a bot finishing a trick it won → `Some(BotPause::TrickClear)` (the
///   `EndTrick` must NOT be applied until the trick-clear pause elapses);
/// * any other single meaningful bot move → `Some(BotPause::Action)`.
///
/// Returns `None` when no bot needs to act (a human's turn, game over, parked
/// awaiting "Done bidding", or no progress possible) — identical to
/// `next_bot_action` returning `None`.
///
/// Honesty is preserved exactly as in [`next_bot_action`]: each honest bot's move
/// is selected from its own redacted [`observed_state`] view.
pub fn plan_next_bot_action(
    game: &InteractiveGame,
    defer_bot_trick_finish: bool,
) -> Result<Option<BotStep>, Error> {
    let (bot_id, action) = match next_bot_action(game, defer_bot_trick_finish)? {
        Some(next) => next,
        None => return Ok(None),
    };

    let pause = if defer_bot_trick_finish && matches!(action, Action::EndTrick) {
        // Stop BEFORE applying the winning bot's EndTrick (longer pause).
        Some(BotPause::TrickClear)
    } else if defer_bot_trick_finish && is_paceable_bot_action(&action) {
        // Apply, then a short per-action pause.
        Some(BotPause::Action)
    } else {
        // A burst step (draw / reset): apply with no pause.
        None
    };

    Ok(Some(BotStep {
        bot_id,
        action,
        pause,
    }))
}

/// Apply a [`BotStep`] that was planned off the lock, under the game lock, AFTER
/// cheaply re-validating that the world still expects this step. This is the
/// on-lock half of the non-blocking bot driver.
///
/// The re-validation (via [`expected_bot_responsibility`], which does NO policy
/// work) confirms the SAME bot is still responsible for the SAME kind of action.
/// If the world changed during the off-lock compute (a human finished the trick,
/// a takeback/reset occurred, the game ended, it is no longer this bot's turn,
/// ...), the stale step is DROPPED and `Ok(None)` is returned so the caller
/// re-plans on the true new state. If it still matches, the precomputed `action`
/// is applied through the same validated [`InteractiveGame::interact`] API a
/// human uses — which independently rejects any concrete illegality — and its
/// broadcasts are returned.
///
/// `respect_human_bid_window` MUST match the value passed to
/// [`plan_next_bot_action`] (the production/deferred driver passes `true`).
///
/// Returns `Ok(None)` if the step was DROPPED because the world moved on (the
/// re-check failed), or `Ok(Some(broadcasts))` if it was applied (the broadcast
/// list may be empty — many legal bot moves, e.g. a draw or a kitty step, produce
/// no broadcast yet still mutate the state).
pub fn apply_planned_bot_action(
    game: &mut InteractiveGame,
    step: &BotStep,
    respect_human_bid_window: bool,
    logger: &Logger,
) -> Result<Option<Vec<(BroadcastMessage, String)>>, Error> {
    let expected = expected_bot_responsibility(game, respect_human_bid_window)?;
    let still_valid = match expected {
        Some((bot_id, BotResponsibility::Bid)) => {
            // Opening-bid case: which seat bids is policy-dependent (the planner
            // picks the first bot that WANTS to bid, which may not be the first
            // seated bot reported cheaply here). So we accept ANY seated bot's bid
            // as long as the world still expects an opening bid and the planned
            // actor is itself a bot. `interact` re-validates the concrete bid's
            // legality. We still require the cheap key to BE `Bid` so a phase
            // change since planning correctly invalidates the step.
            let _ = bot_id;
            matches!(step.action, Action::Bid(..))
                && bot_for(&game.dump_state()?, step.bot_id).is_some()
        }
        Some((bot_id, responsibility)) => {
            bot_id == step.bot_id && responsibility.matches_action(&step.action)
        }
        None => false,
    };
    if !still_valid {
        // The world moved on during the off-lock compute; drop the stale step.
        return Ok(None);
    }
    Ok(Some(game.interact(
        step.action.clone(),
        step.bot_id,
        logger,
    )?))
}

/// Whether the next bot responsibility is an UN-PACED "burst" step that is cheap
/// to compute and must be applied without a human-visible pause: a bot draw
/// (`DrawCard`, a Draw-phase `Select`) or a reset confirmation (`ResetGame`).
/// These are the ONLY steps `advance_bots_burst_unpaced` applies under the lock —
/// crucially, NEVER a Play-phase selection, so the (expensive) determinized
/// search never runs while the lock is held. Everything else (bids, kitty/reveal,
/// plays, trick finishes) is paceable and handled off the lock by the
/// snapshot -> spawn_blocking -> apply driver.
fn is_unpaced_burst_responsibility(state: &GameState, kind: BotResponsibility) -> bool {
    match kind {
        BotResponsibility::ResetGame => true,
        // A `Select` is un-paced ONLY in the Draw phase (a card draw). In the
        // Exchange / Play phases a `Select` is a paceable, possibly-expensive move.
        BotResponsibility::Select => matches!(state, GameState::Draw(_)),
        BotResponsibility::PickUpKitty
        | BotResponsibility::RevealCard
        | BotResponsibility::Bid
        | BotResponsibility::EndTrick => false,
    }
}

/// A cheap classification of what the bot driver would do NEXT, computed WITHOUT
/// running any move-selection policy (no determinized search). Used by the
/// non-blocking handler to decide, from a read-only snapshot, whether it needs to
/// touch the write lock at all — avoiding spurious State republishes when no bot
/// has anything to do.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum NextBotWork {
    /// No bot needs to act: a human's turn, parked awaiting "Done bidding", game
    /// over, or no progress possible. The driver should stop (and NOT write).
    None,
    /// The next bot step is an un-paced burst step (a draw or a reset
    /// confirmation). The driver should burst these under one lock.
    Burst,
    /// The next bot step is a paceable / trick-clear beat (a bid, kitty/reveal,
    /// exchange step, play, or trick finish). The driver should compute it off the
    /// lock and apply it with pacing.
    Paceable,
}

/// Cheaply classify the next bot step from a read-only snapshot, WITHOUT running
/// the move-selection policy/search. See [`NextBotWork`]. The production driver
/// passes `respect_human_bid_window = true`.
pub fn classify_next_bot_work(
    game: &InteractiveGame,
    respect_human_bid_window: bool,
) -> Result<NextBotWork, Error> {
    let state = game.dump_state()?;
    match expected_bot_responsibility(game, respect_human_bid_window)? {
        None => Ok(NextBotWork::None),
        Some((_, kind)) => {
            if is_unpaced_burst_responsibility(&state, kind) {
                Ok(NextBotWork::Burst)
            } else {
                Ok(NextBotWork::Paceable)
            }
        }
    }
}

/// Apply, under the lock, every UN-PACED bot "burst" step (bot draws and reset
/// confirmations) until the next bot step is a paceable / trick-clear beat, a
/// human's turn, or there is nothing to do. Returns the combined broadcasts.
///
/// This is the on-lock complement to the off-lock pacing driver: it batches the
/// many cheap bot draws of the deal into a SINGLE locked operation (one State
/// publish instead of ~75), which keeps the draw phase cheap and avoids flooding
/// clients, WITHOUT ever running the expensive Play-phase search under the lock
/// (it stops before any paceable selection). The caller then drives the paceable
/// steps one at a time off the lock.
pub fn advance_bots_burst_unpaced(
    game: &mut InteractiveGame,
    logger: &Logger,
) -> Result<Vec<(BroadcastMessage, String)>, Error> {
    let mut out = vec![];
    for _ in 0..MAX_BOT_ITERATIONS {
        let state = game.dump_state()?;
        let responsibility = match expected_bot_responsibility(game, true)? {
            Some((_, kind)) if is_unpaced_burst_responsibility(&state, kind) => kind,
            // Either nothing to do, or the next step is paceable/trick-clear:
            // stop and let the off-lock driver handle it.
            _ => break,
        };
        // The un-paced steps are cheap to select (a draw or a reset), so computing
        // and applying them under the lock is fine. `next_bot_action` yields the
        // concrete action; for these responsibilities it never runs the search.
        let (bot_id, action) = match next_bot_action(game, true)? {
            Some(next) => next,
            None => break,
        };
        // Defensive: only apply if the concrete action matches the un-paced kind
        // we expected (it always should). This guards against an unexpected
        // expensive selection ever slipping through.
        if !responsibility.matches_action(&action) {
            break;
        }
        let msgs = game.interact(action, bot_id, logger)?;
        out.extend(msgs);
    }
    Ok(out)
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
        let next = match next_bot_action(game, defer_bot_trick_finish)? {
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
    if let Some((bot_id, action)) = next_bot_action(game, true)? {
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

/// Whether the deferred bot driver is currently PARKED waiting for the seated
/// humans to finish bidding: the deck is fully drawn, a BOT holds the standing
/// (decided) bid, the bot would otherwise pick up the kitty, but not every human
/// seat has clicked "Done bidding" yet (so the window-respecting driver returns
/// `None`).
///
/// There is NO timer: the bot simply waits until every human marks done (or
/// outbids — which re-opens the window and clears the votes). When all humans are
/// done, the next human action (the last "Done bidding" click) re-runs the bot
/// driver, which then finalizes the standing bot itself. This predicate is kept
/// purely as an observability/diagnostic hook for the handler; it never triggers
/// any forced finalization.
pub fn is_parked_awaiting_human_done_bidding(game: &InteractiveGame) -> Result<bool, Error> {
    // If the window-respecting driver wants to make a move, it is NOT parked.
    if next_bot_action(game, true)?.is_some() {
        return Ok(false);
    }
    // It parked. Distinguish the await-human-done park from every other park
    // (human's own turn / nothing to do): only the former has the window-IGNORING
    // driver wanting a bot `PickUpKitty` in a fully-drawn Draw phase where not all
    // humans are done yet.
    match next_bot_action(game, false)? {
        Some((_, Action::PickUpKitty)) => Ok(matches!(
            game.dump_state()?,
            GameState::Draw(p)
                if p.done_drawing() && p.bid_decided() && !p.all_humans_done_bidding()
        )),
        _ => Ok(false),
    }
}

/// Compute the next `(bot_id, action)` pair to apply, or `None` if no bot needs
/// to act / no progress can be made. This is the place where we decide *whose*
/// turn it is and which phase transition (if any) is required, but the actual
/// move is always selected by the policy from the redacted view.
/// `respect_human_bid_window`: when `true` (the production/deferred driver), the
/// bot will NOT finalize the landlord (pick up the kitty) for a bot that holds the
/// standing bid while a seated human could still legally outbid it — it PARKS
/// instead, leaving the human their counter-bid window. When `false` (the
/// synchronous test/harness/e2e drivers), it keeps the original behavior and
/// finalizes immediately, since no human is watching in real time and a
/// non-interactive driver must run to completion.
fn next_bot_action(
    game: &InteractiveGame,
    respect_human_bid_window: bool,
) -> Result<Option<(PlayerID, Action)>, Error> {
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
                    // The human is the standing (winning) bidder: it is THEIR call
                    // to pick up the kitty, so park and wait for them.
                    None => Ok(None),
                    Some(_) => {
                        // A BOT holds the standing bid. Bidding is NOT turn-based:
                        // while the deck is drawn, the bottom is unrevealed, and the
                        // kitty has not been picked up, ANY seated human may still
                        // legally OUTBID (a higher count, a joker, a higher suit,
                        // ...). If we let the bot immediately pick up the kitty here
                        // we lock it in as landlord and race straight into play,
                        // robbing the human of the counter-bid window they're
                        // entitled to (the exact production bug: "an Easy Bot bid
                        // 2♣, but I didn't have a chance to counter-bid before the
                        // game started").
                        //
                        // So, when we are asked to RESPECT the human's bidding
                        // window (`respect_human_bid_window`, set only on the
                        // production/deferred driver path), PARK (return None) until
                        // EVERY human seat has explicitly clicked "Done bidding"
                        // (`Action::MarkBiddingDone`). Each human's click marks their
                        // seat done; a NEW bid (anyone bidding) re-opens the window by
                        // clearing every "done" flag, so the humans re-confirm against
                        // the new standing bid. Once `all_humans_done_bidding` is true
                        // (which, crucially, is TRIVIALLY true for an all-bot table —
                        // zero human seats — so there is no deadlock), the bot
                        // proceeds and picks up the kitty.
                        //
                        // This replaces the old time-based counter-bid grace: there is
                        // no timer; finalization is gated purely on the explicit
                        // per-human "done bidding" votes.
                        //
                        // On the SYNCHRONOUS driver path (`respect_human_bid_window =
                        // false`: every test, the self-play / ladder / dealing
                        // harnesses, the e2e driver) there is no human watching in
                        // real time, so we keep the original behavior and let the bot
                        // pick up immediately — a non-interactive driver must run to
                        // completion without waiting on a human that will never click.
                        let awaiting_human_done =
                            respect_human_bid_window && !p.all_humans_done_bidding();
                        if awaiting_human_done {
                            Ok(None)
                        } else {
                            Ok(Some((responsible, Action::PickUpKitty)))
                        }
                    }
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
