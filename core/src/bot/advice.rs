//! Human-facing play advice — the engine behind the UI's "suggest a play" (✨)
//! button.
//!
//! The advice a human gets is EXACTLY the move the apex HONEST tier,
//! [`BotDifficulty::Grandmaster`], would make in their seat: the same
//! calculation-driven determinized search, which uses the Enoch playbook only to
//! PROPOSE candidates and then commits to whatever the full-hand rollouts value
//! highest. The one deliberate difference is the time budget — a human waiting
//! on a click gets the base per-decision budget rather than Grandmaster's 3×
//! (see [`policy::grandmaster_play`]).
//!
//! Previously the button answered only the RULES question — "what am I *allowed*
//! to play?" — by taking the first structural match out of
//! `TrickFormat::decomposition`. That is legal but frequently terrible (it
//! ignores points, boss cards and who is winning the trick), and it could not
//! advise a LEAD at all, since a lead has no trick format to decompose.
//!
//! # Honesty invariant
//!
//! [`suggest_play`] NEVER calls [`InteractiveGame::dump_state`]. It derives its
//! advice solely from [`InteractiveGame::dump_state_for_player`] — the same
//! redacted view the requesting human already has on their own screen (every
//! other seat's cards are [`Card::Unknown`], the kitty hidden). Asking for
//! advice therefore cannot tell a player anything they could not have worked out
//! themselves, and the button can never become a back door to the `Omniscient`
//! cheat. This parallels [`crate::bot::observed_state`], except that here there
//! is no perfect-information branch at all: the only view this module can
//! obtain is the redacted one.

use anyhow::Error;

use shengji_mechanics::types::{Card, PlayerID};

use crate::bot::policy;
use crate::game_state::GameState;
use crate::interactive::InteractiveGame;

/// Suggest the cards `player` should play right now, as chosen by the
/// Grandmaster policy from `player`'s OWN redacted view.
///
/// Returns `Ok(None)` — meaning "no advice available", not an error — when the
/// game is not in the play phase, the hand is over, or it is not `player`'s turn
/// to play. Returns `Ok(Some(cards))` otherwise; the cards are always a legal
/// play (re-validated against the rules engine before being returned, so the
/// suggestion the UI pre-selects is never one the server would reject).
///
/// Asking twice in the same position will usually, but not always, give the same
/// answer: the policy seeds its RNG from the observable position, but the search
/// underneath it is time-boxed, so how many simulations fit in the budget — and
/// therefore which move wins — can vary with machine load.
pub fn suggest_play(game: &InteractiveGame, player: PlayerID) -> Result<Option<Vec<Card>>, Error> {
    // HONESTY: the requester's own redacted view — never `dump_state()`.
    let view = game.dump_state_for_player(player)?;
    Ok(suggest_play_from_view(&view, player))
}

/// Inner half of [`suggest_play`], taking the ALREADY-REDACTED per-player view.
/// Kept separate so tests can feed a view directly and assert that the advice
/// only ever mentions the advisee's own cards.
fn suggest_play_from_view(view: &GameState, me: PlayerID) -> Option<Vec<Card>> {
    let p = match view {
        GameState::Play(p) => p,
        // Nothing to advise in the lobby / draw / exchange phases.
        _ => return None,
    };
    if p.game_finished() {
        return None;
    }
    // Only the seat that is actually on turn can be advised: the policy (like a
    // bot) only produces a move for the player it is waiting on, and a
    // suggestion for someone else's turn would be meaningless to select.
    if p.trick().next_player() != Some(me) {
        return None;
    }

    let cards = policy::grandmaster_play(p, me);
    if cards.is_empty() {
        return None;
    }
    // Belt-and-braces: the policy always returns a legal play (it falls back to
    // the always-legal dumb policy), but this is user-facing, so we refuse to
    // pre-select anything the rules engine would reject.
    p.can_play_cards(me, &cards).ok()?;
    Some(cards)
}
