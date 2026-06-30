//! Strictly bounded perfect-information solver for tiny Tractor endgames.
//!
//! This is an oracle, not an information leak: callers must provide a fully
//! materialized PlayPhase. Honest search uses it only inside a sampled world;
//! Omniscient/data-label callers may use it directly. Every action is accepted by
//! the mechanics engine before recursion, and any deadline/node-budget overrun
//! aborts the whole result rather than returning a partially searched minimax.

use std::time::{Duration, Instant};

use shengji_mechanics::types::{Card, PlayerID};

use crate::bot::expert;
use crate::game_state::play_phase::PlayPhase;
use crate::settings::GameMode;

const HARD_MAX_CARDS: usize = 12;
const HARD_MAX_NODES: usize = 1_000_000;

/// Hard limits for an exact solve. A max_cards value of zero disables the oracle.
#[derive(Clone, Copy, Debug, Default)]
pub struct ExactEndgameConfig {
    pub max_cards: usize,
    pub max_nodes: usize,
}

impl ExactEndgameConfig {
    pub fn bounded(self) -> Self {
        Self {
            max_cards: self.max_cards.min(HARD_MAX_CARDS),
            max_nodes: self.max_nodes.min(HARD_MAX_NODES),
        }
    }

    pub fn enabled(self) -> bool {
        self.max_cards > 0 && self.max_nodes > 0
    }
}

/// Completed exact root result. Value is normalized signed level utility from
/// the observer team perspective, matching the action-Q/state-V target contract.
#[derive(Clone, Debug)]
pub struct ExactEndgameResult {
    pub cards: Vec<Card>,
    pub value: f64,
    pub nodes: usize,
}

/// Public clean-label/oracle entry point. Returns None when the position is not
/// a fully materialized Tractor endgame, it is not the observer decision, or the
/// strict time/node/card bound cannot complete the whole tree.
pub fn solve_small_endgame_exact(
    play: &PlayPhase,
    observer: PlayerID,
    max_cards: usize,
    max_nodes: usize,
    time_budget: Duration,
) -> Option<ExactEndgameResult> {
    let config = ExactEndgameConfig {
        max_cards,
        max_nodes,
    }
    .bounded();
    if !config.enabled() || play.trick().next_player() != Some(observer) || time_budget.is_zero() {
        return None;
    }
    solve_root_with_deadline(play, observer, config, Some(Instant::now() + time_budget))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum ExactValueAttempt {
    Ineligible,
    Solved(f64),
    Aborted,
}

pub(crate) fn solve_value_if_eligible(
    play: &PlayPhase,
    observer: PlayerID,
    config: ExactEndgameConfig,
    deadline: Option<Instant>,
) -> ExactValueAttempt {
    let config = config.bounded();
    if !eligible_for_exact(play, config) {
        return ExactValueAttempt::Ineligible;
    }
    let mut solver = Solver::new(observer, config.max_nodes, deadline);
    match solver.solve(play) {
        Ok(result) => ExactValueAttempt::Solved(result.value),
        Err(Aborted) => ExactValueAttempt::Aborted,
    }
}

pub(crate) fn solve_root_with_deadline(
    play: &PlayPhase,
    observer: PlayerID,
    config: ExactEndgameConfig,
    deadline: Option<Instant>,
) -> Option<ExactEndgameResult> {
    let config = config.bounded();
    if !eligible_for_exact(play, config) || play.trick().next_player() != Some(observer) {
        return None;
    }
    let mut solver = Solver::new(observer, config.max_nodes, deadline);
    let result = solver.solve(play).ok()?;
    Some(ExactEndgameResult {
        cards: result.best_action?,
        value: result.value,
        nodes: solver.nodes,
    })
}

pub(crate) fn eligible_for_exact(play: &PlayPhase, config: ExactEndgameConfig) -> bool {
    if !config.enabled() || !matches!(play.game_mode(), GameMode::Tractor) {
        return false;
    }
    let mut cards = 0usize;
    for player in play.propagated().players() {
        let Ok(hand) = play.hands().get(player.id) else {
            return false;
        };
        if hand.contains_key(&Card::Unknown) {
            return false;
        }
        cards = cards.saturating_add(hand.values().copied().sum::<usize>());
    }
    let (kitty, removed) = play.piles_for_determinization();
    if kitty.contains(&Card::Unknown) || removed.contains(&Card::Unknown) {
        return false;
    }
    cards <= config.max_cards
}

#[derive(Clone, Debug)]
struct NodeResult {
    value: f64,
    best_action: Option<Vec<Card>>,
}

#[derive(Clone, Copy, Debug)]
struct Aborted;

struct Solver {
    observer: PlayerID,
    max_nodes: usize,
    deadline: Option<Instant>,
    nodes: usize,
}

impl Solver {
    fn new(observer: PlayerID, max_nodes: usize, deadline: Option<Instant>) -> Self {
        Self {
            observer,
            max_nodes,
            deadline,
            nodes: 0,
        }
    }

    fn check_budget(&mut self) -> Result<(), Aborted> {
        if self.nodes >= self.max_nodes || self.deadline_exceeded() {
            return Err(Aborted);
        }
        self.nodes += 1;
        Ok(())
    }

    fn solve(&mut self, play: &PlayPhase) -> Result<NodeResult, Aborted> {
        self.solve_bounded(play, f64::NEG_INFINITY, f64::INFINITY)
    }

    fn solve_bounded(
        &mut self,
        play: &PlayPhase,
        mut alpha: f64,
        mut beta: f64,
    ) -> Result<NodeResult, Aborted> {
        self.check_budget()?;
        if play.game_finished() {
            return terminal_level_utility(play, self.observer)
                .map(|value| NodeResult {
                    value,
                    best_action: None,
                })
                .ok_or(Aborted);
        }

        let Some(actor) = play.trick().next_player() else {
            let mut next = play.clone();
            next.finish_trick().map_err(|_| Aborted)?;
            return self.solve_bounded(&next, alpha, beta);
        };
        let actions = self.legal_actions(play, actor)?;
        if actions.is_empty() {
            return Err(Aborted);
        }

        let observer_landlord = play.landlords_team().contains(&self.observer);
        let actor_landlord = play.landlords_team().contains(&actor);
        let maximize = observer_landlord == actor_landlord;
        let mut best: Option<NodeResult> = None;
        for action in actions {
            if self.deadline_exceeded() {
                return Err(Aborted);
            }
            let mut child = play.clone();
            child.play_cards(actor, &action).map_err(|_| Aborted)?;
            let value = self.solve_bounded(&child, alpha, beta)?.value;
            let improves = best.as_ref().is_none_or(|current| {
                if maximize {
                    value > current.value
                } else {
                    value < current.value
                }
            });
            if improves {
                best = Some(NodeResult {
                    value,
                    best_action: Some(action),
                });
            }
            if maximize {
                alpha = alpha.max(value);
            } else {
                beta = beta.min(value);
            }
            if alpha >= beta {
                break;
            }
        }
        best.ok_or(Aborted)
    }

    fn deadline_exceeded(&self) -> bool {
        self.deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
    }

    fn legal_actions(&self, play: &PlayPhase, actor: PlayerID) -> Result<Vec<Vec<Card>>, Aborted> {
        let hand = play.hands().get(actor).map_err(|_| Aborted)?;
        let target_size = play.trick().trick_format().map(|format| format.size());
        let candidates = multiset_combinations(hand, target_size);
        let mut legal = Vec::new();
        for cards in candidates {
            // Candidate validation can dominate a large lead node. Check between
            // every mechanics probe so action generation observes the same hard
            // wall-clock deadline as recursive minimax.
            if self.deadline_exceeded() {
                return Err(Aborted);
            }
            let mut probe = play.clone();
            if probe.play_cards(actor, &cards).is_ok() {
                legal.push(cards);
            }
        }
        Ok(legal)
    }
}

fn terminal_level_utility(play: &PlayPhase, observer: PlayerID) -> Option<f64> {
    let score = play.current_game_score().ok()?;
    let observer_landlord = play.landlords_team().contains(&observer);
    let observer_won = observer_landlord == score.landlord_won;
    let levels = if score.landlord_won {
        score.landlord_delta
    } else {
        score.non_landlord_delta
    };
    let magnitude = ((1 + levels) as f64 / expert::LEVEL_UTILITY_NORM as f64).min(1.0);
    Some(if observer_won { magnitude } else { -magnitude })
}

fn multiset_combinations(
    hand: &std::collections::HashMap<Card, usize>,
    target_size: Option<usize>,
) -> Vec<Vec<Card>> {
    let mut entries: Vec<(Card, usize)> = hand
        .iter()
        .filter(|(card, count)| **card != Card::Unknown && **count > 0)
        .map(|(card, count)| (*card, *count))
        .collect();
    entries.sort_by_key(|(card, _)| card.as_char());
    let hand_size = entries.iter().map(|(_, count)| *count).sum::<usize>();
    let target_size = target_size.map(|size| size.min(hand_size));
    let mut output = Vec::new();
    let mut current = Vec::new();
    enumerate_multisets(&entries, 0, target_size, &mut current, &mut output);
    output.sort_by(|a, b| {
        a.len().cmp(&b.len()).then_with(|| {
            a.iter()
                .map(|card| card.as_char())
                .cmp(b.iter().map(|card| card.as_char()))
        })
    });
    output
}

fn enumerate_multisets(
    entries: &[(Card, usize)],
    index: usize,
    target_size: Option<usize>,
    current: &mut Vec<Card>,
    output: &mut Vec<Vec<Card>>,
) {
    if index == entries.len() {
        if !current.is_empty() && target_size.is_none_or(|size| current.len() == size) {
            output.push(current.clone());
        }
        return;
    }
    let (card, count) = entries[index];
    for copies in 0..=count {
        if target_size.is_some_and(|size| current.len() + copies > size) {
            break;
        }
        current.extend(std::iter::repeat_n(card, copies));
        enumerate_multisets(entries, index + 1, target_size, current, output);
        current.truncate(current.len() - copies);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shengji_mechanics::deck::Deck;
    use shengji_mechanics::hands::Hands;
    use shengji_mechanics::player::Player;
    use shengji_mechanics::types::{Number, Suit, Trump};

    use crate::settings::PropagatedState;

    fn card(number: Number) -> Card {
        Card::Suited {
            suit: Suit::Spades,
            number,
        }
    }

    fn one_trick_position() -> (PlayPhase, Vec<PlayerID>) {
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
        let mut hands = Hands::new(ids.iter().copied());
        hands.set_trump(trump);
        for (id, number) in ids
            .iter()
            .zip([Number::Ace, Number::King, Number::Queen, Number::Jack])
        {
            hands.add(*id, [card(number)]).unwrap();
        }
        let play = PlayPhase::new(
            propagated,
            1,
            GameMode::Tractor,
            hands,
            vec![],
            trump,
            ids[0],
            ids[0],
            vec![ids[0], ids[2]],
            vec![],
            vec![Deck::default()],
            vec![],
        )
        .unwrap();
        (play, ids)
    }

    #[test]
    fn multiset_enumeration_is_complete_and_unique() {
        let mut hand = std::collections::HashMap::new();
        hand.insert(card(Number::Ace), 2);
        hand.insert(card(Number::King), 1);
        assert_eq!(multiset_combinations(&hand, None).len(), 5);
        assert_eq!(multiset_combinations(&hand, Some(2)).len(), 2);
    }

    #[test]
    fn exact_solver_finishes_a_materialized_one_trick_hand() {
        let (play, ids) = one_trick_position();
        let result = solve_small_endgame_exact(&play, ids[0], 4, 100, Duration::from_secs(1))
            .expect("tiny materialized hand should solve exactly");
        assert_eq!(result.cards, vec![card(Number::Ace)]);
        assert!(result.value > 0.0);
        assert!(result.nodes <= 100);
    }

    #[test]
    fn exact_solver_rejects_hidden_or_underbudgeted_positions() {
        let (play, ids) = one_trick_position();
        assert!(solve_small_endgame_exact(&play, ids[0], 4, 1, Duration::from_secs(1)).is_none());
        let mut hidden = play;
        hidden.destructively_redact_for_player(ids[0]);
        assert!(
            solve_small_endgame_exact(&hidden, ids[0], 4, 100, Duration::from_secs(1)).is_none()
        );
    }
}
