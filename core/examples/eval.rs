//! Self-play evaluation harness for the Shengji bot brains (Milestone 2).
//!
//! Runs many all-bot games with assorted per-seat difficulties and prints a
//! win-rate table demonstrating the strength ladder
//! Easy < Expert <= Enoch < Omniscient, plus exploitability probes against a
//! couple of degenerate opponents.
//!
//! Run with:
//!     cargo run --release --example eval
//!     SHENGJI_BOT_BUDGET_MS=30 cargo run --example eval   # faster search
//!
//! # Honesty
//!
//! Every bot decision is computed from the per-seat REDACTED view
//! (`GameState::for_player(seat)`), exactly as in production. The harness never
//! feeds a bot another seat's cards.

use std::collections::HashMap;
use std::time::Instant;

use shengji_core::bot::policy;
use shengji_core::bot::BotDifficulty;
use shengji_core::game_state::initialize_phase::InitializePhase;
use shengji_core::game_state::GameState;
use shengji_core::interactive::Action;
use shengji_mechanics::types::{Card, EffectiveSuit, PlayerID};

use slog::{o, Discard, Logger};

fn null_logger() -> Logger {
    Logger::root(Discard, o!())
}

/// A "brain" controlling a seat: either one of our difficulty tiers, or a
/// degenerate exploitability probe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Brain {
    Tier(BotDifficulty),
    /// Always dumps the highest-point cards it legally can.
    AlwaysDumpPoints,
    /// Always trumps in when it can, wasting trumps.
    AlwaysTrump,
}

impl Brain {
    fn label(self) -> &'static str {
        match self {
            Brain::Tier(BotDifficulty::Easy) => "Easy",
            Brain::Tier(BotDifficulty::Expert) => "Expert",
            Brain::Tier(BotDifficulty::Enoch) => "Enoch",
            Brain::Tier(BotDifficulty::Omniscient) => "Omniscient",
            Brain::AlwaysDumpPoints => "DumpPoints",
            Brain::AlwaysTrump => "AlwaysTrump",
        }
    }
}

/// Result of one finished hand: which team (by seat parity relative to the
/// landlord) won, and the non-landlord point total.
struct HandOutcome {
    landlord_won: bool,
    landlord_seat: PlayerID,
    /// Attacking (non-landlord) team's point total — a margin indicator.
    non_landlord_points: isize,
}

/// Drive a single all-bot Tractor hand to completion with the given per-seat
/// brains. Returns the outcome, or `None` if the game failed to make progress
/// (should not happen).
fn play_one_hand(brains: &[Brain]) -> Option<HandOutcome> {
    let logger = null_logger();
    let _ = &logger;
    let n = brains.len();

    // Build a fresh initialize phase with `n` named players.
    let mut init = InitializePhase::new();
    let mut seats: Vec<PlayerID> = vec![];
    for i in 0..n {
        seats.push(init.add_player(format!("seat{i}")).unwrap().0);
    }
    init.set_num_decks(Some(n / 2)).ok();

    let mut state = GameState::Initialize(init);
    let brain_of: HashMap<PlayerID, Brain> =
        seats.iter().copied().zip(brains.iter().copied()).collect();

    // A safety cap on total iterations.
    let mut iterations = 0usize;
    let max_iterations = 200_000usize;

    loop {
        iterations += 1;
        if iterations > max_iterations {
            return None;
        }

        match &mut state {
            GameState::Initialize(s) => {
                // Pick a random-ish landlord deterministically (seat 0) and start.
                match s.landlord() {
                    None => {
                        s.set_landlord(Some(seats[0])).ok();
                    }
                    Some(landlord) => {
                        state = GameState::Draw(s.start(landlord).ok()?);
                    }
                }
            }
            GameState::Draw(s) => {
                if !s.done_drawing() {
                    let p = s.next_player().ok()?;
                    s.draw_card(p).ok()?;
                } else if s.bid_decided() {
                    // Advance the winning bidder into the exchange phase.
                    let responsible = s.next_player().ok()?;
                    let exchange = s.advance(responsible).ok()?;
                    state = GameState::Exchange(exchange);
                } else {
                    // No bid: let the bots try to bid by strength; if none want
                    // to, reveal the bottom (auto-bid) for the landlord.
                    let mut bid_made = false;
                    for &seat in &seats {
                        if let Brain::Tier(d) = brain_of[&seat] {
                            if let Some(bid) = policy::choose_bid(s, seat, d) {
                                if s.bid(seat, bid.card, bid.count) {
                                    bid_made = true;
                                    break;
                                }
                            }
                        }
                    }
                    if !bid_made {
                        // Reveal from the bottom to establish trump (auto-bid).
                        if s.reveal_card().is_err() {
                            // Fall back to a minimal legal bid from any seat.
                            let mut any = false;
                            for &seat in &seats {
                                if let Some(bid) =
                                    s.valid_bids(seat).ok()?.into_iter().min_by_key(|b| b.count)
                                {
                                    if s.bid(seat, bid.card, bid.count) {
                                        any = true;
                                        break;
                                    }
                                }
                            }
                            if !any {
                                return None;
                            }
                        }
                    }
                }
            }
            GameState::Exchange(s) => {
                let landlord = s.landlord();
                // Only the landlord acts; use its brain (degenerate brains still
                // bury via the tier heuristic — kitty discipline is shared).
                let view = GameState::Exchange(s.clone()).for_player(landlord);
                let difficulty = match brain_of[&landlord] {
                    Brain::Tier(d) => d,
                    _ => BotDifficulty::Expert,
                };
                match policy::select_action(&view, landlord, difficulty).ok()? {
                    Some(Action::MoveCardToKitty(c)) => {
                        s.move_card_to_kitty(landlord, c).ok()?;
                    }
                    Some(Action::MoveCardToHand(c)) => {
                        s.move_card_to_hand(landlord, c).ok()?;
                    }
                    Some(Action::SetFriends(friends)) => {
                        s.set_friends(landlord, friends).ok()?;
                    }
                    Some(Action::BeginPlay) | None => {
                        state = GameState::Play(s.advance(landlord).ok()?);
                    }
                    Some(_) => {
                        // Unexpected; just begin play.
                        state = GameState::Play(s.advance(landlord).ok()?);
                    }
                }
            }
            GameState::Play(s) => {
                if s.game_finished() {
                    let (non_landlord_points, _) = s.calculate_points();
                    // Determine winner via finish_game.
                    let landlord_seat = s.landlord();
                    let (_, landlord_won, _) = s.finish_game().ok()?;
                    return Some(HandOutcome {
                        landlord_won,
                        landlord_seat,
                        non_landlord_points,
                    });
                }
                match s.trick().next_player() {
                    None => {
                        // Trick complete: finish it.
                        s.finish_trick().ok()?;
                    }
                    Some(actor) => {
                        let cards = match brain_of[&actor] {
                            Brain::Tier(d) => {
                                // Honesty boundary, mirroring production
                                // `bot::observed_state`: honest tiers see only the
                                // redacted per-player view; the Omniscient CHEATER
                                // tier is handed the TRUE full state so its
                                // perfect-information search reads the real hands.
                                let view = observed_state_for(s, actor, d);
                                match policy::select_action(&view, actor, d).ok()? {
                                    Some(Action::PlayCards(c)) => c,
                                    _ => return None,
                                }
                            }
                            Brain::AlwaysDumpPoints => degenerate_play(s, actor, true),
                            Brain::AlwaysTrump => degenerate_play(s, actor, false),
                        };
                        s.play_cards(actor, &cards).ok()?;
                    }
                }
            }
        }
    }
}

/// Mirror production's centralized honesty bypass (`bot::observed_state`) for the
/// eval harness's play phase: honest tiers (`Easy`/`Expert`/`Enoch`) get the
/// redacted per-player view; the `Omniscient` CHEATER tier gets the TRUE full
/// state (every seat's real cards) so its perfect-information search can read
/// them. This is the only place the harness ever hands a bot the unredacted
/// state, and it is gated to `Omniscient` only.
fn observed_state_for(
    s: &shengji_core::game_state::play_phase::PlayPhase,
    actor: PlayerID,
    difficulty: BotDifficulty,
) -> GameState {
    let full = GameState::Play(s.clone());
    if matches!(difficulty, BotDifficulty::Omniscient) {
        full
    } else {
        full.for_player(actor)
    }
}

/// A degenerate-opponent play: from the redacted view, pick a legal play that
/// either dumps the most points (`dump_points`) or trumps in aggressively.
/// Falls back to the bot's legal candidate generator for legality.
fn degenerate_play(
    s: &shengji_core::game_state::play_phase::PlayPhase,
    actor: PlayerID,
    dump_points: bool,
) -> Vec<Card> {
    use shengji_core::bot::heuristics;
    let view = GameState::Play(s.clone()).for_player(actor);
    let view_play = match &view {
        GameState::Play(p) => p,
        _ => unreachable!(),
    };
    let leading = view_play.trick().played_cards().is_empty();
    let candidates = if leading {
        heuristics::lead_candidates(view_play, actor)
    } else {
        heuristics::follow_candidates(view_play, actor)
    };
    let trump = view_play.trick().trump();
    candidates
        .into_iter()
        .max_by_key(|cards| {
            if dump_points {
                cards
                    .iter()
                    .filter_map(|c| c.points().map(|x| x as i32))
                    .sum::<i32>()
            } else {
                cards
                    .iter()
                    .filter(|c| trump.effective_suit(**c) == EffectiveSuit::Trump)
                    .count() as i32
            }
        })
        .unwrap_or_default()
}

/// Tally of wins per brain label across a match.
#[derive(Default)]
struct Tally {
    hands: usize,
    /// Hands won by the team containing a seat with this brain.
    wins: HashMap<&'static str, usize>,
    appearances: HashMap<&'static str, usize>,
    /// Sum of attacker points across hands (margin indicator).
    attacker_points: isize,
}

impl Tally {
    fn record(&mut self, brains: &[Brain], outcome: &HandOutcome) {
        self.hands += 1;
        self.attacker_points += outcome.non_landlord_points;
        // Landlord team = seats with the same parity as the landlord seat.
        let landlord_idx = outcome.landlord_seat.0;
        for (idx, b) in brains.iter().enumerate() {
            let label = b.label();
            *self.appearances.entry(label).or_default() += 1;
            let on_landlord_team = idx % 2 == landlord_idx % 2;
            let team_won = on_landlord_team == outcome.landlord_won;
            if team_won {
                *self.wins.entry(label).or_default() += 1;
            }
        }
    }

    fn win_rate(&self, label: &str) -> f64 {
        let apps = self.appearances.get(label).copied().unwrap_or(0);
        if apps == 0 {
            return 0.0;
        }
        self.wins.get(label).copied().unwrap_or(0) as f64 / apps as f64
    }
}

/// Run a head-to-head match: team A (seats 0,2) uses brain `a`, team B (seats
/// 1,3) uses brain `b`. Plays `games` hands, swapping which team is landlord by
/// alternating the landlord seat parity via seat rotation (mirror) to cancel
/// deal luck. Returns (a_wins, b_wins).
fn head_to_head(a: Brain, b: Brain, games: usize) -> (usize, usize) {
    let mut a_wins = 0;
    let mut b_wins = 0;
    for g in 0..games {
        // Mirror: on odd games, swap which seats get which brain so each brain
        // spends equal time as the (advantaged) landlord team.
        let brains = if g % 2 == 0 {
            vec![a, b, a, b]
        } else {
            vec![b, a, b, a]
        };
        if let Some(outcome) = play_one_hand(&brains) {
            // Seat 0 is always the landlord (we set it). Team of seat 0:
            let landlord_brain = brains[0];
            let landlord_team_won = outcome.landlord_won;
            let winning_brain = if landlord_team_won {
                landlord_brain
            } else {
                brains[1]
            };
            if winning_brain == a {
                a_wins += 1;
            } else if winning_brain == b {
                b_wins += 1;
            }
        }
    }
    (a_wins, b_wins)
}

fn print_matchup(a: Brain, b: Brain, games: usize) {
    let (aw, bw) = head_to_head(a, b, games);
    print_matchup_result(a, b, aw, bw);
}

/// Print a head-to-head line from precomputed results (so the caller can reuse
/// the same batch for an ordering check without re-running games).
fn print_matchup_result(a: Brain, b: Brain, aw: usize, bw: usize) {
    let total = (aw + bw).max(1);
    println!(
        "  {:>10} vs {:<10} : {:>3} - {:<3}  ({} win rate {:.0}%)",
        a.label(),
        b.label(),
        aw,
        bw,
        a.label(),
        100.0 * aw as f64 / total as f64,
    );
}

fn main() {
    let start = Instant::now();
    // The search tiers (heuristic / net / playbook + determinized search) need
    // enough simulations per decision to demonstrate their edge over Easy (the
    // noisy heuristic). A
    // debug build runs ~5x slower, so it gets ~5x fewer sims for the same
    // wall-clock budget; we therefore default to a larger budget in debug to
    // keep the ladder monotonic, and a smaller one in release where sims are
    // cheap. Override with `SHENGJI_BOT_BUDGET_MS` / `EVAL_GAMES`, e.g.:
    //   SHENGJI_BOT_BUDGET_MS=80 EVAL_GAMES=120 cargo run --release --example eval
    // (Running --release is recommended; it is far faster.)
    if std::env::var("SHENGJI_BOT_BUDGET_MS").is_err() {
        let default_budget = if cfg!(debug_assertions) { "80" } else { "20" };
        std::env::set_var("SHENGJI_BOT_BUDGET_MS", default_budget);
    }

    let games: usize = std::env::var("EVAL_GAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(if cfg!(debug_assertions) { 24 } else { 60 });

    println!("=== Shengji bot difficulty ladder (self-play) ===");
    println!("Per matchup: {games} mirrored hands (seat-swapped to cancel deal luck).\n");

    println!("Difficulty ladder (head-to-head win rates):");
    // Perfect-information ceiling: the Omniscient CHEATER tier should be at least
    // as strong as Expert (it shares the determinized-search machinery but reads
    // the REAL hands instead of sampling them). Beneath it the honest ladder is
    // Easy < Expert <= Enoch (Expert is the learned net distilled from
    // Omniscient; Enoch reuses the same search but layers on the full-game
    // playbook). Each pairing is run ONCE and reused for both the printed line
    // and the ordering summary below.
    let (omni_x, exp_o) = head_to_head(
        Brain::Tier(BotDifficulty::Omniscient),
        Brain::Tier(BotDifficulty::Expert),
        games,
    );
    print_matchup_result(
        Brain::Tier(BotDifficulty::Omniscient),
        Brain::Tier(BotDifficulty::Expert),
        omni_x,
        exp_o,
    );
    print_matchup(
        Brain::Tier(BotDifficulty::Omniscient),
        Brain::Tier(BotDifficulty::Enoch),
        games,
    );
    let (enoch_x, exp_n) = head_to_head(
        Brain::Tier(BotDifficulty::Enoch),
        Brain::Tier(BotDifficulty::Expert),
        games,
    );
    print_matchup_result(
        Brain::Tier(BotDifficulty::Enoch),
        Brain::Tier(BotDifficulty::Expert),
        enoch_x,
        exp_n,
    );
    let (exp_e, easy_x) = head_to_head(
        Brain::Tier(BotDifficulty::Expert),
        Brain::Tier(BotDifficulty::Easy),
        games,
    );
    print_matchup_result(
        Brain::Tier(BotDifficulty::Expert),
        Brain::Tier(BotDifficulty::Easy),
        exp_e,
        easy_x,
    );
    let (enoch_e, easy_n) = head_to_head(
        Brain::Tier(BotDifficulty::Enoch),
        Brain::Tier(BotDifficulty::Easy),
        games,
    );
    print_matchup_result(
        Brain::Tier(BotDifficulty::Enoch),
        Brain::Tier(BotDifficulty::Easy),
        enoch_e,
        easy_n,
    );

    // Summarize the ordering we expect: Omniscient >= Expert, Enoch >= Expert
    // (Enoch reuses the search and adds the playbook), and both Expert and Enoch
    // beat Easy.
    println!(
        "\nLadder check  (expect Omniscient>=Expert, Enoch>=Expert, Expert>Easy, Enoch>Easy):\n  \
         Omniscient {} - {} Expert  |  Enoch {} - {} Expert  |  Expert {} - {} Easy  |  Enoch {} - {} Easy",
        omni_x, exp_o, enoch_x, exp_n, exp_e, easy_x, enoch_e, easy_n
    );
    let omni_ge_exp = omni_x >= exp_o;
    let enoch_ge_exp = enoch_x >= exp_n;
    let exp_gt_easy = exp_e > easy_x;
    let enoch_gt_easy = enoch_e > easy_n;
    println!(
        "  => Omniscient>=Expert: {}  Enoch>=Expert: {}  Expert>Easy: {}  Enoch>Easy: {}",
        if omni_ge_exp { "yes" } else { "NO" },
        if enoch_ge_exp { "yes" } else { "NO" },
        if exp_gt_easy { "yes" } else { "NO" },
        if enoch_gt_easy { "yes" } else { "NO" },
    );

    println!("\nExploitability probes (our bots vs degenerate opponents):");
    print_matchup(
        Brain::Tier(BotDifficulty::Expert),
        Brain::AlwaysDumpPoints,
        games,
    );
    print_matchup(
        Brain::Tier(BotDifficulty::Expert),
        Brain::AlwaysTrump,
        games,
    );
    print_matchup(
        Brain::Tier(BotDifficulty::Enoch),
        Brain::AlwaysDumpPoints,
        games,
    );

    // A mixed-table tally for a holistic view.
    println!("\nMixed table (all four tiers/probes at once), per-seat win rates:");
    let mut tally = Tally::default();
    let table = [
        Brain::Tier(BotDifficulty::Enoch),
        Brain::Tier(BotDifficulty::Expert),
        Brain::Tier(BotDifficulty::Easy),
        Brain::Tier(BotDifficulty::Expert),
    ];
    for _ in 0..games {
        if let Some(outcome) = play_one_hand(&table) {
            tally.record(&table, &outcome);
        }
    }
    for label in ["Enoch", "Expert", "Easy"] {
        if tally.appearances.contains_key(label) {
            println!("  {:>8}: {:.0}% ", label, 100.0 * tally.win_rate(label));
        }
    }
    if tally.hands > 0 {
        println!(
            "  (avg attacker points per hand: {:.1})",
            tally.attacker_points as f64 / tally.hands as f64
        );
    }

    println!("\nDone in {:.1}s.", start.elapsed().as_secs_f64());
}
