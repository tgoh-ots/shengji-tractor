//! Headless A/B benchmark: NEW heuristic-direct play vs LEGACY heuristic-direct
//! play, over many full 4-player Tractor hands with SEEDED, reproducible deals.
//!
//! One partnership (2 seats) plays every PLAY-phase decision with the NEW
//! boss-/partner-aware scorer (`HeuristicVersion::New`), the other partnership
//! with the frozen LEGACY scorer (`HeuristicVersion::Legacy`). Both partnerships
//! share the *same* deterministic bidding + kitty policy, so ONLY the play-phase
//! scorer differs. We alternate which partnership is NEW across games to cancel
//! the landlord/dealer positional advantage.
//!
//! Run with:
//!   cargo run --release --example heuristic_benchmark -- [num_games] [base_seed]
//!
//! Reports the NEW side's win-rate, average final point margin, and average net
//! levels gained per game.

use std::env;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use shengji_core::bot::heuristics::{self, HeuristicVersion};
use shengji_core::bot::{policy, BotDifficulty};
use shengji_core::game_state::draw_phase::DrawPhase;
use shengji_core::game_state::initialize_phase::InitializePhase;
use shengji_core::game_state::GameState;
use shengji_core::interactive::Action;
use shengji_core::settings::GameModeSettings;

use shengji_mechanics::deck::Deck;
use shengji_mechanics::scoring::{compute_level_deltas, GameScoringParameters};
use shengji_mechanics::types::PlayerID;

/// The outcome of one finished hand, from the perspective of the NEW partnership.
struct GameOutcome {
    /// Did the NEW partnership WIN the hand? (Either NEW defended successfully as
    /// landlord, or NEW attacked and flipped the landlord.)
    new_won: bool,
    /// Final point MARGIN from NEW's perspective: + means NEW's situation is
    /// better. Concretely:
    ///   - if NEW is the attacking (non-landlord) side: +non_landlord_points
    ///   - if NEW is the defending (landlord) side:    -non_landlord_points
    /// (Attackers want non_landlord_points high; defenders want it low.)
    new_point_margin: isize,
    /// Net levels gained by the NEW side minus net levels gained by the LEGACY
    /// side this hand (level bumps the winning team earns).
    new_net_levels: isize,
}

/// Seats 0 & 2 are the landlord (defending) team; seats 1 & 3 attack. Seat 0 is
/// always the landlord and leads the first trick. `new_is_landlord_team` chooses
/// which partnership uses the NEW heuristic this hand: if true, the NEW side is
/// the landlord team (seats 0 & 2); otherwise NEW is the attacking team (1 & 3).
fn version_for_seat(seat_idx: usize, new_is_landlord_team: bool) -> HeuristicVersion {
    let is_landlord_team = seat_idx % 2 == 0; // seats 0,2 vs 1,3
    let is_new = is_landlord_team == new_is_landlord_team;
    if is_new {
        HeuristicVersion::New
    } else {
        HeuristicVersion::Legacy
    }
}

/// Build a fully-seeded Draw phase for a 4-player, 2-deck Tractor hand with seat 0
/// preselected as landlord. We construct the `DrawPhase` directly with a
/// seed-shuffled deck so the ENTIRE game is reproducible (the engine's own
/// `InitializePhase::start` uses `thread_rng`, which we cannot seed).
fn seeded_draw_phase(seats: &[PlayerID], decks: &[Deck], rng: &mut StdRng) -> DrawPhase {
    // Flatten the decks into a single card vector and seed-shuffle it.
    let mut deck: Vec<_> = decks.iter().flat_map(|d| d.cards()).collect();
    deck.shuffle(rng);

    let num_players = seats.len();
    // Default-kitty branch (mirrors InitializePhase::start's `None` arm): for
    // 4 players / 2 decks this yields a kitty of 8 (108 % 4 = 0 -> 4, < 5 -> +4).
    let mut kitty_size = deck.len() % num_players;
    if kitty_size == 0 {
        kitty_size = num_players;
    }
    if kitty_size < 5 {
        kitty_size += num_players;
    }

    // Rebuild a PropagatedState identical to one started normally: 4 named seats,
    // seat 0 landlord, 2 decks, Tractor.
    let mut init = InitializePhase::new();
    for (i, _) in seats.iter().enumerate() {
        init.add_player(format!("seat{i}")).unwrap();
    }
    init.set_num_decks(Some(decks.len())).unwrap();
    init.set_game_mode(GameModeSettings::Tractor).unwrap();
    // Preselect the landlord = seat 0 so trump is fixed by the reveal path.
    let real_seats: Vec<PlayerID> = init.players().iter().map(|p| p.id).collect();
    init.set_landlord(Some(real_seats[0])).unwrap();
    let propagated = (*init).clone();

    let level = Some(propagated.players()[0].rank());
    let hands_deck = deck[0..deck.len() - kitty_size].to_vec();
    let kitty = deck[deck.len() - kitty_size..].to_vec();

    DrawPhase::new(
        propagated,
        0, // position of the landlord seat
        hands_deck,
        kitty,
        decks.len(),
        shengji_core::settings::GameMode::Tractor,
        level,
        decks.to_vec(),
        vec![],
    )
}

/// Drive a single seeded hand to completion. `new_is_landlord_team` selects which
/// partnership is NEW. Returns `None` only on an unexpected engine error.
fn play_one_hand(new_is_landlord_team: bool, rng: &mut StdRng) -> Option<GameOutcome> {
    // Build the seeded deal and the seat ids.
    let decks = vec![Deck::default(), Deck::default()];
    // Placeholder seats; the real ids come from the constructed phase.
    let placeholder: Vec<PlayerID> = (0..4).map(PlayerID).collect();
    let draw = seeded_draw_phase(&placeholder, &decks, rng);
    let seats: Vec<PlayerID> = draw.propagated().players().iter().map(|p| p.id).collect();

    let mut state = GameState::Draw(draw);
    let mut iters = 0usize;
    loop {
        iters += 1;
        if iters > 2_000_000 {
            return None;
        }
        match &mut state {
            GameState::Initialize(_) => return None,
            GameState::Draw(s) => {
                if !s.done_drawing() {
                    let p = s.next_player().ok()?;
                    s.draw_card(p).ok()?;
                } else if s.bid_decided() {
                    let responsible = s.next_player().ok()?;
                    state = GameState::Exchange(s.advance(responsible).ok()?);
                } else {
                    // Deterministic bidding shared by both sides: let any seat bid
                    // by hand strength; otherwise reveal the bottom to fix trump.
                    let mut bid = false;
                    for &seat in &seats {
                        if let Some(b) = policy::choose_bid(s, seat) {
                            if s.bid(seat, b.card, b.count) {
                                bid = true;
                                break;
                            }
                        }
                    }
                    if !bid && s.reveal_card().is_err() {
                        for &seat in &seats {
                            if let Some(b) =
                                s.valid_bids(seat).ok()?.into_iter().min_by_key(|b| b.count)
                            {
                                if s.bid(seat, b.card, b.count) {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            GameState::Exchange(s) => {
                // Kitty burying is heuristic-version-independent (shared
                // `choose_kitty`); use the Easy policy deterministically.
                let landlord = s.landlord();
                let view = GameState::Exchange(s.clone()).for_player(landlord);
                match policy::select_action(&view, landlord, BotDifficulty::Easy).ok()? {
                    Some(Action::MoveCardToKitty(c)) => s.move_card_to_kitty(landlord, c).ok()?,
                    Some(Action::MoveCardToHand(c)) => s.move_card_to_hand(landlord, c).ok()?,
                    Some(Action::SetFriends(f)) => s.set_friends(landlord, f).ok()?,
                    _ => state = GameState::Play(s.advance(landlord).ok()?),
                }
            }
            GameState::Play(s) => {
                if s.game_finished() {
                    let landlord_seat = s.landlord();
                    let landlord_idx = seats.iter().position(|x| *x == landlord_seat)?;
                    let (non_landlord_points, _) = s.calculate_points();
                    let (_init, landlord_won, _msgs) = s.finish_game().ok()?;

                    // NEW side = landlord team (seats 0,2) iff new_is_landlord_team.
                    // landlord_idx is 0 in our construction, so the landlord TEAM is
                    // the even seats; NEW occupies that team iff new_is_landlord_team.
                    let new_is_defender = new_is_landlord_team;
                    let new_won = landlord_won == new_is_defender;

                    // Point margin from NEW's perspective.
                    let new_point_margin = if new_is_defender {
                        -non_landlord_points
                    } else {
                        non_landlord_points
                    };

                    // Net levels: recompute the level deltas the same way the
                    // engine does in finish_game (no FindingFriends here, so
                    // smaller_landlord_team = false).
                    // Default scoring parameters (the unconfigured game's
                    // parameters, matching the PropagatedState we built).
                    let gsp = GameScoringParameters::default();
                    let result =
                        compute_level_deltas(&gsp, &decks, non_landlord_points, false).ok();
                    let new_net_levels = match result {
                        Some(r) => {
                            let landlord_levels = r.landlord_delta as isize;
                            let non_landlord_levels = r.non_landlord_delta as isize;
                            if new_is_defender {
                                landlord_levels - non_landlord_levels
                            } else {
                                non_landlord_levels - landlord_levels
                            }
                        }
                        None => 0,
                    };

                    let _ = landlord_idx;
                    return Some(GameOutcome {
                        new_won,
                        new_point_margin,
                        new_net_levels,
                    });
                }
                match s.trick().next_player() {
                    None => {
                        s.finish_trick().ok()?;
                    }
                    Some(actor) => {
                        // Heuristic-DIRECT play (greedy argmax, NO search) under the
                        // version assigned to this seat's partnership. Each seat acts
                        // from its OWN redacted, honest per-player view.
                        let actor_idx = seats.iter().position(|x| *x == actor)?;
                        let version = version_for_seat(actor_idx, new_is_landlord_team);
                        let view = GameState::Play(s.clone()).for_player(actor);
                        let pp = match &view {
                            GameState::Play(pp) => pp,
                            _ => return None,
                        };
                        let cards = heuristics::choose_play_direct(pp, actor, version)?;
                        s.play_cards(actor, &cards).ok()?;
                    }
                }
            }
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let num_games: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(400);
    let base_seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0xC0FFEE);

    println!("Heuristic A/B benchmark: NEW heuristic-direct vs LEGACY heuristic-direct");
    println!("Games: {num_games}  base_seed: {base_seed:#x}");
    println!(
        "Policy: greedy argmax, NO search. Shared deterministic bidding/kitty; \
         only PLAY-phase scorer differs. Seats alternate NEW partnership across games."
    );
    println!();

    let start = Instant::now();

    let mut new_wins = 0usize;
    let mut legacy_wins = 0usize;
    let mut total_margin: isize = 0;
    let mut total_net_levels: isize = 0;
    let mut completed = 0usize;
    // Track wins split by which side NEW played, to expose any residual
    // positional bias.
    let mut new_as_landlord_games = 0usize;
    let mut new_as_landlord_wins = 0usize;
    let mut new_as_attacker_games = 0usize;
    let mut new_as_attacker_wins = 0usize;

    for g in 0..num_games {
        // Fixed seed sequence: deterministic, no wall-clock seeding.
        let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(g as u64));
        // Alternate which partnership is NEW to cancel the landlord advantage.
        let new_is_landlord_team = g % 2 == 0;

        match play_one_hand(new_is_landlord_team, &mut rng) {
            Some(outcome) => {
                completed += 1;
                if outcome.new_won {
                    new_wins += 1;
                } else {
                    legacy_wins += 1;
                }
                total_margin += outcome.new_point_margin;
                total_net_levels += outcome.new_net_levels;

                if new_is_landlord_team {
                    new_as_landlord_games += 1;
                    if outcome.new_won {
                        new_as_landlord_wins += 1;
                    }
                } else {
                    new_as_attacker_games += 1;
                    if outcome.new_won {
                        new_as_attacker_wins += 1;
                    }
                }
            }
            None => {
                eprintln!("game {g}: engine error / could not complete (skipped)");
            }
        }
    }

    let elapsed = start.elapsed();

    println!("=== Results over {completed} completed games ===");
    let win_rate = if completed > 0 {
        new_wins as f64 / completed as f64 * 100.0
    } else {
        0.0
    };
    println!("NEW wins:    {new_wins}");
    println!("LEGACY wins: {legacy_wins}");
    println!("NEW win-rate:               {win_rate:.2}%");
    println!(
        "NEW avg final point margin: {:+.2} points/game",
        total_margin as f64 / completed.max(1) as f64
    );
    println!(
        "NEW avg net levels gained:  {:+.3} levels/game",
        total_net_levels as f64 / completed.max(1) as f64
    );
    println!();
    println!("Positional split (sanity check that the edge isn't just the dealer):");
    if new_as_landlord_games > 0 {
        println!(
            "  NEW as LANDLORD team: {}/{} = {:.1}%",
            new_as_landlord_wins,
            new_as_landlord_games,
            new_as_landlord_wins as f64 / new_as_landlord_games as f64 * 100.0
        );
    }
    if new_as_attacker_games > 0 {
        println!(
            "  NEW as ATTACKER team: {}/{} = {:.1}%",
            new_as_attacker_wins,
            new_as_attacker_games,
            new_as_attacker_wins as f64 / new_as_attacker_games as f64 * 100.0
        );
    }
    println!();
    println!("Elapsed: {:.1}s", elapsed.as_secs_f64());
    let verdict = if win_rate >= 60.0 {
        "NEW heuristic is CLEARLY stronger (>=60% win-rate)."
    } else if win_rate > 53.0 {
        "NEW heuristic is stronger (win-rate meaningfully above 50%)."
    } else if win_rate >= 47.0 {
        "NO clear difference (win-rate ~50%)."
    } else {
        "NEW heuristic appears WEAKER (win-rate below 50%)."
    };
    println!("Verdict: {verdict}");
}
