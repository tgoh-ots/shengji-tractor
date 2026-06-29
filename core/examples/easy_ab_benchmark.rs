//! Headless A/B benchmark for the EASY tier's tuning knobs.
//!
//! This measures the effect of the Easy-tier knob change (softmax temperature
//! and blunder rate) by pitting Easy@NEW against Easy@OLD over many seeded
//! 4-player Tractor hands. Both sides play the SAME (greedy heuristic-direct)
//! candidate scorer with NO search; the ONLY thing that differs is the
//! `(epsilon, temperature)` pair applied on top of the shared heuristic ranking:
//!
//!   * Easy@NEW — ε = 0.06, T = 1.1 (the strengthened, less-noisy Easy)
//!   * Easy@OLD — ε = 0.28, T = 3.5 (the original, very noisy Easy)
//!
//! Because both sides share the identical heuristic candidate list (the public
//! `heuristics::ranked_leads` / `ranked_follows`), the bidding/kitty driver, and
//! the deal, any win-rate edge is attributable purely to the knob change. We
//! alternate which partnership is NEW across games to cancel the landlord/dealer
//! positional edge.
//!
//! The Easy policy reproduced here mirrors `bot::policy::choose_play` for the
//! search-less (`search_worlds == 0`) Easy tier: with probability ε pick a random
//! legal candidate (a blunder), otherwise softmax-sample the top-4 heuristic
//! candidates at temperature T. It reads only the actor's own redacted, honest
//! per-player view, exactly like production.
//!
//! Run with:
//!   cargo run --release --example easy_ab_benchmark -- [num_games] [base_seed]
//!
//! We expect a MODEST bump for Easy@NEW (~55-60%), NOT a blowout: Easy must stay
//! the weakest, clearly-beatable casual tier.

use std::env;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use shengji_core::bot::heuristics::{self, ScoredPlay};
use shengji_core::bot::{policy, BotDifficulty};
use shengji_core::game_state::draw_phase::DrawPhase;
use shengji_core::game_state::initialize_phase::InitializePhase;
use shengji_core::game_state::play_phase::PlayPhase;
use shengji_core::game_state::GameState;
use shengji_core::interactive::Action;
use shengji_core::settings::GameModeSettings;

use shengji_mechanics::deck::Deck;
use shengji_mechanics::types::{Card, PlayerID};

/// One Easy knob configuration (the only thing that differs between the A/B
/// sides). Mirrors the `epsilon` / `temperature` fields of `policy::Knobs`.
#[derive(Clone, Copy)]
struct EasyKnobs {
    label: &'static str,
    epsilon: f64,
    temperature: f64,
}

/// The NEW (strengthened) Easy: fewer blunders, a cooler softmax. Keep these in
/// sync with `BotDifficulty::Easy` in `core/src/bot/policy.rs`. The defaults can
/// be overridden via `EASY_NEW_EPS` / `EASY_NEW_TEMP` for quick tuning sweeps.
fn new_easy() -> EasyKnobs {
    EasyKnobs {
        label: "Easy@NEW",
        epsilon: env::var("EASY_NEW_EPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.06),
        temperature: env::var("EASY_NEW_TEMP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.1),
    }
}

/// The OLD (pre-change) Easy: very noisy — frequent blunders, hot softmax.
const OLD_EASY: EasyKnobs = EasyKnobs {
    label: "Easy@OLD",
    epsilon: 0.28,
    temperature: 3.5,
};

/// Reproduce the Easy tier's search-less play policy for `actor` under `knobs`,
/// honoring the honesty boundary (only the actor's own redacted view). Returns
/// `None` only if no legal candidate exists (the engine then errors out).
fn easy_play_for(
    s: &PlayPhase,
    actor: PlayerID,
    knobs: EasyKnobs,
    rng: &mut StdRng,
) -> Option<Vec<Card>> {
    let view = GameState::Play(s.clone()).for_player(actor);
    let pp = match &view {
        GameState::Play(pp) => pp,
        _ => return None,
    };
    let leading = pp.trick().played_cards().is_empty();
    // The shared heuristic candidate list (identical scorer for both sides).
    let ranked: Vec<ScoredPlay> = if leading {
        heuristics::ranked_leads(pp, actor)
    } else {
        heuristics::ranked_follows(pp, actor)
    };
    if ranked.is_empty() {
        return None;
    }

    // ε-blunder: with probability `epsilon`, play a uniformly random legal
    // candidate instead of a scored move (the beginner "obvious blunder").
    if rng.gen_bool(knobs.epsilon.clamp(0.0, 1.0)) {
        let idx = rng.gen_range(0..ranked.len());
        return Some(ranked[idx].cards.clone());
    }

    // Otherwise softmax-sample the top-4 candidates at temperature T (T = 0 would
    // be a greedy argmax; both Easy configs use a warm T). Mirrors
    // `policy::pick_from_ranked`.
    if knobs.temperature <= 0.0 {
        return Some(ranked[0].cards.clone());
    }
    let top = &ranked[..ranked.len().min(4)];
    let max = top.iter().map(|c| c.score).fold(f64::MIN, f64::max);
    let weights: Vec<f64> = top
        .iter()
        .map(|c| ((c.score - max) / knobs.temperature).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    if total <= 0.0 || !total.is_finite() {
        return Some(top[0].cards.clone());
    }
    let mut pick = rng.gen::<f64>() * total;
    for (c, w) in top.iter().zip(weights.iter()) {
        pick -= w;
        if pick <= 0.0 {
            return Some(c.cards.clone());
        }
    }
    Some(top[top.len() - 1].cards.clone())
}

/// Build a fully-seeded 4-player, 2-deck Tractor Draw phase, seat 0 preselected as
/// landlord (mirrors `heuristic_benchmark`'s `seeded_draw_phase`).
fn seeded_draw_phase(decks: &[Deck], rng: &mut StdRng) -> DrawPhase {
    let mut deck: Vec<_> = decks.iter().flat_map(|d| d.cards()).collect();
    deck.shuffle(rng);

    let num_players = 4;
    let mut kitty_size = deck.len() % num_players;
    if kitty_size == 0 {
        kitty_size = num_players;
    }
    if kitty_size < 5 {
        kitty_size += num_players;
    }

    let mut init = InitializePhase::new();
    for i in 0..num_players {
        init.add_player(format!("seat{i}")).unwrap();
    }
    init.set_num_decks(Some(decks.len())).unwrap();
    init.set_game_mode(GameModeSettings::Tractor).unwrap();
    let real_seats: Vec<PlayerID> = init.players().iter().map(|p| p.id).collect();
    init.set_landlord(Some(real_seats[0])).unwrap();
    let propagated = (*init).clone();

    let level = Some(propagated.players()[0].rank());
    let hands_deck = deck[0..deck.len() - kitty_size].to_vec();
    let kitty = deck[deck.len() - kitty_size..].to_vec();

    DrawPhase::new(
        propagated,
        0,
        hands_deck,
        kitty,
        decks.len(),
        shengji_core::settings::GameMode::Tractor,
        level,
        decks.to_vec(),
        vec![],
    )
}

/// The outcome of one finished hand, from the NEW partnership's perspective.
struct GameOutcome {
    new_won: bool,
    new_point_margin: isize,
}

/// Drive one seeded hand. `new_is_landlord_team` selects which partnership plays
/// with the NEW Easy knobs; the other uses the OLD knobs. Both partnerships bid /
/// exchange identically via the shared Easy driver, so only the play-phase knobs
/// differ.
fn play_one_hand(
    new_is_landlord_team: bool,
    new_knobs: EasyKnobs,
    old_knobs: EasyKnobs,
    rng: &mut StdRng,
) -> Option<GameOutcome> {
    let decks = vec![Deck::default(), Deck::default()];
    let draw = seeded_draw_phase(&decks, rng);
    let seats: Vec<PlayerID> = draw.propagated().players().iter().map(|p| p.id).collect();

    // Seats 0,2 are the landlord (defending) team; 1,3 attack. The NEW knobs
    // occupy the landlord team iff `new_is_landlord_team`.
    let knobs_of = |seat_idx: usize| -> EasyKnobs {
        let is_landlord_team = seat_idx % 2 == 0;
        if is_landlord_team == new_is_landlord_team {
            new_knobs
        } else {
            old_knobs
        }
    };

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
                    // Both partnerships bid via the same Easy driver so the trump /
                    // landlord is identical regardless of side.
                    let mut bid = false;
                    for &seat in &seats {
                        if let Some(b) = policy::choose_bid(s, seat, BotDifficulty::Easy) {
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
                // Kitty burying is knob-independent (shared Easy driver).
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
                    let (non_landlord_points, _) = s.calculate_points();
                    let (_init, landlord_won, _msgs) = s.finish_game().ok()?;

                    // NEW is the defending (landlord) team iff new_is_landlord_team.
                    let new_is_defender = new_is_landlord_team;
                    let new_won = landlord_won == new_is_defender;
                    let new_point_margin = if new_is_defender {
                        -non_landlord_points
                    } else {
                        non_landlord_points
                    };
                    return Some(GameOutcome {
                        new_won,
                        new_point_margin,
                    });
                }
                match s.trick().next_player() {
                    None => {
                        s.finish_trick().ok()?;
                    }
                    Some(actor) => {
                        let actor_idx = seats.iter().position(|x| *x == actor)?;
                        let knobs = knobs_of(actor_idx);
                        // Deterministic per-decision seed from the observable state
                        // so each side gets independent but stable randomness.
                        let hand_size = s
                            .hands()
                            .get(actor)
                            .map(|h| h.values().sum::<usize>())
                            .unwrap_or(0);
                        let seed = (actor.0 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                            ^ (hand_size as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
                            ^ (s.trick().played_cards().len() as u64)
                                .wrapping_mul(0x94D0_49BB_1331_11EB);
                        let mut decision_rng = StdRng::seed_from_u64(seed);
                        let cards = easy_play_for(s, actor, knobs, &mut decision_rng)?;
                        s.play_cards(actor, &cards).ok()?;
                    }
                }
            }
        }
    }
}

fn run_match(num_games: usize, base_seed: u64) {
    let start = Instant::now();
    let new_easy = new_easy();
    let mut new_wins = 0usize;
    let mut old_wins = 0usize;
    let mut total_margin: isize = 0;
    let mut completed = 0usize;

    for g in 0..num_games {
        let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(g as u64));
        let new_is_landlord_team = g % 2 == 0;
        match play_one_hand(new_is_landlord_team, new_easy, OLD_EASY, &mut rng) {
            Some(outcome) => {
                completed += 1;
                if outcome.new_won {
                    new_wins += 1;
                } else {
                    old_wins += 1;
                }
                total_margin += outcome.new_point_margin;
            }
            None => eprintln!("  game {g}: engine error / skipped"),
        }
    }

    let win_rate = if completed > 0 {
        new_wins as f64 / completed as f64 * 100.0
    } else {
        0.0
    };

    println!();
    println!(
        "=== {} (ε={}, T={}) vs {} (ε={}, T={}) ===",
        new_easy.label,
        new_easy.epsilon,
        new_easy.temperature,
        OLD_EASY.label,
        OLD_EASY.epsilon,
        OLD_EASY.temperature,
    );
    println!("  Games completed:      {completed}");
    println!("  Easy@NEW wins:        {new_wins}");
    println!("  Easy@OLD wins:        {old_wins}");
    println!("  Easy@NEW win-rate:    {win_rate:.2}%");
    println!(
        "  Easy@NEW avg margin:  {:+.2} pts/game",
        total_margin as f64 / completed.max(1) as f64
    );
    println!("  Elapsed: {:.1}s", start.elapsed().as_secs_f64());

    if completed > 0 {
        let n = completed as f64;
        let p = new_wins as f64 / n;
        let se = (0.25 / n).sqrt(); // SE under the null p=0.5
        let z = (p - 0.5) / se;
        println!("  (z vs 50% null = {z:+.2}; |z|>1.96 ≈ p<0.05 two-sided)");
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let num_games: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(400);
    let base_seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0xEA59);

    println!("EASY A/B benchmark: Easy@NEW vs Easy@OLD (knob change only)");
    println!("Games: {num_games}  base_seed: {base_seed:#x}");
    println!(
        "Both sides share the heuristic candidate scorer and the bidding/kitty \
         driver; only the blunder rate (ε) and softmax temperature (T) differ. \
         Sides alternate landlord/attacker each game.\n"
    );

    run_match(num_games, base_seed);
}
