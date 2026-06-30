//! Self-play Deep Monte-Carlo (DMC) data generator for the **Oracle** tier.
//!
//! DouZero-style: NO teacher, NO tree search. Plays all-bot self-play hands where
//! every seat acts **ε-greedy on the CURRENT Q-net** (loaded via
//! `SHENGJI_EXPERT_MODEL_PATH`; if unset, a heuristic-ish bootstrap), and records,
//! for every PLAY-phase decision, ONE row:
//!
//!     f0..f{D-1}, ret
//!
//! where the features are the SAME honest `bot::expert::candidate_features` for the
//! `(state, CHOSEN candidate)` and `ret` is the REALIZED terminal point-margin
//! oriented for the acting seat's team, normalized by `expert::VALUE_NORM` and
//! clamped to [-1, 1] (back-filled at game end). The Python trainer
//! (`training/train_dmc.py`) regresses Q(s,a) ≈ ret; at serve time Oracle takes
//! `argmax_a Q(s,a)` over the legal candidates — search-free.
//!
//! Because the behavior policy reads the net through the standard
//! `expert::score_candidates_net` (honest redacted view only) and the recorded
//! features are honest, this is fully honesty-safe — no hidden hands are ever read.
//!
//! Env knobs:
//!   DMC_GAMES    self-play hands to export (default 400)
//!   DMC_OUT      output CSV path (default training/dmc_data.csv)
//!   DMC_SEED     deal/behaviour RNG seed (default 0xDМC0; distinct seeds shard)
//!   DMC_EPSILON  exploration rate: pick a uniform-random legal candidate with this
//!                probability, else argmax-Q (default 0.10)
//!   SHENGJI_EXPERT_MODEL_PATH  the current Q-net to act with (unset ⇒ bootstrap:
//!                the first candidate, which the generators order heuristic-ish).
//!   SHENGJI_BOT_BUDGET_MS  bid/exchange (Expert) search budget (default 60).

use std::fs::File;
use std::io::{BufWriter, Write};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use shengji_core::bot::expert::{candidate_features, score_candidates_net, FEATURE_DIM, VALUE_NORM};
use shengji_core::bot::BotDifficulty;
use shengji_core::bot::{heuristics, policy};
use shengji_core::game_state::initialize_phase::InitializePhase;
use shengji_core::game_state::play_phase::PlayPhase;
use shengji_core::game_state::GameState;
use shengji_core::interactive::Action;
use shengji_mechanics::types::{Card, PlayerID};

struct Row {
    features: [f32; FEATURE_DIM],
    actor: PlayerID,
    ret: f32,
}

fn main() {
    if std::env::var("SHENGJI_BOT_BUDGET_MS").is_err() {
        std::env::set_var("SHENGJI_BOT_BUDGET_MS", "60");
    }
    let games: usize = env_parse("DMC_GAMES", 400);
    let seed: u64 = env_parse("DMC_SEED", 0xD_3C_0);
    let epsilon: f64 = env_parse_f64("DMC_EPSILON", 0.10).clamp(0.0, 1.0);
    let out_path = std::env::var("DMC_OUT").unwrap_or_else(|_| "training/dmc_data.csv".to_string());
    let have_model = std::env::var_os("SHENGJI_EXPERT_MODEL_PATH").is_some();
    eprintln!(
        "DMC gen: games={games} seed={seed} eps={epsilon} model={}",
        if have_model { "Q-net (override)" } else { "bootstrap(first-candidate)" }
    );

    let start = std::time::Instant::now();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut rows: Vec<Row> = Vec::new();
    let mut decisions = 0usize;

    for g in 0..games {
        play_one_hand(&mut rng, &mut rows, &mut decisions, epsilon);
        if g % 25 == 0 {
            eprintln!(
                "  game {g}/{games}: {decisions} decisions, {} rows ({:.0}s)",
                rows.len(),
                start.elapsed().as_secs_f64()
            );
        }
    }

    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let file = File::create(&out_path).expect("create output CSV");
    let mut w = BufWriter::new(file);
    write!(w, "f0").unwrap();
    for i in 1..FEATURE_DIM {
        write!(w, ",f{i}").unwrap();
    }
    writeln!(w, ",ret").unwrap();
    for r in &rows {
        let mut it = r.features.iter();
        write!(w, "{:.6}", it.next().unwrap()).unwrap();
        for x in it {
            write!(w, ",{x:.6}").unwrap();
        }
        writeln!(w, ",{:.6}", r.ret).unwrap();
    }
    w.flush().unwrap();
    eprintln!(
        "Wrote {} rows ({} decisions, {} games) to {} in {:.1}s",
        rows.len(),
        decisions,
        games,
        out_path,
        start.elapsed().as_secs_f64()
    );
}

fn env_parse<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn env_parse_f64(key: &str, default: f64) -> f64 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Drive one all-bot self-play hand with ε-greedy-Q play, recording one row per
/// play decision (chosen action's honest features), value back-filled at game end.
fn play_one_hand(rng: &mut StdRng, rows: &mut Vec<Row>, decisions: &mut usize, epsilon: f64) {
    let n = 4;
    let mut init = InitializePhase::new();
    let mut seats: Vec<PlayerID> = vec![];
    for i in 0..n {
        seats.push(init.add_player(format!("seat{i}")).unwrap().0);
    }
    init.set_num_decks(Some(n / 2)).ok();

    let mut game_rows: Vec<Row> = Vec::new();
    let mut game_decisions = 0usize;

    let mut state = GameState::Initialize(init);
    let mut iters = 0usize;
    loop {
        iters += 1;
        if iters > 200_000 {
            return;
        }
        match &mut state {
            GameState::Initialize(s) => match s.landlord() {
                None => {
                    let l = seats[rng.gen_range(0..seats.len())];
                    s.set_landlord(Some(l)).ok();
                }
                Some(l) => {
                    state = match s.start(l) {
                        Ok(d) => GameState::Draw(d),
                        Err(_) => return,
                    };
                }
            },
            GameState::Draw(s) => {
                if !s.done_drawing() {
                    let p = match s.next_player() {
                        Ok(p) => p,
                        Err(_) => return,
                    };
                    if s.draw_card(p).is_err() {
                        return;
                    }
                } else if s.bid_decided() {
                    let responsible = match s.next_player() {
                        Ok(p) => p,
                        Err(_) => return,
                    };
                    state = match s.advance(responsible) {
                        Ok(e) => GameState::Exchange(e),
                        Err(_) => return,
                    };
                } else {
                    let mut bid = false;
                    for &seat in &seats {
                        if let Some(b) = policy::choose_bid(s, seat, BotDifficulty::Expert) {
                            if s.bid(seat, b.card, b.count) {
                                bid = true;
                                break;
                            }
                        }
                    }
                    if !bid && s.reveal_card().is_err() {
                        for &seat in &seats {
                            if let Some(b) = s
                                .valid_bids(seat)
                                .ok()
                                .and_then(|v| v.into_iter().min_by_key(|b| b.count))
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
                let landlord = s.landlord();
                let view = GameState::Exchange(s.clone()).for_player(landlord);
                match policy::select_action(&view, landlord, BotDifficulty::Expert)
                    .ok()
                    .flatten()
                {
                    Some(Action::MoveCardToKitty(c)) => {
                        if s.move_card_to_kitty(landlord, c).is_err() {
                            return;
                        }
                    }
                    Some(Action::MoveCardToHand(c)) => {
                        if s.move_card_to_hand(landlord, c).is_err() {
                            return;
                        }
                    }
                    Some(Action::SetFriends(f)) => {
                        if s.set_friends(landlord, f).is_err() {
                            return;
                        }
                    }
                    _ => {
                        state = match s.advance(landlord) {
                            Ok(p) => GameState::Play(p),
                            Err(_) => return,
                        };
                    }
                }
            }
            GameState::Play(s) => {
                if s.game_finished() {
                    let (non_landlord_points, _) = s.calculate_points();
                    let landlords_team: Vec<PlayerID> = s.landlords_team().to_vec();
                    for mut r in game_rows.drain(..) {
                        let actor_is_defender = landlords_team.contains(&r.actor);
                        let oriented = if actor_is_defender {
                            -(non_landlord_points as f64)
                        } else {
                            non_landlord_points as f64
                        };
                        r.ret = (oriented / VALUE_NORM).clamp(-1.0, 1.0) as f32;
                        rows.push(r);
                    }
                    *decisions += game_decisions;
                    return;
                }
                match s.trick().next_player() {
                    None => {
                        if s.finish_trick().is_err() {
                            return;
                        }
                    }
                    Some(actor) => {
                        // Honest redacted view drives BOTH the behaviour policy and
                        // the recorded features.
                        let view_state = GameState::Play(s.clone()).for_player(actor);
                        let view = match &view_state {
                            GameState::Play(p) => p,
                            _ => return,
                        };
                        let chosen = match choose_eps_greedy_q(view, actor, epsilon, rng) {
                            Some(c) => c,
                            None => return,
                        };
                        game_rows.push(Row {
                            features: candidate_features(view, actor, &chosen),
                            actor,
                            ret: 0.0, // back-filled at game end
                        });
                        game_decisions += 1;
                        if s.play_cards(actor, &chosen).is_err() {
                            return;
                        }
                    }
                }
            }
        }
    }
}

/// ε-greedy over Q: with prob `epsilon` a uniform-random legal candidate
/// (exploration), else argmax of the Q-net's per-candidate score. With no model
/// loaded, "argmax" degrades to the first candidate (the generators order them
/// heuristic-ish), giving a reasonable bootstrap policy for DMC iteration 0.
fn choose_eps_greedy_q(
    view: &PlayPhase,
    actor: PlayerID,
    epsilon: f64,
    rng: &mut StdRng,
) -> Option<Vec<Card>> {
    let leading = view.trick().played_cards().is_empty();
    let candidates: Vec<Vec<Card>> = if leading {
        heuristics::lead_candidates(view, actor)
    } else {
        heuristics::follow_candidates(view, actor)
    };
    if candidates.is_empty() {
        return None;
    }
    if candidates.len() == 1 {
        return Some(candidates.into_iter().next().unwrap());
    }
    if rng.gen_bool(epsilon) {
        let i = rng.gen_range(0..candidates.len());
        return Some(candidates.into_iter().nth(i).unwrap());
    }
    let best_idx = match score_candidates_net(view, actor, &candidates) {
        Some(scores) => {
            let mut bi = 0;
            let mut best = f32::NEG_INFINITY;
            for (i, &s) in scores.iter().enumerate() {
                if s > best {
                    best = s;
                    bi = i;
                }
            }
            bi
        }
        None => 0, // bootstrap: first candidate (heuristic-ordered-ish)
    };
    Some(candidates.into_iter().nth(best_idx).unwrap())
}
