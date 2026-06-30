//! Expert-Iteration (ExIt) data generator for the **Athena** tier.
//!
//! Unlike `gen_training_data` (which clones the *Omniscient cheater* — a target
//! that is often unidentifiable from honest features, an irreducible aliasing
//! floor), Athena clones the **honest determinized SEARCH**. At each PLAY decision
//! the current net-in-search (Expert tier, redacted view, `SHENGJI_EXPERT_MODEL_PATH`)
//! picks a move via lookahead; that move is BOTH the move that advances the game
//! AND the policy LABEL. Because the search consulted only honest information, the
//! label is learnable from honest features — and because search > its own prior,
//! cloning it IMPROVES the net (the AlphaZero policy-improvement operator). Iterating
//! (new net → stronger search → relabel → retrain) compounds the gains.
//!
//! Emits the SAME CSV as `gen_training_data` (group, f0..f{D-1}, label, value) so
//! `train_expert.py` trains it unchanged into a 2-output (policy + value) net. The
//! value target is the realized terminal margin (honest on-policy Monte-Carlo),
//! oriented per acting team, normalized by `expert::VALUE_NORM`.
//!
//! Env knobs:
//!   GEN_GAMES, GEN_OUT, GEN_SEED   as in gen_training_data
//!   EXIT_TIER       honest search tier that BOTH plays and labels: expert | enoch
//!                   (default expert)
//!   SHENGJI_BOT_BUDGET_MS  the search budget per decision (label quality; default 200)
//!   SHENGJI_EXPERT_MODEL_PATH  the CURRENT iteration's net (the search prior)

use std::fs::File;
use std::io::{BufWriter, Write};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use shengji_core::bot::expert::{candidate_features, FEATURE_DIM, VALUE_NORM};
use shengji_core::bot::BotDifficulty;
use shengji_core::bot::{heuristics, policy};
use shengji_core::game_state::initialize_phase::InitializePhase;
use shengji_core::game_state::GameState;
use shengji_core::interactive::Action;
use shengji_mechanics::types::{Card, PlayerID};

struct Row {
    group: u64,
    features: [f32; FEATURE_DIM],
    label: u8,
    actor: PlayerID,
    value: f32,
}

fn main() {
    if std::env::var("SHENGJI_BOT_BUDGET_MS").is_err() {
        std::env::set_var("SHENGJI_BOT_BUDGET_MS", "200");
    }
    let tier = match std::env::var("EXIT_TIER").ok().as_deref() {
        Some("enoch") => BotDifficulty::Enoch,
        _ => BotDifficulty::Expert,
    };
    let games: usize = env_parse("GEN_GAMES", 400);
    let seed: u64 = env_parse("GEN_SEED", 0xE_417);
    let out_path = std::env::var("GEN_OUT").unwrap_or_else(|_| "training/exit_data.csv".to_string());
    eprintln!(
        "ExIt gen: tier={tier:?} budget={}ms games={games} seed={seed} net={}",
        std::env::var("SHENGJI_BOT_BUDGET_MS").unwrap_or_default(),
        std::env::var("SHENGJI_EXPERT_MODEL_PATH").unwrap_or_else(|_| "<embedded>".into()),
    );

    let start = std::time::Instant::now();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut group_counter: u64 = 0;
    let mut rows: Vec<Row> = Vec::new();
    let mut decisions = 0usize;
    for g in 0..games {
        play_one_hand(&mut rng, &mut group_counter, &mut rows, &mut decisions, tier);
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
    let mut w = BufWriter::new(File::create(&out_path).expect("create CSV"));
    write!(w, "group").unwrap();
    for i in 0..FEATURE_DIM {
        write!(w, ",f{i}").unwrap();
    }
    writeln!(w, ",label,value").unwrap();
    for r in &rows {
        write!(w, "{}", r.group).unwrap();
        for x in &r.features {
            write!(w, ",{x:.6}").unwrap();
        }
        writeln!(w, ",{},{:.6}", r.label, r.value).unwrap();
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

fn play_one_hand(
    rng: &mut StdRng,
    group_counter: &mut u64,
    rows: &mut Vec<Row>,
    decisions: &mut usize,
    tier: BotDifficulty,
) {
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
                match policy::select_action(&view, landlord, tier).ok().flatten() {
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
                        r.value = (oriented / VALUE_NORM).clamp(-1.0, 1.0) as f32;
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
                        // Honest redacted view: the search reads only this; the
                        // recorded features are computed from it.
                        let view_state = GameState::Play(s.clone()).for_player(actor);
                        let view = match &view_state {
                            GameState::Play(p) => p,
                            _ => return,
                        };
                        let leading = view.trick().played_cards().is_empty();
                        let candidates: Vec<Vec<Card>> = if leading {
                            heuristics::lead_candidates(view, actor)
                        } else {
                            heuristics::follow_candidates(view, actor)
                        };
                        // The honest search's pick — the ExIt policy-improvement
                        // target AND the move that advances the game (one search).
                        let chosen = match policy::select_action(&view_state, actor, tier)
                            .ok()
                            .flatten()
                        {
                            Some(Action::PlayCards(c)) => c,
                            _ => return,
                        };
                        // Only emit a learning signal when there's a real choice.
                        if candidates.len() >= 2 {
                            if let Some(idx) =
                                candidates.iter().position(|c| same_multiset(c, &chosen))
                            {
                                let group = *group_counter;
                                *group_counter += 1;
                                game_decisions += 1;
                                for (i, cand) in candidates.iter().enumerate() {
                                    game_rows.push(Row {
                                        group,
                                        features: candidate_features(view, actor, cand),
                                        label: if i == idx { 1 } else { 0 },
                                        actor,
                                        value: 0.0,
                                    });
                                }
                            }
                        }
                        if s.play_cards(actor, &chosen).is_err() {
                            return;
                        }
                    }
                }
            }
        }
    }
}

fn same_multiset(a: &[Card], b: &[Card]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    Card::count(a.iter().copied()) == Card::count(b.iter().copied())
}
