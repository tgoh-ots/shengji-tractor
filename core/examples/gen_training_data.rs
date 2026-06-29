//! Training-data exporter for the Expert (learned-net) tier.
//!
//! Plays many all-bot self-play Tractor hands and, at every PLAY-phase decision,
//! emits one CSV row PER LEGAL CANDIDATE move:
//!
//! * a fixed-length **HONEST** feature vector for `(state, candidate)` computed
//!   from the acting seat's REDACTED view (`bot::expert::candidate_features`),
//!   identical to what the Rust inference path will see at serving time; plus
//! * a binary label = 1 if this candidate is the move the STRONG teacher (the
//!   `Omniscient` perfect-information bot) picked from that same position, else
//!   0; plus
//! * a `group` id so the trainer can apply a softmax/cross-entropy over the
//!   candidates of a single decision (exactly one candidate per group is
//!   labelled 1).
//!
//! The teacher chooses with PERFECT INFORMATION (it sees every hand), but the
//! FEATURES are HONEST-only — so the net learns to approximate perfect-info play
//! from the honest observation. This is the behavioral-cloning / distillation
//! signal.
//!
//! Output: `training/data.csv` (header + rows). Run with:
//!
//!     SHENGJI_BOT_BUDGET_MS=10 GEN_GAMES=300 \
//!         cargo run --release --example gen_training_data
//!
//! Env knobs:
//!   GEN_GAMES   number of self-play hands to export (default 200)
//!   GEN_OUT     output CSV path (default training/data.csv)
//!   SHENGJI_BOT_BUDGET_MS  teacher search budget per decision (default small)

use std::fs::File;
use std::io::{BufWriter, Write};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use shengji_core::bot::expert::{candidate_features, FEATURE_DIM};
use shengji_core::bot::BotDifficulty;
use shengji_core::bot::{heuristics, policy};
use shengji_core::game_state::initialize_phase::InitializePhase;
use shengji_core::game_state::play_phase::PlayPhase;
use shengji_core::game_state::GameState;
use shengji_core::interactive::Action;
use shengji_mechanics::types::{Card, PlayerID};

/// The teacher tier whose (perfect-information) choice provides the label.
const TEACHER: BotDifficulty = BotDifficulty::Omniscient;

/// The behaviour tier used to actually ADVANCE the game between recorded
/// decisions. We mix a noisy tier so the exported states are diverse (not just
/// the on-policy trajectory of one strong bot). The teacher's pick is still
/// recorded as the label at every decision regardless of who is acting.
const BEHAVIOUR: BotDifficulty = BotDifficulty::Easy;

struct Row {
    group: u64,
    features: [f32; FEATURE_DIM],
    label: u8,
}

fn main() {
    // Keep the teacher's perfect-info search fast by default (it runs at EVERY
    // recorded decision); override with SHENGJI_BOT_BUDGET_MS for a stronger
    // (slower) teacher.
    if std::env::var("SHENGJI_BOT_BUDGET_MS").is_err() {
        std::env::set_var("SHENGJI_BOT_BUDGET_MS", "8");
    }
    let games: usize = std::env::var("GEN_GAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);
    let out_path = std::env::var("GEN_OUT").unwrap_or_else(|_| "training/data.csv".to_string());

    let start = std::time::Instant::now();
    let mut rng = StdRng::seed_from_u64(0xD157111);
    let mut group_counter: u64 = 0;
    let mut rows: Vec<Row> = Vec::new();
    let mut decisions = 0usize;

    for g in 0..games {
        play_one_hand_collecting(&mut rng, &mut group_counter, &mut rows, &mut decisions);
        if g % 25 == 0 {
            eprintln!(
                "  game {g}/{games}: {decisions} decisions, {} rows so far ({:.0}s)",
                rows.len(),
                start.elapsed().as_secs_f64()
            );
        }
    }

    // Write CSV.
    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let file = File::create(&out_path).expect("create output CSV");
    let mut w = BufWriter::new(file);
    // Header: group, f0..f{D-1}, label.
    write!(w, "group").unwrap();
    for i in 0..FEATURE_DIM {
        write!(w, ",f{i}").unwrap();
    }
    writeln!(w, ",label").unwrap();
    for r in &rows {
        write!(w, "{}", r.group).unwrap();
        for x in &r.features {
            write!(w, ",{x:.6}").unwrap();
        }
        writeln!(w, ",{}", r.label).unwrap();
    }
    w.flush().unwrap();

    eprintln!(
        "Wrote {} rows across {} decisions ({} games) to {} in {:.1}s",
        rows.len(),
        decisions,
        games,
        out_path,
        start.elapsed().as_secs_f64()
    );
}

/// Drive one all-bot hand, recording per-candidate rows at every play decision.
fn play_one_hand_collecting(
    rng: &mut StdRng,
    group_counter: &mut u64,
    rows: &mut Vec<Row>,
    decisions: &mut usize,
) {
    let n = 4;
    let mut init = InitializePhase::new();
    let mut seats: Vec<PlayerID> = vec![];
    for i in 0..n {
        seats.push(init.add_player(format!("seat{i}")).unwrap().0);
    }
    init.set_num_decks(Some(n / 2)).ok();

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
                    // Randomize landlord seat for deal diversity.
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
                match policy::select_action(&view, landlord, BEHAVIOUR)
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
                    return;
                }
                match s.trick().next_player() {
                    None => {
                        if s.finish_trick().is_err() {
                            return;
                        }
                    }
                    Some(actor) => {
                        // Record the decision (per-candidate rows) BEFORE applying
                        // a move to advance the game.
                        record_decision(s, actor, group_counter, rows, decisions);

                        // Advance with the (noisy) behaviour policy from the
                        // honest redacted view.
                        let view = GameState::Play(s.clone()).for_player(actor);
                        let cards = match policy::select_action(&view, actor, BEHAVIOUR)
                            .ok()
                            .flatten()
                        {
                            Some(Action::PlayCards(c)) => c,
                            _ => return,
                        };
                        if s.play_cards(actor, &cards).is_err() {
                            return;
                        }
                    }
                }
            }
        }
    }
}

/// Emit one row per legal candidate for `actor`'s current play decision. The
/// FEATURES come from the redacted view; the LABEL comes from the Omniscient
/// teacher's perfect-information choice on the SAME position.
fn record_decision(
    full: &PlayPhase,
    actor: PlayerID,
    group_counter: &mut u64,
    rows: &mut Vec<Row>,
    decisions: &mut usize,
) {
    // Honest, redacted view: feature computation must only ever see this.
    let view_state = GameState::Play(full.clone()).for_player(actor);
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
    // A degenerate decision with 0 or 1 candidate carries no learning signal.
    if candidates.len() < 2 {
        return;
    }

    // The teacher's pick, computed with PERFECT INFORMATION on the true world
    // `full` (every seat's real cards). We mirror production's Omniscient honesty
    // bypass: hand the teacher the full state.
    let teacher_state = GameState::Play(full.clone());
    let teacher_pick = match policy::select_action(&teacher_state, actor, TEACHER)
        .ok()
        .flatten()
    {
        Some(Action::PlayCards(c)) => c,
        _ => return,
    };

    // Match the teacher's pick to one of our (honest) candidates by multiset of
    // cards. If the teacher chose something outside the candidate set (rare —
    // both use the same generators), skip this decision to keep labels clean.
    let teacher_idx = candidates
        .iter()
        .position(|c| same_multiset(c, &teacher_pick));
    let teacher_idx = match teacher_idx {
        Some(i) => i,
        None => return,
    };

    let group = *group_counter;
    *group_counter += 1;
    *decisions += 1;
    for (i, cand) in candidates.iter().enumerate() {
        rows.push(Row {
            group,
            features: candidate_features(view, actor, cand),
            label: if i == teacher_idx { 1 } else { 0 },
        });
    }
}

/// Whether two card slices are equal as multisets (order-independent).
fn same_multiset(a: &[Card], b: &[Card]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let ca = Card::count(a.iter().copied());
    let cb = Card::count(b.iter().copied());
    ca == cb
}
