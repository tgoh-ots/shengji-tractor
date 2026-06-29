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
//! Each row also carries a `value` column: the realized terminal point-differential
//! (oriented for the acting team, normalized by `expert::VALUE_NORM`), back-filled
//! at game end — the regression target for the learned VALUE head.
//!
//! Env knobs:
//!   GEN_GAMES   number of self-play hands to export (default 200)
//!   GEN_OUT     output CSV path (default training/data.csv)
//!   GEN_SEED    deal-RNG seed (default 0xD157111). Distinct seeds give disjoint
//!               deals, so a run can be SHARDED across processes for parallelism.
//!   GEN_TEACHER_BUDGET_MS  teacher (Omniscient) perfect-info search budget per
//!               decision, in ms (default 400). This directly sets LABEL QUALITY:
//!               an 8ms label is near-noise. Applied via SHENGJI_BOT_BUDGET_MS.
//!   GEN_BEHAVIOUR  which policy ADVANCES the game = the recorded STATE
//!               DISTRIBUTION (DAgger): easy | expert | enoch | mix (default easy).
//!   GEN_MIX_SEARCH_FRAC  for GEN_BEHAVIOUR=mix, fraction of GAMES advanced by the
//!               search tier (default 0.5; the rest by Easy). A search behaviour
//!               shares the teacher's budget and is MUCH slower (hours, not minutes).
//!   SHENGJI_BOT_BUDGET_MS  if set explicitly, OVERRIDES GEN_TEACHER_BUDGET_MS
//!               (back-compat). Caps BOTH the teacher and any search behaviour.

use std::fs::File;
use std::io::{BufWriter, Write};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use shengji_core::bot::expert::{candidate_features, FEATURE_DIM, VALUE_NORM};
use shengji_core::bot::BotDifficulty;
use shengji_core::bot::{heuristics, policy};
use shengji_core::game_state::initialize_phase::InitializePhase;
use shengji_core::game_state::play_phase::PlayPhase;
use shengji_core::game_state::GameState;
use shengji_core::interactive::Action;
use shengji_mechanics::types::{Card, PlayerID};

/// The teacher tier whose (perfect-information) choice provides the label.
const TEACHER: BotDifficulty = BotDifficulty::Omniscient;

/// Which policy ADVANCES the game between recorded decisions — i.e. which state
/// DISTRIBUTION the exported data covers. The teacher's pick is recorded as the
/// label at every decision regardless of who is acting, so changing this is a
/// DAgger-style move: relabel-with-the-teacher on a new state distribution.
///
/// The legacy default (`Easy`) means the net is trained on states the WEAKEST
/// tier reaches but served as the prior of a strong determinized search — a
/// train/serve covariate shift. `GEN_BEHAVIOUR` lets a DAgger pass advance with
/// the actual search tier (or a mix) so recorded states match the serving
/// distribution; it ALSO sharpens the VALUE target (the realized margin is now
/// under strong play, not Easy play).
#[derive(Clone, Copy)]
enum BehaviourMode {
    /// Advance every game with `Easy` (legacy default; rng-stream-identical to the
    /// pre-DAgger exporter, so old policy-only data reproduces byte-for-byte).
    Easy,
    /// Advance every game with a single search tier (`Expert` / `Enoch`).
    Tier(BotDifficulty),
    /// Advance each GAME with `tier` with probability `frac`, else `Easy`. Mixing
    /// per-GAME (not per-decision) keeps each trajectory coherent so the value
    /// target is a clean single-policy playout, while spanning BOTH the
    /// strong-search and the noisy-Easy state distributions across the corpus.
    Mix { tier: BotDifficulty, frac: f64 },
}

impl BehaviourMode {
    /// Parse `GEN_BEHAVIOUR` (easy | expert | enoch | mix; default easy) and, for
    /// `mix`, `GEN_MIX_SEARCH_FRAC` (default 0.5).
    fn from_env() -> Self {
        match std::env::var("GEN_BEHAVIOUR").ok().as_deref() {
            Some("expert") => BehaviourMode::Tier(BotDifficulty::Expert),
            Some("enoch") => BehaviourMode::Tier(BotDifficulty::Enoch),
            Some("mix") => {
                let frac = std::env::var("GEN_MIX_SEARCH_FRAC")
                    .ok()
                    .and_then(|v| v.parse::<f64>().ok())
                    .unwrap_or(0.5)
                    .clamp(0.0, 1.0);
                BehaviourMode::Mix {
                    tier: BotDifficulty::Expert,
                    frac,
                }
            }
            _ => BehaviourMode::Easy,
        }
    }

    fn label(self) -> String {
        match self {
            BehaviourMode::Easy => "Easy".to_string(),
            BehaviourMode::Tier(d) => format!("{d:?}"),
            BehaviourMode::Mix { tier, frac } => {
                format!("mix({tier:?} {frac:.0}% / Easy)", frac = frac * 100.0)
            }
        }
    }

    /// The tier that drives THIS game, chosen once per game. For `Easy`/`Tier` it
    /// does NOT consume `rng` (so the default-Easy rng stream — and thus the deal
    /// sequence — is unchanged); `Mix` consumes one `gen_bool` per game.
    fn pick(self, rng: &mut StdRng) -> BotDifficulty {
        match self {
            BehaviourMode::Easy => BotDifficulty::Easy,
            BehaviourMode::Tier(d) => d,
            BehaviourMode::Mix { tier, frac } => {
                if rng.gen_bool(frac) {
                    tier
                } else {
                    BotDifficulty::Easy
                }
            }
        }
    }
}

struct Row {
    group: u64,
    features: [f32; FEATURE_DIM],
    label: u8,
    /// The acting seat for this decision. Transient: used only to ORIENT the
    /// terminal value target at game end; NOT written to the CSV.
    actor: PlayerID,
    /// The VALUE target: the realized terminal point-differential oriented for
    /// `actor`'s team, normalized by `VALUE_NORM` and clamped to [-1, 1]. Filled
    /// in at game completion (back-filled), since the outcome is only known then.
    value: f32,
}

/// Counts of decisions that were SKIPPED (no row emitted) and why. A quick
/// fact-check on label coverage: in particular `teacher_outside_candidates`
/// should be ~0, because the teacher and the candidate generator both go through
/// `heuristics::{lead,follow}_candidates` — if it is large, the assumption that
/// the teacher always picks a generated candidate is wrong.
#[derive(Default)]
struct DropStats {
    /// Fewer than 2 legal candidates: no learning signal (a forced move).
    degenerate: usize,
    /// The teacher produced no `PlayCards` action (should not happen in Play).
    teacher_no_play: usize,
    /// The teacher's pick was not among the generated candidates (expected ~0).
    teacher_outside_candidates: usize,
}

fn main() {
    // The teacher (Omniscient) runs a perfect-information search at EVERY recorded
    // decision to produce the label, so its budget directly sets LABEL QUALITY: an
    // 8ms label is near-noise. Default to a meaningfully stronger teacher, tunable
    // independently via GEN_TEACHER_BUDGET_MS. We apply it through the shared
    // `SHENGJI_BOT_BUDGET_MS` knob (the only search in this all-Easy-driven process
    // is the teacher's), but an explicit SHENGJI_BOT_BUDGET_MS still wins.
    let teacher_budget_ms: u64 = std::env::var("GEN_TEACHER_BUDGET_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(400);
    if std::env::var("SHENGJI_BOT_BUDGET_MS").is_err() {
        std::env::set_var("SHENGJI_BOT_BUDGET_MS", teacher_budget_ms.to_string());
    }
    eprintln!(
        "teacher budget: SHENGJI_BOT_BUDGET_MS={} ms (GEN_TEACHER_BUDGET_MS default {})",
        std::env::var("SHENGJI_BOT_BUDGET_MS").unwrap_or_default(),
        teacher_budget_ms,
    );
    // DAgger knob: which policy advances the game (= the recorded state
    // distribution). Default Easy (legacy). A search behaviour shares the teacher's
    // SHENGJI_BOT_BUDGET_MS and makes generation MUCH slower (a search per move on
    // top of the teacher search per decision) — expect hours, not minutes.
    let behaviour = BehaviourMode::from_env();
    eprintln!(
        "behaviour: GEN_BEHAVIOUR={} (advances the game; teacher still labels)",
        behaviour.label()
    );
    let games: usize = std::env::var("GEN_GAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);
    let out_path = std::env::var("GEN_OUT").unwrap_or_else(|_| "training/data.csv".to_string());

    // Deal-RNG seed. Distinct seeds generate DISJOINT deal sequences, so a long
    // run can be SHARDED across processes (GEN_SEED=base+i per shard) for parallel,
    // resumable generation. Default keeps the original fixed seed (back-compat).
    let gen_seed: u64 = std::env::var("GEN_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0xD157111);
    let start = std::time::Instant::now();
    let mut rng = StdRng::seed_from_u64(gen_seed);
    let mut group_counter: u64 = 0;
    let mut rows: Vec<Row> = Vec::new();
    let mut decisions = 0usize;
    let mut drops = DropStats::default();

    for g in 0..games {
        play_one_hand_collecting(
            &mut rng,
            &mut group_counter,
            &mut rows,
            &mut decisions,
            &mut drops,
            behaviour,
        );
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
    // Header: group, f0..f{D-1}, label, value. The trailing `value` column is the
    // (normalized, oriented) realized terminal margin for the learned VALUE head;
    // the trainer treats it as optional (back-compat with policy-only CSVs).
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
        "Wrote {} rows across {} decisions ({} games) to {} in {:.1}s",
        rows.len(),
        decisions,
        games,
        out_path,
        start.elapsed().as_secs_f64()
    );
    let total_seen =
        decisions + drops.degenerate + drops.teacher_no_play + drops.teacher_outside_candidates;
    eprintln!(
        "Skipped decisions: {} degenerate(<2 cands), {} teacher-no-play, {} teacher-outside-candidates \
         (of {} positions seen). teacher-outside should be ~0.",
        drops.degenerate, drops.teacher_no_play, drops.teacher_outside_candidates, total_seen,
    );
}

/// Drive one all-bot hand, recording per-candidate rows at every play decision.
fn play_one_hand_collecting(
    rng: &mut StdRng,
    group_counter: &mut u64,
    rows: &mut Vec<Row>,
    decisions: &mut usize,
    drops: &mut DropStats,
    behaviour: BehaviourMode,
) {
    // The tier that advances THIS game (chosen once, before the deal RNG, so the
    // continuation — and thus the value target — is a coherent single-policy
    // playout). `Easy`/`Tier` don't consume rng here, keeping the default deal
    // sequence unchanged.
    let game_behaviour = behaviour.pick(rng);
    let n = 4;
    let mut init = InitializePhase::new();
    let mut seats: Vec<PlayerID> = vec![];
    for i in 0..n {
        seats.push(init.add_player(format!("seat{i}")).unwrap().0);
    }
    init.set_num_decks(Some(n / 2)).ok();

    // Buffer THIS game's rows so the value target (the realized terminal margin,
    // known only at game end) can be back-filled before flushing to the global
    // `rows`. A game that errors out before completion contributes NOTHING (its
    // buffered rows are dropped), so every emitted value target is a real outcome.
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
                match policy::select_action(&view, landlord, game_behaviour)
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
                    // Back-fill the value target from the realized terminal margin
                    // (oriented per decision's acting team) and flush this game's
                    // rows to the global buffer.
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
                        // Record the decision (per-candidate rows) BEFORE applying
                        // a move to advance the game (into the per-game buffer).
                        record_decision(
                            s,
                            actor,
                            group_counter,
                            &mut game_rows,
                            &mut game_decisions,
                            drops,
                        );

                        // Advance with this game's behaviour policy from the
                        // honest redacted view.
                        let view = GameState::Play(s.clone()).for_player(actor);
                        let cards = match policy::select_action(&view, actor, game_behaviour)
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

/// Emit one row per legal candidate for `actor`'s current play decision into the
/// PER-GAME buffer `rows`. The FEATURES come from the redacted view; the LABEL
/// comes from the Omniscient teacher's perfect-information choice on the SAME
/// position; the VALUE target is left as a placeholder (`0.0`) and back-filled
/// from the realized terminal margin when the game finishes.
fn record_decision(
    full: &PlayPhase,
    actor: PlayerID,
    group_counter: &mut u64,
    rows: &mut Vec<Row>,
    decisions: &mut usize,
    drops: &mut DropStats,
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
        drops.degenerate += 1;
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
        _ => {
            drops.teacher_no_play += 1;
            return;
        }
    };

    // Match the teacher's pick to one of our (honest) candidates by multiset of
    // cards. If the teacher chose something outside the candidate set (rare —
    // both use the same generators), skip this decision to keep labels clean.
    let teacher_idx = candidates
        .iter()
        .position(|c| same_multiset(c, &teacher_pick));
    let teacher_idx = match teacher_idx {
        Some(i) => i,
        None => {
            drops.teacher_outside_candidates += 1;
            return;
        }
    };

    let group = *group_counter;
    *group_counter += 1;
    *decisions += 1;
    for (i, cand) in candidates.iter().enumerate() {
        rows.push(Row {
            group,
            features: candidate_features(view, actor, cand),
            label: if i == teacher_idx { 1 } else { 0 },
            actor,
            value: 0.0, // back-filled at game end from the realized terminal margin
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
