//! Seeded training-data exporter for the Expert policy baseline and staged
//! honest action-value learning.
//!
//! Schema v3 retains the original one-hot Omniscient-teacher label, but makes
//! the two value semantics explicit:
//!
//! * v_target is V(o): signed terminal level utility of the behaviour trajectory from
//!   this honest observation. It is constant within a decision and must be fed
//!   to a state-only value head.
//! * q_target is Q(o,a): signed terminal level utility after forcing this candidate in
//!   the same true deal, then continuing from honest per-seat observations. Only
//!   a configurable subset of candidates is evaluated; missing targets are blank.
//!
//! Counterfactual candidates share the exact same compatible world (the seeded
//! true deal), which gives a low-variance within-decision comparison. Across many
//! deals, a network that consumes only honest features learns an expected return
//! rather than a hidden-world-specific oracle argmax.
//!
//! Important env knobs:
//! * `GEN_GAMES` (default 200), `GEN_OUT` (default `training/data.csv`),
//!   `GEN_SEED` (default `0xD157111`);
//! * `GEN_BEHAVIOUR=easy|expert|enoch|mix` and `GEN_MIX_SEARCH_FRAC`;
//! * `GEN_SEAT_BEHAVIOURS` optionally overrides it with exactly four comma-
//!   separated values, one per seat (for partner/opponent league diversity);
//! * `GEN_BEHAVIOUR_BUDGET_MS` is the behavior policy's final per-call budget;
//! * `GEN_TEACHER_BUDGET_MS` (default 400; policy-label quality);
//! * `GEN_Q_CANDIDATES` (default 2, `0` disables Q generation, `all` evaluates
//!   every candidate) and `GEN_Q_ROLLOUT_BEHAVIOUR=easy|expert|enoch`;
//! * `GEN_Q_ROLLOUT_BUDGET_MS` is independent from both budgets above;
//! * `GEN_MANIFEST` overrides the default `<GEN_OUT>.manifest.json` sidecar.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;

use shengji_core::bot::expert::{
    candidate_features_v2, LEVEL_UTILITY_NORM, TRAINING_FEATURE_DIM,
    TRAINING_FEATURE_SCHEMA_VERSION, VALUE_NORM,
};
use shengji_core::bot::harness::seeded_draw_phase;
use shengji_core::bot::{heuristics, policy, BotDifficulty};
use shengji_core::game_state::play_phase::PlayPhase;
use shengji_core::game_state::GameState;
use shengji_core::interactive::Action;
use shengji_mechanics::deck::Deck;
use shengji_mechanics::types::{Card, PlayerID, FULL_DECK};

const DATASET_SCHEMA_VERSION: u32 = 3;
const TEACHER: BotDifficulty = BotDifficulty::Omniscient;
const GAME_CONFIG: &str = "tractor-4p-2deck";

#[derive(Clone, Copy)]
enum BehaviourMode {
    Easy,
    Tier(BotDifficulty),
    Mix { tier: BotDifficulty, frac: f64 },
}

impl BehaviourMode {
    fn from_env() -> Self {
        Self::parse(
            std::env::var("GEN_BEHAVIOUR")
                .ok()
                .as_deref()
                .unwrap_or("easy"),
        )
    }

    fn parse(value: &str) -> Self {
        match value {
            "expert" => Self::Tier(BotDifficulty::Expert),
            "enoch" => Self::Tier(BotDifficulty::Enoch),
            "grandmaster" => Self::Tier(BotDifficulty::Grandmaster),
            "mix" => Self::Mix {
                tier: BotDifficulty::Expert,
                frac: env_f64("GEN_MIX_SEARCH_FRAC", 0.5).clamp(0.0, 1.0),
            },
            "easy" => Self::Easy,
            _ => panic!(
                "unsupported behaviour {:?}; expected easy, expert, enoch, grandmaster, or mix",
                value
            ),
        }
    }

    fn label(self) -> String {
        match self {
            Self::Easy => "easy".to_string(),
            Self::Tier(tier) => tier.as_str().to_ascii_lowercase(),
            Self::Mix { tier, frac } => {
                format!("mix-{}-{frac:.3}", tier.as_str().to_ascii_lowercase())
            }
        }
    }

    fn pick(self, rng: &mut StdRng) -> BotDifficulty {
        match self {
            Self::Easy => BotDifficulty::Easy,
            Self::Tier(tier) => tier,
            Self::Mix { tier, frac } => {
                if rng.gen_bool(frac) {
                    tier
                } else {
                    BotDifficulty::Easy
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
struct BehaviourPlan {
    seats: [BehaviourMode; 4],
    heterogeneous: bool,
}

impl BehaviourPlan {
    fn from_env() -> Self {
        if let Ok(value) = std::env::var("GEN_SEAT_BEHAVIOURS") {
            let parsed = value
                .split(',')
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(BehaviourMode::parse)
                .collect::<Vec<_>>();
            assert_eq!(
                parsed.len(),
                4,
                "GEN_SEAT_BEHAVIOURS must contain exactly four entries, got {}",
                parsed.len()
            );
            let seats = [parsed[0], parsed[1], parsed[2], parsed[3]];
            Self {
                seats,
                heterogeneous: true,
            }
        } else {
            Self {
                seats: [BehaviourMode::from_env(); 4],
                heterogeneous: false,
            }
        }
    }

    fn label(&self) -> String {
        if self.heterogeneous {
            format!(
                "seats[{}]",
                self.seats
                    .iter()
                    .map(|mode| mode.label())
                    .collect::<Vec<_>>()
                    .join(",")
            )
        } else {
            self.seats[0].label()
        }
    }

    fn pick(&self, rng: &mut StdRng) -> [BotDifficulty; 4] {
        self.seats.map(|mode| mode.pick(rng))
    }
}

fn behaviour_for(
    seats: &[PlayerID],
    behaviours: &[BotDifficulty; 4],
    player: PlayerID,
) -> Option<BotDifficulty> {
    seats
        .iter()
        .position(|seat| *seat == player)
        .map(|index| behaviours[index])
}

#[derive(Clone, Copy)]
struct QConfig {
    candidate_cap: Option<usize>,
    rollout: BotDifficulty,
    budget_ms: u64,
}

impl QConfig {
    fn from_env() -> Self {
        let candidate_cap = match std::env::var("GEN_Q_CANDIDATES")
            .unwrap_or_else(|_| "2".to_string())
            .as_str()
        {
            "all" => None,
            value => Some(value.parse::<usize>().unwrap_or(2)),
        };
        let rollout = match std::env::var("GEN_Q_ROLLOUT_BEHAVIOUR")
            .unwrap_or_else(|_| "easy".to_string())
            .as_str()
        {
            "expert" => BotDifficulty::Expert,
            "enoch" => BotDifficulty::Enoch,
            _ => BotDifficulty::Easy,
        };
        Self {
            candidate_cap,
            rollout,
            budget_ms: env_u64("GEN_Q_ROLLOUT_BUDGET_MS", 20).max(1),
        }
    }

    fn enabled(self) -> bool {
        self.candidate_cap != Some(0)
    }

    fn label(self) -> String {
        self.rollout.as_str().to_ascii_lowercase()
    }

    fn cap_label(self) -> String {
        self.candidate_cap
            .map(|cap| cap.to_string())
            .unwrap_or_else(|| "all".to_string())
    }
}

struct Row {
    run_id: String,
    game_id: String,
    game_seed: u64,
    decision_id: u32,
    candidate_id: usize,
    group: String,
    actor: PlayerID,
    actor_team: &'static str,
    behaviour: String,
    rollout_behaviour: String,
    action: String,
    features: [f32; TRAINING_FEATURE_DIM],
    label: u8,
    behaviour_label: u8,
    v_target: f32,
    v_attacker_points: isize,
    v_score_bucket: isize,
    v_win_target: f32,
    v_kitty_target: Option<f32>,
    q_target: Option<f32>,
    q_attacker_points: Option<isize>,
    q_score_bucket: Option<isize>,
    q_win_target: Option<f32>,
    q_kitty_target: Option<f32>,
    q_samples: u32,
}

#[derive(Clone, Debug)]
struct TerminalOutcome {
    attacker_points: isize,
    score_bucket: isize,
    landlord_won: bool,
    landlord_delta: usize,
    attacker_delta: usize,
    landlord_team: Vec<PlayerID>,
    last_trick_winner: Option<PlayerID>,
}

#[derive(Default, Serialize)]
struct DropStats {
    degenerate: usize,
    teacher_no_play: usize,
    teacher_outside_candidates: usize,
    behaviour_no_play: usize,
    behaviour_outside_candidates: usize,
    q_rollout_failed: usize,
    game_failed: usize,
}

#[derive(Serialize)]
struct DatasetManifest {
    manifest_version: u32,
    dataset_schema_version: u32,
    feature_schema_version: u32,
    feature_dim: usize,
    game_config: &'static str,
    output: String,
    gen_seed: u64,
    games_requested: usize,
    games_completed: usize,
    rows: usize,
    decisions: usize,
    q_rows: usize,
    behaviour: String,
    seat_behaviours: [String; 4],
    mix_search_fraction: f64,
    teacher: &'static str,
    teacher_budget_ms: u64,
    behaviour_budget_ms: u64,
    q_candidates: String,
    q_rollout_behaviour: String,
    q_rollout_budget_ms: u64,
    point_feature_norm: f64,
    level_utility_norm: f32,
    target_contract: &'static str,
    auxiliary_targets: [&'static str; 4],
    drops: DropStats,
}

fn main() {
    let teacher_budget_ms = env_u64("GEN_TEACHER_BUDGET_MS", 400).max(1);
    let behaviour_budget_ms = env_u64("GEN_BEHAVIOUR_BUDGET_MS", 80).max(1);
    let games = env_usize("GEN_GAMES", 200);
    let gen_seed = env_u64("GEN_SEED", 0xD157111);
    let out_path = std::env::var("GEN_OUT").unwrap_or_else(|_| "training/data.csv".to_string());
    let manifest_path =
        std::env::var("GEN_MANIFEST").unwrap_or_else(|_| format!("{out_path}.manifest.json"));
    let behaviour = BehaviourPlan::from_env();
    let q_config = QConfig::from_env();

    eprintln!(
        "seeded schema-v{} generation: games={} seed={} behaviour={}@{}ms teacher={}ms q_candidates={} q_rollout={}@{}ms",
        DATASET_SCHEMA_VERSION,
        games,
        gen_seed,
        behaviour.label(),
        behaviour_budget_ms,
        teacher_budget_ms,
        q_config.cap_label(),
        q_config.label(),
        q_config.budget_ms,
    );

    ensure_parent(&out_path);
    ensure_parent(&manifest_path);
    let file = File::create(&out_path).expect("create output CSV");
    let mut writer = BufWriter::new(file);
    write_header(&mut writer);

    let start = std::time::Instant::now();
    let mut drops = DropStats::default();
    let mut completed_games = 0usize;
    let mut total_rows = 0usize;
    let mut total_decisions = 0usize;
    let mut q_rows = 0usize;

    for game_index in 0..games {
        let game_seed = derive_game_seed(gen_seed, game_index as u64);
        match play_one_hand_collecting(
            gen_seed,
            game_index,
            game_seed,
            behaviour,
            behaviour_budget_ms,
            teacher_budget_ms,
            q_config,
            &mut drops,
        ) {
            Some(rows) => {
                if let Err(error) = validate_game_rows(&rows) {
                    eprintln!("  rejecting invalid game {game_index}: {error}");
                    drops.game_failed += 1;
                    continue;
                }
                completed_games += 1;
                // decision_id also advances across observations that are
                // deliberately dropped (for example, when the teacher action
                // is outside the bounded candidate set). The manifest count
                // describes emitted listwise groups, not attempted IDs.
                total_decisions += rows
                    .iter()
                    .map(|row| row.group.as_str())
                    .collect::<BTreeSet<_>>()
                    .len();
                q_rows += rows.iter().filter(|row| row.q_target.is_some()).count();
                total_rows += rows.len();
                for row in rows {
                    write_row(&mut writer, &row);
                }
            }
            None => drops.game_failed += 1,
        }
        if game_index % 25 == 0 {
            eprintln!(
                "  game {game_index}/{games}: completed={completed_games} decisions={total_decisions} rows={total_rows} q_rows={q_rows} ({:.1}s)",
                start.elapsed().as_secs_f64()
            );
        }
    }
    writer.flush().expect("flush dataset");

    let manifest = DatasetManifest {
        manifest_version: 1,
        dataset_schema_version: DATASET_SCHEMA_VERSION,
        feature_schema_version: TRAINING_FEATURE_SCHEMA_VERSION,
        feature_dim: TRAINING_FEATURE_DIM,
        game_config: GAME_CONFIG,
        output: out_path.clone(),
        gen_seed,
        games_requested: games,
        games_completed: completed_games,
        rows: total_rows,
        decisions: total_decisions,
        q_rows,
        behaviour: behaviour.label(),
        seat_behaviours: behaviour.seats.map(BehaviourMode::label),
        mix_search_fraction: env_f64("GEN_MIX_SEARCH_FRAC", 0.5),
        teacher: "omniscient-policy-baseline",
        teacher_budget_ms,
        behaviour_budget_ms,
        q_candidates: q_config.cap_label(),
        q_rollout_behaviour: q_config.label(),
        q_rollout_budget_ms: q_config.budget_ms,
        point_feature_norm: VALUE_NORM,
        level_utility_norm: LEVEL_UTILITY_NORM,
        target_contract:
            "actor_sign*(1+winner_level_delta)/5, clamped[-1,1]; +1 iff actor team won",
        auxiliary_targets: [
            "final_attacker_points",
            "attacker_score_bucket=floor(points/step)",
            "actor_team_win",
            "actor_team_won_final_trick_kitty",
        ],
        drops,
    };
    let manifest_file = File::create(&manifest_path).expect("create dataset manifest");
    serde_json::to_writer_pretty(BufWriter::new(manifest_file), &manifest)
        .expect("write dataset manifest");

    eprintln!(
        "Wrote {total_rows} rows / {total_decisions} decisions / {completed_games} seeded games to {out_path} in {:.1}s; manifest={manifest_path}",
        start.elapsed().as_secs_f64()
    );
}

fn validate_game_rows(rows: &[Row]) -> Result<(), String> {
    let mut groups: BTreeMap<&str, Vec<&Row>> = BTreeMap::new();
    for row in rows {
        if row.features.iter().any(|value| !value.is_finite())
            || !row.v_target.is_finite()
            || row.q_target.is_some_and(|value| !value.is_finite())
        {
            return Err(format!("{} contains NaN/Inf", row.group));
        }
        if row.q_samples != u32::from(row.q_target.is_some()) {
            return Err(format!("{} has inconsistent q_samples", row.group));
        }
        groups.entry(&row.group).or_default().push(row);
    }
    for (group, candidates) in groups {
        if candidates.len() < 2
            || candidates
                .iter()
                .map(|row| row.label as usize)
                .sum::<usize>()
                != 1
            || candidates
                .iter()
                .map(|row| row.behaviour_label as usize)
                .sum::<usize>()
                != 1
        {
            return Err(format!("{group} is not a valid listwise decision"));
        }
        let v = candidates[0].v_target;
        if candidates.iter().any(|row| (row.v_target - v).abs() > 1e-6) {
            return Err(format!("{group} has candidate-dependent state V"));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn play_one_hand_collecting(
    run_seed: u64,
    game_index: usize,
    game_seed: u64,
    behaviour_plan: BehaviourPlan,
    behaviour_budget_ms: u64,
    teacher_budget_ms: u64,
    q_config: QConfig,
    drops: &mut DropStats,
) -> Option<Vec<Row>> {
    // Deal RNG is isolated from policy/mix RNG. Adding a diagnostic or changing
    // behaviour selection cannot silently change the cards for a given game_id.
    let mut deal_rng = StdRng::seed_from_u64(game_seed);
    let decks = [Deck::default(), Deck::default()];
    let draw = seeded_draw_phase(&decks, &mut deal_rng);
    let seats: Vec<PlayerID> = draw.propagated().players().iter().map(|p| p.id).collect();
    let mut policy_rng = StdRng::seed_from_u64(game_seed ^ 0xB3A4_91C2_D5E6_F708);
    let game_behaviours = behaviour_plan.pick(&mut policy_rng);
    let run_id = format!("seed-{run_seed}");
    let game_id = format!("{run_id}-game-{game_index}");

    let mut state = GameState::Draw(draw);
    let mut game_rows: Vec<Row> = Vec::new();
    let mut decision_id = 0u32;
    let mut iterations = 0usize;

    loop {
        iterations += 1;
        if iterations > 2_000_000 {
            return None;
        }
        match &mut state {
            GameState::Initialize(_) => return None,
            GameState::Draw(draw) => {
                if !draw.done_drawing() {
                    let player = draw.next_player().ok()?;
                    draw.draw_card(player).ok()?;
                } else if draw.bid_decided() {
                    let responsible = draw.next_player().ok()?;
                    state = GameState::Exchange(draw.advance(responsible).ok()?);
                } else {
                    let mut bid = false;
                    for &seat in &seats {
                        if let Some(candidate) = policy::choose_bid(
                            draw,
                            seat,
                            behaviour_for(&seats, &game_behaviours, seat)?,
                        ) {
                            if draw.bid(seat, candidate.card, candidate.count) {
                                bid = true;
                                break;
                            }
                        }
                    }
                    if !bid && draw.reveal_card().is_err() {
                        for &seat in &seats {
                            if let Some(candidate) = draw
                                .valid_bids(seat)
                                .ok()?
                                .into_iter()
                                .min_by_key(|candidate| candidate.count)
                            {
                                if draw.bid(seat, candidate.card, candidate.count) {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            GameState::Exchange(exchange) => {
                let landlord = exchange.landlord();
                let landlord_behaviour = behaviour_for(&seats, &game_behaviours, landlord)?;
                let view = GameState::Exchange(exchange.clone()).for_player(landlord);
                match policy::select_action_with_search_budget(
                    &view,
                    landlord,
                    landlord_behaviour,
                    behaviour_budget_ms,
                )
                .ok()
                .flatten()
                {
                    Some(Action::MoveCardToKitty(card)) => {
                        exchange.move_card_to_kitty(landlord, card).ok()?;
                    }
                    Some(Action::MoveCardToHand(card)) => {
                        exchange.move_card_to_hand(landlord, card).ok()?;
                    }
                    Some(Action::SetFriends(friends)) => {
                        exchange.set_friends(landlord, friends).ok()?;
                    }
                    _ => state = GameState::Play(exchange.advance(landlord).ok()?),
                }
            }
            GameState::Play(play) => {
                if play.game_finished() {
                    let terminal = terminal_outcome(play)?;
                    for row in &mut game_rows {
                        row.v_target = orient_level_utility(&terminal, row.actor);
                        row.v_attacker_points = terminal.attacker_points;
                        row.v_score_bucket = terminal.score_bucket;
                        row.v_win_target = actor_team_won(&terminal, row.actor);
                        row.v_kitty_target = actor_team_won_last_trick(&terminal, row.actor);
                    }
                    return Some(game_rows);
                }
                match play.trick().next_player() {
                    None => {
                        play.finish_trick().ok()?;
                    }
                    Some(actor) => {
                        let actor_behaviour = behaviour_for(&seats, &game_behaviours, actor)?;
                        let honest = GameState::Play(play.clone()).for_player(actor);
                        let behaviour_cards = match policy::select_action_with_search_budget(
                            &honest,
                            actor,
                            actor_behaviour,
                            behaviour_budget_ms,
                        )
                        .ok()
                        .flatten()
                        {
                            Some(Action::PlayCards(cards)) => cards,
                            _ => {
                                drops.behaviour_no_play += 1;
                                return None;
                            }
                        };
                        record_decision(
                            play,
                            actor,
                            &behaviour_cards,
                            &run_id,
                            &game_id,
                            game_seed,
                            decision_id,
                            actor_behaviour,
                            teacher_budget_ms,
                            q_config,
                            &mut game_rows,
                            drops,
                        );
                        decision_id += 1;
                        play.play_cards(actor, &behaviour_cards).ok()?;
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn record_decision(
    full: &PlayPhase,
    actor: PlayerID,
    behaviour_cards: &[Card],
    run_id: &str,
    game_id: &str,
    game_seed: u64,
    decision_id: u32,
    game_behaviour: BotDifficulty,
    teacher_budget_ms: u64,
    q_config: QConfig,
    rows: &mut Vec<Row>,
    drops: &mut DropStats,
) {
    let view_state = GameState::Play(full.clone()).for_player(actor);
    let view = match &view_state {
        GameState::Play(play) => play,
        _ => return,
    };
    let leading = view.trick().played_cards().is_empty();
    let candidates = if leading {
        heuristics::lead_candidates(view, actor)
    } else {
        heuristics::follow_candidates(view, actor)
    };
    if candidates.len() < 2 {
        drops.degenerate += 1;
        return;
    }

    let teacher_state = GameState::Play(full.clone());
    let teacher_cards = match policy::select_action_with_search_budget(
        &teacher_state,
        actor,
        TEACHER,
        teacher_budget_ms,
    )
    .ok()
    .flatten()
    {
        Some(Action::PlayCards(cards)) => cards,
        _ => {
            drops.teacher_no_play += 1;
            return;
        }
    };
    let teacher_idx = match candidates
        .iter()
        .position(|candidate| same_multiset(candidate, &teacher_cards))
    {
        Some(index) => index,
        None => {
            drops.teacher_outside_candidates += 1;
            return;
        }
    };
    let behaviour_idx = match candidates
        .iter()
        .position(|candidate| same_multiset(candidate, behaviour_cards))
    {
        Some(index) => index,
        None => {
            drops.behaviour_outside_candidates += 1;
            return;
        }
    };

    let q_indices = selected_q_indices(candidates.len(), teacher_idx, behaviour_idx, q_config);
    let actor_is_landlord = full.landlords_team().contains(&actor);
    let group = format!("{game_id}-decision-{decision_id}");
    for (candidate_id, candidate) in candidates.iter().enumerate() {
        let q_outcome = if q_indices.contains(&candidate_id) {
            match counterfactual_return(
                full,
                actor,
                candidate,
                q_config.rollout,
                q_config.budget_ms,
            ) {
                Some(outcome) => Some(outcome),
                None => {
                    drops.q_rollout_failed += 1;
                    None
                }
            }
        } else {
            None
        };
        let q_target = q_outcome
            .as_ref()
            .map(|outcome| orient_level_utility(outcome, actor));
        rows.push(Row {
            run_id: run_id.to_string(),
            game_id: game_id.to_string(),
            game_seed,
            decision_id,
            candidate_id,
            group: group.clone(),
            actor,
            actor_team: if actor_is_landlord {
                "landlord"
            } else {
                "attacker"
            },
            behaviour: game_behaviour.as_str().to_ascii_lowercase(),
            rollout_behaviour: q_config.label(),
            action: encode_action(candidate),
            features: candidate_features_v2(view, actor, candidate),
            label: u8::from(candidate_id == teacher_idx),
            behaviour_label: u8::from(candidate_id == behaviour_idx),
            v_target: 0.0,
            v_attacker_points: 0,
            v_score_bucket: 0,
            v_win_target: 0.0,
            v_kitty_target: None,
            q_target,
            q_attacker_points: q_outcome.as_ref().map(|outcome| outcome.attacker_points),
            q_score_bucket: q_outcome.as_ref().map(|outcome| outcome.score_bucket),
            q_win_target: q_outcome
                .as_ref()
                .map(|outcome| actor_team_won(outcome, actor)),
            q_kitty_target: q_outcome
                .as_ref()
                .and_then(|outcome| actor_team_won_last_trick(outcome, actor)),
            q_samples: u32::from(q_target.is_some()),
        });
    }
}

fn selected_q_indices(
    candidate_count: usize,
    teacher_idx: usize,
    behaviour_idx: usize,
    q_config: QConfig,
) -> BTreeSet<usize> {
    if !q_config.enabled() {
        return BTreeSet::new();
    }
    let cap = q_config
        .candidate_cap
        .unwrap_or(candidate_count)
        .min(candidate_count);
    let mut selected = BTreeSet::new();
    // Preserve both anchors when possible: the current behaviour action gives an
    // on-policy DMC sample; the teacher action keeps a direct comparison to the
    // old policy target. Fill remaining slots deterministically for reproducibility.
    for index in [behaviour_idx, teacher_idx] {
        if selected.len() < cap {
            selected.insert(index);
        }
    }
    for index in 0..candidate_count {
        if selected.len() >= cap {
            break;
        }
        selected.insert(index);
    }
    selected
}

fn counterfactual_return(
    full: &PlayPhase,
    actor: PlayerID,
    candidate: &[Card],
    rollout: BotDifficulty,
    rollout_budget_ms: u64,
) -> Option<TerminalOutcome> {
    let mut simulation = full.clone();
    simulation.play_cards(actor, candidate).ok()?;
    let mut iterations = 0usize;
    loop {
        iterations += 1;
        if iterations > 1_000_000 {
            return None;
        }
        if simulation.game_finished() {
            return terminal_outcome(&simulation);
        }
        match simulation.trick().next_player() {
            None => {
                simulation.finish_trick().ok()?;
            }
            Some(next) => {
                // Centralized training may inspect the true deal to obtain a
                // return, but every continuation action is chosen from that
                // player's redacted view. No rollout policy receives hidden cards.
                let view = GameState::Play(simulation.clone()).for_player(next);
                let cards = match policy::select_action_with_search_budget(
                    &view,
                    next,
                    rollout,
                    rollout_budget_ms,
                )
                .ok()
                .flatten()
                {
                    Some(Action::PlayCards(cards)) => cards,
                    _ => return None,
                };
                simulation.play_cards(next, &cards).ok()?;
            }
        }
    }
}

fn terminal_outcome(play: &PlayPhase) -> Option<TerminalOutcome> {
    let (attacker_points, _) = play.calculate_points();
    let score = play.current_game_score().ok()?;
    let step = play.bot_step_size().filter(|step| *step > 0).unwrap_or(1);
    Some(TerminalOutcome {
        attacker_points,
        score_bucket: attacker_points.div_euclid(step),
        landlord_won: score.landlord_won,
        landlord_delta: score.landlord_delta,
        attacker_delta: score.non_landlord_delta,
        landlord_team: play.landlords_team().to_vec(),
        last_trick_winner: play.last_trick().and_then(|trick| trick.winner_so_far()),
    })
}

/// Threshold-aware scalar utility. Winning is worth at least one unit even in
/// the turnover/dead-zone where the winner receives zero level increments.
fn orient_level_utility(outcome: &TerminalOutcome, actor: PlayerID) -> f32 {
    let actor_is_landlord = outcome.landlord_team.contains(&actor);
    let actor_won = actor_is_landlord == outcome.landlord_won;
    let awarded_levels = if outcome.landlord_won {
        outcome.landlord_delta
    } else {
        outcome.attacker_delta
    };
    let magnitude = (1 + awarded_levels) as f32 / LEVEL_UTILITY_NORM;
    (if actor_won { magnitude } else { -magnitude }).clamp(-1.0, 1.0)
}

fn actor_team_won(outcome: &TerminalOutcome, actor: PlayerID) -> f32 {
    let actor_is_landlord = outcome.landlord_team.contains(&actor);
    if actor_is_landlord == outcome.landlord_won {
        1.0
    } else {
        0.0
    }
}

fn actor_team_won_last_trick(outcome: &TerminalOutcome, actor: PlayerID) -> Option<f32> {
    let winner = outcome.last_trick_winner?;
    let actor_is_landlord = outcome.landlord_team.contains(&actor);
    let winner_is_landlord = outcome.landlord_team.contains(&winner);
    Some(if actor_is_landlord == winner_is_landlord {
        1.0
    } else {
        0.0
    })
}

fn write_header(writer: &mut impl Write) {
    write!(
        writer,
        "schema_version,run_id,game_id,game_seed,decision_id,candidate_id,group,actor,actor_team,behaviour,rollout_behaviour,config,action"
    )
    .unwrap();
    for index in 0..TRAINING_FEATURE_DIM {
        write!(writer, ",f{index}").unwrap();
    }
    writeln!(
        writer,
        ",label,behaviour_label,v_target,v_attacker_points,v_score_bucket,v_win_target,v_kitty_target,q_target,q_attacker_points,q_score_bucket,q_win_target,q_kitty_target,q_samples"
    )
    .unwrap();
}

fn write_row(writer: &mut impl Write, row: &Row) {
    write!(
        writer,
        "{},{},{},{},{},{},{},{},{},{},{},{},{}",
        DATASET_SCHEMA_VERSION,
        row.run_id,
        row.game_id,
        row.game_seed,
        row.decision_id,
        row.candidate_id,
        row.group,
        row.actor.0,
        row.actor_team,
        row.behaviour,
        row.rollout_behaviour,
        GAME_CONFIG,
        row.action,
    )
    .unwrap();
    for value in row.features {
        write!(writer, ",{value:.6}").unwrap();
    }
    write!(
        writer,
        ",{},{},{:.6},{},{},{:.1},",
        row.label,
        row.behaviour_label,
        row.v_target,
        row.v_attacker_points,
        row.v_score_bucket,
        row.v_win_target,
    )
    .unwrap();
    if let Some(v_kitty_target) = row.v_kitty_target {
        write!(writer, "{v_kitty_target:.1}").unwrap();
    }
    write!(writer, ",").unwrap();
    if let Some(q_target) = row.q_target {
        write!(writer, "{q_target:.6}").unwrap();
    }
    write!(writer, ",").unwrap();
    if let Some(q_attacker_points) = row.q_attacker_points {
        write!(writer, "{q_attacker_points}").unwrap();
    }
    write!(writer, ",").unwrap();
    if let Some(q_score_bucket) = row.q_score_bucket {
        write!(writer, "{q_score_bucket}").unwrap();
    }
    write!(writer, ",").unwrap();
    if let Some(q_win_target) = row.q_win_target {
        write!(writer, "{q_win_target:.1}").unwrap();
    }
    write!(writer, ",").unwrap();
    if let Some(q_kitty_target) = row.q_kitty_target {
        write!(writer, "{q_kitty_target:.1}").unwrap();
    }
    writeln!(writer, ",{}", row.q_samples).unwrap();
}

fn encode_action(cards: &[Card]) -> String {
    let mut ids: Vec<usize> = cards
        .iter()
        .filter_map(|card| FULL_DECK.iter().position(|known| known == card))
        .collect();
    ids.sort_unstable();
    ids.iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(".")
}

fn same_multiset(a: &[Card], b: &[Card]) -> bool {
    a.len() == b.len() && Card::count(a.iter().copied()) == Card::count(b.iter().copied())
}

fn derive_game_seed(base: u64, game_index: u64) -> u64 {
    // SplitMix64: stable random access to a deal seed, so sharding/resume order
    // does not affect any game's cards.
    let mut z = base.wrapping_add(game_index.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn ensure_parent(path: &str) {
    if let Some(parent) = Path::new(path).parent() {
        std::fs::create_dir_all(parent).expect("create output parent");
    }
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heterogeneous_behaviour_plan_tracks_each_seat() {
        let seats = [PlayerID(7), PlayerID(4), PlayerID(9), PlayerID(2)];
        let behaviours = [
            BotDifficulty::Easy,
            BotDifficulty::Expert,
            BotDifficulty::Enoch,
            BotDifficulty::Grandmaster,
        ];
        for (index, seat) in seats.iter().enumerate() {
            assert_eq!(
                behaviour_for(&seats, &behaviours, *seat),
                Some(behaviours[index])
            );
        }
        assert_eq!(behaviour_for(&seats, &behaviours, PlayerID(99)), None);
    }

    #[test]
    fn behaviour_labels_are_manifest_stable() {
        assert_eq!(BehaviourMode::parse("easy").label(), "easy");
        assert_eq!(BehaviourMode::parse("expert").label(), "expert");
        assert_eq!(BehaviourMode::parse("enoch").label(), "enoch");
        assert_eq!(BehaviourMode::parse("grandmaster").label(), "grandmaster");
    }

    #[test]
    fn game_seeds_are_stable_and_distinct() {
        assert_eq!(derive_game_seed(7, 3), derive_game_seed(7, 3));
        assert_ne!(derive_game_seed(7, 3), derive_game_seed(7, 4));
        assert_ne!(derive_game_seed(7, 3), derive_game_seed(8, 3));
    }

    #[test]
    fn q_selection_keeps_behaviour_and_teacher_anchors() {
        let config = QConfig {
            candidate_cap: Some(2),
            rollout: BotDifficulty::Easy,
            budget_ms: 1,
        };
        let selected = selected_q_indices(6, 4, 2, config);
        assert_eq!(selected, BTreeSet::from([2, 4]));
    }

    #[test]
    fn disabled_q_selection_is_empty() {
        let config = QConfig {
            candidate_cap: Some(0),
            rollout: BotDifficulty::Easy,
            budget_ms: 1,
        };
        assert!(selected_q_indices(6, 4, 2, config).is_empty());
    }

    #[test]
    fn level_utility_preserves_deadzone_win_and_level_magnitude() {
        let landlord = PlayerID(0);
        let attacker = PlayerID(1);
        let deadzone = TerminalOutcome {
            attacker_points: 80,
            score_bucket: 2,
            landlord_won: false,
            landlord_delta: 0,
            attacker_delta: 0,
            landlord_team: vec![landlord],
            last_trick_winner: Some(attacker),
        };
        assert_eq!(orient_level_utility(&deadzone, attacker), 0.2);
        assert_eq!(orient_level_utility(&deadzone, landlord), -0.2);

        let two_levels = TerminalOutcome {
            attacker_delta: 2,
            ..deadzone
        };
        assert_eq!(orient_level_utility(&two_levels, attacker), 0.6);
        assert_eq!(actor_team_won_last_trick(&two_levels, attacker), Some(1.0));
    }
}
