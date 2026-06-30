//! Optional learned hidden-card proposal for determinized search.
//!
//! The model never changes legality: hard conservation/capacity/void constraints
//! remain in `determinize`. It only supplies log weights over already-legal hidden
//! destinations, and the constrained sampler applies those weights through its
//! bounded Metropolis refinement. Leave `SHENGJI_BELIEF_MODEL_PATH` unset to use
//! the neutral exact-shuffle/rejection sampler.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

use serde::Deserialize;
use sha2::{Digest, Sha256};
use shengji_mechanics::types::{Card, EffectiveSuit, PlayerID, FULL_DECK};

use crate::bot::determinize::{
    AssignmentProposalContext, HiddenAssignmentProposal, HiddenCardLocation,
};

pub const BELIEF_MODEL_PATH_ENV: &str = "SHENGJI_BELIEF_MODEL_PATH";
pub const BELIEF_MANIFEST_PATH_ENV: &str = "SHENGJI_BELIEF_MODEL_MANIFEST";
pub const BELIEF_WEIGHT_ENV: &str = "SHENGJI_BELIEF_WEIGHT";
pub const FEATURE_DIM: usize = 20;
const DESTINATIONS: usize = 4;
const CONTRACT: &str = "offline_honest_card_location_belief";
const SUPPORTED_GAME_CONTRACT: &str = "tractor:4p:2x-standard:kitty8:no-removed";
const TARGET_CLASSES: [&str; DESTINATIONS] =
    ["next-seat", "opposite-seat", "previous-seat", "kitty"];

type RunnableModel = tract_onnx::prelude::TypedRunnableModel<tract_onnx::prelude::TypedModel>;

#[derive(Deserialize)]
struct Manifest {
    manifest_version: u32,
    contract: String,
    feature_schema_version: u32,
    feature_dim: usize,
    feature_names: Vec<String>,
    inputs: Vec<String>,
    outputs: Vec<String>,
    target_classes: Vec<String>,
    hard_legality_mask_value: f32,
    model_sha256: String,
    supported_game_contract: String,
    serving_status: String,
}

struct OnnxBeliefProposal {
    runnable: RunnableModel,
    healthy: AtomicBool,
    log_weight_scale: f64,
}

pub fn loaded_proposal() -> Option<&'static dyn HiddenAssignmentProposal> {
    static MODEL: OnceLock<Option<OnnxBeliefProposal>> = OnceLock::new();
    MODEL
        .get_or_init(|| match load_from_env() {
            Ok(model) => model,
            Err(error) => {
                eprintln!("[belief-model] load failed; using neutral sampler: {error:#}");
                None
            }
        })
        .as_ref()
        .map(|model| model as &dyn HiddenAssignmentProposal)
}

fn load_from_env() -> tract_onnx::prelude::TractResult<Option<OnnxBeliefProposal>> {
    use tract_onnx::prelude::*;

    let Some(path) = std::env::var_os(BELIEF_MODEL_PATH_ENV).filter(|path| !path.is_empty()) else {
        return Ok(None);
    };
    let log_weight_scale = std::env::var(BELIEF_WEIGHT_ENV)
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);
    if log_weight_scale == 0.0 {
        eprintln!(
            "[belief-model] model path set but {BELIEF_WEIGHT_ENV} is zero/unset; using neutral sampler"
        );
        return Ok(None);
    }
    let path = PathBuf::from(path);
    let manifest_path = std::env::var_os(BELIEF_MANIFEST_PATH_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| companion_manifest_path(&path));
    let manifest: Manifest = serde_json::from_str(&std::fs::read_to_string(&manifest_path)?)?;
    let expected_features: Vec<String> =
        (0..FEATURE_DIM).map(|index| format!("b{index}")).collect();
    if manifest.manifest_version != 1
        || manifest.contract != CONTRACT
        || manifest.feature_schema_version != 1
        || manifest.feature_dim != FEATURE_DIM
        || manifest.feature_names != expected_features
        || manifest
            .inputs
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            != ["features", "legality_mask"]
        || manifest
            .outputs
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            != ["destination_logits"]
        || manifest
            .target_classes
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            != TARGET_CLASSES
        || manifest.hard_legality_mask_value != -10_000.0
        || manifest.supported_game_contract != SUPPORTED_GAME_CONTRACT
        || manifest.serving_status != "experimental_candidate"
    {
        anyhow::bail!("unsupported belief model manifest contract");
    }
    let bytes = std::fs::read(&path)?;
    if bytes.len() < 64 {
        anyhow::bail!("belief model is too small to be ONNX");
    }
    let digest = format!("{:x}", Sha256::digest(&bytes));
    if digest != manifest.model_sha256 {
        anyhow::bail!("belief model SHA-256 does not match its manifest");
    }
    let mut cursor = std::io::Cursor::new(bytes);
    let mut model = tract_onnx::onnx().model_for_read(&mut cursor)?;
    let batch = model.symbols.sym("N");
    model.set_input_fact(
        0,
        f32::fact([batch.to_dim(), (FEATURE_DIM as i64).to_dim()]).into(),
    )?;
    model.set_input_fact(
        1,
        f32::fact([batch.to_dim(), (DESTINATIONS as i64).to_dim()]).into(),
    )?;
    Ok(Some(OnnxBeliefProposal {
        runnable: model.into_optimized()?.into_runnable()?,
        healthy: AtomicBool::new(true),
        log_weight_scale,
    }))
}

fn companion_manifest_path(model_path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.manifest.json", model_path.display()))
}

fn supported_context(context: &AssignmentProposalContext<'_>) -> bool {
    use crate::settings::GameMode;

    let p = context.view;
    let seats = p.propagated().players();
    let (kitty, removed) = p.piles_for_determinization();
    seats.len() == 4
        && seats.iter().any(|seat| seat.id == context.observer)
        && p.num_decks() == 2
        && matches!(p.game_mode(), GameMode::Tractor)
        && kitty.len() == 8
        && removed.is_empty()
        && p.public_history_complete()
        && context.knowledge.total_cards == FULL_DECK.len() * 2
        && FULL_DECK
            .iter()
            .all(|&card| context.knowledge.configured_copies(card) == 2)
}

fn relative_class(
    context: &AssignmentProposalContext<'_>,
    location: HiddenCardLocation,
) -> Option<usize> {
    match location {
        HiddenCardLocation::Kitty => Some(3),
        HiddenCardLocation::Removed => None,
        HiddenCardLocation::Player(player) => {
            let seats: Vec<PlayerID> = context
                .view
                .propagated()
                .players()
                .iter()
                .map(|seat| seat.id)
                .collect();
            if seats.len() != 4 {
                return None;
            }
            let observer = seats.iter().position(|seat| *seat == context.observer)?;
            let destination = seats.iter().position(|seat| *seat == player)?;
            let offset = (destination + seats.len() - observer) % seats.len();
            (offset > 0).then_some(offset - 1)
        }
    }
}

/// Canonical schema-v1 encoder shared by offline data generation and runtime.
/// Keeping it in one place makes a feature change a compile-visible contract
/// change instead of allowing training and serving implementations to drift.
pub fn encode_belief_features(
    p: &crate::game_state::play_phase::PlayPhase,
    observer: PlayerID,
    knowledge: &crate::bot::determinize::Knowledge,
    card: Card,
) -> ([f32; FEATURE_DIM], [f32; DESTINATIONS]) {
    let trump = p.trump();
    let effective = trump.effective_suit(card);
    let card_id = FULL_DECK
        .iter()
        .position(|known| *known == card)
        .unwrap_or(0);
    let mut features = [0.0; FEATURE_DIM];
    features[0] = card_id as f32 / 53.0;
    features[1] = card.points().unwrap_or(0) as f32 / 10.0;
    features[2] = if effective == EffectiveSuit::Trump {
        1.0
    } else {
        0.0
    };
    features[3] = match effective {
        EffectiveSuit::Unknown => 0.0,
        EffectiveSuit::Clubs => 0.2,
        EffectiveSuit::Diamonds => 0.4,
        EffectiveSuit::Spades => 0.6,
        EffectiveSuit::Hearts => 0.8,
        EffectiveSuit::Trump => 1.0,
    };
    features[4] = match card {
        Card::Suited { number, .. } => number.as_u32() as f32 / 14.0,
        Card::SmallJoker => 0.95,
        Card::BigJoker => 1.0,
        Card::Unknown => 0.0,
    };
    features[5] = knowledge.seen.get(&card).copied().unwrap_or(0) as f32
        / knowledge.configured_copies(card).max(1) as f32;
    features[6] =
        p.played_this_hand().values().sum::<usize>() as f32 / knowledge.total_cards.max(1) as f32;
    features[7] = if p.landlords_team().contains(&observer) {
        1.0
    } else {
        0.0
    };
    let (points, _) = p.calculate_points();
    if let Some(step) = p.bot_step_size().filter(|step| *step > 0) {
        features[8] = points.rem_euclid(step) as f32 / step as f32;
    }
    let (kitty, removed) = p.piles_for_determinization();
    let hidden_kitty = kitty.iter().filter(|card| **card == Card::Unknown).count();
    features[9] = hidden_kitty as f32 / 8.0;

    let seats: Vec<PlayerID> = p
        .propagated()
        .players()
        .iter()
        .map(|seat| seat.id)
        .collect();
    let observer_index = seats.iter().position(|seat| *seat == observer).unwrap_or(0);
    let mut mask = [0.0; DESTINATIONS];
    for class in 0..3 {
        if seats.len() != 4 {
            continue;
        }
        let seat = seats[(observer_index + class + 1) % 4];
        let capacity = knowledge.hidden_counts.get(&seat).copied().unwrap_or(0);
        let is_void = knowledge
            .voids
            .get(&seat)
            .is_some_and(|suits| suits.contains(&effective));
        features[10 + class] = capacity as f32 / 27.0;
        features[13 + class] = if is_void { 1.0 } else { 0.0 };
        mask[class] = if capacity > 0 && !is_void { 1.0 } else { 0.0 };
    }
    mask[3] = if hidden_kitty > 0 { 1.0 } else { 0.0 };
    features[16] = p
        .hands()
        .get(observer)
        .map(|hand| hand.values().sum::<usize>() as f32 / 27.0)
        .unwrap_or(0.0);
    let hidden_total = knowledge.hidden_counts.values().sum::<usize>()
        + hidden_kitty
        + removed
            .iter()
            .filter(|card| **card == Card::Unknown)
            .count();
    features[17] = hidden_total as f32 / 100.0;
    features[18] = p.num_decks() as f32 / 4.0;
    features[19] = 1.0;
    (features, mask)
}

impl HiddenAssignmentProposal for OnnxBeliefProposal {
    fn log_weight(
        &self,
        _context: &AssignmentProposalContext<'_>,
        _card: Card,
        _location: HiddenCardLocation,
    ) -> f64 {
        // Serving always calls the batched hook below.
        0.0
    }

    fn batch_log_weights(
        &self,
        context: &AssignmentProposalContext<'_>,
        cards: &[Card],
        slots: &[HiddenCardLocation],
    ) -> Vec<Vec<f64>> {
        use tract_onnx::prelude::*;

        if cards.is_empty() || !supported_context(context) || !self.healthy.load(Ordering::Relaxed)
        {
            return Vec::new();
        }

        // Physical copies of the same identity have identical features. Infer
        // each identity once (at most 54 rows) and fan the logits back out to
        // copies; early-hand inference is roughly halved for two standard decks.
        let mut unique_cards = Vec::new();
        let mut unique_index = HashMap::new();
        let mut card_rows = Vec::with_capacity(cards.len());
        for &card in cards {
            let row = *unique_index.entry(card).or_insert_with(|| {
                let row = unique_cards.len();
                unique_cards.push(card);
                row
            });
            card_rows.push(row);
        }
        let rows: Vec<([f32; FEATURE_DIM], [f32; DESTINATIONS])> = unique_cards
            .iter()
            .map(|&card| {
                encode_belief_features(context.view, context.observer, context.knowledge, card)
            })
            .collect();
        let feature_data: Vec<f32> = rows
            .iter()
            .flat_map(|(features, _)| features.iter().copied())
            .collect();
        let mask_data: Vec<f32> = rows
            .iter()
            .flat_map(|(_, mask)| mask.iter().copied())
            .collect();
        let Ok(features) =
            tract_ndarray::Array2::from_shape_vec((unique_cards.len(), FEATURE_DIM), feature_data)
        else {
            return self.disable("could not construct feature tensor");
        };
        let Ok(mask) =
            tract_ndarray::Array2::from_shape_vec((unique_cards.len(), DESTINATIONS), mask_data)
        else {
            return self.disable("could not construct legality tensor");
        };
        let result = match self.runnable.run(tvec!(
            Tensor::from(features).into(),
            Tensor::from(mask).into()
        )) {
            Ok(result) if result.len() == 1 => result,
            _ => return self.disable("ONNX inference failed or returned the wrong output count"),
        };
        let Ok(logits) = result[0].to_array_view::<f32>() else {
            return self.disable("ONNX output was not f32");
        };
        if logits.shape() != [unique_cards.len(), DESTINATIONS]
            || logits.iter().any(|value| !value.is_finite())
        {
            return self.disable("ONNX output had the wrong shape or a non-finite value");
        }
        (0..cards.len())
            .map(|row| {
                let unique_row = card_rows[row];
                slots
                    .iter()
                    .map(|&location| {
                        relative_class(context, location)
                            .map(|class| logits[[unique_row, class]] as f64 * self.log_weight_scale)
                            .unwrap_or(0.0)
                    })
                    .collect()
            })
            .collect()
    }
}

impl OnnxBeliefProposal {
    fn disable(&self, reason: &str) -> Vec<Vec<f64>> {
        if self.healthy.swap(false, Ordering::Relaxed) {
            eprintln!(
                "[belief-model] inference disabled after contract failure; using neutral sampler: {reason}"
            );
        }
        Vec::new()
    }
}
