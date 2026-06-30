//! Expert tier: a learned neural net that scores legal candidate plays.
//!
//! # Overview
//!
//! The Expert difficulty is trained by **behavioral cloning / distillation** of
//! the `Omniscient` (perfect-information) teacher. For every PLAY-phase decision
//! in a corpus of self-play games we record, for each legal candidate move, a
//! fixed-length **HONEST** feature vector describing `(state, candidate)` from
//! the acting seat's redacted view, together with a binary label = 1 if that
//! candidate is the one the Omniscient teacher actually chose (computed with
//! perfect information), else 0. A small PyTorch MLP is then trained to score
//! candidates so the teacher's choice ranks first (see `training/`). The trained
//! model is exported to ONNX and embedded here via [`include_bytes!`].
//!
//! The crucial honesty property: although the *teacher labels* come from
//! perfect-information play, the *features the net consumes are HONEST only* —
//! they are derived purely from the redacted per-player view. So at inference
//! time the Expert tier approximates perfect-information play using only the
//! information a human in its seat could observe. It NEVER reads hidden hands.
//!
//! # Feature encoding (the contract shared with `gen_training_data` + training)
//!
//! [`candidate_features`] returns a fixed-length `[f32; FEATURE_DIM]` vector for
//! a `(PlayPhase view, me, candidate cards)` triple. Both the Rust data-export
//! example and this inference path call the SAME function, so the encoding can
//! never drift between training and serving. The layout is documented inline on
//! [`candidate_features`]; the upshot is a compact mix of:
//!
//! * candidate shape: card count, points, trump count, max/min strength, whether
//!   it's a lead / follows suit / trumps in, its structural size;
//! * trick context: pot points, whether our team is currently winning, whether
//!   the current winner is our teammate, whether we're last to act, the current
//!   winner's top strength and whether it's trump, and a heuristic estimate of
//!   whether this candidate likely wins the trick;
//! * my-hand summary: hand size, trumps held, points held, aces / kings / jokers
//!   held;
//! * trump info: whether trump is NT, and the trump number's rank;
//! * the heuristic's own score for this candidate (a strong prior the net can
//!   refine).
//!
//! # Inference + fallback
//!
//! [`choose_play_expert`] generates the legal candidates (lead or follow) with
//! the same generators the heuristic uses, scores each with the embedded model,
//! and returns the argmax. If the model fails to load, fails to run, or no
//! candidates exist, it returns `None` and the policy falls back to the
//! hand-written heuristic prior inside the determinized search, so Expert is
//! never illegal/None.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::Deserialize;
use shengji_mechanics::types::{Card, EffectiveSuit, Number, PlayerID, Trump};

use crate::bot::determinize::Knowledge;
use crate::bot::heuristics::{self};
use crate::game_state::play_phase::PlayPhase;

/// The fixed length of the per-candidate feature vector. Must match the training
/// script's input dimension exactly. If you change the encoding, retrain.
///
/// Indices 0..=27 are the original compact encoding; 28..=35 are the richer
/// honest "card-memory" features derived from [`Knowledge::from_play_view`]
/// (remaining unseen trumps / points, per-seat voids of the seats still to act,
/// seat position). Adding these raised the distillation ceiling above the bare
/// heuristic.
pub const FEATURE_DIM: usize = 36;

/// Feature width emitted by the v2 action-value data pipeline. The embedded
/// production model remains on [`FEATURE_DIM`] (schema v1), so the existing
/// policy baseline is byte-for-byte compatible. Candidate models declare their
/// schema in a companion `MODEL.onnx.manifest.json`; inference selects the
/// matching encoder instead of guessing from an ONNX graph.
pub const TRAINING_FEATURE_DIM: usize = 49;

/// Version of [`candidate_features_v2`]. Bump this whenever a v2 feature changes
/// meaning, even if its width does not change.
pub const TRAINING_FEATURE_SCHEMA_VERSION: u32 = 2;

/// Points-scale for the (optional) learned VALUE head's target. The value target
/// is the realized terminal point-differential oriented for the acting seat's team
/// (same sign convention as the search leaf evaluator's `realized` term), DIVIDED
/// by this constant and clamped to `[-1, 1]` so it suits a `tanh` head. Inference
/// scales the `tanh` output back to points by multiplying by `VALUE_NORM` before
/// blending it into the leaf evaluator.
///
/// CONTRACT: this is the single source of truth for the scale, shared by the data
/// exporter (`gen_training_data`) and the inference blend (`search::evaluate_position`).
/// The Python trainer is scale-free — it regresses the already-normalized column —
/// so changing this only requires regenerating data + retraining, no Python edit.
pub const VALUE_NORM: f64 = 200.0;

/// Normalizer for schema-v2 terminal level utility. The scalar target is
/// signed from the acting team's perspective:
///   sign(team won) * (1 + levels awarded to the winning team) / 5
/// and clamped to [-1, 1]. The added one preserves a learning signal for a
/// turnover/dead-zone win that awards zero levels. This is deliberately NOT a
/// point unit and must never be passed through the legacy VALUE_NORM leaf blend.
pub const LEVEL_UTILITY_NORM: f32 = 5.0;

/// The embedded ONNX model (a small MLP scoring one candidate's features to a
/// scalar logit). If training has not produced a model yet, this file may be a
/// placeholder; [`model`] handles a missing/invalid model gracefully by
/// returning `None`, which makes the Expert tier fall back to the heuristic.
///
/// The asset lives under `core/src/bot/` so it travels with the crate (and the
/// pure-Rust `tract-onnx` runtime builds in the musl Docker image — no
/// `onnxruntime` / `ort` C dependency).
static MODEL_BYTES: &[u8] = include_bytes!("expert_model.onnx");
static EMBEDDED_MODEL_MANIFEST: &str = include_str!("expert_model.onnx.manifest.json");

/// Environment variable that, when set to a readable path, overrides the embedded
/// [`MODEL_BYTES`] with an ONNX model loaded from disk AT RUNTIME.
///
/// The net is otherwise [`include_bytes!`]-baked into the binary, so A/B-testing a
/// freshly-trained net costs a full Rust rebuild. This override lets the eval
/// harness (and a manual `cargo run --example ...`) swap in a candidate model with
/// just a flag — the single biggest iteration-speed unlock for net training (see
/// `docs/bot-training-roadmap.md`). Read ONCE, lazily, on the first Expert
/// decision (the model is cached in a [`OnceLock`]), so set it BEFORE the process
/// makes any bot decision. Leave it unset in production to use the embedded net.
pub const MODEL_PATH_ENV: &str = "SHENGJI_EXPERT_MODEL_PATH";
/// Optional explicit companion-manifest path for [`MODEL_PATH_ENV`]. When unset,
/// inference looks for `<model path>.manifest.json`. A missing manifest is
/// accepted only as a legacy schema-v1/36-feature policy model, preserving old
/// A/B commands while making every v2 model self-describing.
pub const MODEL_MANIFEST_PATH_ENV: &str = "SHENGJI_EXPERT_MODEL_MANIFEST";

type RunnableModel = tract_onnx::prelude::TypedRunnableModel<tract_onnx::prelude::TypedModel>;

#[derive(Clone, Debug, Deserialize)]
struct ModelManifest {
    manifest_version: u32,
    feature_schema_version: u32,
    feature_dim: usize,
    outputs: Vec<String>,
    /// Semantic unit for each ONNX output. Schema-v2 manifests must declare
    /// this explicitly so a level-utility V head can never be multiplied by the
    /// legacy point scale inside search.
    #[serde(default)]
    output_semantics: Option<Vec<String>>,
}

struct Model {
    runnable: RunnableModel,
    manifest: ModelManifest,
}

/// Typed runtime contract. Callers branch on semantic units rather than merely
/// on output position, preventing normalized levels from being consumed as
/// normalized points.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpertModelSemantics {
    LegacyPolicy,
    LegacyPointValue,
    V2Policy,
    V2LevelValue,
    V2LevelQ,
}

/// Lazily-parsed model, shared across all Expert decisions. `None` means the
/// model could not be loaded (e.g. the embedded bytes are a placeholder), in
/// which case the caller falls back to the hand-written heuristic.
fn model() -> Option<&'static Model> {
    static MODEL: OnceLock<Option<Model>> = OnceLock::new();
    MODEL
        .get_or_init(|| match load_model() {
            Ok(model) => Some(model),
            Err(error) => {
                // This executes at most once because the failure is cached in the
                // OnceLock. Surface a bad override/schema instead of silently
                // making an A/B compare the heuristic fallback against itself.
                eprintln!("[expert-model] load failed; using heuristic fallback: {error:#}");
                None
            }
        })
        .as_ref()
}

/// Parse and optimize the ONNX model into a runnable plan, choosing the byte
/// source: a runtime override file ([`MODEL_PATH_ENV`]) if set, else the embedded
/// [`MODEL_BYTES`].
///
/// When the override is set but the file cannot be read, we surface the error (so
/// the caller's fall-back to the heuristic is the SAME as a missing net) rather
/// than silently using the embedded net — a silent fallback would make a net A/B
/// quietly compare the embedded net against itself.
fn load_model() -> tract_onnx::prelude::TractResult<Model> {
    match std::env::var_os(MODEL_PATH_ENV) {
        Some(path) if !path.is_empty() => {
            let bytes = std::fs::read(&path)
                .map_err(|e| anyhow::anyhow!("failed to read {MODEL_PATH_ENV} ({path:?}): {e}"))?;
            let path = PathBuf::from(path);
            let manifest_path = std::env::var_os(MODEL_MANIFEST_PATH_ENV)
                .map(PathBuf::from)
                .unwrap_or_else(|| companion_manifest_path(&path));
            let manifest = match std::fs::read_to_string(&manifest_path) {
                Ok(json) => parse_manifest(&json)?,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    eprintln!(
                        "[expert-model] no companion manifest at {}; assuming legacy schema-v1/36-feature policy model",
                        manifest_path.display()
                    );
                    legacy_manifest()
                }
                Err(error) => anyhow::bail!(
                    "failed to read expert model manifest {}: {error}",
                    manifest_path.display()
                ),
            };
            model_from_bytes_with_manifest(&bytes, manifest)
        }
        _ => model_from_bytes_with_manifest(MODEL_BYTES, parse_manifest(EMBEDDED_MODEL_MANIFEST)?),
    }
}

fn companion_manifest_path(model_path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.manifest.json", model_path.display()))
}

fn legacy_manifest() -> ModelManifest {
    ModelManifest {
        manifest_version: 1,
        feature_schema_version: 1,
        feature_dim: FEATURE_DIM,
        outputs: vec!["score".to_string()],
        output_semantics: Some(vec!["policy_logit".to_string()]),
    }
}

fn parse_manifest(json: &str) -> tract_onnx::prelude::TractResult<ModelManifest> {
    let manifest: ModelManifest = serde_json::from_str(json)
        .map_err(|e| anyhow::anyhow!("invalid expert model manifest JSON: {e}"))?;
    validate_manifest(&manifest)?;
    Ok(manifest)
}

fn validate_manifest(manifest: &ModelManifest) -> tract_onnx::prelude::TractResult<()> {
    if manifest.manifest_version != 1 {
        anyhow::bail!(
            "unsupported expert manifest version {} (expected 1)",
            manifest.manifest_version
        );
    }
    let supported_features = (manifest.feature_schema_version == 1
        && manifest.feature_dim == FEATURE_DIM)
        || (manifest.feature_schema_version == TRAINING_FEATURE_SCHEMA_VERSION
            && manifest.feature_dim == TRAINING_FEATURE_DIM);
    if !supported_features {
        anyhow::bail!(
            "unsupported expert feature contract schema={} dim={} (supported: schema=1 dim={}, schema={} dim={})",
            manifest.feature_schema_version,
            manifest.feature_dim,
            FEATURE_DIM,
            TRAINING_FEATURE_SCHEMA_VERSION,
            TRAINING_FEATURE_DIM
        );
    }
    let expected = ["score", "state_value", "action_q"];
    if manifest.outputs.is_empty() || manifest.outputs.len() > expected.len() {
        anyhow::bail!("expert model must declare 1..=3 outputs");
    }
    if manifest.feature_schema_version == 1 && manifest.outputs.len() > 2 {
        anyhow::bail!("schema-v1 models cannot declare action_q");
    }
    for (actual, expected) in manifest.outputs.iter().zip(expected.iter()) {
        if actual != expected {
            anyhow::bail!(
                "unsupported expert output contract {:?}; expected prefix {:?}",
                manifest.outputs,
                expected
            );
        }
    }
    if let Some(semantics) = &manifest.output_semantics {
        if semantics.len() != manifest.outputs.len() {
            anyhow::bail!(
                "expert output_semantics length {} != outputs length {}",
                semantics.len(),
                manifest.outputs.len()
            );
        }
        if semantics.first().map(String::as_str) != Some("policy_logit") {
            anyhow::bail!("expert output 0 semantic must be policy_logit");
        }
        if manifest.feature_schema_version == 1
            && semantics.get(1).map(String::as_str)
                != manifest.outputs.get(1).map(|_| "normalized_point_margin")
        {
            anyhow::bail!("schema-v1 state_value semantic must be normalized_point_margin");
        }
        if manifest.feature_schema_version == TRAINING_FEATURE_SCHEMA_VERSION {
            if semantics.get(1).map(String::as_str)
                != manifest.outputs.get(1).map(|_| "normalized_level_utility")
            {
                anyhow::bail!("schema-v2 state_value semantic must be normalized_level_utility");
            }
            if semantics.get(2).map(String::as_str)
                != manifest.outputs.get(2).map(|_| "normalized_level_utility")
            {
                anyhow::bail!("schema-v2 action_q semantic must be normalized_level_utility");
            }
        }
    } else if manifest.feature_schema_version == TRAINING_FEATURE_SCHEMA_VERSION {
        anyhow::bail!("schema-v2 expert models must declare output_semantics");
    }
    Ok(())
}

fn output_semantic(model: &Model, index: usize) -> Option<&str> {
    if let Some(semantics) = &model.manifest.output_semantics {
        return semantics.get(index).map(String::as_str);
    }
    // Compatibility for old schema-v1 manifests produced before semantic units
    // were recorded. Their optional value head was trained on normalized points.
    match (model.manifest.feature_schema_version, index) {
        (1, 0) => Some("policy_logit"),
        (1, 1) => Some("normalized_point_margin"),
        _ => None,
    }
}

fn semantics_for(model: &Model) -> ExpertModelSemantics {
    match (
        model.manifest.feature_schema_version,
        output_semantic(model, 1),
        output_semantic(model, 2),
    ) {
        (1, Some("normalized_point_margin"), _) => ExpertModelSemantics::LegacyPointValue,
        (1, _, _) => ExpertModelSemantics::LegacyPolicy,
        (
            TRAINING_FEATURE_SCHEMA_VERSION,
            Some("normalized_level_utility"),
            Some("normalized_level_utility"),
        ) => ExpertModelSemantics::V2LevelQ,
        (TRAINING_FEATURE_SCHEMA_VERSION, Some("normalized_level_utility"), _) => {
            ExpertModelSemantics::V2LevelValue
        }
        (TRAINING_FEATURE_SCHEMA_VERSION, _, _) => ExpertModelSemantics::V2Policy,
        _ => unreachable!("manifest validation rejected unsupported feature schema"),
    }
}

/// Semantic contract of the lazily loaded model, or None when model loading
/// failed and Expert is using its heuristic fallback.
pub fn loaded_model_semantics() -> Option<ExpertModelSemantics> {
    model().map(semantics_for)
}

/// Parse and optimize ONNX bytes into a runnable plan. The model takes a single
/// input named `x` of shape `[N, FEATURE_DIM]` (a batch of N candidates) and
/// produces `[N, 1]` logits.
#[cfg(test)]
fn model_from_bytes(bytes: &[u8]) -> tract_onnx::prelude::TractResult<Model> {
    model_from_bytes_with_manifest(bytes, legacy_manifest())
}

fn model_from_bytes_with_manifest(
    bytes: &[u8],
    manifest: ModelManifest,
) -> tract_onnx::prelude::TractResult<Model> {
    use tract_onnx::prelude::*;

    // A near-empty / placeholder file can't be a valid ONNX graph; bail early so
    // we fall back to the heuristic rather than erroring deeper in the parser.
    if bytes.len() < 64 {
        anyhow::bail!("expert model is a placeholder (too small to be ONNX)");
    }

    let mut cursor = std::io::Cursor::new(bytes);
    let mut model = tract_onnx::onnx().model_for_read(&mut cursor)?;
    // Fix the input to a runtime-variable batch (`N`) of FEATURE_DIM-length rows
    // so a single inference call can score a whole candidate set at once.
    let batch = model.symbols.sym("N");
    model.set_input_fact(
        0,
        f32::fact([batch.to_dim(), (manifest.feature_dim as i64).to_dim()]).into(),
    )?;
    let runnable = model.into_optimized()?.into_runnable()?;
    Ok(Model { runnable, manifest })
}

/// Score an explicit set of candidate plays with the learned Expert net,
/// returning one logit per candidate (higher = the net likes it more), or `None`
/// if the model is unavailable / failed to run / the input is empty.
///
/// This is the shared net-policy primitive: both the single-shot
/// [`choose_play_expert`] and the net-guided determinized search
/// ([`crate::bot::search`]) call it so the *same* honest features and the *same*
/// model drive candidate priors, pruning, and rollout moves.
///
/// `p` MUST be the redacted per-player view (the honesty invariant): every
/// feature is computed from observable information only. The caller owns
/// candidate generation, so this never reads hidden hands.
pub fn score_candidates_net(
    p: &PlayPhase,
    me: PlayerID,
    candidates: &[Vec<Card>],
) -> Option<Vec<f32>> {
    if candidates.is_empty() {
        return None;
    }
    let model = model()?;

    let n = candidates.len();
    let flat = feature_batch(model, p, me, candidates)?;
    run_model(model, &flat, n)
}

/// Choose the best legal play for `me` using the learned Expert net, or `None`
/// if the model is unavailable / produced nothing (caller falls back to the heuristic).
///
/// `p` MUST be the redacted per-player view (the honesty invariant): every
/// feature is computed from observable information only.
pub fn choose_play_expert(p: &PlayPhase, me: PlayerID) -> Option<Vec<Card>> {
    let leading = p.trick().played_cards().is_empty();

    // Generate legal candidates with the SAME generators the heuristic uses.
    let candidates: Vec<Vec<Card>> = if leading {
        heuristics::lead_candidates(p, me)
    } else {
        heuristics::follow_candidates(p, me)
    };
    if candidates.is_empty() {
        return None;
    }
    if candidates.len() == 1 {
        return Some(candidates.into_iter().next().unwrap());
    }

    let scores = score_candidates_net(p, me, &candidates)?;

    // Argmax over the candidate logits; ties break toward the earlier candidate
    // (candidates are heuristic-ordered-ish via the generators).
    let mut best_idx = 0;
    let mut best = f32::NEG_INFINITY;
    for (i, &s) in scores.iter().enumerate() {
        if s > best {
            best = s;
            best_idx = i;
        }
    }
    Some(candidates.into_iter().nth(best_idx).unwrap())
}

/// Run the model on a flat `[n * FEATURE_DIM]` buffer, returning `n` scalar
/// logits, or `None` on any inference error (so the caller falls back).
fn run_model(model: &Model, flat: &[f32], n: usize) -> Option<Vec<f32>> {
    run_model_output(model, flat, n, 0)
}

/// Run the model on a flat `[n * FEATURE_DIM]` buffer and return output number
/// `out_idx` flattened to `n` scalars, or `None` if the model has fewer than
/// `out_idx + 1` outputs (e.g. a policy-only / legacy model has no value output)
/// or on any inference error. Output 0 = policy logits (`score`); output 1 = the
/// `tanh` value estimate (`value`), present only on a 2-output value-head model.
fn run_model_output(model: &Model, flat: &[f32], n: usize, out_idx: usize) -> Option<Vec<f32>> {
    use tract_onnx::prelude::*;

    if out_idx >= model.manifest.outputs.len() {
        return None;
    }
    let expected_input_len = n.checked_mul(model.manifest.feature_dim)?;
    if flat.len() != expected_input_len {
        log_inference_error(format!(
            "input length {} != expected {} (n={} dim={})",
            flat.len(),
            expected_input_len,
            n,
            model.manifest.feature_dim
        ));
        return None;
    }
    let input =
        match tract_ndarray::Array2::from_shape_vec((n, model.manifest.feature_dim), flat.to_vec())
        {
            Ok(input) => input,
            Err(error) => {
                log_inference_error(format!("could not construct model input: {error}"));
                return None;
            }
        };
    let tensor: Tensor = input.into();
    let result = match model.runnable.run(tvec!(tensor.into())) {
        Ok(result) => result,
        Err(error) => {
            log_inference_error(format!("ONNX execution failed: {error}"));
            return None;
        }
    };
    if result.len() != model.manifest.outputs.len() {
        log_inference_error(format!(
            "model returned {} outputs but manifest declares {} ({:?})",
            result.len(),
            model.manifest.outputs.len(),
            model.manifest.outputs
        ));
        return None;
    }
    let out = result.get(out_idx)?;
    let view = match out.to_array_view::<f32>() {
        Ok(view) => view,
        Err(error) => {
            log_inference_error(format!(
                "output {} ({}) is not f32: {error}",
                out_idx, model.manifest.outputs[out_idx]
            ));
            return None;
        }
    };
    if view.shape() != [n, 1] {
        log_inference_error(format!(
            "output {} ({}) has shape {:?}, expected [{}, 1]",
            out_idx,
            model.manifest.outputs[out_idx],
            view.shape(),
            n
        ));
        return None;
    }
    let values: Vec<f32> = view.iter().copied().collect();
    if values.iter().any(|v| !v.is_finite()) {
        log_inference_error(format!(
            "output {} ({}) contains NaN/Inf",
            out_idx, model.manifest.outputs[out_idx]
        ));
        return None;
    }
    Some(values)
}

fn log_inference_error(message: String) {
    static LOGGED: OnceLock<()> = OnceLock::new();
    LOGGED.get_or_init(|| {
        eprintln!(
            "[expert-model] inference contract failure; using heuristic/static fallback: {message}"
        );
    });
}

fn feature_batch(
    model: &Model,
    p: &PlayPhase,
    me: PlayerID,
    candidates: &[Vec<Card>],
) -> Option<Vec<f32>> {
    let mut flat = Vec::with_capacity(candidates.len() * model.manifest.feature_dim);
    match model.manifest.feature_schema_version {
        1 => {
            for candidate in candidates {
                flat.extend_from_slice(&candidate_features(p, me, candidate));
            }
        }
        TRAINING_FEATURE_SCHEMA_VERSION => {
            for candidate in candidates {
                flat.extend_from_slice(&candidate_features_v2(p, me, candidate));
            }
        }
        _ => return None,
    }
    Some(flat)
}

/// Score an explicit set of candidates with the learned VALUE head, returning one
/// `tanh` value per candidate in `[-1, 1]` (the normalized terminal-margin estimate
/// oriented for `me`'s team), or `None` if the model is unavailable, failed to
/// run, the input is empty, OR the model has NO value output (a policy-only /
/// legacy model — so the search's value blend transparently stays disabled and it
/// uses the static leaf eval). Multiply by [`VALUE_NORM`] to recover points.
///
/// `p` MUST be the redacted per-player view (the honesty invariant); the search
/// calls this only on SAMPLED determinized worlds, never the real hidden hands.
pub fn point_value_candidates_net(
    p: &PlayPhase,
    me: PlayerID,
    candidates: &[Vec<Card>],
) -> Option<Vec<f32>> {
    if candidates.is_empty() {
        return None;
    }
    let model = model()?;
    // Search converts this head back to points with VALUE_NORM. A schema-v2
    // level-utility state value is deliberately NOT accepted here.
    if output_semantic(model, 1) != Some("normalized_point_margin") {
        return None;
    }
    let n = candidates.len();
    let flat = feature_batch(model, p, me, candidates)?;
    // Output index 1 is the state-value head; `None` here means a 1-output policy
    // model, which correctly disables the value blend.
    run_model_output(model, &flat, n, 1)
}

/// Schema-v2 state value in normalized terminal level-utility units. This is
/// intentionally separate from point_value_candidates_net; callers may replace
/// a level-valued evaluator with it, but must never mix it into point scores.
pub fn level_value_candidates_net(
    p: &PlayPhase,
    me: PlayerID,
    candidates: &[Vec<Card>],
) -> Option<Vec<f32>> {
    if candidates.is_empty() {
        return None;
    }
    let model = model()?;
    if !matches!(
        semantics_for(model),
        ExpertModelSemantics::V2LevelValue | ExpertModelSemantics::V2LevelQ
    ) {
        return None;
    }
    let n = candidates.len();
    let flat = feature_batch(model, p, me, candidates)?;
    run_model_output(model, &flat, n, 1)
}

/// Backward-compatible name for callers predating typed output semantics. It is
/// safe: the implementation delegates to the point-only API and returns None
/// for every schema-v2 level-utility model.
pub fn value_candidates_net(
    p: &PlayPhase,
    me: PlayerID,
    candidates: &[Vec<Card>],
) -> Option<Vec<f32>> {
    point_value_candidates_net(p, me, candidates)
}

/// Score candidates with the optional action-value head (`Q(o,a)`, output 2).
/// The shipped model has no such output; this is the explicit inference contract
/// for staged DMC/action-value models and does not alter production selection yet.
pub fn action_q_candidates_net(
    p: &PlayPhase,
    me: PlayerID,
    candidates: &[Vec<Card>],
) -> Option<Vec<f32>> {
    if candidates.is_empty() {
        return None;
    }
    let model = model()?;
    if semantics_for(model) != ExpertModelSemantics::V2LevelQ {
        return None;
    }
    let n = candidates.len();
    let flat = feature_batch(model, p, me, candidates)?;
    run_model_output(model, &flat, n, 2)
}

/// Normalize a raw card-strength rank into roughly `[0, 1]`.
fn norm_strength(s: i32) -> f32 {
    // card_strength tops out near 1000 (jokers / trump-number); side-suit ranks
    // are <= ~14. Map both bands sensibly: linear for the "normal" band and a
    // saturating tail for the special high cards.
    if s >= 100 {
        // Jokers / trump-number cards: 0.9..1.0.
        0.9 + ((s as f32 - 900.0) / 1000.0).clamp(0.0, 0.1)
    } else {
        (s as f32 / 14.0).clamp(0.0, 1.0)
    }
}

/// Compute the fixed-length HONEST feature vector for `(view, me, cards)`.
///
/// This is the single source of truth for the Expert encoding; the
/// `gen_training_data` example calls it to produce training rows, and
/// [`choose_play_expert`] calls it at inference time, so the two can never
/// disagree. Everything here is derived from the redacted per-player view `p`.
///
/// ## Layout (indices into the returned `[f32; FEATURE_DIM]`)
///
/// Candidate shape:
/// * 0  — number of cards in the candidate / 4
/// * 1  — points in the candidate / 30
/// * 2  — trump cards in the candidate / 4
/// * 3  — max card strength (normalized)
/// * 4  — min card strength (normalized)
/// * 5  — 1 if leading this trick, else 0
/// * 6  — 1 if the candidate follows the led suit, else 0
/// * 7  — 1 if the candidate trumps in (off-suit trump), else 0
/// * 8  — candidate has a point card (0/1)
///
/// Trick context:
/// * 9  — pot points on the table / 30
/// * 10 — our team currently winning (0/1)
/// * 11 — current winner is our teammate (0/1)
/// * 12 — we are the last seat to act (0/1)
/// * 13 — current trick unit size / 4 (0 if leading)
/// * 14 — current winner's top strength (normalized)
/// * 15 — current winner played trump (0/1)
/// * 16 — heuristic estimate: this candidate likely wins the trick (0/1)
/// * 17 — there is a current winner at all (0/1)
///
/// My-hand summary (from my own real cards, which I am allowed to see):
/// * 18 — my hand size / 27
/// * 19 — trumps in my hand / 14
/// * 20 — point cards in my hand / 12
/// * 21 — aces in my hand / 4
/// * 22 — kings in my hand / 4
/// * 23 — jokers in my hand / 4
///
/// Trump info:
/// * 24 — trump is NoTrump (0/1)
/// * 25 — trump number rank / 14 (0 if NT with no number)
///
/// Heuristic prior:
/// * 26 — the heuristic score for this candidate, squashed via tanh
/// * 27 — bias term (always 1.0) so a tiny linear model still has an intercept
///
/// Honest card-memory features (from [`Knowledge::from_play_view`], all derived
/// from the redacted view + public play history — never hidden hands):
/// * 28 — fraction of all trumps still UNSEEN by me (in opponents' hidden hands
///   or the kitty) / total trumps; high ⇒ over-trumping is a real risk
/// * 29 — my trumps as a share of all still-live (unseen + mine) trumps; high ⇒
///   I dominate the trump suit and my trumps/leads are safer
/// * 30 — fraction of the next-to-act opponents that are KNOWN void in the led
///   suit (0 if leading / nobody known void); informs whether a side-suit
///   winner is safe or will be trumped
/// * 31 — at least one opponent yet to act is known void in the led suit (0/1)
/// * 32 — points still unseen (in hidden hands + kitty) / 100; how much is left
///   to fight over in the rest of the hand
/// * 33 — my seat position in the trick: seats that have already acted / 3
///   (0 = I lead, ~1 = I act last); pairs with f12 (am-I-last)
/// * 34 — this candidate's max card is a GUARANTEED current winner given what I
///   can see (no unseen card can beat it in its suit) (0/1)
/// * 35 — game progress: cards already played this hand / deck size (0=start)
pub fn candidate_features(p: &PlayPhase, me: PlayerID, cards: &[Card]) -> [f32; FEATURE_DIM] {
    let mut f = [0.0f32; FEATURE_DIM];
    let trump = p.trick().trump();
    let trick = p.trick();
    let leading = trick.played_cards().is_empty();

    // --- Candidate shape ---
    let n_cards = cards.len();
    let cand_points: i32 = cards
        .iter()
        .filter_map(|c| c.points().map(|x| x as i32))
        .sum();
    let cand_trump = cards
        .iter()
        .filter(|c| trump.effective_suit(**c) == EffectiveSuit::Trump)
        .count();
    let max_strength = cards
        .iter()
        .map(|c| heuristics::legacy_card_strength(trump, *c))
        .max()
        .unwrap_or(0);
    let min_strength = cards
        .iter()
        .map(|c| heuristics::legacy_card_strength(trump, *c))
        .min()
        .unwrap_or(0);

    f[0] = (n_cards as f32 / 4.0).min(1.0);
    f[1] = (cand_points as f32 / 30.0).min(1.0);
    f[2] = (cand_trump as f32 / 4.0).min(1.0);
    f[3] = norm_strength(max_strength);
    f[4] = norm_strength(min_strength);
    f[5] = if leading { 1.0 } else { 0.0 };

    let led_suit = trick.trick_format().map(|tf| tf.suit());
    let following_suit = led_suit
        .map(|s| cards.iter().all(|c| trump.effective_suit(*c) == s))
        .unwrap_or(false);
    let trumping_in = !leading
        && !following_suit
        && cards
            .iter()
            .any(|c| trump.effective_suit(*c) == EffectiveSuit::Trump);
    f[6] = if following_suit { 1.0 } else { 0.0 };
    f[7] = if trumping_in { 1.0 } else { 0.0 };
    f[8] = if cand_points > 0 { 1.0 } else { 0.0 };

    // --- Trick context ---
    let pot_points: i32 = trick
        .played_cards()
        .iter()
        .flat_map(|pc| pc.cards.iter())
        .filter_map(|c| c.points().map(|x| x as i32))
        .sum();
    f[9] = (pot_points as f32 / 30.0).min(1.0);

    let current_winner = trick.winner_so_far();
    let team_winning = current_winner
        .map(|w| heuristics::same_team(p, me, w))
        .unwrap_or(false);
    f[10] = if team_winning { 1.0 } else { 0.0 };
    // Teammate-winning is the same predicate but only when there IS a winner.
    f[11] = if current_winner.is_some() && team_winning {
        1.0
    } else {
        0.0
    };

    let players_left = trick.player_queue().count();
    f[12] = if players_left <= 1 { 1.0 } else { 0.0 };

    let trick_unit_size = trick.trick_format().map(|tf| tf.size()).unwrap_or(0);
    f[13] = (trick_unit_size as f32 / 4.0).min(1.0);

    let winner_top_strength = current_winner
        .and_then(|w| {
            trick.played_cards().iter().find(|pc| pc.id == w).map(|pc| {
                pc.cards
                    .iter()
                    .map(|c| heuristics::legacy_card_strength(trump, *c))
                    .max()
                    .unwrap_or(0)
            })
        })
        .unwrap_or(0);
    f[14] = norm_strength(winner_top_strength);

    let winner_is_trump = current_winner
        .and_then(|w| {
            trick
                .played_cards()
                .iter()
                .find(|pc| pc.id == w)
                .and_then(|pc| pc.cards.first().copied())
        })
        .map(|c| trump.effective_suit(c) == EffectiveSuit::Trump)
        .unwrap_or(false);
    f[15] = if winner_is_trump { 1.0 } else { 0.0 };

    // Heuristic estimate of whether this candidate beats the current winner.
    let likely_win = if leading {
        true
    } else if following_suit {
        (max_strength > winner_top_strength && !winner_is_trump) || current_winner.is_none()
    } else if trumping_in {
        if winner_is_trump {
            max_strength > winner_top_strength
        } else {
            true
        }
    } else {
        false
    };
    f[16] = if likely_win { 1.0 } else { 0.0 };
    f[17] = if current_winner.is_some() { 1.0 } else { 0.0 };

    // --- My-hand summary (my own visible cards) ---
    if let Ok(hand) = p.hands().get(me) {
        let mut hand_size = 0usize;
        let mut trumps = 0usize;
        let mut points = 0usize;
        let mut aces = 0usize;
        let mut kings = 0usize;
        let mut jokers = 0usize;
        for (card, &ct) in hand.iter() {
            hand_size += ct;
            if trump.effective_suit(*card) == EffectiveSuit::Trump {
                trumps += ct;
            }
            if card.points().is_some() {
                points += ct;
            }
            match card {
                Card::BigJoker | Card::SmallJoker => jokers += ct,
                Card::Suited { number, .. } => {
                    if *number == Number::Ace {
                        aces += ct;
                    } else if *number == Number::King {
                        kings += ct;
                    }
                }
                Card::Unknown => {}
            }
        }
        f[18] = (hand_size as f32 / 27.0).min(1.0);
        f[19] = (trumps as f32 / 14.0).min(1.0);
        f[20] = (points as f32 / 12.0).min(1.0);
        f[21] = (aces as f32 / 4.0).min(1.0);
        f[22] = (kings as f32 / 4.0).min(1.0);
        f[23] = (jokers as f32 / 4.0).min(1.0);
    }

    // --- Trump info ---
    f[24] = match trump {
        Trump::NoTrump { .. } => 1.0,
        Trump::Standard { .. } => 0.0,
    };
    f[25] = trump
        .number()
        .map(|num| (num.as_u32() as f32 / 14.0).min(1.0))
        .unwrap_or(0.0);

    // --- Heuristic prior ---
    // FROZEN: this feature was trained against the LEGACY scorer. Keep it on the
    // legacy version so changing the new heuristic doesn't silently shift the
    // net's prior distribution (retrain later to unify).
    let heur = if leading {
        heuristics::score_lead_legacy(p, me, cards)
    } else {
        heuristics::score_follow_legacy(p, me, cards)
    };
    f[26] = (heur as f32 / 10.0).tanh();
    f[27] = 1.0; // bias

    // --- Honest card-memory features (Knowledge from the redacted view) ---
    // `Knowledge` reconstructs, purely from observable info, which cards I have
    // seen (my hand + table + last trick), per-seat established voids, and how
    // many hidden cards each seat holds. We derive a few high-signal aggregates.
    let k = Knowledge::from_play_view(p, me);

    // Trump accounting: total trumps in the deck, how many I can see (mine +
    // played), and therefore how many remain unseen in hidden hands / kitty.
    let seen_trumps: usize = k
        .seen
        .iter()
        .filter(|(c, _)| trump.effective_suit(**c) == EffectiveSuit::Trump)
        .map(|(_, &n)| n)
        .sum();
    // Use the exact configured special-deck multiset; joker-less/short decks
    // must not invent phantom unseen trumps.
    let total_trumps = k.total_trumps;
    let unseen_trumps = total_trumps.saturating_sub(seen_trumps);
    f[28] = if total_trumps > 0 {
        (unseen_trumps as f32 / total_trumps as f32).clamp(0.0, 1.0)
    } else {
        0.0
    };
    // My share of all still-live trumps (mine + unseen). High ⇒ I control trump.
    let my_trumps = f[19] * 14.0; // recover the raw count we stored above
    let my_trumps = my_trumps.round() as usize;
    let live_trumps = my_trumps + unseen_trumps;
    f[29] = if live_trumps > 0 {
        (my_trumps as f32 / live_trumps as f32).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // Void awareness for the seats still to act AFTER me in this trick.
    let led_suit_eff = led_suit;
    let mut yet_to_act: Vec<PlayerID> = trick.player_queue().collect();
    // `player_queue` includes me as the head; drop me so we look at opponents
    // that will respond to this candidate.
    if yet_to_act.first() == Some(&me) {
        yet_to_act.remove(0);
    }
    let n_after = yet_to_act.len();
    if let Some(ls) = led_suit_eff {
        let void_after = yet_to_act
            .iter()
            .filter(|pid| k.voids.get(pid).map(|vs| vs.contains(&ls)).unwrap_or(false))
            .count();
        f[30] = if n_after > 0 {
            (void_after as f32 / n_after as f32).clamp(0.0, 1.0)
        } else {
            0.0
        };
        f[31] = if void_after > 0 { 1.0 } else { 0.0 };
    }

    // Points still unseen (in hidden hands + kitty): total deck points minus the
    // points I have already seen on the table / in my hand / last trick.
    let total_points = k.total_points;
    let mut seen_points = 0usize;
    for (card, &seen) in &k.seen {
        if let Some(pts) = card.points() {
            seen_points += pts * seen;
        }
    }
    let unseen_points = total_points.saturating_sub(seen_points);
    f[32] = (unseen_points as f32 / 100.0).min(1.0);

    // Seat position: how many seats already acted this trick (0 = I lead).
    let acted = trick.played_cards().len();
    f[33] = (acted as f32 / 3.0).min(1.0);

    // Guaranteed winner: this candidate's strongest card cannot be beaten in its
    // effective suit by any card I have NOT seen (so it is currently uncatchable
    // by a same-suit response). Only meaningful when leading or following suit.
    let strongest = cards
        .iter()
        .max_by_key(|c| heuristics::legacy_card_strength(trump, **c))
        .copied();
    f[34] = strongest
        .map(|c| heuristics::is_guaranteed_top(&k, trump, c))
        .map(|g| if g { 1.0 } else { 0.0 })
        .unwrap_or(0.0);

    // Game progress: roughly how much of the hand has been revealed in play.
    // `k.seen` counts my hand + the table + the last trick; subtracting my own
    // hand gives a proxy for cards already played, normalized by the deck size.
    let seen_total: usize = k.seen.values().sum();
    let my_hand = (f[18] * 27.0).round() as usize;
    let revealed = seen_total.saturating_sub(my_hand);
    let deck_size = k.total_cards.max(1);
    f[35] = (revealed as f32 / deck_size as f32).clamp(0.0, 1.0);

    f
}

/// Version-2 honest `(observation, action)` encoding used by the action-value
/// pipeline. Its first 36 entries are the frozen production-policy contract from
/// [`candidate_features`]; the appended entries add signals that were previously
/// aliased away. Keeping the prefix frozen lets experiments measure the added
/// information while the model manifest prevents a v2 network from ever being
/// run with v1 inputs.
///
/// Appended layout:
/// * 36 — actor is on the landlord team;
/// * 37 — realized points so far, oriented to the actor's team / VALUE_NORM;
/// * 38 — progress through the current scoring threshold interval;
/// * 39 — scoring threshold step size / 80;
/// * 40 — exact public game progress from the full played-card log;
/// * 41 — actor's remaining-hand fraction;
/// * 42 — fraction of distinct card identities in the candidate;
/// * 43 — duplicate-card fraction in the candidate (pair/triple structure);
/// * 44 — all candidate cards have one effective suit;
/// * 45 — fraction of all trumps still unseen using full public memory;
/// * 46 — fraction of all point value still unseen using full public memory;
/// * 47 — fraction of possible opponent/effective-suit void facts established;
/// * 48 — exact mechanics-engine answer: this candidate takes the lead now.
pub fn candidate_features_v2(
    p: &PlayPhase,
    me: PlayerID,
    cards: &[Card],
) -> [f32; TRAINING_FEATURE_DIM] {
    let legacy = candidate_features(p, me, cards);
    let mut f = [0.0f32; TRAINING_FEATURE_DIM];
    f[..FEATURE_DIM].copy_from_slice(&legacy);

    let actor_is_landlord = p.landlords_team().contains(&me);
    f[36] = if actor_is_landlord { 1.0 } else { 0.0 };

    let (non_landlord_points, _) = p.calculate_points();
    let oriented = if actor_is_landlord {
        -non_landlord_points
    } else {
        non_landlord_points
    };
    f[37] = (oriented as f64 / VALUE_NORM).clamp(-1.0, 1.0) as f32;

    if let Some(step) = p.bot_step_size().filter(|step| *step > 0) {
        f[38] = (non_landlord_points.rem_euclid(step) as f32 / step as f32).clamp(0.0, 1.0);
        f[39] = (step as f32 / 80.0).clamp(0.0, 1.0);
    }

    let deck_cards = p
        .configured_cards_for_determinization()
        .map(|cards| cards.len())
        .unwrap_or_else(|| 54 * p.num_decks().max(1));
    let completed_cards: usize = p.played_this_hand().values().copied().sum();
    let table_cards: usize = p
        .trick()
        .played_cards()
        .iter()
        .map(|played| played.cards.len())
        .sum();
    f[40] = ((completed_cards + table_cards) as f32 / deck_cards as f32).clamp(0.0, 1.0);
    let hand_size = p
        .hands()
        .get(me)
        .map(|hand| hand.values().copied().sum::<usize>())
        .unwrap_or(0);
    f[41] = (hand_size as f32 / 27.0).clamp(0.0, 1.0);

    if !cards.is_empty() {
        let counts = Card::count(cards.iter().copied());
        let distinct = counts.len();
        f[42] = (distinct as f32 / cards.len() as f32).clamp(0.0, 1.0);
        f[43] = (1.0 - f[42]).clamp(0.0, 1.0);
        let trump = p.trump();
        let first_suit = trump.effective_suit(cards[0]);
        f[44] = if cards
            .iter()
            .all(|card| trump.effective_suit(*card) == first_suit)
        {
            1.0
        } else {
            0.0
        };
    }

    let full = Knowledge::from_play_view(p, me);
    let trump = p.trump();
    let total_trumps = full.total_trumps;
    let seen_trumps: usize = full
        .seen
        .iter()
        .filter(|(card, _)| trump.effective_suit(**card) == EffectiveSuit::Trump)
        .map(|(_, count)| *count)
        .sum();
    if total_trumps > 0 {
        f[45] =
            (total_trumps.saturating_sub(seen_trumps) as f32 / total_trumps as f32).clamp(0.0, 1.0);
    }
    let total_points = full.total_points.max(1);
    let seen_points: usize = full
        .seen
        .iter()
        .map(|(card, count)| card.points().unwrap_or(0) * *count)
        .sum();
    f[46] = (total_points.saturating_sub(seen_points) as f32 / total_points as f32).clamp(0.0, 1.0);
    let known_voids: usize = full
        .voids
        .iter()
        .filter(|(player, _)| **player != me)
        .map(|(_, suits)| suits.len())
        .sum();
    // Three other seats × five effective suits is the maximum useful count.
    f[47] = (known_voids as f32 / 15.0).clamp(0.0, 1.0);

    // Unlike frozen f16, this asks the rules engine to evaluate the complete
    // follow structure. It is therefore correct for pairs, tractors, throws,
    // bombs, trump-led tricks, and mixed forced follows. A lead trivially owns
    // the current trick, matching the engine result after applying the play.
    f[48] = if heuristics::candidate_wins_current_trick(p, me, cards) {
        1.0
    } else {
        0.0
    };

    f
}

#[cfg(test)]
mod model_path_tests {
    use super::*;

    /// The embedded model + the runtime override ([`MODEL_PATH_ENV`]) round-trip,
    /// and an unreadable / placeholder source errors (so the caller falls back to
    /// the heuristic). These call the private `load_model` / `model_from_bytes`
    /// directly so the test is deterministic and does NOT touch the cached
    /// `model()` `OnceLock` other tests rely on. The single test serializes its own
    /// env mutations (sibling tests run in parallel threads).
    #[test]
    fn model_path_override_round_trips() {
        // 1) No override → the embedded net loads (it is a real trained model,
        //    not the <64-byte placeholder).
        std::env::remove_var(MODEL_PATH_ENV);
        assert!(
            load_model().is_ok(),
            "the embedded expert model should load with no override"
        );

        // 2) Override pointing at the real embedded asset → loads identically.
        let real = concat!(env!("CARGO_MANIFEST_DIR"), "/src/bot/expert_model.onnx");
        std::env::set_var(MODEL_PATH_ENV, real);
        assert!(
            load_model().is_ok(),
            "override pointing at the real onnx should load"
        );

        // 3) Override pointing at a missing file → Err (caller falls back to the
        //    heuristic; it does NOT silently use the embedded net).
        std::env::set_var(MODEL_PATH_ENV, "/nonexistent/does-not-exist.onnx");
        assert!(
            load_model().is_err(),
            "a missing override file must error, not silently use the embedded net"
        );

        // Clean up so the cached model() and sibling tests are unaffected.
        std::env::remove_var(MODEL_PATH_ENV);
    }

    /// A placeholder / truncated source is rejected before the ONNX parser runs.
    #[test]
    fn placeholder_bytes_rejected() {
        assert!(
            model_from_bytes(&[0u8; 16]).is_err(),
            "a <64-byte source is not a valid ONNX graph"
        );
    }

    #[test]
    fn manifest_rejects_width_drift_and_untyped_v2_outputs() {
        assert!(parse_manifest(
            r#"{"manifest_version":1,"feature_schema_version":2,"feature_dim":48,
                 "outputs":["score"],"output_semantics":["policy_logit"]}"#
        )
        .is_err());
        assert!(parse_manifest(
            r#"{"manifest_version":1,"feature_schema_version":2,"feature_dim":49,
                 "outputs":["score","state_value","action_q"]}"#
        )
        .is_err());
        let valid = parse_manifest(
            r#"{"manifest_version":1,"feature_schema_version":2,"feature_dim":49,
                 "outputs":["score","state_value","action_q"],
                 "output_semantics":["policy_logit","normalized_level_utility",
                                     "normalized_level_utility"]}"#,
        )
        .expect("typed schema-v2 manifest");
        assert_eq!(valid.feature_dim, TRAINING_FEATURE_DIM);
    }

    /// Backward-compat (the load-bearing value-head safety property): the SHIPPED
    /// embedded model is policy-only, so output index 1 (value) must be ABSENT.
    /// `value_candidates_net` → `run_model_output(.., 1)` therefore returns `None`,
    /// which transparently disables the value blend and keeps the static leaf eval.
    #[test]
    fn embedded_model_has_no_value_output() {
        let model = model_from_bytes(MODEL_BYTES).expect("embedded model should load");
        let n = 2;
        let flat = vec![0.0f32; n * FEATURE_DIM];
        assert!(
            run_model_output(&model, &flat, n, 0).is_some(),
            "embedded model must have a policy output (index 0)"
        );
        assert!(
            run_model_output(&model, &flat, n, 1).is_none(),
            "embedded (policy-only) model must have NO value output (index 1)"
        );
    }

    /// Manual validation that a freshly-trained multi-output value model loads in tract
    /// and exposes a readable `tanh` value output in [-1, 1]. `#[ignore]`d because
    /// it needs an external model file; run after training one:
    ///   SHENGJI_TEST_VALUE_MODEL=/path/to/value_model.onnx \
    ///     cargo +1.92.0 test -p shengji-core --lib value_output_readable -- --ignored
    #[test]
    #[ignore]
    fn value_output_readable_from_multioutput_model() {
        let path = std::env::var("SHENGJI_TEST_VALUE_MODEL")
            .expect("set SHENGJI_TEST_VALUE_MODEL to a multi-output ONNX");
        let bytes = std::fs::read(&path).expect("read value model");
        let manifest_json = std::fs::read_to_string(companion_manifest_path(Path::new(&path)))
            .expect("read companion model manifest");
        let manifest = parse_manifest(&manifest_json).expect("parse companion model manifest");
        let dim = manifest.feature_dim;
        let model = model_from_bytes_with_manifest(&bytes, manifest)
            .expect("multi-output value model should load in tract");
        let n = 3;
        let flat = vec![0.0f32; n * dim];
        let policy = run_model_output(&model, &flat, n, 0).expect("policy output (index 0)");
        let value = run_model_output(&model, &flat, n, 1).expect("value output (index 1)");
        assert_eq!(policy.len(), n);
        assert_eq!(value.len(), n);
        for v in value {
            assert!(
                (-1.0..=1.0).contains(&v),
                "tanh value must be in [-1,1], got {}",
                v
            );
        }
    }
}
