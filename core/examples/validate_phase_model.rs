//! Strict PyTorch -> ONNX -> tract parity validator for bid/kitty phase models.
//!
//! Usage:
//!   cargo run --release -p shengji-core --example validate_phase_model -- \
//!     model.onnx [model.onnx.manifest.json] [model.onnx.golden.json]

use std::fs;
use std::io::Cursor;

use serde::Deserialize;
use sha2::{Digest, Sha256};
use shengji_core::bot::phase::{
    BID_FEATURE_DIM, BID_FEATURE_NAMES, FEATURE_SCHEMA_VERSION, KITTY_FEATURE_DIM,
    KITTY_FEATURE_NAMES,
};
use tract_onnx::prelude::*;

const MANIFEST_VERSION: u32 = 1;
const BID_CONTRACT: &str = "honest_bid_action_ranker";
const KITTY_CONTRACT: &str = "honest_kitty_card_ranker";
const SERVING_STATUS: &str = "experimental_candidate";
const LOGIT_SEMANTICS: &str = "relative_listwise_rank_only";
const MAX_TOLERANCE: f32 = 1e-3;

#[derive(Debug, Deserialize)]
struct ModelManifest {
    manifest_version: u32,
    contract: String,
    feature_schema_version: u32,
    feature_dim: usize,
    feature_names: Vec<String>,
    inputs: Vec<String>,
    outputs: Vec<String>,
    output_semantics: Vec<String>,
    logit_semantics: String,
    training_domain: String,
    model_sha256: String,
    golden_path: String,
    golden_sha256: String,
    dataset_sha256: String,
    dataset_manifest_sha256: Option<String>,
    dataset_manifest_declared_content_sha256: String,
    serving_status: String,
    research_only: bool,
    automatic_production_promotion_allowed: bool,
    unsafe_training_data: bool,
}

#[derive(Debug, Deserialize)]
struct Golden {
    manifest_version: u32,
    phase: String,
    feature_dim: usize,
    inputs: Vec<Vec<f32>>,
    action_logits: Vec<f32>,
    atol: f32,
    rtol: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PhaseContract {
    Bid,
    Kitty,
}

impl PhaseContract {
    fn from_name(name: &str) -> Option<Self> {
        match name {
            BID_CONTRACT => Some(Self::Bid),
            KITTY_CONTRACT => Some(Self::Kitty),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Bid => BID_CONTRACT,
            Self::Kitty => KITTY_CONTRACT,
        }
    }

    fn phase(self) -> &'static str {
        match self {
            Self::Bid => "bid",
            Self::Kitty => "kitty",
        }
    }

    fn feature_dim(self) -> usize {
        match self {
            Self::Bid => BID_FEATURE_DIM,
            Self::Kitty => KITTY_FEATURE_DIM,
        }
    }

    fn feature_names(self) -> Vec<String> {
        match self {
            Self::Bid => BID_FEATURE_NAMES
                .iter()
                .map(|name| (*name).to_owned())
                .collect(),
            Self::Kitty => KITTY_FEATURE_NAMES
                .iter()
                .map(|name| (*name).to_owned())
                .collect(),
        }
    }

    fn training_domain(self) -> &'static str {
        match self {
            Self::Bid => "four_player_tractor_two_full_standard_decks_deal_complete_heuristic_v1",
            Self::Kitty => {
                "four_player_tractor_two_full_standard_decks_initial_exchange_heuristic_v1"
            }
        }
    }
}

fn main() -> TractResult<()> {
    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("missing model.onnx argument"))?;
    let manifest_path = args
        .next()
        .unwrap_or_else(|| format!("{model_path}.manifest.json"));
    let golden_path = args
        .next()
        .unwrap_or_else(|| format!("{model_path}.golden.json"));
    if args.next().is_some() {
        anyhow::bail!("expected at most three arguments");
    }

    let manifest_bytes = fs::read(&manifest_path)?;
    let manifest: ModelManifest = serde_json::from_slice(&manifest_bytes)?;
    let golden_bytes = fs::read(&golden_path)?;
    let golden: Golden = serde_json::from_slice(&golden_bytes)?;
    let contract = validate_contract(&manifest, &golden)?;

    let golden_digest = sha256(&golden_bytes);
    if manifest.golden_sha256 != golden_digest {
        anyhow::bail!(
            "{} golden SHA-256 mismatch: manifest={} actual={golden_digest}",
            contract.phase(),
            manifest.golden_sha256
        );
    }
    let actual_golden_name = std::path::Path::new(&golden_path)
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow::anyhow!("phase golden path has no UTF-8 file name"))?;
    if manifest.golden_path != actual_golden_name {
        anyhow::bail!(
            "{} golden file name does not match manifest",
            contract.phase()
        );
    }

    let model_bytes = fs::read(&model_path)?;
    if model_bytes.len() < 64 {
        anyhow::bail!("{} phase model is too small to be ONNX", contract.phase());
    }
    let model_digest = sha256(&model_bytes);
    if manifest.model_sha256 != model_digest {
        anyhow::bail!(
            "{} model SHA-256 mismatch: manifest={} actual={model_digest}",
            contract.phase(),
            manifest.model_sha256
        );
    }

    let n = golden.inputs.len();
    let flat_inputs = golden.inputs.iter().flatten().copied().collect::<Vec<_>>();
    let mut cursor = Cursor::new(model_bytes);
    let mut model = tract_onnx::onnx().model_for_read(&mut cursor)?;
    let batch = model.symbols.sym("N");
    model.set_input_fact(
        0,
        f32::fact([batch.to_dim(), (contract.feature_dim() as i64).to_dim()]).into(),
    )?;
    let runnable = model.into_optimized()?.into_runnable()?;
    let inputs = tract_ndarray::Array2::from_shape_vec((n, contract.feature_dim()), flat_inputs)?;
    let input: Tensor = inputs.into();
    let result = runnable.run(tvec!(input.into()))?;
    if result.len() != 1 {
        anyhow::bail!("tract returned {} outputs; expected 1", result.len());
    }
    let actual = result[0].to_array_view::<f32>()?;
    if actual.shape() != [n, 1] {
        anyhow::bail!("tract action_logit shape {:?} != [{n}, 1]", actual.shape());
    }

    let mut worst_absolute = 0.0f32;
    let mut worst_relative = 0.0f32;
    for (row, (&got, &want)) in actual.iter().zip(&golden.action_logits).enumerate() {
        if !got.is_finite() {
            anyhow::bail!("tract action_logit[{row}] is non-finite");
        }
        let absolute = (got - want).abs();
        let relative = absolute / want.abs().max(1e-12);
        let tolerance = golden.atol + golden.rtol * want.abs();
        worst_absolute = worst_absolute.max(absolute);
        worst_relative = worst_relative.max(relative);
        if absolute > tolerance {
            anyhow::bail!(
                "parity failure action_logit[{row}]: tract={got:.9} pytorch={want:.9} \
                 abs={absolute:.3e} tolerance={tolerance:.3e}"
            );
        }
    }

    println!(
        "PASS: {} phase model, {} rows x {} features, worst_abs={:.3e}, worst_rel={:.3e}",
        contract.phase(),
        n,
        contract.feature_dim(),
        worst_absolute,
        worst_relative
    );
    Ok(())
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn validate_contract(manifest: &ModelManifest, golden: &Golden) -> TractResult<PhaseContract> {
    let contract = PhaseContract::from_name(&manifest.contract).ok_or_else(|| {
        anyhow::anyhow!("unsupported phase-model contract {:?}", manifest.contract)
    })?;
    if manifest.manifest_version != MANIFEST_VERSION
        || manifest.feature_schema_version != FEATURE_SCHEMA_VERSION
        || manifest.feature_dim != contract.feature_dim()
        || manifest.feature_names != contract.feature_names()
    {
        anyhow::bail!(
            "{} manifest version/dimension/feature-name contract mismatch",
            contract.phase()
        );
    }
    if manifest.contract != contract.name()
        || manifest.inputs != ["features"]
        || manifest.outputs != ["action_logit"]
        || manifest.output_semantics != ["policy_logit"]
        || manifest.logit_semantics != LOGIT_SEMANTICS
        || manifest.training_domain != contract.training_domain()
    {
        anyhow::bail!(
            "{} manifest input/output contract mismatch",
            contract.phase()
        );
    }
    if manifest.serving_status != SERVING_STATUS
        || !manifest.research_only
        || manifest.automatic_production_promotion_allowed
        || manifest.unsafe_training_data
        || !is_sha256(&manifest.dataset_sha256)
        || !is_sha256(&manifest.dataset_manifest_declared_content_sha256)
        || manifest.dataset_sha256 != manifest.dataset_manifest_declared_content_sha256
        || !manifest
            .dataset_manifest_sha256
            .as_deref()
            .is_some_and(is_sha256)
    {
        anyhow::bail!(
            "{} manifest is not a safe experimental candidate",
            contract.phase()
        );
    }

    if golden.manifest_version != MANIFEST_VERSION
        || golden.phase != contract.phase()
        || golden.feature_dim != contract.feature_dim()
        || golden.inputs.is_empty()
        || golden.action_logits.len() != golden.inputs.len()
    {
        anyhow::bail!(
            "{} golden header or tensor shape mismatch",
            contract.phase()
        );
    }
    if golden.inputs.iter().any(|row| {
        row.len() != contract.feature_dim() || row.iter().any(|value| !value.is_finite())
    }) || golden.action_logits.iter().any(|value| !value.is_finite())
    {
        anyhow::bail!(
            "{} golden tensors have invalid shape or values",
            contract.phase()
        );
    }
    if !golden.atol.is_finite()
        || !golden.rtol.is_finite()
        || golden.atol < 0.0
        || golden.rtol < 0.0
        || golden.atol > MAX_TOLERANCE
        || golden.rtol > MAX_TOLERANCE
    {
        anyhow::bail!(
            "{} golden tolerances are invalid or excessive",
            contract.phase()
        );
    }
    Ok(contract)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_manifest(contract: PhaseContract) -> ModelManifest {
        ModelManifest {
            manifest_version: MANIFEST_VERSION,
            contract: contract.name().to_owned(),
            feature_schema_version: FEATURE_SCHEMA_VERSION,
            feature_dim: contract.feature_dim(),
            feature_names: contract.feature_names(),
            inputs: vec!["features".to_owned()],
            outputs: vec!["action_logit".to_owned()],
            output_semantics: vec!["policy_logit".to_owned()],
            logit_semantics: LOGIT_SEMANTICS.to_owned(),
            training_domain: contract.training_domain().to_owned(),
            model_sha256: "00".repeat(32),
            golden_path: "model.onnx.golden.json".to_owned(),
            golden_sha256: "11".repeat(32),
            dataset_sha256: "22".repeat(32),
            dataset_manifest_sha256: Some("33".repeat(32)),
            dataset_manifest_declared_content_sha256: "22".repeat(32),
            serving_status: SERVING_STATUS.to_owned(),
            research_only: true,
            automatic_production_promotion_allowed: false,
            unsafe_training_data: false,
        }
    }

    fn valid_golden(contract: PhaseContract) -> Golden {
        Golden {
            manifest_version: MANIFEST_VERSION,
            phase: contract.phase().to_owned(),
            feature_dim: contract.feature_dim(),
            inputs: vec![vec![0.0; contract.feature_dim()]],
            action_logits: vec![0.0],
            atol: 2e-5,
            rtol: 2e-5,
        }
    }

    #[test]
    fn accepts_exact_bid_and_kitty_contracts() {
        for contract in [PhaseContract::Bid, PhaseContract::Kitty] {
            assert_eq!(
                validate_contract(&valid_manifest(contract), &valid_golden(contract)).unwrap(),
                contract
            );
        }
    }

    #[test]
    fn rejects_placeholder_feature_names() {
        let mut manifest = valid_manifest(PhaseContract::Bid);
        manifest.feature_names = (0..BID_FEATURE_DIM)
            .map(|index| format!("bid.f{index}"))
            .collect();
        assert!(validate_contract(&manifest, &valid_golden(PhaseContract::Bid)).is_err());
    }

    #[test]
    fn rejects_nonfinite_or_vacuous_golden_values() {
        let manifest = valid_manifest(PhaseContract::Kitty);
        let mut golden = valid_golden(PhaseContract::Kitty);
        golden.action_logits[0] = f32::NAN;
        assert!(validate_contract(&manifest, &golden).is_err());
        golden.action_logits[0] = 0.0;
        golden.atol = 1.0;
        assert!(validate_contract(&manifest, &golden).is_err());
    }
}
