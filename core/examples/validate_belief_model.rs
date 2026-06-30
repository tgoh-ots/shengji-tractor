//! Numerical PyTorch -> ONNX -> tract parity validator for the experimental
//! honest card-location belief model.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;
use sha2::{Digest, Sha256};
use shengji_core::bot::belief::{
    belief_encoder_contract, belief_encoder_source_sha256, belief_feature_names, FEATURE_DIM,
    FEATURE_DIM_V2, GOLDEN_VECTOR_CONTRACT,
};
use tract_onnx::prelude::*;

const TARGET_DIM: usize = 4;
const MODEL_CONTRACT: &str = "offline_honest_card_location_belief";
const GAME_CONTRACT: &str = "tractor:4p:2x-standard:kitty8:no-removed";
const SERVING_STATUS: &str = "experimental_candidate";
const TARGET_CLASSES: [&str; TARGET_DIM] = ["next-seat", "opposite-seat", "previous-seat", "kitty"];
const TRAINING_BEHAVIOUR_POLICY_DOMAIN: &str = "bidding=expert;exchange=easy;play=easy";
const TARGET_SEMANTICS: &str =
    "per-hidden-card destination marginals excluding publicly pinned holdings; rows in a snapshot are correlated physical copies";
const PROPOSAL_FACTORIZATION: &str =
    "per-card destination marginals multiplied over physical-copy assignments; approximate joint";

#[derive(Deserialize)]
struct ModelManifest {
    manifest_version: u32,
    contract: String,
    feature_schema_version: u32,
    feature_dim: usize,
    feature_names: Vec<String>,
    supported_game_contract: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    target_classes: Vec<String>,
    model_sha256: String,
    golden_path: String,
    golden_sha256: String,
    dataset_sha256: String,
    dataset_manifest_sha256: Option<String>,
    dataset_manifest_declared_csv_sha256: Option<String>,
    training_behaviour_policy_domain: String,
    proposal_factorization: String,
    encoder_contract: String,
    encoder_source_sha256: String,
    golden_vector_contract: String,
    research_only: bool,
    auto_promotion: bool,
    #[serde(rename = "unsafe")]
    unsafe_artifact: bool,
    serving_status: String,
}

#[derive(Deserialize)]
struct DatasetManifest {
    csv_sha256: String,
    behaviour_policy_domain: String,
    target_semantics: String,
    publicly_pinned_targets_excluded: bool,
    encoder_contract: String,
    encoder_source_sha256: String,
}

#[derive(Deserialize)]
struct Golden {
    manifest_version: u32,
    vector_contract: String,
    feature_dim: usize,
    target_dim: usize,
    atol: f32,
    rtol: f32,
    features: Vec<Vec<f32>>,
    legality_mask: Vec<Vec<f32>>,
    outputs: BTreeMap<String, Vec<Vec<f32>>>,
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
    let dataset_path = args.next();
    if args.next().is_some() {
        anyhow::bail!("expected model, optional manifest, optional golden, and optional dataset");
    }

    let manifest: ModelManifest = serde_json::from_slice(&std::fs::read(&manifest_path)?)?;
    let golden_bytes = std::fs::read(&golden_path)?;
    let golden: Golden = serde_json::from_slice(&golden_bytes)?;
    validate_contract(&manifest, &golden)?;
    validate_artifact_hashes(&manifest, &model_path, &golden_path, &golden_bytes)?;
    validate_dataset_lineage(&manifest, dataset_path.as_deref())?;

    let n = golden.features.len();
    let flat_features: Vec<f32> = golden.features.into_iter().flatten().collect();
    let flat_masks: Vec<f32> = golden.legality_mask.into_iter().flatten().collect();
    let mut model = tract_onnx::onnx().model_for_path(&model_path)?;
    let batch = model.symbols.sym("N");
    model.set_input_fact(
        0,
        f32::fact([batch.to_dim(), (manifest.feature_dim as i64).to_dim()]).into(),
    )?;
    model.set_input_fact(
        1,
        f32::fact([batch.to_dim(), (TARGET_DIM as i64).to_dim()]).into(),
    )?;
    let runnable = model.into_optimized()?.into_runnable()?;
    let features = tract_ndarray::Array2::from_shape_vec((n, manifest.feature_dim), flat_features)?;
    let masks = tract_ndarray::Array2::from_shape_vec((n, TARGET_DIM), flat_masks)?;
    let features: Tensor = features.into();
    let masks: Tensor = masks.into();
    let result = runnable.run(tvec!(features.into(), masks.into()))?;
    if result.len() != 1 {
        anyhow::bail!("tract returned {} outputs; expected 1", result.len());
    }
    let actual = result[0].to_array_view::<f32>()?;
    if actual.shape() != [n, TARGET_DIM] {
        anyhow::bail!(
            "destination_logits shape {:?} != [{n}, {TARGET_DIM}]",
            actual.shape()
        );
    }
    let expected = golden
        .outputs
        .get("destination_logits")
        .ok_or_else(|| anyhow::anyhow!("golden file has no destination_logits"))?;
    if expected.len() != n
        || expected
            .iter()
            .any(|row| row.len() != TARGET_DIM || row.iter().any(|value| !value.is_finite()))
    {
        anyhow::bail!("golden destination_logits has invalid shape or values");
    }

    let mut worst_absolute = 0.0f32;
    let mut worst_relative = 0.0f32;
    for (index, (&got, &want)) in actual.iter().zip(expected.iter().flatten()).enumerate() {
        if !got.is_finite() {
            anyhow::bail!("tract destination_logits[{index}] is non-finite");
        }
        let absolute = (got - want).abs();
        let relative = absolute / want.abs().max(1e-12);
        worst_absolute = worst_absolute.max(absolute);
        worst_relative = worst_relative.max(relative);
        let tolerance = golden.atol + golden.rtol * want.abs();
        if absolute > tolerance {
            anyhow::bail!(
                "parity failure destination_logits[{index}]: tract={got:.9} +                 pytorch={want:.9} abs={absolute:.3e} tolerance={tolerance:.3e}"
            );
        }
    }
    println!(
        "PASS: belief model {} rows x {} features, worst_abs={:.3e}, worst_rel={:.3e}",
        n, manifest.feature_dim, worst_absolute, worst_relative
    );
    Ok(())
}

fn validate_contract(manifest: &ModelManifest, golden: &Golden) -> TractResult<()> {
    let supported_schema = matches!(
        (
            manifest.manifest_version,
            manifest.feature_schema_version,
            manifest.feature_dim,
        ),
        (1, 1, FEATURE_DIM) | (2, 2, FEATURE_DIM_V2)
    );
    if !supported_schema || manifest.contract != MODEL_CONTRACT {
        anyhow::bail!("unsupported belief model manifest/schema/feature dimension");
    }
    let expected_features = belief_feature_names(manifest.feature_schema_version);
    if manifest.feature_names != expected_features {
        anyhow::bail!("belief feature_names do not match the declared schema");
    }
    if manifest.supported_game_contract != GAME_CONTRACT {
        anyhow::bail!(
            "belief game contract {:?} != {:?}",
            manifest.supported_game_contract,
            GAME_CONTRACT
        );
    }
    if manifest.serving_status != SERVING_STATUS {
        anyhow::bail!(
            "belief serving status {:?} != {:?}",
            manifest.serving_status,
            SERVING_STATUS
        );
    }
    if manifest.training_behaviour_policy_domain != TRAINING_BEHAVIOUR_POLICY_DOMAIN
        || manifest.proposal_factorization != PROPOSAL_FACTORIZATION
        || manifest.encoder_contract
            != belief_encoder_contract(manifest.feature_schema_version).unwrap_or_default()
        || manifest.encoder_source_sha256 != belief_encoder_source_sha256()
        || manifest.golden_vector_contract != GOLDEN_VECTOR_CONTRACT
        || !manifest.research_only
        || manifest.auto_promotion
        || manifest.unsafe_artifact
    {
        anyhow::bail!("belief research, encoder, domain, or proposal contract mismatch");
    }
    if manifest.inputs != ["features", "legality_mask"]
        || manifest.outputs != ["destination_logits"]
        || manifest.target_classes
            != TARGET_CLASSES
                .iter()
                .map(|value| value.to_string())
                .collect::<Vec<_>>()
    {
        anyhow::bail!("belief input/output/target ordering contract mismatch");
    }
    if golden.manifest_version != manifest.manifest_version
        || golden.vector_contract != GOLDEN_VECTOR_CONTRACT
        || golden.feature_dim != manifest.feature_dim
        || golden.target_dim != TARGET_DIM
        || golden.features.is_empty()
        || golden.features.len() != golden.legality_mask.len()
    {
        anyhow::bail!("invalid belief golden header or batch size");
    }
    if golden
        .features
        .iter()
        .any(|row| row.len() != manifest.feature_dim || row.iter().any(|value| !value.is_finite()))
    {
        anyhow::bail!("invalid belief golden features");
    }
    if golden.legality_mask.iter().any(|row| {
        row.len() != TARGET_DIM
            || !row.contains(&1.0)
            || row.iter().any(|value| *value != 0.0 && *value != 1.0)
    }) {
        anyhow::bail!("invalid belief golden legality mask");
    }
    Ok(())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_artifact_hashes(
    manifest: &ModelManifest,
    model_path: &str,
    golden_path: &str,
    golden_bytes: &[u8],
) -> TractResult<()> {
    if !is_sha256(&manifest.model_sha256)
        || sha256_bytes(&std::fs::read(model_path)?) != manifest.model_sha256
    {
        anyhow::bail!("belief model SHA-256 does not match its manifest");
    }
    if !is_sha256(&manifest.golden_sha256) || sha256_bytes(golden_bytes) != manifest.golden_sha256 {
        anyhow::bail!("belief golden SHA-256 does not match its manifest");
    }
    let actual_golden_name = Path::new(golden_path)
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow::anyhow!("belief golden path has no UTF-8 file name"))?;
    if actual_golden_name != manifest.golden_path {
        anyhow::bail!("belief golden file name does not match its manifest");
    }
    Ok(())
}

fn validate_dataset_lineage(
    manifest: &ModelManifest,
    dataset_path: Option<&str>,
) -> TractResult<()> {
    let Some(dataset_manifest_sha256) = manifest.dataset_manifest_sha256.as_deref() else {
        anyhow::bail!("experimental belief candidate is missing dataset-manifest lineage");
    };
    let Some(declared_csv_sha256) = manifest.dataset_manifest_declared_csv_sha256.as_deref() else {
        anyhow::bail!("experimental belief candidate is missing the dataset-declared CSV hash");
    };
    if !is_sha256(&manifest.dataset_sha256)
        || !is_sha256(dataset_manifest_sha256)
        || !is_sha256(declared_csv_sha256)
        || manifest.dataset_sha256 != declared_csv_sha256
    {
        anyhow::bail!("belief dataset lineage hashes are missing, malformed, or inconsistent");
    }

    let Some(dataset_path) = dataset_path else {
        return Ok(());
    };
    let dataset_bytes = std::fs::read(dataset_path)?;
    if sha256_bytes(&dataset_bytes) != manifest.dataset_sha256 {
        anyhow::bail!("belief dataset SHA-256 does not match model lineage");
    }
    let sidecar_path = format!("{dataset_path}.manifest.json");
    let sidecar_bytes = std::fs::read(&sidecar_path)?;
    if sha256_bytes(&sidecar_bytes) != dataset_manifest_sha256 {
        anyhow::bail!("belief dataset-manifest SHA-256 does not match model lineage");
    }
    let sidecar: DatasetManifest = serde_json::from_slice(&sidecar_bytes)?;
    if sidecar.csv_sha256 != manifest.dataset_sha256
        || sidecar.behaviour_policy_domain != TRAINING_BEHAVIOUR_POLICY_DOMAIN
        || sidecar.target_semantics != TARGET_SEMANTICS
        || !sidecar.publicly_pinned_targets_excluded
        || sidecar.encoder_contract != manifest.encoder_contract
        || sidecar.encoder_source_sha256 != manifest.encoder_source_sha256
    {
        anyhow::bail!("belief dataset sidecar does not match model lineage semantics");
    }
    Ok(())
}
