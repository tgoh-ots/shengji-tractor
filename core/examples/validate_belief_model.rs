//! Numerical PyTorch -> ONNX -> tract parity validator for the experimental
//! honest card-location belief model.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufReader;

use serde::Deserialize;
use tract_onnx::prelude::*;

const FEATURE_DIM: usize = 20;
const TARGET_DIM: usize = 4;
const MODEL_CONTRACT: &str = "offline_honest_card_location_belief";
const GAME_CONTRACT: &str = "tractor:4p:2x-standard:kitty8:no-removed";
const SERVING_STATUS: &str = "experimental_candidate";
const TARGET_CLASSES: [&str; TARGET_DIM] = ["next-seat", "opposite-seat", "previous-seat", "kitty"];

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
    serving_status: String,
}

#[derive(Deserialize)]
struct Golden {
    manifest_version: u32,
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
    if args.next().is_some() {
        anyhow::bail!("expected at most three arguments");
    }

    let manifest: ModelManifest =
        serde_json::from_reader(BufReader::new(File::open(&manifest_path)?))?;
    let golden: Golden = serde_json::from_reader(BufReader::new(File::open(&golden_path)?))?;
    validate_contract(&manifest, &golden)?;

    let n = golden.features.len();
    let flat_features: Vec<f32> = golden.features.into_iter().flatten().collect();
    let flat_masks: Vec<f32> = golden.legality_mask.into_iter().flatten().collect();
    let mut model = tract_onnx::onnx().model_for_path(&model_path)?;
    let batch = model.symbols.sym("N");
    model.set_input_fact(
        0,
        f32::fact([batch.to_dim(), (FEATURE_DIM as i64).to_dim()]).into(),
    )?;
    model.set_input_fact(
        1,
        f32::fact([batch.to_dim(), (TARGET_DIM as i64).to_dim()]).into(),
    )?;
    let runnable = model.into_optimized()?.into_runnable()?;
    let features = tract_ndarray::Array2::from_shape_vec((n, FEATURE_DIM), flat_features)?;
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
        n, FEATURE_DIM, worst_absolute, worst_relative
    );
    Ok(())
}

fn validate_contract(manifest: &ModelManifest, golden: &Golden) -> TractResult<()> {
    if manifest.manifest_version != 1
        || manifest.contract != MODEL_CONTRACT
        || manifest.feature_schema_version != 1
        || manifest.feature_dim != FEATURE_DIM
    {
        anyhow::bail!("unsupported belief model manifest/schema/feature dimension");
    }
    let expected_features: Vec<String> = (0..FEATURE_DIM).map(|i| format!("b{i}")).collect();
    if manifest.feature_names != expected_features {
        anyhow::bail!("belief feature_names must be exactly b0..b19");
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
    if golden.manifest_version != 1
        || golden.feature_dim != FEATURE_DIM
        || golden.target_dim != TARGET_DIM
        || golden.features.is_empty()
        || golden.features.len() != golden.legality_mask.len()
    {
        anyhow::bail!("invalid belief golden header or batch size");
    }
    if golden
        .features
        .iter()
        .any(|row| row.len() != FEATURE_DIM || row.iter().any(|value| !value.is_finite()))
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
