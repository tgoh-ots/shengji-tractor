//! Numerical PyTorch -> ONNX -> tract parity check for an Expert model.
//!
//! Usage:
//!   cargo run --release -p shengji-core --example validate_expert_model -- \
//!     model.onnx model.onnx.manifest.json model.onnx.golden.json

use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufReader;

use serde::Deserialize;
use tract_onnx::prelude::*;

#[derive(Deserialize)]
struct ModelManifest {
    feature_dim: usize,
    outputs: Vec<String>,
}

#[derive(Deserialize)]
struct Golden {
    manifest_version: u32,
    feature_dim: usize,
    atol: f32,
    rtol: f32,
    inputs: Vec<Vec<f32>>,
    outputs: BTreeMap<String, Vec<f32>>,
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
    if golden.manifest_version != 1 {
        anyhow::bail!(
            "unsupported golden manifest version {}",
            golden.manifest_version
        );
    }
    if golden.feature_dim != manifest.feature_dim {
        anyhow::bail!(
            "golden feature_dim {} != model manifest feature_dim {}",
            golden.feature_dim,
            manifest.feature_dim
        );
    }
    if golden.inputs.is_empty() {
        anyhow::bail!("golden input batch is empty");
    }
    if golden
        .inputs
        .iter()
        .any(|row| row.len() != manifest.feature_dim || row.iter().any(|v| !v.is_finite()))
    {
        anyhow::bail!("golden inputs have wrong width or non-finite values");
    }

    let n = golden.inputs.len();
    let flat: Vec<f32> = golden.inputs.into_iter().flatten().collect();
    let mut model = tract_onnx::onnx().model_for_path(&model_path)?;
    let batch = model.symbols.sym("N");
    model.set_input_fact(
        0,
        f32::fact([batch.to_dim(), (manifest.feature_dim as i64).to_dim()]).into(),
    )?;
    let runnable = model.into_optimized()?.into_runnable()?;
    let input = tract_ndarray::Array2::from_shape_vec((n, manifest.feature_dim), flat)?;
    let tensor: Tensor = input.into();
    let result = runnable.run(tvec!(tensor.into()))?;
    if result.len() != manifest.outputs.len() {
        anyhow::bail!(
            "tract returned {} outputs; manifest declares {}",
            result.len(),
            manifest.outputs.len()
        );
    }

    let mut worst_absolute = 0.0f32;
    let mut worst_relative = 0.0f32;
    for (index, name) in manifest.outputs.iter().enumerate() {
        let expected = golden
            .outputs
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("golden file has no {name} output"))?;
        if expected.len() != n || expected.iter().any(|v| !v.is_finite()) {
            anyhow::bail!("golden {name} output must contain {n} finite scalars");
        }
        let actual = result[index].to_array_view::<f32>()?;
        if actual.shape() != [n, 1] {
            anyhow::bail!("tract {name} shape {:?} != [{n}, 1]", actual.shape());
        }
        for (row, (&got, &want)) in actual.iter().zip(expected).enumerate() {
            if !got.is_finite() {
                anyhow::bail!("tract {name}[{row}] is non-finite");
            }
            let absolute = (got - want).abs();
            let relative = absolute / want.abs().max(1e-12);
            worst_absolute = worst_absolute.max(absolute);
            worst_relative = worst_relative.max(relative);
            if absolute > golden.atol + golden.rtol * want.abs() {
                anyhow::bail!(
                    "parity failure {name}[{row}]: tract={got:.9} pytorch={want:.9} abs={absolute:.3e} tolerance={:.3e}",
                    golden.atol + golden.rtol * want.abs()
                );
            }
        }
    }
    println!(
        "PASS: {} rows x {} features, outputs={:?}, worst_abs={:.3e}, worst_rel={:.3e}",
        n, manifest.feature_dim, manifest.outputs, worst_absolute, worst_relative
    );
    Ok(())
}
