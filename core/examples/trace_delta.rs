//! Paired cross-checkout delta analysis for `version_ab --trace` and
//! `refinement_search_ab --trace` output.
//!
//! Usage:
//!   cargo run --release --example trace_delta -- BASELINE CANDIDATE TAG

use std::collections::BTreeMap;
use std::env;
use std::fs;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

type Observation = [f64; 3];

fn read_trace(path: &str, tag: &str) -> BTreeMap<usize, Observation> {
    let contents =
        fs::read_to_string(path).unwrap_or_else(|error| panic!("read {}: {}", path, error));
    contents
        .lines()
        .filter_map(|line| {
            let mut fields = line.split('\t');
            if fields.next()? != "TRACE" || fields.next()? != tag {
                return None;
            }
            let index = fields.next()?.parse().ok()?;
            let win = fields.next()?.parse().ok()?;
            let margin = fields.next()?.parse().ok()?;
            let level = fields.next()?.parse().ok()?;
            Some((index, [win, margin, level]))
        })
        .collect()
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn bootstrap_ci(values: &[f64], iterations: usize, seed: u64) -> (f64, f64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut means = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let total = (0..values.len())
            .map(|_| values[rng.gen_range(0..values.len())])
            .sum::<f64>();
        means.push(total / values.len() as f64);
    }
    means.sort_by(f64::total_cmp);
    let low = ((iterations as f64 * 0.025).floor() as usize).min(iterations - 1);
    let high = ((iterations as f64 * 0.975).floor() as usize).min(iterations - 1);
    (means[low], means[high])
}

fn sign_flip_p(values: &[f64], iterations: usize, seed: u64) -> f64 {
    let observed = mean(values).abs();
    let mut rng = StdRng::seed_from_u64(seed);
    let extreme = (0..iterations)
        .filter(|_| {
            let flipped = values
                .iter()
                .map(|value| if rng.gen_bool(0.5) { *value } else { -*value })
                .sum::<f64>()
                / values.len() as f64;
            flipped.abs() >= observed
        })
        .count();
    (extreme + 1) as f64 / (iterations + 1) as f64
}

fn report(name: &str, values: &[f64], scale: f64, suffix: &str, seed: u64) {
    let (low, high) = bootstrap_ci(values, 10_000, seed);
    let p = sign_flip_p(values, 50_000, seed ^ 0x51A7_F11F);
    println!(
        "{name}: {:+.3}{suffix}  paired-bootstrap95 [{:+.3}, {:+.3}]{suffix}  sign-flip p={p:.4}",
        mean(values) * scale,
        low * scale,
        high * scale,
    );
}

fn main() {
    let args = env::args().collect::<Vec<_>>();
    let baseline_path = args.get(1).expect("baseline trace path");
    let candidate_path = args.get(2).expect("candidate trace path");
    let tag = args.get(3).map(String::as_str).unwrap_or("heuristic");

    let baseline = read_trace(baseline_path, tag);
    let candidate = read_trace(candidate_path, tag);
    assert!(
        !baseline.is_empty(),
        "no baseline TRACE rows for tag {}",
        tag
    );
    assert_eq!(
        baseline.keys().collect::<Vec<_>>(),
        candidate.keys().collect::<Vec<_>>(),
        "baseline and candidate traces must contain identical paired-deck indices"
    );

    let mut deltas = [Vec::new(), Vec::new(), Vec::new()];
    for (index, before) in &baseline {
        let after = candidate[index];
        for metric in 0..3 {
            deltas[metric].push(after[metric] - before[metric]);
        }
    }

    println!(
        "PAIRED CROSS-CHECKOUT DELTA: candidate - baseline, tag={tag}, decks={}",
        baseline.len()
    );
    report("win rate", &deltas[0], 100.0, "pp", 0xD311_A001);
    report("point margin", &deltas[1], 1.0, " pts/hand", 0xD311_A002);
    report(
        "level utility",
        &deltas[2],
        1.0,
        " levels/hand",
        0xD311_A003,
    );
}
