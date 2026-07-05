//! Machine-readable, direct Enoch candidate-versus-control evaluation.
//!
//! The candidate and frozen Enoch-0 control play in the same process and on the
//! same mirrored deals. Enoch hypotheses are carried in `SearchConfig`, so no
//! process-global feature switch can contaminate the other partnership.
//!
//! Fixed-work example:
//! ```text
//! cargo run --release -p shengji-core --example enoch_eval -- \
//!   --pairs 800 --base-seed 0x5eed \
//!   --features bid-ownership,compound-follow \
//!   --worlds 8 --candidates 6 --rollout-tricks 12 \
//!   --fixed-work --deadline-ms 30000
//! ```
//!
//! Product-budget example:
//! ```text
//! cargo run --release -p shengji-core --example enoch_eval -- \
//!   --pairs 800 --base-seed 0x5eed --features default-kitty \
//!   --worlds 144 --candidates 6 --rollout-tricks 12 \
//!   --budget-ms 2200
//! ```

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::convert::{TryFrom, TryInto};
use std::env;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process;
use std::time::Duration;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use rand::rngs::StdRng;
use rand::SeedableRng;
use shengji_core::bot::enoch::EnochFeatures;
use shengji_core::bot::harness::{
    bootstrap_mean_ci, mean, minimum_detectable_effect, play_one_hand_with_config,
    play_one_hand_with_config_audited, Contestant, HandResult, HarnessConfig, PlayBrain,
    PlayDecision, Seat,
};
use shengji_core::bot::search::{Policy, SearchConfig, SearchTelemetry};
use shengji_core::bot::BotDifficulty;
use shengji_core::game_state::play_phase::PlayPhase;
use shengji_core::settings::{
    FriendSelectionPolicy, GameModeSettings, KittyPenalty, MultipleJoinPolicy, ThrowPenalty,
};
use shengji_mechanics::deck::Deck;
use shengji_mechanics::scoring::{BonusLevelPolicy, GameScoringParameters};
use shengji_mechanics::trick::TrickFormat;
use shengji_mechanics::types::{Card, EffectiveSuit, Number, PlayerID, Rank};

const BOOTSTRAP_ITERATIONS: usize = 10_000;
const Z_ALPHA_95: f64 = 1.96;
const Z_POWER_80: f64 = 0.84;
const WEEK1_PROTOCOL_KIND: &str = "enoch-week1-seed-protocol";
const WEEK1_SELECTION_KIND: &str = "enoch-week1-seed-selection";
const WEEK1_DERIVATION_DOMAIN: &str = "shengji/enoch-week1/seed/v1";
const WEEK1_DERIVATION_DOMAIN_BYTES: &[u8] = b"shengji/enoch-week1/seed/v1\0";
const BLOCKED_ENV_PREFIXES: [&str; 4] = ["SHENGJI_", "GM_", "OMNI_", "GEN_"];
const RUNNER_STYLE_METRICS: [&str; 9] = [
    "compound-format-follow-rate",
    "empty-trick-ruff-rate",
    "failed-throw-rate",
    "follow-rate",
    "lead-rate",
    "multi-card-play-rate",
    "point-card-play-rate",
    "throw-rate",
    "trump-play-rate",
];
const QUALIFICATION_SCENARIOS: [(&str, &str); 21] = [
    ("intended", "qual/intended"),
    ("equal", "qual/equal"),
    ("rank-low", "qual/rank/low"),
    ("rank-middle", "qual/rank/middle"),
    ("rank-high", "qual/rank/high"),
    ("crossplay-assignment-01", "qual/crossplay/assignment-01"),
    ("crossplay-assignment-02", "qual/crossplay/assignment-02"),
    ("crossplay-assignment-03", "qual/crossplay/assignment-03"),
    ("crossplay-assignment-04", "qual/crossplay/assignment-04"),
    ("configuration-slot-01", "qual/configuration/slot-01"),
    ("configuration-slot-02", "qual/configuration/slot-02"),
    ("configuration-slot-03", "qual/configuration/slot-03"),
    (
        "finding-friends-contract-01",
        "qual/finding-friends/contract-01",
    ),
    (
        "finding-friends-contract-02",
        "qual/finding-friends/contract-02",
    ),
    ("scoring-kitty-ruleset-01", "qual/scoring/kitty-ruleset-01"),
    ("scoring-kitty-ruleset-02", "qual/scoring/kitty-ruleset-02"),
    ("scoring-kitty-ruleset-03", "qual/scoring/kitty-ruleset-03"),
    ("threshold-situation-01", "qual/threshold/situation-01"),
    ("threshold-situation-02", "qual/threshold/situation-02"),
    ("threshold-situation-03", "qual/threshold/situation-03"),
    ("threshold-situation-04", "qual/threshold/situation-04"),
];
const THRESHOLD_SELECTION_MAX_ATTEMPTS: u64 = 256;
const THRESHOLD_DEAL_DOMAIN: &[u8] = b"shengji/enoch-week1/threshold-deal/v1\0";

#[derive(Clone, Debug)]
struct SeedEntry {
    registry_index: Option<u64>,
    seed: u64,
}

#[derive(Clone, Debug)]
struct ProtocolIdentity {
    domain_status: String,
    protocol_kind: Option<String>,
    manifest_version: Option<u64>,
    protocol_fingerprint: Option<String>,
    seed_registry_sha256: Option<String>,
    derivation_domain: Option<String>,
    master_seed: Option<u64>,
    namespace: Option<String>,
    registry_namespace_count: Option<usize>,
    environment_allowlist: Vec<String>,
    environment_policy_verified: bool,
}

impl ProtocolIdentity {
    fn unverified(namespace: Option<String>, status: &str) -> Self {
        Self {
            domain_status: status.to_owned(),
            protocol_kind: None,
            manifest_version: None,
            protocol_fingerprint: None,
            seed_registry_sha256: None,
            derivation_domain: None,
            master_seed: None,
            namespace,
            registry_namespace_count: None,
            environment_allowlist: Vec::new(),
            environment_policy_verified: false,
        }
    }
}

#[derive(Clone, Debug)]
struct SeedPlan {
    entries: Vec<SeedEntry>,
    source_kind: String,
    source_path: Option<String>,
    protocol: ProtocolIdentity,
}

#[derive(Debug)]
struct ValidatedRegistry {
    master_seed: u64,
    registry_sha256: String,
    namespaces: BTreeMap<String, Vec<u64>>,
}

#[derive(Clone, Debug)]
enum AuditRouting {
    Static(Vec<EvaluationArm>),
    DynamicTeam { candidate_is_landlord_team: bool },
}

#[derive(Clone, Debug)]
struct OrientationPlan {
    name: &'static str,
    seats: Vec<Seat>,
    routing: AuditRouting,
    candidate_is_landlord_team: bool,
}

#[derive(Clone, Copy, Debug)]
struct ThresholdBand {
    minimum_inclusive: isize,
    maximum_exclusive: Option<isize>,
}

impl ThresholdBand {
    fn contains(self, points: isize) -> bool {
        points >= self.minimum_inclusive
            && self
                .maximum_exclusive
                .map(|maximum| points < maximum)
                .unwrap_or(true)
    }
}

#[derive(Clone, Debug)]
struct ScenarioPlan {
    id: String,
    expected_namespace: Option<&'static str>,
    config: HarnessConfig,
    orientations: [OrientationPlan; 2],
    threshold_band: Option<ThresholdBand>,
    identity: Value,
}

#[derive(Clone, Copy, Debug)]
struct DealSelection {
    registry_seed: u64,
    effective_seed: u64,
    selection_attempt: u64,
    selector_non_landlord_points: Option<isize>,
}

#[derive(Clone, Copy, Debug)]
enum WorkMode {
    FixedWork { deadline_ms: u64 },
    Budget { budget_ms: u64 },
}

impl WorkMode {
    fn time_budget_ms(self) -> u64 {
        match self {
            WorkMode::FixedWork { deadline_ms } => deadline_ms,
            WorkMode::Budget { budget_ms } => budget_ms,
        }
    }

    fn name(self) -> &'static str {
        match self {
            WorkMode::FixedWork { .. } => "fixed-work",
            WorkMode::Budget { .. } => "budget",
        }
    }

    fn time_budget_kind(self) -> &'static str {
        match self {
            WorkMode::FixedWork { .. } => "safety-deadline",
            WorkMode::Budget { .. } => "wall-clock-budget",
        }
    }
}

#[derive(Debug)]
struct Args {
    pairs: usize,
    seed_plan: SeedPlan,
    environment_identity_only: bool,
    feature_input: String,
    features: EnochFeatures,
    worlds: usize,
    candidates: usize,
    rollout_tricks: usize,
    runner_style_metrics: Vec<String>,
    scenario_id: String,
    mode: WorkMode,
}

fn usage() -> &'static str {
    "usage:\n  enoch_eval --pairs N SEEDS --features SPEC \\\n     --worlds N --candidates N --rollout-tricks N \\\n     (--fixed-work --deadline-ms MS | --budget-ms MS)\n\nSEEDS is exactly one of:\n  --base-seed SEED\n  --seed SEED [--seed SEED ...] [--seed-domain NAMESPACE]\n  --seeds-json FILE [--seed-namespace NAMESPACE] [--seed-index INDEX ...]\n\n`--seeds-json` (aliases: --seed-file, --seed-registry) accepts the frozen\nWeek-1 protocol/registry or an exact JSON seed array/selection. Registry inputs\nrequire --seed-namespace; repeated --seed-index selects a noncontiguous shard.\nSEED is decimal or 0x-prefixed hexadecimal. SPEC is a comma-separated list\nfrom EnochFeatures (or `none` / `all`). --pairs must equal the exact seed count."
}

fn style_usage() -> &'static str {
    "Optional runner bridge:\n  --style-metric NAME [--style-metric NAME ...]\n  --environment-identity-only\n\nSupported NAME values:\n  compound-format-follow-rate, empty-trick-ruff-rate, failed-throw-rate,\n  follow-rate, lead-rate, multi-card-play-rate, point-card-play-rate,\n  throw-rate, trump-play-rate\n\n`--environment-identity-only` validates the normal command and frozen protocol,\nemits only the environment contract and hash, and performs no scenario/deal/game work."
}

fn scenario_usage() -> &'static str {
    "Scenario:\n  --scenario ID\n\nID is standard, development-finding-friends for the friend-revelation arm,\nor one frozen W1.5 comparison ID:\n  intended, equal, rank-low, rank-middle, rank-high,\n  crossplay-assignment-01 .. crossplay-assignment-04,\n  configuration-slot-01 .. configuration-slot-03,\n  finding-friends-contract-01 .. finding-friends-contract-02,\n  scoring-kitty-ruleset-01 .. scoring-kitty-ruleset-03,\n  threshold-situation-01 .. threshold-situation-04"
}

fn take_value(args: &[String], index: &mut usize, flag: &str) -> Result<String, String> {
    *index += 1;
    args.get(*index)
        .cloned()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn set_once<T>(slot: &mut Option<T>, value: T, flag: &str) -> Result<(), String> {
    if slot.is_some() {
        return Err(format!("{flag} was provided more than once"));
    }
    *slot = Some(value);
    Ok(())
}

fn positive_usize(value: &str, flag: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("{flag} must be a positive integer, got {value:?}"))
}

fn positive_u64(value: &str, flag: &str) -> Result<u64, String> {
    parse_u64(value)
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("{flag} must be a positive integer, got {value:?}"))
}

fn parse_u64(value: &str) -> Option<u64> {
    let value = value.trim();
    if let Some(hex) = value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
    {
        u64::from_str_radix(hex, 16).ok()
    } else {
        value.parse().ok()
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn canonical_json_sha256(value: &Value) -> Result<String, String> {
    serde_json::to_vec(value)
        .map(|bytes| sha256_hex(&bytes))
        .map_err(|error| format!("canonicalize JSON for SHA-256: {error}"))
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = File::open(path)
        .map_err(|error| format!("open {} for hashing: {error}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("read {} for hashing: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn object<'a>(value: &'a Value, label: &str) -> Result<&'a serde_json::Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} must be a JSON object"))
}

fn array<'a>(value: &'a Value, label: &str) -> Result<&'a Vec<Value>, String> {
    value
        .as_array()
        .ok_or_else(|| format!("{label} must be a JSON array"))
}

fn exact_keys(
    value: &serde_json::Map<String, Value>,
    expected: &[&str],
    label: &str,
) -> Result<(), String> {
    let actual: BTreeSet<&str> = value.keys().map(String::as_str).collect();
    let expected: BTreeSet<&str> = expected.iter().copied().collect();
    if actual == expected {
        Ok(())
    } else {
        Err(format!(
            "{label} keys differ; expected={expected:?}, actual={actual:?}"
        ))
    }
}

fn value_u64(value: &Value, label: &str) -> Result<u64, String> {
    match value {
        Value::Number(number) => number
            .as_u64()
            .ok_or_else(|| format!("{label} must be an unsigned 64-bit integer")),
        Value::String(text) => parse_u64(text).ok_or_else(|| {
            format!("{label} must be decimal or 0x-prefixed unsigned 64-bit integer")
        }),
        _ => Err(format!("{label} must be an unsigned 64-bit integer")),
    }
}

fn value_usize(value: &Value, label: &str) -> Result<usize, String> {
    usize::try_from(value_u64(value, label)?)
        .map_err(|_| format!("{label} is too large for this platform"))
}

fn required<'a>(
    value: &'a serde_json::Map<String, Value>,
    key: &str,
    label: &str,
) -> Result<&'a Value, String> {
    value
        .get(key)
        .ok_or_else(|| format!("{label} is missing required field {key:?}"))
}

fn required_string(
    value: &serde_json::Map<String, Value>,
    key: &str,
    label: &str,
) -> Result<String, String> {
    required(value, key, label)?
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| format!("{label}.{key} must be a string"))
}

fn optional_string(
    value: &serde_json::Map<String, Value>,
    keys: &[&str],
    label: &str,
) -> Result<Option<String>, String> {
    let mut selected: Option<String> = None;
    for key in keys {
        if let Some(raw) = value.get(*key) {
            let candidate = raw
                .as_str()
                .ok_or_else(|| format!("{label}.{key} must be a string"))?
                .to_owned();
            if let Some(previous) = &selected {
                if previous != &candidate {
                    return Err(format!("{label} has conflicting namespace fields"));
                }
            } else {
                selected = Some(candidate);
            }
        }
    }
    Ok(selected)
}

fn validate_namespace(namespace: &str) -> Result<(), String> {
    let bytes = namespace.as_bytes();
    let valid_edge = |byte: u8| byte.is_ascii_lowercase() || byte.is_ascii_digit();
    let valid_body = |byte: u8| valid_edge(byte) || matches!(byte, b'.' | b'/' | b'-');
    if bytes.is_empty()
        || !valid_edge(bytes[0])
        || !valid_edge(bytes[bytes.len() - 1])
        || !bytes.iter().copied().all(valid_body)
        || namespace.contains("//")
        || namespace.contains("/./")
        || namespace.contains("/../")
    {
        return Err(format!("invalid Week-1 seed namespace {namespace:?}"));
    }
    Ok(())
}

fn derive_week1_seed(master_seed: u64, namespace: &str, index: u64) -> Result<u64, String> {
    validate_namespace(namespace)?;
    let namespace_bytes = namespace.as_bytes();
    let namespace_len = u32::try_from(namespace_bytes.len())
        .map_err(|_| "Week-1 seed namespace exceeds u32 bytes".to_owned())?;
    let mut digest = Sha256::new();
    digest.update(WEEK1_DERIVATION_DOMAIN_BYTES);
    digest.update(master_seed.to_be_bytes());
    digest.update(namespace_len.to_be_bytes());
    digest.update(namespace_bytes);
    digest.update(index.to_be_bytes());
    let digest = digest.finalize();
    Ok(u64::from_be_bytes(
        digest[..8]
            .try_into()
            .expect("SHA-256 always contains at least eight bytes"),
    ))
}

fn expected_derivation() -> Value {
    json!({
        "algorithm": "sha256-first-64-bits",
        "byte_order": "big",
        "domain": WEEK1_DERIVATION_DOMAIN,
        "index_encoding": "u64-big-endian",
        "master_seed_encoding": "u64-big-endian",
        "namespace_encoding": "u32-length-prefixed-utf8",
    })
}

fn validate_registry(
    registry_value: &Value,
    declared_sha256: Option<&str>,
) -> Result<ValidatedRegistry, String> {
    let registry = object(registry_value, "seed registry")?;
    exact_keys(
        registry,
        &[
            "derivation",
            "global_seed_count",
            "master_seed",
            "namespaces",
        ],
        "seed registry",
    )?;
    if required(registry, "derivation", "seed registry")? != &expected_derivation() {
        return Err("Week-1 seed derivation domain/contract mismatch".to_owned());
    }
    let master_seed = value_u64(
        required(registry, "master_seed", "seed registry")?,
        "seed registry master_seed",
    )?;
    let namespace_values = array(
        required(registry, "namespaces", "seed registry")?,
        "seed registry namespaces",
    )?;
    let mut namespaces = BTreeMap::new();
    let mut global_seeds = HashSet::new();
    let mut observed_count = 0usize;

    for (namespace_position, namespace_value) in namespace_values.iter().enumerate() {
        let label = format!("seed registry namespace[{namespace_position}]");
        let namespace_entry = object(namespace_value, &label)?;
        exact_keys(namespace_entry, &["count", "name", "seeds"], &label)?;
        let name = required_string(namespace_entry, "name", &label)?;
        validate_namespace(&name)?;
        let count = value_usize(
            required(namespace_entry, "count", &label)?,
            "namespace count",
        )?;
        let seeds = array(
            required(namespace_entry, "seeds", &label)?,
            "namespace seeds",
        )?;
        if seeds.len() != count {
            return Err(format!(
                "{label} declares {count} seeds but contains {}",
                seeds.len()
            ));
        }
        let mut parsed = Vec::with_capacity(seeds.len());
        for (index, value) in seeds.iter().enumerate() {
            let seed = value_u64(value, &format!("{name}[{index}]"))?;
            let expected = derive_week1_seed(master_seed, &name, index as u64)?;
            if seed != expected {
                return Err(format!(
                    "Week-1 derivation mismatch for {name}[{index}]: expected {expected}, got {seed}"
                ));
            }
            if !global_seeds.insert(seed) {
                return Err(format!("duplicate seed {seed} in Week-1 registry"));
            }
            parsed.push(seed);
        }
        observed_count = observed_count
            .checked_add(count)
            .ok_or_else(|| "seed registry global count overflow".to_owned())?;
        if namespaces.insert(name.clone(), parsed).is_some() {
            return Err(format!("duplicate Week-1 namespace {name:?}"));
        }
    }
    let declared_count = value_usize(
        required(registry, "global_seed_count", "seed registry")?,
        "seed registry global_seed_count",
    )?;
    if declared_count != observed_count {
        return Err(format!(
            "seed registry global count mismatch: declared {declared_count}, observed {observed_count}"
        ));
    }

    let registry_sha256 = canonical_json_sha256(registry_value)?;
    if let Some(declared) = declared_sha256 {
        if !is_sha256(declared) || declared != registry_sha256 {
            return Err(format!(
                "seed registry SHA-256 mismatch: declared {declared:?}, actual {registry_sha256}"
            ));
        }
    }
    Ok(ValidatedRegistry {
        master_seed,
        registry_sha256,
        namespaces,
    })
}

fn validate_environment_policy(value: &Value) -> Result<Vec<String>, String> {
    let policy = object(value, "evaluator environment policy")?;
    exact_keys(
        policy,
        &["allowlist", "blocked_prefixes"],
        "evaluator environment policy",
    )?;
    let blocked = array(
        required(policy, "blocked_prefixes", "environment policy")?,
        "environment blocked_prefixes",
    )?;
    let parsed_blocked: Result<Vec<&str>, String> = blocked
        .iter()
        .map(|value| {
            value
                .as_str()
                .ok_or_else(|| "environment blocked prefix must be a string".to_owned())
        })
        .collect();
    if parsed_blocked? != BLOCKED_ENV_PREFIXES {
        return Err("Week-1 evaluator environment blocked-prefix domain mismatch".to_owned());
    }
    let allowlist = array(
        required(policy, "allowlist", "environment policy")?,
        "environment allowlist",
    )?;
    let mut parsed = Vec::with_capacity(allowlist.len());
    for value in allowlist {
        let name = value
            .as_str()
            .ok_or_else(|| "environment allowlist entry must be a string".to_owned())?;
        if !BLOCKED_ENV_PREFIXES
            .iter()
            .any(|prefix| name.starts_with(prefix))
        {
            return Err(format!(
                "allowlisted evaluator variable {name:?} is outside the blocked domain"
            ));
        }
        parsed.push(name.to_owned());
    }
    let mut sorted = parsed.clone();
    sorted.sort();
    sorted.dedup();
    if sorted != parsed {
        return Err("evaluator environment allowlist must be sorted and unique".to_owned());
    }
    Ok(parsed)
}

fn select_registry_entries(
    registry: &ValidatedRegistry,
    namespace: &str,
    indices: &[u64],
) -> Result<Vec<SeedEntry>, String> {
    validate_namespace(namespace)?;
    let seeds = registry
        .namespaces
        .get(namespace)
        .ok_or_else(|| format!("unknown Week-1 seed namespace {namespace:?}"))?;
    if indices.is_empty() {
        return seeds
            .iter()
            .copied()
            .enumerate()
            .map(|(index, seed)| {
                Ok(SeedEntry {
                    registry_index: Some(
                        u64::try_from(index)
                            .map_err(|_| "registry index does not fit in u64".to_owned())?,
                    ),
                    seed,
                })
            })
            .collect();
    }
    let mut seen = HashSet::new();
    let mut selected = Vec::with_capacity(indices.len());
    for &index in indices {
        if !seen.insert(index) {
            return Err(format!("duplicate requested seed index {index}"));
        }
        let position = usize::try_from(index)
            .map_err(|_| format!("seed index {index} is too large for this platform"))?;
        let seed = seeds.get(position).copied().ok_or_else(|| {
            format!(
                "seed index {index} is outside namespace {namespace:?} (count {})",
                seeds.len()
            )
        })?;
        selected.push(SeedEntry {
            registry_index: Some(index),
            seed,
        });
    }
    Ok(selected)
}

fn parse_week1_protocol(
    root: &Value,
    namespace: Option<String>,
    indices: &[u64],
    source_path: String,
) -> Result<SeedPlan, String> {
    let protocol = object(root, "Week-1 seed protocol")?;
    exact_keys(
        protocol,
        &[
            "automatic_production_promotion_allowed",
            "evaluator_environment_policy",
            "manifest_version",
            "protocol_fingerprint",
            "protocol_kind",
            "seed_registry",
            "seed_registry_sha256",
        ],
        "Week-1 seed protocol",
    )?;
    let manifest_version = value_u64(
        required(protocol, "manifest_version", "Week-1 seed protocol")?,
        "protocol manifest_version",
    )?;
    if manifest_version != 1 {
        return Err(format!(
            "unsupported Week-1 protocol manifest version {manifest_version}"
        ));
    }
    let protocol_kind = required_string(protocol, "protocol_kind", "Week-1 seed protocol")?;
    if protocol_kind != WEEK1_PROTOCOL_KIND {
        return Err(format!(
            "seed protocol domain mismatch: expected {WEEK1_PROTOCOL_KIND:?}, got {protocol_kind:?}"
        ));
    }
    if required(
        protocol,
        "automatic_production_promotion_allowed",
        "Week-1 seed protocol",
    )? != &Value::Bool(false)
    {
        return Err("Week-1 protocol must disable automatic production promotion".to_owned());
    }
    let environment_allowlist = validate_environment_policy(required(
        protocol,
        "evaluator_environment_policy",
        "Week-1 seed protocol",
    )?)?;
    let declared_registry_sha =
        required_string(protocol, "seed_registry_sha256", "Week-1 seed protocol")?;
    let registry_value = required(protocol, "seed_registry", "Week-1 seed protocol")?;
    let registry = validate_registry(registry_value, Some(&declared_registry_sha))?;

    let protocol_fingerprint =
        required_string(protocol, "protocol_fingerprint", "Week-1 seed protocol")?;
    if !is_sha256(&protocol_fingerprint) {
        return Err("Week-1 protocol fingerprint must be lowercase SHA-256".to_owned());
    }
    let mut body = root.clone();
    object(&body, "Week-1 seed protocol")?;
    body.as_object_mut()
        .expect("object was checked")
        .remove("protocol_fingerprint");
    let actual_fingerprint = canonical_json_sha256(&body)?;
    if protocol_fingerprint != actual_fingerprint {
        return Err(format!(
            "Week-1 protocol fingerprint mismatch: declared {protocol_fingerprint}, actual {actual_fingerprint}"
        ));
    }

    let namespace =
        namespace.ok_or_else(|| "--seed-namespace is required for a Week-1 registry".to_owned())?;
    let entries = select_registry_entries(&registry, &namespace, indices)?;
    let registry_namespace_count = registry
        .namespaces
        .get(&namespace)
        .map(Vec::len)
        .expect("selection verified the namespace");
    Ok(SeedPlan {
        entries,
        source_kind: "week1-protocol-registry".to_owned(),
        source_path: Some(source_path),
        protocol: ProtocolIdentity {
            domain_status: "verified-week1-protocol".to_owned(),
            protocol_kind: Some(protocol_kind),
            manifest_version: Some(manifest_version),
            protocol_fingerprint: Some(protocol_fingerprint),
            seed_registry_sha256: Some(registry.registry_sha256),
            derivation_domain: Some(WEEK1_DERIVATION_DOMAIN.to_owned()),
            master_seed: Some(registry.master_seed),
            namespace: Some(namespace),
            registry_namespace_count: Some(registry_namespace_count),
            environment_allowlist,
            environment_policy_verified: true,
        },
    })
}

fn parse_registry_only(
    root: &Value,
    namespace: Option<String>,
    indices: &[u64],
    source_path: String,
) -> Result<SeedPlan, String> {
    let registry = validate_registry(root, None)?;
    let namespace =
        namespace.ok_or_else(|| "--seed-namespace is required for a Week-1 registry".to_owned())?;
    let entries = select_registry_entries(&registry, &namespace, indices)?;
    let registry_namespace_count = registry
        .namespaces
        .get(&namespace)
        .map(Vec::len)
        .expect("selection verified the namespace");
    Ok(SeedPlan {
        entries,
        source_kind: "week1-registry".to_owned(),
        source_path: Some(source_path),
        protocol: ProtocolIdentity {
            domain_status: "verified-week1-registry".to_owned(),
            protocol_kind: None,
            manifest_version: None,
            protocol_fingerprint: None,
            seed_registry_sha256: Some(registry.registry_sha256),
            derivation_domain: Some(WEEK1_DERIVATION_DOMAIN.to_owned()),
            master_seed: Some(registry.master_seed),
            namespace: Some(namespace),
            registry_namespace_count: Some(registry_namespace_count),
            environment_allowlist: Vec::new(),
            environment_policy_verified: false,
        },
    })
}

fn parse_seed_value(
    value: &Value,
    ordinal: usize,
    expected_namespace: Option<&str>,
) -> Result<SeedEntry, String> {
    if !value.is_object() {
        return Ok(SeedEntry {
            registry_index: None,
            seed: value_u64(value, &format!("seed[{ordinal}]"))?,
        });
    }
    let entry = object(value, &format!("seed[{ordinal}]"))?;
    let raw_seed = entry
        .get("seed")
        .or_else(|| entry.get("seed_u64"))
        .ok_or_else(|| format!("seed[{ordinal}] is missing seed/seed_u64"))?;
    let seed = value_u64(raw_seed, &format!("seed[{ordinal}]"))?;
    let registry_index = match (entry.get("index"), entry.get("registry_index")) {
        (Some(left), Some(right)) => {
            let left = value_u64(left, &format!("seed[{ordinal}].index"))?;
            let right = value_u64(right, &format!("seed[{ordinal}].registry_index"))?;
            if left != right {
                return Err(format!("seed[{ordinal}] has conflicting index fields"));
            }
            Some(left)
        }
        (Some(value), None) | (None, Some(value)) => {
            Some(value_u64(value, &format!("seed[{ordinal}].index"))?)
        }
        (None, None) => None,
    };
    if let Some(record_namespace) =
        optional_string(entry, &["namespace", "seed_namespace"], "seed")?
    {
        if let Some(expected) = expected_namespace {
            if record_namespace != expected {
                return Err(format!(
                    "seed[{ordinal}] namespace {record_namespace:?} does not match {expected:?}"
                ));
            }
        }
    }
    Ok(SeedEntry {
        registry_index,
        seed,
    })
}

fn parse_exact_seed_json(
    root: &Value,
    cli_namespace: Option<String>,
    indices: &[u64],
    source_path: String,
) -> Result<SeedPlan, String> {
    if !indices.is_empty() {
        return Err("--seed-index is only valid with a full Week-1 registry".to_owned());
    }
    let (seed_values, object_metadata) = match root {
        Value::Array(values) => (values, None),
        Value::Object(metadata) => {
            let values = array(
                required(metadata, "seeds", "exact seed selection")?,
                "exact seed selection seeds",
            )?;
            (values, Some(metadata))
        }
        _ => return Err("seed JSON root must be an array or object".to_owned()),
    };

    let embedded_namespace = match object_metadata {
        Some(metadata) => optional_string(
            metadata,
            &["namespace", "seed_namespace", "name"],
            "seed selection",
        )?,
        None => None,
    };
    if let (Some(cli), Some(embedded)) = (&cli_namespace, &embedded_namespace) {
        if cli != embedded {
            return Err(format!(
                "CLI seed namespace {cli:?} does not match JSON namespace {embedded:?}"
            ));
        }
    }
    let namespace = cli_namespace.or(embedded_namespace);
    if let Some(namespace) = namespace.as_deref() {
        validate_namespace(namespace)?;
    }
    if let Some(metadata) = object_metadata {
        if let Some(count) = metadata.get("count") {
            let declared = value_usize(count, "seed selection count")?;
            if declared != seed_values.len() {
                return Err(format!(
                    "seed selection count mismatch: declared {declared}, observed {}",
                    seed_values.len()
                ));
            }
        }
    }
    let entries: Result<Vec<_>, _> = seed_values
        .iter()
        .enumerate()
        .map(|(ordinal, value)| parse_seed_value(value, ordinal, namespace.as_deref()))
        .collect();

    let mut identity = ProtocolIdentity::unverified(
        namespace,
        if object_metadata.is_some() {
            "declared-json-selection"
        } else {
            "unverified-json-list"
        },
    );
    if let Some(metadata) = object_metadata {
        if let Some(kind) = metadata.get("protocol_kind") {
            let kind = kind
                .as_str()
                .ok_or_else(|| "seed selection protocol_kind must be a string".to_owned())?;
            if kind != WEEK1_SELECTION_KIND {
                return Err(format!(
                    "exact seed selection domain mismatch: expected {WEEK1_SELECTION_KIND:?}, got {kind:?}"
                ));
            }
            identity.protocol_kind = Some(kind.to_owned());
        }
        if let Some(version) = metadata.get("manifest_version") {
            let version = value_u64(version, "seed selection manifest_version")?;
            if version != 1 {
                return Err(format!(
                    "unsupported seed selection manifest version {version}"
                ));
            }
            identity.manifest_version = Some(version);
        }
        if let Some(fingerprint) = metadata.get("protocol_fingerprint") {
            let fingerprint = fingerprint
                .as_str()
                .ok_or_else(|| "protocol_fingerprint must be a string".to_owned())?;
            if !is_sha256(fingerprint) {
                return Err("protocol_fingerprint must be lowercase SHA-256".to_owned());
            }
            identity.protocol_fingerprint = Some(fingerprint.to_owned());
        }
        if let Some(registry_sha) = metadata.get("seed_registry_sha256") {
            let registry_sha = registry_sha
                .as_str()
                .ok_or_else(|| "seed_registry_sha256 must be a string".to_owned())?;
            if !is_sha256(registry_sha) {
                return Err("seed_registry_sha256 must be lowercase SHA-256".to_owned());
            }
            identity.seed_registry_sha256 = Some(registry_sha.to_owned());
        }
        if let Some(domain) = metadata.get("derivation_domain") {
            let domain = domain
                .as_str()
                .ok_or_else(|| "derivation_domain must be a string".to_owned())?;
            if domain != WEEK1_DERIVATION_DOMAIN {
                return Err(format!(
                    "seed derivation domain mismatch: expected {WEEK1_DERIVATION_DOMAIN:?}, got {domain:?}"
                ));
            }
            identity.derivation_domain = Some(domain.to_owned());
        }
        identity.registry_namespace_count = metadata
            .get("count")
            .map(|value| value_usize(value, "seed selection count"))
            .transpose()?;
    }
    Ok(SeedPlan {
        entries: entries?,
        source_kind: "exact-json-selection".to_owned(),
        source_path: Some(source_path),
        protocol: identity,
    })
}

fn load_seed_json(
    path: &Path,
    namespace: Option<String>,
    indices: &[u64],
) -> Result<SeedPlan, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    let root: Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse seed JSON {}: {error}", path.display()))?;
    let source_path = path.display().to_string();
    if root.get("seed_registry").is_some() {
        parse_week1_protocol(&root, namespace, indices, source_path)
    } else if root.get("namespaces").is_some() {
        parse_registry_only(&root, namespace, indices, source_path)
    } else {
        parse_exact_seed_json(&root, namespace, indices, source_path)
    }
}

fn validate_seed_plan(plan: &SeedPlan, pairs: usize) -> Result<(), String> {
    if plan.entries.len() != pairs {
        return Err(format!(
            "--pairs {pairs} does not match exact seed count {}",
            plan.entries.len()
        ));
    }
    let mut seen_seeds = HashSet::new();
    let mut seen_registry_indices = HashSet::new();
    for (position, entry) in plan.entries.iter().enumerate() {
        if !seen_seeds.insert(entry.seed) {
            return Err(format!(
                "duplicate seed {} at requested paired index {position}",
                entry.seed
            ));
        }
        if let Some(registry_index) = entry.registry_index {
            if !seen_registry_indices.insert(registry_index) {
                return Err(format!(
                    "duplicate registry index {registry_index} at requested paired index {position}"
                ));
            }
        }
    }
    Ok(())
}

fn parse_args() -> Result<Args, String> {
    let raw: Vec<String> = env::args().skip(1).collect();
    if raw.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("{}\n\n{}\n\n{}", usage(), style_usage(), scenario_usage());
        process::exit(0);
    }

    let mut pairs = None;
    let mut base_seed = None;
    let mut explicit_seeds = Vec::new();
    let mut seed_json_path = None;
    let mut seed_namespace = None;
    let mut seed_indices = Vec::new();
    let mut feature_input = None;
    let mut worlds = None;
    let mut candidates = None;
    let mut rollout_tricks = None;
    let mut runner_style_metrics = Vec::new();
    let mut scenario_id = None;
    let mut environment_identity_only = false;
    let mut fixed_work = false;
    let mut deadline_ms = None;
    let mut budget_ms = None;

    let mut index = 0usize;
    while index < raw.len() {
        let flag = raw[index].as_str();
        match flag {
            "--pairs" => {
                let value = take_value(&raw, &mut index, flag)?;
                set_once(&mut pairs, positive_usize(&value, flag)?, flag)?;
            }
            "--base-seed" => {
                let value = take_value(&raw, &mut index, flag)?;
                let parsed = parse_u64(&value).ok_or_else(|| {
                    format!("{flag} must be decimal or 0x-prefixed hexadecimal, got {value:?}")
                })?;
                set_once(&mut base_seed, parsed, flag)?;
            }
            "--seed" => {
                let value = take_value(&raw, &mut index, flag)?;
                let parsed = parse_u64(&value).ok_or_else(|| {
                    format!("{flag} must be decimal or 0x-prefixed hexadecimal, got {value:?}")
                })?;
                explicit_seeds.push(parsed);
            }
            "--seeds-json" | "--seed-file" | "--seed-registry" => {
                let value = take_value(&raw, &mut index, flag)?;
                set_once(&mut seed_json_path, PathBuf::from(value), flag)?;
            }
            "--seed-namespace" | "--seed-domain" => {
                let value = take_value(&raw, &mut index, flag)?;
                validate_namespace(&value)?;
                set_once(&mut seed_namespace, value, flag)?;
            }
            "--seed-index" => {
                let value = take_value(&raw, &mut index, flag)?;
                let parsed = parse_u64(&value).ok_or_else(|| {
                    format!("{flag} must be decimal or 0x-prefixed hexadecimal, got {value:?}")
                })?;
                seed_indices.push(parsed);
            }
            "--features" | "--candidate-features" => {
                let value = take_value(&raw, &mut index, flag)?;
                set_once(&mut feature_input, value, flag)?;
            }
            "--worlds" => {
                let value = take_value(&raw, &mut index, flag)?;
                set_once(&mut worlds, positive_usize(&value, flag)?, flag)?;
            }
            "--candidates" => {
                let value = take_value(&raw, &mut index, flag)?;
                set_once(&mut candidates, positive_usize(&value, flag)?, flag)?;
            }
            "--rollout-tricks" => {
                let value = take_value(&raw, &mut index, flag)?;
                set_once(&mut rollout_tricks, positive_usize(&value, flag)?, flag)?;
            }
            "--style-metric" => {
                let value = take_value(&raw, &mut index, flag)?;
                if !RUNNER_STYLE_METRICS.contains(&value.as_str()) {
                    return Err(format!(
                        "unsupported {flag} {value:?}; supported names are {}",
                        RUNNER_STYLE_METRICS.join(",")
                    ));
                }
                if runner_style_metrics
                    .iter()
                    .any(|existing| existing == &value)
                {
                    return Err(format!("{flag} {value:?} was provided more than once"));
                }
                runner_style_metrics.push(value);
            }
            "--scenario" => {
                let value = take_value(&raw, &mut index, flag)?;
                if value != "standard"
                    && value != "development-finding-friends"
                    && !QUALIFICATION_SCENARIOS
                        .iter()
                        .any(|(scenario, _)| *scenario == value)
                {
                    return Err(format!("unknown frozen scenario {value:?}"));
                }
                set_once(&mut scenario_id, value, flag)?;
            }
            "--environment-identity-only" => {
                if environment_identity_only {
                    return Err(
                        "--environment-identity-only was provided more than once".to_owned()
                    );
                }
                environment_identity_only = true;
            }
            "--fixed-work" => {
                if fixed_work {
                    return Err("--fixed-work was provided more than once".to_owned());
                }
                fixed_work = true;
            }
            "--deadline-ms" => {
                let value = take_value(&raw, &mut index, flag)?;
                set_once(&mut deadline_ms, positive_u64(&value, flag)?, flag)?;
            }
            "--budget-ms" => {
                let value = take_value(&raw, &mut index, flag)?;
                set_once(&mut budget_ms, positive_u64(&value, flag)?, flag)?;
            }
            _ => return Err(format!("unknown argument {flag:?}")),
        }
        index += 1;
    }

    let pairs = pairs.ok_or_else(|| "--pairs is required".to_owned())?;
    let feature_input = feature_input.ok_or_else(|| "--features is required".to_owned())?;
    let features = EnochFeatures::parse(&feature_input)?;
    let worlds = worlds.ok_or_else(|| "--worlds is required".to_owned())?;
    let candidates = candidates.ok_or_else(|| "--candidates is required".to_owned())?;
    let rollout_tricks = rollout_tricks.ok_or_else(|| "--rollout-tricks is required".to_owned())?;

    let mode = match (fixed_work, deadline_ms, budget_ms) {
        (true, Some(deadline_ms), None) => WorkMode::FixedWork { deadline_ms },
        (false, None, Some(budget_ms)) => WorkMode::Budget { budget_ms },
        (true, None, None) => {
            return Err("--fixed-work requires an explicit --deadline-ms".to_owned())
        }
        (false, Some(_), None) => return Err("--deadline-ms requires --fixed-work".to_owned()),
        (false, None, None) => {
            return Err("choose --fixed-work with --deadline-ms, or --budget-ms".to_owned())
        }
        _ => {
            return Err(
                "fixed-work/deadline and wall-clock budget settings are mutually exclusive"
                    .to_owned(),
            )
        }
    };

    let seed_source_count = usize::from(base_seed.is_some())
        + usize::from(!explicit_seeds.is_empty())
        + usize::from(seed_json_path.is_some());
    if seed_source_count != 1 {
        return Err(
            "choose exactly one seed source: --base-seed, repeated --seed, or --seeds-json"
                .to_owned(),
        );
    }
    let seed_plan = if let Some(base_seed) = base_seed {
        if seed_namespace.is_some() || !seed_indices.is_empty() {
            return Err(
                "--seed-namespace/--seed-index cannot be combined with --base-seed".to_owned(),
            );
        }
        let last_index = u64::try_from(pairs - 1)
            .map_err(|_| "--pairs is too large for a u64 seed sequence".to_owned())?;
        base_seed
            .checked_add(last_index)
            .ok_or_else(|| "base seed plus pair count overflows u64".to_owned())?;
        SeedPlan {
            entries: (0..pairs)
                .map(|index| SeedEntry {
                    registry_index: None,
                    seed: base_seed + index as u64,
                })
                .collect(),
            source_kind: "legacy-contiguous-base-seed".to_owned(),
            source_path: None,
            protocol: ProtocolIdentity::unverified(None, "legacy-contiguous-unverified-domain"),
        }
    } else if !explicit_seeds.is_empty() {
        if !seed_indices.is_empty() {
            return Err("--seed-index cannot be combined with repeated --seed".to_owned());
        }
        SeedPlan {
            entries: explicit_seeds
                .into_iter()
                .map(|seed| SeedEntry {
                    registry_index: None,
                    seed,
                })
                .collect(),
            source_kind: "repeated-cli-seed".to_owned(),
            source_path: None,
            protocol: ProtocolIdentity::unverified(
                seed_namespace,
                "caller-declared-or-unknown-domain",
            ),
        }
    } else {
        load_seed_json(
            seed_json_path
                .as_deref()
                .expect("seed source count requires a JSON path"),
            seed_namespace,
            &seed_indices,
        )?
    };
    validate_seed_plan(&seed_plan, pairs)?;
    runner_style_metrics.sort();
    let scenario_id = scenario_id.unwrap_or_else(|| "standard".to_owned());

    Ok(Args {
        pairs,
        seed_plan,
        environment_identity_only,
        feature_input,
        features,
        worlds,
        candidates,
        rollout_tricks,
        runner_style_metrics,
        scenario_id,
        mode,
    })
}

fn search_config(args: &Args, features: EnochFeatures) -> SearchConfig {
    SearchConfig {
        time_budget: Duration::from_millis(args.mode.time_budget_ms()),
        require_full_work: matches!(args.mode, WorkMode::FixedWork { .. }),
        max_candidates: args.candidates,
        max_worlds: args.worlds,
        rollout_tricks: args.rollout_tricks,
        // The harness replaces this with a seed derived from the honest root
        // observation before every decision.
        seed: 0,
        policy: Policy::EnochHeuristic,
        rollout_policy: Policy::EnochHeuristic,
        enoch_features: features,
    }
}

fn contestant(args: &Args, features: EnochFeatures, label: String) -> Contestant {
    let candidate_uses_default_kitty =
        !features.is_empty() && features.contains(EnochFeatures::DEFAULT_KITTY);
    Contestant::new(
        label,
        Seat {
            play: PlayBrain::SearchStrict(search_config(args, features)),
            bid: BotDifficulty::Enoch,
            kitty: if candidate_uses_default_kitty {
                // Easy and the plain heuristic tier share the deterministic
                // point/shape burial, without Expert's optional phase model.
                BotDifficulty::Easy
            } else {
                BotDifficulty::Enoch
            },
        },
    )
}

#[derive(Clone, Copy, Debug)]
enum Checkpoint {
    Candidate,
    Control,
}

impl Checkpoint {
    fn name(self) -> &'static str {
        match self {
            Self::Candidate => "candidate",
            Self::Control => "Enoch-0",
        }
    }

    fn features(self, args: &Args) -> EnochFeatures {
        match self {
            Self::Candidate => args.features,
            Self::Control => EnochFeatures::empty(),
        }
    }
}

fn checkpoint_seat(args: &Args, checkpoint: Checkpoint) -> Seat {
    contestant(
        args,
        checkpoint.features(args),
        checkpoint.name().to_owned(),
    )
    .seat
}

fn audit_arm_name(arm: EvaluationArm) -> &'static str {
    match arm {
        EvaluationArm::Candidate => "candidate-subject-team",
        EvaluationArm::Control => "control-opponent-team",
    }
}

fn checkpoint_json(args: &Args, checkpoint: Checkpoint) -> Value {
    let features = checkpoint.features(args);
    json!({
        "checkpoint": checkpoint.name(),
        "features": feature_json(features),
        "play_brain": "SearchStrict",
        "bid": "Enoch",
        "kitty": if features.contains(EnochFeatures::DEFAULT_KITTY) {
            "DefaultHeuristic"
        } else {
            "Enoch"
        },
    })
}

fn harness_config_json(config: &HarnessConfig) -> Value {
    json!({
        "num_players": config.num_players,
        "deck_count": config.decks.len(),
        "deck_contract": "standard-54-card-deck repeated deck_count times",
        "game_mode": config.game_mode,
        "rank": config.rank,
        "friend_selection_policy": config.friend_selection_policy,
        "multiple_join_policy": config.multiple_join_policy,
        "kitty_penalty": config.kitty_penalty,
        "throw_penalty": config.throw_penalty,
        "game_scoring_parameters": config.game_scoring_parameters,
        "tractor_requirements": config.tractor_requirements,
        "trick_draw_policy": config.trick_draw_policy,
        "bomb_policy": config.bomb_policy,
        "compound_formats": config.compound_formats,
    })
}

fn static_orientation(
    args: &Args,
    name: &'static str,
    checkpoints: Vec<Checkpoint>,
    arms: Vec<EvaluationArm>,
    roles: Vec<&'static str>,
    candidate_is_landlord_team: bool,
) -> Result<(OrientationPlan, Value), String> {
    if checkpoints.len() != arms.len() || checkpoints.len() != roles.len() {
        return Err(format!("{name} static seat declaration length mismatch"));
    }
    let seats: Vec<Seat> = checkpoints
        .iter()
        .copied()
        .map(|checkpoint| checkpoint_seat(args, checkpoint))
        .collect();
    let seat_identity: Vec<Value> = checkpoints
        .iter()
        .copied()
        .zip(arms.iter().copied())
        .zip(roles.iter().copied())
        .enumerate()
        .map(|(seat_index, ((checkpoint, arm), role))| {
            json!({
                "seat_index": seat_index,
                "role": role,
                "audit_arm": audit_arm_name(arm),
                "brain": checkpoint_json(args, checkpoint),
            })
        })
        .collect();
    Ok((
        OrientationPlan {
            name,
            seats,
            routing: AuditRouting::Static(arms),
            candidate_is_landlord_team,
        },
        json!({
            "name": name,
            "candidate_is_landlord_team": candidate_is_landlord_team,
            "audit_routing": "static-seat-subject-partnership",
            "seats": seat_identity,
        }),
    ))
}

fn standard_orientations(
    args: &Args,
    num_players: usize,
) -> Result<([OrientationPlan; 2], [Value; 2]), String> {
    let candidate_landlord_checkpoints = (0..num_players)
        .map(|index| {
            if index % 2 == 0 {
                Checkpoint::Candidate
            } else {
                Checkpoint::Control
            }
        })
        .collect::<Vec<_>>();
    let control_landlord_checkpoints = (0..num_players)
        .map(|index| {
            if index % 2 == 0 {
                Checkpoint::Control
            } else {
                Checkpoint::Candidate
            }
        })
        .collect::<Vec<_>>();
    let candidate_landlord_arms = (0..num_players)
        .map(|index| {
            if index % 2 == 0 {
                EvaluationArm::Candidate
            } else {
                EvaluationArm::Control
            }
        })
        .collect::<Vec<_>>();
    let control_landlord_arms = (0..num_players)
        .map(|index| {
            if index % 2 == 0 {
                EvaluationArm::Control
            } else {
                EvaluationArm::Candidate
            }
        })
        .collect::<Vec<_>>();
    let candidate_landlord_roles = (0..num_players)
        .map(|index| {
            if index % 2 == 0 {
                "candidate-landlord-team"
            } else {
                "control-attacker-team"
            }
        })
        .collect::<Vec<_>>();
    let control_landlord_roles = (0..num_players)
        .map(|index| {
            if index % 2 == 0 {
                "control-landlord-team"
            } else {
                "candidate-attacker-team"
            }
        })
        .collect::<Vec<_>>();
    let (first, first_identity) = static_orientation(
        args,
        "candidate-landlord-team",
        candidate_landlord_checkpoints,
        candidate_landlord_arms,
        candidate_landlord_roles,
        true,
    )?;
    let (second, second_identity) = static_orientation(
        args,
        "control-landlord-team",
        control_landlord_checkpoints,
        control_landlord_arms,
        control_landlord_roles,
        false,
    )?;
    Ok(([first, second], [first_identity, second_identity]))
}

fn crossplay_orientations(
    args: &Args,
    partner: Checkpoint,
    opponent: Checkpoint,
) -> Result<([OrientationPlan; 2], [Value; 2]), String> {
    let (first, first_identity) = static_orientation(
        args,
        "focal-candidate-landlord-team",
        vec![Checkpoint::Candidate, opponent, partner, opponent],
        vec![
            EvaluationArm::Candidate,
            EvaluationArm::Control,
            EvaluationArm::Candidate,
            EvaluationArm::Control,
        ],
        vec![
            "focal-candidate",
            "opponent-checkpoint",
            "partner-checkpoint",
            "opponent-checkpoint",
        ],
        true,
    )?;
    let (second, second_identity) = static_orientation(
        args,
        "focal-candidate-attacker-team",
        vec![opponent, Checkpoint::Candidate, opponent, partner],
        vec![
            EvaluationArm::Control,
            EvaluationArm::Candidate,
            EvaluationArm::Control,
            EvaluationArm::Candidate,
        ],
        vec![
            "opponent-checkpoint",
            "focal-candidate",
            "opponent-checkpoint",
            "partner-checkpoint",
        ],
        false,
    )?;
    Ok(([first, second], [first_identity, second_identity]))
}

fn dynamic_team_orientation(
    args: &Args,
    name: &'static str,
    num_players: usize,
    candidate_is_landlord_team: bool,
) -> (OrientationPlan, Value) {
    let candidate_config = search_config(args, args.features);
    let control_config = search_config(args, EnochFeatures::empty());
    let (landlord_team, non_landlord_team) = if candidate_is_landlord_team {
        (candidate_config, control_config)
    } else {
        (control_config, candidate_config)
    };
    let landlord_checkpoint = if candidate_is_landlord_team {
        Checkpoint::Candidate
    } else {
        Checkpoint::Control
    };
    let non_landlord_checkpoint = if candidate_is_landlord_team {
        Checkpoint::Control
    } else {
        Checkpoint::Candidate
    };
    let kitty = checkpoint_seat(args, landlord_checkpoint).kitty;
    let seats = (0..num_players)
        .map(|_| Seat {
            play: PlayBrain::SearchStrictByTeam {
                landlord_team,
                non_landlord_team,
            },
            bid: BotDifficulty::Enoch,
            kitty,
        })
        .collect::<Vec<_>>();
    let seat_identity = (0..num_players)
        .map(|seat_index| {
            json!({
                "seat_index": seat_index,
                "role": "dynamic-after-friend-revelation",
                "play_brain": "SearchStrictByTeam",
                "landlord_team_brain": checkpoint_json(args, landlord_checkpoint),
                "non_landlord_team_brain": checkpoint_json(args, non_landlord_checkpoint),
                "bid": "Enoch",
                "kitty": if args.features.contains(EnochFeatures::DEFAULT_KITTY)
                    && candidate_is_landlord_team
                {
                    "DefaultHeuristic"
                } else {
                    "Enoch"
                },
            })
        })
        .collect::<Vec<_>>();
    (
        OrientationPlan {
            name,
            seats,
            routing: AuditRouting::DynamicTeam {
                candidate_is_landlord_team,
            },
            candidate_is_landlord_team,
        },
        json!({
            "name": name,
            "candidate_is_landlord_team": candidate_is_landlord_team,
            "audit_routing": "public-current-landlords-team-membership",
            "join_switch_contract":
                "a revealed friend uses its new landlord-team config on its next decision",
            "seats": seat_identity,
        }),
    )
}

fn scenario_category(id: &str) -> &'static str {
    if id == "intended" || id == "equal" {
        "budget"
    } else if id.starts_with("rank-") {
        "rank"
    } else if id.starts_with("crossplay-") {
        "crossplay"
    } else if id.starts_with("configuration-") {
        "configuration"
    } else if id.starts_with("finding-friends-") {
        "finding-friends"
    } else if id.starts_with("scoring-kitty-") {
        "scoring-kitty"
    } else if id.starts_with("threshold-") {
        "threshold"
    } else {
        "development"
    }
}

fn scenario_total_pairs(id: &str) -> Option<usize> {
    if id == "intended" || id == "equal" {
        Some(800)
    } else if id.starts_with("threshold-") {
        Some(50)
    } else if id == "standard" || id == "development-finding-friends" {
        None
    } else {
        Some(100)
    }
}

fn build_scenario(args: &Args) -> Result<ScenarioPlan, String> {
    let id = args.scenario_id.as_str();
    let expected_namespace = QUALIFICATION_SCENARIOS
        .iter()
        .find_map(|(scenario, namespace)| (*scenario == id).then_some(*namespace));
    let mut config = HarnessConfig::default();
    let mut threshold_band = None;
    let (orientations, orientation_identity) = match id {
        "standard" | "intended" | "equal" => standard_orientations(args, config.num_players)?,
        "rank-low" => {
            config.rank = Rank::Number(Number::Two);
            standard_orientations(args, config.num_players)?
        }
        "rank-middle" => {
            config.rank = Rank::Number(Number::Seven);
            standard_orientations(args, config.num_players)?
        }
        "rank-high" => {
            config.rank = Rank::Number(Number::Ace);
            standard_orientations(args, config.num_players)?
        }
        "crossplay-assignment-01" => {
            crossplay_orientations(args, Checkpoint::Control, Checkpoint::Control)?
        }
        "crossplay-assignment-02" => {
            crossplay_orientations(args, Checkpoint::Candidate, Checkpoint::Control)?
        }
        "crossplay-assignment-03" => {
            crossplay_orientations(args, Checkpoint::Control, Checkpoint::Candidate)?
        }
        "crossplay-assignment-04" => {
            crossplay_orientations(args, Checkpoint::Candidate, Checkpoint::Candidate)?
        }
        "configuration-slot-01" => {
            config.num_players = 2;
            config.decks = vec![Deck::default()];
            standard_orientations(args, config.num_players)?
        }
        "configuration-slot-02" => {
            config.num_players = 4;
            config.decks = vec![Deck::default(); 3];
            standard_orientations(args, config.num_players)?
        }
        "configuration-slot-03" => {
            config.num_players = 6;
            config.decks = vec![Deck::default(); 3];
            standard_orientations(args, config.num_players)?
        }
        "development-finding-friends" | "finding-friends-contract-01" => {
            config.num_players = 5;
            config.decks = vec![Deck::default(); 2];
            config.game_mode = GameModeSettings::FindingFriends {
                num_friends: Some(1),
            };
            config.friend_selection_policy = FriendSelectionPolicy::Unrestricted;
            config.multiple_join_policy = MultipleJoinPolicy::Unrestricted;
            let (first, first_identity) =
                dynamic_team_orientation(args, "candidate-landlord-team", config.num_players, true);
            let (second, second_identity) =
                dynamic_team_orientation(args, "control-landlord-team", config.num_players, false);
            ([first, second], [first_identity, second_identity])
        }
        "finding-friends-contract-02" => {
            config.num_players = 7;
            config.decks = vec![Deck::default(); 3];
            config.game_mode = GameModeSettings::FindingFriends {
                num_friends: Some(2),
            };
            config.friend_selection_policy = FriendSelectionPolicy::HighestCardNotAllowed;
            config.multiple_join_policy = MultipleJoinPolicy::NoDoubleJoin;
            let (first, first_identity) =
                dynamic_team_orientation(args, "candidate-landlord-team", config.num_players, true);
            let (second, second_identity) =
                dynamic_team_orientation(args, "control-landlord-team", config.num_players, false);
            ([first, second], [first_identity, second_identity])
        }
        "scoring-kitty-ruleset-01" => {
            config.kitty_penalty = KittyPenalty::Times;
            standard_orientations(args, config.num_players)?
        }
        "scoring-kitty-ruleset-02" => {
            config.kitty_penalty = KittyPenalty::Power;
            standard_orientations(args, config.num_players)?
        }
        "scoring-kitty-ruleset-03" => {
            config.kitty_penalty = KittyPenalty::Times;
            config.throw_penalty = ThrowPenalty::TenPointsPerAttempt;
            config.game_scoring_parameters =
                GameScoringParameters::new(25, 2, 0, true, BonusLevelPolicy::NoBonusLevel);
            standard_orientations(args, config.num_players)?
        }
        "threshold-situation-01" => {
            threshold_band = Some(ThresholdBand {
                minimum_inclusive: 0,
                maximum_exclusive: Some(40),
            });
            standard_orientations(args, config.num_players)?
        }
        "threshold-situation-02" => {
            threshold_band = Some(ThresholdBand {
                minimum_inclusive: 40,
                maximum_exclusive: Some(80),
            });
            standard_orientations(args, config.num_players)?
        }
        "threshold-situation-03" => {
            threshold_band = Some(ThresholdBand {
                minimum_inclusive: 80,
                maximum_exclusive: Some(120),
            });
            standard_orientations(args, config.num_players)?
        }
        "threshold-situation-04" => {
            threshold_band = Some(ThresholdBand {
                minimum_inclusive: 120,
                maximum_exclusive: None,
            });
            standard_orientations(args, config.num_players)?
        }
        _ => return Err(format!("unsupported scenario {id:?}")),
    };
    let threshold_identity = threshold_band.map(|band| {
        json!({
            "selector": "homogeneous-searchless-Enoch-0-control-v1",
            "candidate_independent": true,
            "registry_seed_is_attempt_zero": true,
            "derived_seed_domain": "shengji/enoch-week1/threshold-deal/v1",
            "maximum_attempts": THRESHOLD_SELECTION_MAX_ATTEMPTS,
            "band": {
                "attacker_points_minimum_inclusive": band.minimum_inclusive,
                "attacker_points_maximum_exclusive": band.maximum_exclusive,
            },
        })
    });
    let identity = json!({
        "scenario_contract_version": 1,
        "id": id,
        "category": scenario_category(id),
        "qualification_total_pairs": scenario_total_pairs(id),
        "evaluator_pairs_are_one_declared_shard_subset": true,
        "expected_seed_namespace": expected_namespace,
        "rules": harness_config_json(&config),
        "orientations": orientation_identity,
        "threshold_deal_selection": threshold_identity,
    });
    Ok(ScenarioPlan {
        id: id.to_owned(),
        expected_namespace,
        config,
        orientations,
        threshold_band,
        identity,
    })
}

fn validate_scenario_seed_domain(args: &Args, scenario: &ScenarioPlan) -> Result<(), String> {
    let actual = args.seed_plan.protocol.namespace.as_deref();
    match scenario.expected_namespace {
        Some(expected) if actual != Some(expected) => Err(format!(
            "scenario {:?} requires seed namespace {expected:?}, got {actual:?}",
            scenario.id
        )),
        None if actual.map(|value| value.starts_with("qual/")) == Some(true) => Err(format!(
            "qualification namespace {actual:?} requires its explicit frozen --scenario"
        )),
        _ => Ok(()),
    }
}

fn derive_threshold_deal_seed(registry_seed: u64, attempt: u64) -> u64 {
    if attempt == 0 {
        return registry_seed;
    }
    let mut digest = Sha256::new();
    digest.update(THRESHOLD_DEAL_DOMAIN);
    digest.update(registry_seed.to_be_bytes());
    digest.update(attempt.to_be_bytes());
    let digest = digest.finalize();
    u64::from_be_bytes(
        digest[..8]
            .try_into()
            .expect("SHA-256 always contains eight bytes"),
    )
}

fn resolve_deal_selection(
    scenario: &ScenarioPlan,
    registry_seed: u64,
) -> Result<DealSelection, String> {
    let Some(band) = scenario.threshold_band else {
        return Ok(DealSelection {
            registry_seed,
            effective_seed: registry_seed,
            selection_attempt: 0,
            selector_non_landlord_points: None,
        });
    };
    let selector_seat = Seat {
        play: PlayBrain::EnochGreedy,
        bid: BotDifficulty::Enoch,
        kitty: BotDifficulty::Enoch,
    };
    let selector_seats = vec![selector_seat; scenario.config.num_players];
    for attempt in 0..THRESHOLD_SELECTION_MAX_ATTEMPTS {
        let effective_seed = derive_threshold_deal_seed(registry_seed, attempt);
        let mut rng = StdRng::seed_from_u64(effective_seed);
        let Some(result) = play_one_hand_with_config(&selector_seats, &scenario.config, &mut rng)
        else {
            continue;
        };
        if band.contains(result.non_landlord_points) {
            return Ok(DealSelection {
                registry_seed,
                effective_seed,
                selection_attempt: attempt,
                selector_non_landlord_points: Some(result.non_landlord_points),
            });
        }
    }
    Err(format!(
        "unsupported-threshold-stratum: scenario={:?} registry_seed={} band=[{}, {}) no homogeneous Enoch-0 control deal found in {} deterministic attempts",
        scenario.id,
        registry_seed,
        band.minimum_inclusive,
        band.maximum_exclusive
            .map(|value| value.to_string())
            .unwrap_or_else(|| "infinity".to_owned()),
        THRESHOLD_SELECTION_MAX_ATTEMPTS,
    ))
}

fn resolve_deal_selections(
    args: &Args,
    scenario: &ScenarioPlan,
) -> Result<Vec<DealSelection>, String> {
    let mut effective_seeds = HashSet::new();
    let registry_seeds = args
        .seed_plan
        .entries
        .iter()
        .map(|entry| entry.seed)
        .collect::<HashSet<_>>();
    let mut selections = Vec::with_capacity(args.seed_plan.entries.len());
    for entry in &args.seed_plan.entries {
        let selection = resolve_deal_selection(scenario, entry.seed)?;
        if !effective_seeds.insert(selection.effective_seed) {
            return Err(format!(
                "effective deal seed collision in scenario {:?}: {}",
                scenario.id, selection.effective_seed
            ));
        }
        if selection.effective_seed != selection.registry_seed
            && registry_seeds.contains(&selection.effective_seed)
        {
            return Err(format!(
                "effective deal seed {} collides with another registry seed in scenario {:?}",
                selection.effective_seed, scenario.id
            ));
        }
        selections.push(selection);
    }
    Ok(selections)
}

fn deal_selection_manifest(
    args: &Args,
    scenario: &ScenarioPlan,
    selections: &[DealSelection],
) -> Result<(Value, String), String> {
    if selections.len() != args.seed_plan.entries.len() {
        return Err("deal selection count does not match exact registry seed count".to_owned());
    }
    let records = args
        .seed_plan
        .entries
        .iter()
        .zip(selections.iter())
        .enumerate()
        .map(|(paired_index, (entry, selection))| {
            if selection.registry_seed != entry.seed {
                return Err(format!(
                    "deal selection registry seed mismatch at paired index {paired_index}"
                ));
            }
            Ok(json!({
                "paired_index": paired_index,
                "registry_index": entry.registry_index,
                "registry_seed_u64": selection.registry_seed,
                "registry_seed_hex": format!("{:#018x}", selection.registry_seed),
                "effective_deal_seed_u64": selection.effective_seed,
                "effective_deal_seed_hex": format!("{:#018x}", selection.effective_seed),
                "selection_attempt_zero_based": selection.selection_attempt,
                "selector_non_landlord_points": selection.selector_non_landlord_points,
            }))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let records_sha256 = canonical_json_sha256(&Value::Array(records.clone()))?;
    let manifest = json!({
        "contract": if scenario.threshold_band.is_some() {
            "candidate-independent-threshold-control-selection-v1"
        } else {
            "direct-registry-seed-v1"
        },
        "scenario_id": scenario.id,
        "records_sha256": records_sha256,
        "records": records,
    });
    Ok((manifest, records_sha256))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EvaluationArm {
    Candidate,
    Control,
}

#[derive(Clone, Debug, Default)]
struct StyleMetrics {
    action_decisions: u64,
    lead_plays: u64,
    follow_plays: u64,
    multi_card_plays: u64,
    compound_lead_attempts: u64,
    compound_format_follows: u64,
    failed_throw_leads: u64,
    trump_play_decisions: u64,
    all_trump_play_decisions: u64,
    trump_cards_played: u64,
    point_card_play_decisions: u64,
    point_cards_played: u64,
    point_value_played: u64,
    empty_trick_ruff_plays: u64,
    lead_classification_failures: u64,
}

fn ratio(numerator: u64, denominator: u64) -> Option<f64> {
    (denominator > 0).then(|| numerator as f64 / denominator as f64)
}

impl StyleMetrics {
    fn runner_value(&self, name: &str) -> Option<f64> {
        let value = match name {
            "lead-rate" => ratio(self.lead_plays, self.action_decisions),
            "follow-rate" => ratio(self.follow_plays, self.action_decisions),
            "multi-card-play-rate" => ratio(self.multi_card_plays, self.action_decisions),
            "throw-rate" => ratio(self.compound_lead_attempts, self.lead_plays),
            "compound-format-follow-rate" => ratio(self.compound_format_follows, self.follow_plays),
            "failed-throw-rate" => {
                Some(ratio(self.failed_throw_leads, self.compound_lead_attempts).unwrap_or(0.0))
            }
            "trump-play-rate" => ratio(self.trump_play_decisions, self.action_decisions),
            "point-card-play-rate" => ratio(self.point_card_play_decisions, self.action_decisions),
            "empty-trick-ruff-rate" => ratio(self.empty_trick_ruff_plays, self.follow_plays),
            _ => return None,
        };
        Some(value.unwrap_or(0.0))
    }

    fn merge(&mut self, other: &Self) {
        self.action_decisions = self.action_decisions.saturating_add(other.action_decisions);
        self.lead_plays = self.lead_plays.saturating_add(other.lead_plays);
        self.follow_plays = self.follow_plays.saturating_add(other.follow_plays);
        self.multi_card_plays = self.multi_card_plays.saturating_add(other.multi_card_plays);
        self.compound_lead_attempts = self
            .compound_lead_attempts
            .saturating_add(other.compound_lead_attempts);
        self.compound_format_follows = self
            .compound_format_follows
            .saturating_add(other.compound_format_follows);
        self.failed_throw_leads = self
            .failed_throw_leads
            .saturating_add(other.failed_throw_leads);
        self.trump_play_decisions = self
            .trump_play_decisions
            .saturating_add(other.trump_play_decisions);
        self.all_trump_play_decisions = self
            .all_trump_play_decisions
            .saturating_add(other.all_trump_play_decisions);
        self.trump_cards_played = self
            .trump_cards_played
            .saturating_add(other.trump_cards_played);
        self.point_card_play_decisions = self
            .point_card_play_decisions
            .saturating_add(other.point_card_play_decisions);
        self.point_cards_played = self
            .point_cards_played
            .saturating_add(other.point_cards_played);
        self.point_value_played = self
            .point_value_played
            .saturating_add(other.point_value_played);
        self.empty_trick_ruff_plays = self
            .empty_trick_ruff_plays
            .saturating_add(other.empty_trick_ruff_plays);
        self.lead_classification_failures = self
            .lead_classification_failures
            .saturating_add(other.lead_classification_failures);
    }

    fn record_action(&mut self, state: &PlayPhase, config: &HarnessConfig, cards: &[Card]) {
        self.action_decisions = self.action_decisions.saturating_add(1);
        let leading = state.trick().played_cards().is_empty();
        if leading {
            self.lead_plays = self.lead_plays.saturating_add(1);
        } else {
            self.follow_plays = self.follow_plays.saturating_add(1);
        }
        if cards.len() > 1 {
            self.multi_card_plays = self.multi_card_plays.saturating_add(1);
        }

        let trump = state.trump();
        let trump_cards = cards
            .iter()
            .filter(|card| trump.effective_suit(**card) == EffectiveSuit::Trump)
            .count();
        if trump_cards > 0 {
            self.trump_play_decisions = self.trump_play_decisions.saturating_add(1);
            self.trump_cards_played = self.trump_cards_played.saturating_add(trump_cards as u64);
        }
        let all_trump = !cards.is_empty() && trump_cards == cards.len();
        if all_trump {
            self.all_trump_play_decisions = self.all_trump_play_decisions.saturating_add(1);
        }

        let point_cards = cards.iter().filter(|card| card.points().is_some()).count();
        if point_cards > 0 {
            self.point_card_play_decisions = self.point_card_play_decisions.saturating_add(1);
            self.point_cards_played = self.point_cards_played.saturating_add(point_cards as u64);
        }
        let point_value: usize = cards.iter().filter_map(|card| card.points()).sum();
        self.point_value_played = self.point_value_played.saturating_add(point_value as u64);

        if leading {
            match TrickFormat::from_cards(
                trump,
                config.tractor_requirements,
                cards,
                None,
                config.compound_formats.clone(),
            ) {
                Ok(format) if format.units().len() > 1 || format.is_rainbow() => {
                    self.compound_lead_attempts = self.compound_lead_attempts.saturating_add(1);
                }
                Ok(_) => {}
                Err(_) => {
                    self.lead_classification_failures =
                        self.lead_classification_failures.saturating_add(1);
                }
            }
        } else if let Some(format) = state.trick().trick_format() {
            if format.units().len() > 1 || format.is_rainbow() {
                self.compound_format_follows = self.compound_format_follows.saturating_add(1);
            }
            let points_before: usize = state
                .trick()
                .played_cards()
                .iter()
                .flat_map(|played| played.cards.iter())
                .filter_map(|card| card.points())
                .sum();
            if points_before == 0 && format.suit() != EffectiveSuit::Trump && all_trump {
                self.empty_trick_ruff_plays = self.empty_trick_ruff_plays.saturating_add(1);
            }
        }
    }

    fn to_json(&self) -> Value {
        json!({
            "action_decisions": self.action_decisions,
            "lead_plays": self.lead_plays,
            "follow_plays": self.follow_plays,
            "multi_card_plays": self.multi_card_plays,
            "compound_lead_attempts": self.compound_lead_attempts,
            "compound_format_follows": self.compound_format_follows,
            "failed_throw_leads": self.failed_throw_leads,
            "trump_play_decisions": self.trump_play_decisions,
            "all_trump_play_decisions": self.all_trump_play_decisions,
            "trump_cards_played": self.trump_cards_played,
            "point_card_play_decisions": self.point_card_play_decisions,
            "point_cards_played": self.point_cards_played,
            "point_value_played": self.point_value_played,
            "empty_trick_ruff_plays": self.empty_trick_ruff_plays,
            "lead_classification_failures": self.lead_classification_failures,
            "rates": {
                "lead-rate": ratio(self.lead_plays, self.action_decisions),
                "follow-rate": ratio(self.follow_plays, self.action_decisions),
                "multi-card-play-rate": ratio(self.multi_card_plays, self.action_decisions),
                "throw-rate": ratio(self.compound_lead_attempts, self.lead_plays),
                "compound-format-follow-rate": ratio(
                    self.compound_format_follows,
                    self.follow_plays,
                ),
                "failed-throw-rate": ratio(
                    self.failed_throw_leads,
                    self.compound_lead_attempts,
                ),
                "trump-play-rate": ratio(self.trump_play_decisions, self.action_decisions),
                "point-card-play-rate": ratio(
                    self.point_card_play_decisions,
                    self.action_decisions,
                ),
                "empty-trick-ruff-rate": ratio(
                    self.empty_trick_ruff_plays,
                    self.follow_plays,
                ),
            },
        })
    }
}

#[derive(Clone, Debug, Default)]
struct ArmAudit {
    decisions: u64,
    strict_decisions: u64,
    non_strict_decisions: u64,
    missing_search_telemetry: u64,
    decisions_without_action: u64,
    policy_fallback_used: u64,
    prior_fallback_used: u64,
    requested_worlds: u64,
    worlds_attempted: u64,
    worlds_accepted: u64,
    worlds_completed: u64,
    initial_candidate_count: u64,
    candidate_pool_count: u64,
    candidate_work_budget: u64,
    candidate_evaluations: u64,
    effective_time_budget_micros: u64,
    time_bound_decisions: u64,
    work_bound_decisions: u64,
    forced_action_decisions: u64,
    timeout_failures: u64,
    failure_reasons: BTreeMap<String, u64>,
    latency_micros: Vec<u64>,
    style: StyleMetrics,
}

fn add_usize(total: &mut u64, value: usize) {
    *total = total.saturating_add(u64::try_from(value).unwrap_or(u64::MAX));
}

fn duration_micros(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

fn percentile_nearest_rank(values: &[u64], percentile: usize) -> Option<u64> {
    if values.is_empty() || percentile == 0 || percentile > 100 {
        return None;
    }
    let mut ordered = values.to_vec();
    ordered.sort_unstable();
    let rank = percentile.saturating_mul(ordered.len()).saturating_add(99) / 100;
    ordered.get(rank.saturating_sub(1)).copied()
}

impl ArmAudit {
    fn record_decision(
        &mut self,
        state: &PlayPhase,
        config: &HarnessConfig,
        decision: &PlayDecision,
    ) {
        self.decisions = self.decisions.saturating_add(1);
        if decision.policy_fallback_used {
            self.policy_fallback_used = self.policy_fallback_used.saturating_add(1);
        }
        if decision.cards.is_none() {
            self.decisions_without_action = self.decisions_without_action.saturating_add(1);
        }
        match &decision.search {
            Some(telemetry) => self.record_search(telemetry),
            None => {
                self.missing_search_telemetry = self.missing_search_telemetry.saturating_add(1);
            }
        }
        if let Some(cards) = &decision.cards {
            self.style.record_action(state, config, cards);
        }
    }

    fn record_search(&mut self, telemetry: &SearchTelemetry) {
        if telemetry.strict {
            self.strict_decisions = self.strict_decisions.saturating_add(1);
        } else {
            self.non_strict_decisions = self.non_strict_decisions.saturating_add(1);
        }
        add_usize(&mut self.requested_worlds, telemetry.requested_worlds);
        add_usize(&mut self.worlds_attempted, telemetry.worlds_attempted);
        add_usize(&mut self.worlds_accepted, telemetry.worlds_accepted);
        add_usize(&mut self.worlds_completed, telemetry.worlds_completed);
        add_usize(
            &mut self.initial_candidate_count,
            telemetry.initial_candidate_count,
        );
        add_usize(
            &mut self.candidate_pool_count,
            telemetry.candidate_pool_count,
        );
        add_usize(
            &mut self.candidate_work_budget,
            telemetry.candidate_work_budget,
        );
        add_usize(
            &mut self.candidate_evaluations,
            telemetry.candidate_evaluations,
        );
        self.effective_time_budget_micros = self
            .effective_time_budget_micros
            .saturating_add(duration_micros(telemetry.effective_time_budget));
        self.latency_micros.push(duration_micros(telemetry.elapsed));
        if telemetry.time_bound {
            self.time_bound_decisions = self.time_bound_decisions.saturating_add(1);
        }
        if telemetry.work_bound {
            self.work_bound_decisions = self.work_bound_decisions.saturating_add(1);
        }
        if telemetry.forced_action {
            self.forced_action_decisions = self.forced_action_decisions.saturating_add(1);
        }
        if telemetry.prior_fallback_used {
            self.prior_fallback_used = self.prior_fallback_used.saturating_add(1);
        }
        if let Some(reason) = telemetry.failure {
            if reason.as_str() == "budget_expired_before_work"
                || (reason.as_str() == "no_completed_worlds" && telemetry.time_bound)
            {
                self.timeout_failures = self.timeout_failures.saturating_add(1);
            }
            *self
                .failure_reasons
                .entry(reason.as_str().to_owned())
                .or_default() += 1;
        }
    }

    fn merge(&mut self, other: &Self) {
        self.decisions = self.decisions.saturating_add(other.decisions);
        self.strict_decisions = self.strict_decisions.saturating_add(other.strict_decisions);
        self.non_strict_decisions = self
            .non_strict_decisions
            .saturating_add(other.non_strict_decisions);
        self.missing_search_telemetry = self
            .missing_search_telemetry
            .saturating_add(other.missing_search_telemetry);
        self.decisions_without_action = self
            .decisions_without_action
            .saturating_add(other.decisions_without_action);
        self.policy_fallback_used = self
            .policy_fallback_used
            .saturating_add(other.policy_fallback_used);
        self.prior_fallback_used = self
            .prior_fallback_used
            .saturating_add(other.prior_fallback_used);
        self.requested_worlds = self.requested_worlds.saturating_add(other.requested_worlds);
        self.worlds_attempted = self.worlds_attempted.saturating_add(other.worlds_attempted);
        self.worlds_accepted = self.worlds_accepted.saturating_add(other.worlds_accepted);
        self.worlds_completed = self.worlds_completed.saturating_add(other.worlds_completed);
        self.initial_candidate_count = self
            .initial_candidate_count
            .saturating_add(other.initial_candidate_count);
        self.candidate_pool_count = self
            .candidate_pool_count
            .saturating_add(other.candidate_pool_count);
        self.candidate_work_budget = self
            .candidate_work_budget
            .saturating_add(other.candidate_work_budget);
        self.candidate_evaluations = self
            .candidate_evaluations
            .saturating_add(other.candidate_evaluations);
        self.effective_time_budget_micros = self
            .effective_time_budget_micros
            .saturating_add(other.effective_time_budget_micros);
        self.time_bound_decisions = self
            .time_bound_decisions
            .saturating_add(other.time_bound_decisions);
        self.work_bound_decisions = self
            .work_bound_decisions
            .saturating_add(other.work_bound_decisions);
        self.forced_action_decisions = self
            .forced_action_decisions
            .saturating_add(other.forced_action_decisions);
        self.timeout_failures = self.timeout_failures.saturating_add(other.timeout_failures);
        for (reason, count) in &other.failure_reasons {
            let total = self.failure_reasons.entry(reason.clone()).or_default();
            *total = total.saturating_add(*count);
        }
        self.latency_micros.extend_from_slice(&other.latency_micros);
        self.style.merge(&other.style);
    }

    fn failure_count(&self) -> u64 {
        self.failure_reasons
            .values()
            .copied()
            .fold(0u64, u64::saturating_add)
    }

    fn invalid_counter_total(&self) -> u64 {
        self.failure_count()
            .saturating_add(self.missing_search_telemetry)
            .saturating_add(self.decisions_without_action)
            .saturating_add(self.policy_fallback_used)
            .saturating_add(self.prior_fallback_used)
            .saturating_add(self.non_strict_decisions)
            .saturating_add(self.style.lead_classification_failures)
    }

    fn mean_latency_ms(&self) -> Option<f64> {
        (!self.latency_micros.is_empty()).then(|| {
            self.latency_micros
                .iter()
                .copied()
                .map(|value| value as f64)
                .sum::<f64>()
                / self.latency_micros.len() as f64
                / 1_000.0
        })
    }

    fn to_json(&self) -> Value {
        let failure_reason = |name: &str| self.failure_reasons.get(name).copied().unwrap_or(0);
        json!({
            "decision_count": self.decisions,
            "strict_decisions": self.strict_decisions,
            "non_strict_decisions": self.non_strict_decisions,
            "missing_search_telemetry": self.missing_search_telemetry,
            "decisions_without_action": self.decisions_without_action,
            "latency_micros": {
                "p50": percentile_nearest_rank(&self.latency_micros, 50),
                "p95": percentile_nearest_rank(&self.latency_micros, 95),
                "max": self.latency_micros.iter().copied().max(),
                "raw_for_deterministic_merge": self.latency_micros,
            },
            "worlds": {
                "requested": self.requested_worlds,
                "attempted": self.worlds_attempted,
                "accepted": self.worlds_accepted,
                "completed": self.worlds_completed,
            },
            "candidate_work": {
                "initial_candidates": self.initial_candidate_count,
                "candidate_pool": self.candidate_pool_count,
                "evaluation_budget": self.candidate_work_budget,
                "evaluations_completed": self.candidate_evaluations,
            },
            "effective_time_budget_micros_total": self.effective_time_budget_micros,
            "bound_counts": {
                "time": self.time_bound_decisions,
                "work": self.work_bound_decisions,
            },
            "forced_action_decisions": self.forced_action_decisions,
            "timeout_failures": self.timeout_failures,
            "internal_prior_fallback": self.prior_fallback_used,
            "external_policy_fallback": self.policy_fallback_used,
            "search_failure_count": self.failure_count(),
            "search_failure_reasons": {
                "no_legal_candidates": failure_reason("no_legal_candidates"),
                "budget_expired_before_work": failure_reason("budget_expired_before_work"),
                "no_completed_worlds": failure_reason("no_completed_worlds"),
            },
            "invalid_counter_total": self.invalid_counter_total(),
            "style": self.style.to_json(),
        })
    }
}

#[derive(Clone, Debug, Default)]
struct PairAudit {
    candidate: ArmAudit,
    control: ArmAudit,
    attribution_failures: u64,
}

impl PairAudit {
    fn arm_mut(&mut self, arm: EvaluationArm) -> &mut ArmAudit {
        match arm {
            EvaluationArm::Candidate => &mut self.candidate,
            EvaluationArm::Control => &mut self.control,
        }
    }

    fn merge(&mut self, other: &Self) {
        self.candidate.merge(&other.candidate);
        self.control.merge(&other.control);
        self.attribution_failures = self
            .attribution_failures
            .saturating_add(other.attribution_failures);
    }

    fn invalid_counter_total(&self) -> u64 {
        self.candidate
            .invalid_counter_total()
            .saturating_add(self.control.invalid_counter_total())
            .saturating_add(self.attribution_failures)
    }

    fn to_json(&self) -> Value {
        json!({
            "candidate": self.candidate.to_json(),
            "control": self.control.to_json(),
            "attribution_failures": self.attribution_failures,
            "invalid_counter_total": self.invalid_counter_total(),
        })
    }
}

fn failure_evidence(count: Option<u64>, authority: &str) -> Value {
    json!({
        "count": count,
        "authority": authority,
    })
}

fn runner_record_inputs(
    args: &Args,
    seed_entry: &SeedEntry,
    complete: bool,
    winrate: Option<f64>,
    margin: Option<f64>,
    level_utility: Option<f64>,
    audit: &PairAudit,
) -> Value {
    let mut style_metrics = serde_json::Map::new();
    for name in &args.runner_style_metrics {
        style_metrics.insert(
            name.clone(),
            json!(audit.candidate.style.runner_value(name)),
        );
    }
    let model_fallbacks = audit
        .candidate
        .prior_fallback_used
        .saturating_add(audit.candidate.policy_fallback_used)
        .saturating_add(audit.control.prior_fallback_used)
        .saturating_add(audit.control.policy_fallback_used);
    let timeouts = audit
        .candidate
        .timeout_failures
        .saturating_add(audit.control.timeout_failures);
    let failure_counter_evidence = json!({
        "illegal_action": failure_evidence(
            complete.then_some(0),
            if complete {
                "both audited orientations completed; the driver rejects illegal play application"
            } else {
                "an incomplete audited orientation cannot distinguish illegal play from other driver failure"
            },
        ),
        "honesty_violation": failure_evidence(
            None,
            "requires a fingerprinted external hidden-information preflight artifact",
        ),
        "model_fallback": failure_evidence(
            Some(model_fallbacks),
            "SearchTelemetry.prior_fallback_used plus PlayDecision.policy_fallback_used",
        ),
        "model_contract_failure": failure_evidence(
            Some(0),
            "both arms use EnochHeuristic search and non-model bid/kitty policies",
        ),
        "incomplete_pair": failure_evidence(
            Some(u64::from(!complete)),
            "two synchronous audited orientation return values",
        ),
        "hidden_information_leak": failure_evidence(
            None,
            "requires a fingerprinted external honesty-suite artifact",
        ),
        "artifact_mismatch": failure_evidence(
            None,
            "requires comparison-bound external artifact fingerprint verification",
        ),
        "cancellation": failure_evidence(
            Some(0),
            "the synchronous evaluator emitted this completed pair record; interrupted processes emit no record",
        ),
        "fixture_failure": failure_evidence(
            None,
            "requires a fingerprinted external arm-fixture artifact",
        ),
        "machine_contention": failure_evidence(
            None,
            "requires a fingerprinted external worker-isolation artifact",
        ),
        "timeout": failure_evidence(
            Some(timeouts),
            "strict search failures that expired before work or completed no world at the deadline",
        ),
    });
    json!({
        "seed_index": seed_entry.registry_index,
        "seed": seed_entry.seed,
        "level_utility_delta": level_utility,
        "point_margin_delta": margin,
        "win_rate_delta": winrate.map(|value| value - 0.5),
        "candidate_latency_ms": audit.candidate.mean_latency_ms(),
        "control_latency_ms": audit.control.mean_latency_ms(),
        "candidate_completed_worlds": audit.candidate.worlds_completed,
        "control_completed_worlds": audit.control.worlds_completed,
        "style_metrics": Value::Object(style_metrics),
        "failure_counter_evidence": failure_counter_evidence,
    })
}

fn seat_index(state: &PlayPhase, actor: PlayerID) -> Option<usize> {
    state
        .propagated()
        .players()
        .iter()
        .position(|player| player.id == actor)
}

fn dynamic_team_audit_arm(
    actor_is_landlord_team: bool,
    candidate_is_landlord_team: bool,
) -> EvaluationArm {
    if actor_is_landlord_team == candidate_is_landlord_team {
        EvaluationArm::Candidate
    } else {
        EvaluationArm::Control
    }
}

fn routed_arm(
    state: &PlayPhase,
    actor: PlayerID,
    seat_index: usize,
    routing: &AuditRouting,
) -> Option<EvaluationArm> {
    match routing {
        AuditRouting::Static(arms) => arms.get(seat_index).copied(),
        AuditRouting::DynamicTeam {
            candidate_is_landlord_team,
        } => {
            let actor_is_landlord_team = state.landlords_team().contains(&actor);
            Some(dynamic_team_audit_arm(
                actor_is_landlord_team,
                *candidate_is_landlord_team,
            ))
        }
    }
}

fn play_audited_orientation(
    orientation: &OrientationPlan,
    config: &HarnessConfig,
    seed: u64,
) -> (Option<HandResult>, PairAudit) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut audit = PairAudit::default();
    let mut pending_lead_arm = None;
    let result = play_one_hand_with_config_audited(
        &orientation.seats,
        config,
        &mut rng,
        &mut |state, actor, decision| {
            if state.trick().played_cards().len() == 1 {
                if !state.trick().played_cards()[0].bad_throw_cards.is_empty() {
                    if let Some(arm) = pending_lead_arm {
                        let style = &mut audit.arm_mut(arm).style;
                        style.failed_throw_leads = style.failed_throw_leads.saturating_add(1);
                    } else {
                        audit.attribution_failures = audit.attribution_failures.saturating_add(1);
                    }
                }
                pending_lead_arm = None;
            }
            if let Some(index) = seat_index(state, actor) {
                if let Some(arm) = routed_arm(state, actor, index, &orientation.routing) {
                    let leading = state.trick().played_cards().is_empty();
                    audit.arm_mut(arm).record_decision(state, config, decision);
                    if leading && decision.cards.is_some() {
                        pending_lead_arm = Some(arm);
                    }
                } else {
                    audit.attribution_failures = audit.attribution_failures.saturating_add(1);
                }
            } else {
                audit.attribution_failures = audit.attribution_failures.saturating_add(1);
            }
        },
    );
    (result, audit)
}

fn orientation_outcome(
    name: &str,
    result: Option<HandResult>,
    candidate_is_landlord_team: bool,
) -> Value {
    match result {
        Some(result) => {
            let (candidate_won, candidate_margin) =
                result.subject_outcome(candidate_is_landlord_team);
            json!({
                "orientation": name,
                "complete": true,
                "candidate_is_landlord_team": candidate_is_landlord_team,
                "candidate_won": candidate_won,
                "candidate_point_margin": candidate_margin,
                "candidate_level_utility": result.subject_level_utility(candidate_is_landlord_team),
                "landlord_won": result.landlord_won,
                "landlord_seat": result.landlord_seat.0,
                "non_landlord_points": result.non_landlord_points,
                "landlord_level_delta": result.landlord_level_delta,
                "non_landlord_level_delta": result.non_landlord_level_delta,
            })
        }
        None => json!({
            "orientation": name,
            "complete": false,
            "candidate_is_landlord_team": candidate_is_landlord_team,
            "candidate_won": null,
            "candidate_point_margin": null,
            "candidate_level_utility": null,
            "landlord_won": null,
            "landlord_seat": null,
            "non_landlord_points": null,
            "landlord_level_delta": null,
            "non_landlord_level_delta": null,
        }),
    }
}

fn optional_finite(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

fn metric(values: &[f64], null_value: f64, bootstrap_seed: u64) -> Value {
    if values.is_empty() {
        return json!({
            "estimate": null,
            "delta_from_null": null,
            "null_value": null_value,
            "paired_bootstrap95": null,
            "delta_paired_bootstrap95": null,
            "mde_80pct_two_sided": null,
            "paired_observations": 0,
        });
    }

    let estimate = mean(values);
    let (low, high) = bootstrap_mean_ci(values, BOOTSTRAP_ITERATIONS, bootstrap_seed);
    let mde = minimum_detectable_effect(values, Z_ALPHA_95, Z_POWER_80);
    json!({
        "estimate": estimate,
        "delta_from_null": estimate - null_value,
        "null_value": null_value,
        "paired_bootstrap95": [low, high],
        "delta_paired_bootstrap95": [low - null_value, high - null_value],
        "mde_80pct_two_sided": optional_finite(mde),
        "paired_observations": values.len(),
    })
}

fn feature_json(features: EnochFeatures) -> Value {
    json!({
        "bits": features.bits(),
        "canonical_spec": features.to_string(),
        "names": features.names(),
    })
}

fn protocol_identity_json(protocol: &ProtocolIdentity) -> Value {
    json!({
        "domain_status": protocol.domain_status,
        "protocol_kind": protocol.protocol_kind,
        "manifest_version": protocol.manifest_version,
        "protocol_fingerprint": protocol.protocol_fingerprint,
        "seed_registry_sha256": protocol.seed_registry_sha256,
        "derivation_domain": protocol.derivation_domain,
        "master_seed_u64": protocol.master_seed,
        "namespace": protocol.namespace,
        "registry_namespace_count": protocol.registry_namespace_count,
        "environment_policy_verified": protocol.environment_policy_verified,
        "environment_allowlist": protocol.environment_allowlist,
    })
}

fn relevant_environment(protocol: &ProtocolIdentity) -> Result<BTreeMap<String, String>, String> {
    let mut relevant = BTreeMap::new();
    for (name, value) in env::vars() {
        if BLOCKED_ENV_PREFIXES
            .iter()
            .any(|prefix| name.starts_with(prefix))
        {
            if protocol.environment_policy_verified
                && !protocol
                    .environment_allowlist
                    .iter()
                    .any(|allowed| allowed == &name)
            {
                return Err(format!(
                    "ambient evaluator variable {name:?} is blocked by the frozen Week-1 protocol"
                ));
            }
            relevant.insert(name, value);
        }
    }
    Ok(relevant)
}

fn evaluator_environment_identity(protocol: &ProtocolIdentity) -> Result<(Value, String), String> {
    let executable =
        env::current_exe().map_err(|error| format!("resolve evaluator executable: {error}"))?;
    let binary_sha256 = sha256_file(&executable)?;
    let parallelism = std::thread::available_parallelism()
        .map(usize::from)
        .map_err(|error| format!("read available hardware parallelism: {error}"))?;
    let contract = json!({
        "binary_sha256": binary_sha256,
        "crate_name": env!("CARGO_PKG_NAME"),
        "crate_version": env!("CARGO_PKG_VERSION"),
        "build_profile": if cfg!(debug_assertions) { "debug" } else { "release" },
        "target_os": env::consts::OS,
        "target_arch": env::consts::ARCH,
        "target_family": env::consts::FAMILY,
        "target_pointer_width": usize::BITS,
        "target_endian": if cfg!(target_endian = "little") { "little" } else { "big" },
        "available_parallelism": parallelism,
        "source_revision": option_env!("SHENGJI_SOURCE_SHA")
            .or(option_env!("GIT_COMMIT"))
            .or(option_env!("SOURCE_SHA")),
        "blocked_environment_prefixes": BLOCKED_ENV_PREFIXES,
        "effective_experiment_environment": relevant_environment(protocol)?,
        "frozen_environment_policy_verified": protocol.environment_policy_verified,
        "frozen_environment_allowlist": protocol.environment_allowlist,
    });
    let fingerprint = canonical_json_sha256(&contract)?;
    Ok((contract, fingerprint))
}

fn search_identity(args: &Args, scenario: &ScenarioPlan) -> Result<(Value, String), String> {
    let candidate_default_kitty = args.features.contains(EnochFeatures::DEFAULT_KITTY);
    let contract = json!({
        "harness": "play_one_hand_with_config_audited/configured-mirrored-deal",
        "play_brain": "SearchStrict or SearchStrictByTeam as declared by scenario",
        "strict_no_fallback": true,
        "candidate": {
            "features": feature_json(args.features),
            "bid_difficulty": "Enoch",
            "kitty_difficulty": if candidate_default_kitty { "Easy" } else { "Enoch" },
            "kitty_semantics": if candidate_default_kitty {
                "default-point-shape-heuristic"
            } else {
                "Enoch"
            },
        },
        "control": {
            "features": feature_json(EnochFeatures::empty()),
            "bid_difficulty": "Enoch",
            "kitty_difficulty": "Enoch",
            "kitty_semantics": "Enoch",
        },
        "work": {
            "mode": args.mode.name(),
            "require_full_work": matches!(args.mode, WorkMode::FixedWork { .. }),
            "time_budget_kind": args.mode.time_budget_kind(),
            "time_budget_ms": args.mode.time_budget_ms(),
            "max_worlds": args.worlds,
            "max_candidates": args.candidates,
            "rollout_tricks": args.rollout_tricks,
            "root_policy": "EnochHeuristic",
            "rollout_policy": "EnochHeuristic",
        },
        "audit_contract": {
            "version": "enoch-search-audit-v1",
            "latency_unit": "microseconds",
            "latency_percentiles": "nearest-rank over individual play decisions",
            "runner_pair_latency": "arithmetic mean of individual play-decision latency",
            "world_counts": "sums of SearchTelemetry counters",
            "style_attribution": "actor partnership in each mirrored orientation",
            "compound_leads": "mechanics TrickFormat decomposition under HarnessConfig",
            "empty_trick_ruff": "follow on zero-point nontrump-led pot using all trump",
        },
        "runner_style_metrics": args.runner_style_metrics,
        "scenario": scenario.identity,
    });
    let fingerprint = canonical_json_sha256(&contract)?;
    Ok((contract, fingerprint))
}

fn seed_consumption(args: &Args) -> Result<(Value, Value, String), String> {
    let records: Vec<Value> = args
        .seed_plan
        .entries
        .iter()
        .enumerate()
        .map(|(paired_index, entry)| {
            json!({
                "paired_index": paired_index,
                "registry_index": entry.registry_index,
                "seed_u64": entry.seed,
                "seed_hex": format!("{:#018x}", entry.seed),
            })
        })
        .collect();
    let ordered_seed_sha256 = canonical_json_sha256(&Value::Array(records.clone()))?;
    let protocol_contract = protocol_identity_json(&args.seed_plan.protocol);
    let consumption = json!({
        "source_kind": args.seed_plan.source_kind,
        "source_path": args.seed_plan.source_path,
        "pairs_requested": args.pairs,
        "ordered_seed_records_sha256": ordered_seed_sha256,
        "records": records,
    });
    Ok((protocol_contract, consumption, ordered_seed_sha256))
}

fn run() -> Result<i32, String> {
    let args = parse_args()?;
    if args.environment_identity_only && !args.seed_plan.protocol.environment_policy_verified {
        return Err(
            "--environment-identity-only requires a verified frozen Week-1 protocol".to_owned(),
        );
    }
    let (environment_contract, environment_identity_sha256) =
        evaluator_environment_identity(&args.seed_plan.protocol)?;
    if args.environment_identity_only {
        let payload = json!({
            "environment": environment_contract,
            "environment_identity_sha256": environment_identity_sha256,
        });
        let stdout = io::stdout();
        let mut output = stdout.lock();
        serde_json::to_writer_pretty(&mut output, &payload)
            .map_err(|error| format!("serialize environment identity JSON: {error}"))?;
        writeln!(output).map_err(|error| format!("write environment identity JSON: {error}"))?;
        output
            .flush()
            .map_err(|error| format!("flush environment identity JSON: {error}"))?;
        return Ok(0);
    }
    let scenario = build_scenario(&args)?;
    validate_scenario_seed_domain(&args, &scenario)?;
    // Resolve every effective deal before candidate evaluation. Threshold
    // enrichment therefore either has a complete, candidate-independent shard
    // plan or fails without publishing a partial comparison.
    let deal_selections = resolve_deal_selections(&args, &scenario)?;
    let (deal_selection_manifest, deal_selection_records_sha256) =
        deal_selection_manifest(&args, &scenario, &deal_selections)?;
    let (protocol_contract, seed_consumption, ordered_seed_sha256) = seed_consumption(&args)?;
    let scenario_identity_sha256 = canonical_json_sha256(&scenario.identity)?;
    let compatibility_contract = json!({
        "seed_protocol": protocol_contract,
        "scenario": scenario.identity,
    });
    let protocol_compatibility_sha256 = canonical_json_sha256(&compatibility_contract)?;
    let (search_contract, search_identity_sha256) = search_identity(&args, &scenario)?;
    let candidate_label = format!("Enoch(candidate:{})", args.features);
    let control_label = "Enoch-0".to_owned();

    let mut completed_hands = 0usize;
    let mut complete_pairs = 0usize;
    let mut candidate_wins = 0usize;
    let mut control_wins = 0usize;
    let mut complete_registry_indices = Vec::with_capacity(args.pairs);
    let mut complete_seeds_u64 = Vec::with_capacity(args.pairs);
    let mut complete_seeds_hex = Vec::with_capacity(args.pairs);
    let mut complete_effective_seeds_u64 = Vec::with_capacity(args.pairs);
    let mut complete_effective_seeds_hex = Vec::with_capacity(args.pairs);
    let mut per_deck_winrate = Vec::with_capacity(args.pairs);
    let mut per_deck_margin = Vec::with_capacity(args.pairs);
    let mut per_deck_level_utility = Vec::with_capacity(args.pairs);
    let mut pair_records = Vec::with_capacity(args.pairs);
    let mut aggregate_audit = PairAudit::default();

    for (index, (seed_entry, deal_selection)) in args
        .seed_plan
        .entries
        .iter()
        .zip(deal_selections.iter())
        .enumerate()
    {
        let registry_seed = seed_entry.seed;
        let effective_seed = deal_selection.effective_seed;
        let (candidate_landlord, candidate_landlord_audit) =
            play_audited_orientation(&scenario.orientations[0], &scenario.config, effective_seed);
        let (control_landlord, control_landlord_audit) =
            play_audited_orientation(&scenario.orientations[1], &scenario.config, effective_seed);
        let mut pair_audit = PairAudit::default();
        pair_audit.merge(&candidate_landlord_audit);
        pair_audit.merge(&control_landlord_audit);
        aggregate_audit.merge(&pair_audit);

        let mut pair_wins = Vec::with_capacity(2);
        let mut pair_margins = Vec::with_capacity(2);
        let mut pair_levels = Vec::with_capacity(2);
        let mut pair_candidate_wins = 0usize;
        let mut pair_control_wins = 0usize;
        if let Some(result) = candidate_landlord {
            completed_hands = completed_hands.saturating_add(1);
            let subject_is_landlord = scenario.orientations[0].candidate_is_landlord_team;
            let (won, margin) = result.subject_outcome(subject_is_landlord);
            if won {
                candidate_wins = candidate_wins.saturating_add(1);
                pair_candidate_wins = pair_candidate_wins.saturating_add(1);
            } else {
                control_wins = control_wins.saturating_add(1);
                pair_control_wins = pair_control_wins.saturating_add(1);
            }
            pair_wins.push(if won { 1.0 } else { 0.0 });
            pair_margins.push(margin as f64);
            pair_levels.push(result.subject_level_utility(subject_is_landlord) as f64);
        }
        if let Some(result) = control_landlord {
            completed_hands = completed_hands.saturating_add(1);
            let subject_is_landlord = scenario.orientations[1].candidate_is_landlord_team;
            let (won, margin) = result.subject_outcome(subject_is_landlord);
            if won {
                candidate_wins = candidate_wins.saturating_add(1);
                pair_candidate_wins = pair_candidate_wins.saturating_add(1);
            } else {
                control_wins = control_wins.saturating_add(1);
                pair_control_wins = pair_control_wins.saturating_add(1);
            }
            pair_wins.push(if won { 1.0 } else { 0.0 });
            pair_margins.push(margin as f64);
            pair_levels.push(result.subject_level_utility(subject_is_landlord) as f64);
        }
        let complete = pair_wins.len() == 2;
        if complete {
            complete_pairs = complete_pairs.saturating_add(1);
        }
        let winrate = complete.then(|| (pair_wins[0] + pair_wins[1]) / 2.0);
        let margin = complete.then(|| (pair_margins[0] + pair_margins[1]) / 2.0);
        let level_utility = complete.then(|| (pair_levels[0] + pair_levels[1]) / 2.0);
        if let (Some(winrate), Some(margin), Some(level_utility)) = (winrate, margin, level_utility)
        {
            complete_registry_indices.push(seed_entry.registry_index);
            complete_seeds_u64.push(registry_seed);
            complete_seeds_hex.push(format!("{registry_seed:#018x}"));
            complete_effective_seeds_u64.push(effective_seed);
            complete_effective_seeds_hex.push(format!("{effective_seed:#018x}"));
            per_deck_winrate.push(winrate);
            per_deck_margin.push(margin);
            per_deck_level_utility.push(level_utility);
        }
        let runner_inputs = runner_record_inputs(
            &args,
            seed_entry,
            complete,
            winrate,
            margin,
            level_utility,
            &pair_audit,
        );
        pair_records.push(json!({
            "paired_index": index,
            "registry_index": seed_entry.registry_index,
            "seed_u64": registry_seed,
            "seed_hex": format!("{registry_seed:#018x}"),
            "registry_seed_u64": deal_selection.registry_seed,
            "registry_seed_hex": format!("{:#018x}", deal_selection.registry_seed),
            "effective_deal_seed_u64": effective_seed,
            "effective_deal_seed_hex": format!("{effective_seed:#018x}"),
            "deal_selection": {
                "status": if scenario.threshold_band.is_some() {
                    "candidate-independent-control-selected"
                } else {
                    "direct-registry-seed"
                },
                "selection_attempt_zero_based": deal_selection.selection_attempt,
                "attempts_examined": deal_selection.selection_attempt.saturating_add(1),
                "selector_non_landlord_points": deal_selection.selector_non_landlord_points,
            },
            "complete": complete,
            "hands_completed": pair_wins.len(),
            "hands_failed": 2usize.saturating_sub(pair_wins.len()),
            "candidate_wins": pair_candidate_wins,
            "control_wins": pair_control_wins,
            "candidate_win_rate": winrate,
            "candidate_point_margin": margin,
            "candidate_level_utility": level_utility,
            "runner_record_inputs": runner_inputs,
            "orientations": [
                orientation_outcome(
                    scenario.orientations[0].name,
                    candidate_landlord,
                    scenario.orientations[0].candidate_is_landlord_team,
                ),
                orientation_outcome(
                    scenario.orientations[1].name,
                    control_landlord,
                    scenario.orientations[1].candidate_is_landlord_team,
                ),
            ],
            "audit": pair_audit.to_json(),
        }));
    }

    let hands_expected = args.pairs.saturating_mul(2);
    let failed_hands = hands_expected.saturating_sub(completed_hands);
    let incomplete_pairs = args.pairs.saturating_sub(complete_pairs);
    let audit_invalid_counter_total = aggregate_audit.invalid_counter_total();
    let valid =
        complete_pairs == args.pairs && failed_hands == 0 && audit_invalid_counter_total == 0;

    let payload = json!({
        "manifest_version": 4,
        "evaluator": "enoch-eval-v4-scenarios",
        "method": "direct in-process audited configured-subject mirrored-deal pairs",
        "valid": valid,
        "merge_identity": {
            "schema": "enoch-eval-deterministic-shard-merge-v3-scenarios",
            "merge_safe_seed_domain": args.seed_plan.protocol.domain_status.starts_with("verified-"),
            "protocol_compatibility_sha256": protocol_compatibility_sha256,
            "scenario_identity_sha256": scenario_identity_sha256,
            "search_identity_sha256": search_identity_sha256,
            "environment_identity_sha256": environment_identity_sha256,
            "ordered_shard_seed_records_sha256": ordered_seed_sha256,
            "ordered_effective_deal_records_sha256": deal_selection_records_sha256,
            "protocol": protocol_contract,
            "scenario": &scenario.identity,
            "search": search_contract,
            "environment": environment_contract,
        },
        "scenario": {
            "id": &scenario.id,
            "expected_namespace": scenario.expected_namespace,
            "identity_sha256": scenario_identity_sha256,
            "contract": &scenario.identity,
        },
        "candidate": {
            "label": candidate_label,
            "feature_input": &args.feature_input,
            "features": feature_json(args.features),
            "bid": "Enoch",
            "kitty": if args.features.contains(EnochFeatures::DEFAULT_KITTY) {
                "DefaultHeuristic"
            } else {
                "Enoch"
            },
        },
        "control": {
            "label": control_label,
            "features": feature_json(EnochFeatures::empty()),
            "bid": "Enoch",
            "kitty": "Enoch",
        },
        "settings": {
            "scenario_id": &scenario.id,
            "mode": args.mode.name(),
            "require_full_work": matches!(args.mode, WorkMode::FixedWork { .. }),
            "time_budget_kind": args.mode.time_budget_kind(),
            "time_budget_ms": args.mode.time_budget_ms(),
            "max_worlds": args.worlds,
            "max_candidates": args.candidates,
            "rollout_tricks": args.rollout_tricks,
            "policy": "EnochHeuristic",
            "rollout_policy": "EnochHeuristic",
            "runner_style_metrics": &args.runner_style_metrics,
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "bootstrap_unit": "mirrored-deal-pair",
        },
        "seed_consumption": seed_consumption,
        "deal_selection": deal_selection_manifest,
        "completion": {
            "pairs_requested": args.pairs,
            "pairs_complete": complete_pairs,
            "pairs_incomplete": incomplete_pairs,
            "hands_expected": hands_expected,
            "hands_completed": completed_hands,
            "hands_failed": failed_hands,
            "candidate_wins": candidate_wins,
            "control_wins": control_wins,
            "audit_invalid_counter_total": audit_invalid_counter_total,
        },
        "audit": aggregate_audit.to_json(),
        "paired_records": pair_records,
        "per_deck": {
            "complete_registry_index": complete_registry_indices,
            "complete_seed_u64": complete_seeds_u64,
            "complete_seed_hex": complete_seeds_hex,
            "complete_effective_deal_seed_u64": complete_effective_seeds_u64,
            "complete_effective_deal_seed_hex": complete_effective_seeds_hex,
            "candidate_win_rate": per_deck_winrate,
            "candidate_point_margin": per_deck_margin,
            "candidate_level_utility": per_deck_level_utility,
        },
        "metrics": {
            "win_rate": metric(&per_deck_winrate, 0.5, 0xE10C_0001),
            "point_margin": metric(&per_deck_margin, 0.0, 0xE10C_0002),
            "level_utility": metric(
                &per_deck_level_utility,
                0.0,
                0xE10C_0003,
            ),
        },
    });

    let stdout = io::stdout();
    let mut output = stdout.lock();
    serde_json::to_writer_pretty(&mut output, &payload)
        .map_err(|error| format!("serialize evaluation JSON: {error}"))?;
    writeln!(output).map_err(|error| format!("write evaluation JSON: {error}"))?;
    output
        .flush()
        .map_err(|error| format!("flush evaluation JSON: {error}"))?;

    Ok(if valid { 0 } else { 3 })
}

fn main() {
    match run() {
        Ok(0) => {}
        Ok(code) => process::exit(code),
        Err(error) => {
            eprintln!(
                "{error}\n\n{}\n\n{}\n\n{}",
                usage(),
                style_usage(),
                scenario_usage()
            );
            process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        build_scenario, canonical_json_sha256, deal_selection_manifest, derive_threshold_deal_seed,
        derive_week1_seed, dynamic_team_audit_arm, parse_exact_seed_json, parse_week1_protocol,
        percentile_nearest_rank, runner_record_inputs, validate_scenario_seed_domain,
        validate_seed_plan, Args, ArmAudit, AuditRouting, DealSelection, EvaluationArm,
        GameModeSettings, PairAudit, ProtocolIdentity, SeedEntry, SeedPlan, WorkMode,
        QUALIFICATION_SCENARIOS, WEEK1_DERIVATION_DOMAIN,
    };
    use serde_json::{json, Value};
    use shengji_core::bot::enoch::EnochFeatures;
    use shengji_core::bot::search::{SearchFailureReason, SearchTelemetry};
    use std::collections::{BTreeMap, HashSet};
    use std::time::Duration;

    fn miniature_protocol(master_seed: u64, namespace: &str, count: usize) -> Value {
        let seeds: Vec<u64> = (0..count)
            .map(|index| derive_week1_seed(master_seed, namespace, index as u64).unwrap())
            .collect();
        let registry = json!({
            "derivation": {
                "algorithm": "sha256-first-64-bits",
                "byte_order": "big",
                "domain": WEEK1_DERIVATION_DOMAIN,
                "index_encoding": "u64-big-endian",
                "master_seed_encoding": "u64-big-endian",
                "namespace_encoding": "u32-length-prefixed-utf8",
            },
            "global_seed_count": count,
            "master_seed": master_seed,
            "namespaces": [{
                "count": count,
                "name": namespace,
                "seeds": seeds,
            }],
        });
        let registry_sha256 = canonical_json_sha256(&registry).unwrap();
        let body = json!({
            "automatic_production_promotion_allowed": false,
            "evaluator_environment_policy": {
                "allowlist": [],
                "blocked_prefixes": ["SHENGJI_", "GM_", "OMNI_", "GEN_"],
            },
            "manifest_version": 1,
            "protocol_kind": "enoch-week1-seed-protocol",
            "seed_registry": registry,
            "seed_registry_sha256": registry_sha256,
        });
        let fingerprint = canonical_json_sha256(&body).unwrap();
        let mut protocol = body;
        protocol
            .as_object_mut()
            .unwrap()
            .insert("protocol_fingerprint".to_owned(), json!(fingerprint));
        protocol
    }

    fn args_for_scenario(scenario_id: &str, namespace: Option<&str>) -> Args {
        Args {
            pairs: 1,
            seed_plan: SeedPlan {
                entries: vec![SeedEntry {
                    registry_index: Some(0),
                    seed: 99,
                }],
                source_kind: "test".to_owned(),
                source_path: None,
                protocol: ProtocolIdentity::unverified(namespace.map(str::to_owned), "test-domain"),
            },
            environment_identity_only: false,
            feature_input: "bid-ownership".to_owned(),
            features: EnochFeatures::BID_OWNERSHIP,
            worlds: 8,
            candidates: 3,
            rollout_tricks: 2,
            runner_style_metrics: vec![],
            scenario_id: scenario_id.to_owned(),
            mode: WorkMode::FixedWork { deadline_ms: 100 },
        }
    }

    #[test]
    fn week1_derivation_matches_frozen_python_vector() {
        assert_eq!(
            derive_week1_seed(123, "qual/intended", 0).unwrap(),
            9_567_484_213_682_665_997
        );
    }

    #[test]
    fn environment_identity_hash_matches_frozen_python_vector() {
        let contract = json!({
            "available_parallelism": 10,
            "binary_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
            "blocked_environment_prefixes": ["SHENGJI_", "GM_", "OMNI_", "GEN_"],
            "build_profile": "release",
            "crate_name": "shengji-core",
            "crate_version": "0.1.0",
            "effective_experiment_environment": {"SHENGJI_ALPHA": "café/quote\""},
            "frozen_environment_allowlist": ["SHENGJI_ALPHA"],
            "frozen_environment_policy_verified": true,
            "source_revision": "1111111111111111111111111111111111111111111111111111111111111111",
            "target_arch": "aarch64",
            "target_endian": "little",
            "target_family": "unix",
            "target_os": "macos",
            "target_pointer_width": 64,
        });
        assert_eq!(
            canonical_json_sha256(&contract).unwrap(),
            "552326c51b0a5acc67e3df86bb97a337a34472e900e64242991638894c2b86e7"
        );
    }

    #[test]
    fn registry_accepts_exact_noncontiguous_indices_in_requested_order() {
        let protocol = miniature_protocol(0x5eed, "qual/intended", 5);
        let plan = parse_week1_protocol(
            &protocol,
            Some("qual/intended".to_owned()),
            &[4, 1, 3],
            "memory.json".to_owned(),
        )
        .unwrap();
        assert_eq!(
            plan.entries
                .iter()
                .map(|entry| entry.registry_index)
                .collect::<Vec<_>>(),
            vec![Some(4), Some(1), Some(3)]
        );
        validate_seed_plan(&plan, 3).unwrap();
    }

    #[test]
    fn exact_list_rejects_duplicate_seeds_and_count_mismatch() {
        let duplicate =
            parse_exact_seed_json(&json!([17, 19, 17]), None, &[], "memory.json".to_owned())
                .unwrap();
        assert!(validate_seed_plan(&duplicate, 3)
            .unwrap_err()
            .contains("duplicate seed"));

        let error = parse_exact_seed_json(
            &json!({"count": 3, "seeds": [17, 19]}),
            None,
            &[],
            "memory.json".to_owned(),
        )
        .unwrap_err();
        assert!(error.contains("count mismatch"));
    }

    #[test]
    fn protocol_fails_closed_on_domain_or_fingerprint_mismatch() {
        let mut wrong_domain = miniature_protocol(0x5eed, "qual/intended", 2);
        wrong_domain["seed_registry"]["derivation"]["domain"] = json!("wrong/domain");
        assert!(parse_week1_protocol(
            &wrong_domain,
            Some("qual/intended".to_owned()),
            &[],
            "memory.json".to_owned(),
        )
        .unwrap_err()
        .contains("domain/contract mismatch"));

        let mut wrong_fingerprint = miniature_protocol(0x5eed, "qual/intended", 2);
        wrong_fingerprint["protocol_fingerprint"] = json!("00".repeat(32));
        assert!(parse_week1_protocol(
            &wrong_fingerprint,
            Some("qual/intended".to_owned()),
            &[],
            "memory.json".to_owned(),
        )
        .unwrap_err()
        .contains("fingerprint mismatch"));
    }

    #[test]
    fn every_frozen_qualification_scenario_builds_with_unique_identity() {
        assert_eq!(QUALIFICATION_SCENARIOS.len(), 21);
        let mut ids = HashSet::new();
        let mut namespaces = HashSet::new();
        let mut fingerprints = HashSet::new();
        for (id, namespace) in QUALIFICATION_SCENARIOS {
            assert!(ids.insert(id));
            assert!(namespaces.insert(namespace));
            let args = args_for_scenario(id, Some(namespace));
            let scenario = build_scenario(&args).unwrap();
            validate_scenario_seed_domain(&args, &scenario).unwrap();
            assert_eq!(scenario.id, id);
            assert_eq!(scenario.expected_namespace, Some(namespace));
            let expected_pairs = if id == "intended" || id == "equal" {
                800
            } else if id.starts_with("threshold-") {
                50
            } else {
                100
            };
            assert_eq!(
                scenario.identity["qualification_total_pairs"],
                json!(expected_pairs)
            );
            assert_eq!(scenario.orientations.len(), 2);
            assert!(scenario
                .orientations
                .iter()
                .all(|orientation| orientation.seats.len() == scenario.config.num_players));
            assert!(fingerprints.insert(canonical_json_sha256(&scenario.identity).unwrap()));
        }
        assert_eq!(fingerprints.len(), 21);
    }

    #[test]
    fn friend_revelation_development_scenario_exercises_finding_friends() {
        let args = args_for_scenario(
            "development-finding-friends",
            Some("dev/ablation/friend-revelation"),
        );
        let scenario = build_scenario(&args).unwrap();
        validate_scenario_seed_domain(&args, &scenario).unwrap();
        assert!(matches!(
            scenario.config.game_mode,
            GameModeSettings::FindingFriends {
                num_friends: Some(1)
            }
        ));
        assert_eq!(scenario.identity["qualification_total_pairs"], Value::Null);
        assert!(scenario
            .orientations
            .iter()
            .all(|orientation| matches!(orientation.routing, AuditRouting::DynamicTeam { .. })));
    }

    #[test]
    fn scenario_contracts_freeze_crossplay_variants_and_rule_strata() {
        let crossplay = [
            ("crossplay-assignment-01", "Enoch-0", "Enoch-0"),
            ("crossplay-assignment-02", "candidate", "Enoch-0"),
            ("crossplay-assignment-03", "Enoch-0", "candidate"),
            ("crossplay-assignment-04", "candidate", "candidate"),
        ];
        for (id, partner, opponent) in crossplay {
            let namespace = QUALIFICATION_SCENARIOS
                .iter()
                .find_map(|(scenario, namespace)| (*scenario == id).then_some(*namespace))
                .unwrap();
            let scenario = build_scenario(&args_for_scenario(id, Some(namespace))).unwrap();
            let seats = scenario.identity["orientations"][0]["seats"]
                .as_array()
                .unwrap();
            assert_eq!(seats[2]["brain"]["checkpoint"], json!(partner));
            assert_eq!(seats[1]["brain"]["checkpoint"], json!(opponent));
            assert_eq!(seats[3]["brain"]["checkpoint"], json!(opponent));
        }

        let slot_1 = build_scenario(&args_for_scenario(
            "configuration-slot-01",
            Some("qual/configuration/slot-01"),
        ))
        .unwrap();
        assert_eq!(slot_1.config.num_players, 2);
        assert_eq!(slot_1.config.decks.len(), 1);
        let slot_3 = build_scenario(&args_for_scenario(
            "configuration-slot-03",
            Some("qual/configuration/slot-03"),
        ))
        .unwrap();
        assert_eq!(slot_3.config.num_players, 6);
        assert_eq!(slot_3.config.decks.len(), 3);

        let finding_friends = build_scenario(&args_for_scenario(
            "finding-friends-contract-02",
            Some("qual/finding-friends/contract-02"),
        ))
        .unwrap();
        assert_eq!(finding_friends.config.num_players, 7);
        assert!(finding_friends
            .orientations
            .iter()
            .all(|orientation| matches!(orientation.routing, AuditRouting::DynamicTeam { .. })));
        assert_eq!(
            finding_friends.identity["orientations"][0]["join_switch_contract"],
            json!("a revealed friend uses its new landlord-team config on its next decision")
        );

        let scoring = build_scenario(&args_for_scenario(
            "scoring-kitty-ruleset-03",
            Some("qual/scoring/kitty-ruleset-03"),
        ))
        .unwrap();
        assert_eq!(
            scoring.identity["rules"]["throw_penalty"],
            json!("TenPointsPerAttempt")
        );
    }

    #[test]
    fn scenario_namespace_and_dynamic_team_routing_fail_closed() {
        let wrong = args_for_scenario("intended", Some("qual/equal"));
        let scenario = build_scenario(&wrong).unwrap();
        assert!(validate_scenario_seed_domain(&wrong, &scenario)
            .unwrap_err()
            .contains("requires seed namespace"));

        let implicit = args_for_scenario("standard", Some("qual/intended"));
        let scenario = build_scenario(&implicit).unwrap();
        assert!(validate_scenario_seed_domain(&implicit, &scenario)
            .unwrap_err()
            .contains("requires its explicit frozen --scenario"));

        assert_eq!(dynamic_team_audit_arm(true, true), EvaluationArm::Candidate);
        assert_eq!(dynamic_team_audit_arm(false, true), EvaluationArm::Control);
        assert_eq!(
            dynamic_team_audit_arm(false, false),
            EvaluationArm::Candidate
        );
        assert_eq!(dynamic_team_audit_arm(true, false), EvaluationArm::Control);
    }

    #[test]
    fn threshold_seed_derivation_is_deterministic_and_domain_separated() {
        let registry_seed = 0x1234_5678_9abc_def0;
        assert_eq!(derive_threshold_deal_seed(registry_seed, 0), registry_seed);
        assert_eq!(
            derive_threshold_deal_seed(registry_seed, 7),
            derive_threshold_deal_seed(registry_seed, 7)
        );
        assert_eq!(
            derive_threshold_deal_seed(registry_seed, 7),
            13_150_005_208_752_510_302
        );
        assert_ne!(
            derive_threshold_deal_seed(registry_seed, 1),
            derive_threshold_deal_seed(registry_seed, 2)
        );
        let bands = (1..=4)
            .map(|index| {
                let id = format!("threshold-situation-{index:02}");
                let namespace = format!("qual/threshold/situation-{index:02}");
                build_scenario(&args_for_scenario(&id, Some(&namespace)))
                    .unwrap()
                    .threshold_band
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert!(bands[0].contains(0) && bands[0].contains(39) && !bands[0].contains(40));
        assert!(bands[1].contains(40) && bands[1].contains(79) && !bands[1].contains(80));
        assert!(bands[2].contains(80) && bands[2].contains(119) && !bands[2].contains(120));
        assert!(bands[3].contains(120) && bands[3].contains(500));

        let args = args_for_scenario(
            "threshold-situation-02",
            Some("qual/threshold/situation-02"),
        );
        let scenario = build_scenario(&args).unwrap();
        let effective = derive_threshold_deal_seed(99, 3);
        let (manifest, digest) = deal_selection_manifest(
            &args,
            &scenario,
            &[DealSelection {
                registry_seed: 99,
                effective_seed: effective,
                selection_attempt: 3,
                selector_non_landlord_points: Some(55),
            }],
        )
        .unwrap();
        assert_eq!(
            manifest["contract"],
            json!("candidate-independent-threshold-control-selection-v1")
        );
        assert_eq!(manifest["records_sha256"], json!(digest));
        assert_eq!(manifest["records"][0]["registry_seed_u64"], json!(99));
        assert_eq!(
            manifest["records"][0]["effective_deal_seed_u64"],
            json!(effective)
        );
    }

    #[test]
    fn telemetry_aggregation_is_fail_closed_and_uses_nearest_rank_latency() {
        let mut audit = ArmAudit::default();
        audit.record_search(&SearchTelemetry {
            strict: true,
            requested_worlds: 8,
            initial_candidate_count: 3,
            candidate_pool_count: 4,
            candidate_work_budget: 24,
            worlds_attempted: 5,
            worlds_accepted: 4,
            worlds_completed: 0,
            candidate_evaluations: 12,
            per_candidate_evaluations: vec![4, 4, 4, 0],
            elapsed: Duration::from_micros(2_500),
            effective_time_budget: Duration::from_millis(2),
            time_bound: true,
            work_bound: false,
            forced_action: false,
            prior_fallback_used: false,
            failure: Some(SearchFailureReason::NoCompletedWorlds),
        });
        assert_eq!(audit.requested_worlds, 8);
        assert_eq!(audit.worlds_attempted, 5);
        assert_eq!(audit.candidate_evaluations, 12);
        assert_eq!(audit.timeout_failures, 1);
        assert_eq!(audit.failure_count(), 1);
        assert!(audit.invalid_counter_total() > 0);
        assert_eq!(percentile_nearest_rank(&[40, 10, 30, 20], 50), Some(20));
        assert_eq!(percentile_nearest_rank(&[40, 10, 30, 20], 95), Some(40));
    }

    #[test]
    fn runner_bridge_has_exact_evidence_and_requested_style_keys() {
        let seed_entry = SeedEntry {
            registry_index: Some(7),
            seed: 99,
        };
        let args = Args {
            pairs: 1,
            seed_plan: SeedPlan {
                entries: vec![seed_entry.clone()],
                source_kind: "test".to_owned(),
                source_path: None,
                protocol: ProtocolIdentity::unverified(Some("qual/intended".to_owned()), "test"),
            },
            environment_identity_only: false,
            feature_input: "none".to_owned(),
            features: EnochFeatures::empty(),
            worlds: 8,
            candidates: 3,
            rollout_tricks: 2,
            runner_style_metrics: vec!["throw-rate".to_owned(), "trump-play-rate".to_owned()],
            scenario_id: "standard".to_owned(),
            mode: WorkMode::FixedWork { deadline_ms: 100 },
        };
        let mut audit = PairAudit::default();
        audit.candidate.decisions = 2;
        audit.candidate.latency_micros = vec![1_000, 3_000];
        audit.candidate.worlds_completed = 4;
        audit.candidate.style.action_decisions = 2;
        audit.candidate.style.lead_plays = 1;
        audit.candidate.style.compound_lead_attempts = 1;
        audit.candidate.style.trump_play_decisions = 1;
        audit.control.decisions = 1;
        audit.control.latency_micros = vec![2_000];
        audit.control.worlds_completed = 3;

        let bridge = runner_record_inputs(
            &args,
            &seed_entry,
            true,
            Some(0.75),
            Some(10.0),
            Some(1.0),
            &audit,
        );
        let object = bridge.as_object().unwrap();
        let expected: BTreeMap<&str, ()> = [
            "seed_index",
            "seed",
            "level_utility_delta",
            "point_margin_delta",
            "win_rate_delta",
            "candidate_latency_ms",
            "control_latency_ms",
            "candidate_completed_worlds",
            "control_completed_worlds",
            "style_metrics",
            "failure_counter_evidence",
        ]
        .iter()
        .map(|name| (*name, ()))
        .collect();
        assert_eq!(
            object
                .keys()
                .map(|name| (name.as_str(), ()))
                .collect::<BTreeMap<_, _>>(),
            expected
        );
        assert_eq!(bridge["candidate_latency_ms"], json!(2.0));
        assert_eq!(bridge["win_rate_delta"], json!(0.25));
        assert_eq!(bridge["style_metrics"].as_object().unwrap().len(), 2);
        assert_eq!(
            bridge["failure_counter_evidence"]
                .as_object()
                .unwrap()
                .len(),
            11
        );
    }
}
