//! Optional learned hidden-card proposal for determinized search.
//!
//! The model never changes legality: hard conservation/capacity/void constraints
//! remain in `determinize`. It only supplies log weights over already-legal hidden
//! destinations, and the constrained sampler applies those weights through its
//! bounded Metropolis refinement. The network predicts per-card destination
//! marginals; multiplying those scores over physical copies is an approximate
//! joint proposal, not a calibrated posterior. Leave `SHENGJI_BELIEF_WEIGHT`
//! zero/unset to use the fresh neutral constrained sampler.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use rand::Rng;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use shengji_mechanics::bidding::Bid;
use shengji_mechanics::trick::PlayedCards;
use shengji_mechanics::types::{Card, EffectiveSuit, PlayerID, Trump, FULL_DECK};

use crate::bot::determinize::{
    sample_hidden_hands_with_proposal, AssignmentProposalContext, DeterminizedWorld,
    HiddenAssignmentProposal, HiddenCardLocation, PersistentBelief,
};

pub const BELIEF_MODEL_PATH_ENV: &str = "SHENGJI_BELIEF_MODEL_PATH";
pub const BELIEF_MANIFEST_PATH_ENV: &str = "SHENGJI_BELIEF_MODEL_MANIFEST";
pub const BELIEF_WEIGHT_ENV: &str = "SHENGJI_BELIEF_WEIGHT";
pub const PERSISTENT_BELIEF_ENV: &str = "SHENGJI_PERSISTENT_BELIEF";
pub const GOLDEN_VECTOR_CONTRACT: &str =
    "synthetic deterministic tensor parity only; state-derived encoder golden pending";
/// Frozen schema-v1 feature dimension. The layout remains supported, but old
/// artifacts must be regenerated with the current strict lineage manifest.
pub const FEATURE_DIM: usize = 20;
/// Schema v2 appends public progress, four ordered bids, and eight ordered play
/// events to the frozen schema-v1 row. The power-of-two width is convenient for
/// small MLPs while retaining the phase signals omitted by schema v1.
pub const FEATURE_DIM_V2: usize = 128;
const BID_EVENTS: usize = 4;
const BID_EVENT_DIM: usize = 6;
const PLAY_EVENTS: usize = 8;
const PLAY_EVENT_DIM: usize = 10;
const DESTINATIONS: usize = 4;
const CONTRACT: &str = "offline_honest_card_location_belief";
const SUPPORTED_GAME_CONTRACT: &str = "tractor:4p:2x-standard:kitty8:no-removed";
const TRAINING_BEHAVIOUR_POLICY_DOMAIN: &str = "bidding=expert;exchange=easy;play=easy";
const PROPOSAL_FACTORIZATION: &str =
    "per-card destination marginals multiplied over physical-copy assignments; approximate joint";
const ENCODER_SOURCE: &str = include_str!("belief.rs");
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
    golden_path: String,
    golden_sha256: String,
    dataset_sha256: String,
    dataset_manifest_sha256: String,
    dataset_manifest_declared_csv_sha256: String,
    encoder_contract: String,
    encoder_source_sha256: String,
    golden_vector_contract: String,
    supported_game_contract: String,
    training_behaviour_policy_domain: String,
    proposal_factorization: String,
    research_only: bool,
    auto_promotion: bool,
    #[serde(rename = "unsafe")]
    unsafe_artifact: bool,
    serving_status: String,
}

struct OnnxBeliefProposal {
    runnable: RunnableModel,
    healthy: AtomicBool,
    log_weight_scale: f64,
    feature_schema_version: u32,
    feature_dim: usize,
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

struct CachedBelief {
    last_used: u64,
    belief: Arc<Mutex<PersistentBelief>>,
}

fn persistent_beliefs() -> &'static Mutex<HashMap<[u8; 32], CachedBelief>> {
    static CACHE: OnceLock<Mutex<HashMap<[u8; 32], CachedBelief>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn persistent_belief_flag(value: Option<&str>) -> bool {
    value == Some("1")
}

fn persistent_belief_enabled() -> bool {
    persistent_belief_flag(std::env::var(PERSISTENT_BELIEF_ENV).ok().as_deref())
}

fn update_usize(digest: &mut Sha256, value: usize) {
    Digest::update(digest, value.to_le_bytes());
}

fn update_card(digest: &mut Sha256, card: Card) {
    Digest::update(digest, (card.as_char() as u32).to_le_bytes());
}

/// Stable, observer-private hand key. It is derived only from public setup plus
/// the observer's own initial post-exchange cards (current cards + their public
/// plays), never from an opponent hand. A rare cross-room collision is harmless:
/// every cached particle is reconditioned and revalidated against the encoded
/// constraints before use.
fn persistent_hand_key(
    p: &crate::game_state::play_phase::PlayPhase,
    observer: PlayerID,
) -> Option<[u8; 32]> {
    if !p.public_history_complete()
        || p.public_play_history()
            .iter()
            .flat_map(|trick| trick.iter())
            .chain(p.trick().played_cards())
            .flat_map(|played| played.cards.iter())
            .any(|card| *card == Card::Unknown)
    {
        return None;
    }
    let mut digest = Sha256::new();
    Digest::update(&mut digest, b"shengji-persistent-belief-v1");
    update_usize(&mut digest, observer.0);
    update_usize(&mut digest, p.landlord().0);
    update_usize(&mut digest, p.exchanger().0);
    update_usize(&mut digest, p.num_decks());
    Digest::update(&mut digest, format!("{:?}", p.trump()).as_bytes());
    for player in p.propagated().players() {
        update_usize(&mut digest, player.id.0);
    }
    for bid in p.public_bids() {
        update_usize(&mut digest, bid.id.0);
        update_card(&mut digest, bid.card);
        update_usize(&mut digest, bid.count);
        update_usize(&mut digest, bid.epoch);
    }
    let mut configured = p.configured_cards_for_determinization()?;
    configured.sort_unstable_by_key(|card| card.as_char());
    for card in configured {
        update_card(&mut digest, card);
    }

    let mut initial_observer_cards = p
        .hands()
        .get(observer)
        .ok()?
        .iter()
        .filter(|(card, _)| **card != Card::Unknown)
        .flat_map(|(&card, &count)| std::iter::repeat_n(card, count))
        .collect::<Vec<_>>();
    initial_observer_cards.extend(
        p.public_play_history()
            .iter()
            .flat_map(|trick| trick.iter())
            .chain(p.trick().played_cards())
            .filter(|played| played.id == observer)
            .flat_map(|played| played.cards.iter().copied()),
    );
    initial_observer_cards.sort_unstable_by_key(|card| card.as_char());
    for card in initial_observer_cards {
        update_card(&mut digest, card);
    }
    Some(digest.finalize().into())
}

/// Search entry point for hidden-world sampling. Fresh sampling is the default.
///
/// `SHENGJI_PERSISTENT_BELIEF=1` opts into an experimental retained-particle
/// cache. That path is not an exact posterior: conditioning duplicate physical
/// copies needs multiplicity weighting that the current transition does not yet
/// implement. Learned proposals are always sampled fresh because their
/// sequence-conditioned weights change after each public action.
pub fn sample_persistent_world<R: Rng>(
    p: &crate::game_state::play_phase::PlayPhase,
    observer: PlayerID,
    rng: &mut R,
) -> Option<DeterminizedWorld> {
    let proposal = loaded_proposal();
    if proposal.is_some() || !persistent_belief_enabled() {
        return sample_hidden_hands_with_proposal(p, observer, rng, proposal);
    }
    let Some(key) = persistent_hand_key(p, observer) else {
        return sample_hidden_hands_with_proposal(p, observer, rng, proposal);
    };

    const MAX_HANDS: usize = 128;
    const PARTICLES_PER_HAND: usize = 48;
    static CLOCK: AtomicU64 = AtomicU64::new(1);
    let now = CLOCK.fetch_add(1, Ordering::Relaxed);
    let belief = {
        let mut cache = persistent_beliefs()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !cache.contains_key(&key) && cache.len() >= MAX_HANDS {
            if let Some(oldest) = cache
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| *key)
            {
                cache.remove(&oldest);
            }
        }
        let entry = cache.entry(key).or_insert_with(|| CachedBelief {
            last_used: now,
            belief: Arc::new(Mutex::new(PersistentBelief::new(
                observer,
                PARTICLES_PER_HAND,
            ))),
        });
        entry.last_used = now;
        Arc::clone(&entry.belief)
    };
    let mut belief = belief
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    crate::bot::determinize::sample_hidden_hands_with_persistent_belief(
        p,
        observer,
        rng,
        None,
        &mut belief,
    )
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
    let supported_schema = matches!(
        (
            manifest.manifest_version,
            manifest.feature_schema_version,
            manifest.feature_dim,
        ),
        (1, 1, FEATURE_DIM) | (2, 2, FEATURE_DIM_V2)
    );
    let expected_features = belief_feature_names(manifest.feature_schema_version);
    let expected_encoder_contract =
        belief_encoder_contract(manifest.feature_schema_version).unwrap_or_default();
    if !supported_schema
        || manifest.contract != CONTRACT
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
        || manifest.training_behaviour_policy_domain != TRAINING_BEHAVIOUR_POLICY_DOMAIN
        || manifest.proposal_factorization != PROPOSAL_FACTORIZATION
        || manifest.encoder_contract != expected_encoder_contract
        || manifest.encoder_source_sha256 != belief_encoder_source_sha256()
        || manifest.golden_vector_contract != GOLDEN_VECTOR_CONTRACT
        || !manifest.research_only
        || manifest.auto_promotion
        || manifest.unsafe_artifact
        || !is_sha256(&manifest.model_sha256)
        || !is_sha256(&manifest.golden_sha256)
        || !is_sha256(&manifest.dataset_sha256)
        || !is_sha256(&manifest.dataset_manifest_sha256)
        || !is_sha256(&manifest.dataset_manifest_declared_csv_sha256)
        || manifest.dataset_sha256 != manifest.dataset_manifest_declared_csv_sha256
        || manifest.serving_status != "experimental_candidate"
    {
        anyhow::bail!("unsupported belief model manifest contract");
    }
    let golden_name = Path::new(&manifest.golden_path);
    if manifest.golden_path.is_empty()
        || golden_name.components().count() != 1
        || golden_name.file_name() != Some(golden_name.as_os_str())
    {
        anyhow::bail!("belief golden_path must be one relative file name");
    }
    let golden_path = manifest_path
        .parent()
        .unwrap_or_else(|| Path::new(""))
        .join(golden_name);
    let golden_bytes = std::fs::read(golden_path)?;
    if format!("{:x}", Sha256::digest(golden_bytes)) != manifest.golden_sha256 {
        anyhow::bail!("belief golden SHA-256 does not match its manifest");
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
        f32::fact([batch.to_dim(), (manifest.feature_dim as i64).to_dim()]).into(),
    )?;
    model.set_input_fact(
        1,
        f32::fact([batch.to_dim(), (DESTINATIONS as i64).to_dim()]).into(),
    )?;
    Ok(Some(OnnxBeliefProposal {
        runnable: model.into_optimized()?.into_runnable()?,
        healthy: AtomicBool::new(true),
        log_weight_scale,
        feature_schema_version: manifest.feature_schema_version,
        feature_dim: manifest.feature_dim,
    }))
}

/// Ordered feature contract for offline generators and parity validators.
pub fn belief_feature_names(schema_version: u32) -> Vec<String> {
    let mut names = (0..FEATURE_DIM)
        .map(|index| format!("b{index}"))
        .collect::<Vec<_>>();
    if schema_version != 2 {
        return names;
    }
    names.extend([
        "seq.completed_tricks".to_owned(),
        "seq.current_trick_occupancy".to_owned(),
        "seq.bid_count".to_owned(),
        "seq.failed_throw_count".to_owned(),
    ]);
    const BID_NAMES: [&str; BID_EVENT_DIM] = [
        "present",
        "relative_actor",
        "sequence_position",
        "card_identity",
        "count",
        "epoch",
    ];
    for event in 0..BID_EVENTS {
        names.extend(
            BID_NAMES
                .iter()
                .map(|name| format!("seq.bid_{event}.{name}")),
        );
    }
    const PLAY_NAMES: [&str; PLAY_EVENT_DIM] = [
        "present",
        "relative_actor",
        "trick_recency",
        "position_in_trick",
        "card_count",
        "point_density",
        "trump_fraction",
        "led_suit_fraction",
        "failed_throw_count",
        "first_effective_suit",
    ];
    for event in 0..PLAY_EVENTS {
        names.extend(
            PLAY_NAMES
                .iter()
                .map(|name| format!("seq.play_{event}.{name}")),
        );
    }
    names
}

fn companion_manifest_path(model_path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.manifest.json", model_path.display()))
}

pub fn belief_encoder_contract(schema_version: u32) -> Option<&'static str> {
    match schema_version {
        1 => Some("shengji-belief-encoder-v1"),
        2 => Some("shengji-belief-encoder-v2"),
        _ => None,
    }
}

pub fn belief_encoder_source_sha256() -> String {
    format!("{:x}", Sha256::digest(ENCODER_SOURCE.as_bytes()))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
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

fn effective_suit_feature(suit: EffectiveSuit) -> f32 {
    match suit {
        EffectiveSuit::Unknown => 0.0,
        EffectiveSuit::Clubs => 0.2,
        EffectiveSuit::Diamonds => 0.4,
        EffectiveSuit::Spades => 0.6,
        EffectiveSuit::Hearts => 0.8,
        EffectiveSuit::Trump => 1.0,
    }
}

#[derive(Clone, Copy)]
struct SequenceEvent<'a> {
    played: &'a PlayedCards,
    trick_index: usize,
    position: usize,
    led_suit: EffectiveSuit,
}

/// Encode the ordered tail of the honest public action stream. This deliberately
/// consumes only completed public history and cards already on the table. It is
/// kept separate from the frozen schema-v1 encoder so old artifacts remain byte-
/// for-byte compatible and schema-v2 artifacts must opt in explicitly.
fn encode_public_sequence_features(
    history: &[Vec<PlayedCards>],
    current: &[PlayedCards],
    seats: &[PlayerID],
    observer: PlayerID,
    trump: Trump,
    bids: &[Bid],
) -> [f32; FEATURE_DIM_V2 - FEATURE_DIM] {
    let mut encoded = [0.0; FEATURE_DIM_V2 - FEATURE_DIM];
    encoded[0] = (history.len() as f32 / 32.0).min(1.0);
    encoded[1] = current.len() as f32 / seats.len().max(1) as f32;
    encoded[2] = (bids.len() as f32 / 16.0).min(1.0);
    encoded[3] = history
        .iter()
        .flat_map(|trick| trick.iter())
        .chain(current.iter())
        .filter(|played| !played.bad_throw_cards.is_empty())
        .count() as f32
        / 16.0;
    encoded[3] = encoded[3].min(1.0);

    let observer_index = seats.iter().position(|seat| *seat == observer).unwrap_or(0);
    let relative_actor = |actor: PlayerID| {
        let actor_index = seats
            .iter()
            .position(|seat| *seat == actor)
            .unwrap_or(observer_index);
        if seats.is_empty() {
            0.0
        } else {
            let offset = (actor_index + seats.len() - observer_index) % seats.len();
            offset as f32 / seats.len().saturating_sub(1).max(1) as f32
        }
    };
    let bid_start = bids.len().saturating_sub(BID_EVENTS);
    let bid_padding = BID_EVENTS - (bids.len() - bid_start);
    for (tail_index, bid) in bids[bid_start..].iter().enumerate() {
        let base = 4 + (bid_padding + tail_index) * BID_EVENT_DIM;
        encoded[base] = 1.0;
        encoded[base + 1] = relative_actor(bid.id);
        encoded[base + 2] = (bid_start + tail_index + 1) as f32 / bids.len().max(1) as f32;
        encoded[base + 3] = FULL_DECK
            .iter()
            .position(|card| *card == bid.card)
            .unwrap_or(0) as f32
            / 53.0;
        encoded[base + 4] = (bid.count as f32 / 4.0).min(1.0);
        encoded[base + 5] = (bid.epoch as f32 / 4.0).min(1.0);
    }

    let led_suit = |trick: &[PlayedCards]| {
        trick
            .first()
            .and_then(|played| played.cards.iter().find(|card| **card != Card::Unknown))
            .map(|card| trump.effective_suit(*card))
            .unwrap_or(EffectiveSuit::Unknown)
    };
    let mut events = Vec::new();
    for (trick_index, trick) in history.iter().enumerate() {
        let led = led_suit(trick);
        events.extend(
            trick
                .iter()
                .enumerate()
                .map(|(position, played)| SequenceEvent {
                    played,
                    trick_index,
                    position,
                    led_suit: led,
                }),
        );
    }
    let current_index = history.len();
    let current_led = led_suit(current);
    events.extend(
        current
            .iter()
            .enumerate()
            .map(|(position, played)| SequenceEvent {
                played,
                trick_index: current_index,
                position,
                led_suit: current_led,
            }),
    );

    let start = events.len().saturating_sub(PLAY_EVENTS);
    // Right-align short histories so event_7 is always the most recent action.
    let padding = PLAY_EVENTS - (events.len() - start);
    let trick_denominator = (history.len() + usize::from(!current.is_empty())).max(1) as f32;
    for (tail_index, event) in events[start..].iter().enumerate() {
        let base = 4 + BID_EVENTS * BID_EVENT_DIM + (padding + tail_index) * PLAY_EVENT_DIM;
        let cards = &event.played.cards;
        let visible_cards = cards
            .iter()
            .copied()
            .filter(|card| *card != Card::Unknown)
            .collect::<Vec<_>>();
        let points = visible_cards
            .iter()
            .map(|card| card.points().unwrap_or(0))
            .sum::<usize>();
        let first_suit = visible_cards
            .first()
            .map(|card| trump.effective_suit(*card))
            .unwrap_or(EffectiveSuit::Unknown);
        encoded[base] = 1.0;
        encoded[base + 1] = relative_actor(event.played.id);
        encoded[base + 2] = (event.trick_index + 1) as f32 / trick_denominator;
        encoded[base + 3] = event.position as f32 / seats.len().saturating_sub(1).max(1) as f32;
        encoded[base + 4] = (cards.len() as f32 / 8.0).min(1.0);
        encoded[base + 5] = if visible_cards.is_empty() {
            0.0
        } else {
            points as f32 / (10 * visible_cards.len()) as f32
        };
        encoded[base + 6] = if visible_cards.is_empty() {
            0.0
        } else {
            visible_cards
                .iter()
                .filter(|card| trump.effective_suit(**card) == EffectiveSuit::Trump)
                .count() as f32
                / visible_cards.len() as f32
        };
        encoded[base + 7] = if visible_cards.is_empty() {
            0.0
        } else {
            visible_cards
                .iter()
                .filter(|card| trump.effective_suit(**card) == event.led_suit)
                .count() as f32
                / visible_cards.len() as f32
        };
        encoded[base + 8] = (event.played.bad_throw_cards.len() as f32 / 8.0).min(1.0);
        encoded[base + 9] = effective_suit_feature(first_suit);
    }
    encoded
}

/// Strict schema-v2 encoder. It preserves the first twenty schema-v1 values and
/// appends an ordered public-history tail. Callers cannot accidentally serve a
/// v2 row to a v1 artifact because model loading checks both schema and width.
pub fn encode_belief_features_v2(
    p: &crate::game_state::play_phase::PlayPhase,
    observer: PlayerID,
    knowledge: &crate::bot::determinize::Knowledge,
    card: Card,
) -> ([f32; FEATURE_DIM_V2], [f32; DESTINATIONS]) {
    let (v1, mask) = encode_belief_features(p, observer, knowledge, card);
    let seats = p
        .propagated()
        .players()
        .iter()
        .map(|seat| seat.id)
        .collect::<Vec<_>>();
    let sequence = encode_public_sequence_features(
        p.public_play_history(),
        p.trick().played_cards(),
        &seats,
        observer,
        p.trump(),
        p.public_bids(),
    );
    let mut features = [0.0; FEATURE_DIM_V2];
    features[..FEATURE_DIM].copy_from_slice(&v1);
    features[FEATURE_DIM..].copy_from_slice(&sequence);
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
        let rows: Vec<(Vec<f32>, [f32; DESTINATIONS])> = unique_cards
            .iter()
            .map(|&card| {
                if self.feature_schema_version == 2 {
                    let (features, mask) = encode_belief_features_v2(
                        context.view,
                        context.observer,
                        context.knowledge,
                        card,
                    );
                    (features.to_vec(), mask)
                } else {
                    let (features, mask) = encode_belief_features(
                        context.view,
                        context.observer,
                        context.knowledge,
                        card,
                    );
                    (features.to_vec(), mask)
                }
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
        let Ok(features) = tract_ndarray::Array2::from_shape_vec(
            (unique_cards.len(), self.feature_dim),
            feature_data,
        ) else {
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

#[cfg(test)]
mod sequence_feature_tests {
    use super::{
        belief_feature_names, encode_public_sequence_features, persistent_belief_flag, BID_EVENTS,
        BID_EVENT_DIM, FEATURE_DIM_V2, PLAY_EVENT_DIM,
    };
    use shengji_mechanics::bidding::Bid;
    use shengji_mechanics::trick::PlayedCards;
    use shengji_mechanics::types::{Card, Number, PlayerID, Suit, Trump};

    fn play(id: usize, cards: Vec<Card>) -> PlayedCards {
        PlayedCards {
            id: PlayerID(id),
            cards,
            bad_throw_cards: vec![],
            better_player: None,
        }
    }

    fn card(suit: Suit, number: Number) -> Card {
        Card::Suited { suit, number }
    }

    #[test]
    fn schema_v2_names_and_ordered_tail_are_stable() {
        let seats = [PlayerID(0), PlayerID(1), PlayerID(2), PlayerID(3)];
        let trump = Trump::Standard {
            suit: Suit::Hearts,
            number: Number::Two,
        };
        let history = vec![vec![
            play(0, vec![card(Suit::Clubs, Number::Five)]),
            play(1, vec![card(Suit::Clubs, Number::Ten)]),
        ]];
        let current = vec![play(2, vec![card(Suit::Hearts, Number::King)])];
        let bids = vec![
            Bid {
                id: PlayerID(1),
                card: card(Suit::Clubs, Number::Two),
                count: 1,
                epoch: 0,
            },
            Bid {
                id: PlayerID(2),
                card: Card::SmallJoker,
                count: 2,
                epoch: 1,
            },
        ];
        let encoded =
            encode_public_sequence_features(&history, &current, &seats, PlayerID(0), trump, &bids);

        assert_eq!(belief_feature_names(2).len(), FEATURE_DIM_V2);
        assert_eq!(belief_feature_names(1).len(), super::FEATURE_DIM);
        assert_eq!(encoded[0], 1.0 / 32.0);
        assert_eq!(encoded[1], 0.25);
        assert_eq!(encoded[2], 2.0 / 16.0);
        let first_bid = 4 + 2 * BID_EVENT_DIM;
        assert_eq!(encoded[first_bid], 1.0);
        assert!((encoded[first_bid + 1] - 1.0 / 3.0).abs() < f32::EPSILON);
        assert_eq!(encoded[first_bid + 2], 0.5);
        // Three actions are right-aligned into slots 5, 6, 7.
        let play_offset = 4 + BID_EVENTS * BID_EVENT_DIM;
        let first = play_offset + 5 * PLAY_EVENT_DIM;
        let latest = play_offset + 7 * PLAY_EVENT_DIM;
        assert_eq!(encoded[first], 1.0);
        assert!((encoded[first + 2] - 0.5).abs() < f32::EPSILON);
        assert_eq!(encoded[first + 7], 1.0);
        assert_eq!(encoded[latest], 1.0);
        assert!((encoded[latest + 1] - 2.0 / 3.0).abs() < f32::EPSILON);
        assert_eq!(encoded[latest + 2], 1.0);
        assert_eq!(encoded[latest + 6], 1.0);
        assert_eq!(encoded[latest + 7], 1.0);
        assert_eq!(encoded[latest + 9], 1.0);
    }

    #[test]
    fn persistent_belief_requires_an_exact_opt_in() {
        assert!(!persistent_belief_flag(None));
        assert!(!persistent_belief_flag(Some("")));
        assert!(!persistent_belief_flag(Some("0")));
        assert!(!persistent_belief_flag(Some("true")));
        assert!(persistent_belief_flag(Some("1")));
    }
}
