//! Optional learned policies for the non-play phases.
//!
//! Trick play and bidding/kitty exchange have very different action spaces and
//! horizons.  Reusing the play model for all three phases therefore gives the
//! network an unnecessarily hard job.  This module defines small, strict ONNX
//! contracts for a bidding ranker and a kitty-card ranker.  Both are residuals
//! over the existing, mechanics-aware heuristics: with no model configured (the
//! production default), behavior is byte-for-byte the heuristic baseline.
//!
//! Models are deliberately runtime-only candidates.  They require a companion
//! manifest, an exact SHA-256 match, and an explicit non-zero weight.  A bad
//! artifact or inference error permanently opens the circuit breaker and falls
//! back to the heuristic for the rest of the process.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

use serde::Deserialize;
use sha2::{Digest, Sha256};
use shengji_mechanics::bidding::Bid;
use shengji_mechanics::deck::Deck;
use shengji_mechanics::types::{Card, EffectiveSuit, PlayerID, Trump};

use crate::bot::heuristics;
use crate::game_state::draw_phase::DrawPhase;
use crate::game_state::exchange_phase::ExchangePhase;
use crate::settings::{GameMode, PropagatedState};

pub const BID_MODEL_PATH_ENV: &str = "SHENGJI_BID_MODEL_PATH";
pub const BID_MODEL_MANIFEST_ENV: &str = "SHENGJI_BID_MODEL_MANIFEST";
pub const KITTY_MODEL_PATH_ENV: &str = "SHENGJI_KITTY_MODEL_PATH";
pub const KITTY_MODEL_MANIFEST_ENV: &str = "SHENGJI_KITTY_MODEL_MANIFEST";
pub const PHASE_MODEL_WEIGHT_ENV: &str = "SHENGJI_PHASE_MODEL_WEIGHT";

pub const FEATURE_SCHEMA_VERSION: u32 = 1;
pub const BID_FEATURE_DIM: usize = 20;
pub const KITTY_FEATURE_DIM: usize = 20;

pub const BID_FEATURE_NAMES: [&str; BID_FEATURE_DIM] = [
    "hand_size",
    "deal_fraction",
    "bid_count",
    "bid_is_joker",
    "bid_is_big_joker",
    "candidate_is_no_trump",
    "trump_fraction",
    "trump_points",
    "hand_points",
    "pair_count",
    "trump_pair_count",
    "has_trump_tractor",
    "heuristic_strength",
    "joker_count",
    "has_current_bid",
    "current_bid_count",
    "same_suit_as_current_bid",
    "deal_complete",
    "kitty_size",
    "player_count",
];

pub const KITTY_FEATURE_NAMES: [&str; KITTY_FEATURE_DIM] = [
    "card_points",
    "card_is_trump",
    "card_is_joker",
    "card_strength",
    "card_copies",
    "card_is_paired",
    "effective_suit_fraction",
    "would_void_effective_suit",
    "pool_trump_fraction",
    "pool_points",
    "kitty_fraction",
    "heuristic_selected",
    "card_is_ace",
    "card_is_king",
    "card_is_level",
    "card_is_trump_suit",
    "pool_size",
    "effective_suit_remaining_fraction",
    "is_pool_suit_boss",
    "bias",
];

const BID_CONTRACT: &str = "honest_bid_action_ranker";
const KITTY_CONTRACT: &str = "honest_kitty_card_ranker";
const BID_TRAINING_DOMAIN: &str =
    "four_player_tractor_two_full_standard_decks_deal_complete_heuristic_v1";
const KITTY_TRAINING_DOMAIN: &str =
    "four_player_tractor_two_full_standard_decks_initial_exchange_heuristic_v1";
const LOGIT_SEMANTICS: &str = "relative_listwise_rank_only";

type RunnableModel = tract_onnx::prelude::TypedRunnableModel<tract_onnx::prelude::TypedModel>;

#[derive(Debug, Deserialize)]
struct Manifest {
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

struct PhaseModel {
    runnable: RunnableModel,
    feature_dim: usize,
    weight: f64,
    healthy: AtomicBool,
    label: &'static str,
}

#[derive(Clone, Copy)]
struct Contract {
    label: &'static str,
    name: &'static str,
    dim: usize,
    path_env: &'static str,
    manifest_env: &'static str,
}

const BID: Contract = Contract {
    label: "bid",
    name: BID_CONTRACT,
    dim: BID_FEATURE_DIM,
    path_env: BID_MODEL_PATH_ENV,
    manifest_env: BID_MODEL_MANIFEST_ENV,
};

const KITTY: Contract = Contract {
    label: "kitty",
    name: KITTY_CONTRACT,
    dim: KITTY_FEATURE_DIM,
    path_env: KITTY_MODEL_PATH_ENV,
    manifest_env: KITTY_MODEL_MANIFEST_ENV,
};

fn bid_model() -> Option<&'static PhaseModel> {
    static MODEL: OnceLock<Option<PhaseModel>> = OnceLock::new();
    MODEL
        .get_or_init(|| {
            load_from_env(BID).unwrap_or_else(|error| {
                eprintln!("[phase-model:bid] load failed; using heuristic: {error:#}");
                None
            })
        })
        .as_ref()
}

fn kitty_model() -> Option<&'static PhaseModel> {
    static MODEL: OnceLock<Option<PhaseModel>> = OnceLock::new();
    MODEL
        .get_or_init(|| {
            load_from_env(KITTY).unwrap_or_else(|error| {
                eprintln!("[phase-model:kitty] load failed; using heuristic: {error:#}");
                None
            })
        })
        .as_ref()
}

fn companion_manifest_path(model_path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.manifest.json", model_path.display()))
}

fn phase_weight() -> f64 {
    std::env::var(PHASE_MODEL_WEIGHT_ENV)
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(0.0)
        .clamp(0.0, 1.0)
}

fn feature_names(contract: Contract) -> Vec<String> {
    match contract.name {
        BID_CONTRACT => BID_FEATURE_NAMES
            .iter()
            .map(|name| (*name).to_owned())
            .collect(),
        KITTY_CONTRACT => KITTY_FEATURE_NAMES
            .iter()
            .map(|name| (*name).to_owned())
            .collect(),
        _ => Vec::new(),
    }
}

fn training_domain(contract: Contract) -> &'static str {
    match contract.name {
        BID_CONTRACT => BID_TRAINING_DOMAIN,
        KITTY_CONTRACT => KITTY_TRAINING_DOMAIN,
        _ => "",
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn load_from_env(contract: Contract) -> tract_onnx::prelude::TractResult<Option<PhaseModel>> {
    use tract_onnx::prelude::*;

    let weight = phase_weight();
    if weight == 0.0 {
        return Ok(None);
    }
    let Some(path) = std::env::var_os(contract.path_env).filter(|path| !path.is_empty()) else {
        return Ok(None);
    };
    let path = PathBuf::from(path);
    let manifest_path = std::env::var_os(contract.manifest_env)
        .map(PathBuf::from)
        .unwrap_or_else(|| companion_manifest_path(&path));
    let manifest: Manifest = serde_json::from_str(&std::fs::read_to_string(&manifest_path)?)?;
    if manifest.manifest_version != 1
        || manifest.contract != contract.name
        || manifest.feature_schema_version != FEATURE_SCHEMA_VERSION
        || manifest.feature_dim != contract.dim
        || manifest.feature_names != feature_names(contract)
        || manifest.inputs != ["features"]
        || manifest.outputs != ["action_logit"]
        || manifest.output_semantics != ["policy_logit"]
        || manifest.logit_semantics != LOGIT_SEMANTICS
        || manifest.training_domain != training_domain(contract)
        || manifest.serving_status != "experimental_candidate"
        || !manifest.research_only
        || manifest.automatic_production_promotion_allowed
        || manifest.unsafe_training_data
        || !is_sha256(&manifest.model_sha256)
        || !is_sha256(&manifest.golden_sha256)
        || !is_sha256(&manifest.dataset_sha256)
        || !is_sha256(&manifest.dataset_manifest_declared_content_sha256)
        || manifest.dataset_sha256 != manifest.dataset_manifest_declared_content_sha256
        || !manifest
            .dataset_manifest_sha256
            .as_deref()
            .is_some_and(is_sha256)
    {
        anyhow::bail!(
            "unsupported {} phase-model manifest contract",
            contract.label
        );
    }
    let golden_name = Path::new(&manifest.golden_path);
    if manifest.golden_path.is_empty()
        || golden_name.components().count() != 1
        || golden_name.file_name() != Some(golden_name.as_os_str())
    {
        anyhow::bail!("{} phase golden_path must be one file name", contract.label);
    }
    let golden_path = manifest_path
        .parent()
        .unwrap_or_else(|| Path::new(""))
        .join(golden_name);
    let golden_bytes = std::fs::read(golden_path)?;
    if format!("{:x}", Sha256::digest(golden_bytes)) != manifest.golden_sha256 {
        anyhow::bail!("{} phase golden SHA-256 mismatch", contract.label);
    }
    let bytes = std::fs::read(&path)?;
    if bytes.len() < 64 {
        anyhow::bail!("{} phase model is too small to be ONNX", contract.label);
    }
    let digest = format!("{:x}", Sha256::digest(&bytes));
    if digest != manifest.model_sha256 {
        anyhow::bail!("{} phase-model SHA-256 mismatch", contract.label);
    }
    let mut cursor = std::io::Cursor::new(bytes);
    let mut model = tract_onnx::onnx().model_for_read(&mut cursor)?;
    let batch = model.symbols.sym("N");
    model.set_input_fact(
        0,
        f32::fact([batch.to_dim(), (contract.dim as i64).to_dim()]).into(),
    )?;
    Ok(Some(PhaseModel {
        runnable: model.into_optimized()?.into_runnable()?,
        feature_dim: contract.dim,
        weight,
        healthy: AtomicBool::new(true),
        label: contract.label,
    }))
}

fn is_full_standard_deck(deck: &Deck) -> bool {
    !deck.exclude_small_joker
        && !deck.exclude_big_joker
        && deck.min == shengji_mechanics::types::Number::Two
}

fn is_supported_common_domain(
    propagated: &PropagatedState,
    game_mode: &GameMode,
    removed_cards: &[Card],
) -> bool {
    propagated.players().len() == 4
        && propagated.num_decks() == 2
        && matches!(game_mode, GameMode::Tractor)
        && removed_cards.is_empty()
        && propagated
            .decks()
            .ok()
            .is_some_and(|decks| decks.len() == 2 && decks.iter().all(is_full_standard_deck))
}

/// Whether a draw state lies inside the bid model's deliberately narrow
/// training support. This is separate from artifact validation: even an exact,
/// hash-bound model falls back to the heuristic outside this domain.
pub fn bid_domain_supported(p: &DrawPhase) -> bool {
    is_supported_common_domain(p.propagated(), p.game_mode(), p.removed_cards())
        && p.deck().is_empty()
        && p.kitty().len() == 8
        && p.cards_in_play() == 108
}

/// Whether an exchange state lies inside the kitty model's deliberately narrow
/// training support. The combined pool length is invariant while cards move
/// between hand and kitty.
pub fn kitty_domain_supported(p: &ExchangePhase, combined_pool_len: usize) -> bool {
    is_supported_common_domain(p.propagated(), p.game_mode(), p.removed_cards())
        && !p.finalized()
        && p.kitty_size() == 8
        && combined_pool_len == 33
}

impl PhaseModel {
    fn score(&self, features: &[f32]) -> Option<f64> {
        use tract_onnx::prelude::*;

        if !self.healthy.load(Ordering::Relaxed) || features.len() != self.feature_dim {
            return None;
        }
        let input =
            tract_ndarray::Array2::from_shape_vec((1, self.feature_dim), features.to_vec()).ok()?;
        let result = self.runnable.run(tvec!(Tensor::from(input).into()));
        let score = result
            .ok()
            .filter(|outputs| outputs.len() == 1)
            .and_then(|outputs| {
                let output = outputs[0].to_array_view::<f32>().ok()?;
                (output.shape() == [1, 1]).then(|| output[[0, 0]] as f64)
            })
            .filter(|value| value.is_finite());
        if score.is_none() && self.healthy.swap(false, Ordering::Relaxed) {
            eprintln!(
                "[phase-model:{}] inference contract failed; circuit breaker opened",
                self.label
            );
        }
        score
    }
}

/// Schema-stable honest bid features. Hidden hands and the hidden kitty are
/// never consulted; `p` is the acting player's redacted view.
pub fn bid_features(
    p: &DrawPhase,
    me: PlayerID,
    bid: Bid,
    candidate_trump: Trump,
    heuristic_strength: f64,
) -> [f32; BID_FEATURE_DIM] {
    let cards: Vec<Card> = p
        .hands()
        .get(me)
        .ok()
        .map(|hand| Card::cards(hand.iter()).copied().collect())
        .unwrap_or_default();
    let hand_len = cards.len().max(1);
    let trump_cards = cards
        .iter()
        .filter(|card| candidate_trump.effective_suit(**card) == EffectiveSuit::Trump)
        .count();
    let trump_points = cards
        .iter()
        .filter(|card| candidate_trump.effective_suit(**card) == EffectiveSuit::Trump)
        .filter_map(|card| card.points())
        .sum::<usize>();
    let total_points = cards.iter().filter_map(|card| card.points()).sum::<usize>();
    let pairs = Card::count(cards.iter().copied())
        .values()
        .filter(|count| **count >= 2)
        .count();
    let (trump_pairs, tractor) = heuristics::trump_pair_structure(&cards, candidate_trump);
    let full_hand =
        p.cards_in_play().saturating_sub(p.kitty().len()) / p.propagated().players().len().max(1);
    let current = p.winning_bid();
    let same_suit = current.is_some_and(|winning| winning.card.suit() == bid.card.suit());
    let jokers = cards.iter().filter(|card| card.is_joker()).count();

    [
        cards.len() as f32 / 40.0,
        cards.len() as f32 / full_hand.max(1) as f32,
        bid.count as f32 / p.propagated().players().len().max(1) as f32,
        bid.card.is_joker() as u8 as f32,
        (bid.card == Card::BigJoker) as u8 as f32,
        candidate_trump.suit().is_none() as u8 as f32,
        trump_cards as f32 / hand_len as f32,
        trump_points as f32 / 40.0,
        total_points as f32 / 80.0,
        pairs as f32 / 12.0,
        trump_pairs as f32 / 8.0,
        tractor as u8 as f32,
        heuristic_strength as f32 / 40.0,
        jokers as f32 / 4.0,
        current.is_some() as u8 as f32,
        current.map(|winning| winning.count).unwrap_or(0) as f32 / 4.0,
        same_suit as u8 as f32,
        p.deck().is_empty() as u8 as f32,
        p.kitty().len() as f32 / 16.0,
        p.propagated().players().len() as f32 / 8.0,
    ]
}

fn minmax(values: &[f64]) -> Option<Vec<f64>> {
    let minimum = values.iter().copied().reduce(f64::min)?;
    let maximum = values.iter().copied().reduce(f64::max)?;
    let span = maximum - minimum;
    if !minimum.is_finite() || !maximum.is_finite() || span <= 1e-9 {
        return None;
    }
    Some(
        values
            .iter()
            .map(|value| (value - minimum) / span)
            .collect(),
    )
}

fn blend_rankings(heuristic: &[f64], logits: &[f64], weight: f64) -> Option<Vec<f64>> {
    if heuristic.len() != logits.len() || heuristic.is_empty() {
        return None;
    }
    let model = minmax(logits)?;
    let heuristic = minmax(heuristic).unwrap_or_else(|| vec![0.5; logits.len()]);
    Some(
        heuristic
            .into_iter()
            .zip(model)
            .map(|(heuristic, model)| (1.0 - weight) * heuristic + weight * model)
            .collect(),
    )
}

/// Rank fully-dealt bid candidates with an explicitly enabled phase model.
/// Listwise logits have no calibrated absolute zero, so they are normalized
/// jointly and blended only as ranks. The caller must use the unmodified
/// heuristic strength for bid/pass thresholds.
pub fn rank_bid_candidates(
    p: &DrawPhase,
    me: PlayerID,
    candidates: &[(Bid, Trump, f64)],
) -> Vec<f64> {
    let heuristic = candidates
        .iter()
        .map(|(_, _, strength)| *strength)
        .collect::<Vec<_>>();
    let Some(model) = bid_model() else {
        return heuristic;
    };
    let logits = candidates
        .iter()
        .map(|(bid, trump, strength)| model.score(&bid_features(p, me, *bid, *trump, *strength)))
        .collect::<Option<Vec<_>>>();
    logits
        .and_then(|logits| blend_rankings(&heuristic, &logits, model.weight))
        .unwrap_or(heuristic)
}

/// Schema-stable per-physical-card features for a kitty burial candidate.
pub fn kitty_features(
    pool: &[Card],
    trump: Trump,
    kitty_size: usize,
    card: Card,
    baseline_selected: bool,
) -> [f32; KITTY_FEATURE_DIM] {
    let hand_len = pool.len().max(1);
    let suit = trump.effective_suit(card);
    let suit_len = pool
        .iter()
        .filter(|candidate| trump.effective_suit(**candidate) == suit)
        .count();
    let copies = pool.iter().filter(|candidate| **candidate == card).count();
    let total_trumps = pool
        .iter()
        .filter(|candidate| trump.effective_suit(**candidate) == EffectiveSuit::Trump)
        .count();
    let total_points = pool
        .iter()
        .filter_map(|candidate| candidate.points())
        .sum::<usize>();
    let points = card.points().unwrap_or(0);
    let strength = heuristics::card_strength(trump, card);
    let number = card.number();
    let is_pool_suit_boss = pool
        .iter()
        .filter(|candidate| trump.effective_suit(**candidate) == suit)
        .all(|candidate| trump.compare(card, *candidate) != std::cmp::Ordering::Less);

    [
        points as f32 / 10.0,
        (suit == EffectiveSuit::Trump) as u8 as f32,
        card.is_joker() as u8 as f32,
        strength as f32 / 20.0,
        copies as f32 / 4.0,
        (copies >= 2) as u8 as f32,
        suit_len as f32 / hand_len as f32,
        (suit_len == 1) as u8 as f32,
        total_trumps as f32 / hand_len as f32,
        total_points as f32 / 160.0,
        kitty_size as f32 / hand_len as f32,
        baseline_selected as u8 as f32,
        number.is_some_and(|n| n == shengji_mechanics::types::Number::Ace) as u8 as f32,
        number.is_some_and(|n| n == shengji_mechanics::types::Number::King) as u8 as f32,
        trump
            .number()
            .is_some_and(|number| card.number() == Some(number)) as u8 as f32,
        trump.suit().is_some_and(|suit| card.suit() == Some(suit)) as u8 as f32,
        pool.len() as f32 / 64.0,
        suit_len.saturating_sub(1) as f32 / hand_len as f32,
        is_pool_suit_boss as u8 as f32,
        1.0,
    ]
}

fn baseline_membership(cards: &[Card]) -> std::collections::HashMap<Card, usize> {
    Card::count(cards.iter().copied())
}

/// Optionally refine the heuristic burial as a bounded learned residual.  The
/// model ranks physical cards, so duplicate copies retain correct multiplicity.
/// Both teacher membership and model logits are normalized jointly: listwise
/// logits have no meaningful absolute offset or scale.
pub fn choose_kitty(pool: &[Card], trump: Trump, kitty_size: usize, advanced: bool) -> Vec<Card> {
    let baseline = if advanced {
        heuristics::choose_kitty_enoch(pool, trump, kitty_size)
    } else {
        heuristics::choose_kitty(pool, trump, kitty_size)
    };
    let Some(model) = kitty_model() else {
        return baseline;
    };
    let mut remaining = baseline_membership(&baseline);
    let mut physical = Vec::with_capacity(pool.len());
    for (index, &card) in pool.iter().enumerate() {
        let selected = remaining.get_mut(&card).is_some_and(|count| {
            if *count == 0 {
                false
            } else {
                *count -= 1;
                true
            }
        });
        let features = kitty_features(pool, trump, kitty_size, card, selected);
        physical.push((index, card, selected, features));
    }
    let heuristic = physical
        .iter()
        .map(|(_, _, selected, _)| *selected as u8 as f64)
        .collect::<Vec<_>>();
    let logits = physical
        .iter()
        .map(|(_, _, _, features)| model.score(features))
        .collect::<Option<Vec<_>>>();
    let Some(scores) = logits.and_then(|logits| blend_rankings(&heuristic, &logits, model.weight))
    else {
        return baseline;
    };
    let mut ranked = physical
        .into_iter()
        .zip(scores)
        .map(|((index, card, _, _), score)| (score, index, card))
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .0
            .partial_cmp(&left.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.1.cmp(&right.1))
    });
    ranked
        .into_iter()
        .take(kitty_size.min(pool.len()))
        .map(|(_, _, card)| card)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use shengji_mechanics::deck::Deck;
    use shengji_mechanics::types::{Number, Suit};

    use crate::bot::harness;

    #[test]
    fn phase_feature_contract_names_are_stable() {
        assert_eq!(feature_names(BID).len(), BID_FEATURE_DIM);
        assert_eq!(feature_names(KITTY).len(), KITTY_FEATURE_DIM);
        assert_eq!(feature_names(BID)[0], "hand_size");
        assert_eq!(feature_names(KITTY)[KITTY_FEATURE_DIM - 1], "bias");
    }

    #[test]
    fn kitty_features_are_finite_and_track_baseline() {
        let ace = Card::Suited {
            suit: Suit::Hearts,
            number: Number::Ace,
        };
        let five = Card::Suited {
            suit: Suit::Clubs,
            number: Number::Five,
        };
        let trump = Trump::Standard {
            suit: Suit::Spades,
            number: Number::Two,
        };
        let features = kitty_features(&[ace, five], trump, 1, five, true);
        assert!(features.iter().all(|value| value.is_finite()));
        assert_eq!(features[11], 1.0);
    }

    #[test]
    fn disabled_kitty_model_is_exact_heuristic_baseline() {
        // Tests run without candidate-model environment variables. This guards
        // the important production-default invariant.
        let cards = [
            Card::Suited {
                suit: Suit::Hearts,
                number: Number::Ace,
            },
            Card::Suited {
                suit: Suit::Clubs,
                number: Number::Five,
            },
        ];
        let trump = Trump::Standard {
            suit: Suit::Spades,
            number: Number::Two,
        };
        assert_eq!(
            choose_kitty(&cards, trump, 1, false),
            heuristics::choose_kitty(&cards, trump, 1)
        );
    }

    #[test]
    fn bid_rank_blend_is_invariant_to_logit_offset_and_scale() {
        let heuristic = [0.1, 0.7, 1.0];
        let first = blend_rankings(&heuristic, &[-3.0, 0.0, 2.0], 0.4).unwrap();
        let shifted = blend_rankings(&heuristic, &[4.0, 10.0, 14.0], 0.4).unwrap();
        assert!(first
            .iter()
            .zip(shifted)
            .all(|(left, right)| (left - right).abs() < 1e-12));
    }

    #[test]
    fn bid_model_domain_is_exactly_the_exported_standard_setup() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let mut standard =
            harness::seeded_draw_phase(&[Deck::default(), Deck::default()], &mut rng);
        while !standard.deck().is_empty() {
            let actor = standard.next_player().unwrap();
            standard.draw_card(actor).unwrap();
        }
        assert!(bid_domain_supported(&standard));

        let short = Deck {
            min: Number::Five,
            ..Deck::default()
        };
        let mut unsupported = harness::seeded_draw_phase(&[short.clone(), short], &mut rng);
        while !unsupported.deck().is_empty() {
            let actor = unsupported.next_player().unwrap();
            unsupported.draw_card(actor).unwrap();
        }
        assert!(!bid_domain_supported(&unsupported));
    }
}
