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
//! candidates exist, it returns `None` and the policy falls back to the `Hard`
//! determinized search (which itself falls back to the heuristic), so Expert is
//! never illegal/None.

use std::sync::OnceLock;

use shengji_mechanics::types::{Card, EffectiveSuit, Number, PlayerID, Trump};

use crate::bot::heuristics::{self};
use crate::game_state::play_phase::PlayPhase;

/// The fixed length of the per-candidate feature vector. Must match the training
/// script's input dimension exactly. If you change the encoding, retrain.
pub const FEATURE_DIM: usize = 28;

/// The embedded ONNX model (a small MLP scoring one candidate's features to a
/// scalar logit). If training has not produced a model yet, this file may be a
/// placeholder; [`model`] handles a missing/invalid model gracefully by
/// returning `None`, which makes the Expert tier fall back to `Hard`.
///
/// The asset lives under `core/src/bot/` so it travels with the crate (and the
/// pure-Rust `tract-onnx` runtime builds in the musl Docker image — no
/// `onnxruntime` / `ort` C dependency).
static MODEL_BYTES: &[u8] = include_bytes!("expert_model.onnx");

type Model = tract_onnx::prelude::TypedRunnableModel<tract_onnx::prelude::TypedModel>;

/// Lazily-parsed model, shared across all Expert decisions. `None` means the
/// model could not be loaded (e.g. the embedded bytes are a placeholder), in
/// which case the caller falls back to the Hard heuristic.
fn model() -> Option<&'static Model> {
    static MODEL: OnceLock<Option<Model>> = OnceLock::new();
    MODEL
        .get_or_init(|| match load_model() {
            Ok(m) => Some(m),
            Err(_) => None,
        })
        .as_ref()
}

/// Parse and optimize the embedded ONNX model into a runnable plan. The model
/// takes a single input named `x` of shape `[N, FEATURE_DIM]` (a batch of N
/// candidates) and produces `[N, 1]` logits.
fn load_model() -> tract_onnx::prelude::TractResult<Model> {
    use tract_onnx::prelude::*;

    // A near-empty / placeholder file can't be a valid ONNX graph; bail early so
    // we fall back to Hard rather than erroring deeper in the parser.
    if MODEL_BYTES.len() < 64 {
        anyhow::bail!("expert model is a placeholder (too small to be ONNX)");
    }

    let mut cursor = std::io::Cursor::new(MODEL_BYTES);
    let mut model = tract_onnx::onnx().model_for_read(&mut cursor)?;
    // Fix the input to a runtime-variable batch (`N`) of FEATURE_DIM-length rows
    // so a single inference call can score a whole candidate set at once.
    let batch = model.symbols.sym("N");
    model.set_input_fact(
        0,
        f32::fact([batch.to_dim(), (FEATURE_DIM as i64).to_dim()]).into(),
    )?;
    let model = model.into_optimized()?.into_runnable()?;
    Ok(model)
}

/// Choose the best legal play for `me` using the learned Expert net, or `None`
/// if the model is unavailable / produced nothing (caller falls back to Hard).
///
/// `p` MUST be the redacted per-player view (the honesty invariant): every
/// feature is computed from observable information only.
pub fn choose_play_expert(p: &PlayPhase, me: PlayerID) -> Option<Vec<Card>> {
    let model = model()?;
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

    // Build a [N, FEATURE_DIM] batch and score it in one inference call.
    let n = candidates.len();
    let mut flat: Vec<f32> = Vec::with_capacity(n * FEATURE_DIM);
    for cand in &candidates {
        flat.extend_from_slice(&candidate_features(p, me, cand));
    }

    let scores = run_model(model, &flat, n)?;

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
    use tract_onnx::prelude::*;

    let input = tract_ndarray::Array2::from_shape_vec((n, FEATURE_DIM), flat.to_vec()).ok()?;
    let tensor: Tensor = input.into();
    let result = model.run(tvec!(tensor.into())).ok()?;
    let view = result[0].to_array_view::<f32>().ok()?;
    // The model outputs [N, 1]; flatten to N scores.
    Some(view.iter().copied().collect())
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
        .map(|c| card_strength_pub(trump, *c))
        .max()
        .unwrap_or(0);
    let min_strength = cards
        .iter()
        .map(|c| card_strength_pub(trump, *c))
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
                    .map(|c| card_strength_pub(trump, *c))
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
    let heur = if leading {
        heuristics::score_lead(p, me, cards)
    } else {
        heuristics::score_follow(p, me, cards)
    };
    f[26] = (heur as f32 / 10.0).tanh();
    f[27] = 1.0; // bias

    f
}

/// Public re-export of the heuristic's card-strength metric so this module's
/// feature encoding stays identical to what training observed. We replicate the
/// small mapping here rather than `pub`-ifying the private one to keep the
/// heuristic module's surface area unchanged.
fn card_strength_pub(trump: Trump, card: Card) -> i32 {
    match card {
        Card::BigJoker => 1000,
        Card::SmallJoker => 999,
        Card::Unknown => -1,
        Card::Suited { number, suit } => {
            if trump.number() == Some(number) {
                if Some(suit) == trump.suit() {
                    998
                } else {
                    997
                }
            } else {
                number.as_u32() as i32
            }
        }
    }
}
