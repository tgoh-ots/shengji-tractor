//! Cross-version, machine-readable Enoch control probe.
//!
//! This source deliberately uses only public APIs available at production
//! reference `c813c8a`. The freezer can inject the same file into that archived
//! tree, build it there, and compare its output with a build from the current
//! tree. Each requested seed is played in both mirrored orientations against an
//! explicit frozen-legacy greedy opponent.

use std::collections::HashSet;
use std::env;
use std::process;

use rand::rngs::StdRng;
use rand::SeedableRng;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use shengji_core::bot::harness::{play_one_hand_instrumented, PlayBrain, Seat};
use shengji_core::bot::heuristics::HeuristicVersion;
use shengji_core::bot::BotDifficulty;
use shengji_core::game_state::play_phase::PlayPhase;
use shengji_mechanics::types::{Card, EffectiveSuit, PlayerID};

const MANIFEST_VERSION: u64 = 1;
const PROBE_KIND: &str = "enoch-control-probe";
const OPPONENT_ID: &str = "legacy-greedy/easy-phases-v1";

fn usage() -> &'static str {
    "usage: enoch_control_probe --policy POLICY --seed SEED [--seed SEED ...]\n\
     POLICY is exactly one of: enoch-tier, enoch-greedy\n\
     SEED is decimal or 0x-prefixed hexadecimal. Seed order is preserved;\n\
     duplicate seed values are rejected."
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProbePolicy {
    EnochTier,
    EnochGreedy,
}

impl ProbePolicy {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "enoch-tier" => Some(Self::EnochTier),
            "enoch-greedy" => Some(Self::EnochGreedy),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::EnochTier => "enoch-tier",
            Self::EnochGreedy => "enoch-greedy",
        }
    }

    fn seat(self) -> Seat {
        match self {
            Self::EnochTier => Seat::tier(BotDifficulty::Enoch),
            Self::EnochGreedy => Seat {
                play: PlayBrain::EnochGreedy,
                bid: BotDifficulty::Enoch,
                kitty: BotDifficulty::Enoch,
            },
        }
    }
}

fn stable_opponent() -> Seat {
    Seat {
        play: PlayBrain::HeuristicDirect(HeuristicVersion::Legacy),
        // Easy phase decisions are deterministic in the shared harness and do
        // not load a model or invoke play search.
        bid: BotDifficulty::Easy,
        kitty: BotDifficulty::Easy,
    }
}

#[derive(Debug, Eq, PartialEq)]
struct Args {
    policy: ProbePolicy,
    seeds: Vec<u64>,
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

fn parse_args(raw: &[String]) -> Result<Args, String> {
    let mut policy = None;
    let mut seeds = Vec::new();
    let mut seen = HashSet::new();
    let mut index = 0usize;
    while index < raw.len() {
        let flag = &raw[index];
        index += 1;
        match flag.as_str() {
            "--policy" => {
                if policy.is_some() {
                    return Err("--policy was provided more than once".to_owned());
                }
                let value = raw
                    .get(index)
                    .ok_or_else(|| "--policy requires a value".to_owned())?;
                index += 1;
                policy = ProbePolicy::parse(value);
                if policy.is_none() {
                    return Err(format!("unsupported policy {value:?}"));
                }
            }
            "--seed" => {
                let value = raw
                    .get(index)
                    .ok_or_else(|| "--seed requires a value".to_owned())?;
                index += 1;
                let seed = parse_u64(value)
                    .ok_or_else(|| format!("invalid unsigned 64-bit seed {value:?}"))?;
                if !seen.insert(seed) {
                    return Err(format!("duplicate seed value {seed}"));
                }
                seeds.push(seed);
            }
            _ => return Err(format!("unknown argument {flag:?}")),
        }
    }
    let policy = policy.ok_or_else(|| "--policy is required".to_owned())?;
    if seeds.is_empty() {
        return Err("at least one --seed is required".to_owned());
    }
    Ok(Args { policy, seeds })
}

#[derive(Default)]
struct DecisionMetrics {
    decisions: u64,
    lead_decisions: u64,
    follow_decisions: u64,
    single_card_plays: u64,
    multi_card_plays: u64,
    cards_played: u64,
    point_cards_played: u64,
    point_value_played: u64,
    trump_cards_played: u64,
    joker_cards_played: u64,
    trace: Vec<u8>,
}

impl DecisionMetrics {
    fn observe(&mut self, state: &PlayPhase, actor: PlayerID, cards: &[Card]) {
        let leading = state.trick().played_cards().is_empty();
        self.decisions += 1;
        if leading {
            self.lead_decisions += 1;
        } else {
            self.follow_decisions += 1;
        }
        if cards.len() == 1 {
            self.single_card_plays += 1;
        } else {
            self.multi_card_plays += 1;
        }
        self.cards_played += cards.len() as u64;
        self.point_cards_played +=
            cards.iter().filter(|card| card.points().is_some()).count() as u64;
        self.point_value_played +=
            cards.iter().filter_map(|card| card.points()).sum::<usize>() as u64;
        self.trump_cards_played += cards
            .iter()
            .filter(|card| state.trump().effective_suit(**card) == EffectiveSuit::Trump)
            .count() as u64;
        self.joker_cards_played += cards.iter().filter(|card| card.is_joker()).count() as u64;

        // Length-prefixed, fixed-width trace encoding. Card order within one
        // physical multiset is canonicalized so harmless Vec ordering cannot
        // perturb the policy-equivalence digest.
        self.trace
            .extend_from_slice(&(actor.0 as u64).to_be_bytes());
        self.trace.push(u8::from(leading));
        self.trace
            .extend_from_slice(&(cards.len() as u64).to_be_bytes());
        let mut encoded_cards = cards
            .iter()
            .map(|card| card.as_char() as u32)
            .collect::<Vec<_>>();
        encoded_cards.sort_unstable();
        for card in encoded_cards {
            self.trace.extend_from_slice(&card.to_be_bytes());
        }
    }

    fn value(&self) -> Value {
        json!({
            "cards_played": self.cards_played,
            "decision_sha256": sha256_hex(&self.trace),
            "decisions": self.decisions,
            "follow_decisions": self.follow_decisions,
            "joker_cards_played": self.joker_cards_played,
            "lead_decisions": self.lead_decisions,
            "multi_card_plays": self.multi_card_plays,
            "point_cards_played": self.point_cards_played,
            "point_value_played": self.point_value_played,
            "single_card_plays": self.single_card_plays,
            "trump_cards_played": self.trump_cards_played,
        })
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn canonical_json_bytes(value: &Value) -> Result<Vec<u8>, String> {
    serde_json::to_vec(value).map_err(|error| format!("serialize canonical JSON: {error}"))
}

fn canonical_json_sha256(value: &Value) -> Result<String, String> {
    canonical_json_bytes(value).map(|bytes| sha256_hex(&bytes))
}

fn orientation(
    seed: u64,
    focal: Seat,
    focal_is_landlord_team: bool,
) -> (Value, bool, DecisionMetrics) {
    let opponent = stable_opponent();
    let seats = if focal_is_landlord_team {
        [focal, opponent, focal, opponent]
    } else {
        [opponent, focal, opponent, focal]
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let mut metrics = DecisionMetrics::default();
    let result = play_one_hand_instrumented(&seats, &mut rng, &mut |state, actor, cards| {
        let actor_is_landlord_team = state.landlords_team().contains(&actor);
        if actor_is_landlord_team == focal_is_landlord_team {
            metrics.observe(state, actor, cards);
        }
    });
    let value = if let Some(result) = result {
        let (focal_won, focal_point_margin) = result.subject_outcome(focal_is_landlord_team);
        json!({
            "complete": true,
            "focal_is_landlord_team": focal_is_landlord_team,
            "focal_level_utility": result.subject_level_utility(focal_is_landlord_team),
            "focal_point_margin": focal_point_margin,
            "focal_won": focal_won,
            "landlord_level_delta": result.landlord_level_delta,
            "landlord_won": result.landlord_won,
            "non_landlord_level_delta": result.non_landlord_level_delta,
            "non_landlord_points": result.non_landlord_points,
            "style": metrics.value(),
        })
    } else {
        json!({
            "complete": false,
            "focal_is_landlord_team": focal_is_landlord_team,
            "focal_level_utility": Value::Null,
            "focal_point_margin": Value::Null,
            "focal_won": Value::Null,
            "landlord_level_delta": Value::Null,
            "landlord_won": Value::Null,
            "non_landlord_level_delta": Value::Null,
            "non_landlord_points": Value::Null,
            "style": metrics.value(),
        })
    };
    (value, result.is_some(), metrics)
}

fn signed_field(value: &Value, name: &str) -> Option<i64> {
    value.get(name).and_then(Value::as_i64)
}

fn bool_field(value: &Value, name: &str) -> Option<bool> {
    value.get(name).and_then(Value::as_bool)
}

fn run_probe(args: &Args) -> Result<(Value, bool), String> {
    let focal = args.policy.seat();
    let mut pairs = Vec::with_capacity(args.seeds.len());
    let mut complete_pairs = 0u64;
    let mut completed_hands = 0u64;
    let mut focal_wins = 0u64;
    let mut focal_decisions = 0u64;

    for (request_index, &seed) in args.seeds.iter().enumerate() {
        let (as_landlord, landlord_complete, landlord_metrics) = orientation(seed, focal, true);
        let (as_attacker, attacker_complete, attacker_metrics) = orientation(seed, focal, false);
        completed_hands += u64::from(landlord_complete) + u64::from(attacker_complete);
        focal_decisions += landlord_metrics.decisions + attacker_metrics.decisions;

        let pair_complete = landlord_complete && attacker_complete;
        let (pair_wins, point_margin_sum, level_utility_sum) = if pair_complete {
            let wins = u64::from(bool_field(&as_landlord, "focal_won").unwrap_or(false))
                + u64::from(bool_field(&as_attacker, "focal_won").unwrap_or(false));
            let margins = signed_field(&as_landlord, "focal_point_margin")
                .and_then(|first| {
                    signed_field(&as_attacker, "focal_point_margin").map(|second| first + second)
                })
                .ok_or_else(|| "completed pair lacked point margins".to_owned())?;
            let levels = signed_field(&as_landlord, "focal_level_utility")
                .and_then(|first| {
                    signed_field(&as_attacker, "focal_level_utility").map(|second| first + second)
                })
                .ok_or_else(|| "completed pair lacked level utilities".to_owned())?;
            complete_pairs += 1;
            focal_wins += wins;
            (json!(wins), json!(margins), json!(levels))
        } else {
            (Value::Null, Value::Null, Value::Null)
        };

        pairs.push(json!({
            "complete": pair_complete,
            "focal_as_attacker": as_attacker,
            "focal_as_landlord": as_landlord,
            "focal_level_utility_sum": level_utility_sum,
            "focal_point_margin_sum": point_margin_sum,
            "focal_wins": pair_wins,
            "request_index": request_index,
            "seed": seed,
        }));
    }

    let pairs_requested = args.seeds.len() as u64;
    let incomplete_pairs = pairs_requested.saturating_sub(complete_pairs);
    let frozen_policy = json!({
        "kind": PROBE_KIND,
        "manifest_version": MANIFEST_VERSION,
        "opponent": OPPONENT_ID,
        "pairs": pairs,
        "policy": args.policy.name(),
        "seed_count": pairs_requested,
        "seeds": args.seeds,
        "summary": {
            "complete_pairs": complete_pairs,
            "completed_hands": completed_hands,
            "focal_decisions": focal_decisions,
            "focal_wins": focal_wins,
            "incomplete_pairs": incomplete_pairs,
            "pairs_requested": pairs_requested,
        },
    });
    let digest = canonical_json_sha256(&frozen_policy)?;
    let output = json!({
        "equivalence_sha256": digest,
        "frozen_policy": frozen_policy,
    });
    Ok((output, incomplete_pairs == 0))
}

fn real_main() -> Result<bool, String> {
    let raw = env::args().skip(1).collect::<Vec<_>>();
    if raw.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("{}", usage());
        return Ok(true);
    }
    let args = parse_args(&raw)?;
    let (output, complete) = run_probe(&args)?;
    let bytes = canonical_json_bytes(&output)?;
    println!(
        "{}",
        String::from_utf8(bytes).map_err(|error| format!("JSON was not UTF-8: {error}"))?
    );
    Ok(complete)
}

fn main() {
    match real_main() {
        Ok(true) => {}
        Ok(false) => process::exit(2),
        Err(error) => {
            eprintln!("error: {error}\n{}", usage());
            process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn parser_preserves_exact_noncontiguous_seed_order() {
        let parsed = parse_args(&strings(&[
            "--seed",
            "0x2a",
            "--policy",
            "enoch-greedy",
            "--seed",
            "7",
            "--seed",
            "18446744073709551615",
        ]))
        .unwrap();
        assert_eq!(parsed.policy, ProbePolicy::EnochGreedy);
        assert_eq!(parsed.seeds, vec![42, 7, u64::MAX]);
    }

    #[test]
    fn parser_rejects_duplicate_or_implicit_seeds() {
        assert!(parse_args(&strings(&[
            "--policy",
            "enoch-tier",
            "--seed",
            "7",
            "--seed",
            "0x7",
        ]))
        .unwrap_err()
        .contains("duplicate seed"));
        assert!(parse_args(&strings(&["--policy", "enoch-tier"]))
            .unwrap_err()
            .contains("at least one --seed"));
        assert!(parse_args(&strings(&["--policy", "other", "--seed", "1"]))
            .unwrap_err()
            .contains("unsupported policy"));
    }

    #[test]
    fn canonical_digest_is_key_order_independent_and_known() {
        let mut first = serde_json::Map::new();
        first.insert("b".to_owned(), json!(2));
        first.insert("a".to_owned(), json!(1));
        let mut second = serde_json::Map::new();
        second.insert("a".to_owned(), json!(1));
        second.insert("b".to_owned(), json!(2));
        let first = Value::Object(first);
        let second = Value::Object(second);
        assert_eq!(canonical_json_bytes(&first).unwrap(), br#"{"a":1,"b":2}"#);
        assert_eq!(
            canonical_json_sha256(&first).unwrap(),
            "43258cff783fe7036d8a43033f830adfc60ec037382473548ac742b888292777"
        );
        assert_eq!(
            canonical_json_sha256(&first).unwrap(),
            canonical_json_sha256(&second).unwrap()
        );
    }

    #[test]
    fn digest_commits_to_seed_sequence_without_runtime_metadata() {
        let first = json!({"policy": "enoch-greedy", "seeds": [9, 3]});
        let second = json!({"policy": "enoch-greedy", "seeds": [3, 9]});
        assert_ne!(
            canonical_json_sha256(&first).unwrap(),
            canonical_json_sha256(&second).unwrap()
        );
        assert!(first.get("elapsed_ms").is_none());
        assert!(first.get("wall_time").is_none());
    }

    #[test]
    fn known_hash_order_regression_seed_is_repeatable() {
        let args = Args {
            policy: ProbePolicy::EnochGreedy,
            seeds: vec![0x58b7_6b70_3ff0_e148],
        };
        let (first, complete) = run_probe(&args).unwrap();
        assert!(complete);
        let first = canonical_json_bytes(&first).unwrap();

        // This seed previously exposed randomized HashMap iteration in bidding
        // and Enoch's whole-suit throw candidates. Rebuild and replay the full
        // mirrored pair enough times to exercise fresh map states; compare bytes
        // rather than a historical digest so the test binds determinism, not one
        // accidental member of the old nondeterministic output set.
        for _ in 0..16 {
            let (repeated, complete) = run_probe(&args).unwrap();
            assert!(complete);
            assert_eq!(canonical_json_bytes(&repeated).unwrap(), first);
        }
    }
}
