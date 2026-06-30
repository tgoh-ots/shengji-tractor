//! Machine-readable matched-deal Expert-vs-Easy control arm.
//! Run embedded and candidate models in separate processes, then subtract the
//! per-deck outcomes by index to obtain a direct model-effect estimate.

use shengji_core::bot::harness::{run_paired_ab, Contestant};
use shengji_core::bot::BotDifficulty;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let pairs = args
        .get(1)
        .map(|value| {
            value
                .parse::<usize>()
                .ok()
                .filter(|pairs| *pairs > 0)
                .unwrap_or_else(|| panic!("pairs must be a positive integer, got {:?}", value))
        })
        .unwrap_or(200);
    let seed = args
        .get(2)
        .map(|value| {
            parse_u64(value).unwrap_or_else(|| {
                panic!(
                    "seed must be decimal or 0x-prefixed hexadecimal, got {:?}",
                    value
                )
            })
        })
        .unwrap_or(0x5EED);
    let result = run_paired_ab(
        &Contestant::tier(BotDifficulty::Expert),
        &Contestant::tier(BotDifficulty::Easy),
        pairs,
        seed,
    );
    println!(
        "{}",
        serde_json::json!({
            "manifest_version": 1,
            "pairs_requested": pairs,
            "seed": seed,
            "complete_pairs": result.complete_pairs,
            "failed_hands": result.failed_hands(),
            "per_deck_winrate": result.per_deck_winrate,
            "per_deck_margin": result.per_deck_margin,
            "per_deck_level_utility": result.per_deck_level_utility,
        })
    );
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

#[cfg(test)]
mod tests {
    use super::parse_u64;

    #[test]
    fn parses_decimal_and_hex_seeds_without_silent_fallback() {
        assert_eq!(parse_u64("24301"), Some(0x5eed));
        assert_eq!(parse_u64("0x5EED"), Some(0x5eed));
        assert_eq!(parse_u64("0X5eed"), Some(0x5eed));
        assert_eq!(parse_u64("not-a-seed"), None);
    }
}
