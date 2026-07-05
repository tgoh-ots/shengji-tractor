//! Explicit, per-contestant controls for the Enoch-1 ablation program.
//!
//! These controls deliberately live in [`crate::bot::search::SearchConfig`]
//! instead of process-global environment variables. A Week-1 comparison plays
//! the candidate and frozen control in the same process and often in the same
//! hand; a global flag would therefore contaminate both arms.

use std::fmt;

/// A compact set of independently selectable Enoch-1 hypotheses.
///
/// The bit assignments are part of experiment manifests. Append new features;
/// never reuse an existing bit for a different hypothesis.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct EnochFeatures(u64);

impl EnochFeatures {
    pub const BID_OWNERSHIP: Self = Self(1 << 0);
    pub const COMPOUND_FOLLOW: Self = Self(1 << 1);
    pub const FAILED_THROW_WITNESS: Self = Self(1 << 2);
    pub const FRIEND_REVELATION: Self = Self(1 << 3);
    pub const TERMINAL_LEVEL: Self = Self(1 << 4);
    pub const DEFAULT_KITTY: Self = Self(1 << 5);
    pub const RUFF_SHAPE: Self = Self(1 << 6);
    pub const CONTEXTUAL_EMPTY_TRICK: Self = Self(1 << 7);
    pub const LIVE_SUIT_CONTROL: Self = Self(1 << 8);
    pub const TEAM_VOID: Self = Self(1 << 9);
    pub const ENTRY_RETURN: Self = Self(1 << 10);
    pub const HANDOFF_PROTECTION: Self = Self(1 << 11);
    pub const STRUCTURAL_FAMILIES: Self = Self(1 << 12);
    pub const PROGRESSIVE_ADMISSION: Self = Self(1 << 13);
    pub const UNCERTAIN_THROWS: Self = Self(1 << 14);

    pub const ALL: Self = Self((1 << 15) - 1);
    /// Features that change the admissible hidden-world distribution. Search
    /// must route every one of these through the feature-aware determinizer;
    /// otherwise an apparently enabled ablation would silently sample the
    /// Enoch-0 posterior and have no effect.
    pub const HARD_EVIDENCE: Self = Self(
        Self::BID_OWNERSHIP.0
            | Self::COMPOUND_FOLLOW.0
            | Self::FAILED_THROW_WITNESS.0
            | Self::FRIEND_REVELATION.0,
    );

    pub const fn empty() -> Self {
        Self(0)
    }

    pub const fn bits(self) -> u64 {
        self.0
    }

    pub const fn contains(self, feature: Self) -> bool {
        self.0 & feature.0 == feature.0
    }

    pub const fn intersects(self, features: Self) -> bool {
        self.0 & features.0 != 0
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn without(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Parse the stable comma-separated names used by Week-1 protocols.
    pub fn parse(spec: &str) -> Result<Self, String> {
        let mut result = Self::empty();
        for raw in spec.split(',') {
            let name = raw.trim();
            if name.is_empty() || name == "none" {
                continue;
            }
            let feature =
                Self::named(name).ok_or_else(|| format!("unknown Enoch feature {name:?}"))?;
            result = result.union(feature);
        }
        Ok(result)
    }

    pub fn names(self) -> Vec<&'static str> {
        Self::DEFINITIONS
            .iter()
            .filter_map(|(name, feature)| self.contains(*feature).then_some(*name))
            .collect()
    }

    fn named(name: &str) -> Option<Self> {
        if name == "all" {
            return Some(Self::ALL);
        }
        Self::DEFINITIONS
            .iter()
            .find_map(|(candidate, feature)| (*candidate == name).then_some(*feature))
    }

    const DEFINITIONS: [(&'static str, Self); 15] = [
        ("bid-ownership", Self::BID_OWNERSHIP),
        ("compound-follow", Self::COMPOUND_FOLLOW),
        ("failed-throw-witness", Self::FAILED_THROW_WITNESS),
        ("friend-revelation", Self::FRIEND_REVELATION),
        ("terminal-level", Self::TERMINAL_LEVEL),
        ("default-kitty", Self::DEFAULT_KITTY),
        ("ruff-shape", Self::RUFF_SHAPE),
        ("contextual-empty-trick", Self::CONTEXTUAL_EMPTY_TRICK),
        ("live-suit-control", Self::LIVE_SUIT_CONTROL),
        ("team-void", Self::TEAM_VOID),
        ("entry-return", Self::ENTRY_RETURN),
        ("handoff-protection", Self::HANDOFF_PROTECTION),
        ("structural-families", Self::STRUCTURAL_FAMILIES),
        ("progressive-admission", Self::PROGRESSIVE_ADMISSION),
        ("uncertain-throws", Self::UNCERTAIN_THROWS),
    ];
}

impl fmt::Display for EnochFeatures {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names = self.names();
        if names.is_empty() {
            formatter.write_str("none")
        } else {
            formatter.write_str(&names.join(","))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::EnochFeatures;

    #[test]
    fn manifest_names_round_trip_without_reassigning_bits() {
        let parsed =
            EnochFeatures::parse("bid-ownership,failed-throw-witness,progressive-admission")
                .unwrap();
        assert_eq!(
            parsed.names(),
            vec![
                "bid-ownership",
                "failed-throw-witness",
                "progressive-admission"
            ]
        );
        assert_eq!(parsed.bits(), (1 << 0) | (1 << 2) | (1 << 13));
        assert_eq!(EnochFeatures::parse(&parsed.to_string()).unwrap(), parsed);
    }

    #[test]
    fn unknown_feature_fails_closed() {
        assert!(EnochFeatures::parse("bid-ownership,typo").is_err());
    }

    #[test]
    fn every_manifest_feature_is_one_isolated_stable_bit() {
        let mut union = EnochFeatures::empty();
        for (index, name) in EnochFeatures::ALL.names().into_iter().enumerate() {
            let feature = EnochFeatures::parse(name).unwrap();
            assert_eq!(feature.bits().count_ones(), 1, "{name} is not isolated");
            assert_eq!(feature.bits(), 1u64 << index, "{name} bit moved");
            assert!(
                !union.intersects(feature),
                "{} aliases an earlier arm",
                name
            );
            union = union.union(feature);
        }
        assert_eq!(union, EnochFeatures::ALL);
        assert_eq!(
            EnochFeatures::HARD_EVIDENCE,
            EnochFeatures::BID_OWNERSHIP
                .union(EnochFeatures::COMPOUND_FOLLOW)
                .union(EnochFeatures::FAILED_THROW_WITNESS)
                .union(EnochFeatures::FRIEND_REVELATION)
        );
    }
}
