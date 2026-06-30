//! Security-hardening helpers for Milestone 4.
//!
//! This module centralizes the configuration and primitives used to protect the
//! WebSocket / HTTP surface of the server when it is exposed to an untrusted
//! network (e.g. a free hosting tier):
//!
//! - Origin allow-listing for the WebSocket handshake (anti CSWSH).
//! - Inbound message size and rate limiting.
//! - Resource caps (players/observers per room, total rooms, room-creation rate).
//! - Chat / name sanitization (control-char stripping + length limits).
//!
//! Tunables can be overridden via environment variables so the defaults stay
//! conservative without requiring a redeploy to adjust.

use std::time::Instant;

/// Maximum size (in bytes) of a single inbound WebSocket message we are willing
/// to buffer/parse. A legal move is at most a few hundred bytes; 16 KiB leaves
/// generous headroom for the largest settings payloads while preventing memory
/// exhaustion from a single frame.
pub const MAX_INBOUND_MESSAGE_BYTES: usize = 16 * 1024;

/// Maximum length (in bytes) of a chat message after trimming.
pub const MAX_CHAT_MESSAGE_BYTES: usize = 2048;

/// Maximum length (in bytes) of a player display name. Matches the pre-existing
/// `name.len() < 32` check in the join path.
pub const MAX_NAME_BYTES: usize = 32;

/// Exact required length (in bytes) of a room name. Matches the pre-existing
/// `room_name.len() == 16` check in the join path.
pub const ROOM_NAME_BYTES: usize = 16;

/// Read an environment-variable-overridable `usize` tunable.
fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

/// Read an environment-variable-overridable `f64` tunable.
fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .unwrap_or(default)
}

/// Resource caps and rate-limit tunables. Values are conservative defaults
/// suitable for a small free-host deployment and can be raised via env vars.
#[derive(Clone, Debug)]
pub struct ResourceLimits {
    /// Maximum number of seated players allowed in a single room.
    pub max_players_per_room: usize,
    /// Maximum number of observers allowed in a single room.
    pub max_observers_per_room: usize,
    /// Maximum number of concurrent rooms (games) held in storage.
    pub max_total_rooms: usize,
    /// Inbound message rate limit: sustained messages per second.
    pub msg_rate_per_sec: f64,
    /// Inbound message rate limit: burst capacity (bucket size).
    pub msg_burst: f64,
    /// Maximum number of CPU-heavy bot planners allowed to run concurrently
    /// across the process.  Search is deliberately wall-clock bounded, so
    /// oversubscribing a small VM both weakens every bot and can starve request
    /// handling.  Keep this at one on the production single-vCPU machine.
    pub max_parallel_bot_searches: usize,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            // Tractor supports up to ~6 seated players with multiple decks; 16
            // is a generous ceiling that still bounds per-room memory and keeps
            // the bot self-play setup (host + 4 bots) well within range.
            max_players_per_room: env_usize("MAX_PLAYERS_PER_ROOM", 16),
            max_observers_per_room: env_usize("MAX_OBSERVERS_PER_ROOM", 16),
            max_total_rooms: env_usize("MAX_TOTAL_ROOMS", 1000),
            msg_rate_per_sec: env_f64("WS_MSG_RATE_PER_SEC", 40.0),
            msg_burst: env_f64("WS_MSG_BURST", 60.0),
            max_parallel_bot_searches: env_usize("MAX_PARALLEL_BOT_SEARCHES", 1).max(1),
        }
    }
}

impl ResourceLimits {
    pub fn from_env() -> Self {
        Self::default()
    }
}

/// A simple token-bucket rate limiter used to throttle inbound WebSocket
/// messages on a per-connection basis. Not shared across connections, so no
/// locking is required.
#[derive(Debug)]
pub struct TokenBucket {
    capacity: f64,
    tokens: f64,
    refill_per_sec: f64,
    last: Instant,
}

impl TokenBucket {
    pub fn new(capacity: f64, refill_per_sec: f64) -> Self {
        Self {
            capacity,
            tokens: capacity,
            refill_per_sec,
            last: Instant::now(),
        }
    }

    /// Attempt to consume a single token. Returns `true` if the message is
    /// allowed, `false` if the connection is over its rate budget.
    pub fn try_acquire(&mut self) -> bool {
        self.try_acquire_at(Instant::now())
    }

    fn try_acquire_at(&mut self, now: Instant) -> bool {
        let elapsed = now.saturating_duration_since(self.last).as_secs_f64();
        if elapsed > 0.0 {
            self.tokens = (self.tokens + elapsed * self.refill_per_sec).min(self.capacity);
            self.last = now;
        }
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    /// Construct a bucket whose internal clock is pinned to `last`, for
    /// deterministic testing (avoids racing against the real wall clock).
    #[cfg(test)]
    fn new_at(capacity: f64, refill_per_sec: f64, last: Instant) -> Self {
        Self {
            capacity,
            tokens: capacity,
            refill_per_sec,
            last,
        }
    }
}

/// Parse the `CORS_ALLOWED_ORIGINS` env var into an origin allow-list policy.
///
/// The same variable governs both the CORS layer (for `/api/rpc`) and the
/// WebSocket handshake Origin check, so the two stay in sync.
#[derive(Clone, Debug)]
pub enum OriginPolicy {
    /// Allow any origin (configured via `*`). Logged as a warning at startup.
    AllowAny,
    /// Allow only the listed origins (compared case-insensitively, exact match).
    Allowlist(Vec<String>),
}

/// The default development origin allow-list, used when `CORS_ALLOWED_ORIGINS`
/// is unset. Kept identical to the CORS default.
pub const DEFAULT_ALLOWED_ORIGINS: &str =
    "http://localhost:3000,http://localhost:3030,http://127.0.0.1:3000,http://127.0.0.1:3030";

impl OriginPolicy {
    /// Build the policy from the raw `CORS_ALLOWED_ORIGINS` value (or its
    /// default when unset).
    pub fn from_raw(raw: &str) -> Self {
        if raw.trim() == "*" {
            OriginPolicy::AllowAny
        } else {
            let origins = raw
                .split(',')
                .map(|o| o.trim().to_ascii_lowercase())
                .filter(|o| !o.is_empty())
                .collect::<Vec<_>>();
            OriginPolicy::Allowlist(origins)
        }
    }

    /// Decide whether a request bearing the given `Origin` header value (if any)
    /// is allowed to establish a WebSocket connection.
    ///
    /// A missing Origin header is allowed: native (non-browser) clients do not
    /// send one, and the Origin check exists specifically to defend against
    /// cross-site requests *from browsers*, which always send Origin on WS
    /// upgrades. Rejecting missing-Origin would break legitimate non-browser
    /// clients without adding browser protection.
    pub fn is_ws_origin_allowed(&self, origin: Option<&str>) -> bool {
        match (self, origin) {
            (OriginPolicy::AllowAny, _) => true,
            (OriginPolicy::Allowlist(_), None) => true,
            (OriginPolicy::Allowlist(list), Some(origin)) => {
                let origin = origin.trim().to_ascii_lowercase();
                list.contains(&origin)
            }
        }
    }
}

/// Strip ASCII/Unicode control characters (other than ordinary spaces) from a
/// user-supplied string and trim surrounding whitespace. This neutralizes
/// terminal-escape / newline-injection / null-byte tricks before the value is
/// stored or broadcast. The frontend additionally renders everything as text.
pub fn sanitize_text(input: &str) -> String {
    input
        .chars()
        // Drop C0/C1 control characters (including NUL, CR, LF, ESC) but keep
        // normal printable content and regular spaces.
        .filter(|c| !c.is_control())
        .collect::<String>()
        .trim()
        .to_string()
}

/// Validate and sanitize a chat message. Returns `None` if, after stripping
/// control characters and trimming, the message is empty or exceeds the length
/// cap (oversized messages are rejected rather than silently truncated).
pub fn sanitize_chat_message(input: &str) -> Option<String> {
    let cleaned = sanitize_text(input);
    if cleaned.is_empty() || cleaned.len() > MAX_CHAT_MESSAGE_BYTES {
        None
    } else {
        Some(cleaned)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn origin_allowlist_exact_match() {
        let policy = OriginPolicy::from_raw("https://example.com,http://localhost:3000");
        assert!(policy.is_ws_origin_allowed(Some("https://example.com")));
        // Case-insensitive scheme/host comparison.
        assert!(policy.is_ws_origin_allowed(Some("HTTPS://EXAMPLE.COM")));
        assert!(policy.is_ws_origin_allowed(Some("http://localhost:3000")));
    }

    #[test]
    fn origin_allowlist_rejects_unknown() {
        let policy = OriginPolicy::from_raw("https://example.com");
        assert!(!policy.is_ws_origin_allowed(Some("https://evil.example")));
        assert!(!policy.is_ws_origin_allowed(Some("http://example.com"))); // wrong scheme
        assert!(!policy.is_ws_origin_allowed(Some("https://example.com.evil.test")));
    }

    #[test]
    fn origin_allowlist_allows_missing_origin() {
        // Non-browser clients (which don't send Origin) are permitted.
        let policy = OriginPolicy::from_raw("https://example.com");
        assert!(policy.is_ws_origin_allowed(None));
    }

    #[test]
    fn origin_allow_any_allows_everything() {
        let policy = OriginPolicy::from_raw("*");
        assert!(matches!(policy, OriginPolicy::AllowAny));
        assert!(policy.is_ws_origin_allowed(Some("https://evil.example")));
        assert!(policy.is_ws_origin_allowed(None));
    }

    #[test]
    fn default_origins_cover_localhost() {
        let policy = OriginPolicy::from_raw(DEFAULT_ALLOWED_ORIGINS);
        assert!(policy.is_ws_origin_allowed(Some("http://localhost:3000")));
        assert!(policy.is_ws_origin_allowed(Some("http://127.0.0.1:3030")));
        assert!(!policy.is_ws_origin_allowed(Some("http://localhost:9999")));
    }

    #[test]
    fn sanitize_strips_control_chars() {
        assert_eq!(sanitize_text("hel\u{0}lo\n"), "hello");
        assert_eq!(sanitize_text("  spaced  "), "spaced");
        assert_eq!(sanitize_text("a\u{1b}[31mred"), "a[31mred");
        // Ordinary internal spaces are preserved.
        assert_eq!(sanitize_text("hello world"), "hello world");
    }

    #[test]
    fn sanitize_chat_rejects_empty_and_oversized() {
        assert_eq!(sanitize_chat_message("   \n\t  "), None);
        assert_eq!(sanitize_chat_message("hi"), Some("hi".to_string()));
        let big = "a".repeat(MAX_CHAT_MESSAGE_BYTES + 1);
        assert_eq!(sanitize_chat_message(&big), None);
        let ok = "a".repeat(MAX_CHAT_MESSAGE_BYTES);
        assert_eq!(sanitize_chat_message(&ok), Some(ok));
    }

    #[test]
    fn token_bucket_enforces_rate() {
        let start = Instant::now();
        let mut bucket = TokenBucket::new_at(3.0, 1.0, start);
        // Burst of 3 allowed.
        assert!(bucket.try_acquire_at(start));
        assert!(bucket.try_acquire_at(start));
        assert!(bucket.try_acquire_at(start));
        // 4th in the same instant is denied.
        assert!(!bucket.try_acquire_at(start));
        // After 1 second, one token refills.
        assert!(bucket.try_acquire_at(start + Duration::from_secs(1)));
        assert!(!bucket.try_acquire_at(start + Duration::from_secs(1)));
    }
}
