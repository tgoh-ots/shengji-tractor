use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use shengji_core::{game_state, interactive};
use shengji_mechanics::types::Card;

pub mod wasm_rpc;

#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub enum GameMessage {
    State {
        state: game_state::GameState,
    },
    Message {
        from: String,
        message: String,
    },
    Broadcast {
        data: interactive::BroadcastMessage,
        message: String,
    },
    Beep {
        target: String,
    },
    ReadyCheck {
        from: String,
    },
    Error(String),
    Header {
        messages: Vec<String>,
    },
    Kicked {
        target: String,
    },
    /// The answer to a `UserMessage::RequestPlaySuggestion`: the cards the
    /// Grandmaster policy would play in the requesting player's seat,
    /// computed from THAT PLAYER'S OWN redacted view (see
    /// `shengji_core::bot::advice`). Published only to the requesting socket, and
    /// it only ever contains cards that socket's player already holds.
    ///
    /// `cards` is empty when no advice is available — it isn't their turn, the
    /// game isn't in the play phase, or no legal suggestion could be produced —
    /// so the client always gets a reply to clear its pending state.
    PlaySuggestion {
        cards: Vec<Card>,
    },
}

/// zstd dictionary, compressed with zstd.
pub const ZSTD_ZSTD_DICT: &[u8] = include_bytes!("../dict.zstd");
