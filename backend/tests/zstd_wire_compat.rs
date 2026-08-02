//! The server-to-client compression contract.
//!
//! Live game state is pushed over the WebSocket zstd-compressed with a trained
//! dictionary (`shengji_handler::send_to_user_with_compression` via
//! `ZSTD_COMPRESSOR`), and the browser decompresses it in
//! `frontend/shengji-wasm` — which does NOT use the same library. The server
//! compresses with the `zstd` crate (bindings to libzstd); the client decodes
//! with `ruzstd`, an independent pure-Rust implementation.
//!
//! That is a cross-implementation wire contract with no other coverage: the
//! end-to-end tests deliberately join with `disable_compression: true`, so they
//! exercise the plain-JSON path only. If a `zstd` bump ever emitted frames
//! `ruzstd` cannot decode, every client would fail to read game state and no
//! existing test would notice — it would surface as a total outage.
//!
//! This test pins that contract by compressing exactly the way the server does
//! and decoding exactly the way the client does.

use std::io::Read;

use ruzstd::decoding::dictionary::Dictionary;
use ruzstd::frame_decoder::FrameDecoder;
use ruzstd::streaming_decoder::StreamingDecoder;

use shengji_types::ZSTD_ZSTD_DICT;

/// Decode the embedded dictionary the same way both sides do.
fn dictionary_bytes() -> Vec<u8> {
    let mut reader = ZSTD_ZSTD_DICT;
    let mut decoder =
        StreamingDecoder::new(&mut reader).expect("dictionary must be a valid zstd frame");
    let mut dict = Vec::new();
    decoder
        .read_to_end(&mut dict)
        .expect("embedded dictionary must decode");
    dict
}

/// Compress `payload` exactly as `ZSTD_COMPRESSOR` does in `backend/src/lib.rs`.
fn compress_like_server(payload: &[u8]) -> Vec<u8> {
    let dict = dictionary_bytes();
    let mut compressor =
        zstd::bulk::Compressor::with_dictionary(0, &dict).expect("compressor must build");
    compressor.compress(payload).expect("compression must work")
}

/// Decompress exactly as `frontend/shengji-wasm` does.
fn decompress_like_client(frame: &[u8]) -> Vec<u8> {
    let dict = dictionary_bytes();
    let mut decoder = FrameDecoder::new();
    decoder
        .add_dict(Dictionary::decode_dict(&dict).expect("dictionary must parse"))
        .expect("dictionary must attach");
    let mut reader = frame;
    let mut streaming =
        StreamingDecoder::new_with_decoder(&mut reader, decoder).expect("frame must open");
    let mut out = Vec::new();
    streaming
        .read_to_end(&mut out)
        .expect("client must be able to decode what the server produced");
    out
}

#[test]
fn server_compressed_frames_are_decodable_by_the_client_decoder() {
    // A spread of payload shapes: tiny, dictionary-friendly game JSON (the real
    // traffic), something incompressible, and something large.
    let game_like = serde_json::json!({
        "State": {
            "Play": {
                "trump": { "Standard": { "suit": "♡", "number": "2" } },
                "landlord": 0,
                "hands": { "hands": { "0": { "🂡": 1, "🂮": 2 } } },
                "points": { "0": ["🃍"], "1": [] },
            }
        }
    })
    .to_string();

    let payloads: Vec<Vec<u8>> = vec![
        b"{}".to_vec(),
        game_like.clone().into_bytes(),
        game_like.repeat(64).into_bytes(),
        (0u8..=255).cycle().take(200_000).collect(),
    ];

    for (i, payload) in payloads.iter().enumerate() {
        let frame = compress_like_server(payload);
        let round_tripped = decompress_like_client(&frame);
        assert_eq!(
            round_tripped, *payload,
            "payload {i} did not survive the server->client compression contract"
        );
    }
}

#[test]
fn the_embedded_dictionary_is_usable_by_both_implementations() {
    let dict = dictionary_bytes();
    assert!(!dict.is_empty(), "dictionary decoded to nothing");
    // Both sides must accept it: the server as a compression dictionary, the
    // client as a decoding dictionary.
    zstd::bulk::Compressor::with_dictionary(0, &dict).expect("server side must accept the dict");
    Dictionary::decode_dict(&dict).expect("client side must accept the dict");
}
