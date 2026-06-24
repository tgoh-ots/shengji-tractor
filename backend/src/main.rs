#![deny(warnings)]

//! Thin binary shim over the `shengji` library. All of the app's wiring lives in
//! `lib.rs` so that integration tests (`tests/e2e_game.rs`) can boot the same
//! real Axum app and `shengji_handler` over an actual WebSocket.

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    shengji::run().await
}
