#![deny(warnings)]

//! Library surface of the `shengji` backend.
//!
//! Historically all of this lived in `main.rs` (a binary-only crate). It was
//! lifted into a library so that integration tests in `tests/` (notably the
//! end-to-end WebSocket game in `tests/e2e_game.rs`) can boot the *real* Axum
//! app and the *real* `shengji_handler` over an actual socket, rather than only
//! testing through the in-process `axum-test` harness (which cannot drive
//! WebSockets in this version). `main.rs` is now a thin shim over [`run`].

use std::net::SocketAddr;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use axum::{
    extract::ws::{Message, WebSocketUpgrade},
    http::StatusCode,
    response::{IntoResponse, Redirect},
    routing::{get, post},
    Extension, Json, Router,
};
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use slog::{debug, error, info, o, warn, Drain, Logger};
use tokio::sync::{mpsc, Mutex};

#[cfg(feature = "dynamic")]
use axum::routing::get_service;
#[cfg(not(feature = "dynamic"))]
use axum::{
    body::{Empty, Full},
    extract::Path,
    response::Response,
};
use http::{header, HeaderName, HeaderValue, Method};
use tower_http::cors::{AllowOrigin, CorsLayer};
#[cfg(feature = "dynamic")]
use tower_http::services::ServeDir;
use tower_http::set_header::SetResponseHeaderLayer;

use shengji_core::settings;
use shengji_mechanics::types::FULL_DECK;
use shengji_types::ZSTD_ZSTD_DICT;
use storage::{HashMapStorage, Storage};

pub mod security;
pub mod serving_types;
pub mod shengji_handler;
pub mod state_dump;
pub mod utils;
pub mod wasm_rpc_handler;

use security::{OriginPolicy, ResourceLimits, DEFAULT_ALLOWED_ORIGINS, MAX_INBOUND_MESSAGE_BYTES};

use serving_types::{CardsBlob, VersionedGame};
use state_dump::InMemoryStats;

/// Our global unique user id counter.
static NEXT_USER_ID: AtomicUsize = AtomicUsize::new(1);

lazy_static::lazy_static! {
    static ref CARDS_JSON: CardsBlob = CardsBlob {
        cards: FULL_DECK.iter().map(|c| c.as_info()).collect()
    };

    pub static ref ROOT_LOGGER: Logger = {
        #[cfg(not(feature = "dynamic"))]
        let drain = slog_bunyan::default(std::io::stdout());
        #[cfg(feature = "dynamic")]
        let drain = slog_term::FullFormat::new(slog_term::TermDecorator::new().build()).build();

        let version = std::env::var("VERSION").unwrap_or_else(|_| "unknown_dev".to_string());

        Logger::root(
            slog_async::Async::new(drain.fuse()).build().fuse(),
            o!("version" => version)
        )
    };

    pub(crate) static ref ZSTD_COMPRESSOR: std::sync::Mutex<zstd::bulk::Compressor<'static>> = {
        // default zstd dictionary size is 112_640
        let comp = zstd::bulk::Compressor::with_dictionary(0, &zstd::bulk::decompress(ZSTD_ZSTD_DICT, 112_640).unwrap()).unwrap();
        std::sync::Mutex::new(comp)
    };

    static ref VERSION: String = {
        std::env::var("VERSION").unwrap_or_else(|_| "unknown_dev".to_string())
    };

    pub(crate) static ref DUMP_PATH: String = {
        std::env::var("DUMP_PATH").unwrap_or_else(|_| "/tmp/shengji_state.json".to_string())
    };
    pub(crate) static ref MESSAGE_PATH: String = {
        std::env::var("MESSAGE_PATH").unwrap_or_else(|_| "/tmp/shengji_messages.json".to_string())
    };
    static ref WEBSOCKET_HOST: Option<String> = {
        std::env::var("WEBSOCKET_HOST").ok()
    };
}

async fn runtime_settings() -> impl IntoResponse {
    let body = match WEBSOCKET_HOST.as_ref() {
        Some(s) => format!(
            "window._WEBSOCKET_HOST = \"{}\";window._VERSION = \"{}\";",
            s, *VERSION,
        ),
        None => format!(
            "window._WEBSOCKET_HOST = null;window._VERSION = \"{}\";",
            *VERSION
        ),
    };
    (
        [(http::header::CONTENT_TYPE, "text/javascript; charset=utf-8")],
        body,
    )
}

/// Build the full application router with all routes, the CORS layer, the
/// WebSocket Origin guard, the shared extensions, and the baseline security
/// headers. This is shared by the production [`run`] entrypoint and the
/// end-to-end integration tests, so the tests exercise exactly the same wiring
/// real clients hit.
pub fn build_app(
    backend_storage: HashMapStorage<VersionedGame>,
    stats: Arc<Mutex<InMemoryStats>>,
    origin_policy: OriginPolicy,
    resource_limits: Arc<ResourceLimits>,
) -> Router {
    let app = Router::new()
        .route(
            "/api",
            get(handle_websocket).layer(axum::middleware::from_fn(enforce_ws_origin)),
        )
        .route("/api/rpc", post(wasm_rpc_handler::handle_wasm_rpc))
        .route(
            "/default_settings.json",
            get(|| async { Json(settings::PropagatedState::default()) }),
        )
        .route("/full_state.json", get(state_dump::dump_state))
        .route("/stats", get(get_stats))
        .route("/runtime.js", get(runtime_settings))
        .route("/cards.json", get(|| async { Json(CARDS_JSON.clone()) }))
        .route(
            "/rules",
            get(|| async { Redirect::permanent("/rules.html") }),
        )
        .route("/public_games.json", get(state_dump::public_games));

    #[cfg(feature = "dynamic")]
    let app = app.fallback_service(get_service(
        ServeDir::new("../frontend/dist").fallback(ServeDir::new("../favicon")),
    ));
    #[cfg(not(feature = "dynamic"))]
    let app = app
        .route(
            "/",
            get(|| async { serve_static_routes(Path("index.html".to_string())).await }),
        )
        .route("/*path", get(serve_static_routes));

    let cors = build_cors_layer();

    let app = app
        .layer(cors)
        .layer(Extension(backend_storage))
        .layer(Extension(stats))
        .layer(Extension(origin_policy))
        .layer(Extension(resource_limits));

    apply_security_headers(app)
}

/// Construct the CORS layer from the `CORS_ALLOWED_ORIGINS` env var, matching
/// the production policy.
fn build_cors_layer() -> CorsLayer {
    let allowed_origins_raw = std::env::var("CORS_ALLOWED_ORIGINS")
        .unwrap_or_else(|_| DEFAULT_ALLOWED_ORIGINS.to_string());
    if allowed_origins_raw.trim() == "*" {
        CorsLayer::new()
            .allow_origin(tower_http::cors::Any)
            .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
            .allow_headers(tower_http::cors::Any)
    } else {
        let origins: Vec<HeaderValue> = allowed_origins_raw
            .split(',')
            .filter_map(|origin| origin.trim().parse::<HeaderValue>().ok())
            .collect();

        if origins.is_empty() {
            CorsLayer::new()
        } else {
            CorsLayer::new()
                .allow_origin(AllowOrigin::list(origins))
                .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
                .allow_headers(tower_http::cors::Any)
        }
    }
}

/// Production entrypoint: load persisted state, start the periodic dump task,
/// build the app, and serve forever on `0.0.0.0:3030`.
pub async fn run() -> Result<(), anyhow::Error> {
    ctrlc::set_handler(move || {
        info!(ROOT_LOGGER, "Received SIGTERM, shutting down");
        std::process::exit(0);
    })
    .unwrap();

    let (backend_storage, stats) = state_dump::load_state().await?;

    tokio::task::spawn(periodically_dump_state(
        backend_storage.clone(),
        stats.clone(),
    ));

    // The CORS allow-list (for /api/rpc) and the WebSocket Origin check (for the
    // /api handshake) are both derived from the single CORS_ALLOWED_ORIGINS env
    // var, so the two policies always stay in sync.
    let allowed_origins_raw = std::env::var("CORS_ALLOWED_ORIGINS")
        .unwrap_or_else(|_| DEFAULT_ALLOWED_ORIGINS.to_string());
    let origin_policy = OriginPolicy::from_raw(&allowed_origins_raw);
    if let OriginPolicy::AllowAny = origin_policy {
        warn!(
            ROOT_LOGGER,
            "CORS/WebSocket configured to allow ANY origin - not recommended for production"
        );
    }

    let resource_limits = ResourceLimits::from_env();
    info!(ROOT_LOGGER, "Resource limits configured"; "limits" => format!("{resource_limits:?}"));

    let app = build_app(
        backend_storage,
        stats,
        origin_policy,
        Arc::new(resource_limits),
    );

    axum::Server::bind(&SocketAddr::from(([0, 0, 0, 0], 3030)))
        .serve(app.into_make_service())
        .await?;

    info!(ROOT_LOGGER, "Shutting down");
    Ok(())
}

/// Build the `Content-Security-Policy` value, allowing the app to load its own
/// JS/WASM/CSS and to open a WebSocket to the configured `WEBSOCKET_HOST` (in
/// addition to same-origin, used in local/dev where the socket is same-origin).
fn content_security_policy() -> String {
    // Derive the wss:// (or ws://) origin to allow in connect-src from the
    // configured WEBSOCKET_HOST, which looks like "wss://host/api".
    let ws_connect = WEBSOCKET_HOST.as_ref().and_then(|h| {
        let trimmed = h.trim();
        if trimmed.is_empty() {
            return None;
        }
        // Keep only scheme://host[:port] for the CSP source expression.
        url_scheme_and_authority(trimmed)
    });

    let connect_src = match ws_connect {
        Some(origin) => format!("connect-src 'self' {origin}"),
        None => "connect-src 'self'".to_string(),
    };

    [
        "default-src 'self'",
        connect_src.as_str(),
        // WASM execution requires 'wasm-unsafe-eval' in modern browsers; the app
        // also inlines a small runtime.js bootstrap, hence 'unsafe-inline' for
        // scripts is intentionally avoided in favor of allowing only self + wasm.
        "script-src 'self' 'wasm-unsafe-eval'",
        // The React/CSS app uses inline styles in places.
        "style-src 'self' 'unsafe-inline'",
        "img-src 'self' data:",
        "font-src 'self' data:",
        "media-src 'self'",
        "object-src 'none'",
        "base-uri 'self'",
        "frame-ancestors 'none'",
        "form-action 'self'",
    ]
    .join("; ")
}

/// Extract the `scheme://authority` portion of a URL-ish string (e.g.
/// `wss://example.com/api` -> `wss://example.com`). Returns `None` if there is
/// no `scheme://` prefix.
fn url_scheme_and_authority(s: &str) -> Option<String> {
    let (scheme, rest) = s.split_once("://")?;
    let authority = rest.split('/').next().unwrap_or(rest);
    if authority.is_empty() {
        None
    } else {
        Some(format!("{scheme}://{authority}"))
    }
}

/// Wrap the router in baseline security-header layers. Each header is only set
/// if not already present, so route-specific content-types are preserved.
fn apply_security_headers(app: Router) -> Router {
    let csp = content_security_policy();
    app.layer(SetResponseHeaderLayer::if_not_present(
        header::CONTENT_SECURITY_POLICY,
        HeaderValue::from_str(&csp).expect("valid CSP header value"),
    ))
    .layer(SetResponseHeaderLayer::if_not_present(
        header::STRICT_TRANSPORT_SECURITY,
        HeaderValue::from_static("max-age=63072000; includeSubDomains"),
    ))
    .layer(SetResponseHeaderLayer::if_not_present(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    ))
    .layer(SetResponseHeaderLayer::if_not_present(
        header::X_FRAME_OPTIONS,
        HeaderValue::from_static("DENY"),
    ))
    .layer(SetResponseHeaderLayer::if_not_present(
        header::REFERRER_POLICY,
        HeaderValue::from_static("strict-origin-when-cross-origin"),
    ))
    .layer(SetResponseHeaderLayer::if_not_present(
        HeaderName::from_static("permissions-policy"),
        HeaderValue::from_static(
            "accelerometer=(), camera=(), geolocation=(), gyroscope=(), \
             magnetometer=(), microphone=(), payment=(), usb=()",
        ),
    ))
}

#[derive(Debug, Serialize, Deserialize)]
struct GameStats {
    num_games_created: u64,
    num_active_games: usize,
    num_players_online_now: usize,
    sha: &'static str,
}

async fn get_stats(
    Extension(backend_storage): Extension<HashMapStorage<VersionedGame>>,
) -> Result<Json<GameStats>, &'static str> {
    let num_games_created = backend_storage
        .clone()
        .get_states_created()
        .await
        .map_err(|_| "failed to get number of games created")?;
    let (num_active_games, num_players_online_now) = backend_storage
        .clone()
        .stats()
        .await
        .map_err(|_| "failed to get number of active games and online players")?;
    Ok(Json(GameStats {
        num_games_created,
        num_players_online_now,
        num_active_games,
        sha: &VERSION,
    }))
}

async fn periodically_dump_state(
    backend_storage: HashMapStorage<VersionedGame>,
    stats: Arc<Mutex<InMemoryStats>>,
) {
    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(60));
    loop {
        interval.tick().await;
        let _ =
            state_dump::dump_state(Extension(backend_storage.clone()), Extension(stats.clone()))
                .await;
    }
}

/// Middleware that rejects a WebSocket upgrade to `/api` when the browser
/// `Origin` header is not in the configured allow-list. This runs *before* the
/// `WebSocketUpgrade` extractor, closing the Cross-Site WebSocket Hijacking hole
/// the CORS layer (which only guards `/api/rpc`) leaves open. Non-browser
/// clients (which omit `Origin`) are permitted; see `OriginPolicy`.
pub async fn enforce_ws_origin<B>(
    Extension(origin_policy): Extension<OriginPolicy>,
    request: axum::http::Request<B>,
    next: axum::middleware::Next<B>,
) -> axum::response::Response {
    let origin = request
        .headers()
        .get(header::ORIGIN)
        .and_then(|v| v.to_str().ok());
    if !origin_policy.is_ws_origin_allowed(origin) {
        warn!(
            ROOT_LOGGER,
            "Rejecting WebSocket upgrade due to disallowed Origin";
            "origin" => origin.unwrap_or("<none>")
        );
        return (StatusCode::FORBIDDEN, "origin not allowed").into_response();
    }
    next.run(request).await
}

pub async fn handle_websocket(
    ws: WebSocketUpgrade,
    Extension(backend_storage): Extension<HashMapStorage<VersionedGame>>,
    Extension(stats): Extension<Arc<Mutex<InMemoryStats>>>,
    Extension(resource_limits): Extension<Arc<ResourceLimits>>,
) -> impl IntoResponse {
    // Cap the size of inbound frames so a single oversized message can't be used
    // to exhaust memory. A move is at most a few hundred bytes.
    let ws = ws.max_message_size(MAX_INBOUND_MESSAGE_BYTES);

    ws.on_upgrade(move |ws| {
        let ws_id = NEXT_USER_ID.fetch_add(1, Ordering::Relaxed);
        let logger = ROOT_LOGGER.new(o!("ws_id" => ws_id));
        info!(logger, "Websocket connection initialized");
        // Split the socket into a sender and receive of messages.
        let (mut user_ws_tx, mut user_ws_rx) = ws.split();

        // Use an unbounded channel to handle buffering and flushing of messages
        // to the websocket...
        let logger_ = logger.clone();
        let (tx, mut rx) = mpsc::unbounded_channel();
        tokio::task::spawn(async move {
            while let Some(v) = rx.recv().await {
                let _ = user_ws_tx.send(Message::Binary(v)).await;
            }
            debug!(logger_, "Ending tx task");
        });

        // And another channel to receive messages from the websocket
        let logger_ = logger.clone();
        let (tx2, rx2) = mpsc::unbounded_channel();
        // Per-connection inbound rate limiter (token bucket). Abusive connections
        // that exceed the burst/rate budget are disconnected.
        let mut rate_limiter =
            security::TokenBucket::new(resource_limits.msg_burst, resource_limits.msg_rate_per_sec);
        tokio::task::spawn(async move {
            while let Some(result) = user_ws_rx.next().await {
                match result {
                    Ok(Message::Close(_)) => {
                        break;
                    }
                    Ok(Message::Binary(r)) => {
                        // Defense-in-depth: drop oversized frames even though
                        // max_message_size should already have rejected them.
                        if r.len() > MAX_INBOUND_MESSAGE_BYTES {
                            warn!(logger_, "Dropping oversized inbound frame"; "len" => r.len());
                            continue;
                        }
                        if !rate_limiter.try_acquire() {
                            warn!(
                                logger_,
                                "Inbound message rate limit exceeded; disconnecting"
                            );
                            break;
                        }
                        let _ = tx2.send(r);
                    }
                    Ok(Message::Text(r)) => {
                        if r.len() > MAX_INBOUND_MESSAGE_BYTES {
                            warn!(logger_, "Dropping oversized inbound frame"; "len" => r.len());
                            continue;
                        }
                        if !rate_limiter.try_acquire() {
                            warn!(
                                logger_,
                                "Inbound message rate limit exceeded; disconnecting"
                            );
                            break;
                        }
                        let _ = tx2.send(r.into_bytes());
                    }
                    Ok(Message::Ping(_)) | Ok(Message::Pong(_)) => (),
                    Err(e) => {
                        error!(logger_, "Failed to fetch message"; "error" => format!("{e:?}"));
                        break;
                    }
                }
            }
            debug!(logger_, "Ending rx task");
        });

        shengji_handler::entrypoint(
            tx,
            rx2,
            ws_id,
            logger,
            backend_storage,
            stats,
            resource_limits,
        )
    })
    .into_response()
}

#[cfg(not(feature = "dynamic"))]
async fn serve_static_routes(Path(path): Path<String>) -> impl IntoResponse {
    static DIST: include_dir::Dir<'_> = include_dir::include_dir!("frontend/dist");
    static FAVICON: include_dir::Dir<'_> = include_dir::include_dir!("favicon");
    let mime_type = mime_guess::from_path(&path).first_or_text_plain();

    match DIST.get_file(&path).or_else(|| FAVICON.get_file(&path)) {
        Some(f) => Response::builder()
            .status(axum::http::StatusCode::OK)
            .header(
                http::header::CONTENT_TYPE,
                http::HeaderValue::from_str(mime_type.as_ref()).unwrap(),
            )
            .body(axum::body::boxed(Full::from(f.contents())))
            .unwrap(),
        None => Response::builder()
            .status(axum::http::StatusCode::NOT_FOUND)
            .body(axum::body::boxed(Empty::new()))
            .unwrap(),
    }
}

#[cfg(test)]
mod tests {
    use super::CARDS_JSON;

    static CARDS_JSON_FROM_FILE: &str = include_str!("../../frontend/src/generated/cards.json");

    #[test]
    fn test_cards_json_compatibility() {
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(
                &serde_json::to_string(&*CARDS_JSON).unwrap()
            )
            .unwrap(),
            serde_json::from_str::<serde_json::Value>(CARDS_JSON_FROM_FILE).unwrap(),
            "Run `yarn download-cards-json` with the backend running to sync the generated cards.json file"
        );
    }
}

#[cfg(test)]
mod ws_origin_tests {
    //! Integration tests for the WebSocket-handshake Origin allow-list (P0:
    //! Cross-Site WebSocket Hijacking defense).

    use std::sync::Arc;

    use axum::{routing::get, Extension, Router};
    use axum_test::TestServer;
    use slog::o;
    use tokio::sync::Mutex;

    use crate::security::{OriginPolicy, ResourceLimits};
    use crate::serving_types::VersionedGame;
    use crate::state_dump::InMemoryStats;
    use storage::HashMapStorage;

    fn test_router(policy: OriginPolicy) -> TestServer {
        let logger = slog::Logger::root(slog::Discard, o!());
        let backend_storage = HashMapStorage::<VersionedGame>::new(logger);
        let stats = Arc::new(Mutex::new(InMemoryStats::default()));
        let app = Router::new()
            .route(
                "/api",
                get(super::handle_websocket)
                    .layer(axum::middleware::from_fn(super::enforce_ws_origin)),
            )
            .layer(Extension(backend_storage))
            .layer(Extension(stats))
            .layer(Extension(policy))
            .layer(Extension(Arc::new(ResourceLimits::default())));
        TestServer::new(app).unwrap()
    }

    /// Issue a GET to `/api` carrying valid WebSocket upgrade headers plus the
    /// given Origin, so the `WebSocketUpgrade` extractor succeeds and the Origin
    /// allow-list check (not the upgrade validation) governs the outcome.
    async fn ws_handshake(server: &TestServer, origin: &'static str) -> http::StatusCode {
        server
            .get("/api")
            .add_header(
                http::header::CONNECTION,
                http::HeaderValue::from_static("upgrade"),
            )
            .add_header(
                http::header::UPGRADE,
                http::HeaderValue::from_static("websocket"),
            )
            .add_header(
                http::header::SEC_WEBSOCKET_VERSION,
                http::HeaderValue::from_static("13"),
            )
            .add_header(
                http::header::SEC_WEBSOCKET_KEY,
                http::HeaderValue::from_static("dGhlIHNhbXBsZSBub25jZQ=="),
            )
            .add_header(http::header::ORIGIN, http::HeaderValue::from_static(origin))
            .await
            .status_code()
    }

    #[tokio::test]
    async fn rejects_disallowed_origin_with_403() {
        let server = test_router(OriginPolicy::from_raw("https://allowed.example"));
        let status = ws_handshake(&server, "https://evil.example").await;
        assert_eq!(
            status,
            http::StatusCode::FORBIDDEN,
            "a WebSocket upgrade from a disallowed Origin must be rejected with 403"
        );
    }

    #[tokio::test]
    async fn allowed_origin_passes_origin_check() {
        // With an allowed Origin, the Origin check passes and the handshake is
        // accepted (101 Switching Protocols), not 403.
        let server = test_router(OriginPolicy::from_raw("https://allowed.example"));
        let status = ws_handshake(&server, "https://allowed.example").await;
        assert_ne!(
            status,
            http::StatusCode::FORBIDDEN,
            "an allowed Origin must pass the Origin check (got 403)"
        );
    }
}
