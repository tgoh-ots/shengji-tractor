#!/usr/bin/env bash
#
# Vercel build script for the OPTIONAL standalone frontend deploy (Path B in
# DEPLOY.md). The webpack build compiles the `shengji-wasm` crate via
# WasmPackPlugin, so the Vercel build image needs a Rust toolchain, the
# wasm32-unknown-unknown target, and wasm-pack — none of which Vercel provides
# by default. This script bootstraps them, then builds the frontend.
#
# The frontend bakes process.env.WEBSOCKET_HOST into the bundle (webpack
# DefinePlugin). Set WEBSOCKET_HOST in the Vercel project's build env, e.g.
#   WEBSOCKET_HOST=wss://<your-backend-host>/api
#
set -euo pipefail

echo "==> Bootstrapping Rust toolchain for the WASM build"
if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
# shellcheck disable=SC1090
source "$HOME/.cargo/env"

rustup target add wasm32-unknown-unknown

if ! command -v wasm-pack >/dev/null 2>&1; then
  echo "==> Installing wasm-pack"
  curl -fsSL https://rustwasm.github.io/wasm-pack/installer/init.sh | sh
fi

echo "==> Building frontend (WEBSOCKET_HOST=${WEBSOCKET_HOST:-<unset>})"
cd frontend
yarn install --frozen-lockfile
yarn build
