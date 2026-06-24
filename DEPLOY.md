# Deploying Shengji

This is a step-by-step runbook for deploying your fork of Shengji.

The whole app is **one Rust binary** (`shengji`). It serves BOTH the static
frontend (embedded into the binary at build time) AND the game WebSocket at
`/api` from a single process listening on `0.0.0.0:3030`. So a single backend
service is fully playable on its own — no separate frontend host is required.

Everything deploys from one self-contained image: **`Dockerfile.deploy`** (at
the repo root). It uses only official base images (no private build images),
builds the frontend and backend from source, and embeds the frontend into the
binary.

There are two shapes you can deploy:

- **Path A (recommended): single service.** One container serves everything.
  Simplest and reliable. This is the default.
- **Path B (optional): split frontend.** Host the static frontend on Vercel
  (faster first paint, served from a CDN) pointing its WebSocket at your
  backend. The backend still runs the same image.

---

## What YOU must do vs. what's automated

**You must do (accounts / one-time, cannot be automated for you):**

- Create a GitHub repo for your fork and push to it.
- Create a hosting account for the backend (Fly.io — see Path A) and run
  `fly auth login`. Fly is pay-as-you-go and requires a credit card (~$2/mo,
  cheaper with auto-stop). Free alternatives are documented below.
- (Path B only) Create a Vercel account and run `vercel login`.
- Set the production environment variables / secrets in the host's dashboard or
  CLI (listed below).

**Automated (already in the repo):**

- `Dockerfile.deploy` — builds the whole app, single-arch `linux/amd64`.
- `fly.toml` — Fly.io service config.
- `vercel.json` + `scripts/vercel-build.sh` — optional Vercel frontend build.
- `.github/workflows/ci.yml` — builds + fast tests on every push.
- `.github/workflows/docker-publish.yml` — builds `Dockerfile.deploy` and
  pushes to `ghcr.io/<your-username>/shengji` (used by the "deploy from a
  prebuilt image" options).

---

## Environment variables

| Var | Required | Value | Notes |
|-----|----------|-------|-------|
| `CORS_ALLOWED_ORIGINS` | **Yes (prod)** | Comma-separated origin allow-list, e.g. `https://your-app.fly.dev` | Guards both CORS (`/api/rpc`) and the WebSocket `Origin` check. **Do NOT use `*` in production.** Defaults to localhost dev origins if unset. |
| `WEBSOCKET_HOST` | No | `wss://<backend-host>/api` | **Leave UNSET for Path A** (the frontend uses same-origin). Set it only for Path B so the standalone frontend knows where the backend WebSocket is. |
| `VERSION` | No | any label, e.g. `fly` or a git SHA | Shown in logs and the UI footer. |
| `DUMP_PATH` / `MESSAGE_PATH` | No | file paths | Periodic in-memory state snapshot for crash/restart recovery. Default `/tmp/...`. |
| `SENTRY_DSN` | No | secret | Optional server-side error reporting. |
| `UPSTASH_REDIS_REST_URL` / `_TOKEN` | No | secret | Optional snapshot durability across host restarts. |

The server **hard-binds `0.0.0.0:3030`** — it does NOT read a `$PORT` variable.
Configure the host's service/internal port to **3030**.

See `.env.example` for a copy-pasteable template.

---

## Path A (recommended): single service on Fly.io

Fly.io is the primary path: cheapest always-on option (~$2/mo, less with
auto-stop), managed, native WebSocket/WSS support, deploys our Docker image
directly.

### 1. Push your fork to GitHub (one-time)

```bash
# from the repo root
git remote add origin git@github.com:<your-username>/shengji.git   # or https://
git push -u origin master
```

### 2. Install + authenticate the Fly CLI (one-time, you do this)

```bash
# macOS
brew install flyctl
# or: curl -L https://fly.io/install.sh | sh

fly auth login          # opens a browser; create the account here if needed
```

You'll be asked for a credit card (pay-as-you-go). A single `shared-cpu-1x`
256mb machine is ~$2/mo always-on, and less if you let it auto-stop when idle.

### 3. Configure `fly.toml`

The repo ships a `fly.toml`. Before deploying, edit it:

- Set `app = "..."` to a unique name (or let `fly launch` choose one).
- Set `primary_region` to one near your players
  (<https://fly.io/docs/reference/regions/>).
- Set `CORS_ALLOWED_ORIGINS` under `[env]` to your real Fly app URL, e.g.
  `https://your-app.fly.dev` (you'll know the exact URL after `fly launch`).

Key settings already in `fly.toml`:

- `[build] dockerfile = "Dockerfile.deploy"` — builds our image.
- `[http_service] internal_port = 3030`, `force_https = true` — HTTPS/WSS.
- `auto_stop_machines = "stop"`, `min_machines_running = 0` — pay-per-use:
  the machine stops when idle and wakes (~1-2s) on the next request. The game's
  reconnect-by-name already handles the brief wake, so mid-game wakes are
  seamless. Set `min_machines_running = 1` for always-on (no cold start).
- `[[vm]]` `shared-cpu-1x` / `256mb` — bump `memory` to `512mb` if you hit OOM.

### 4. Launch + deploy

```bash
# First time: create the app from the existing fly.toml WITHOUT deploying yet,
# so you can review/adjust the generated config, then deploy.
fly launch --no-deploy --copy-config

# Set the production CORS origin to your app's real URL (shown by `fly launch`):
fly secrets set CORS_ALLOWED_ORIGINS="https://your-app.fly.dev"
# (or put it in fly.toml [env] since it's non-secret — either works)

fly deploy
```

Subsequent deploys are just `fly deploy`. The first build is a cold Rust
release build and takes several minutes; later builds reuse layer cache.

### 5. Play

Open `https://your-app.fly.dev/` and play. Share the URL with friends.

---

## Path B (optional): Vercel frontend + Fly.io backend

Use this only if you want the page to load from Vercel's CDN (faster first
paint). The backend stays on Fly (or any host from the alternatives). With a
split, an idle backend's cold start only delays the WebSocket connect, not the
initial page render.

### 1. Deploy the backend (Path A above)

Note its WebSocket URL: `wss://your-app.fly.dev/api`.

### 2. Deploy the frontend to Vercel

```bash
npm i -g vercel
vercel login          # you do this; create the account here if needed
```

In the Vercel project settings (or during `vercel`), set a **build-time**
environment variable:

```
WEBSOCKET_HOST = wss://your-app.fly.dev/api
```

This is baked into the bundle by webpack's `DefinePlugin` (see
`frontend/webpack.config.js`). The repo's `vercel.json` already points the
build at `scripts/vercel-build.sh`, which installs the Rust + wasm-pack
toolchain the WASM build needs and runs `yarn build`, outputting `frontend/dist`.

Deploy:

```bash
vercel            # preview
vercel --prod     # production
```

### 3. Point the backend's CORS at the Vercel origin

The backend must allow the Vercel origin for both CORS and the WebSocket
handshake:

```bash
fly secrets set CORS_ALLOWED_ORIGINS="https://your-frontend.vercel.app"
```

(Comma-separate if you serve from multiple origins, e.g. a custom domain.)

### Trade-off

- **Pro:** the page loads instantly from Vercel's CDN even if the backend is
  asleep; only the live game socket waits for the backend to wake.
- **Con:** two services to manage, and you must keep `CORS_ALLOWED_ORIGINS` and
  `WEBSOCKET_HOST` consistent.

For most people **Path A is simpler and good enough.**

---

## Backend host alternatives

All of these run the **same `Dockerfile.deploy` image** on port **3030**.

### Render (free, no credit card) — primary free option

Render's free web service runs a Docker image, supports WebSockets, and needs
no credit card.

1. Push your fork to GitHub.
2. Render dashboard → **New → Web Service** → connect your repo.
3. Runtime: **Docker**; Dockerfile path: `Dockerfile.deploy`.
4. Set the **HTTP port** to `3030`.
5. Add env vars: `CORS_ALLOWED_ORIGINS` (your `https://<app>.onrender.com`),
   `VERSION`. Leave `WEBSOCKET_HOST` unset for single-service.
6. Deploy.

(Or deploy from the prebuilt image `ghcr.io/<you>/shengji` that
`docker-publish.yml` pushes.)

**Caveat — sleep:** the free instance **sleeps after 15 minutes of inactivity**
and takes ~30–50s to cold-start on the next request. To stay warm, add an
**optional keep-alive ping** every ≤14 minutes hitting `/stats` (a cheap JSON
endpoint) — e.g. a free external cron (cron-job.org / UptimeRobot) or a GitHub
Actions cron. Keep total uptime within Render's ~750 free instance-hours/month
(one always-warm service ≈ 730 h, so a single service fits).

### Oracle Cloud "Always Free" ARM VM — free, always-on (no cold start)

Oracle's Always Free tier includes an **Ampere A1 (ARM64)** VM you can run 24/7.

1. Create an **Always Free** Ampere A1 VM (e.g. 1 OCPU / 6 GB), Ubuntu image.
2. Install Docker on the VM:
   `curl -fsSL https://get.docker.com | sh`.
3. **Architecture note:** the VM is **ARM64**, but `Dockerfile.deploy` defaults
   to `linux/amd64`. Build/pull an arm64 image:
   - Build on the VM (simplest): clone the repo and
     `docker build --platform linux/arm64 -f Dockerfile.deploy -t shengji .`
   - Or publish multi-arch from CI: run the **Docker publish** workflow via
     "Run workflow" with `platforms = linux/amd64,linux/arm64`, then
     `docker pull ghcr.io/<you>/shengji` on the VM.
4. Run it:
   ```bash
   docker run -d --restart=always -p 3030:3030 \
     -e CORS_ALLOWED_ORIGINS="https://your-domain" \
     -e VERSION=oracle \
     --name shengji ghcr.io/<you>/shengji
   ```
5. **Open port 3030:** add an ingress rule in the VCN Security List / Network
   Security Group for TCP 3030 (and run `sudo ufw allow 3030` if the OS
   firewall is on).
6. **TLS/WSS:** browsers require `wss://` (secure) WebSockets from an `https://`
   page. Front the container with **Caddy** (easiest — automatic Let's Encrypt)
   or nginx to terminate TLS on 443 and proxy to `127.0.0.1:3030`. With Caddy,
   a two-line `Caddyfile` (`your-domain { reverse_proxy 127.0.0.1:3030 }`) gets
   you HTTPS + WSS automatically.

**Pro:** truly always-on, no cold starts, free. **Con:** more setup (VM, ARM
build, TLS); Oracle Always Free capacity can be hard to obtain in some regions.

### Fly.io is the recommended paid option

Already covered in Path A (~$2/mo always-on, or pay-per-use with auto-stop).
Cheapest managed always-on path; native WSS; deploys our image directly.

---

## Local verification (what was tested before shipping)

The deploy image was built and run locally to de-risk the cloud deploy:

```bash
# Build (slow: a cold Rust release build, ~10-20 min)
docker build -f Dockerfile.deploy -t shengji-deploy .

# Run
docker run --rm -d -p 3030:3030 -e CORS_ALLOWED_ORIGINS='*' \
  --name shengji-test shengji-deploy

# Verify
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:3030/   # -> 200
curl -s http://127.0.0.1:3030/stats                               # -> JSON
curl -sI http://127.0.0.1:3030/ | grep -i content-security-policy # -> CSP header

docker stop shengji-test
```

Note: `CORS_ALLOWED_ORIGINS='*'` is for LOCAL testing only. The server logs a
warning when configured to allow any origin. **Never ship `*` in production.**

---

## Production security checklist

- [ ] **`CORS_ALLOWED_ORIGINS` set to the real origin(s)** — your deployed
      frontend URL(s), NOT `*`. This is both the CORS allow-list and the
      WebSocket `Origin` check (anti cross-site WebSocket hijacking).
- [ ] **HTTPS / WSS only.** Fly/Render terminate TLS for you (`force_https`);
      on a raw VM, front the container with Caddy/nginx so the page is served
      over `https://` and the socket over `wss://`.
- [ ] **`WEBSOCKET_HOST`** matches your setup: unset for Path A (same-origin),
      `wss://<backend>/api` for Path B.
- [ ] **Security headers are already applied** by the backend on every response
      (CSP, HSTS, `X-Content-Type-Options`, `X-Frame-Options: DENY`,
      `Referrer-Policy`, `Permissions-Policy`) — verify with
      `curl -sI https://<your-app>/`.
- [ ] Keep secrets (`SENTRY_DSN`, Upstash tokens) in the host's secret store —
      never in git. `.env` is gitignored; `.env.example` documents the shape.

---

## Notes & caveats

- **Build time:** the first cloud build is a cold Rust release build and takes
  several minutes. LTO is disabled in the deploy build
  (`CARGO_PROFILE_RELEASE_LTO=false`) to keep it reasonable; this is fine for a
  hobby game. Subsequent builds reuse cache.
- **Port:** the server hard-binds **3030** and ignores `$PORT`. Always set the
  host's service/internal port to 3030.
- **Cold starts & reconnect:** any scale-to-zero host (Fly auto-stop, Render
  free sleep) has a short wake delay on the first request after idle. The game
  supports **reconnect-by-name**, so a player who was mid-game can rejoin the
  same seat after a brief disconnect — wakes are seamless in practice.
- **Single arch:** `Dockerfile.deploy` targets `linux/amd64` (what Fly and
  Render run). For Oracle ARM, build/pull `linux/arm64` as noted above.
