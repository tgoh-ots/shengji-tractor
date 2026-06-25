# Shengji Online (升级 Online)

Play 升级 — the popular Chinese trick-taking card game also known as **Tractor**,
**Finding Friends**, **拖拉机**, or **找朋友** — online with friends, or against
computer opponents.

**▶ Play now: https://shengji-tractor.fly.dev**

This is an open-source fork of the [rbtying/shengji](https://github.com/rbtying/shengji)
engine, fully redesigned with a responsive bilingual (中文 / English) interface,
cheat-proof AI bots, and one-command deployment on [Fly.io](https://fly.io).

## Features

- **Play with friends online** — share a room link; supports the full ruleset
  (Tractor / Finding Friends), with extensive per-round settings (number of
  decks, scoring thresholds, kitty size, throw-evaluation policies, bombs, and
  more).
- **Computer opponents** — fill empty seats with AI bots across several skill
  tiers (Easy / Hard / Expert and an "Omniscient" perfect-information tier).
  The honest tiers are **cheat-proof**: they only see their own hand. The
  Omniscient tier is a deliberate, clearly-badged perfect-information opponent.
  > Note: tier names and behavior are actively being tuned, so treat the above
  > as a general description rather than a fixed contract.
- **Modern responsive UI** — a redesigned Tailwind-based interface with light
  and dark themes, four-color suits, and a built-in **About** panel describing
  the project and its sources.
- **Bilingual** — toggle between 中文 and English from the toolbar.

## Play

The hosted instance lives at **https://shengji-tractor.fly.dev** — no install
required. Just open the link, pick a name, and share the room with friends.

## Run locally

```bash
cd frontend && yarn build && cd .. && cd backend && cargo run
```

The server is a self-contained static binary and does not terminate TLS. It
listens on `127.0.0.1:3030`, and should only be exposed to an external network
behind a proxy that supports both HTTP and WebSocket protocols.

### Development

```bash
# in one terminal
cd frontend && yarn watch
# in another
cd backend && cargo run --features dynamic
```

See [CLAUDE.md](CLAUDE.md) for the full set of build, test, lint, and
type-generation commands.

### Environment variables

- `CORS_ALLOWED_ORIGINS`: Comma-separated list of allowed origins for CORS
  requests to the `/api/rpc` endpoint (e.g.
  `"https://example.com,https://app.example.com"`). Set to `"*"` to allow any
  origin (not recommended for production). If not set, defaults to allowing
  common localhost origins for development.

## Deploy

The project deploys to Fly.io. See **[DEPLOY.md](DEPLOY.md)** for the full,
step-by-step deployment guide (Docker image, `fly.toml`, and configuration).

## Technical details

The entire state of each game is stored in the memory of the server process.
Restarting the server kicks all players, and games are automatically closed
when all players have disconnected. The bulk of the game logic is implemented
in Rust (shared with the client via WebAssembly); the frontend is React +
TypeScript. See [CLAUDE.md](CLAUDE.md) for the architecture overview.

## Credits

This project is a fork and would not exist without the work it builds on:

- **Original engine:** [github.com/rbtying/shengji](https://github.com/rbtying/shengji),
  released under the MIT License — the foundation of this fork. See
  [LICENSE](LICENSE) and [NOTICE](NOTICE), whose original copyright is preserved intact.

### Game rules

- [Wikipedia: "Sheng ji"](https://en.wikipedia.org/wiki/Sheng_ji)
- [pagat.com: Tractor (Tuo La Ji)](https://www.pagat.com/kt5/tractor.html)

### AI research

- [Berkeley EECS-2023-127: ShengJi+](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2023/EECS-2023-127.html)
  — search/learning approaches for ShengJi.
- [DouZero](https://github.com/kwai/DouZero) — deep reinforcement learning for
  related Chinese card games.

## License

[MIT](LICENSE). The original copyright notice is preserved; this fork adds a
[NOTICE](NOTICE) clarifying the relationship to the upstream project.
