# Superior-Play Bot Training Plan

> **Goal:** train a new bot that plays 升级 (Shengji/Tractor) *materially better* than the
> current strongest honest tier (Enoch), proven by the repo's paired-on-mirrored-deck
> win-rate harness. Several distinct approaches are proposed, spanning the major game-AI
> traditions, and benchmarked against each other and the existing ladder.
>
> **Constraints (from the user):** no human-feedback budget, no pre-existing game DB
> (self-play only); be creative / not limited to the existing net; token & training time
> are *not* a concern given reasonable return; torch/tooling install OK; GPU/cloud OK if it
> pays off.
>
> **Drafted:** 2026-06-29. Companion to `docs/bot-training-roadmap.md` (the prior 1-month
> plan) and `docs/bot-eval-baseline.md` (the measurement substrate). Grounded in a full
> code survey (file:line refs throughout).

---

## 0. What already exists (so we don't rebuild it)

The repo is further along than a greenfield project — this plan *stands on* three things
that are already built and tested:

1. **A rigorous measurement substrate.** `core/src/bot/harness.rs::run_paired_ab` plays
   each deck in *both* orientations to cancel deal luck, and reports win-rate with a
   **Wilson CI**, a **paired bootstrap CI over decks**, and a **minimum-detectable-effect**.
   It resolves **±3–5pp** strength gains. `core/tests/baseline_gate.rs` commits floors.
   → *We can prove a new bot is stronger, not just claim it.* This is the hard part of game-AI
   work and it's done.

2. **A net→search plumbing path that's ready for new heads.** `expert.rs` runs an ONNX net
   via pure-Rust `tract-onnx` (no C deps, musl-deployable). It already supports:
   - a **runtime model override** (`SHENGJI_EXPERT_MODEL_PATH`) → A/B a net with **no rebuild**;
   - a **2-output (policy + value) model** with a **gated value-head leaf blend**
     (`SHENGJI_VALUE_WEIGHT`, default 0 = OFF) in `search.rs::evaluate_position`;
   - graceful fallback to the hand-written heuristic if a net fails to load/run.

3. **A self-play data + training pipeline.** `core/examples/gen_training_data.rs`
   (Omniscient teacher labels, honest-only features, DAgger `GEN_BEHAVIOUR` knob,
   value-target back-fill) → `training/train_expert.py` (shared trunk + policy head +
   value head, ONNX export) → `training/run_value_pipeline.sh` (resumable, sharded).

**The single biggest unrealized lever:** the value-head pipeline is **fully coded but
INERT** — *no value net has ever been trained*; both embedded models
(`expert_model.onnx`, `expert_model_old.onnx`) are **policy-only**, and the search leaf is
still the hand-rolled `evaluate_position` with a documented **hoarding bias** (it credits
only your own jokers/aces/trumps and ignores opponents' winners). The prior roadmap calls
training a value head "the change worth a month." Nobody has spent that month.

### The key constraints the user just lifted

The prior roadmap's opinionated **"AVOID" list** (bigger nets, learned belief models, CFR,
augmentation) was justified *explicitly* by two assumptions: a **~5,000-game corpus** and
**no compute budget**. The user has lifted both — we can generate **millions** of self-play
games and train for as long as it takes. That re-opens the most interesting, highest-ceiling
directions, which is where the creative options below live.

### The honest game characterization (why method choice matters)

- **4 players, fixed 2v2 partnership** (Tractor mode; seats 0&2 vs 1&3 in the harness).
  Multi-deck (typically 2 decks / 108 cards). Phases: Initialize → **Draw** (draw + declare
  trump) → **Exchange** (kitty/扣底 swap) → **Play** (trick-taking).
- **Imperfect information, team game, long horizon (~25 tricks).** Hidden: every other hand
  (redacted to `Card::Unknown`) + the kitty. Public: completed tricks
  (`played_this_hand`), inferred voids, bids. This is the **Skat/Mahjong/Dou-Dizhu** family,
  *not* the perfect-info Go/chess family.
- **Huge, variable, structured action space:** leads can be singles, pairs, tractors, and
  multi-unit **throws**; follows are constrained by the led format. The action space is
  closer to Dou Dizhu (which motivated DouZero's design) than to chess.
- **Reward signal:** `non_landlords_points` (isize) → level deltas via
  `GameScoringParameters`. The kitty is multiplied (×trick-unit-size) and awarded to the
  last-trick winner. The **per-hand oriented point margin** is the natural value target
  (already what `gen_training_data.rs` back-fills, normalized by `VALUE_NORM = 200`).

---

## 1. How the great game bots are trained, and what transfers here

| Tradition | Method | Perfect/Imperfect info | What transfers to Shengji |
|---|---|---|---|
| **AlphaGo→AlphaZero→MuZero** (Go/chess/shogi) | Self-play + MCTS-guided **expert iteration**; policy+value net | Perfect | The ExIt *loop* + value-as-leaf. For imperfect info we use **determinized (IS)MCTS** — already in `search.rs`. |
| **Poker** Libratus/Pluribus/DeepStack/**ReBeL** | **CFR** / regret minimization, abstraction, **depth-limited subgame solving** with value nets on belief states | Imperfect | Full-game CFR is intractable here (prior roadmap correctly flags worst-ROI); but **subgame/endgame solving** with a value net is the tractable slice. |
| **Bridge** GIB / WBridge5 / NN bidders | **Double-dummy solver** (perfect-info) labels millions of layouts → NN; **PIMC** + constraint **sampling** of hidden hands at play time | Imperfect | This *is* the current Omniscient-teacher distillation. The upgrade: a **learned hidden-hand sampler** (constraint sampling done with a net). |
| **Skat / Hearts / Spades** | **PIMC / ISMCTS** + **opponent-card inference** + **alpha-mu** (combats strategy fusion) | Imperfect | Direct: a **belief/inference net** to replace uniform world sampling; alpha-mu as a search upgrade. |
| **Mahjong — Suphx** (Microsoft) | Supervised from logs **+ RL** + **oracle guiding** (a perfect-info "oracle" critic gradually distilled into an honest agent) + global reward | Imperfect | **Oracle guiding** = use our Omniscient state as a *privileged critic during training only*, not as a label to clone. Principled fix for the aliasing floor. |
| **Dou Dizhu — DouZero / PerfectDou** | **Deep Monte-Carlo** (action-value net, sampled returns, **no search, no human data, pure self-play**); PerfectDou adds **perfect-info distillation** | Imperfect | Most analogous game. DMC is a genuinely different, search-free architecture that *beat* MCTS/CFR bots on the huge Dou-Dizhu action space. |
| **StarCraft/Dota** AlphaStar / OpenAI Five | Large-scale **self-play RL** (PPO/LSTM) + **league / population-based training** for robustness | Imperfect, team | League/PBT to avoid overfitting one opponent and to make team play robust. |

**The two most relevant lessons for *our* exact constraints** (no human data, imperfect-info
Chinese trick game, team play):
- **DouZero** proves you can reach top strength with *pure self-play Deep Monte-Carlo and no
  tree search* on a Dou-Dizhu-sized action space — directly applicable, and architecturally
  the biggest departure from the current MCTS bot.
- **Suphx oracle-guiding / PerfectDou perfect-info distillation** give the *principled* way
  to use our cheating Omniscient agent: as a **privileged critic during training**, not a
  policy to clone. This dissolves the "behavioral-cloning-of-an-unidentifiable-teacher"
  aliasing floor the prior roadmap flagged as the current approach's hard ceiling.

---

## 2. Where compute actually helps (torch / GPU / cloud — honest answer)

The serve-time net runs in the **Rust search hot loop** via `tract-onnx`: the value leaf is
called **O(worlds × candidates)** per decision (≈ 144 × 6), inside a **2,200 ms** budget.
**This caps net size at inference, not at training.** Consequences:

- **The real bottleneck is CPU self-play data generation**, which is embarrassingly parallel
  (sharded by `GEN_SEED`). Search-driven generation (`GEN_BEHAVIOUR=mix/expert`) is *hours*
  for thousands of games on 10 cores; ExIt/DMC want **millions** of decisions.
  → **Highest-value compute = many CPU cores.** A 64–128 vCPU cloud box gives a ~6–13×
  speedup on every data-gen and self-play-RL phase. This is the lever I recommend.
- **Net training is cheap.** The MLP trains in minutes on CPU; Apple-Silicon **MPS** is
  available locally for free. **A GPU barely helps the small leaf/value nets.**
- **A GPU *is* worth it only if we commit to a larger encoder** (DeepSets/transformer over
  the full hand + history + belief) used as a **root-prior / belief net** — i.e. called
  **O(1) per decision**, *outside* the per-leaf hot loop. Such a net can be bigger and
  GPU-trained. The per-leaf value net must stay small regardless.
- **torch install:** CPU/MPS wheel in a venv (`training/requirements.txt` already lists
  torch/onnx/numpy). Done in Phase 0.

**Recommendation:** rent a big **CPU** box for the heavy self-play phases; add a GPU only if
we green-light the large-encoder belief net (Option D / the "Maestro" variant). I'll provide
exact instance specs + a turन-key setup script; provisioning needs your cloud account.

---

## 3. The bot options

Five candidates, each a different tradition and a distinct playing style. All stay **honest**
(never flip `sees_perfect_information`); the Omniscient cheater is used only as a
training-time teacher/critic, which is allowed and already how Expert is built.

### A — "Sage" — Value-Augmented Expert  *(AlphaZero leaf-value + bridge DDS-labeling)*
- **Idea:** finally *train the inert value head* and turn on the leaf blend. Policy stays the
  distilled teacher prior; the **learned value net replaces the hoarding-biased static leaf**.
- **Recipe:** big `GEN_BEHAVIOUR=mix` dataset with a strong teacher budget → train the
  existing 2-output net (`train_expert.py`) → embed → sweep `SHENGJI_VALUE_WEIGHT` and
  `rollout_tricks` via `paired_eval`. Mostly **runs the existing resumable pipeline**.
- **Risk:** lowest. **Cost:** ~1 day wall (data-gen dominated). **Expected:** +3–8pp vs
  Expert (the roadmap's headline bet), validating the whole stack.
- **Style:** *the efficient closer* — stops hoarding honors, spends high cards to seize point
  tricks at the right tempo, better trump economy, noticeably stronger mid/endgame.

### B — "Athena" — Self-play Actor-Critic with a privileged critic  *(Suphx oracle-guiding / PerfectDou + AlphaZero ExIt)*
- **Idea:** stop *cloning* the cheating teacher (aliasing floor). Instead train an **honest
  policy+value net by self-play**, using the **perfect-information state as a privileged
  critic during training only** (asymmetric actor-critic). Iterate: search(net_g) generates
  improved targets + realized outcomes → train net_{g+1} → **paired-gate** vs net_g → repeat.
- **Recipe:** new data-gen mode that records the **search's visit distribution** (not the
  Omniscient action) as the policy target + realized margin as the value target; a perfect-info
  critic head trained on the Omniscient view, distilled toward the honest value. A small
  **league** (keep past generations as opponents) for robustness.
- **Risk:** medium (new training loop). **Cost:** ~3–5 days, iterative. **Expected:** highest
  *ceiling* of the policy-net options — can exceed the Omniscient teacher because search +
  honest value compound.
- **Style:** *the alien optimizer* — not bound to the human playbook; well-calibrated risk;
  surprising-but-effective tempo and throws.

### C — "Oracle" — Deep Monte-Carlo, search-free  *(DouZero)*
- **Idea:** the biggest architectural departure. A net **Q(state, candidate-action) →
  expected oriented margin**, trained by **self-play Deep Monte-Carlo** (epsilon-greedy
  rollouts, sampled MC returns, no bootstrapping, no tree search). At serve time: **argmax Q
  over legal candidates** — *no determinized search at all*.
- **Recipe:** new action-conditioned net + a DMC self-play trainer; a new `PlayBrain`/tier to
  serve it. Reuses the candidate enumerators and feature encoder.
- **Risk:** medium-high (most new code). **Cost:** ~2–4 days. **Expected:** DouZero beat
  MCTS/CFR bots in Dou Dizhu *specifically because* of the huge action space — Shengji has the
  same property. Also **far faster at serve time** (no 144-world search).
- **Style:** *the instinctive speed-demon* — fast, pattern-rich, superb value calibration on
  common shapes; may miss rare deep-tactical endgames that explicit search would find.

### D — "Seer" — Learned-belief determinizer  *(Skat/bridge inference + alpha-mu)*
- **Idea:** PIMC's core weakness is **uniform world sampling** that ignores what bids/plays
  reveal. Train an **inference net P(hidden hands | my view + public history)** from self-play
  (true hidden hands are known in self-play) and use it to **importance-sample/weight** the
  determinized worlds in `determinize.rs`, and feed **belief features** into the policy/value
  net. Layers on top of A or B.
- **Risk:** medium. **Cost:** ~2–3 days. **Expected:** the fix the prior roadmap deferred
  ("Bayesian weighting"); directly attacks strategy-fusion blunders. Compounds with A/B/C.
- **Style:** *the card-counter / table-reader* — reads voids and partner shape, finesses,
  coordinates with partner like a strong human pair; fewest "fantasy-world" blunders.

### E — "Endgame" — Exact tail solver  *(chess tablebase / poker subgame solving / alpha-mu)* — **optional stretch**
- **Idea:** build the **complete legal-move enumerator** in `mechanics` (the deferred blocker
  — the current generators emit only a heuristic subset and can't represent multi-unit
  throws), then exact alpha-beta over the last ≤K tricks. De-noises teacher value labels and
  serves as a perfect endgame leaf; optionally **alpha-mu** for the imperfect-info endgame.
- **Risk:** high (correctness-critical `mechanics` work). **Cost:** multi-day + careful review.
  **Expected:** flawless endgames + better teacher labels, but the enumerator is the
  high-cost prerequisite the prior roadmap flagged. **Recommend last**, only if A–D leave a
  measurable endgame gap.
- **Style:** *the endgame surgeon* — perfect last-trick squeezes and throw-ins.

> **Architecture note (the "Maestro" variant):** B/C/D can each be run with a **richer
> encoder** (full-hand multi-hot + history + belief features, bigger trunk) used as the
> **root prior / belief net** (O(1) per decision, so inference cost is fine). This is the one
> place a GPU earns its keep. Treated as a knob on B/C/D, not a separate bot.

### F — "Grandmaster" — the composition (the whole point of building A–D separately)

A–D are **not rivals; they are four learnable *slots* of one determinized-search agent**, so
building them separately is an **ablation study** — the paired harness measures each
component's marginal win-rate, and the final bot keeps the ones that help:

| Search slot | Owned by | Effect |
|---|---|---|
| World sampling (`determinize.rs`) | **Seer** belief net | realistic worlds, not uniform |
| Root prior + rollout policy | **Athena** policy (or Sage's distilled policy / Oracle's Q) | which moves to search & how to roll out |
| Leaf evaluator (`evaluate_position`) | **Sage / Athena** value head | un-hoarded leaf value |
| Search-free fast path | **Oracle** DMC Q | instant argmax-Q |

The codebase is *built* for this composition: the `Policy` enum (prior/rollout), the gated
`SHENGJI_VALUE_WEIGHT` leaf blend, and the `determinize.rs` sampler are exactly the pluggable
slots. Concrete hybrids we'll measure:
- **Full learned ISMCTS:** Seer beliefs → Athena policy prior + rollouts → Athena/Sage value
  leaf. The "AlphaZero/ReBeL-for-imperfect-info" agent with every component learned.
- **System-1/System-2 (fast/slow):** serve **Oracle's argmax-Q instantly** when its top-2
  candidates are far apart or the move is forced; fall back to the **full belief+search**
  only on contested decisions / the endgame. (Suphx-style run-time allocation; also keeps
  serve latency low on average.)
- **Human-knowledge + learned-value:** keep **Enoch's playbook** as a prior/rollout influence
  (its principles — no high-trump opens, point-dumping for a winning partner, defender
  low-trump hand-off, endgame kitty protection — are real signal) and layer the learned value
  + beliefs on top, like an engine's opening book + search.
- **Leaf ensemble:** blend Oracle-Q, Sage/Athena value, and the static eval at the leaf
  (the existing `(1-w)·static + w·value` blend generalizes to a 3-way mix).

The composition is **Phase 5**; it's expected to be the strongest bot of all, *because* it
stacks the independently-verified gains.

---

## 4. How win rates get measured (the comparison)

All claims go through the existing paired harness — **no new methodology, just new contestants.**

1. **Freeze a reference.** Pin a search budget (`SHENGJI_BOT_BUDGET_MS`) and fixed seed sets;
   record current Easy/Expert/Enoch/Omniscient paired win-rates as the frozen baseline.
2. **Head-to-head, paired-on-mirrored-deck.** For each new bot, `run_paired_ab` vs **Easy,
   Expert, Enoch, Omniscient** → win-rate + bootstrap CI + MDE. Because search-less and
   modest-budget runs are cheap, use **large pair counts (≥1,000 deck-pairs)** for tight CIs.
3. **Full round-robin matrix.** Extend `core/examples/tournament.rs` to include the new
   contestants → a **win-rate matrix + avg point-margin matrix + implied (Elo-like) ladder**
   across *all* bots, new and old.
4. **Serve-as-deployed via `PlayBrain::Search`/new tiers.** New search configs (value weight,
   belief sampler, PUCT) A/B *without* new enum variants where possible (the harness's
   intended extension pattern); a shipped tier gets a `BotDifficulty` variant + `Knobs` +
   `choose_play` dispatch.
5. **Guardrails every run:** the **honesty gate**
   (`e2e_game_no_hidden_card_leakage`) must stay green for every honest bot, and the
   **baseline_gate** floors must not regress. Compare **CIs, never byte-diffs** (HashMap-order
   non-reproducibility is documented).
6. **Playing-style writeup:** each bot's style is described from (a) its design and (b)
   **logged self-play traces** (lead/follow/kitty/bid tendencies, trump economy, point-dump
   timing) — concrete, not just adjectives.

---

## 5. Proposed execution sequence (phased, resumable, checkpointed)

- **Phase 0 — Setup & baseline (non-destructive, ~½ day).** venv + CPU/MPS torch; build
  release binaries; **smoke-test + time** data-gen and `paired_eval` (calibrates all later
  estimates); freeze the current-ladder baseline numbers. *(Safe to run immediately on
  approval, independent of scope.)*
- **Phase 1 — Sage (value head).** Highest ROI, exercises the whole pipeline end-to-end via
  `run_value_pipeline.sh`. Ship if it gates over Expert.
- **Phase 2 — Oracle (DMC).** Most novel; biggest code lift; search-free serve path.
- **Phase 3 — Athena (self-play actor-critic + ExIt).** Builds on Phase-1 infra.
- **Phase 4 — Seer (belief net).** Layers onto the best policy/value net from 1–3.
- **Phase 5 — Grandmaster (the hybrid).** Compose the components that beat their baseline in
  Phases 1–4 into one agent (Seer beliefs → Athena/Sage policy + value in the search, Oracle-Q
  as the System-1 fast path), and paired-measure the *combined* lift vs each part alone. This
  is expected to be the strongest bot — it stacks the independently-verified gains. Includes
  the fast/slow latency allocation and the optional Enoch-playbook prior blend.
- **Phase 6 — Grand tournament + style writeup + recommendation** on what to ship (replace
  Expert's net / add a new lobby tier / keep as benchmark). Full win-rate + margin matrix with
  CIs across *all* bots (new + existing), styles described from logged self-play traces.

Each phase ends with a **paired A/B gate** and a checkpoint; the pipeline is resumable so
multi-day runs survive interruives. Phases 1–4 are largely independent and can be parallelized
across machines/shards.

---

## 6. Open risks / honest caveats

- **Aliasing floor (A):** Sage's *policy* is still cloned from an unidentifiable perfect-info
  teacher — the value head helps the leaf, not the prior. B/C escape this; that's their point.
- **Serve-time latency:** any net in the per-leaf loop must stay small. Big encoders are
  confined to root-prior/belief roles. Measured, not assumed.
- **Self-play collapse / cycling (B, C):** mitigated by the league + paired-gating (only keep
  a generation if it beats the last by CI).
- **DMC variance (C):** needs lots of self-play; this is exactly where the CPU box pays off.
- **Non-reproducibility:** all numbers are CIs over distributions; gates carry a 4–6pp buffer.
- **`mechanics` correctness (E):** the enumerator is correctness-critical and warrants human
  review — hence "optional/last."
