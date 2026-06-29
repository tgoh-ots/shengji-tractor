# Bot strength — committed evaluation baseline

> Measured 2026-06 on the paired-on-mirrored-deck harness
> (`core/src/bot/harness.rs`). This is the reference the non-inferiority gate
> (`core/tests/baseline_gate.rs`) and the `paired_eval` example check against.
> Companion to `docs/bot-training-roadmap.md` (the "measurement substrate").

## How strength is measured now

`harness::run_paired_ab` plays each deck seed in **both** orientations (A as the
landlord team, then B as the landlord team, on the **identical** deal). Pairing the
deal out of the comparison cancels the dominant deal-luck variance, so a given
number of hands resolves a much smaller strength difference than the legacy
unpaired benchmarks (which alternated the landlord across *different* seeds). Each
matchup reports:

- win-rate with a **Wilson** 95% interval (over individual hands) and a **paired
  bootstrap** 95% interval (resampling over *decks* — pairing-aware);
- mean paired point margin;
- the 95% **minimum detectable effect (MDE)** on win-rate, so "no difference" is
  distinguishable from "underpowered".

Run it yourself:

```bash
# Fast, stable, search-less matchups (Easy knobs, NEW-vs-LEGACY heuristic):
cargo run --release --example paired_eval -- 400 0x5EED fast

# Search/net matchups (honor SHENGJI_BOT_BUDGET_MS; slower):
SHENGJI_BOT_BUDGET_MS=400 cargo run --release --example paired_eval -- 200 0x5EED search

# A/B a freshly-trained net WITHOUT rebuilding (runtime model override):
SHENGJI_EXPERT_MODEL_PATH=/path/to/candidate.onnx \
  cargo run --release --example paired_eval -- 200 0x5EED search
```

## Committed baselines (200 pairs = 400 hands, `base_seed = 0x5EED`)

Search-less (budget-independent → fast & stable; these back the CI gate):

| Matchup (A vs B) | A win-rate | paired-bootstrap95 | paired margin | gate floor |
|---|---|---|---|---|
| Easy@NEW (ε .06 / T 1.1) vs Easy@OLD (ε .28 / T 3.5) | ~54% | ~[50.5, 58.5] | ~+4 pts/hand | **A > 0.50** |
| Heuristic NEW vs LEGACY (greedy, bid Expert / bury Easy) | ~57% | ~[53.5, 60.0] | ~+6.7 pts/hand | **A > 0.51** |

Search/net (budget-dependent → run in `--release`; coarse, not in fast CI):

| Matchup | notes |
|---|---|
| Expert(search) vs Easy | release gate (`#[ignore]`), floor "A ≥ 0.50"; run via `--ignored` |
| Enoch(search) vs Expert(search) | tracked via `paired_eval ... search` / `enoch_benchmark` |
| Expert net A/B (embedded vs candidate) | `SHENGJI_EXPERT_MODEL_PATH` + `expert_ab` / `paired_eval` |

## The gate (`core/tests/baseline_gate.rs`)

Fast CI runs the two **search-less** assertions above (each ~3–4 s in debug). The
floors sit ~4–6pp below the measured baselines, so:

- a green run is *signal* (a real regression of >~4–6pp on the shared scorer / Easy
  knobs fails the gate);
- run-to-run jitter does **not** flake it (see the nondeterminism note below).

The search/net relationship is the **release-only** `#[ignore]`d
`baseline_expert_beats_easy_search` (debug search is too starved/noisy to gate
tightly — the "tight gate on budget-independent tiers, coarse net on search tiers"
split from the roadmap):

```bash
SHENGJI_BOT_BUDGET_MS=60 cargo test -p shengji-core --release \
    --test baseline_gate -- --ignored
```

## ⚠️ Reproducibility caveat — `HashMap` iteration order

These benchmarks are **not byte-reproducible run-to-run**, even the search-less
ones: Rust's `std::collections::HashMap` seeds its iteration order per process, and
that order leaks into tie-breaks somewhere on the candidate-generation / scoring
path. Empirically a fixed 40-game search-less run jitters by ~±1–2 wins; a 200-pair
paired win-rate jitters by ~±1pp. The paired bootstrap CI already quantifies the
combined (deal + ordering) noise, and the gate floors are set well clear of it.

A future hardening (out of scope for the measurement substrate) would make the
benchmarks fully reproducible by removing that ordering dependence (e.g. a
deterministic candidate ordering / `BTreeMap` in the hot path). Until then: compare
**distributions / CIs**, not exact win counts, and never gate on a byte-diff.

## Regenerating these numbers

Re-run the `paired_eval` commands above and update the table. If you change the
shared scorer (`heuristics::score_lead`/`score_follow`) or the Easy knobs
(`bot::policy::Knobs`), re-measure and, if the new baseline genuinely moved, update
both this file and the floors in `core/tests/baseline_gate.rs` in the same change.
