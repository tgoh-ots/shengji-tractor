# Expert tier training (distillation → ONNX → tract)

> ⚠️ **PARTIALLY SUPERSEDED — `CLAUDE.md` (the "Retrain the Expert net" section) is
> the source of truth for the CURRENT pipeline.** This file is accurate on the
> distillation *concept* but lags on specifics. What changed since:
> - The net is now **multi-task**: a policy head **and** an optional `tanh` **VALUE
>   head**, exported as a **2-output ONNX** (`score`, `value`) when value targets
>   are present. The CSV is now `group, f0..f35, label, value` (a `value` column).
> - The leaf-eval **value blend** is gated by `SHENGJI_VALUE_WEIGHT` (default 0=OFF);
>   inference reads ONNX `output[1]`, scaled by `expert::VALUE_NORM`.
> - **DAgger**: `GEN_BEHAVIOUR` (easy|expert|enoch|mix) picks the data-gen state
>   distribution; `GEN_TEACHER_BUDGET_MS` (default 400) sets label quality;
>   `GEN_SEED` shards data-gen.
> - The whole generate→train→A/B run is automated + **resumable** by
>   `training/run_value_pipeline.sh`.
> - Build/run with `cargo +1.92.0` (deps need rustc ≥ 1.87).
> - The tiers are `Easy < Expert <= Enoch < Omniscient` (the old **"Hard" tier was
>   removed** — ignore any "Hard" mention below).
> Treat the architecture/CSV/tier/invocation details below as historical.

The **Expert** bot difficulty is a small learned neural net that scores each
legal candidate play and picks the best. It is trained by **behavioral cloning /
distillation** of the **Omniscient** (perfect-information) teacher: for every
play decision we record, per legal candidate, an *honest* feature vector plus a
label = 1 on the candidate the teacher (which sees every hand) chose. The net
therefore learns to approximate perfect-information play from the *honest*
redacted view it will actually have at serving time. It never reads hidden hands.

The whole pipeline is pure-Rust at inference (via `tract-onnx`), so it builds in
the musl deploy image with no `onnxruntime`/`ort` C dependency.

## Pipeline

```
                 self-play (Rust)            PyTorch                 tract (Rust)
gen_training_data ─────────────► data.csv ──► train_expert.py ──► expert_model.onnx ──► bot::expert
   (Omniscient teacher labels,                (listwise softmax    (opset 13, [N,D]→[N,1])
    HONEST features)                            cross-entropy)
```

### 1. Generate training data (Rust)

```sh
# From the repo root. The Omniscient teacher runs a perfect-info search at every
# recorded decision. A LARGER budget gives stronger (less noisy) labels; the
# per-game wall-clock barely grows because the perfect-info search usually
# converges well before the budget. The shipped model was distilled from
# 5000 games at a 500ms teacher budget.
GEN_GAMES=5000 SHENGJI_BOT_BUDGET_MS=500 \
  cargo run --release --example gen_training_data
# -> writes training/data.csv  (group, f0..f35, label)
```

Each row is one *candidate*; rows sharing a `group` are the candidates of a
single decision, and exactly one of them has `label == 1` (the teacher's pick).
The feature encoding is defined once in `core/src/bot/expert.rs`
(`candidate_features`, `FEATURE_DIM = 36`) and reused by both the exporter and the
Rust inference path, so training and serving can never drift.

The 36-feature encoding mixes a compact candidate/trick/hand summary (indices
0–27) with **honest card-memory** features (28–35) derived from
`Knowledge::from_play_view`: the fraction of trumps still unseen, my share of the
live trumps, whether the opponents still to act are known void in the led suit,
points left unseen, my seat position, whether the candidate is an uncatchable
top of its suit, and overall game progress. These were the lever that pushed the
distilled net clearly above Hard — they are all computed from the redacted view
+ public play history (never hidden hands), preserving the honesty invariant.

### 2. Train + export ONNX (Python)

```sh
cd training
python3 -m venv .venv && . .venv/bin/activate     # recommended
pip install -r requirements.txt                   # torch + onnx + numpy (large)
python train_expert.py --data data.csv \
  --out ../core/src/bot/expert_model.onnx          # defaults: hidden 128, 200 epochs
```

The model is a 3-hidden-layer MLP (`FEATURE_DIM → 128 → 128 → 64 → 1`) with ReLU
+ dropout, trained with a softmax cross-entropy over each decision's candidates
(the teacher's candidate should get the top logit). It uses a cosine LR schedule
and **early stopping** on val top-1 accuracy (`--patience`, default 25). The
headline metric is **top-1 accuracy** = fraction of decisions where the net's
argmax candidate equals the teacher's pick; compare it against the printed
random-guess baseline (≈ 1 / avg-candidates).

Dropout is a no-op in eval/export, so the exported graph is just Gemm/ReLU and
loads cleanly in tract. It exports `expert_model.onnx` with input `x:[N,36]` →
output `[N,1]` (opset 13, legacy TorchScript exporter via `dynamo=False`).

### 3. Use it (Rust)

`core/src/bot/expert.rs` `include_bytes!`s `expert_model.onnx`, parses it with
`tract-onnx`, and scores all legal candidates of a decision in one inference
call. If the model can't load/run (e.g. it's still the committed placeholder),
the Expert tier transparently falls back to the **Hard** determinized search, so
Expert is never illegal/None.

### 4. Evaluate the ladder

```sh
cargo run --release --example eval     # prints Easy/Hard/Expert/Omniscient ladder
```

## Training longer / stronger

- More data: raise `GEN_GAMES` (the shipped model used 5000; 8000+ helps a bit
  more with diminishing returns).
- Stronger teacher labels: raise `SHENGJI_BOT_BUDGET_MS` during generation so the
  Omniscient search is deeper. This is nearly free in wall-clock because the
  perfect-info search usually converges before the budget.
- Bigger/longer net: `--hidden 192 --epochs 300` (early stopping caps the cost).
- Richer honest features are the highest-leverage knob and already include the
  remaining-trump counts, per-seat voids of the seats still to act, seat
  position, and uncatchable-top detection (indices 28–35 of
  `candidate_features`). Further honest signals (e.g. per-suit remaining high
  cards, landlord-relative seat) can be added there — bump `FEATURE_DIM`, keep
  the Python `FEATURE_DIM` in sync, and regenerate + retrain.
