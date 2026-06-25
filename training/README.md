# Expert tier training (distillation → ONNX → tract)

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
# recorded decision, so a small budget keeps it fast; bump it for a stronger
# (slower) teacher.
GEN_GAMES=800 SHENGJI_BOT_BUDGET_MS=8 \
  cargo run --release --example gen_training_data
# -> writes training/data.csv  (group, f0..f27, label)
```

Each row is one *candidate*; rows sharing a `group` are the candidates of a
single decision, and exactly one of them has `label == 1` (the teacher's pick).
The feature encoding is defined once in `core/src/bot/expert.rs`
(`candidate_features`, `FEATURE_DIM`) and reused by both the exporter and the
Rust inference path, so training and serving can never drift.

### 2. Train + export ONNX (Python)

```sh
cd training
python3 -m venv .venv && . .venv/bin/activate     # optional
pip install -r requirements.txt                   # torch + onnx + numpy (large)
python train_expert.py --data data.csv \
  --out ../core/src/bot/expert_model.onnx --epochs 60
```

The model is a tiny 2-hidden-layer MLP (`FEATURE_DIM → 64 → 64 → 1`) trained with
a softmax cross-entropy over each decision's candidates (the teacher's candidate
should get the top logit). The headline metric is **top-1 accuracy** = fraction
of decisions where the net's argmax candidate equals the teacher's pick; compare
it against the printed random-guess baseline (≈ 1 / avg-candidates).

It exports `expert_model.onnx` with input `x:[N,28]` → output `[N,1]` (opset 13).

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

- More data: raise `GEN_GAMES` (e.g. 2000–5000).
- Stronger teacher labels: raise `SHENGJI_BOT_BUDGET_MS` during generation so the
  Omniscient search is deeper (slower).
- Bigger/longer net: `--hidden 128 --epochs 150`.
- The feature set is deliberately compact; richer features (per-suit voids
  inferred from play, remaining-trump counts, seat position relative to the
  landlord) would raise the ceiling — add them to `candidate_features` in
  `expert.rs` (bump `FEATURE_DIM`) and regenerate + retrain.
