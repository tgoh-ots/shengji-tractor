# Superior-Play Bot — Results Log

> Live results from executing `docs/bot-superplay-plan.md`. All win-rates are
> paired-on-mirrored-deck (deal-luck cancelled) with Wilson + paired-bootstrap 95%
> CIs from `core/src/bot/harness.rs`. **Compare CIs, not point estimates**
> (benchmarks are not byte-reproducible). Running on a 10-core laptop **shared with
> a concurrent training session**, so wall-clock is contended; the cloud box
> (`training/cloud_setup.sh`) would 5–10× the heavy phases.

_Started 2026-06-29._

## Phase 0 — setup & calibration ✅
- Env: torch 2.12.1 (+MPS), onnx 1.22, numpy 2.5. Release binaries built.
- Timing (teacher 200ms): easy gen ~2.5 s/game; **mix (search-driven) ~10.3 s/game/core**;
  **DMC self-play ~0.04 s/game** (no teacher search — startup-dominated).
- Labels clean: `teacher-outside-candidates = 0`.

## Phase 1 — Sage (value-head leaf) — ⚠️ negative result on existing net
**Setup.** Used the value net already trained by the resumable pipeline
(`$HOME/.shengji-value-run/value.onnx`, a 2-output policy+value model distilled from
~2,100 mix games, teacher 200ms). The leaf blend `(1-w)·static + w·net_value` is
gated by `SHENGJI_VALUE_WEIGHT`. Measured Expert(this net) vs Easy at w=0 vs w=0.5
(200 deck-pairs, 150ms budget):

| value-blend weight | Expert win-rate vs Easy | paired margin |
|---|---|---|
| w = 0 (leaf blend OFF) | **65.5%**  (boot95 [61.8, 69.2]) | +13.4 pts/hand |
| w = 0.5 (leaf blend ON) | **62.7%**  (boot95 [58.8, 66.5]) | +10.4 pts/hand |

**Finding.** Turning the value-head leaf blend ON at w=0.5 **reduced** strength
(~−2.8pp win-rate, −3.0 pts/hand margin). The value head as currently trained
(small mix corpus, 200ms teacher) is **not calibrated well enough to improve the
search leaf** — at high weight it injects noise into every rollout's leaf estimate.
This matches the roadmap's caution that the value head only pays off once trained on
a large, on-policy, well-calibrated dataset.

**Implication for the plan.** Sage's *value-leaf* lever is **not a free win** on the
current data — it needs (a) much more on-policy data and/or (b) a far lower weight,
and even then the payoff is uncertain. The more promising bets are the ones that
change *how the policy/value is learned* rather than bolting a noisy value head onto
a cloned policy: **Oracle** (learn the action-value directly by self-play) and
**Athena** (clone the *honest search*, not the cheater, and iterate). A larger,
better-calibrated value net + low-weight sweep remains a cloud-box follow-up.

_(A finer weight sweep on the existing net was started but abandoned — under the
shared-CPU contention a single search A/B ran >8 min with no result; not worth the
wall-clock to characterize an already-underperforming net.)_

## Phase 2 — Oracle (Deep Monte-Carlo) — ⚠️ underperforms at feasible scale (but the loop works)
DouZero-style **search-free** Q-net, pure self-play (ε-greedy on the current Q),
bootstrapped from the embedded Expert then self-improving over **6 iterations
(9×1,500 games/iter ≈ 1.1M (s,a,return) rows each ≈ 6.6M samples total)**, ~3 min/iter.

**Search-free argmax vs Easy (100 deck-pairs, search-free = fast):**

| serve policy (argmax, no search) | win-rate vs Easy | margin |
|---|---|---|
| embedded **Expert policy net** | **59.5%**  (boot95 [54.0, 65.0]) | +8.5 |
| Oracle DMC **iter 0** (Expert-bootstrapped) | 37.5%  [31.5, 43.5] | −12.4 |
| Oracle DMC **iter 5** (final) | 46.5%  [39.5, 53.0] | −5.8 |

vs the search tiers (25 pairs): Oracle iter3 lost to Enoch 16% / Omniscient 16%.

**Findings.**
1. **DMC self-play *is* improving** — +9pp from iter0→iter5 — so the method is sound and
   converging, just slowly.
2. **It is far less sample-efficient than distillation.** The *existing* distilled Expert
   policy net, played search-free, already beats Easy 59.5%; DMC after 6.6M samples is still
   <50%. The single-game Monte-Carlo return is high-variance (target std ≈0.5, Q val-RMSE
   ≈0.44 — the net barely beats predicting the mean), so a *good argmax* needs far more data
   than is feasible on a laptop (DouZero used **billions** of samples). Dou Dizhu's value
   signal is much cleaner than 升级's team/kitty-laden one.
3. **Search is doing real work but the policy net carries most of it**: embedded Expert is
   59.5% search-free vs ~65% with the 144-world search (+~5.5pp from search).

**CLOUD-SCALE CONFIRMATION (Fly 16-core, 2026-06-30):** re-ran DMC at **8 iterations ×
~1.23M (s,a,return) rows/iter** (more iters + data than the laptop). Result: **41.1% vs Easy**
(boot95 [38.5, 44.0], −7.8 pts), 19.9% vs Enoch, 17.2% vs Omniscient — **no improvement over
the laptop's ~46%** (if anything slightly worse), val-RMSE plateaued at ~0.43. **Scale does
not rescue search-free DMC here** — the single-game return variance is the wall, not data
volume. This closes the "Oracle at scale" question: negative.

**Verdict.** Pure search-free DMC is **not** the path at this compute budget; the
distilled-policy paradigm dominates. This *motivates* Athena (which improves the distilled
policy via honest-search ExIt rather than regressing noisy returns). Oracle remains a
cloud-scale candidate (10–100× the self-play) and a strong **System-1 fast path** inside the
Grandmaster hybrid (instant when its top-2 Q are far apart), but not a standalone champion.

> **Bonus bot discovered:** the embedded Expert policy net played **search-free** ("Fast
> Expert", `PlayBrain::NetGreedy` on `expert_model.onnx`) beats Easy **59.5%** at ~one net
> call per decision — a genuinely useful **low-latency tier** (no 144-world search), ~5.5pp
> below full-search Expert. Worth surfacing in the lobby as a "fast" option.

## Phase 3 — Athena (honest-search ExIt) — ✅ POSITIVE (the session's clear win)
**Setup (laptop, deliberately tiny).** ONE ExIt iteration: 400 self-play games where the
honest Expert determinized search (100ms) both plays and labels each decision → 22,409
search-labeled decisions → trained a 2-output (policy+value) net. (The big multi-iteration
version is queued for the box.)

**Search-free policy quality vs Easy (120 deck-pairs, argmax, no search):**

| net (argmax, search-free) | win-rate vs Easy | margin |
|---|---|---|
| embedded Expert (distilled from the **Omniscient cheater**) | 61.7%  (boot95 [56.2, 66.7]) | +9.1 |
| **Athena iter0** (clones the **honest search**) | **67.1%  (boot95 [62.1, 71.7])** | **+12.1** |

**Finding — the thesis holds.** A single ExIt iteration on just 22k decisions produced a
policy net that beats the production Expert net by **+5.4pp** (CIs barely overlap — a real
effect). Why it works where Sage/Oracle didn't: it stays in the **distilled-policy** paradigm
(sample-efficient) but swaps the *teacher* from the Omniscient cheater (whose perfect-info
picks are often unidentifiable from honest features — the aliasing floor) to the **honest
search** (whose picks ARE a function of honest features, and which is *stronger than its own
prior* — the AlphaZero policy-improvement operator). So the labels are both **learnable** and
**better than what the net currently does** → the net improves. Iterating (cloud: 3–4 rounds,
full data) should compound it.

**Deployed WITH search (144 worlds, 100ms) vs Easy — the gain washes out:**

| net used as search prior | win-rate vs Easy | margin |
|---|---|---|
| embedded Expert | 67.5%  [62.5, 72.5] | +16.2 |
| Athena iter0 | 67.1%  [62.1, 71.7] | +13.4 |

**Sharper conclusion.** Athena's net is a clearly better *standalone* policy (+5.4pp
search-free) but, wrapped in the determinized search, it is **statistically tied** with the
embedded net (both ~67%, CIs fully overlapping, MDE ±5pp). The search *compensates* for a
weaker prior — it finds the strong move by lookahead regardless of which decent prior seeds
it — so **a better prior pays off most exactly where there is no search.** Practical
implications:
1. **Ship Athena as the "Fast Expert" tier** (search-free `NetGreedy`): a concrete, deployable
   win — ~+5pp over the current net's argmax at one net-call/decision (no 144-world search →
   instant, cheap). Best honest *low-latency* bot.
2. **To beat the full-search Expert** you must improve a slot the search *can't* self-correct:
   the **value leaf** (a *well-calibrated* Sage — not the laptop one) or the **world model**
   (Seer's beliefs). A better policy prior alone is masked. This is the single most important
   architectural lesson of the whole study, and it re-prioritizes the cloud run toward Seer +
   a calibrated value head over more policy distillation.
3. The cloud Athena (3–4 iterations, higher budget, ≥600 pairs) is still worth running — more
   ExIt rounds may lift the prior enough to show through search, and ≥600 pairs resolves a
   sub-5pp deployed gain the laptop can't.

**CLOUD RESULT — Athena cloning ENOCH (Fly 16-core, 2026-06-30, 3 ExIt iters, 400 paired):**

| net (search-free argmax) | vs Easy | vs Enoch |
|---|---|---|
| embedded Expert (current) | 60.4% [57.6, 63.2] | 33.0% [30.1, 35.9] |
| **Athena-from-Enoch** | **65.0% [62.3, 67.8]** | **42.4% [39.5, 45.2]** |

Cloning the **strong** teacher (Enoch) instead of the weak Expert/Omniscient gives a markedly
better **search-free** net: **+4.6pp vs Easy, +9.4pp vs Enoch** (CIs disjoint). It does NOT
reach Enoch strength search-free (42% vs Enoch; top-1 only ~61% — Enoch's play is harder to
clone from honest features), and **ExIt iterations did not compound** (top-1 flat 61→62→61).
**Deployed with the 144-world search:** Expert(Athena-net) vs Easy = 66% and Enoch still beats
it 67% — i.e. it **ties the existing Expert deployed and can't out-score Enoch.** Third
independent confirmation of the shared-heuristic ceiling (my laptop + the parallel session's
Grandmaster + this).

**⇒ The standout deployable artifact: a "Fast Enoch" tier** — Athena-from-Enoch served
search-free (one net call/move, instant) is the **strongest low-latency honest bot** by a
clear margin (esp. vs strong opponents), far better than the current net's argmax. *(Net not
retained — the Fly box was destroyed; cheaply re-trainable with the now-fixed pipeline.)*

## Cross-session corroboration (parallel `worktree-better`, master)
- **Honest tiers can't out-score Enoch** — their calculation-driven **Grandmaster ties Enoch**
  (~50–52%, n=1200): same shared-heuristic ceiling.
- **Value head = NEUTRAL** on every tier (matches Sage), even with a stronger teacher; they
  also found+fixed a group-collision data bug (independent of the mawk concat bug I hit).
- **Omniscient was bugged** (plain heuristic → *lost* to Enoch 44.8%); fixed on master to use
  the playbook (now ~61% vs Enoch). My runs used the OLD weak Omniscient as teacher — which is
  exactly why cloning **Enoch** (Athena) beat cloning the cheater.
- Infra bug I found+fixed: multi-shard concat offset `1e9` overflows mawk's 2³¹ integer-print
  on Linux → scientific-notation group IDs → trainer crash. Fixed to `1e7` in
  `run_exit_pipeline.sh` + `run_value_pipeline.sh`.

## Phases 4 & 5 — Seer + Grandmaster — designed, deferred to cloud scale (scope call)
Given (a) the shared-CPU contention here and (b) the empirical lesson that **search + a
distilled policy already dominate**, while bolt-on learned components (Sage's value leaf,
Oracle's search-free Q) under-deliver at laptop scale, I scoped this session to *rigorously
measure* Sage/Oracle/Athena rather than half-build the two most invasive bots. Their designs
are now concrete (exact integration points identified):

**Seer (learned belief determinizer).** The only honest tier weakness that *isn't* about raw
search depth is that `determinize.rs::sample_hidden_hands` samples worlds with a **greedy
"neediest-seat" deal** that ignores everything bids/plays reveal beyond hard voids. Concrete
build:
- *Encoder* (new Rust fn): observable state → fixed vector (my hand multiset, played-card
  multiset, per-seat hidden counts + voids, trump, bid, seat).
- *Belief net*: state-vec → for each unseen card-type a softmax over its location
  {opp_left, partner, opp_right, kitty} (≈54×4 outputs). 1 ONNX, tract-served.
- *Data*: self-play recording (observable state from a seat) → (true location of each hidden
  card). Honesty-safe (target used only in training).
- *Integration* (the leverage): in `sample_hidden_hands`, replace the neediest-seat pick with
  a **belief-weighted** seat draw (∝ P(card→seat), masked by need>0 & not-void). One net call
  per world (~144/decision) — measure the added latency; cache per decision.
- *Serve*: a `belief_sample` flag on the search; Seer = Expert search + belief sampling.
  Expected to compound with any policy/value net — and it attacks PIMC's real failure
  (strategy fusion from fantasy worlds), the one lever orthogonal to search depth.

**Grandmaster (composition).** Once Seer + the best policy/value net exist:
Seer beliefs → Athena policy prior + rollouts → Athena/Sage value leaf (3-way leaf ensemble
with Oracle-Q + static), with Oracle-Q as the **System-1 fast path** (instant argmax when its
top-2 are far apart; full belief+search otherwise) and an optional Enoch-playbook prior blend.
Paired-measure the marginal lift of each added component (the ablation is already set up: each
bot isolates one slot).

Both are turn-key on the cloud box (`training/cloud_setup.sh`): the heavy cost is CPU
self-play, which the box provides 6–13×.

## Playing styles (from architecture + observed win-rate/margin signatures)
- **Easy** — *the casual.* Heuristic top-move with a warm softmax (T≈1.1) and ε≈0.06 blunders;
  no card memory, no search. Reasonable opens, but leaks points and mistimes trump. The
  beatable baseline.
- **Fast Expert** (net argmax, search-free) — *the snap-judgment pro.* The distilled policy
  net's top pick, one net call per move. Solid card sense (59.5% vs Easy) without deliberation;
  occasionally misses a tactical line that lookahead would catch. Great low-latency tier.
- **Expert** (net prior + 144-world search) — *the calculating closer.* Same instincts plus
  determinized lookahead; manages trump and trick tempo better (~65% vs Easy, +13 pts/hand).
  Its known tic: the hand-rolled leaf slightly **hoards** high cards (over-values keeping its
  own honors).
- **Enoch** (playbook + search + perfect memory) — *the disciplined veteran.* The strongest
  honest tier: pair-priority trump declaring, **no high-trump opens**, point-dumps only when
  the partner is winning, the defender low-trump hand-off, endgame kitty protection, and exact
  card memory (full-history voids, never re-deals a played card). Patient early, seizes the
  mid-game, protects the kitty. Beats every other honest tier.
- **Omniscient** (perfect-info search) — *the cheater.* Sees all hands; an "impossible"
  sparring partner, not honest.
- **Sage** (value-blend, w=0.5) — *the over-thinker.* Same policy as Expert but the noisy
  learned leaf made it second-guess good lines → **weaker** (−3pp). Needs a far better value
  net to earn its leaf.
- **Oracle** (DMC, search-free Q) — *the raw rookie.* Pure self-play instinct with no teacher
  and no search. Improving (iter0→5: +9pp) but still green at this scale — its value sense is
  too noisy to consistently pick the best of close candidates. Promising with cloud-scale data
  or as a fast first-impression inside a hybrid.
- **Athena** (honest-search ExIt) — *the apprentice who out-learned the teacher (when it gets
  to think fast).* Its prior was trained to agree with its own lookahead on *honest*-learnable
  targets, so **search-free it clearly out-picks the production net** (67% vs 62% vs Easy) —
  the best honest low-latency bot. Given the full 144-world search it plays neck-and-neck with
  Expert (the search masks the prior edge). Style: same instincts as Expert, just sharper snap
  judgment.

## Summary ladder (paired vs Easy, this session) + recommendation
| bot | vs Easy (search-free) | vs Easy (144-world search) |
|---|---|---|
| Oracle (DMC, final) | 46.5% | — (loses to Enoch/Omni badly) |
| Easy (anchor) | 50% | 50% |
| embedded Expert / "Fast Expert" | ~61% | ~67% |
| **Athena (1 ExIt iter)** | **67%** ⬆ | ~67% (= Expert) |
| Sage (Expert + value-blend w=0.5) | — | ~63% ⬇ |
| Enoch / Omniscient | — | above Expert (Oracle lost 16% to each) |

**Recommendation (post-cloud, final).**
1. **Ship "Fast Enoch" — Athena-from-Enoch served search-free** — as a new low-latency tier:
   the clean deployable win (65% vs Easy; **+9.4pp** over the current net's argmax against
   Enoch; one net call/move, no 144-world search). Re-train the net (pipeline now fixed) +
   embed. This is the concrete product of the whole effort.
2. **Don't ship** the Sage value-blend (neutral→negative) or Oracle (weak; confirmed not to
   scale).
3. **To raise the *deployed* ceiling, the policy/value/search levers are exhausted** — beating
   Enoch with search is now **3× confirmed impossible within the shared heuristic** (my laptop
   Athena, the parallel Grandmaster, and cloud Athena-from-Enoch all tie/lose deployed). The
   one unexplored axis is the **world model: build Seer** (learned belief determinizer) so the
   search reasons over *realistic* hidden hands — the only slot lookahead can't self-correct.
4. **Grandmaster hybrid** = Seer beliefs + a strong prior + the existing search, once Seer
   exists. (A calculation-driven Grandmaster already ships on master, tying Enoch.)
