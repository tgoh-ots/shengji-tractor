# Four-player bot exploration closure — 2026-07-11

> **Status:** terminal human decision; no active bot-strength campaign
> **Scope:** four-player, two-deck Tractor with fixed opposite-seat partnerships
> **Excluded from scoring:** Finding Friends and every other player-count or
> dynamic-team variant
> **Result:** no candidate proved superior to Enoch-0
> **Authority:** this disposition supersedes every open task, “next step,”
> continuation loop, and unspent-seed instruction in the 2026-07-02
> strongest-bot roadmap and its later Stage 2 research branches

## Decision

The exploration ends here. Enoch-0 remains both the permanent scientific
control and the strongest validated honest yardstick from this program. There
is no Enoch-1, no selected or frozen Grandmaster v2 teacher, no Expert v2
promotion, and no deployment authorization.

This is a program stop, not a mathematical claim that no stronger Shengji bot
can exist. It means the bounded avenues accepted into this exploration are
terminal, the remaining speculative avenues are not active work, and no agent
should resume this campaign from an old plan, branch, worktree, seed registry,
or unchecked roadmap item.

## Closest game-level evidence

Positive level utility favors the named candidate. Intervals are paired
bootstrap 95% intervals unless otherwise stated.

| Avenue | Scored comparison | Result | Terminal disposition |
|---|---|---:|---|
| Week 1 four-change Enoch candidate | 300-pair qualification against Enoch-0 | `-0.006667 [-0.068333, +0.053333]` | Qualification failed; the candidate was rejected as `combination-regressed`. |
| Week 1 four-change Enoch candidate | Separate 800-pair development screen | `+0.010625 [-0.023750, +0.046250]` | Directionally positive but insufficient because both predeclared stages had to pass. |
| Grandmaster-0 | 300 fixed-work pairs against Enoch-0 | `+0.00500 [-0.05833, +0.07000]`; point margin `-0.34167` | Compatible with neutral, not superior. |
| `paired-racing-v1` | 300 equal-work pairs against flat Grandmaster-0 | `+0.03000 [-0.03167, +0.09000]` | Largest positive game-level point estimate, but inconclusive and never compared with Enoch-0. |
| `paired-racing-v2` | 800 equal-wall pairs against flat Grandmaster-0 | `-0.124375 [-0.164375, -0.086875]`; point margin `-5.15625` | Failed admission and was retired before any Enoch-0 strength comparison. |
| `enoch-bid-ownership-v1` | 300 fixed-work pairs against Enoch-0 | `+0.006667`, lower 95% bound `-0.011667`; point margin `+0.275` | Passed only the permissive development screen. The admitted product pilot attempted 22 pairs, completed zero, failed after launch/restart, and terminally retired the slot. |

None of these results establishes superiority. The strongest positive estimates
either crossed zero, were measured against Grandmaster rather than Enoch, or
failed a separate required stage.

## Diagnostic evidence that did not become strength-eligible

The outcome-blind diagnostics were useful for rejecting weak ideas cheaply,
but they are not complete-hand strength evidence.

| Avenue | Best-looking signal | Why it stopped |
|---|---:|---|
| Tiny exact continuation v2 | Mean regret delta `-0.0007324`, interval `[-0.0023926, +0.0004395]` where negative favors the candidate | Untouched confirmation crossed zero. |
| Landlord-follow HEH | Discovery mean `-0.945833` | Untouched confirmation reversed to `+0.518229`, interval `[-0.601042, +1.666406]`. |
| Paired-confidence selection | Averaged reference mean `-0.180469` | The second reference was `+0.278125` and the landlord-role guardrail was `+0.184375`. |
| Role/action adaptive width | Mean regret delta `-0.013787` | Held-out interval `[-0.121140, +0.091912]` crossed zero. |
| Phase-aware flat breadth | Reference A `-0.082813` | Independent reference B was `+0.171354`; the conjunctive discovery gate rejected it. |
| Attacker-lead prior `lambda=0.10` | Historical replay motivation only | Fresh discovery was harmful on both references: pooled means `+0.059180` and `+0.066895`. |
| Global prior `lambda=0.40` | Development-selected replay `+0.939` normalized percentage points | Interval `[-0.537, +2.550]` crossed zero and the role/action split was mixed; no strength candidate was registered. |

Other rollout, localized-breadth, team-control, kitty-set, bid-policy,
root/partner/opponent, exact-phase, greedy-rollout, and final-prior variants were
mixed, materially harmful, operationally invalidated before strength, or failed
their frozen discovery/confirmation rule. No rejected diagnostic authorizes a
new candidate by reinterpretation.

## Multi-avenue terminal state

The four-player multi-avenue registry created 12 candidate slots, numbered
0–11. Every one is terminal:

- slots 0–6 and 9 were retired before strength outcomes because their frozen
  integration, identity, capture, or command contracts failed closed;
- slot 7 (`kitty-set-local-swap-family-v2`) found no arm that passed both
  independent references and coverage gates;
- slot 8 (`bid-policy-partnership-family-v1`) found no eligible bidding arm;
- slot 10 (`phase-exact12-k4-top3-v2`) failed its D1 diagnostic gate; and
- slot 11 (`enoch-bid-ownership-v1`) consumed strength outcomes, passed its
  300-pair development screen, then terminally failed its intended-product
  pilot with no completed pair.

The soft-Q proxy was separately rejected before full-corpus completion and
before strength evaluation. It completed 512 games and 503 Q roots, but its
offline selector did not reproduce real serving semantics, its held-out and
lineage controls were insufficient, and its dirty admission patch was removed.

The threshold-CVaR / Seed Protocol V2 stack stopped at incomplete design and
test-contract work. Its Ledger V2 successor draft failed independent review on
identity, ordering, type-validation, and graph-validation boundaries. It was
never activated and produced no candidate census, registration, gameplay,
strength evidence, or superiority decision. The untracked RED draft and tests
were deleted rather than committed.

There are no active slots, jobs, operators, or background evaluations. All
previously consumed, reserved, or conditionally unspent seed namespaces are
retired for this closed program and must not be recycled.

## Locked superiority gate was never reached

Only a complete passing locked `four-player-s2-v2` package could have supported
the statement that a bot is stronger than Enoch-0. No candidate reached it.
The never-completed package comprised:

- 2,000 intended-product pairs;
- 2,000 equal-compute pairs;
- 1,300 robustness pairs;
- two independent 2,500-pair confirmations; and
- 500 non-strength style pairs.

The final mandatory regression check was green on both the terminal research
source (`7ad1dab`, `274.03s`) and the production branch source (`159aa64`,
`276.15s`):

```text
SHENGJI_BOT_BUDGET_MS=60 cargo test -p shengji-core --release \
  --test baseline_gate baseline_expert_beats_easy_search \
  -- --ignored --exact --nocapture

result: 1 passed; 0 failed
```

That gate confirms the ordinary Expert-over-Easy floor remained intact. It is
not evidence of superiority over Enoch.

## Preserved evidence and source history

The authoritative Week 1 chain remains described in
[`strongest-bot-program-2026-07-02.md`](strongest-bot-program-2026-07-02.md).
Its terminal master record is `159aa64`; production reference `c813c8a` and
Enoch-0 identity
`5243de72c6669e233c1528f42f5de8e4b578165f55b0964dd48c3326df551e62`
remain unchanged.

The post-Week-1 research source history ends at `7ad1dab` and remains preserved
locally on `codex/landlord-next-threshold-cvar25-seed-v2-v1`. It is neither
merged into `master` nor exported as a remote branch because it contains
experimental and rejected mechanisms. Sealed local evidence remains under
these ignored run roots:

- `.enoch-week1-runs/authoritative-2026-07-05-d048b79`;
- `.enoch-week1-runs/stage2-first-targets-2026-07-06`;
- `.enoch-week1-runs/stage2-superiority-v2-2026-07-07-w2`;
- `.enoch-week1-runs/four-player-superiority-2026-07-08`;
- `.enoch-week1-runs/four-player-exact-late-state-v2-2026-07-09-r2`; and
- `.enoch-week1-runs/landlord-follow-heh-v1-2026-07-09-r2`.

Preserve those sealed results as historical evidence. They are not active
instructions and do not belong in the production source branch.

## Resumption contract

There is no current next step. Future bot-strength work is a new initiative,
not a continuation of this one. It requires all of the following:

1. explicit new human authorization and a newly versioned research objective;
2. a clean source revision and a fresh protocol that imports prior evidence as
   read-only history;
3. fresh candidate, diagnostic, development, and locked seed domains proven
   disjoint from every prior registry;
4. the hard `baseline_expert_beats_easy_search` gate before candidate work;
5. scoring limited to standard four-player Tractor, with Finding Friends
   excluded; and
6. the complete locked superiority package before any superiority, teacher,
   promotion, or deployment wording.

Do not resume an old operator, reuse an old slot, finish the rejected soft-Q or
Ledger V2 patches, or infer authorization from an unchecked item in a
historical roadmap.
