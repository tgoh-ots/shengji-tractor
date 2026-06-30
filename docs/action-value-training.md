# Honest action-value and belief training

Policy distillation answers what a perfect-information teacher chose. It does
not identify action value, inherits hidden-information teacher behavior, and
cannot credit near-equivalent legal actions. The schema-3 pipeline keeps policy
as a baseline while adding honest Monte Carlo state/action returns.

The first-stage Q estimator remains biased: each sampled candidate uses one real
hidden deal, sparse coverage, and a fixed continuation policy. Same-world
comparisons reduce variance but do not integrate over the full posterior.

High-leverage next steps:

- sample several compatible worlds per observation with common random numbers;
- stratify candidate sampling and use importance weights;
- alternate behavior collection and retraining while retaining a frozen champion;
- train quantile/categorical returns from score buckets;
- calibrate and integrate the offline destination belief model, independently A/B
  tested against constraint-only determinization;
- try recurrent or set encoders over complete trick history and hand multisets;
- test mixed human-like partners and rule/deck variants, not homogeneous self-play.

Promotion requires tract parity, no illegal moves or inference fallbacks, held-out
whole games/deals, and paired candidate-minus-embedded improvement with uncertainty.
See training/README.md for commands and exact contracts.
