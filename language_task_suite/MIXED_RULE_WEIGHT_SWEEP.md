# Mixture interventions on the original toy language

This extends the [mixed-rule bridge](EXISTING_GENERATOR_BRIDGE.md) with a controlled change to the **joint** autoregressive DGP: vary bigram versus induction sampling weight while holding the rule definitions, vocabulary, model architecture, and other class weights fixed. The point is to expose whether a learned computation follows the rule competition, not merely whether it predicts the next token on one stream. This directly follows the motivation in the [original project proposal](https://www.lesswrong.com/posts/GncmyuKmp526RJpmj/ambitious-mech-interp-w-tensor-transformers-on-toy-languages).

## DGP and verification

The baseline weights are bigram 0.25, trigram 0.22, skip-bigram 0.22, induction 0.20, bracket 0.01, noise 0.10. `bigram_heavy` sets bigram/induction to 0.35/0.10, while `induction_heavy` sets them to 0.10/0.35. These are **weights among active rule candidates**, not unconditional token probabilities. The setup path also changes with these weights, so the marginal context distributions are not matched. The generator is the original Python `LanguageGenerator`, not its separate fast implementation. The exact observable oracle was Monte Carlo checked under both new mixtures (`mixed_rule_weight_sweep_v2/oracle_check_*.json`); maximum controlled-context TV was 0.016.

## Training and cross-mixture prediction

`train_mixed_rule_weight_sweep.py` trains two-layer, four-head, attention-only bilinear transformers on 64-token sequences with the Python generator. The small sweep uses width 32, 2,500 steps, two seeds; the capacity check uses width 64, 5,000 steps, one seed. Batch size is 64; all models are evaluated on the **same** 256 held-out sequences per test mixture with expected KL to the exact prefix-conditioned DGP. All three trained models see all three test mixtures. The table reports KL in nats/token **on visible contexts where active rule candidates propose different tokens**; small-sweep cells average two seeds, capacity-check cells are one seed.

| Training mixture | Width/steps | Test baseline | Test bigram-heavy | Test induction-heavy |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 32/2,500 | 0.106 | 0.081 | 0.350 |
| Bigram-heavy | 32/2,500 | 0.177 | 0.074 | 0.601 |
| Induction-heavy | 32/2,500 | 0.146 | 0.207 | 0.167 |
| Baseline | 64/5,000 | 0.049 | 0.050 | 0.206 |
| Bigram-heavy | 64/5,000 | 0.107 | 0.042 | 0.440 |
| Induction-heavy | 64/5,000 | 0.122 | 0.221 | 0.068 |

The most sensitive subgroup is an **exactly bigram+induction** active pair with different proposed tokens. At width 64, the bigram-heavy model's KL on this group is 0.143 on its own stream but **1.006** on the induction-heavy stream. The induction-heavy model's corresponding values are 0.566 on bigram-heavy and **0.148** on its own stream. Increasing capacity and steps reduces in-distribution error but does not eliminate mixture sensitivity. Two seeds at width 32 and one at width 64 are insufficient to claim a stable scaling law.

## Fixed-prefix counterfactual

`eval_mixed_rule_fixed_context_shift.py` holds **520 visible prefixes** from a baseline-generated test set fixed. Each has exactly a bigram and induction candidate proposing different tokens, with no trigram or skip-bigram candidate; prefixes where either proposal has zero observable oracle probability are excluded. It recomputes the exact oracle under each mixture and compares the log odds of the two proposed tokens to the separately trained models' log odds on the *same input tokens*. Thus context-frequency shift is removed from this comparison, although models were still trained on different context distributions.

For baseline→induction-heavy, the oracle's mean log-odds shift is **−1.441** with SD **0.214**. The width-32 models' mean shifts are −1.477 and −1.550, but SDs are 1.172 and 0.945; their shift RMSEs are 1.146 and 0.960. The width-64 model's mean is −1.499, SD 1.206, RMSE 1.188. A constant oracle-mean shift would have RMSE 0.214 on these prefixes. The models track the *aggregate* direction and size but introduce much more context-by-context variation than the known rule-weight intervention requires. This remains true in the wider run despite its better ordinary KL. It is a negative result for any claim that the currently learned circuit precisely implements the mixture operation on each conflict context.

## Local bond compression under mixture shift

`eval_existing_mixed_bond_gauges.py` was applied to the width-64 **baseline-trained** model, fitting a basis on either baseline or induction-heavy sequences and testing both on fresh induction-heavy sequences. This is a local orthogonal basis of an attention-output bond, not full-function canonicalization. Full-width gauges reproduce logits within numerical tolerance. At 16/64 coordinates in the **second** attention bond, the weight-only SVD basis gives model-to-truncated KL **0.0024** on induction-heavy conflict contexts; activation PCA fitted on baseline gives **0.0537**, and physical axes give **0.1006**. At 32/64, SVD is approximately lossless by this metric (0.000004). Fitting PCA on induction-heavy rather than baseline contexts only modestly changes these values. However, the untruncated baseline model itself has oracle KL **0.197** on this shifted conflict set. Faithful compression of an imperfect model is not recovery of the DGP, and this result does not identify rule-specific directions.

## Decision and next experiment

The exact oracle and fixed-prefix shift now give a ground-truth target for *causal supervision*. A candidate component should predict the intervention-induced change in bigram-versus-induction log odds on held-out conflicts, with a matched no-conflict control. The next experiment should fit and evaluate such components across whole conflict-pattern holdouts, comparing physical axes, weight SVD, activation PCA, and any sparse/intervention-aware basis at matched width and reconstruction. Rule-weight changes alter both conditional probabilities and the training context distribution; a stronger control must match marginal context/token frequencies or use a teacher that exposes the mixture parameter as input. The current DGP is intentionally diagnostic but has only 26 artificial tokens and is not a realism result for FineWeb-like text.

Artifacts: `mixed_rule_weight_sweep_v1/` and `mixed_rule_weight_sweep_v2/` contain configs, saved checkpoints, cross-mixture expected-KL results, fixed-context log-odds results, oracle checks, and bond truncation results. Their scripts are adjacent to this note.
