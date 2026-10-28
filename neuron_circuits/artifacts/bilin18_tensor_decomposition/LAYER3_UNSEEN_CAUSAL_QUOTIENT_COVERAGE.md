# Unseen-word causal validation of the Layer-3 quotient

Ten manually frozen source/target word pairs per family are disjoint from every seed pair used to construct the channel banks and input metrics. Natural occurrences are placed at the same edit and causal positions as the original test, the target token is substituted, and the model suffix is recomputed. The rank-eight shared/residual quotient is never refit.

## Locked hypotheses

- **U1_unseen_causal_panel_has_coverage: supported.**
- **U2_sentiment_and_polarity_remain_shared_dominant: supported.**
- **U3_gender_and_tense_remain_residual_dominant: supported.**
- **U4_scale_remains_a_mixed_causal_regime: not supported.**
- **U5_unseen_residuals_are_family_specific: not supported.**
- **U6_seed_and_unseen_regime_labels_agree: supported.**
- **U7_unseen_cross_term_is_nontrivial: supported.**
- **U8_unseen_regimes_transfer_ood: supported.**

![Unseen causal quotient](layer3_unseen_causal_quotient_coverage.png)

| Corpus / family | N | Unique source tokens | Shared-like | Residual | Shared−residual | Best wrong residual | Cross norm | Semantic projection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fineweb / sentiment | 60 | 10 | 0.943 | 0.026 | 0.916 | 0.111 | 0.141 | 0.048 |
| fineweb / gender | 60 | 10 | 0.268 | 0.901 | -0.632 | 0.175 | 0.209 | 0.002 |
| fineweb / tense | 60 | 10 | 0.150 | 0.902 | -0.752 | 0.105 | 0.234 | 0.007 |
| fineweb / increasing_scale | 60 | 10 | 0.832 | 0.135 | 0.697 | 0.167 | 0.078 | 0.007 |
| fineweb / polarity | 60 | 10 | 0.902 | 0.238 | 0.663 | 0.218 | 0.173 | 0.026 |
| tinystories / sentiment | 42 | 7 | 0.950 | -0.033 | 0.983 | 0.053 | 0.144 | 0.033 |
| tinystories / gender | 25 | 5 | 0.321 | 0.902 | -0.582 | 0.169 | 0.193 | 0.003 |
| tinystories / tense | 29 | 7 | 0.020 | 0.740 | -0.720 | 0.110 | 0.329 | 0.002 |
| tinystories / increasing_scale | 17 | 6 | 0.538 | 0.401 | 0.136 | 0.192 | 0.087 | 0.006 |
| tinystories / polarity | 6 | 1 | 0.936 | 0.453 | 0.483 | 0.404 | 0.257 | 0.063 |

Fidelity is relative to the frozen family's own top-32 causal patch, not to the entire MLP. This asks whether the internal regime transfers to new lexical realizations. Semantic projection is reported separately because an internally faithful family patch can still be a small or opposing contributor to the complete final-logit counterfactual.

Repeated contexts for the same source token improve context coverage but do not improve lexical coverage. Any sensitivity run with few unique source tokens must therefore be read as a context-robustness test, not as independent evidence for family-wide lexical generalization.
