# Metric red team of output-core semantic roles

This separates three meanings of an output role: a uniform-vocabulary unembedding direction, the same raw direction under the natural output-token distribution, and the exact context-dependent metric after final RMSNorm and logit saturation.

## Locked hypotheses

- **R0_forward_backward_duality_is_exact: supported.** The cap-weighted RMSNorm JVP score matches the residual-adjoint projection to relative error at most 1e-5.
- **R1_uniform_and_natural_raw_roles_agree: not supported.** Uniform-vocabulary weight labels and natural-frequency raw-unembedding labels agree on at least 8 of 16 axes for target or top1.
- **R2_raw_and_exact_natural_roles_agree: not supported.** Natural-frequency raw-unembedding and exact-adjoint labels agree on at least 8 of 16 axes for both target and top1.
- **R3_rms_and_exact_natural_roles_agree: supported.** RMS-aware and exact-adjoint labels agree on at least 12 of 16 axes for both target and top1.
- **R4_rms_metric_predicts_exact_scores: not supported.** Median per-axis RMS-aware versus exact score correlation is at least 0.98 on every corpus for target and top1.
- **R5_raw_unembedding_is_numerically_adequate: supported.** Median per-axis raw-unembedding versus exact score correlation is at least 0.80 on every corpus for target and top1.
- **R6_output_metric_effect_transfers: supported.** The RMS-aware correlation advantage over raw unembedding is nonnegative on every corpus and endpoint.
- **R7_three_role_definitions_are_reported_separately: supported.** Uniform vocabulary, natural-frequency raw, and context-metric roles are retained as distinct interpretation targets rather than silently merged.

Maximum forward-JVP/backward-VJP duality error: `3.42e-07`.

## Numerical agreement

| Corpus | Endpoint | raw vs exact | RMS-aware vs exact | cap-weighted formula vs exact |
|---|---|---:|---:|---:|
| fineweb | target | 0.887 | 0.969 | 1.000 |
| fineweb | top1 | 0.885 | 0.974 | 1.000 |
| wikitext | target | 0.893 | 0.983 | 1.000 |
| wikitext | top1 | 0.898 | 0.987 | 1.000 |
| tinystories | target | 0.927 | 0.982 | 1.000 |
| tinystories | top1 | 0.924 | 0.985 | 1.000 |

## Label agreement out of 16 modes

| Endpoint | uniform→raw | uniform→RMS | uniform→exact | raw→RMS | raw→exact | RMS→exact |
|---|---:|---:|---:|---:|---:|---:|
| target | 2 | 0 | 1 | 8 | 7 | 14 |
| top1 | 2 | 0 | 0 | 9 | 9 | 16 |

Uniform-vocabulary meaning asks what a direction could promote over the tokenizer. Natural-frequency meaning asks which questions the model actually presents to it. Context-metric meaning includes the state-dependent final normalization. These are different use cases; a mismatch is not automatically polysemanticity.
