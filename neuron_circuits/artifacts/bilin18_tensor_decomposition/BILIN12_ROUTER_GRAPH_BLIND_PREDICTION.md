# Blind sitewise prediction of sparse-graph errors

For each natural site, the graph's omitted pooled hidden vector is propagated only through the analytic local Jacobian of final RMSNorm, unembedding, and bounded logits. The resulting Fisher energy and margin score are frozen predictions of the subsequently measured finite graph-replacement error.

## Locked hypotheses

- **B0_local_jacobian_predicts_logit_direction: supported.**
- **B1_fisher_energy_ranks_finite_kl: supported.**
- **B2_top_decile_localizes_error: not supported.**
- **B3_predicted_margin_identifies_top1_flips: supported.**
- **B4_output_geometry_beats_hidden_norm: supported.**
- **B5_second_order_scale_is_calibrated: supported.**

| Corpus | Fisher cosine | Fisher/hidden Spearman with KL | Top-decile KL share | Flip rate / AUROC | Actual/predicted KL median |
|---|---:|---:|---:|---:|---:|
| fineweb | 0.998 | 0.984/0.393 | 0.360 | 0.113/1.000 | 1.017 |
| wikitext | 0.998 | 0.985/0.532 | 0.329 | 0.124/0.999 | 1.023 |
| tinystories | 0.997 | 0.979/0.342 | 0.330 | 0.103/1.000 | 0.987 |

This is a prospective calculation from the native state, omitted graph vector, and output weights; it does not use the finite replacement to choose sites or fit a scale. Success would make the graph diagnostically useful even where it is not exact. Failure would show that aggregate reconstruction hides nonlinear, example-specific errors requiring a richer downstream model.
