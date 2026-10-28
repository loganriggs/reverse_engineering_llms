# Layer-16 compact program at the final logits and loss

The exact and approximate Layer-16 local derivative programs pass through dynamic Layer-17 attention, the final bilinear MLP, final RMS normalization, unembedding, and tanh logit cap. `Fisher` is probability-weighted centered-logit squared fidelity, the local quadratic geometry of predictive KL. `CE` is fidelity of the true-next-token loss directional derivatives over the 32 local input axes.

## Locked hypotheses

- **L0_full_suffix_logit_derivative_is_exact: supported.**
- **L1_inner8_preserves_full_logit_tangents: not supported.**
- **L2_inner4_preserves_full_logit_tangents: not supported.**
- **L3_inner8_preserves_fisher_geometry: not supported.**
- **L4_inner8_preserves_cross_entropy_derivatives: not supported.**
- **L5_common_operator_is_loss_sufficient: not supported.**
- **L6_contextual_program_materially_improves_loss: supported.**
- **L7_loss_program_transfers_ood: supported.**
- **L8_inner8_nearly_matches_full64_on_loss: not supported.**

Autograd error: `5.06e-07`; maximum native-forward error: `0.00e+00`.

| Corpus | Native CE | Program | Logit | Fisher | CE derivative | Margin |
|---|---:|---|---:|---:|---:|---:|
| fineweb | 3.821 | mean | 0.749 | 0.542 | 0.508 | 0.499 |
| fineweb | 3.821 | full64 | 0.924 | 0.834 | 0.840 | 0.828 |
| fineweb | 3.821 | inner4 | 0.843 | 0.732 | 0.717 | 0.709 |
| fineweb | 3.821 | inner8 | 0.888 | 0.769 | 0.758 | 0.744 |
| wikitext | 4.130 | mean | 0.595 | 0.385 | 0.344 | 0.300 |
| wikitext | 4.130 | full64 | 0.911 | 0.826 | 0.832 | 0.823 |
| wikitext | 4.130 | inner4 | 0.825 | 0.737 | 0.749 | 0.740 |
| wikitext | 4.130 | inner8 | 0.874 | 0.766 | 0.774 | 0.762 |
| tinystories | 2.725 | mean | 0.689 | 0.544 | 0.089 | 0.141 |
| tinystories | 2.725 | full64 | 0.932 | 0.854 | 0.916 | 0.852 |
| tinystories | 2.725 | inner4 | 0.838 | 0.738 | 0.827 | 0.748 |
| tinystories | 2.725 | inner8 | 0.887 | 0.780 | 0.862 | 0.783 |

These are first-order causal comparisons around native text states. CE and margin can be noisier than full-logit fidelity because each collapses a 50,304-dimensional tangent to a single scalar per direction; aggregate denominators and OOD replication are therefore reported rather than per-example ratios.
