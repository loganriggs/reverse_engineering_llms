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

Autograd error: `4.52e-07`; maximum native-forward error: `0.00e+00`.

| Corpus | Native CE | Program | Logit | Fisher | CE derivative | Margin |
|---|---:|---|---:|---:|---:|---:|
| fineweb | 3.433 | mean | 0.445 | 0.348 | 0.149 | 0.097 |
| fineweb | 3.433 | full64 | 0.880 | 0.803 | 0.803 | 0.785 |
| fineweb | 3.433 | inner4 | 0.796 | 0.709 | 0.708 | 0.682 |
| fineweb | 3.433 | inner8 | 0.839 | 0.741 | 0.746 | 0.720 |
| wikitext | 3.951 | mean | 0.371 | 0.175 | 0.203 | 0.115 |
| wikitext | 3.951 | full64 | 0.880 | 0.787 | 0.779 | 0.758 |
| wikitext | 3.951 | inner4 | 0.801 | 0.700 | 0.700 | 0.670 |
| wikitext | 3.951 | inner8 | 0.842 | 0.732 | 0.732 | 0.702 |
| tinystories | 2.635 | mean | 0.770 | 0.519 | 0.289 | 0.289 |
| tinystories | 2.635 | full64 | 0.927 | 0.819 | 0.838 | 0.810 |
| tinystories | 2.635 | inner4 | 0.850 | 0.696 | 0.719 | 0.673 |
| tinystories | 2.635 | inner8 | 0.889 | 0.739 | 0.759 | 0.719 |

These are first-order causal comparisons around native text states. CE and margin can be noisier than full-logit fidelity because each collapses a 50,304-dimensional tangent to a single scalar per direction; aggregate denominators and OOD replication are therefore reported rather than per-example ratios.
