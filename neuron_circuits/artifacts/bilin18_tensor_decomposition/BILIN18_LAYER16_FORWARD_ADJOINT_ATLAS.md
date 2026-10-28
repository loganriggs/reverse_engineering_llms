# Exact forward-tangent and backward-adjoint atlas for the Layer-16 loss path

Each 32D behavioral vector is exactly factorized as `y_s = J_s h_s`, where `J_s` transports the fixed Layer-12 basis into the Layer-16 state and `h_s = B(x_s)^T g_s` is the effective state adjoint induced by the bilinear MLP and downstream loss adjoint. Frozen PCA spans are learned separately from forward tangent rows, states, raw write adjoints, and effective adjoints.

## Locked hypotheses

- **A0_forward_adjoint_contraction_is_exact: supported.**
- **A1_rank64_effective_adjoint_subspace_is_sufficient: not supported.**
- **A2_rank64_forward_subspace_is_sufficient: not supported.**
- **A3_rank128_forward_or_adjoint_is_high_fidelity: not supported.**
- **A4_bilinear_interaction_broadens_the_adjoint: supported.**
- **A5_natural_subspaces_beat_random: supported.**
- **A6_factor_subspaces_transfer_ood: not supported.**
- **A7_state_or_write_adjoint_alone_is_inferior: supported.**

Autograd error: `6.27e-07`; maximum contraction error: `7.78e-07`.

| Discovery family | Energy R90 | Effective rank | Top share |
|---|---:|---:|---:|
| forward | 588 | 210.5 | 0.028 |
| state | 207 | 14.8 | 0.655 |
| write_adjoint | 275 | 268.6 | 0.033 |
| effective_adjoint | 310 | 214.7 | 0.147 |

| Corpus | Basis | Behavior R32 / R64 / R128 / R256 |
|---|---|---:|
| fineweb | forward | 0.718 / 0.806 / 0.829 / 0.873 |
| fineweb | effective_adjoint | 0.390 / 0.436 / 0.489 / 0.592 |
| fineweb | state | 0.470 / 0.560 / 0.634 / 0.709 |
| fineweb | write_adjoint | 0.252 / 0.308 / 0.402 / 0.498 |
| fineweb | random rank64 p95 | 0.101 |
| wikitext | forward | 0.626 / 0.702 / 0.746 / 0.801 |
| wikitext | effective_adjoint | 0.283 / 0.347 / 0.459 / 0.572 |
| wikitext | state | 0.326 / 0.413 / 0.513 / 0.592 |
| wikitext | write_adjoint | 0.154 / 0.213 / 0.276 / 0.428 |
| wikitext | random rank64 p95 | 0.093 |
| tinystories | forward | 0.792 / 0.848 / 0.866 / 0.899 |
| tinystories | effective_adjoint | 0.388 / 0.475 / 0.564 / 0.674 |
| tinystories | state | 0.475 / 0.589 / 0.675 / 0.762 |
| tinystories | write_adjoint | 0.179 / 0.261 / 0.392 / 0.521 |
| tinystories | random rank64 p95 | 0.106 |

A basis derived from one factor is scored only after inserting it into the exact contraction. This is more behaviorally relevant than its raw explained variance. The factor bases are shared coordinates; the site-specific `J`, `x`, `g`, and `h` remain part of the executable explanation.
