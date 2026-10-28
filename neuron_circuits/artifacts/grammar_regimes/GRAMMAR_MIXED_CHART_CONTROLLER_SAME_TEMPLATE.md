# Internal controller models for mixed causal charts

PCA codes of the local bilinear activation, final-token residual, and full residual sequence are fit on unselected training facts in one template and evaluated on correctly generalized held-out pairs. Linear codes and a frozen top-16 quadratic expansion test whether a short tensor-like controller generates the high-rank margin-operator family.

## Locked hypotheses

- **S0_full_context_is_a_compact_within_template_controller: not supported.**
- **S1_attention_context_is_needed_within_template: not supported.**
- **S2_low_degree_tensor_map_improves_within_template: not supported.**
- **S3_local_bilinear_state_is_predictive_within_template: not supported.**
- **S4_controller_generalizes_to_new_pairs: supported.**

S4 is a vacuous threshold pass: validation and held-out R2 are both approximately
zero. It is not positive evidence for a useful controller.

| Seed | Layer | Source | lin16 | lin32 | lin64 | lin128 | lin16+quad16 | validation lin64 |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | 0 | activation | 0.000 | 0.000 | -0.000 | -0.001 | 0.001 | 0.001 |
| 0 | 0 | final_token | -0.001 | -0.000 | -0.001 | -0.000 | 0.000 | 0.002 |
| 0 | 0 | full_residual | -0.001 | -0.000 | 0.001 | 0.000 | -0.000 | 0.001 |
| 0 | 1 | activation | 0.001 | 0.001 | 0.001 | 0.000 | 0.002 | 0.000 |
| 0 | 1 | final_token | 0.000 | 0.001 | 0.001 | 0.001 | 0.000 | 0.002 |
| 0 | 1 | full_residual | 0.001 | 0.001 | 0.001 | 0.001 | 0.000 | 0.001 |
| 0 | 2 | activation | 0.001 | 0.001 | 0.001 | 0.000 | 0.002 | 0.002 |
| 0 | 2 | final_token | 0.002 | 0.001 | 0.000 | 0.000 | 0.008 | 0.002 |
| 0 | 2 | full_residual | 0.002 | 0.001 | -0.004 | 0.000 | 0.001 | 0.005 |
| 0 | 3 | activation | 0.008 | 0.001 | 0.002 | -0.000 | 0.018 | 0.011 |
| 0 | 3 | final_token | -0.005 | 0.001 | 0.007 | -0.009 | 0.053 | 0.013 |
| 0 | 3 | full_residual | 0.013 | -0.000 | -0.004 | 0.002 | 0.028 | 0.018 |
| 1 | 0 | activation | -0.001 | -0.001 | -0.001 | -0.000 | -0.000 | 0.001 |
| 1 | 0 | final_token | -0.001 | -0.001 | -0.001 | -0.000 | 0.000 | 0.001 |
| 1 | 0 | full_residual | -0.000 | -0.000 | -0.000 | 0.000 | -0.000 | 0.000 |
| 1 | 1 | activation | 0.000 | -0.000 | 0.000 | 0.000 | -0.001 | 0.001 |
| 1 | 1 | final_token | 0.002 | 0.002 | 0.001 | 0.001 | 0.007 | 0.001 |
| 1 | 1 | full_residual | 0.000 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| 1 | 2 | activation | 0.003 | 0.006 | 0.008 | 0.002 | 0.005 | 0.004 |
| 1 | 2 | final_token | 0.013 | 0.005 | 0.005 | 0.004 | 0.019 | 0.003 |
| 1 | 2 | full_residual | 0.003 | 0.003 | 0.008 | 0.002 | 0.001 | 0.004 |
| 1 | 3 | activation | 0.009 | 0.017 | 0.022 | 0.012 | 0.028 | 0.022 |
| 1 | 3 | final_token | 0.017 | 0.020 | 0.025 | 0.018 | 0.101 | 0.012 |
| 1 | 3 | full_residual | 0.004 | 0.015 | 0.004 | 0.013 | 0.023 | 0.003 |
| 2 | 0 | activation | -0.000 | -0.001 | -0.001 | 0.000 | -0.007 | 0.002 |
| 2 | 0 | final_token | -0.002 | -0.001 | -0.001 | -0.000 | -0.000 | 0.002 |
| 2 | 0 | full_residual | -0.002 | -0.001 | -0.001 | -0.001 | -0.001 | 0.002 |
| 2 | 1 | activation | 0.001 | 0.001 | 0.000 | -0.000 | 0.000 | 0.001 |
| 2 | 1 | final_token | -0.000 | 0.000 | -0.000 | 0.000 | 0.001 | 0.000 |
| 2 | 1 | full_residual | 0.001 | 0.000 | 0.001 | 0.001 | 0.000 | 0.001 |
| 2 | 2 | activation | 0.001 | 0.001 | 0.001 | 0.001 | 0.000 | 0.001 |
| 2 | 2 | final_token | 0.002 | 0.002 | 0.001 | 0.001 | 0.003 | 0.001 |
| 2 | 2 | full_residual | 0.000 | 0.001 | 0.002 | 0.001 | 0.001 | 0.002 |
| 2 | 3 | activation | 0.001 | -0.003 | 0.003 | 0.001 | 0.010 | 0.005 |
| 2 | 3 | final_token | 0.001 | 0.004 | -0.002 | 0.001 | 0.036 | 0.005 |
| 2 | 3 | full_residual | 0.008 | 0.011 | 0.011 | 0.007 | 0.017 | 0.017 |

A positive controller result would shift interpretation from many unrelated programs to a compact latent code generating many operators. A negative result is restricted to linear PCA and a small quadratic chart; it does not exclude a different tensor factorization or nonlinear coordinate system.
