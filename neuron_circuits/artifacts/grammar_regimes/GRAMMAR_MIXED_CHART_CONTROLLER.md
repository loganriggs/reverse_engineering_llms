# Internal controller models for mixed causal charts

PCA codes of the local bilinear activation, final-token residual, and full residual sequence are fit on unselected training facts in one template and evaluated on correctly generalized held-out pairs. Linear codes and a frozen top-16 quadratic expansion test whether a short tensor-like controller generates the high-rank margin-operator family.

## Locked hypotheses

- **T0_full_context_is_a_compact_controller: not supported.**
- **T1_attention_context_is_needed: not supported.**
- **T2_low_degree_tensor_map_improves_controller: not supported.**
- **T3_local_bilinear_state_is_predictive: not supported.**
- **T4_controller_transfers_across_pairs: supported.**

T4 is a vacuous threshold pass: validation and held-out R2 are both approximately
zero. It is not positive evidence for a useful controller.

| Seed | Layer | Source | lin16 | lin32 | lin64 | lin128 | lin16+quad16 | validation lin64 |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | 0 | activation | -0.000 | 0.000 | 0.001 | 0.001 | -0.000 | 0.001 |
| 0 | 0 | final_token | 0.000 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| 0 | 0 | full_residual | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0 | 1 | activation | 0.001 | 0.001 | 0.001 | 0.001 | -0.001 | 0.000 |
| 0 | 1 | final_token | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| 0 | 1 | full_residual | 0.001 | 0.001 | 0.000 | 0.001 | 0.001 | 0.000 |
| 0 | 2 | activation | 0.001 | 0.001 | 0.001 | 0.001 | 0.003 | 0.001 |
| 0 | 2 | final_token | 0.003 | 0.002 | 0.002 | 0.002 | 0.009 | 0.001 |
| 0 | 2 | full_residual | 0.002 | 0.002 | 0.001 | 0.001 | 0.001 | 0.001 |
| 0 | 3 | activation | 0.009 | 0.004 | 0.009 | 0.019 | 0.055 | 0.015 |
| 0 | 3 | final_token | 0.002 | 0.002 | 0.006 | 0.009 | 0.051 | 0.019 |
| 0 | 3 | full_residual | -0.001 | 0.001 | -0.001 | -0.002 | 0.016 | 0.000 |
| 1 | 0 | activation | 0.001 | 0.001 | 0.001 | 0.000 | 0.001 | 0.001 |
| 1 | 0 | final_token | 0.000 | 0.001 | 0.002 | 0.001 | 0.001 | 0.001 |
| 1 | 0 | full_residual | -0.000 | -0.001 | -0.001 | -0.001 | -0.000 | 0.000 |
| 1 | 1 | activation | 0.002 | 0.001 | 0.001 | 0.001 | 0.003 | 0.001 |
| 1 | 1 | final_token | 0.003 | 0.003 | 0.002 | 0.002 | 0.009 | 0.003 |
| 1 | 1 | full_residual | -0.000 | 0.000 | 0.001 | 0.001 | 0.000 | 0.001 |
| 1 | 2 | activation | 0.007 | 0.010 | 0.004 | 0.005 | 0.006 | 0.006 |
| 1 | 2 | final_token | 0.007 | 0.017 | 0.004 | 0.003 | 0.020 | 0.003 |
| 1 | 2 | full_residual | 0.002 | 0.001 | 0.003 | -0.000 | 0.001 | 0.005 |
| 1 | 3 | activation | 0.014 | 0.024 | 0.024 | 0.013 | 0.027 | 0.010 |
| 1 | 3 | final_token | 0.018 | 0.023 | 0.030 | 0.019 | 0.067 | 0.022 |
| 1 | 3 | full_residual | 0.003 | 0.012 | 0.003 | 0.003 | 0.007 | 0.002 |
| 2 | 0 | activation | -0.000 | 0.001 | 0.001 | 0.001 | 0.000 | 0.001 |
| 2 | 0 | final_token | 0.001 | 0.001 | 0.001 | 0.001 | 0.000 | 0.002 |
| 2 | 0 | full_residual | 0.001 | 0.001 | 0.001 | -0.000 | 0.000 | 0.001 |
| 2 | 1 | activation | 0.001 | 0.001 | 0.001 | 0.000 | 0.001 | 0.001 |
| 2 | 1 | final_token | 0.001 | 0.000 | 0.001 | 0.000 | 0.003 | 0.002 |
| 2 | 1 | full_residual | 0.000 | 0.001 | 0.000 | -0.001 | 0.000 | 0.001 |
| 2 | 2 | activation | 0.001 | 0.001 | 0.001 | 0.001 | 0.005 | 0.002 |
| 2 | 2 | final_token | 0.004 | 0.003 | 0.002 | 0.001 | 0.012 | 0.002 |
| 2 | 2 | full_residual | 0.001 | 0.001 | 0.001 | 0.001 | 0.000 | 0.001 |
| 2 | 3 | activation | -0.017 | -0.021 | -0.015 | -0.003 | -0.018 | 0.011 |
| 2 | 3 | final_token | 0.013 | 0.011 | 0.008 | -0.001 | 0.032 | 0.009 |
| 2 | 3 | full_residual | 0.008 | 0.011 | 0.004 | 0.003 | 0.009 | 0.005 |

A positive controller result would shift interpretation from many unrelated programs to a compact latent code generating many operators. A negative result is restricted to linear PCA and a small quadratic chart; it does not exclude a different tensor factorization or nonlinear coordinate system.
