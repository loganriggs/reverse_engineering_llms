# Exact block-factor chain across rule and mixed regimes

For final-position interventions, causality makes the local chain exact: centered answer readout O times downstream final-position block Jacobians A_k times the source MLP down-projection D_l. Every A_k is computed analytically from RMSNorm, softmax QK/value attention, and the bilinear MLP at the natural state.

## Locked hypotheses

- **F0_analytic_factor_chain_is_exact: supported.**
- **F1_rules_have_more_reusable_mean_factors: not supported.**
- **F2_mixed_block_factors_are_more_diverse: supported.**
- **F3_final_readout_factor_is_reusable: not supported.**
- **F4_mixed_diversity_has_a_causal_factor_location: supported.**
- **F5_no_single_mixed_factor_explains_the_chart: supported.**

| Seed | rho | Exact min | rank A1/A2/A3/O | Layer-0 all-mean | Freeze O | Worst freeze-one | Best only-one |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0 | 1.000000 | 28.4/16.4/9.4/10.5 | 0.020 | 0.764 | 0.166 | 0.245 |
| 0 | 0.5 | 1.000000 | 52.5/50.7/45.3/40.3 | 0.016 | 0.930 | 0.166 | 0.193 |
| 1 | 0.0 | 1.000000 | 28.7/22.3/21.8/9.0 | 0.083 | 0.816 | 0.256 | 0.322 |
| 1 | 0.5 | 1.000000 | 47.3/49.0/43.6/41.4 | 0.017 | 0.923 | 0.195 | 0.193 |

All-mean asks whether one global chain suffices. Freeze-one asks which state-conditioned factor is necessary given all the others. Only-one asks whether a single varying factor is sufficient. These are exact operator reconstructions, not probes.
