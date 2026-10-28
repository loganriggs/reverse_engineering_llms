# Signed factual constraint-network pilot

The design and thresholds were locked before these kernels or edits were computed. See `../constraint_network_preregistration.md`.

## Primary causal results

Pooled scores use all four transformer blocks and exclude each edit's source fact.

| rho | width | scale | Pearson | origin R² | null Pearson | mean |collateral| | rival switches |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 128 | 0.25 | 0.996 | 0.987 | -0.020 | 0.01339 | 0.012 |
| 0 | 128 | 1 | 0.953 | 0.856 | -0.018 | 0.05191 | 0.044 |
| 0 | 128 | 4 | 0.509 | 0.083 | -0.014 | 0.22796 | 0.150 |
| 0 | 256 | 0.25 | 1.000 | 0.999 | 0.001 | 0.00999 | 0.021 |
| 0 | 256 | 1 | 0.995 | 0.988 | 0.002 | 0.04059 | 0.055 |
| 0 | 256 | 4 | 0.892 | 0.782 | 0.005 | 0.18408 | 0.130 |
| 0.5 | 128 | 0.25 | 1.000 | 1.000 | 0.009 | 0.00243 | 0.001 |
| 0.5 | 128 | 1 | 1.000 | 1.000 | 0.009 | 0.00971 | 0.002 |
| 0.5 | 128 | 4 | 1.000 | 0.999 | 0.009 | 0.03881 | 0.009 |
| 0.5 | 256 | 0.25 | 1.000 | 1.000 | 0.041 | 0.00243 | 0.001 |
| 0.5 | 256 | 1 | 1.000 | 1.000 | 0.041 | 0.00973 | 0.003 |
| 0.5 | 256 | 4 | 1.000 | 0.999 | 0.041 | 0.03894 | 0.008 |
| 1 | 128 | 0.25 | 1.000 | 1.000 | 0.007 | 0.00290 | 0.001 |
| 1 | 128 | 1 | 1.000 | 1.000 | 0.007 | 0.01161 | 0.002 |
| 1 | 128 | 4 | 0.999 | 0.999 | 0.007 | 0.04645 | 0.010 |
| 1 | 256 | 0.25 | 1.000 | 1.000 | -0.030 | 0.00183 | 0.000 |
| 1 | 256 | 1 | 1.000 | 1.000 | -0.029 | 0.00731 | 0.001 |
| 1 | 256 | 4 | 1.000 | 0.999 | -0.029 | 0.02924 | 0.009 |

## Preregistered verdict

- Local constraint explanation: **validated**
- Finite constraint explanation: **validated**
- Load induced web: **falsified**

## Load comparison

Values average the four blockwise controlled residual-cosine summaries; collateral uses 1-logit edits.

| width | rho=0 residual | rho=1 residual | rho=0 collateral | rho=1 collateral |
|---:|---:|---:|---:|---:|
| 128 | 0.04192 | 0.01081 | 0.05191 | 0.01161 |
| 256 | 0.05173 | 0.00688 | 0.04059 | 0.00731 |

## Scope

A successful local Taylor prediction validates the signed constraint network as a causal local model, not as a unique or complete account of storage. The load test asks the stronger empirical question: whether arbitrary facts make that network broader after controlling for obvious shared fields. Semantic coexistence must be assessed with the locked activation audit and a separate transfer intervention; it is not inferred from gradient coupling alone.
