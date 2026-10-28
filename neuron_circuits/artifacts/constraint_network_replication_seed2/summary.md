# Signed factual constraint-network pilot

The design and thresholds were locked before these kernels or edits were computed. See `../constraint_network_preregistration.md`.

## Primary causal results

Pooled scores use all four transformer blocks and exclude each edit's source fact.

| rho | width | scale | Pearson | origin R² | null Pearson | mean absolute collateral | rival switches |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 128 | 0.25 | 0.994 | 0.989 | -0.080 | 0.02603 | 0.009 |
| 0 | 128 | 1 | 0.809 | 0.651 | -0.090 | 0.11074 | 0.048 |
| 0 | 128 | 4 | 0.569 | 0.298 | -0.059 | 0.57055 | 0.153 |
| 0 | 256 | 0.25 | 0.999 | 0.998 | -0.024 | 0.01346 | 0.017 |
| 0 | 256 | 1 | 0.993 | 0.982 | -0.022 | 0.05374 | 0.062 |
| 0 | 256 | 4 | 0.725 | 0.514 | -0.023 | 0.25417 | 0.133 |
| 1 | 128 | 0.25 | 1.000 | 1.000 | -0.025 | 0.00330 | 0.001 |
| 1 | 128 | 1 | 1.000 | 1.000 | -0.025 | 0.01319 | 0.006 |
| 1 | 128 | 4 | 1.000 | 0.999 | -0.025 | 0.05275 | 0.020 |
| 1 | 256 | 0.25 | 1.000 | 1.000 | -0.005 | 0.00187 | 0.000 |
| 1 | 256 | 1 | 1.000 | 1.000 | -0.005 | 0.00748 | 0.001 |
| 1 | 256 | 4 | 1.000 | 0.999 | -0.005 | 0.02992 | 0.007 |

## Preregistered verdict

- Local constraint explanation: **validated**
- Finite constraint explanation: **validated**
- Load induced web: **falsified**

## Load comparison

Values average the four blockwise controlled residual-cosine summaries; collateral uses 1-logit edits.

| width | rho=0 residual | rho=1 residual | rho=0 collateral | rho=1 collateral |
|---:|---:|---:|---:|---:|
| 128 | 0.04863 | 0.01211 | 0.11074 | 0.01319 |
| 256 | 0.06853 | 0.00791 | 0.05374 | 0.00748 |

## Scope

A successful local Taylor prediction validates the signed constraint network as a causal local model, not as a unique or complete account of storage. The load test asks the stronger empirical question: whether arbitrary facts make that network broader after controlling for obvious shared fields. Semantic coexistence must be assessed with the locked activation audit and a separate transfer intervention; it is not inferred from gradient coupling alone.
