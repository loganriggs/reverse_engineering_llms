# Signed factual constraint-network pilot

The design and thresholds were locked before these kernels or edits were computed. See `../constraint_network_preregistration.md`.

## Primary causal results

Pooled scores use all four transformer blocks and exclude each edit's source fact.

| rho | width | scale | Pearson | origin R² | null Pearson | mean absolute collateral | rival switches |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 128 | 0.25 | 0.993 | 0.984 | 0.004 | 0.01746 | 0.026 |
| 0 | 128 | 1 | 0.895 | 0.765 | -0.011 | 0.07733 | 0.100 |
| 0 | 128 | 4 | 0.596 | 0.343 | -0.012 | 0.43450 | 0.263 |
| 0 | 256 | 0.25 | 1.000 | 0.999 | -0.020 | 0.00841 | 0.024 |
| 0 | 256 | 1 | 0.994 | 0.987 | -0.018 | 0.03455 | 0.064 |
| 0 | 256 | 4 | 0.881 | 0.768 | -0.015 | 0.16853 | 0.148 |
| 0.5 | 128 | 0.25 | 1.000 | 1.000 | 0.030 | 0.00242 | 0.002 |
| 0.5 | 128 | 1 | 1.000 | 1.000 | 0.030 | 0.00967 | 0.009 |
| 0.5 | 128 | 4 | 1.000 | 0.999 | 0.030 | 0.03863 | 0.038 |
| 0.5 | 256 | 0.25 | 1.000 | 1.000 | -0.021 | 0.00300 | 0.000 |
| 0.5 | 256 | 1 | 1.000 | 1.000 | -0.021 | 0.01199 | 0.000 |
| 0.5 | 256 | 4 | 1.000 | 0.999 | -0.021 | 0.04800 | 0.013 |
| 1 | 128 | 0.25 | 1.000 | 1.000 | -0.003 | 0.00287 | 0.004 |
| 1 | 128 | 1 | 1.000 | 1.000 | -0.003 | 0.01149 | 0.006 |
| 1 | 128 | 4 | 0.999 | 0.999 | -0.003 | 0.04610 | 0.021 |
| 1 | 256 | 0.25 | 1.000 | 1.000 | 0.034 | 0.00174 | 0.002 |
| 1 | 256 | 1 | 1.000 | 1.000 | 0.034 | 0.00696 | 0.005 |
| 1 | 256 | 4 | 1.000 | 0.999 | 0.034 | 0.02785 | 0.029 |

## Preregistered verdict

- Local constraint explanation: **validated**
- Finite constraint explanation: **validated**
- Load induced web: **falsified**

## Load comparison

Values average the four blockwise controlled residual-cosine summaries; collateral uses 1-logit edits.

| width | rho=0 residual | rho=1 residual | rho=0 collateral | rho=1 collateral |
|---:|---:|---:|---:|---:|
| 128 | 0.04747 | 0.01082 | 0.07733 | 0.01149 |
| 256 | 0.05040 | 0.00710 | 0.03455 | 0.00696 |

## Scope

A successful local Taylor prediction validates the signed constraint network as a causal local model, not as a unique or complete account of storage. The load test asks the stronger empirical question: whether arbitrary facts make that network broader after controlling for obvious shared fields. Semantic coexistence must be assessed with the locked activation audit and a separate transfer intervention; it is not inferred from gradient coupling alone.
