# Signed factual constraint-network pilot

The design and thresholds were locked before these kernels or edits were computed. See `../constraint_network_preregistration.md`.

## Primary causal results

Pooled scores use all four transformer blocks and exclude each edit's source fact.

| rho | width | scale | Pearson | origin R² | null Pearson | mean absolute collateral | rival switches |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 128 | 0.25 | 1.000 | 0.999 | -0.047 | 0.01357 | 0.015 |
| 0 | 128 | 1 | 0.996 | 0.988 | -0.048 | 0.05340 | 0.050 |
| 0 | 128 | 4 | 0.749 | 0.478 | -0.036 | 0.20950 | 0.130 |
| 0 | 256 | 0.25 | 1.000 | 0.999 | -0.047 | 0.01921 | 0.014 |
| 0 | 256 | 1 | 0.995 | 0.988 | -0.045 | 0.07594 | 0.042 |
| 0 | 256 | 4 | 0.936 | 0.854 | -0.035 | 0.29590 | 0.115 |
| 1 | 128 | 0.25 | 1.000 | 1.000 | -0.016 | 0.00262 | 0.001 |
| 1 | 128 | 1 | 1.000 | 1.000 | -0.017 | 0.01047 | 0.003 |
| 1 | 128 | 4 | 1.000 | 0.999 | -0.017 | 0.04186 | 0.011 |
| 1 | 256 | 0.25 | 1.000 | 1.000 | -0.003 | 0.00177 | 0.001 |
| 1 | 256 | 1 | 1.000 | 1.000 | -0.003 | 0.00708 | 0.004 |
| 1 | 256 | 4 | 1.000 | 0.999 | -0.003 | 0.02835 | 0.012 |

## Preregistered verdict

- Local constraint explanation: **validated**
- Finite constraint explanation: **validated**
- Load induced web: **falsified**

## Load comparison

Values average the four blockwise controlled residual-cosine summaries; collateral uses 1-logit edits.

| width | rho=0 residual | rho=1 residual | rho=0 collateral | rho=1 collateral |
|---:|---:|---:|---:|---:|
| 128 | 0.04709 | 0.00958 | 0.05340 | 0.01047 |
| 256 | 0.06923 | 0.00694 | 0.07594 | 0.00708 |

## Scope

A successful local Taylor prediction validates the signed constraint network as a causal local model, not as a unique or complete account of storage. The load test asks the stronger empirical question: whether arbitrary facts make that network broader after controlling for obvious shared fields. Semantic coexistence must be assessed with the locked activation audit and a separate transfer intervention; it is not inferred from gradient coupling alone.
