# Validated grammar-regime results

The design and thresholds were frozen in `../preregistration.md` before the causal results.

## Behavioral regimes

| MLP | rho | seeds passing recall + unseen-template gate | train recall mean | unseen-template mean | structured held-out-pair mean |
|---|---:|---:|---:|---:|---:|
| swiglu | 0 | 3/3 | 0.999 | 0.965 | 0.999 |
| swiglu | 0.5 | 2/3 | 0.988 | 0.929 | 0.431 |
| swiglu | 1 | 3/3 | 0.976 | 0.953 | n/a |
| bilinear | 0 | 3/3 | 0.998 | 0.968 | 0.997 |
| bilinear | 0.5 | 2/3 | 0.983 | 0.926 | 0.429 |
| bilinear | 1 | 2/3 | 0.981 | 0.945 | n/a |

At rho=0.5, the structured held-out score is computed only on unrandomized pairs; arbitrary held-outs are intentionally not capability targets.

## Final-layer representation and causal transfer (gate-passing seeds only)

| MLP | rho | n | semantic R2 top decile | residual fact stability | full semantic patch acc | native top-32 | rotated top-32 | random native-32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| swiglu | 0 | 3 | 0.906 | 0.470 | 0.658 | 0.629 | 0.032 | 0.002 |
| swiglu | 0.5 | 2 | 0.176 | 0.862 | 0.154 | 0.101 | 0.076 | 0.019 |
| swiglu | 1 | 3 | 0.072 | 0.905 | nan | nan | nan | nan |
| bilinear | 0 | 3 | 0.935 | 0.240 | 0.722 | 0.617 | 0.037 | 0.001 |
| bilinear | 0.5 | 2 | 0.141 | 0.854 | 0.263 | 0.143 | 0.125 | 0.037 |
| bilinear | 1 | 2 | 0.056 | 0.853 | nan | nan | nan | nan |

Pure-random rows have almost no naturally correct counterfactual pairs, so their patch accuracy is not interpreted; their residual fact-stability score remains diagnostic.

## Constraint-network audit

| MLP | rho | n | early effective support | early abs cosine | finite-edit Pearson (scale 1) | Pearson (scale 4) |
|---|---:|---:|---:|---:|---:|---:|
| swiglu | 0 | 3 | 11.438 | 0.051 | 0.977 | 0.777 |
| swiglu | 0.5 | 2 | 22.378 | 0.034 | 1.000 | 0.999 |
| swiglu | 1 | 3 | 24.582 | 0.068 | 1.000 | 0.996 |
| bilinear | 0 | 3 | 12.919 | 0.051 | 0.995 | 0.931 |
| bilinear | 0.5 | 2 | 22.714 | 0.041 | 1.000 | 0.998 |
| bilinear | 1 | 2 | 25.080 | 0.068 | 1.000 | 0.997 |

## Verdict

1. **Validated:** pure reusable grammar produces a causally transferable computation. In all six pure-rule depth-4 models, a mean activation change transfers across unrelated entity names. The effect is sparse in the trained channel basis: 32 native channels substantially outperform both 32 random native channels and 32 coordinates of a random rotation.
2. **Validated:** arbitrary assignments produce stable idiosyncratic activation state and broader signed constraint coupling. Residual fact stability remains high after the restricted semantic audit, while early-block effective support is larger than for pure rules.
3. **Validated but not memorization-specific:** signed gradient overlap is an accurate local causal account in rule, mixed, and random regimes. It is therefore a general tangent-space description, not a classifier for memorization and not proof of KKT convergence.
4. **Supported:** mixing rules with 50% arbitrary exceptions is a distinct competition regime. Models can recall both subsets, yet unrandomized held-out rule accuracy falls near 0.42 and semantic patches transfer poorly. Two seed-2 mixed runs also fail the unseen-template fidelity gate even with near-perfect trained-template recall.
5. **Falsified:** the regimes are not a single axis from monosemantic rules to a KKT web. Neuron-axis alignment, semantic transfer, residual fact state, constraint breadth, and behavioral fidelity can vary independently.
6. **Not established:** one neuron equals one human concept; global KKT optimality; or that all natural-language factual memory has the synthetic mixed regime's organization.
