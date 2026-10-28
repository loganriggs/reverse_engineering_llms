# Three-seed constraint-network confirmation

All runs use independently generated manifests and initializations. The seed-0 row is the 128-fact sensitivity panel on the original checkpoint; seeds 1 and 2 are newly trained matched cells. All arbitrary models met the same factual fidelity gate before auditing.

## Paired load effects

Ratios are fully arbitrary (`rho=1`) divided by structured (`rho=0`). Values below one mean weaker coupling/collateral under arbitrary load; support above one means effects are distributed over more facts.

| seed | width | residual-coupling ratio | collateral ratio | effective-support ratio |
|---:|---:|---:|---:|---:|
| 0 | 128 | 0.258 | 0.224 | 1.869 |
| 0 | 256 | 0.133 | 0.180 | 3.830 |
| 1 | 128 | 0.204 | 0.196 | 2.050 |
| 1 | 256 | 0.100 | 0.093 | 2.850 |
| 2 | 128 | 0.249 | 0.119 | 2.264 |
| 2 | 256 | 0.115 | 0.139 | 3.658 |

## Replicated causal prediction

The table reports the one-logit finite-edit prediction on fully arbitrary models.

| seed | width | Pearson | origin R-squared | permuted-source Pearson |
|---:|---:|---:|---:|---:|
| 0 | 128 | 1.000 | 1.000 | 0.007 |
| 0 | 256 | 1.000 | 1.000 | -0.029 |
| 1 | 128 | 1.000 | 1.000 | -0.017 |
| 1 | 256 | 1.000 | 1.000 | -0.003 |
| 2 | 128 | 1.000 | 1.000 | -0.025 |
| 2 | 256 | 1.000 | 1.000 | -0.005 |

## Verdict

- Finite constraint prediction replicated: **True**
- Stronger arbitrary web falsified in all six paired cells: **True**
- Broader arbitrary support observed in all six paired cells: **True**

The robust result is not that arbitrary facts lack a network. Their signed sensitivities touch more of the sampled facts, but each interaction and its finite collateral are much smaller. The high-fit AdamW transformers therefore realize a diffuse, weakly coupled geometry rather than the taut KKT-like web seen in the saturated SGD bilinear model.
