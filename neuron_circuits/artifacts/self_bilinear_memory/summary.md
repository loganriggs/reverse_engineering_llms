# Residual self-bilinear memory and compositional load

The main model has a constant-width residual stream with `x_(l+1) = x_l + D_l((L_l x_l) ⊙ (R_l x_l)) + b_l` and a bias-free final readout. `L` and `R` are also bias-free, so disabling decoder bias removes every learned constant write. The model memorizes 128 random binary facts with 64 bilinear neurons per layer. A computation query asks for the parity/product of exactly 2, 4, or 8 stored facts; held-out queries contain unseen fact combinations.

A separate exact-architecture capacity sweep reached 100% lookup from one through eight memorized facts per neuron; all runs met the same loss-convergence criterion.

## Computation-load sweep (three layers; means over three seeds)

| distinct computation tasks | loss | lookup | held-out 2-fact | held-out 4-fact | held-out 8-fact |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.00e+00 | 1.000 | 0.563 | 0.554 | 0.537 |
| 192 | 4.58e-09 | 1.000 | 0.499 | 0.505 | 0.487 |
| 768 | 2.70e-07 | 1.000 | 0.548 | 0.502 | 0.490 |
| 3072 | 1.35e-05 | 1.000 | 0.587 | 0.493 | 0.479 |
| 12288 | 3.88e-06 | 1.000 | 1.000 | 1.000 | 0.999 |
| 18432 | 6.33e-05 | 1.000 | 1.000 | 1.000 | 1.000 |

## Residual-stream and decoder-bias ablation

All four architectures have identical stream width, bilinear-neuron count, depth, and bias-free output readout. Only the identity additions and/or the biases on `D` are switched off. Values are means over seeds at the maximum computation load.

| residual | decoder bias | converged seeds | loss | lookup | held-out 2 | held-out 4 | held-out 8 |
|---|---|---:|---:|---:|---:|---:|---:|
| yes | yes | 3/3 | 6.33e-05 | 1.000 | 1.000 | 1.000 | 1.000 |
| yes | no | 3/3 | 6.00e-05 | 1.000 | 1.000 | 1.000 | 0.823 |
| no | yes | 2/3 | 2.31e-01 | 0.833 | 0.826 | 0.839 | 0.835 |
| no | no | 0/3 | 6.93e-01 | 0.503 | 0.487 | 0.514 | 0.507 |

All models in the computation-load sweep met the convergence criterion on their training distribution. The ablations are deliberately reported separately because some did not converge.

## Neuron requirements at maximum computation load

Accuracy counts use one shared mask for the entire task family, ranked by mean intact `|activation × gradient|` and verified with fresh masked forwards. The final column instead uses input-specific masks to measure each individual fact/query's margin fidelity.

| family | 80% accuracy | 90% accuracy | 99% accuracy | median neurons for 90% margin |
|---|---:|---:|---:|---:|
| lookup | 16 | 16 | 24 | 128 |
| parity_2 | 48 | 96 | 96 | 128 |
| parity_4 | 96 | 96 | 96 | 128 |
| parity_8 | 96 | 96 | 96 | 128 |

The per-example margin statistic answers the fact-level question more strictly than correctness: it asks how many neuron activations preserve 90% of that particular fact or query's intact decision margin.

## Cross-entropy recovery at maximum computation load

CE masks are ranked independently using `|activation × ∂CE/∂activation|`. Recovery is `(log(2) - masked CE) / (log(2) - intact CE)`, so 0% is chance CE and 100% is the intact model. Counts below use one shared mask for the task family.

| family | intact CE | 80% CE recovery | 90% CE recovery | 99% CE recovery | CE/margin rank correlation |
|---|---:|---:|---:|---:|---:|
| lookup | 0.000e+00 | 128 | 128 | 128 | 0.585 |
| parity_2 | 0.000e+00 | 96 | 96 | 96 | 0.735 |
| parity_4 | 0.000e+00 | 96 | 96 | 96 | 0.922 |
| parity_8 | 9.479e-05 | 96 | 96 | 128 | 0.947 |

For a single example in this binary task, CE is a monotone function of the correct-logit margin, so CE and margin give the same within-example neuron ordering up to numerical ties. Their shared-mask rankings can differ because CE's softmax gradient weights uncertain or incorrect examples more heavily.

## Operational monosemanticity

For every neuron, candidate explanatory features are individual fact presence, query arity, and target sign. Its score is the strongest feature's r² multiplied by that feature's share of total candidate-feature r². This distinguishes a strong single-feature neuron from a neuron weakly correlated with many facts.

The corresponding per-layer cleanliness fractions, top features, and final-layer value signal-to-interference measurements are recorded directly in `results.json`; the summary no longer hard-codes values from the pre-residual architecture.

## Ablation interpretation

The full residual-plus-bias model was the only condition with perfect held-out generalization in every seed. Keeping residuals but removing decoder bias still converged in 3/3 seeds, although one seed failed to generalize 8-fact parity (mean 0.823). Removing residuals while retaining bias was brittle: 2/3 seeds converged and solved the task, while one remained at chance, yielding means near 0.83. Removing both produced loss `log(2) = 0.693`, chance accuracy, and 0/3 convergence. Thus residual transmission is the stronger stabilizer, decoder bias can partly substitute for it, and removing both causes a multiplicative signal/gradient collapse under the matched initialization.

The load sweep in `results.json` reports this distribution by layer, alongside a value signal-to-interference ratio, shared versus input-specific circuit sizes, and exact quadratic matrices for reconstruction from the saved model.

## Limits

The facts are random binary values and computation is parity, chosen because dependency size and polynomial degree are exact. The first-layer input has one coordinate per fact, so storage parameters still scale with the number of possible keys; the experiment concerns shared neuron width, interference, and reusable computation rather than parameter-free compression.
