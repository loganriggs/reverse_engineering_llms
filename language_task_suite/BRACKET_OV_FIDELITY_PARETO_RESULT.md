# Balancing rule fidelity with ordinary predictions

The earlier rank-2 OV fit recovered bracket behavior but raised teacher-to-student KL at ordinary report positions. This follow-up trains the same two-channel-per-head factors with

`loss = bracket-site KL + λ × background KL`,

where background sites are every 32nd valid next-token position, excluding all three rule sites. The frozen teacher and fit reports are the same for each λ. Only retained OV factors are updated for 300 steps; the unused channels remain exactly zero. Evaluation uses 40 heldout game-disjoint reports with all eight controlled edits and a separate background KL sample. The λ=0 baselines are from the preceding run under the same initialization and panel seed.

| Checkpoint | λ | Heldout bracket paired exact | Bracket KL | Background KL | Token NLL |
|---|---:|---:|---:|---:|---:|
| 55301 | 0 | .950 | .032 | .379 | 5.247 |
| 55301 | .03 | .950 | .032 | .310 | 5.157 |
| 55301 | .1 | .931 | .040 | .257 | 5.106 |
| 55301 | .3 | .906 | .050 | .230 | 5.097 |
| 55301 | 1 | .919 | .057 | .217 | 5.093 |
| 55302 | 0 | .913 | .172 | 1.154 | 5.575 |
| 55302 | .03 | .900 | .163 | .881 | 5.287 |
| 55302 | .1 | .900 | .162 | .748 | 5.156 |
| 55302 | .3 | .900 | .167 | .618 | 5.046 |
| 55302 | 1 | .875 | .178 | .518 | 4.959 |

Native teacher paired bracket scores on this panel are .931/.906; native token NLL is 5.093/4.685. The rank-2 initialization has paired bracket .700/.625 and background KL .346/.850. Moderate background weighting protects much of the ordinary distribution while retaining nearly the native bracket behavior. For seed55302, λ=.3 reduces background KL by .537 relative to λ=0 with only .013 lower paired bracket accuracy; it still does not match the teacher's token NLL. The seed55301 paired score has small nonmonotonic changes across λ, consistent with a 40-report panel and an accuracy threshold; KL gives a smoother tradeoff.

This demonstrates that the optimization target matters at fixed rank, but it is **not** evidence of an independently discovered or uniquely canonical circuit. Rule sites were supplied, and all eight task combinations were in the fit set. The next convincing test is to train a distribution with at least two behaviorally competent rules and withhold whole combinations or realizations during factor fitting; then ask whether a rule-preserving factor transfers. A larger heldout panel and additional checkpoints would sharpen the Pareto estimate. Separately, the synthetic-language form gap and the natural report's lack of a complete next-token oracle remain open DGP issues.

Run `fit_three_rule_bracket_ov.py --background-weight {0.03,0.1,0.3,1.0}`. The aggregate sweep files are `results/fit_three_rule_bracket_ov_seed{55301,55302}_rank2_bg*.json`; no source prose or model checkpoints are included in Git.
