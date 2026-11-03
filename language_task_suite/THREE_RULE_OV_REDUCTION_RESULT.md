# OV rank reduction on the three-rule natural-report model

This experiment asks whether a canonical value/output factorization preserves a **learned rule**, rather than merely reconstructing weights or average activations. It uses the two trained three-rule bilinear checkpoints and a fresh game-disjoint panel of 40 heldout reports × eight edits per checkpoint. The native models get 100% paired bracket closure on this particular panel. Alias remains behaviorally weak; capitalization is confounded by the common `The/the` continuation.

For every attention head in both layers, form the exact product `O_h V_h`. At full rank 16, an SVD with balanced factors changes no logits (probe maximum < .001). For lower ranks, compare truncation by ordinary weight Frobenius norm with truncation by the covariance of value inputs on 80 separate fit reports. The latter is data-aware but **unsupervised with respect to the rule**. Teacher-to-reduced KL is measured at all three task sites and at sampled background positions; paired answers are measured on the heldout eight-arm reports.

| Checkpoint | Rank/head | Method | Bracket paired exact | Bracket KL | Mean input-weighted OV error | Background KL |
|---|---:|---|---:|---:|---:|---:|
| 55301 | 4 | weight | 1.000 | .057 | .610 | .228 |
| 55301 | 4 | data | .875 | .188 | .565 | .146 |
| 55302 | 4 | weight | .988 | .035 | .627 | .492 |
| 55302 | 4 | data | .494 | 1.465 | .586 | .356 |
| 55301 | 8 | weight | 1.000 | .006 | .367 | .061 |
| 55301 | 8 | data | .988 | .002 | .328 | .046 |
| 55302 | 8 | weight | 1.000 | .001 | .388 | .150 |
| 55302 | 8 | data | 1.000 | .001 | .349 | .138 |

The data-aware method does what its objective promises: it lowers mean activation-weighted OV reconstruction error and background KL at the same rank. Yet at rank 4 it harms bracket fidelity in both checkpoints. This is direct evidence that average activation covariance is an insufficient interpretability prior for this rule. Rank 8 largely closes the gap, but it retains half the head channels. On checkpoint 55302, rank 2 bracket paired exact is .744 weight-only versus .000 data-aware; rank 1 is .125 versus .000. These severe reductions expose a useful target for a **rule-site or causal-effect-preserving** low-rank objective.

This is an OV-only result. It does not establish a canonical attention function, a sparse circuit, or a recovered ground-truth algorithm. The weight-only win could reflect the learned bracket direction's weight singular value, while the data covariance emphasizes common non-rule contexts. The next comparison should fit rank-constrained factors to preserve teacher logits and/or opener-edit responses specifically at bracket sites, while keeping game-disjoint fit/test reports and reporting background KL as a tradeoff. A separate whole-combination holdout is still needed for claims about reusable computation across rules.

Reproduce with `eval_three_rule_ov_canonical.py`. The underlying decomposition helpers are in `eval_sports_ov_canonical_truncation.py`; aggregate outputs are `results/three_rule_ov_canonical_seed55301_n80_n40.json`, `results/three_rule_ov_canonical_seed55302_n80_n40.json`, and the seed55302 rank1/2 file. Model checkpoints stay local.
