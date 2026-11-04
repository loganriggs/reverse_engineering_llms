# Weight-L1 OV gauge on a learned natural-carrier rule

This tests a weight-only prior on the two previously trained depth-2 sports alias/bracket models. The learned bracket behavior is eligible for interpretation; alias is weaker and remains a fidelity control. It is a local value/output (`O @ V`) experiment, not a canonicalization of attention or a recovered human-readable circuit.

For each of eight heads, the evaluator takes the exact rank-16 SVD of `O @ V`, balances its two factors, and fits a 16-dimensional orthogonal rotation minimizing their combined smooth L1 norm. This rotation preserves `O @ V` exactly at full rank. It then retains the 4 or 8 whole internal coordinates with highest factor-norm product. The comparator is the optimal weight-Frobenius rank-4 or rank-8 SVD of the same `O @ V`. Neither gauge fit sees examples, activations, or task labels. Both methods are evaluated on the same 40 held-out-game panel per seed.

| Seed | Sum of native / balanced-SVD / rotated factor L1 | Rank | Rotated bracket paired top-1 | SVD bracket paired top-1 | Rotated / SVD bracket teacher KL | Rotated / SVD background teacher KL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 98201 | 1270.9 / 1177.8 / 1066.4 | 4 | .575 | .900 | .465 / .043 | .477 / .268 |
| 98201 | same | 8 | .950 | .925 | .069 / .008 | .191 / .085 |
| 98202 | 1262.3 / 1167.9 / 1061.8 | 4 | .300 | .500 | 1.281 / .973 | .447 / .316 |
| 98202 | same | 8 | .413 | .763 | .891 / .362 | .177 / .130 |

Native full-rank bracket paired top-1 is .925/.950 across seeds; native alias paired top-1 is .288/.338. At rank 8, rotated/SVD alias top-1 is .100/.125 for seed 98201 and .000/.088 for seed 98202. The rank-16 rotated models recover native behavior, with per-head `O @ V` relative error below `6.2e-15`, maximum sampled logit change below `4.2e-5`, and numerical-zero teacher KL. Thus the local gauge implementation is function preserving before truncation.

The rotation lowers factor L1 by about 16% versus native and 9–10% versus balanced SVD, yet SVD has lower bracket and background teacher KL at both truncated ranks in both seeds. A single seed's bracket top-1 reversal at rank 8 (.950 versus .925) is misleading without the KL and second seed. This rejects the simple hypothesis that factor L1 minimization alone selects a functionally better small coordinate basis. Smooth L1 rotation also does not create exact zero weights, so this is a basis-selection test, not a sparse circuit extraction result. A more relevant next prior would optimize edge sparsity *and* task-agnostic functional fidelity, then test causal response and cross-rule transfer under matched total edge and background-fidelity budgets. Keep the primary work on a varied sampled, meaning-first document DGP; this bounded method result should not become another extended single-task detour.

The [evaluator](eval_sports_sparse_ov_gauge.py) and aggregate [seed 98201](results/sports_sparse_ov_gauge_seed98201.json) and [seed 98202](results/sports_sparse_ov_gauge_seed98202.json) reproduce this result. Checkpoints stay local.
