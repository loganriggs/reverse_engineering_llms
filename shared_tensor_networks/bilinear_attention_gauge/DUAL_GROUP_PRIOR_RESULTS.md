# Family-group priors on the necessary two-head toy language

The [dual-output toy](DUAL_OUTPUT_LANGUAGE_RESULTS.md) needs two scalar heads to predict both independent bits. Ten trained checkpoints are 100% accurate on all 96 exhaustive sequences, including the rare A0/B1 swap combination omitted during training. This follow-up asks whether a **weakly supervised token-family prior** makes a rank-one-per-head reduction more legible or more faithful under interventions.

The supplied groups are key-token families `{A0,A1}`, `{B0,B1}`, and `{F0,F1}`. They come from the generator, so this condition is **not unsupervised discovery**. The fit receives no bit targets, output-rule labels, or rare held-out combination labels. Two group penalties are compared with pair-response L1 using identical rank-one parameterization, four-start L-BFGS optimizer, lambda grid, and per-head function-error allowance:

- **Pair L1:** sum of norms of individual query–key output responses.
- **Family group:** sum of response norms for each key family across both queries.
- **Query-conditional family group:** separate family norms for each query type. This allows a head to favor different key families for common versus rare queries without specifying which family should win.

The fidelity measure is either uniform over the role-valid 2×6 query–key table or empirical pair frequency from 100,000 generated training-law sentences. Candidate strength is chosen by the smallest corresponding prior subject to no more than **0.02 additional relative pair-function MSE** beyond the best rank-one fidelity fit. Exhaustive labels are used only after selection. Analytic prior gradients match central differences to at most `2.2e-10`; the new zero-penalty fits match the earlier rank-one baselines to within `1.1e-9` MSE. Selected gradient norms are below `2.2e-8`.

## Outcomes across ten seeds

Means over 20 head reductions; the final column is mean accuracy on the held-out rare combination across the ten complete two-head models. Effective family support is an inverse-concentration index ranging from 1 to 3; **lower is more concentrated**, not necessarily more meaningful. QK2-constant error measures the same reduced QK1 bond after replacing the other attention branch by a constant.

| Fit measure / prior | Pair MSE | Effective family support | QK2-constant MSE | Rare held-out accuracy |
|---|---:|---:|---:|---:|
| Uniform / pair L1 | .0255 | 2.164 | .0290 | 90% |
| Uniform / family group | .0143 | 2.199 | .0162 | 90% |
| Uniform / conditional family | .0181 | 2.185 | .0203 | 90% |
| Empirical / pair L1 | .0060 | 2.175 | .0638 | 90% |
| Empirical / family group | .0115 | 2.149 | .1074 | 90% |
| Empirical / conditional family | .0037 | 2.184 | .0405 | 90% |

The rare accuracy of 90% means nine seeds remain perfect and **seed 2 fails completely**, as in the earlier fidelity-only comparison. A seed-2 sensitivity run expanding the allowable function error from .02 to .10 still does not recover its held-out rare rule under any of the three priors. The selected candidates spend different amounts of their error allowance, so lower intervention error in a row with lower pair MSE must not be attributed to its prior. At the tested lambda grid, the group-prior advantage in family concentration is small and changes sign across error budgets. This is not evidence for a better mechanistic decomposition.

The intervention panel also scores removal of each key family, flipping each family’s bit token, switching common↔rare query response, replacing QK2 by a constant, and shifting QK2 scores across key tokens. The JSON files contain every candidate and selected response. Pooled A/B-bit-flip effect errors under uniform fitting are `.023/.020` for pair L1, `.013/.010` for global family groups, and `.016/.014` for conditional groups. Those differences track their pair-function error budgets; they do not establish causal superiority. Under empirical fitting, pooled A/B flip errors range from `.030` to `.108` depending on prior. Family F has very low original effect energy, so its relative errors are unstable and should not be used to name a feature.

One structural explanation is that family sparsity is the wrong target for this architecture. The correct task needs both A and B information, and the two queries can request opposite output orders. A rank-one bond per head may need to preserve mixed family responses even when the token families are known. A family-group penalty can make a table look cleaner while discarding a useful cross-family computation. This remains a hypothesis, not a proof: the next experiment should give the toy a representation with explicit reusable components and test whether query-conditioned group assignments predict unseen combinations and internal interventions. More generally, the evaluation should require a proposed component to make **specific held-out intervention predictions**, not just have a small support index.

Reproduce with `taskset -c 2`, `OMP_NUM_THREADS=1`, and `OPENBLAS_NUM_THREADS=1` using `analyze_dual_group.py --seed 0` through `--seed 9`. `dual_group_seed*.json` contain the full sweeps; `dual_group_seed2_allowance0.1.json` is the larger-error sensitivity run. The experiment uses only CPU and does not test the 546M model.
