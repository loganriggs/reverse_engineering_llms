# A fixed first-layer token basis depends on its input metric

The [per-prefix joint source search](NESTED_JOINT_SOURCE_SEARCH_RESULTS.md) found compact source subsets, but each mask required teacher-output fitting on that prefix. This experiment tests a **single reusable basis** of the first-layer attention-score function instead. For each head and relative lag, the trained model defines a vocabulary-by-vocabulary matrix of the product `QK1 × QK2`, with token embeddings and per-token RMSNorm folded in. `eval_nested_shared_token_basis.py` forms this weight-derived table over all 319×319 token pairs, then chooses one query-token and one key-token subspace **per head, shared across all 128 lags**. It projects every lag matrix through those same two subspaces. Rank is token-mode dimension per head, not the number of retained attention source positions and not a count of model parameters.

The subspace is selected by eigendecomposing the query and key Grams of the score table. Three metrics define the Grams:

* **Uniform:** all vocabulary entries receive equal weight; no examples are used.
* **Nested-unlabeled:** smoothed token frequencies from 2,048 independently generated nested-stream sequences (262,144 tokens); no task labels or causal edits are used.
* **No-bracket fit:** the same amount of unlabeled data from the earlier varied-language generator, which contains the verb rules but no brackets. Only basis weighting holds brackets out; the underlying model was trained on bracket data.

All bases are fitted separately from the causal-edit and verb evaluation panels. Full-table replay has mean teacher KL below `3e-6` at verb sites and below `6e-7` at outer bracket sites; the small discrepancy comes from representing bfloat16 RoPE phases by relative lag. A built-in random-tensor check confirms that full-rank two-sided projection reconstructs its input to `3.5e-6` maximum absolute error and that weighted reconstruction error decreases with rank.

| Independent panel and basis | Rank | Outer opener-edit L1 | Outer edited accuracy | Verb oracle KL |
|---|---:|---:|---:|---:|
| Seed 20261213, native model | full | .0022 | 100% | .2545 |
| Uniform | 8 | .2887 | 96.2% | .2989 |
| Nested-unlabeled | 8 | **.0088** | **100%** | .2721 |
| No-bracket fit | 8 | 1.0045 | 82.8% | .2681 |
| Uniform / nested-unlabeled / no-bracket | 16 | .0212 / **.0026** / .0619 | 100% each | .2655 / **.2606** / .2605 |
| Seed 20261216, native model | full | .0022 | 100% | .2524 |
| Uniform | 8 | .3187 | 93.4% | .2954 |
| Nested-unlabeled | 8 | **.0088** | **100%** | .2722 |
| No-bracket fit | 8 | 1.0167 | 76.1% | .2707 |
| Uniform / nested-unlabeled / no-bracket | 16 | .020 / **.003** / .073 | 100% each | .263 / **.258** / .258 |

Each panel has 238–259 outer and 114–120 inner relevant edits, matched no-effect controls, and 256 verb sites. At rank 8, all three bases retain the inner opener-edit effect within L1 .008–.013, so the large separation is specifically in the **outer** dependency. On the two panels, outer no-effect control L1 is about .0009 for the nested-unlabeled basis, versus .027–.029 for uniform and .050–.051 for no-bracket fit. The nested-unlabeled rank-8 basis is not fully faithful to the verb task: its verb KL remains about .018–.020 above native. Rank 16 narrows that gap and almost fully preserves outer edits. This is a tradeoff across tasks, not a universal win for data weighting.

`audit_nested_shared_basis_stability.py` recomputes bases from two independent 2,048-stream fit samples. For rank 8, mean squared subspace overlap across the four heads is **.9991** for query modes and **.9994** for key modes (1 means identical). The same overlap between nested-fit and no-bracket-fit bases is .857/.893, so the task-weighting change is much larger than fit-sample noise. In the weighted orthonormal coordinates, mean rank-8 bracket-token leverage across heads is .161 query/.165 key for nested fit, versus .0025/.0011 for no-bracket fit. Uniform gives .076/.097. These leverage values show which token rows enter the selected weighted subspaces; they do not by themselves identify a semantic circuit.

A new seed-20261218 edit panel checks the frequency pseudocount. At rank 8, changing smoothing from 1 to 100 leaves nested-fit outer edit L1 at **.004** and uniform at .333. The no-bracket fit remains poor, though it improves from 1.001 to .787. Thus the main separation is not specific to smoothing 16; the exact size of the held-out-task failure does depend on that prior.

This is a limited but concrete canonicalization result: the **effective first-layer score function** has a reproducible low-dimensional token basis under a specified metric, and choosing that metric changes which behaviors survive truncation. The score table is invariant to internal Q/K factor gauges that preserve its product, but the result does not canonicalize OV, redistribute redundant heads, or prove a unique decomposition of the full language-model function. Nor is the table-basis rank directly a parameter-compression ratio against the already factored native model. The synthetic corpus remains far from FineWeb form statistics; its frequencies are a controlled prior, not a model of natural-language prevalence.

Next, test whether the basis supports **legible and causal** token-pair components: identify bracket-sensitive modes on one panel, intervene on those modes on disjoint bracket and verb panels, and ask whether a basis fitted to more realistic task mixtures retains the same behavior. Raw results: `surface_mixture_v2_nested_varlen_depth1/shared_token_basis_seed20261213.json`, `shared_token_basis_seed20261216.json`, `shared_token_basis_smooth1_seed20261218.json`, `shared_token_basis_smooth100_seed20261218.json`, and `shared_basis_stability.json`.
