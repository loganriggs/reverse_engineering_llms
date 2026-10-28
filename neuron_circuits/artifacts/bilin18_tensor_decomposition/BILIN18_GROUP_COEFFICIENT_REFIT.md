## H729 — Coefficient refitting and the best possible pair within frozen memberships

H728 showed that exact group selection with frozen encoder coefficients recovered only a small part of overlapping groups' reconstruction deficit. Here we distinguish coefficient mismatch from the limitations of the frozen membership inventory.

We evaluate three nested procedures: (1) H728's best pair with original encoder coefficients; (2) least-squares projection onto that same support; and (3) exhaustive selection of the group pair with the best least-squares projection. The third optimizes both pair selection and coefficients within the existing dictionary and memberships. The comparator refits coefficients on encoder-selected scalar Top-8 supports. This gives the grouped model an exhaustive support search while leaving the scalar supports fixed; it is not a globally optimized scalar sparse-coding comparison.

The dictionary, bias and memberships are unchanged. Each projection uses only its input activation; no parameters are fit across validation examples. The evaluation reuses H728's 32 documents at 16 sampled positions, 512 vectors per head. All three dictionary seeds and all 54 membership sets are included.

Relative write MSE (lower is better; averages over dictionary seeds and, for controls, three rewires per seed):

| Head / memberships | Frozen coefficients, best pair | Refit same support | Best pair and refitted coefficients | Refit scalar Top-8 |
|---|---:|---:|---:|---:|
| Agreement, raw | 0.4377 | 0.4135 | 0.4070 | 0.3223 |
| Agreement, raw rewires | 0.4612 | 0.4320 | 0.4262 | 0.3223 |
| Agreement, conditioned | 0.4547 | 0.4275 | 0.4218 | 0.3223 |
| Agreement, conditioned rewires | 0.4737 | 0.4424 | 0.4364 | 0.3223 |
| Agreement, disjoint | 0.5154 | 0.4808 | 0.4779 | 0.3223 |
| Capitals, raw | 0.5191 | 0.5026 | 0.5008 | 0.4193 |
| Capitals, raw rewires | 0.5482 | 0.5275 | 0.5244 | 0.4193 |
| Capitals, conditioned | 0.5236 | 0.5057 | 0.5037 | 0.4193 |
| Capitals, conditioned rewires | 0.5502 | 0.5293 | 0.5267 | 0.4193 |
| Capitals, disjoint | 0.5918 | 0.5694 | 0.5677 | 0.4193 |

Refitting coefficients helps, and allowing the selected pair to change helps further. Nevertheless, even the best pair with optimally refitted coefficients remains worse than scalar Top-8 with the same refitting. For these frozen dictionaries and memberships, the membership inventory is a substantive constraint. This does not prove the model fundamentally uses scalar features: a dictionary trained for scalar selection, heuristic correlation memberships, fixed four-atom groups, and a two-group budget all remain assumptions that could disadvantage a grouped representation.

The candidate enumeration accounts for overlap exactly: shared atoms contribute once. Projection handles rank deficiency by SVD rather than assuming independent directions. Exhaustive search uses float32 bases, and the winning reconstructions are recomputed with double-precision SVD. Near ties and rank thresholds remain numerical qualifications to the optimum claim.

Audits: all input/source hashes remained unchanged, all recorded MSEs were finite, maximum float32 score versus double-projection error divided by input energy was 3.33e-06, maximum best-pair excess error versus refitting the old support was 0, and maximum basis orthogonality error was 9e-06. Two tests passed: rank-deficient supports against an independent pseudoinverse, and exhaustive selection against independent least-squares solves. Zero model forwards, new documents or reserved-test evaluations were used.

This is a reconstruction diagnostic, not evidence of causal reuse, semantic categories, hierarchy, or manifolds. No new output-KL, CE or task-effect claim follows from it. Higher-fidelity scalar featurizers remain a useful baseline for geometry and intervention studies while group-learning alternatives are improved.

Next: test a membership-learning objective tied directly to captured support energy or reconstruction, with identical inventory and active budgets, and disjoint/rewired controls. That distinguishes poor correlation heuristics from the broader group restriction without attributing the failure to gates again. Continue toward fresh diverse-task and selective shared-component tests for any successful candidate; do not extend the current grouping into a hierarchy before it clears those checks.

### Hypothesis refresh: what the current evidence does and does not favor

These are competing explanations, not twenty discoveries. Priorities describe the next experiments' information value, not calibrated probabilities.

| # | Explanation | Current standing | Discriminating test |
|---|---|---|---|
| 1 | Useful computation uses flexible combinations of scalar directions rather than a small fixed inventory of blocks. | Leading description of current reconstruction results; not established ontology. | Learn memberships directly for reconstruction and test whether group restrictions still lose at matched support-description budgets. |
| 2 | Correlation-derived memberships miss the true functional groupings. | Leading alternative to 1. | Train memberships on reconstruction with frozen dictionaries; validate across seeds and unseen contexts. |
| 3 | The scalar-trained encoder gives poor coefficients after group selection. | Directly tested here; some contribution, magnitude measured above. | Compare frozen-support refitting and joint pair/refit optimum against equally refitted scalar supports. |
| 4 | Greedy coefficient-energy gating is the dominant problem. | Weakened by H728; cannot explain most of its gap. | Exact pair enumeration already supplies the relevant counterfactual. |
| 5 | Two groups of four atoms is an inappropriate granularity. | Open. | Predeclare group-size and group-count sweeps, comparing error versus active dimension and support-description cost. |
| 6 | Multiple groups need many shared atoms active simultaneously. | Weak support in this pilot: selected unions almost always contain eight different atoms. | Vary overlap while preserving atom degrees and test shared-atom intervention specificity. |
| 7 | Group dictionaries must be trained jointly; freezing scalar dictionaries handicaps blocks. | Open and important. | Warm-start controlled joint training, with matched scalar retraining and synthetic recovery checks. |
| 8 | Decoder nonorthogonality explains most group failures. | Partially addressed by exact reconstruction and projection; not automatically ruled out for training. | Compare whitened/orthogonal and free dictionaries with identical group-search procedures. |
| 9 | Only a minority of coordinates form meaningful groups; forcing every atom into a group dilutes them. | Open. | Mixed model allowing singleton residual atoms plus selected reusable groups, charging memberships explicitly. |
| 10 | Geometry is curved and context dependent, so linear blocks are the wrong representation. | Open; current experiments test unions of linear spans. | Compare local tangent prediction against dimension- and density-matched linear/null models on disjoint contexts. |
| 11 | Apparent groups mainly reflect lexical identity or position. | Not settled: conditioned groups retain some usefulness but conditioning changes sample coverage. | Equal-sample matched conditioning, then lexical/position transfer with frozen groups. |
| 12 | Groups reflect genuine joint computation rather than mere input covariance. | Not established by reconstruction or whole-head swaps. | Remove a shared component and test predicted effects on each consumer, with unrelated consumers as controls. |
| 13 | Capital recall is more distributed than agreement at the chosen heads. | Supported descriptively by PCA/sparse intervention retention, site specific. | Match competence and intervention effect size across new prompts and additional heads. |
| 14 | Agreement's apparent simplicity is specific to the task panel. | Open. | Fresh grammatical constructions, lexicons and nuisance transformations with frozen featurizers. |
| 15 | Aggregate effect retention hides failure on individual examples. | Relevant risk; RMS effect errors already show losses. | Per-example and per-family error distributions, sign reversals and worst-group fidelity. |
| 16 | Directions that dominate write variance are not those that matter to downstream logits. | Supported by differing reconstruction/task/KL rankings. | Compare input-only fitting with explicitly output-weighted fitting, reporting objectives separately. |
| 17 | Redundant alternative bases describe the same computation, so atom instability need not mean failure. | Plausible; raw matching alone cannot decide. | Compare reconstructed subspaces and intervention effects under aligned basis transformations. |
| 18 | Hierarchy is a useful description of activation patterns but not an identifiable additive computation. | Compatible with earlier synthetic nonidentifiability results. | Test implication-gate predictions separately from additive ancestor/child interventions. |
| 19 | Reuse occurs across layers or attention routing rather than within one head's activation groups. | Open and relevant to prior causal-head results. | Trace a validated component through upstream routing and downstream consumers across contexts. |
| 20 | Reconstruction improvements can coexist with worse output faithfulness or task performance. | Already observed as a practical distinction. | Keep KL, CE, per-example task effects, nuisance sensitivity and accuracy as separate evaluation axes. |

The high-value fork is between flexible scalar combinations (1) and poorly inferred or constrained groups (2, 5, 7, 9). Gate-only tuning (4) is now lower priority. Geometry/manifold analysis should use the strongest available scalar baseline alongside group candidates, rather than waiting for a grouped architecture to win by assumption. Any interpretation must survive fresh diverse tasks and selective interventions; these two-head assays do not satisfy that requirement alone.
