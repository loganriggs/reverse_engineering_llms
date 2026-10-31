# Grouping sparse attention atoms by their function, without sentence co-occurrence

The [grammar-shift red team](GROUPING_GRAMMAR_SHIFT_RESULTS.md) found that unlabeled co-occurrence can stop pairing A0/A1 and B0/B1 atoms when token counts vary. The same sparse dictionary still describes the trained attention function, so this experiment asks whether a **weight-derived functional relation** can group those atoms without relying on a particular sentence distribution.

For each of ten trained two-head toys, use the saved rank-five key dictionary fitted to original and branch-edited head tables, then sparsified without task labels. For each atom, multiply its key direction by the exact **combined original** \(4\times6\) query/output token table to get a four-dimensional response vector. Connect two atoms when the absolute cosine between their response vectors exceeds a fixed threshold; return graph connected components. The graph sees no sentence samples, A/B labels, or bit-flip data. Test thresholds .6, .7, .8, and .9; apply the same rule to SVD and 128 random orthogonal gauges per seed.

Evaluation is stricter than the original effect-only grouping test: a recovered A or B group must have >90% post-fit bit-flip effect purity **and** cover >90% of the full signed A0↔A1 or B0↔B1 key contrast. Extra groups are allowed.

| Absolute response-cosine threshold | Sparse complete groups | SVD complete groups | Random-gauge success rate |
|---:|---:|---:|---:|
| .6 | 8/10 | 1/10 | .117 |
| .7 | 8/10 | 2/10 | .151 |
| .8 | **9/10** | 2/10 | .207 |
| .9 | **9/10** | 2/10 | .102 |

At threshold .8, the only sparse failure is seed 5. Its A-like singleton has a selective effect, but the graph does not reunite enough axes to cover the entire A contrast; this is the same case flagged by the corrected co-occurrence criterion. In the other nine seeds, functional grouping recovers complete A/B variables **without seeing any unlabeled sentence distribution**. This avoids the independent-Poisson failure of co-occurrence grouping by using a different source of information: the atoms' response pattern across query types and output channels.

This does **not** imply data-free recovery of arbitrary semantic concepts. The toy has a four-dimensional query/output signature tailored to two known task bits, the dictionary is sparse in a token basis, and success is evaluated post-fit with known contrasts. Functional similarity can also merge atoms that share outputs for different reasons. The useful object tested here is the **ordinary combined function** for semantic grouping; intervention tables remain necessary for a separate claim of branch-robust causal fidelity.

This suggests a concrete next featurizer comparison on a richer trained grammar: group sparse atoms using unlabeled co-occurrence, functional-response similarity, and their combination, with task labels hidden until evaluation. Include independent or variable token counts, distractor outputs, whole-task holdouts, and held-out internal interventions so that semantic grouping, task prediction, and causal transfer remain distinct.

Run `dual_functional_grouping.py` after `dual_intervention_sparse_basis.py`. `dual_functional_grouping.json` contains per-seed group graphs, response similarities, post-fit contrast coverage and effect purity, plus SVD and random-gauge controls.
