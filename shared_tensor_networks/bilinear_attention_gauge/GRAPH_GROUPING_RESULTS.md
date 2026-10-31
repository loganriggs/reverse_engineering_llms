# Discovering variable-size sparse-atom groups without a pairing oracle

The [first unlabeled grouping test](UNSUPERVISED_GROUPING_RESULTS.md) recovered A/B groups from four sparse axes, but it supplied the answer's structure: exactly two groups of two. This follow-up uses the **rank-five** sparse dictionary fitted on ordinary behavior and all three branch-edit conditions. It lets a correlation graph decide how many groups exist and how large they are. The extra direction makes the two-pairs-of-two rule inapplicable.

For each of ten trained two-head toys, take 32, 128, or 512 newly generated unlabeled training-distribution sentences. Sum each of the five key-dictionary coordinates over the sentence's three key tokens. Connect two axes when the absolute Pearson correlation of these summed activations exceeds a fixed threshold; return connected components. No group count, size, token-family name, output target, or bit-flip response enters the graph. Thresholds .6, .7, .8, and .9 are swept to expose sensitivity rather than selecting one by its task score.

After grouping, score the full 96-sentence exhaustive set, including the rare A0/B1 swap combination withheld from training. The initial table below uses an **effect-only** criterion: at least one group with >90% A-flip effect energy and one with >90% B-flip effect energy; extra groups are permitted. This is a post-fit test on the reduced explanatory function. A later [grammar-shift red team](GROUPING_GRAMMAR_SHIFT_RESULTS.md) found that this criterion can count an isolated token-specific axis as a whole variable. The stricter **complete-group** criterion also requires >90% coverage of each signed A0↔A1 and B0↔B1 contrast by its group. Neither criterion enters fitting. The same graph rule is applied to SVD and 128 random rotations per seed.

| Unlabeled sentences | Correlation threshold | Sparse effect-only success | SVD effect-only success | Random-gauge effect-only success rate |
|---:|---:|---:|---:|---:|
| 32 | .6 | **10/10** | 3/10 | .09 |
| 32 | .8 | **10/10** | 3/10 | .10 |
| 32 | .9 | **10/10** | 3/10 | .12 |
| 128 | .8 | **10/10** | 3/10 | .10 |
| 512 | .8 | **10/10** | 3/10 | .10 |

At threshold .8 and 32 sentences, the sparse graph returns **three groups in every seed**. Two have selective A/B effects in all ten seeds, but only **9/10** recover both *complete signed contrasts*: seed 5 leaves an A-like singleton. The third group is filler-dominant with >99.9% effect purity in nine seeds; in the remaining seed it is an A-dominant mixed group with .762 purity. Deleting either selected A/B group from the reduced original function leaves the other bit at **100% exhaustive accuracy** in every run, while the affected bit falls to mean **50.3%**. At thresholds .6 and .7, the graph also returns three groups in all ten seeds; at .9 it splits groups in two seeds and complete-group recovery is 8/10. Effect-only results remain stable across tested sample sizes, but the grammar-shift red team shows they are not a sufficient measure of full variable recovery.

This removes the explicit two-pairs-of-two prior and supports a practical featurizer pattern: first find a function/intervention-aware subspace, rotate to sparse atoms, then infer reusable groups from unlabeled activation statistics. It is still a highly structured toy with five axes, six key tokens, and a grammar that places exactly one token from each family in every sentence. The graph threshold is hand-specified, and the criterion for calling a group A/B is post-fit supervised evaluation. The corrected result is **9/10 complete-variable recovery under a simple grammar**, not unrestricted discovery in real language or physical-node identification. The grammar-shift test shows that variable counts can erase the correlation signal entirely.

Run `dual_graph_grouping.py` after `dual_intervention_sparse_basis.py`. `dual_graph_grouping.json` records groups, correlations, causal group effects, ablation accuracies, and random-gauge controls across seeds, sample sizes, and thresholds.
