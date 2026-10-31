# Discovering variable-size sparse-atom groups without a pairing oracle

The [first unlabeled grouping test](UNSUPERVISED_GROUPING_RESULTS.md) recovered A/B groups from four sparse axes, but it supplied the answer's structure: exactly two groups of two. This follow-up uses the **rank-five** sparse dictionary fitted on ordinary behavior and all three branch-edit conditions. It lets a correlation graph decide how many groups exist and how large they are. The extra direction makes the two-pairs-of-two rule inapplicable.

For each of ten trained two-head toys, take 32, 128, or 512 newly generated unlabeled training-distribution sentences. Sum each of the five key-dictionary coordinates over the sentence's three key tokens. Connect two axes when the absolute Pearson correlation of these summed activations exceeds a fixed threshold; return connected components. No group count, size, token-family name, output target, or bit-flip response enters the graph. Thresholds .6, .7, .8, and .9 are swept to expose sensitivity rather than selecting one by its task score.

After grouping, score the full 96-sentence exhaustive set, including the rare A0/B1 swap combination withheld from training. Success requires at least one group with >90% A-flip effect energy and one with >90% B-flip effect energy; extra groups are permitted. This is a **post-fit causal label test** on the reduced explanatory function. Apply the same graph rule to SVD and 128 random rotations per seed.

| Unlabeled sentences | Correlation threshold | Sparse success | SVD success | Random-gauge success rate |
|---:|---:|---:|---:|---:|
| 32 | .6 | **10/10** | 3/10 | .09 |
| 32 | .8 | **10/10** | 3/10 | .10 |
| 32 | .9 | **10/10** | 3/10 | .12 |
| 128 | .8 | **10/10** | 3/10 | .10 |
| 512 | .8 | **10/10** | 3/10 | .10 |

At threshold .8 and 32 sentences, the sparse graph returns **three groups in every seed**. Two have selective A/B effects in all ten seeds. The third is filler-dominant with >99.9% effect purity in nine seeds; in the remaining seed it is an A-dominant mixed group with .762 purity. Deleting either recovered A/B group from the reduced original function leaves the other bit at **100% exhaustive accuracy** in every run, while the affected bit falls to mean **50.3%**. At thresholds .6 and .7, the graph also returns three groups in all ten seeds; at .9 it splits groups in two seeds, but still finds selective A/B groups in all ten. Results are stable across the tested unlabeled sample sizes.

This removes the explicit two-pairs-of-two prior and supports a practical featurizer pattern: first find a function/intervention-aware subspace, rotate to sparse atoms, then infer reusable groups from unlabeled activation statistics. It is still a highly structured toy with five axes, six key tokens, and a grammar that places exactly one token from each family in every sentence. The graph threshold is hand-specified, and the criterion for calling a group A/B is post-fit supervised evaluation. Thus the experiment establishes **variable-size group recovery under a simple grammar**, not unrestricted discovery in real language or physical-node identification. The next decisive step is a richer grammar with variable token counts, distractors, and tasks that do not reduce to mutually exclusive token pairs.

Run `dual_graph_grouping.py` after `dual_intervention_sparse_basis.py`. `dual_graph_grouping.json` records groups, correlations, causal group effects, ablation accuracies, and random-gauge controls across seeds, sample sizes, and thresholds.
