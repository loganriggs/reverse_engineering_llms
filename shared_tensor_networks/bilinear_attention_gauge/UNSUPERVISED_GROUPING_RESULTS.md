# Pairing sparse atoms into task variables from unlabeled sentences

The [intervention-aware sparse-basis result](INTERVENTION_SPARSE_BASIS_RESULTS.md) found four selective token-level axes rather than one axis per A/B bit. This experiment tests whether a simple *unlabeled* co-occurrence rule can pair those axes into reusable groups. It is a small, task-grounded version of the proposed shared-dictionary-plus-gates featurizer: sparse atoms come from the exact token function, while groups are inferred from their activations on sentences.

For each of ten trained two-head toys, use either the rank-four key space fitted on the original function and both single QK-branch edits, or the rank-four space fitted on those plus the double edit. Rotate each space by the previously selected label-free key-L1 objective. From 32, 128, 512, or 2,048 newly generated **unlabeled training-distribution sentences**, sum each dictionary row's coordinate over the three key tokens. Assume only that four axes should form **two disjoint pairs**. Among the three possible pairings, select the one maximizing the sum of absolute within-pair Pearson correlations. This rule sees no A/B family names, target outputs, or bit-flip responses. The training distribution still has its rare A0/B1 swap combination withheld.

After selecting groups, evaluate their effects on the full 96-sentence exhaustive set, including the withheld rare combination. A group counts as selectively causal if the two groups have opposite A/B-dominant bit-flip responses and each puts more than 90% of its A+B+filler flip energy on its dominant bit. Controls apply the **same grouping rule** to SVD axes and to 128 random orthogonal rotations per seed.

| Fit space | Unlabeled sentences | Sparse success across 10 seeds | SVD success | Random-rotation success rate |
|---|---:|---:|---:|---:|
| Original + two single edits, rank 4 | 32 | **10/10** | 4/10 | .30 |
| Same | 2,048 | **10/10** | 4/10 | .31 |
| All four conditions, rank 4 | 32 | **10/10** | 2/10 | .28 |
| Same | 2,048 | **10/10** | 2/10 | .30 |

The 128- and 512-sentence results are the same success counts. At 32 sentences, mean post-fit group effect purity is **.994** for the sparse basis fitted on three conditions and **.991** for the one fitted on four. For the corresponding SVD bases it is .767 and .709. Group deletion from the reduced original function leaves the **unaffected bit at 100% accuracy** over all exhaustive sentences in every sparse run; the affected bit falls to mean **50.0%** or **50.9%** respectively. The base reduced functions themselves retain 100% joint task accuracy before deletion. These are interventions on explanatory function components, not evidence that the trained model contains those groups as physical nodes.

This is positive evidence for the featurizer loop: a sparse dictionary plus a small amount of unlabeled activation data can recover groups whose effects match independent task variables, even when individual axes no longer match whole signed bit contrasts. But the result depends on a strong **two-pairs-of-two prior** and a grammar with exactly one A token and one B token per sentence. Its correlations exploit mutual exclusivity within each token family. The names A/B and the 90% causal criterion are used after fitting for validation, so this is **group recovery under a supplied structure**, not unrestricted discovery of an unknown number of tasks. The next test should remove the pairing oracle by adding distractor directions, variable group sizes, or more bit families, then ask whether an unlabeled graph clustering rule still recovers groups and predicts held-out causal edits.

Run `dual_unsupervised_grouping.py` after `dual_intervention_sparse_basis.py`. `dual_unsupervised_grouping.json` records per-seed selected pairs, correlation margins, post-fit group effects, full-set deletion accuracy, and random-gauge controls for every sample size.
