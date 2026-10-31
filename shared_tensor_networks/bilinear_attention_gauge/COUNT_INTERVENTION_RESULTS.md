# Ordinary semantic basis versus branch-robust basis in variable-count attention

The [variable-count language result](COUNT_LANGUAGE_CANONICAL_RESULTS.md) found an almost perfect two-direction semantic basis for the combined ordinary attention function. This follow-up asks whether that compression survives **internal QK-branch edits** in the same ten trained models.

For each seed, extract the exact \(8\times6\) head-preserving token table (two heads × four query/output rows) under four conditions: original scores, QK1 replaced by a constant one, QK2 replaced by one, and both replaced by one. Fit a shared key subspace by SVD on selected conditions, normalizing each fitted condition to unit Frobenius norm. Project every condition onto it and measure relative table MSE. The decomposition uses weights and specified branch edits, not sampled sentences or task labels. An edit omitted from the fit is a genuine **held-out structural condition** for this subspace test.

| Fitted conditions | Key rank | Original MSE | QK1 constant | QK2 constant | Both constant |
|---|---:|---:|---:|---:|---:|
| Original only | 2 | **.0004** | .468 | .381 | .565 |
| Original + both single edits | 2 | .241 | .159 | .223 | .538 |
| Original + both single edits | 3 | .0333 | .0583 | .0314 | .288 |
| Original + both single edits | 4 | **.0009** | **.0052** | **.0033** | **.0588** |
| All four conditions | 4 | .0061 | .0097 | .0064 | **.0080** |
| All four conditions | 5 | .0005 | .0011 | .0005 | .0011 |

Values average ten seeds. The ordinary rank-two space is excellent for the unedited function but misses the single and double branch edits badly. At the same rank, fitting the two single edits trades away ordinary fidelity. Four directions cover the original and two single edits below .006 mean error, but their unseen **double** composition remains at .0588. Including it lowers that error to .0080 at rank four, with some cost on the other three conditions. Full rank six represents all exact token tables.

This repeats the conceptual split seen in the first toy under a more demanding **trained variable-count grammar**. Label-free sparsity on the ordinary rank-two function identified A/B task axes, yet those axes are not a faithful basis for the model's branch counterfactuals. Conversely, intervention coverage costs dimensions and may change the natural sparse orientation. The double-edit failure follows from the QK product's interaction term; good fidelity on each single edit does not certify the combination. These are token-function and branch-response statements, not an identification of reusable physical nodes or downstream language-model behavior.

Run `count_intervention_dictionary.py` after the ten saved count-language checkpoints. `count_intervention_dictionary.json` contains every seed, rank two through six, singular values, and errors for all fit/test condition combinations.
