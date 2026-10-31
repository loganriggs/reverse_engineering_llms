# Weight-only canonical function on a trained variable-count attention language

The earlier two-head toy always placed one A token and one B token in each sentence. Its [co-occurrence grouping](GRAPH_GROUPING_RESULTS.md) benefited from that grammar, and the [grammar-shift red team](GROUPING_GRAMMAR_SHIFT_RESULTS.md) showed how quickly the signal can disappear. This experiment trains **new** two-head bilinear-attention models on variable token counts, then repeats the weight-only function reduction and label-free sparse-axis test.

Each sentence has up to eight key tokens and a final common or rare query. Counts of A0, A1, B0, B1, F0, and F1 are initially drawn from independent Poisson(.75) variables; examples with more than eight keys or tied A/B contrasts are rejected. Thus the actual accepted counts are **not independent**—conditioning induces about -.39 within-A and within-B count correlation—but they vary substantially and can contain both opposing tokens from a family. The A bit is the sign of count(A1)-count(A0), and B is analogous. The common query requests (A,B); the rare query swaps them. Training excludes rare-query cases with A=0,B=1, while independent test sets include them. A masked-pad version of the two-head bilinear attention architecture learns from **50,000 training sentences** per seed; validation has 10,000 and test has 20,000. Ten seeds were trained for 1,500 steps, except seed 3 continued to 4,000 because its rare holdout was still improving.

All ten models reach **100% validation joint accuracy** on the training distribution. On independent test sets, nine solve the rare withheld combination perfectly; seed 3 reaches **98.66%** there despite 100% accuracy on common and rare-seen cases. Removing either physical head lowers mean full-test joint accuracy to **52.9%** across 20 head deletions, so both heads matter. The exact role-valid \(2\times6\times2\) token table extracted from weights reproduces the direct masked model to maximum absolute error **4.3×10^-14**. This is possible without enumerating sentences because the head response sums independently over key-token occurrences.

## Which reduction preserves the count task?

Flatten the **sum across heads** to a \(4\times6\) query/output-by-key matrix and truncate by ordinary SVD. No sampled data or task labels choose the reduction. Test scores below average ten seeds on 20,000 fresh variable-count sentences per seed; rare holdout has 718–781 examples per seed.

| Function representation | Mean relative uniform table MSE | Mean relative test-logit MSE | Mean full-test joint accuracy | Mean rare-heldout joint accuracy |
|---|---:|---:|---:|---:|
| Original exact table | 0 | ~0 | .99995 | .99866 |
| Combined rank 1 | .461 | .491 | .5008 | .1005 |
| **Combined rank 2** | **.000436** | **.000963** | **.999985** | **.99960** |
| Separate rank 1 per head | .0778 | .0432 | .9687 | .7957 |
| Combined rank 3 | .0000325 | .0000292 | .99995 | .99866 |

The mean singular values of the combined matrix are **18.97, 17.52, .420, .105**. Rank two retains the two independent count rules almost perfectly. In seed 3 it even improves the original model's rare-heldout accuracy from .9866 to .9960 by removing a small residual term; this is an observed regularization effect on one seed, **not** evidence that truncation generally improves the task. Separate headwise rank-one tables have more error and much worse rare transfer despite the same total rank count. The teacher's remaining error limits any absolute task-success claim.

### Uniform versus dataset-aware metric

The weight-only SVD gives every role-valid token-pair/output cell equal weight. A second fit weights cells by their empirical frequency in each seed's **50,000 unlabeled training sentences**; a third uses a half-uniform mixture. All twelve query–key cells occur in training, although the rare *sequence-level* A=0/B=1 combination does not. Weighted low-rank factorization starts from SVD and is a local least-squares optimization; the test set is independent and not used to fit or select a metric.

| Joint rank-2 metric | Mean uniform table MSE | Mean independent test-logit MSE | Mean full-test joint accuracy | Mean rare-heldout joint accuracy |
|---|---:|---:|---:|---:|
| Uniform | **.000436** | .000963 | .999985 | .99960 |
| Training-frequency | .000669 | **.000519** | **1.000000** | **1.00000** |
| Half uniform / frequency | .000492 | .000586 | .999985 | .99960 |

Empirical weighting gives up some uniform table fidelity and improves logit fidelity on the independent count distribution. Its perfect task accuracy here comes from correcting a handful of seed-3 rare cases, so it is not a robust claim that data-aware fitting is universally better. At rank one, all three metrics fail to represent both independent bits: full-test joint accuracy stays near .50 and rare-heldout accuracy ranges from .10 to .20. The metric choice matters at the margin **after** sufficient rank is available. The per-seed weighted fits are saved in `count_joint_metric.json` and reproduced by `count_joint_metric.py`.

## Does label-free key sparsity find A and B?

Factor the combined rank-two table as \(LD\) by SVD, then rotate its two-dimensional key dictionary through 4,096 angles to minimize \(\|D\|_1\). The rotation preserves the **entire reduced function** exactly. A/B contrast vectors, target labels, and bit-flip effects are used only after selection.

| Gauge of rank-two function | Mean matched A/B axis cosine | Mean intended bit-flip effect fraction |
|---|---:|---:|
| SVD | .9741 | .9511 |
| 128 random rotations per seed, mean | .8980 | .8153 |
| **Key-L1 sparse rotation** | **.99927** | **.99969** |

The sparse cosine is at least **.99865** in every seed. Mean key L1 falls from **14.02 to 12.39**. Removing one sparse component from the reduced explanatory function leaves the **unaffected bit at 100% accuracy** over independent samples at Poisson rates .35, .75, and 1.2; the affected bit falls to about 50%. The original full model's mean joint accuracy at those rates is 1.00000, .99994, and .99994; the rank-two reduced function scores 1.00000, .99998, and 1.00000. These shifted-rate tests change the input distribution within the same count rule, not the task grammar itself.

This is a positive data-free canonicalization case: a shared functional reduction followed by sparse orientation identifies the two task variables even with repeated, variably counted tokens and both physical heads required. It does **not** establish recovery of physical head modules, robustness to QK-branch interventions, discovery of an unknown number of tasks, or success on the 546M model. The task was designed to be additive in key counts, making a low-rank first-layer token function plausible. The next test should add a distractor rule or a third task variable, compare weight-only with data-aware and intervention-aware metrics under a genuine rank budget, and keep whole-task and causal holdouts distinct.

A [branch-intervention follow-up](COUNT_INTERVENTION_RESULTS.md) now tests that causal limitation: the ordinary rank-two key space has mean relative errors .468/.381/.565 under QK1-constant, QK2-constant, and double-constant edits. A rank-four space fitted to all four conditions brings each below .010, while fitting only the single edits leaves .0588 error on the held-out double edit. This does not change the semantic result for ordinary behavior; it defines its intervention scope.

Run `train_count_language.py --seed 0` through `--seed 9` (seed 3 at 4,000 steps), then `analyze_count_language.py`, `count_joint_metric.py`, and `count_sparse_canonical.py`. Saved `count_language_seed*.pt` checkpoints, training histories, and the corresponding JSON analyses contain the per-seed evidence.
