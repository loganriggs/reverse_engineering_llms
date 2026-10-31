# Three trained count rules create a real canonicalization rank budget

The [two-bit variable-count toy](COUNT_LANGUAGE_CANONICAL_RESULTS.md) was almost exactly rank two, so all reasonable rank-two metrics preserved its task. This follow-up trains a **three-bit, three-head bilinear-attention language** and tests rank two versus rank three. It also compares weight-only reconstruction, training-frequency reconstruction, and supervised cross-entropy under the insufficient rank-two budget.

Each sentence has up to twelve key tokens drawn initially from eight Poisson(.75) token counts: A0/A1, B0/B1, C0/C1, and two filler types. Samples with tied A/B/C contrasts or more than twelve keys are rejected, so accepted token counts are conditioned and not independent. The three target bits are signs of the A, B, and C count contrasts. A common final query outputs (A,B,C), while a rare query rotates them to (B,C,A). Rare-query A=0/B=1/C=0 examples are absent from training and validation but present in the independent test set. Ten seeds use **100,000 training, 20,000 validation, and 30,000 test sentences** each. A width-24, three-head bilinear-attention model with masked padding trains for 1,500 steps per seed. All ten models reach **100%** validation, full-test, and rare-heldout joint accuracy. Deleting a physical head drops mean full-test joint accuracy to **64.5%** across 30 deletions.

The weight-derived exact token table is \(T\in\mathbb R^{2\times8\times3}\), with two query types, eight key tokens, and three output logits. Because the first-layer head sums pair contributions over key occurrences, the table describes every allowed variable-count sentence. Its direct forward matches the saved model to maximum absolute error **3.6×10^-14**. Flattening query and output gives a \(6\times8\) matrix whose mean singular values across seeds are **18.07, 17.54, 17.19, .495, .267, .084**: three large task directions and three small residual directions.

## Rank really matters

No sampled sentences or task labels enter the uniform SVD fits. The test scores below average ten independent 30,000-sentence sets.

| Effective token-function representation | Mean uniform table MSE | Mean test-logit MSE | Full-test joint accuracy | Rare-heldout joint accuracy |
|---|---:|---:|---:|---:|
| Original exact model | 0 | ~0 | 1.000 | 1.000 |
| Joint rank 1 | .649 | .655 | .252 | .013 |
| **Joint rank 2** | **.318** | **.323** | **.571** | **.204** |
| **Joint rank 3** | **.000392** | **.000290** | **1.000** | **1.000** |
| Separate rank 1 per physical head | .233 | .131 | .883 | .315 |

Joint rank three and three separate headwise rank-one tables both use **42 raw factor entries** (three times \(6+8\)), but the former shares a key dictionary across heads and preserves all three rules; the latter loses much of the rare held-out behavior. They are different function classes, so equal raw count does not isolate an optimization or parameter-efficiency effect. These are SVDs of effective head tables, not rank-one truncations of physical QK bonds.

The rank-two failure has a simple **capacity proof** for the common query. Let \(c\in\mathbb R^8\) be the key-token count vector. A rank-two query/output-by-key matrix factors as \(M=LD\), so all three common-query logits are affine functions of the same two coordinates \(z=Dc\in\mathbb R^2\). Each bit is the sign of an affine function of \(z\), corresponding to one line in the plane. Three lines divide the plane into at most \(1+3+\binom{3}{2}=7\) open sign regions, while the task needs **eight** distinct (A,B,C) sign patterns. All eight occur in the common-query test set of every seed, and each has positive probability under the generator. Thus no rank-two additive token function with three affine output logits can solve the entire common-query task, regardless of how the factors are fitted. Rank three removes this particular obstruction; the trained models and rank-three SVD empirically realize the task. This proof uses the additive first-layer token-table form and does not claim a universal lower bound for deeper nonlinear networks.

## What changes under a hard rank-two budget?

Uniform fit weights every role-valid query–key/output cell equally. The empirical metric weights cells by token-pair frequency in each seed's **100,000 unlabeled training sentences**, with no rare held-out *sequence*. The half mixture averages those weights. All fits start from SVD and optimize weighted squared table error; they are local low-rank solutions, not global-optimum certificates. A separate **supervised** condition directly optimizes rank-two factors for cross-entropy on labeled training examples, using three starts and selecting checkpoints by validation BCE. Its training and validation both exclude the rare whole-task holdout.

| Rank-two objective | Mean full-test joint accuracy | Mean rare-heldout joint accuracy | Mean test BCE | Mean test-logit MSE |
|---|---:|---:|---:|---:|
| Uniform table MSE | .571 | .204 | .497 | .323 |
| Training-frequency table MSE | .566 | .402 | .426 | .319 |
| Half uniform / frequency | .567 | .311 | .453 | .319 |
| **Supervised cross-entropy** | **.623** | .179 | **.221** | .548 |

The empirical metric gives up a little full-test joint accuracy but improves mean rare-heldout accuracy relative to uniform fitting, despite never seeing the withheld combination. This is **seed-sensitive**: rank-two approximations cannot preserve all eight A/B/C bit patterns, and the rare score is often near zero or one depending on which rule the fit sacrifices. The supervised CE condition improves average ordinary-task accuracy and BCE, while moving much farther from the teacher's logits and leaving the held-out combination unreliable. Eight of its ten seeds score below 2% on that rare slice; two score much better. This is a concrete illustration of the fidelity-to-task-loss spectrum the project set out to compare, **not** a universal ranking of metrics. At rank three, uniform, empirical, and mixed fits all achieve 100% full and rare-heldout accuracy, so the distinctions only become visible under actual capacity pressure.

## A label-free sparse basis for all three task variables

Take the joint rank-three weight-only SVD \(M=LD\) and minimize the L1 norm of the three-row key dictionary \(D\) over orthogonal rotations. Sixteen starts and pairwise Givens sweeps select the lowest-L1 orientation. The entire reduced function is invariant to this rotation. A/B/C contrast vectors and bit-flip effects are used only *after* orientation selection.

| Rank-three basis | Mean matched A/B/C axis cosine | Mean intended bit-flip effect fraction |
|---|---:|---:|
| SVD | .862 | .752 |
| 128 random rotations per seed, mean | .829 | .699 |
| **Label-free key-L1 rotation** | **.99950** | **.99991** |

Sparse alignment is at least **.99899** in every seed. Mean key L1 falls from **25.89 to 18.25**. On independent variable-count test sets at Poisson rates .35, .75, and 1.2, the rank-three reduced function retains 100% joint accuracy. Removing one sparse explanatory component leaves the **other two semantic bits at 100% accuracy** in every test, while the affected bit falls to about 50%. These edits target components of the *reduced token function*, not nodes proven to exist as such in the physical network.

The positive result is sharply scoped. This first-layer task is additive over token counts, so its three independent bits naturally induce a low-rank key function. The sparse token basis is meaningful here because token identities were designed around those bits. The experiment demonstrates a working **weight-only reduction → sparse orientation → post-fit causal validation** pipeline under a real rank bottleneck and a whole-combination holdout. It does not establish that ordinary function components predict QK-branch interventions, that semantic modules are physically localized to heads, or that the same prior works for the 546M model. The next high-information test is to add a non-additive or context-dependent rule and compare intervention-aware, weight-only, and data-aware decompositions at matched fidelity and task budgets.

Reproduce with `train_three_count_language.py --seed 0` through `--seed 9`, followed by `analyze_three_count_language.py`, `three_count_metric.py`, `three_count_task_fit.py`, and `three_count_sparse.py`. The ten checkpoints and history files plus the four JSON analyses preserve per-seed scores and fitted factors.
