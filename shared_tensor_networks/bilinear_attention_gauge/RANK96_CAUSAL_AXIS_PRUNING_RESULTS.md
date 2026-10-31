# Sparse-looking axes do not survive finite causal pruning better than PCA

The [exact-gauge prior comparison](RANK96_EXACT_GAUGE_PRIORS_RESULTS.md)
showed that edit-activation varimax makes familiar edit responses appear
more concentrated than pooled-edit PCA, while both encode **the same
fixed rank-96 reduced head function**. We tested whether that proxy
advantage yields a useful mechanistic intervention.

For L9H7, each gauge encodes a native edit-induced head-state difference
in **output-isometric** coordinates. For each edit we keep its 8, 16,
or 32 largest absolute coefficients, decode those axes back into a
head-state difference, and patch that difference into the same native
base model. This top-k selector uses no edit-family or answer labels.
All gauges retain the same number of axes, and keeping all 96 would
give exactly the same reduced function. The target here is the
**final-logit head-patch path of the full rank-96 projected edit**;
we also saved error against the full native head edit separately.
The familiar evaluation has 12 independent cases from each of ten old
edit types. The unseen evaluation has 12 cases from each of ten whole
new designed types. Neither panel fitted the rotations.

| Panel | Axes kept | Gauge | Immediate output energy retained ↑ | Finite reduced-path relative MSE ↓ |
|---|---:|---|---:|---:|
| Familiar | 8 | Edit varimax | .608 | .391 |
| Familiar | 8 | Pooled PCA | **.623** | **.369** |
| Familiar | 8 | Weight varimax | .419 | .610 |
| Familiar | 8 | Random-gauge mean | .384 | .603 |
| Familiar | 16 | Edit varimax | .748 | .241 |
| Familiar | 16 | Pooled PCA | **.772** | **.211** |
| Familiar | 16 | Weight varimax | .610 | .390 |
| Familiar | 16 | Random-gauge mean | .585 | .395 |
| Familiar | 32 | Edit varimax | .892 | .101 |
| Familiar | 32 | Pooled PCA | **.906** | **.081** |
| Unseen | 8 | Edit varimax | .477 | .520 |
| Unseen | 8 | Pooled PCA | **.523** | **.458** |
| Unseen | 8 | Weight varimax | .423 | .594 |
| Unseen | 8 | Random-gauge mean | .382 | .589 |
| Unseen | 16 | Edit varimax | .658 | .316 |
| Unseen | 16 | Pooled PCA | **.692** | **.285** |
| Unseen | 16 | Weight varimax | .614 | .383 |
| Unseen | 16 | Random-gauge mean | .582 | .393 |
| Unseen | 32 | Edit varimax | .849 | .137 |
| Unseen | 32 | Pooled PCA | **.867** | **.115** |

The table averages per-edit retained energy and per-edit path relative
MSE. Random means use four predeclared exact rotations. At 16 axes on
the unseen panel, varimax minus PCA path error is **+.031**. Nine of
ten new-family means favor PCA. A descriptive cluster bootstrap over
the ten designed families gives a 95% interval of **+.015 to +.049**
for that difference; the families are not a random sample of natural
language tasks, so this is not a population confidence claim. PCA also
beats varimax on familiar cases at 16 axes by .030 mean path error.

This directly corrects an appealing but misleading reading of the
previous report. Edit varimax had a slightly lower **inverse-Simpson
effective-axis count** on familiar edits (13.70 versus 13.86), and much
higher familiar edit-type purity. Neither metric optimizes the amount
of output energy in the largest *k* coefficients. PCA retains more of
that energy and, in this experiment, more of the downstream finite path.
Weight sparsity is even less predictive of causal compression: the
weight-varimax decoder is sparser, but its top-k intervention paths
are close to random-gauge controls at small k.

The comparison holds **axis count** fixed, not the norm of the partial
intervention. That is appropriate for testing whether a gauge affords
better sparse coding at the same component budget; it does not by
itself isolate causal selectivity at matched intervention dose. Nor
does any axis here identify a reusable semantic rule. The next test
should compare predeclared *groups* or shared-dictionary modules at
matched output dose and use named answer-level behaviors plus whole
task-family holdouts. The negative result is a useful stopping rule
for optimizing edit-response L1 or varimax merely to improve top-k
causal compression of this head.

Reproduce with `flagship_rank96_axis_pruning_finite.py`. All 6,480
case/gauge/axis-budget observations, including native-path errors, are
in `full_width_results/rank96_axis_pruning_finite.json`.
