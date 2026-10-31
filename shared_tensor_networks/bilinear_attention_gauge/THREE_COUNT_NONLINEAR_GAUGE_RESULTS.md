# Exact-gauge intervention sparsity can hide a real interaction

The [rank-two nonlinear toy](THREE_COUNT_NONLINEAR_METRIC_RESULTS.md)
separated reduction metrics but did not compare legibility priors **after**
fixing a reduced function. Here each of three saved rank-two attention
tables—uniform, frequency-weighted, and downstream-aware—is held exactly
fixed. We choose orthonormal coordinates for its two-dimensional shared key
space, then rotate those coordinates without changing any token-table entry,
logit, or bilinear parity prediction. This is a gauge comparison, with zero
additional reconstruction error.

We compare the SVD orientation, **key L1** (minimize the absolute entries of
the two key directions), and **context-edit sparsity**. The latter uses
4,096 unlabeled training-law sentences excluding the rare whole-combination
holdout. In each sentence, replace one randomly selected present key with a
different random key. For each candidate axis, measure how deleting that
axis from the *reduced explanation* changes the bilinear output's response
to this input edit. Search 512 angles and choose the one minimizing the
edit-energy-weighted effective number of active axis effects:

`N_eff = (|effect_1|+|effect_2|)^2 / (effect_1^2+effect_2^2)`.

The weighting is the squared *total* output response to the edit. No A/B/C
family labels or task targets enter gauge selection. Test on 4,096 new
sentences, including the rare combination, and 4,096 new high-count-rate
sentences. Sixty-four random rotations per model are reference controls.
Ten trained attention checkpoints × three fixed reductions give 30 paired
comparisons.

## Results at fixed function

Means over ten checkpoints; lower effective-axis count means a more
concentrated axis-deletion response. The **interaction residual** is
`||total edit response - sum of the two axis-deletion effects||² /
||total edit response||²`. It measures the part not assigned additively to
individual deletions by this particular nonlinear readout and gauge.
For two axis contributions `a` and `c` to the two logits, the bilinear
readout contains the cross-axis term `-(a[0]*c[1]+c[0]*a[1])`. Deleting
either axis removes that term, so summing both single-axis deletion effects
counts its edit response twice. The residual detects this interaction; it
is not missing function output. The code constructs that cross node
separately and checks the identity against every sampled edit to below
`7e-13` absolute error.

| Fixed rank-two reduction | Gauge | Held-out effective axes | Interaction residual | Matched A/B/C key cosine | A/B/C edit-family purity |
|---|---|---:|---:|---:|---:|
| Uniform table | SVD | 1.509 | .233 | .855 | .556 |
| Uniform table | Key L1 | 1.594 | .534 | **.937** | .541 |
| Uniform table | Context-edit | **1.319** | **.103** | .823 | .551 |
| Frequency table | SVD | 1.627 | .361 | .878 | .564 |
| Frequency table | Key L1 | 1.837 | .788 | **.968** | .533 |
| Frequency table | Context-edit | **1.449** | **.096** | .780 | .533 |
| Downstream-aware | SVD | 1.741 | .461 | .890 | .550 |
| Downstream-aware | Key L1 | 1.995 | .993 | **.996** | .501 |
| Downstream-aware | Context-edit | **1.501** | **.004** | .710 | .506 |

On the downstream-aware reduction, key L1 lowers the dictionary's mean L1
from 3.828 (SVD) to 3.071 but leaves the nonlinear edit response almost
entirely in a cross-axis interaction: residual .993. The context-edit gauge
raises key L1 to 4.199, yet reduces that residual to .004 and lowers held-out
effective axes from 1.995 to 1.501. It beats key L1 on held-out
effective-axis count in **10/10 seeds** for each of the three reductions.
For the downstream-aware cut, the paired mean difference is -.494 axes
(seed bootstrap 95% interval [-.499, -.488]); interaction-residual
difference is -.988 ([-.991, -.985]). The training/test and count-shift
scores are close, so this is not merely memorization of edit contexts.

The key cosine is a **post-fit** match of the two learned directions to two
distinct designed A/B/C count contrasts. On the downstream-aware cut,
key L1 aligns almost exactly with **A and B** in every seed (mean .996,
minimum .993), while the context-edit gauge rotates them into roughly
45-degree A/B mixtures (mean .710). The average of 64 random rotations is
.897. This reverses the conclusion one would draw from deletion sparsity
alone: the sparse-key gauge best recovers the variables, even though its
single-axis deletions look less additive.

The A/B/C edit-family purity is a **post-fit** check: for each axis, group
the squared effect of A0→A1, B0→B1, and C0→C1 edits, then measure its
dominant family's share. On the downstream-aware cut, the context-edit
gauge scores only .506, versus .501 for key L1 and .549 for random
rotations on average. This score **does not measure key-variable recovery**
in a parity circuit: an edit to either input variable changes their product,
so even a perfectly aligned A axis can affect both A- and B-family edit
responses. The combination of near-perfect key alignment and near-unit
interaction residual for key L1 is direct evidence that a meaningful
variable decomposition can have non-additive single-axis causal effects.

The context-edit prior chooses coordinates in which deletion effects are
nearly additive, but those coordinates mix the original variables. A
computational DAG with explicit A, B, and A×B nodes would express the
compositional explanation more directly. Forcing the interaction into
individual linear axes is the wrong ontology for this toy.

## Exactness and limits

The maximum projector discrepancy among fitted and random gauges is
`2.3e-16`; total held-out edit-response energy agrees across selected
gauges to relative `1.6e-15`. Every function-logit implementation check
passes below `1e-9`. Thus the differences above are coordinate choices,
not changes to the reduced attention function or its task accuracy.

This is a two-dimensional grid search, not a full uniqueness theorem. The
component deletions modify an explanatory token table, not physical model
nodes. The second-stage readout is analytic, and the toy grammar has only
three designed variables; neither result directly certifies legibility in
the 546M model. The next test should train the nonlinear composition and
compare both single-axis and explicit cross-axis or DAG explanations,
requiring held-out task and intervention selectivity rather than sparsity
alone.

The known A/B/C contrasts were used only after fitting; using them to
choose a gauge would be a different, supervised experiment.

Reproduce with `three_count_nonlinear_gauge.py`. The full per-seed gauge
choices, training/test/shift metrics, random controls, and family profiles
are in `three_count_nonlinear_gauge.json`. The [plot](three_count_nonlinear_gauge.png)
shows the variable-alignment/edit-sparsity tradeoff, with individual seeds.
