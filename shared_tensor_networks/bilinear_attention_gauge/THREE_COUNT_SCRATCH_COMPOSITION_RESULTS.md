# From-scratch bilinear composition separates quadratic and variable gauges

The [earlier trained composition](THREE_COUNT_TRAINED_COMPOSITION_RESULTS.md)
started from attention models already trained on the three count bits. To
test that initialization confound, this experiment trains **all** parameters
of the same three-head bilinear attention plus quadratic readout from
random weights. The task has three A/B/C count-sign outputs and a fourth
XOR of the first two outputs. Ten seeds each see 100,000 training and
20,000 validation sentences; a rare-query `A=0, B=1, C=0` combination is
absent from both. The checkpoint is selected by validation loss after up
to 2,500 AdamW steps, then evaluated on 30,000 independent test sentences.

All ten seeds reach **100% four-output accuracy** on the full test and the
withheld rare combination. The learned quadratic readout places mean
**83.3%** of its absolute effective coefficient weight on the two
off-diagonal first/second-output terms (minimum across seeds 73.6%). The
model therefore learned an interaction rather than receiving an analytic
parity formula. Its exact first-layer token table and learned quadratic
readout reproduce direct model outputs to maximum absolute error
`4.6e-13`.

## Reduction remains a separate choice

The first layer's exact `6 × 8` query/output-by-key table is reduced in
the shared key dimension. Uniform SVD uses weights alone. Frequency
weighted rank-two fitting uses 20,000 unlabeled training-law contexts;
neither fit sees task labels. The rank-three uniform control preserves all
four outputs perfectly in every seed.

| Reduction | Full-test parity accuracy | Rare-holdout parity accuracy | Full-test four-output accuracy | Uniform table MSE |
|---|---:|---:|---:|---:|
| Native | 1.000 | 1.000 | 1.000 | 0 |
| Uniform rank 2 | .733 | .158 | .578 | .315 |
| Frequency rank 2 | .661 | .734 | .523 | .324 |
| Uniform rank 3 | **1.000** | **1.000** | **1.000** | .00041 |

The rank-two metric tradeoff depends on the trained function: here the
frequency measure helps the withheld slice but hurts average parity
relative to uniform. The opposite ordering occurred for some pretrained
and jointly tuned checkpoints. No metric has a task-independent ranking.

## Fixed-function gauges and an explicit product node

Within each rank-two reduction, compare four exact rotations of its key
subspace: SVD, minimum key-entry L1, minimum single-axis deletion sparsity
on 2,048 unlabeled random key edits, and **weight-only quadratic
diagonalization**. The last minimizes the off-diagonal entries of
`H_q = C_qᵀ Q C_q`, where `C_q` maps reduced key coordinates to the three
readout logits and `Q` is the learned quadratic coefficient matrix. The
known A/B/C contrasts are used only after fitting to score whether two
distinct key directions recover designed variables. Test edits and a
higher key-count rate are held out.

| Fixed reduction | Gauge | Held-out effective axes ↓ | Cross-axis edit residual ↓ | Matched variable cosine ↑ |
|---|---|---:|---:|---:|
| Uniform rank 2 | Key L1 | 1.587 | .484 | **.947** |
| Uniform rank 2 | Context-edit | **1.278** | **.035** | .829 |
| Uniform rank 2 | Weight-only quadratic diagonal | 1.363 | .050 | .833 |
| Frequency rank 2 | Key L1 | 1.601 | .499 | **.965** |
| Frequency rank 2 | Context-edit | **1.302** | .103 | .866 |
| Frequency rank 2 | Weight-only quadratic diagonal | 1.414 | .109 | .802 |

Context-edit gives sparser deletion
effects than key L1 in **10/10 seeds** under each reduction. Key L1 gives
better variable alignment in **10/10**. For the frequency cut, the paired
mean advantages are -.299 effective axes (seed-bootstrap 95% interval
[-.412, -.188]) for context-edit and +.099 key alignment ([.035, .171])
for key L1. The 64-rotation random-control mean alignment is .871, below
key L1's .965. Scores transfer closely to the held-out high-count panel.

The weight-only quadratic gauge reproduces much of the context-edit
effect but is not identical on these scratch models: for the frequency
cut their angles differ by mean **11.6°**, compared with 0.7° in the
jointly fine-tuned pretrained control. Query frequencies and linear
terms of the readout can matter after diagonalizing its quadratic part.
The exact algebra is still clear: a sparse variable basis can represent
the computation as A, B, and an explicit A×B product node, while a
quadratic eigenbasis represents the same product as a difference of
squares of A/B mixtures: with `u=(A+B)/√2` and `v=(A-B)/√2`,
`AB=(u²-v²)/2`. Single-axis deletion sparsity prefers the
latter, even when the former better recovers the designed variables.

All gauge projectors agree to `2.3e-16`, total edit-response energy to
relative `2.3e-15`, and the explicit cross-node identity to absolute
`1e-12`. These are different descriptions of the same fixed reduced
function. The designed variables give a post-fit oracle unavailable in
natural language; the next step is to define a reuse-aware graph prior
that chooses product nodes without those labels, then test transfer on
held-out edits and the 546M model.

The first follow-up is a [label-free quadratic graph-prior sweep](THREE_COUNT_GRAPH_PRIOR_RESULTS.md).
A moderate graph penalty retains most A/B variable alignment while reducing
the error from pruning one quadratic node on held-out and shifted contexts.
It has not yet been selected on independent model families or tested for
physical circuit recovery.

Reproduce with `train_three_count_composed_scratch.py`,
`three_count_composed_scratch_analysis.py`, and
`three_count_composed_gauge.py --scratch`. Ten checkpoints and histories,
`three_count_composed_scratch_analysis.json`, and
`three_count_composed_scratch_gauge.json` retain all measurements.
