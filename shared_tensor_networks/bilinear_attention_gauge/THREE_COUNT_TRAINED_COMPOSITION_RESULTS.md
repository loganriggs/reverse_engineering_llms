# Trained bilinear composition preserves the key-sparsity / edit-sparsity conflict

The earlier [analytic-readout toy](THREE_COUNT_NONLINEAR_GAUGE_RESULTS.md)
showed that two exact gauges of the same reduced attention function can
favor different interpretations: key sparsity recovered A/B variables,
while context-edit sparsity mixed them to make single-axis deletion effects
look additive. This experiment trains the nonlinear readout and checks
whether that conflict survives.

Ten previously trained three-head bilinear-attention models provide a
first-stage representation of three count-comparison bits. For each seed,
a symmetric quadratic readout of those three logits is initialized at zero
and trained for 1,200 steps to predict the XOR of the first two output bits.
The attention backbone is frozen during this phase. A second, separately
saved version fine-tunes both stages jointly for 400 steps, selecting its
checkpoint by validation loss. Each seed uses 100,000 training, 20,000
validation, and 30,000 independent test sentences. The rare-query
combination `A=0, B=1, C=0` is absent from both training and validation.
This is a learned nonlinear composition with pretrained attention, not
training both stages from random initialization.

All ten frozen-backbone models reach **100%** full-test four-output and
rare-holdout accuracy. After joint fine-tuning, mean full-test four-output
accuracy is **99.992%** and rare-holdout accuracy **99.642%**; four seeds
lose a little rare accuracy, with the worst at 97.9%. The learned
quadratic is concentrated on the interaction between its first two logits:
their two symmetric off-diagonal entries account for mean **95.4%** of
absolute effective quadratic weight in frozen-readout models and **87.7%**
after joint fine-tuning. Thus the composition is learned rather than
inserted as the analytic `-logit_0 * logit_1` rule.

## Exact function and rank controls

For each checkpoint, the first-stage attention has an exact `6 × 8`
query/output-by-key token table. Replaying the whole learned two-stage
function from that table and the quadratic readout matches direct model
output to maximum absolute error **5.0e-14** on test sequences. This
weight-only object needs no sampled sequences to define it.

The table is reduced to a shared rank-two key dimension by uniform SVD or
by training-frequency weighted table least squares. The latter fits 20,000
unlabeled training-law contexts but no task labels. A rank-three uniform
SVD is a capacity control. Means over ten models:

| Stage / reduction | Full-test parity accuracy | Rare-holdout parity accuracy | Full-test four-output accuracy |
|---|---:|---:|---:|
| Frozen / native | 1.000 | 1.000 | 1.000 |
| Frozen / uniform rank 2 | .693 | .512 | .564 |
| Frozen / frequency rank 2 | .720 | .761 | .552 |
| Frozen / uniform rank 3 | **1.000** | **1.000** | **1.000** |
| Joint / native | 1.000 | 1.000 | 1.000* |
| Joint / uniform rank 2 | .791 | .797 | .548 |
| Joint / frequency rank 2 | .933 | .465 | .543 |
| Joint / uniform rank 3 | **1.000** | **1.000** | **1.000*** |

`*` Rounded from .999923 and .999957. Frequency weighting can improve
overall parity while damaging rare behavior after joint tuning; the
ordering differs from the frozen control. This is a seed-averaged,
architecture-specific tradeoff, not a universal claim about data-aware
reductions. Rank three largely removes the bottleneck for this task.

## Exact-gauge legibility test

Within each saved rank-two table, rotate its two orthonormal key directions
without changing the projector or any model output. Compare SVD orientation,
minimum key-entry L1, and minimum effective number of single-axis deletion
effects on a random key-edit response. The edit-sparsity gauge fits 2,048
unlabeled, holdout-free contexts with a 512-angle search; evaluation uses
fresh 2,048-context panels and a high-count-rate shift. A/B/C contrast
labels are used only after fitting to score the key directions.

| Model / rank-two cut | Gauge | Held-out effective axes ↓ | Cross-axis response residual ↓ | Matched A/B/C key cosine ↑ |
|---|---|---:|---:|---:|
| Frozen / frequency | SVD | 1.619 | .353 | .874 |
| Frozen / frequency | Key L1 | 1.811 | .766 | **.963** |
| Frozen / frequency | Context-edit | **1.425** | **.082** | .779 |
| Joint / frequency | SVD | 1.819 | .646 | .926 |
| Joint / frequency | Key L1 | 1.990 | .981 | **.989** |
| Joint / frequency | Context-edit | **1.476** | **.002** | .728 |

The same ordering holds under uniform rank-two cuts. Context-edit wins
held-out edit concentration over key L1 in **10/10 seeds** for each of the
four stage × reduction conditions. Key L1 wins known-variable alignment in
the corresponding comparisons; for the joint frequency cut its .989
alignment exceeds the 64-rotation random-control mean .884, whereas the
context-edit gauge's .728 falls below it. Results transfer to the
high-count-rate panel: joint frequency effective axes are 1.990 for key
L1 and 1.448 for context-edit.
For this joint-frequency comparison, the paired seed mean changes from key
L1 to context-edit are -.514 effective axes (bootstrap 95% interval
[-.533, -.496]) and -.979 cross-axis residual ([-.990, -.967]); key L1's
alignment advantage is +.261 ([.254, .269]).

A further **weight-only** gauge minimizes the off-diagonal energy of the
two-dimensional quadratic form induced by the learned readout. If `C_q`
maps key coordinates to readout logits for query type `q`, the form is
`H_q = C_qᵀ Q C_q`; we minimize the summed squared off-diagonal entries
over both query types, without sampled sentences. On the joint frequency
cut, this gauge has 1.477 held-out effective axes, .0015 interaction
residual, and .721 key-variable alignment. The data-fitted context-edit
gauge has 1.476, .0023, and .728 respectively; their selected angles
differ by only **0.7° on average**. For the frozen frequency cut,
quadratic diagonalization is less exact (1.531 effective axes versus
1.425 for context-edit) but has the same direction of tradeoff. Thus
much of the edit-sparsity gain can be obtained from weights alone, but
that canonical choice still mixes the task's input variables.

The apparent conflict has an exact algebraic explanation. If the two key
axes contribute vectors `a` and `c` to the readout logits, a quadratic
readout contains the explicit interaction node `2 aᵀ Q c`. Summing the
effects of deleting `a` and `c` counts this term twice. The reported
cross-axis residual equals the edit response of that node; the identity
passes to maximum absolute error below **8e-13**. The key-sparse gauge
keeps recognizable input variables and a large interaction node. The
context-edit gauge rotates the variables into mixtures that absorb the
interaction into individual terms. Both are exact descriptions of the
same reduced function, but only one closely follows the designed
computation's variable factorization.

Projector error stays below `2.3e-16`, and total edit-response energy
agrees across gauges to relative `1.5e-15`. This establishes a precise
warning for human-legibility priors: **single-axis causal additivity can
reward a gauge that hides compositional structure**. A reuse-aware
canonicalization should allow explicit product nodes and evaluate their
transfer, not force every effect into independent linear components.

The designed A/B/C variables supply post-fit ground truth here; natural
language has no such oracle. The readout starts from pretrained attention,
and deletion edits act on a reduced explanatory function rather than
physical model nodes. Testing graph recovery on a model trained from scratch,
then on named behaviors in the 546M model, remains necessary.

That [from-scratch control](THREE_COUNT_SCRATCH_COMPOSITION_RESULTS.md) is
now complete: all ten random-initialized two-stage models solve the
holdout, and the same variable-versus-deletion-gauge tradeoff persists.

Reproduce with `train_three_count_composed.py`,
`three_count_composed_analysis.py`, and `three_count_composed_gauge.py`.
Ten frozen and ten jointly tuned checkpoints, per-seed training histories,
`three_count_composed_analysis.json`, and `three_count_composed_gauge.json`
contain all fit and test records.
