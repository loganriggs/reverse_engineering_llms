# Can internal head responses reveal a rule without its output readout?

The previous [exact dictionary counterexample](DENSE_WEAK_GROUP_IDENTIFIABILITY_RESULTS.md)
showed that output-link sparsity prefers output-specific atoms over reusable
input-rule atoms. We therefore fit a dictionary to the **internal four-head
QK1 × QK2 × V response**, before the learned 4-output readout, on the ten
trained dense-coded four-rule models. The dictionary fit receives neither
the D readout weights nor task labels. Only after fitting do we map its
components through the readout and compare them to the known A/B/C/D rules.

This is a distinct form of held-out-task analysis: the models were trained
with D supervision, so their heads can contain D information; we withhold D
from the *post-hoc featurizer fit*, not from model training. The four raw-bit
labels and intended membership graph enter only the audit. The sole
exception is the designed `no_direct_D_flip` training-edit set: synthetic
bit identities identify which eight direct D-flip edges to exclude. Those
identities are not used to optimize the basis within the remaining 112
token-pair edits.

## Protocol

We extract the exact `2 query × 4 head × 16 key` response table from QK1,
QK2, and V weights. Passing it through the held-out readout reproduces the
native model's forward pass within `2e-10` in float64. For each seed we fit
rank four using (1) uniform weight-only SVD on the full token table and
(2) weighted least squares with an empirical measure estimated from
100,000 unlabeled training-law sentences. No output readout or task label
enters either reduction. Both cuts retain more than 99.98% D prediction
agreement and more than 99.98% joint prediction agreement with the native
model on a separate 30,000-example test. Thus the legibility comparison is
made at a nearly task-faithful rank.

Within each fixed cut, exact orthogonal rotations optimize edit-response
L1 using all 120 token pairs or 112 pairs excluding the eight direct D
flips. We use 32 Powell starts for each objective. SVD and 32 seeded Haar
rotations per cut are controls. All rotations represent the same cut;
reconstruction and predictions are identical across gauges. The known
raw-bit code is used *afterward* to match axes and score the eight-row
output membership graph. A component's D-flip error asks how much of the
full D-output change under those withheld direct flips is reproduced by
the single atom matched to raw D. This is a narrower causal-response audit
than deleting physical heads.

| Reduction and gauge; mean over 10 models | Raw-bit atom alignment | Intended graph-cell accuracy | D-flip D-output component relative error |
|---|---:|---:|---:|
| Weight-only, SVD | .691 | .563 | .323 |
| Weight-only, random rotation | .767 | .582 | .222 |
| Weight-only, all-pair edit L1 | **.994** | **.912** | **.003** |
| Weight-only, exclude direct D flips | .924 | .866 | .157 |
| Empirical usage, SVD | .691 | .563 | .324 |
| Empirical usage, random rotation | .769 | .580 | .213 |
| Empirical usage, all-pair edit L1 | **.994** | **.919** | **.003** |
| Empirical usage, exclude direct D flips | .953 | .878 | .124 |

The response table itself is nearly rank four: the mean relative head-table
MSE is `4.89e-5` for the weight-only cut and `6.48e-5` for empirical usage.
Native joint accuracy on this test panel is 99.985%; the two reductions
agree with native joint predictions at 99.986% and 99.985%, respectively.
The sparse basis improves the *decomposition*, not the function.
Against SVD, the all-pair gauge improves atom alignment, graph-cell accuracy,
and D-flip component error in **all ten seeds** under each reduction measure.

The all-pair prior identifies reusable raw-bit atoms from internal head
responses without consulting the D output mapping. In this toy, simply
choosing a sparse output-link dictionary would select the wrong graph.
When we remove the eight clean D-flip edits, the result degrades under both
measures, particularly on the D component's held-out response. Increasing
from eight to 32 optimizer starts improved recovery, so residual failure
may combine an objective ambiguity with nonconvex search; this experiment
does not isolate those causes. The remaining training pairs can still vary
D jointly with other bits, so this is not a *whole-D-information* holdout.

The next discriminating experiment should create a *new* trained toy with
multiple independent compositions of the same input rules, and withhold
one whole composition and its direct edit family from featurizer selection.
Then compare an actual shared-dictionary/group-gate model with edit-sparse
rotations at matched function error. The current result supports the
input-intervention prior but does not prove physical circuit recovery or
real-language generalization. It also warns against calling a component
“discovered” merely because its output label was hidden: the internal
weights were trained on that label.

Reproduce with `dense_weak_unlabeled_heads.py`. Per-seed metrics, random
controls, fit diagnostics, and summaries are in
`dense_weak_unlabeled_heads.json`; factors are in
`dense_weak_unlabeled_head_factors.npz`.
