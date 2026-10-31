# A legible group cannot repair a reduction that lost the head path

The canonicalization program has two stages: choose a reduced function,
then choose human-readable coordinates *within that function*. The
[competent bracket identity test](COMPETENT_BRACKET_IDENTITY_TRANSFER_RESULTS.md)
showed that a weight-only rank-four cut of layer 13, head 8 transfers to
new parenthesis/square token IDs. The
[group red team](BRACE_COMPETENCE_AND_GROUP_TRANSFER_RESULTS.md) showed
that retaining two edit-derived directions inside it predicts the retained
finite path better than one. Here we cross the two choices directly.

We freeze **five previously fitted rank-four head-output subspaces**:
weight-only output-projection SVD, FineWeb head-state PCA, task-edit-delta
PCA, their mixed moment, and a random orthogonal control. The task-edit
reduction saw discovery edit vectors, including bracket examples, but no
validation token-ID substitutions. Inside *each* fixed subspace, fit the
top one or two directions of the **same 80 answer-free original
parenthesis/square head-state differences**. No output-answer labels or
validation contexts enter this group fit. All five bases are orthonormal
to Frobenius error below `2.3e-15` before float32 conversion.

We then test 21 native-competent prompts using six opening-token-ID
substitutions absent from group fitting. For each input edit, we
transplant the projected head-state difference at the base prompt and
measure the finite change in the `)` versus `]` logit margin after the
remaining model layers. The reference is the **complete native head**
patch; a separate comparison asks how well one or two axes reproduce
their *own* rank-four cut. Both questions matter.

| Rank-four reduction | Rank-four path error vs native head ↓ | Two-axis error vs own rank four ↓ | Two-axis error vs native head ↓ | Rank-four forced-choice pairs correct |
|---|---:|---:|---:|---:|
| Weight only | **2.29%** | 0.56% | **2.32%** | 21/21 |
| Corpus PCA | **80.25%** | 1.25% | **80.10%** | 21/21 |
| Task-edit PCA | 9.25% | **0.28%** | 9.42% | 21/21 |
| Mixed | 5.97% | 0.85% | 6.06% | 21/21 |
| Random | 93.66% | 3.32% | 93.77% | 21/21 |

Every cell is mean absolute margin-path error divided by mean absolute
reference path magnitude; the native head path averages **5.43 logits**.
The parenthesis/square task survives all five cuts, even when the native
contribution of this head is almost erased. This confirms that task
accuracy alone cannot certify a mechanistic reduction. The corpus cut
misses 73.8–88.4% of the native path in each of the six token-ID families;
the random control misses 90.7–96.4%. The result is not driven by just one
held-out identity.

The two-axis group inside corpus PCA is internally precise—1.25% error
against that cut—but remains 80.10% wrong against the native head. A
task-edit rank-one axis already reproduces its own cut to 0.41%, yet that
cut misses 9.25% of the native path. **Legibility conditional on a bad
reduction can be misleadingly excellent.** An exact orthogonal rotation
within a projector cannot recover the component of the head state that
the projector discarded. Downstream nonlinearity changes finite margin
effects, but cannot restore the omitted input state by reorienting the
retained coordinates.

This does not establish that weight-only reduction is universally best:
the selected head and bracket task favor its output geometry, while other
heads and natural-text KL can favor different measures. The finding is
local and actionable: choose the reduction by the *specific functional or
causal fidelity target*, then compare sparse/grouped bases within a
cut that passes that gate. In this head, answer-free edit covariance gives
a transferable small group after the weight-only cut; task, path, and
group-legibility scores remain distinct.

Reproduce with `flagship_bracket_reduction_group_factorial.py`. Per-case
finite paths, task margins, spectra, and summaries are in
`full_width_results/bracket_reduction_group_factorial.json`; frozen
within-cut rotations are in
`full_width_results/bracket_reduction_group_factorial_rotations.npz`.
