# The reduction-before-legibility gate transfers to the agreement head

The [bracket-head factorial](BRACKET_REDUCTION_GROUP_FACTORIAL_RESULTS.md)
showed that an answer-free two-direction group can describe a rank-four
reduction almost perfectly even when that reduction erases most of the
native head's finite effect. We repeat the same design on a different
competent behavior: singular/plural subject–verb agreement at layer 11,
head 3 of the 546M model. The measured path is the change in the ` is`
versus ` are` logit margin when the edited prompt's final-position head
state is patched into the base prompt.

We freeze the five previously fitted rank-four reductions: output-weight
SVD, natural-corpus PCA, task-edit PCA, mixed moment, and random. The
task-edit reduction includes discovery agreement edits, but no validation
prompts. *Within* each reduction, we fit rank-one and rank-two response
groups from noun-number input-edit deltas without answer labels. The
ordinary fit uses 80 edits. The stronger fit removes all edits with any
of the six validation token-swap identities, leaving 44 edits. This
holdout applies to the **group fit**, not to the previously frozen
rank-four reductions.

The model gives the correct forced choice on all 20 native validation
pairs: six lexical holdouts and 14 changed-format prompts. The table
uses the stricter 44-edit group fit. Errors are mean absolute finite
margin-path differences divided by the mean absolute native head path
(4.85 logits), except the middle column, whose denominator is the
corresponding rank-four path.

| Rank-four reduction | Rank-four error vs native head ↓ | Rank-two error vs own rank four ↓ | Rank-two error vs native head ↓ | Correct rank-four forced-choice pairs |
|---|---:|---:|---:|---:|
| Weight only | 5.16% | 2.28% | 7.32% | 20/20 |
| Corpus PCA | 72.31% | **0.54%** | 72.33% | 18/20 |
| Task-edit PCA | **4.07%** | 3.23% | **5.05%** | 20/20 |
| Mixed | 5.48% | **0.31%** | 5.47% | 20/20 |
| Random | 95.95% | 32.53% | 97.27% | 20/20 |

The ordinary 80-edit fit predicts its own weight-derived rank-four path
more closely (0.41% rank-two error), and the identity-held-out fit
degrades to 2.28%. The transfer is therefore real but not invariant to
which edit identities are available. For the stricter fit, weight rank-two
error against its own cut is 2.2% on the six lexical holdouts and 2.3%
on the 14 changed-format prompts. Corpus rank-four path loss is 73.3%
and 71.8%, respectively; it is not confined to one panel.

This second task reinforces the stage ordering. A **dataset-aware** cut
is not automatically useful merely because it explains common natural
activations. The corpus-PCA two-direction group is exceptionally precise
*inside* its cut, yet the cut misses most of this head's native agreement
path and flips two correct answers. Conversely, the random cut leaves all
20 answers correct while missing 96% of the selected head's path. The
model has other routes to the answer, and forced-choice success does not
establish that a reduction preserves this particular mechanism.

This is still a selected-head, selected-task result. The 20 contexts
share only six noun-number contrasts, and the rank-four reductions were
fitted before the strict group-identity holdout. It does not establish
weight-only optimality for natural-text KL, a unique gauge, or necessity
of this head for agreement. It does establish that the bracket-head
failure mode of *excellent within-cut legibility with poor native-path
fidelity* also occurs in a separate competent attention head.

Reproduce with `flagship_agreement_reduction_group_factorial.py`.
Per-case paths, task margins, fit spectra, and panel summaries are in
`full_width_results/agreement_reduction_group_factorial.json`; the
within-cut rotations are in
`full_width_results/agreement_reduction_group_factorial_rotations.npz`.
