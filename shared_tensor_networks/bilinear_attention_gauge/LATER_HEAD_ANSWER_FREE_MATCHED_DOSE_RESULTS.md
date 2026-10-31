# Answer-free edit axes pass the matched-dose gate only with task-type coverage

The [task-trained matched-dose result](LATER_HEAD_MATCHED_AXIS_DOSE_RESULTS.md)
showed a modest advantage for effect- and edit-sparse coordinates in two
546M attention heads. This follow-up applies the *same* finite
interventions and strong random controls to gauges fitted **without
answer labels**. It tests whether diverse input edits alone provide a
useful legibility prior, and whether a completely absent edit family can
be recovered from the others.

## Frozen design

The physical heads and weight-only rank-four projectors stay fixed:
layer 13/head 8 for bracket closure, layer 11/head 3 for subject–verb
agreement. Their original selection used those task labels. Each of
two pre-existing 800-edit panels contains 80 designed one-token
substitutions in each of ten types. The panels have disjoint full
sequences but reuse some token-substitution identities. They are
separate **gauge-fitting samples**, not independent trained models.
No answer logits or margin gradients enter these gauge fits.

For each panel, we fit raw edit-vector PCA, normalized edit-vector PCA,
and edit-vector L1 rotations. We select each axis by **global raw edit
response energy**. The full condition includes all ten edit types.
The exclusion condition removes every opening-bracket edit for the
bracket head, or every noun-number edit for the agreement head, from
both rotation fitting and axis selection. The exclusions are
whole-edit-type holdouts; other edits remain.

The test is unchanged from the supervised dose assay: 18 bracket and
20 agreement held-out lexical/new-format prompt pairs, with a finite
single-axis patch at the final position of the rank-four projected
model. Each candidate is compared with eight of 64 random rotations.
Every random rotation first gets its *best task axis on discovery
answer labels*; the eight with largest local output dose are then
chosen on the test edit without inspecting the answer. Candidate and
control are scaled to the same local residual norm **after the learned
output projection**. The readout is signed correct-minus-foil margin
response per unit dose. All 38 prompt pairs pass a fresh native
forced-choice competence check. This control is deliberately stronger
than an arbitrary random coordinate.

## Result

Ratio of mean candidate to mean paired-random signed margin response
per matched output dose. Values above one favor the answer-free gauge.
The columns are the two independent edit-fitting panels; the
held-out answer prompts are the same in both columns.

| Target | Answer-free gauge and coverage | Fit panel 1 | Fit panel 2 |
|---|---|---:|---:|
| Bracket | Raw edit PCA, all edit types | **1.050** | **1.051** |
| Bracket | Edit L1, all edit types | **1.050** | **1.052** |
| Bracket | Normalized PCA, all types | 0.975 | 1.013 |
| Bracket | Raw PCA, bracket edits excluded | 1.005 | 0.972 |
| Bracket | Edit L1, bracket edits excluded | 0.499 | 0.583 |
| Agreement | Raw edit PCA, all edit types | **1.046** | **1.051** |
| Agreement | Edit L1, all edit types | **1.055** | **1.048** |
| Agreement | Normalized PCA, all types | 0.790 | 0.862 |
| Agreement | Raw PCA, noun-number edits excluded | 0.691 | 0.723 |
| Agreement | Edit L1, noun-number edits excluded | 0.160 | 0.112 |

With all edit types present, raw PCA and edit L1 beat the random
control mean on **all 38 cases in each fit panel**. For agreement,
both also beat every one of the eight per-case controls in both panels.
The all-type bracket candidates beat the control *mean* on every case,
but not every individual random direction. The raw-PCA bracket
exclusion is essentially tied in panel 1 and worse in panel 2; it
does not give a stable positive result. The other exclusions fail
clearly. Normalizing edit vectors before PCA loses most of the
amplitude signal; its bracket result varies across panels, and
agreement is below control in both.

The saved case-level analysis includes paired prompt bootstraps for
each cell. These intervals condition on one model, two designed edit
panels, and templated answer prompts. Repeated token identities and
related templates limit statistical independence. We therefore focus
on the replicated direction and large exclusion contrast, rather
than treating the narrow prompt-bootstrap intervals as population
uncertainty.

## Interpretation and stopping rule

Answer labels are **not necessary for fitting a useful coordinate** in
these two previously task-selected heads, provided the input-edit
measure actually excites the relevant behavior. Under a matched
intervention dose, raw edit PCA and edit L1 are about 5% more effective
than unusually strong task-selected random controls. This is a
mechanistic advance over a sparsity proxy: the coordinates produce
more of the held-out task-margin response per unit residual edit.

It is still **not new-task discovery**. The head choice was supervised,
the edit generator explicitly included bracket and noun-number
substitutions, and the same model/task families have been studied
repeatedly. Whole-type exclusion shows the limit directly: unrelated
input edits do not recover the specialist reliably. There is no reason
to keep tuning gauges on this pair of heads. The next experiment needs
either broader task-blind head/operation selection with held-out
behavioral predictions, or a trained toy where shared computational
nodes are known and can be challenged by internal interventions.

Run flagship_later_head_matched_axis_dose.py with mode unlabeled and
unlabeled_validation, then analyze_later_head_matched_axis_dose.py
with matching modes. The four JSON files under full_width_results
retain all 3,040 finite candidate-control comparisons and case-level
summaries. The scripts assert competence, dose equality, and complete
case/control coverage.
