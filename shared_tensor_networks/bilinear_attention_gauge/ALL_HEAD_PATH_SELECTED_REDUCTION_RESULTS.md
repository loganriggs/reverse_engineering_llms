# Selecting reduction targets by native downstream paths

The [all-head rank frontier](ALL_HEAD_RANK_FIDELITY_FRONTIER_RESULTS.md)
showed that a small head-state cut can preserve the dominant edit response
while missing entire edit families. We tested whether a task-blind downstream
path scan can choose which families merit fidelity. This makes the stage-one
metric more behavior-aware, but it remains distinct from the stage-two
question of which surviving coordinates are human-legible.

For every head in layers 9, 11, 13, and 15 of the 546M model, a **training**
finite-path scan ranked ten designed token-edit families by the mean L2 norm
of their native full-vocabulary final-logit head paths. We selected the top
three families per head. This uses no answer tokens or task labels; it does
use designed edit types and observed native model responses. The selected
families overlapped the validation-panel top three by a mean 2.33/3 heads
(15 exact matches, 18 with two, three with one).

We then fitted three orthogonal spectral cuts to each head's *training*
head-state edit responses: pooled edit covariance; all ten types with equal
immediate-`c_proj` output energy; and the selected three types with equal
output energy plus a fixed 10% all-type floor. The path scan **selects**
families; it does not provide continuous per-family weights or gradients
through the reduced model. The cuts are spectral surrogates rather than
solutions of the exact output- or final-logit-constrained problem. Independent
same-type validation edits evaluate every cut without choosing a winning
rank on that panel.

| Rank | Fit measure | Mean aggregate immediate-output relative MSE | Mean selected-type relative MSE | Mean worst unselected-type relative MSE |
|---:|---|---:|---:|---:|
| 4 | pooled | .592 | .582 | .921 |
| 4 | all-type balanced | .669 | .654 | .897 |
| 4 | path-selected + floor | .632 | **.535** | .941 |
| 32 | pooled | **.223** | .204 | .532 |
| 32 | all-type balanced | .253 | .252 | **.442** |
| 32 | path-selected + floor | .269 | **.159** | .589 |
| 64 | pooled | **.095** | .087 | .258 |
| 64 | all-type balanced | .107 | .107 | **.199** |
| 64 | path-selected + floor | .115 | **.071** | .277 |

These are averages over 36 heads. The selected-type column first averages
each selected type's separately normalized MSE within a head, then heads.
The final column averages each head's worst of seven unselected types. The
path-selected measure improves fidelity on its declared families at every
shown rank, at the expense of aggregate and unselected fidelity. This is a
**coverage choice**, not evidence that those families are intrinsically the
only useful computations.

## Finite full-model check

We also tested layer 9/head 7 at rank 32 with 12 independent validation
cases from each of its training-selected families (noun number,
preposition, name) and bracket as an unselected guardrail. For each case,
we compare the final-logit vector change caused by patching the native
head-state edit with the change caused by patching its reduced projection
into the reduced-head model. The denominator is that case's squared native
path norm; numbers below average those per-case ratios. This is a causal
path-fidelity measure, distinct from immediate-output error and from task
accuracy.

| Edit family | Pooled path MSE | All-type balanced path MSE | Path-selected path MSE |
|---|---:|---:|---:|
| Noun number | .114 | .169 | **.097** |
| Preposition | .231 | .253 | **.186** |
| Name | .236 | .255 | **.200** |
| Bracket (unselected) | **.122** | .152 | .372 |

Across the 36 selected-family cases, mean path MSE is .194 pooled, .226
all-type balanced, and .161 path-selected. On the same ordinary base
prompts, mean forward KL from the native model is .00284, .00291, and
.00339 respectively. Thus the selected fit also slightly worsens ordinary
forward fidelity; good edit-path fidelity is not a substitute for checking
it. These KL values cover only the 48 edited-panel base prompts, not a
fresh natural-text corpus.

This confirms the direction of the stage-one tradeoff downstream in this
one head: target selection improves the three intended edit paths, while a
bracket path gets much worse. The twelve-case means are exploratory, and
the held-out panel contains new instances of the **same** designed families.
No new-family transfer, task prediction, or human-legibility gain has been
established by this experiment. The native path norm selects families even
when a path is poorly aligned with the whole input edit; future selection
should distinguish path magnitude, signed task effect, and output KL.

**Decision.** Keep an explicit broad-coverage guardrail when selecting
important edit families. At the current 10% floor, rank 32 is too small
to preserve the bracket family for this broad head while favoring its
three selected families. Next compare floors and ranks with a training-only
selection rule, then check disjoint edit families and ordinary forward KL.
Only after a declared reduction passes those fidelity gates should a sparse
or reusable-coordinate prior be judged as an interpretability improvement.

Reproduce the all-head proxy with `flagship_path_weighted_rank.py` and the
finite check with `flagship_path_selected_finite.py`. Results are in
`full_width_results/all_head_path_weighted_rank.json` and
`full_width_results/path_selected_finite.json`.
The [training-only floor follow-up](PATH_FLOOR_GUARDRAIL_RESULTS.md) adds a
fourth finite-check method to the latter JSON and tests whether a local
coverage guardrail preserves downstream paths.
