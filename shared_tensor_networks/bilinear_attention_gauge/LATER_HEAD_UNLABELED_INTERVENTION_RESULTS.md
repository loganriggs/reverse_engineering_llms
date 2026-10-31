# Answer-free interventions recover two later-head coordinates

## Why this test

Passive natural-activation sparsity [failed a stability red team](LATER_HEAD_UNLABELED_GAUGE_RESULTS.md).
We kept the same faithful, weight-only rank-4 projectors for layer 13 head 8
and layer 11 head 3, but changed the **gauge-fitting measure** from ordinary
activations to changes caused by diverse token substitutions. No answer
tokens, task margins, or model gradients enter the fitting or axis-selection
rules. The input-edit distribution is deliberately designed, so this is
answer-free intervention supervision, not a fully blind discovery method.

We generated two independent panels of 800 one-token edits each: 80 in
each of ten types (opening bracket, singular/plural noun, adjective, color,
country, name, number, punctuation, preposition, and a random substitution
in natural text). The panels have no exact or reversed pair overlap;
random-context edits use disjoint FineWeb document ranges. The fitting
prompts differ from the held-out bracket/grammar answer prompts. We
captured each edit's final-position head-output difference in the frozen
four-dimensional subspace.

## Exact rotations and finite tests

We fitted rotations using raw edit-vector PCA, row-normalized PCA, or an
edit-vector L1 prior. For the primary answer-free rule, the selected axis
is the coordinate with the largest **global raw edit variance**. The
rotation and selection use no edit-type label. A positive control uses the
earlier answer-margin-supervised effect axis. The model's rank-4 projected
function is identical under every rotation.

The table reports finite single-axis patch error relative to transplanting
the complete rank-4 head vector on prompt panels unseen during fitting.
Lower is better. Each cell is **lexical/numeric holdout / new prompt format**.

| Fitting and axis rule | Bracket | Agreement |
|---|---:|---:|
| Answer-free raw PCA, panel 1 | **.015 / .021** | .022 / .028 |
| Answer-free raw PCA, independent panel 2 | **.015 / .020** | **.019 / .024** |
| Answer-free edit L1, panel 1 | .015 / .020 | .021 / .025 |
| Answer-free edit L1, independent panel 2 | .015 / .020 | .039 / .046 |
| Row-normalized PCA, panel 1 | .11 / .11 | .45 / .47 |
| Raw PCA, matching edit type excluded | .25 / .24 | .58 / .59 |
| Edit L1, matching edit type excluded | .79 / .78 | .98 / .98 |
| Answer-margin-supervised axis | .006 / .008 | .006 / .004 |

The two independently fitted raw-PCA axes have absolute cosine **1.000**
for bracket and **.998** for agreement. They retain the finite path on
held-out edits and changed formats much more reliably than passive
activation sparsity. The supervised axes remain more precise, but answer
labels are not necessary to obtain a useful coordinate **when the relevant
input edit type is included**.

This recovery is driven largely by response amplitude. In the training
panel, bracket edits have mean rank-4 response norm about **203** in the
bracket head, versus roughly **1–9** for other edit types. Noun-number edits
have norm about **46** in the agreement head, versus roughly **1–4** for most
others. Normalizing every edit before PCA erases part of this useful
importance signal and worsens the path prediction. This is a choice of
measure: raw PCA prioritizes large *absolute* effects, not equal treatment
of every edit type.

## What is and is not discovered

On the independent validation intervention panel, the largest 10% of
absolute selected-axis responses are **100% opening-bracket edits** in the
bracket head and **95% noun-number edits** in the agreement head. The
axis responses are enriched 136-fold and 35-fold, respectively, relative
to other edit types. Those are useful input-side candidate labels that one
could inspect before supplying answer labels.

However, ranking edits by the **full rank-4 head-response norm** already
recovers 100% of bracket edits and 91% of noun-number edits at the same
top-decile threshold. The coordinate improves agreement precision only
modestly over this simpler head-level baseline. The heads were originally
selected using task labels, and the intervention generator explicitly
contains bracket and number edits. This does **not** establish genuine
discovery of an unseen task, a unique canonical basis, or a necessary
answer circuit.

Whole-edit-type exclusion is a meaningful negative control. Removing
opening-bracket edits from fitting raises raw-PCA finite path error to
roughly 25%; removing noun-number edits raises it to roughly 59%.
Edit L1 degrades more. The model can expose the corresponding variable to
an answer-free procedure, but the fitting measure must excite it.

## How much rare-edit coverage is enough?

We resampled the fixed intervention pool with a declared probability for
the matching edit type, fitted the top raw-PCA direction, and scored the
held-out local task path. Across 100 bootstrap repetitions, the fraction
below 10% relative local error was:

| Matching edit probability | Bracket, 800 samples | Bracket, 8,000 | Agreement, 800 | Agreement, 8,000 |
|---:|---:|---:|---:|---:|
| 0 | 0% | 0% | 0% | 0% |
| 0.1% | 57% | 100% | 10% | 31% |
| 0.5% | 99% | 100% | 59% | 93% |
| 1% | 100% | 100% | 83% | 100% |

These are resamples of **80 unique matching edits**, not thousands of new
model forwards. They quantify the sensitivity of this *declared* edit
measure, not an empirical estimate of natural edit frequencies. A small
coverage floor can make recovery reliable if enough examples are drawn;
more samples cannot help when the edit type has zero support. The stronger
bracket signal needs fewer matching examples than the agreement signal.

## Next discriminator

Apply the same answer-free edit panel to **all heads and several layers**
without choosing heads from known tasks. Rank heads and candidate axes by
reproducible edit responses, then have a separate validation panel infer
input-side labels and predict new downstream behaviors. Compare directly
with full-head response norm, raw edit PCA, and random axes. Hold complete
edit types and task families out of each stage. Only then can an apparent
component count as a newly discovered task rather than rediscovery of
human-designed probes at previously selected heads.

Reproduce with `flagship_later_head_unlabeled_edits.py`,
`flagship_later_head_unlabeled_edit_gauge.py`,
`flagship_later_head_unlabeled_edit_finite.py`,
`flagship_later_head_unlabeled_edit_analysis.py`, and
`flagship_later_head_edit_frequency.py`. Their cases, arrays, rotations,
finite patch records, and frequency-sweep results are saved under
`full_width_results/later_head_unlabeled_edit*` and
`later_head_edit_frequency.json`.
