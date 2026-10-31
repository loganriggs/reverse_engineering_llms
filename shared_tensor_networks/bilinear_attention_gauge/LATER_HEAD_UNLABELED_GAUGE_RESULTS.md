# Can unlabeled natural activations orient the same head subspaces?

## Question and setup

The [supervised exact-gauge test](LATER_HEAD_EXACT_GAUGE_RESULTS.md) found
one influential coordinate in fixed rank-4 bracket and agreement head
subspaces. Removing the entire target task from its edit-based fit failed.
We asked whether a data-aware but **unlabeled** prior could recover those
coordinates from ordinary text, without task edits, answer labels, or
target-based axis selection.

The rank-4 weight-selected projector stays fixed. We captured final-position
head activations on disjoint FineWeb documents at context lengths 5 and 32,
and on WikiText-103 and TinyStories. We compared uncentered and centered PCA,
activation L1, and a fourth-moment concentration rotation. Each rotation is
orthogonal inside the same four-dimensional subspace, so the reduced model's
function does not change. The axis-selection rule was also unlabeled: choose
the coordinate with the smallest centered mean-absolute/RMS activation ratio
on the fitting text. We then evaluated that frozen axis against the held-out
bracket or agreement edit path. The task labels enter **only evaluation**.

## Apparent small-sample success and its failure

With 512 FineWeb contexts of length 5, centered activation L1 selected a
bracket axis with **1.2% local linear path error**. A finite patch through
the model also looked strong: **1.5%** error relative to the full rank-4
head patch on six new lexical/numeric edits, and **1.6%** on twelve new
prompt formats. This was an exploratory success among multiple methods and
context lengths, not a validated discovery.

One training context had a projected activation norm of **158.8**, far
above most examples; its text included an opening bracket (`" follows:\n[I"`).
Removing only that highest-norm context and refitting the same unlabeled
prior changed target local error from **.012 to .888**. Removing two gave
`.880`; removing more yielded unstable values. Independent 256-example
halves of the same 512-context panel gave `.023` and `.673`. The natural
training set's identity, not simply its size, was determining the axis.

We therefore expanded to **8,192** fresh FineWeb training contexts of
length 5, with two disjoint 4,096-context halves, a separate 2,048-context
FineWeb test panel, and 1,024 contexts each from WikiText-103 and
TinyStories. The full-data centered activation-L1 axis had **.942 local
path error**, and its finite path error was **.940** on the lexical panel
and **.953** on the format-shift panel. Its two large halves also failed
(`.819` and `.971` local error). Trimming the largest 0.2% of full-data
activation norms still gave `.923` local error. A separate 2,048-context
FineWeb fit happened to work (`.039` local error), reinforcing the
sample-dependence rather than showing a stable prior.

## Balanced multicorpus test

We next fit from **512 unlabeled contexts per corpus**, balanced across
FineWeb, WikiText-103, and TinyStories, in two independent splits. We
selected either the lowest *mean* activation sparsity ratio or the lowest
*worst-corpus* ratio, with no task labels. Neither rule recovered the
bracket path: centered activation L1 had about **.99 local error** in both
splits, confirmed by finite path errors **.994/.992** on the lexical
panel and **.987/.983** on changed prompt formats.

Agreement showed partial transfer under the multicorpus fourth-moment
rotation: finite single-axis error was **.077/.158** across the two
lexical splits and **.090/.172** across the format-shift splits. That is
better than many unlabeled controls but far weaker than the supervised
effect-axis error of about **.006/.004**. Its orientation also varied
between splits. Neither task has a validated, stable unlabeled axis under
these priors.

| Frozen axis | Bracket lexical / format error | Agreement lexical / format error |
|---|---:|---:|
| Supervised effect axis (positive control) | .006 / .008 | .006 / .004 |
| Small natural fit (exploratory) | .015 / .016 | .195 / .195 |
| Large 8,192-context natural fit | .940 / .953 | .806 / .813 |
| Balanced multicorpus, split 0 | .994 / .987 | .077 / .090 |
| Balanced multicorpus, split 1 | .992 / .983 | .158 / .172 |

Errors are mean absolute **finite single-axis versus full rank-4 head-path
effects**, normalized by the full path magnitude. The small natural fit uses
5-token centered activation L1 for bracket and 32-token centered PCA for
agreement; the large fit uses 5-token centered activation L1 for bracket
and 5-token centered PCA for agreement; the multicorpus fit uses centered
activation L1 for bracket and fourth-moment concentration for agreement.
These representatives were selected after inspecting a larger method grid,
so the table is diagnostic, not a fair preselected competition. The full
grid, including failed variants and random controls, is in the saved JSON.

## Interpretation and next test

The weight-only rank-4 *subspaces* remain faithful to these head paths, and
the supervised intervention gauges still provide compact coordinates.
Ordinary activation sparsity and PCA do not reliably choose those axes from
natural text. More natural samples did not repair the mismatch: they
changed which rare activation directions the objective valued. This is a
concrete example of why the **measure used to define importance** must be
declared separately from the **prior used to make a basis legible**.

The negative result is specific to these unlabeled priors, samples,
head outputs, and axis-selection rule. It does not show that unsupervised
canonicalization is impossible. A better next test is an unlabeled but
**balanced intervention ensemble**: sample diverse token substitutions and
context edits without answer labels, fit a response-sparse gauge, then hold
whole edit types and task families out. That may supply coverage of rare
functions without encoding their answers. It must beat the present natural
activation baselines on finite path predictions and show stable axes across
independent samples before any discovery claim.

Reproduce with `flagship_later_head_unlabeled_capture.py`,
`flagship_later_head_unlabeled_gauge.py`,
`flagship_later_head_unlabeled_stability.py`,
`flagship_later_head_unlabeled_outlier.py`,
`flagship_later_head_unlabeled_large.py`,
`flagship_later_head_unlabeled_large_eval.py`,
`flagship_later_head_multicorpus_unlabeled.py`, and
`flagship_later_head_unlabeled_finite.py`. Their arrays and numerical
records are under `full_width_results/later_head_unlabeled_*` and
`later_head_multicorpus_unlabeled*`.
