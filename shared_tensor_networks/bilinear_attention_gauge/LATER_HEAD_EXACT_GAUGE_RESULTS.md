# Exact gauges of later attention-head responses

## What is fixed

The [matched-rank pilot](LATER_HEAD_REDUCTION_METRIC_RESULTS.md) identified
two faithful rank-4 cuts: layer 13 head 8 for bracket closure and layer 11
head 3 for subject–verb agreement. We freeze each 128-to-4 orthonormal
projector `P = B Bᵀ`. For every orthogonal four-by-four rotation `R`,
`(B R)(B R)ᵀ = P`; the reduced attention output and every downstream logit
are therefore identical. Across 142 saved rotations, the maximum projector
entry difference was `2.50e-16`. Only the *coordinates used to describe and
intervene on that fixed response* change.

This is the second choice in the research matrix: a legibility prior after
selecting the reduction metric. It is not a canonical form of the complete
attention function, and these rank-4 cuts omit other head directions.

## Priors and evaluation split

We captured final-position head edit vectors `d` and local correct-minus-foil
margin gradients `g` for 23 discovery token-swap pairs across four task
families, then held out 21 new lexical/numeric pairs. The priors were:

* **Weight L1:** rotate the four projected output-projection columns to have
  sparse physical coefficients. No examples enter this rotation.
* **Edit L1:** rotate the four coordinates of normalized `d` to be sparse.
  It uses input interventions but not the answer margin in its fit loss.
* **Effect concentration:** rotate to concentrate local axis responses
  `(d·u_j)(g·u_j)`, using a smoothed `|response|^1/2` penalty. This uses the
  answer margin. A synthetic one-axis gate checks that the loss prefers one
  axis over a 45-degree split; ordinary smoothed L1 failed this gate because
  it could favor spreading a fixed signed effect.

Each fit used eight optimization starts. We compared the native SVD gauge,
64 random rotations with the same *target-label-based axis selection*,
all-family fits, target-only fits, and fits excluding the entire target
family. The named axis for ordinary variants was selected on discovery
examples of that family, then frozen. For excluded-target variants, both
rotation and axis selection used only other families. These are distinct
supervised and target-excluded evaluations, not unsupervised discovery.
Central finite differences of one projected coordinate per head agree with
the captured analytic margin gradients within `4.5e-5` absolute error.

## Held-out finite effects

The table reports the error of a **single selected-axis finite patch**
relative to the full rank-4 head-vector patch on fresh token pairs. Lower
means that one coordinate reproduces more of the fixed reduced head path.
All patches run through the native downstream network, rather than relying
on the local gradient approximation used in fitting.

| Gauge | Bracket (6 pairs) | Agreement (6 pairs) |
|---|---:|---:|
| SVD basis | .487 | .037 |
| Sparse output weights | .436 | .464 |
| Sparse edit vectors, all families | .017 | **.004** |
| Sparse local effects, all families | **.006** | .006 |
| Sparse edit vectors, target excluded | .418 | .999 |
| Sparse local effects, target excluded | .935 | .826 |
| Random rotations, median of 64 | .412 | .363 |

The effect-concentrating gauge beats all 64 sampled random rotations for
both tasks under this metric (empirical rank `1/65`). The edit-vector gauge
beats all 64 for agreement, and 61 of 64 for bracket. The SVD basis already
has a useful agreement axis, so the gain there is smaller than for brackets.
The output-weight-sparse gauge does not produce a useful single-axis
explanation, despite the **weight-selected subspace** being highly faithful.
Choosing a good subspace and choosing a good basis inside it are different
problems.

The held-out mean effective number of *local response axes* mirrors the
finite test: for bracket, SVD `2.62`, weight L1 `2.56`, edit L1 `1.04`,
effect concentration `1.03`; for agreement, `1.09`, `2.24`, `1.01`, and
`1.02`. Excluding the target task raises edit-gauge counts to `2.51` and
`2.00`. This failure is important: the current procedure does not discover
these axes from unrelated task edits alone. It needs target-family examples
or a more informative shared prior.

## Frozen format-shift test

We kept the rank cuts, rotations, and selected axes frozen for another
26 unfiltered prompt pairs: 12 brackets with new operators or prefixes,
and 14 agreement cases with new prepositions or prefixes. All **26/26**
base and edited prompts retained the expected forced-choice answer.

| Family | Rank-4 path vs native | Effect-axis vs full rank-4 path | Effect-axis vs native path | SVD axis vs full rank-4 path |
|---|---:|---:|---:|---:|
| Bracket | .020 | **.0076** | .0236 | .506 |
| Agreement | .046 | **.0044** | .0479 | .038 |

All numbers are relative mean absolute finite path-effect errors. The
effect-axis patch has the native edit's sign in every format-shift pair.
The finite target-pair and format-shift panels therefore support a
**behaviorally compact coordinate** for these specific head paths. They do
not establish necessity, full-vocabulary semantics, or a uniquely determined
canonical axis. The agreement head and bracket head were themselves chosen
for task-selective patch effects, and the axis selection saw target-family
training labels. We have not established transfer to a task family absent
from fitting or to broad natural text.

## Reverse-patch red team

We also started from each edited prompt and transplanted the base prompt's
head state back into it. The effect-optimized axis had the expected reverse
sign on **all 38** bracket/agreement pairs across the lexical and format
panels. Its reverse finite effect matched the complete rank-4 reverse patch
within `8.5%`/`9.5%` for bracket and `0.9%`/`0.6%` for agreement on the two
panels, respectively. The rank-4 reverse patch itself matched the full
native-head reverse effect within `1.3%`/`1.5%` for bracket and
`6.7%`/`3.9%` for agreement.

Yet restoring the **entire native head vector** changed the edited
forced-choice answer back to the base choice in only `1/18` bracket cases
and `2/20` agreement cases. The axis matters in both directions, but the
rest of the edited computation usually still supports the edited answer.
This is evidence against treating this coordinate as a necessary, complete
answer circuit. Reverse path control is not the same as a necessity proof
under all interventions.

## Next discriminator

Test richer tasks or longer contexts in which these same heads participate,
and identify what other paths maintain the edited answer when the head is
restored. A causal necessity claim needs broader interventions or a model
of redundant paths; the reverse patch alone does not establish one.
For genuine discovery, withhold a whole family from both rotation fitting
and axis selection; the current excluded-target result is negative. A
shared dictionary or graph prior would need to improve that holdout rather
than only making seen-family edits sparse.

Reproduce with `flagship_later_head_gauge_capture.py`,
`flagship_later_head_exact_gauge.py`,
`flagship_later_head_gauge_finite.py`, and
`flagship_later_head_format_shift.py`, and
`flagship_later_head_reverse_patch.py`. The corresponding `later_head_gauge_*`
and `later_head_exact_gauge_*` arrays and JSON files are in
`full_width_results/`, along with `later_head_reverse_patch.json`.
