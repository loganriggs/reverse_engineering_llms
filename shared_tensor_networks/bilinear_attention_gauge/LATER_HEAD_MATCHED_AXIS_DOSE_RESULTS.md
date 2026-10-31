# Task-trained later-head axes retain a modest advantage at matched causal dose

The [later-head exact-gauge study](LATER_HEAD_EXACT_GAUGE_RESULTS.md)
found that sparse edit and task-effect orientations each concentrate a
rank-four head's bracket or agreement edit path into one coordinate.
One-coordinate patches can have different sizes after the learned output
projection, however. This test asks whether their **finite task-margin
effect per unit residual edit** is better than strong random-axis
controls at the same dose.

## Frozen objects and paired test

The 546M bilinear model, the weight-only rank-four projectors, rotations,
and train-selected axes are unchanged. The bracket target is layer 13,
head 8; the agreement target is layer 11, head 3. Both physical heads were
originally chosen with task knowledge. The effect and edit gauges were
fitted with target-family discovery examples; they are **not**
unsupervised discoveries. The random controls are also given a favorable
supervised selection: each of 64 pre-existing random rotations uses its
best axis by mean absolute local task effect on discovery pairs.

We use the 21 lexical/numeric held-out pairs and 26 new-format pairs from
the earlier study, retaining only bracket and agreement examples: **18
bracket and 20 agreement pairs**. Each base and edited prompt has the
expected native forced-choice answer, checked again in this run. For
each pair, we obtain the edited-minus-base final-position head state and
patch only one coordinate into the **base** prompt of the rank-four
projected model. The signed readout is the change in
correct-minus-foil margin, oriented by the full rank-four patch's sign.
All model layers downstream of the patched head run normally.

The dose is the Euclidean norm of the patch **after that head's learned
output projection**, before downstream nonlinearities. For each
prompt, we take the eight random axes with largest *unscaled* dose from
the 64-axis pool, using no held-out answer labels. For each
candidate-versus-control pair, scale both injections down to the smaller
of their original doses. The script asserts equality of their calculated
post-projection doses. This controls edit magnitude more directly than
comparing unscaled coordinate deletions.

## Result

The table reports the ratio of mean signed finite margin response per
unit matched residual dose to the corresponding random controls.
The final column counts prompt cases in which the candidate exceeds the
mean of its eight paired random controls.

| Target | Effect-sparse | Edit-sparse | SVD axis | Weight-sparse | Effect-axis case wins |
|---|---:|---:|---:|---:|---:|
| Bracket closure, 18 pairs | **1.084** | 1.057 | 0.818 | 0.874 | **18/18** |
| Agreement, 20 pairs | **1.057** | 1.053 | 1.025 | 0.764 | **20/20** |

The effect axis exceeds **all eight** per-case random controls on 17/18
bracket pairs and 20/20 agreement pairs. Edit sparsity exceeds the
control *mean* in all 38 pairs, but not every individual control for
brackets. Weight sparsity falls below the random mean in all 38 pairs.
The SVD axis is helpful for agreement, consistent with the earlier
finding that its unrotated coordinate already captures much of that path.

For the effect axis, the mean candidate-minus-control response per unit
dose is +0.000193 for brackets and +0.000116 for agreement. A 10,000-draw
paired **prompt** bootstrap, stratified by lexical versus new-format
panel, gives [0.000177, 0.000209] and [0.000109, 0.000123].
These narrow intervals describe sampling of these templated prompts for
one trained model. Repeated words, operators, and constructions make
the prompts dependent; the intervals should not be read as evidence
across model seeds, arbitrary language, or a pre-registered task search.
The saved analysis includes all four orientations and each case's
matched dose and control effects.

## Interpretation

This passes a specific **dose-matched selectivity gate** for two
task-trained explanatory coordinates: their advantage is not merely
that they inject more residual energy than a random coordinate.
It also distinguishes legibility priors. Sparse physical output weights
do not predict task impact here, while sparse edit responses and
task-effect concentration do. The effect-axis gain over a fairly strong
random null is modest, around 6–8% under this readout.

It does **not** identify a unique physical node. All rotations represent
the same rank-four projected head function; axis patches intervene on
the chosen explanatory gauge. Earlier reverse patches of the complete
native head rarely restored the original answer, so this coordinate is
not a necessary or sufficient full answer circuit. The two heads and
task families were selected with labels, and the rank-four path itself
omits alternative routes. The next discovery test must withhold entire
task families from head and gauge selection, or use an operation-level
toy with known graph structure, then apply the same matched-dose gate on
new model seeds and broader contexts.

Reproduce with flagship_later_head_matched_axis_dose.py, then
analyze_later_head_matched_axis_dose.py. The two JSON artifacts under
full_width_results/later_head_matched_axis_dose*.json contain all 1,216
candidate-control comparisons and case-level bootstrap results.
