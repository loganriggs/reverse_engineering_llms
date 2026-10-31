# Later-layer attention loci for competent token-swap tasks

## Question and method

The [explicit-task gauge test](EXPLICIT_TASK_GAUGE_RESULTS.md) found that
first-layer QK1-key axes can look sparse and task-selective while carrying
almost none of the natural answer-changing edit. We tested whether attention
at *later* layers supplies a causal target for canonicalization.

For each of 23 previously screened, equal-length single-token prompt pairs,
we ran the 546M native model twice. At each of its 18 layers, we transplanted
the **entire edited-minus-base attention output** at all positions into the
base run and measured the correct-minus-foil next-token logit margin. We then
split four promising layers into their nine pre-output-projection heads and
patched one head at a time. No model weights or prompts were fit in this scan.
The selected layer/head pairs were frozen before a new 21-pair panel with new
countries, nouns, antonyms, and arithmetic expressions. Every pair changes
exactly one input token, retains the same two answer tokens, and is reported
without outcome filtering.

## Layer and head localization

| Family | Discovery pairs | Native edit, mean absolute margin | Best whole-attention layer | Whole-layer patch | Largest head there | Single-head patch |
|---|---:|---:|---:|---:|---:|---:|
| Country capital | 6 | 17.91 | 13 | 11.75 | 0 | 10.55 |
| Bracket closure | 8 | 14.70 | 13 | 6.10 | 8 | 5.44 |
| Subject–verb agreement | 5 | 8.06 | 11 | 5.54 | 3 | 5.52 |
| Antonym | 4 | 5.04 | 14 | 2.42 | 4 | 1.51 |

Antonym has a second comparable locus at **layer 8, head 7** (2.46 mean
absolute patch effect), which we selected for holdout because it is the
largest single-head effect in the four scanned layers. This is a deliberately
exploratory choice, not a preregistered layer. All selected discovery-head
effects have the same sign as the corresponding input edit for every pair.

The all-layer result is sharply different from the earlier layer-0 patch:
for example, the layer-13 capital effect is 11.75 on average, whereas the
layer-0 attention effect was .024. The point is a concrete, later-layer
attention path with substantial behavioral control, not an estimated
percentage of mediation.

## Frozen-selection holdout

| Family | New pairs correct / total | Frozen layer, head | Native edit | Selected-head patch | Selected head's share of summed absolute head effects | Last-position patch | Earlier-position patch |
|---|---:|---|---:|---:|---:|---:|---:|
| Country capital | 5/5 | 13, 0 | 14.33 | 7.39 | .884 | 7.36 | .017 |
| Bracket closure | 6/6 | 13, 8 | 18.57 | 6.57 | .885 | 6.50 | .100 |
| Subject–verb agreement | 6/6 | 11, 3 | 8.00 | 5.35 | .980 | 5.33 | .022 |
| Antonym | 4/4 | 8, 7 | 4.26 | 2.26 | .676 | 2.26 | .022 |

All values are mean absolute margin changes. The selected head remains the
largest at its layer for every holdout family, and its effect has the native
edit's sign on **all 21 pairs**. Almost all of that single-head effect is
caused by transplanting its vector at the **final prompt position**; the
earlier-position-only patch is small. This gives a precise representation to
reduce and analyze: the 128-dimensional, final-position output of a selected
later-layer attention head before its output projection.

## Interpretation and next experiment

This is path **sufficiency**, not a complete causal mediation estimate.
The patched attention vector is computed from an edited run whose upstream
residual stream already differs. A large transplant effect does not prove
the head generated the relevant variable independently, nor that its head
is necessary for the native computation. Single-head effects need not add
to the whole-layer effect because downstream processing is nonlinear. The
head shares reported above are descriptive fractions of summed absolute
single-head patch effects, not fractions of the native input edit.
The small, templated forced-choice panels do not establish broad language
generalization or human-readable features.

The next canonicalization comparison should work on this later-layer
128-dimensional head response, first at a fixed faithful function. Compare
weight-only directions from its output projection, broad corpus-variance
directions, a mixed broad/rare metric, and task/intervention-aware response
directions at matched rank. Separately test coordinate sparsity, edit-effect
sparsity, and reuse across the capital/bracket/grammar heads. Fit on training
prompts only; keep whole task families and new token edits held out. A
candidate component must retain native response/logit fidelity and predict
unseen finite edit effects substantially better than controls before being
called a mechanism. Intervene directly on head outputs in the native model;
do not reuse the failed first-layer QK1-key gauge as evidence.

Reproduce with `flagship_explicit_task_layer_scan.py`,
`flagship_explicit_task_head_scan.py`, and
`flagship_explicit_task_head_holdout.py`. Exact per-pair results are in
`full_width_results/all_head_explicit_task_layer_attention_scan.json`,
`all_head_explicit_task_head_attention_scan.json`, and
`all_head_explicit_task_head_holdout.json`.
