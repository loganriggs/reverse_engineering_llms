# Balanced output-stratum gauge on the 546M model

## Question and protocol

Does the toy result—task-response supervision finding reusable coordinates—
survive even a coarse test in the 546M bilinear-attention model? We held a
previously fitted rank-256 first-layer QK1-key projector fixed and collected
gradients of the observed next token's log probability through all 18 layers.
The target tokens were balanced across six token-form categories, with
separate FineWeb training, validation, and test document ranges plus
WikiText-103 and TinyStories tests. One document supplied one 64-token
context and one target; the 64 key-position gradients within it are
correlated observations, so uncertainty is computed at document level.

The new varimax rotation saw only **digits, punctuation, space-prefixed
words, and word fragments** in 96 FineWeb training documents (16 per
category were collected; four categories were used for fitting). Entire
capitalized-word and mixed-token categories were held out from gauge
fitting. The reduction, model, and output readout were frozen. A weight-based
rotation, a prior pair-response rotation, and an all-six-category rotation
serve as controls. The latter sees the held categories and is an oracle,
not a valid held-out method.

The hooked head output reproduced the original exactly in the pilot and
main collection, and central finite differences agreed with autograd to
within 3.2e-5 absolute error. The main panel used 96 train, 48 validation,
96 FineWeb test, 48 WikiText test, and 40 TinyStories test documents. The
GPU was occupied by the separate task-coverage job; all work here ran
on CPU without interrupting it.

## What transferred, and what did not

Effective gradient coordinates are `(sum |g_i|)^2 / sum g_i^2`, averaged
over positions and then documents; smaller means a gradient is concentrated
in fewer coordinates. On the two **held-out target categories**, the
rotation fitted without them improves this score in every test document
relative to the prior weight rotation:

| Test corpus | Weight rotation | Four-category rotation | All-category oracle |
|---|---:|---:|---:|
| FineWeb | 127.66 | 89.98 | 88.06 |
| WikiText-103 | 136.15 | 106.88 | 103.97 |
| TinyStories | 129.68 | 87.56 | 82.44 |

Each row averages the two held categories equally. Paired document
bootstrap intervals for the four-category-minus-weight difference exclude
zero for both categories in all three corpora. For example, on FineWeb
capitalized words the difference is -36.76 [−41.28, −31.91], and on mixed
tokens it is -38.60 [−41.13, −36.18]. The full per-category intervals are
in the document-analysis JSON. The all-category oracle is only modestly
sparser on held test data than the four-category rotation, but their axes
match at mean absolute cosine .582; a rotation fitted only on the two held
categories matches at .223. A good sparsity score therefore does not identify
one stable coordinate system.

We preselected one axis per **fit** category using its training gradient
energy enrichment over the other fit categories, then kept those axes fixed.
The strongest labels do not transfer reliably:

| Training-selected label | Train enrichment | Validation | FineWeb test | WikiText test | TinyStories test |
|---|---:|---:|---:|---:|---:|
| Digits | 2.94 | .77 | 1.25 | .77 | unavailable |
| Punctuation | 5.71 | .67 | .84 | .63 | .57 |
| Space-prefixed word | 14.72 | 1.64 | .71 | 2.37 | 1.14 |
| Word fragment | 6.37 | .59 | 1.14 | .75 | .87 |

Those figures use all key positions and normalize each document's total
gradient energy. An independent fixed-position version likewise fails
consistent transfer. Thus the category-balanced gauge gives broad gradient
sparsity, but **no supported named target-category component**. We stopped
before selecting finite interventions on these unstable labels. This avoids
repeating the earlier punctuation-axis detour with a new set of categories.

These categories are output strata, not separately verified tasks. They also
have unequal model competence: WikiText mixed targets had zero top-1
accuracy in eight sampled documents, so that stratum cannot validate a
useful computation. The present evidence neither establishes nor rules out
task-supervised canonicalization on richer, competent task families.

## Course correction

For the 546M model, gradient sparsity alone is too permissive. The next
comparison should use explicit behaviors with measurable task success and
disjoint task-family holdouts, then score finite interventions and downstream
logit/KL fidelity at a fixed reduction. The toy's one-axis token-edit result
is a useful target for that comparison; this coarse token-form panel did not
replicate its semantic selectivity.

## Reproduction

`flagship_balanced_output_gauge.py` collects the balanced panel and fits
the exact rotations. `analyze_balanced_output_gauge.py` produces paired
document intervals and frozen-axis label-transfer checks. The panel,
rotation, and analysis artifacts use the stem
`all_head_balanced_output_gauge_train16_val8_test16_ext8_s9171` under
`full_width_results/`. The smaller `train1_val1_test1_ext1_s9071` run was
used only to validate the hook and sampler.
