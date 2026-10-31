# A branch-aware first-layer key basis has a modest, corpus-dependent full-model gain

The [toy branch tests](SHARED_PAIR_BRANCH_CAUSAL_RESULTS.md) showed that an
ordinary semantic reduction can miss physical QK-branch response directions.
The 546M model already has four saved, **matched rank-256** shared QK1-key
bases: weight-only uniform-vocabulary SVD, usage-weighted key SVD, a basis
optimized for ordinary nine-head pair-output fidelity, and one optimized
for a 50/50 mixture of ordinary and QK2-constant pair-output fidelity.
This experiment asks whether the last basis better predicts a *small QK2
edit's effect on final logits* after all 18 layers, without refitting any
basis on the evaluation documents.

The first-layer edit replaces every QK2 score `s₂` with
`(1-α)s₂ + α·1` **before** multiplication by QK1. For each basis and
document, run the full model unedited and edited, then subtract to get
its predicted logit response. Compare that with the native model's
edited-minus-unedited response. The primary score is pooled squared
response error divided by pooled native response energy. Ordinary and
edited KL to the native model are reported separately. An identity
rank-1152 key projection reproduces native logits within `3.0e-5`
before the edit and `2.2e-5` after it across the saved panels.

At `α=.05`, two independently sampled panels contribute **24 disjoint
128-token documents per corpus** (FineWeb test documents 17,500–20,000,
WikiText-103, TinyStories). The FineWeb fit used documents 0–15,000 and
validation 15,000–17,500; the other two corpora are external tests.
The score uses all output positions and vocabulary logits, not isolated
attention pairs or a selected task label.

| Basis, rank 256 | FineWeb response MSE ↓ | WikiText response MSE ↓ | TinyStories response MSE ↓ |
|---|---:|---:|---:|
| Uniform key-table SVD | .0343 | .0755 | .0570 |
| Usage-weighted key-table SVD | .0171 | .0396 | .0165 |
| Ordinary function-aware | .0175 | **.0238** | .0108 |
| QK2-branch-aware | **.0163** | .0245 | **.0107** |

Paired document bootstraps (10,000 draws within each corpus) give the
branch-aware minus ordinary-function response-MSE difference as
**−.0013** on FineWeb, 95% interval `[−.0020, −.0005]`, with 20/24
documents favoring branch-aware. The difference is **+.0007** on
WikiText, interval `[−.0006, +.0022]`, 13/24 wins; and **−.0001** on
TinyStories, interval `[−.0005, +.0003]`, 13/24 wins. Thus the
branch-aware objective yields a reproducible but small advantage on
FineWeb and no established advantage over ordinary function-aware
fitting on the two external corpora. Both function-aware bases beat
uniform key-table SVD on every corpus; the usage-only basis slightly
beats ordinary function-aware fitting on FineWeb response MSE, while
losing on the two external corpora. Mean ordinary and edited KL are
very close for the two
function-aware bases; the response score detects a difference that KL
alone does not summarize.

The pooled response-energy effective sample sizes are 21.3, 22.2, and
23.0 documents in corpus order, with the largest document carrying
under 8% of energy. These panels are broad enough to avoid a single
energy-dominating outlier, but 24 documents per corpus are still an
initial transfer check, not proof of universal ordering.

An `α=.01` check on the same first eight documents per corpus preserves
the direction of the comparison: branch-aware versus ordinary-function
response MSE is `.0152/.0162` on FineWeb, `.0264/.0232` on WikiText,
and `.0127/.0130` on TinyStories. The identity projection gate remains
small relative to the native 1% response. The small-dose test therefore
does not turn the FineWeb-specific improvement into an external-corpus
gain.

This tests a **structured QK1-key reduction** whose fitting objective
includes a QK2 edit. It is not yet a canonicalization of the complete
attention function across both branches and heads. A
separate [held-out QK2 edit-family test](FULL_MODEL_HELDOUT_BRANCH_EDIT_RESULTS.md)
preserves the small FineWeb benefit but finds no established external-corpus
gain over ordinary function-aware fitting. The toy's
condition-wise product and joint-nullspace priors cannot be transplanted
unchanged: the real model has many heads, RoPE positions, and no supplied
output/query cells known to each contain one product. The next
legibility experiment should define condition-indexed *full-context
causal response signatures*, fit on diverse training contexts and edit
families, and hold out both documents and entire edit families. It
should compare ordinary, branch-aware, and uniform reduction metrics at
matched rank and intervention dose, and validate any components through
the full model rather than token-pair labels.

Reproduce with `flagship_shared_key_branch_response.py` for `--n-docs 8
--seed 891 --length 128 --dose .05`, `--n-docs 16 --seed 892 --length
128 --dose .05`, and the eight-document `.01` dose check; then run
`analyze_flagship_branch_response.py`. JSON records and the paired
bootstrap summary are in `full_width_results/all_head_shared_key_branch_response_*`.
