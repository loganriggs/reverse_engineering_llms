# Branch-aware key reduction has a small, context-dependent full-model gain

The [function-aware shared-key fit](FUNCTION_AWARE_SHARED_KEY_RESULTS.md)
compared rank-256 projections of the 546M model's first-layer QK1 keys
under natural pair response and a 50:50 mixture of natural and QK2-constant
pair response. An [earlier full-model branch-response test](FULL_MODEL_BRANCH_RESPONSE_RESULTS.md)
found a small FineWeb advantage at `α=.05` when predicting the
**edited-minus-unedited logit response**. Here we vary dose from zero to
one and ask a complementary question: how faithfully does each cut
reproduce the complete branch-edited model's next-token distribution?

We freeze four saved rank-256 shared-key subspaces: usage-weighted SVD,
natural-function fitting, 50:50 natural/QK2-constant fitting, and
QK2-constant-only fitting. In every first-layer head, interpolate the QK2
score `s₂` toward one: `s₂(α)=(1-α)s₂+α`, with `α∈{0,.25,.5,1}`. For each
strength, the **reference** is the same 546M model with this branch edit
and no QK1 key projection. Candidate cuts are compared with that reference
by final-logit KL after all 18 layers. This measures faithfulness to the
edited computation, not an improvement in the original model's task.

The panel has 16 length-128 document windows from each of two sampling
seeds per corpus: **32 windows / 4,096 prediction positions per corpus per
strength**, paired exactly across the four strengths and four cuts. FineWeb
test documents exclude training/selection ranges; WikiText-103 and
TinyStories are external corpora. We also excluded 191 documents per corpus
used by earlier shared-key full-model panels. The `α=0` branch reference
matches native logits and CE exactly; identity key projection changes final
logits by at most `2.2e-5`.

Mean KL to the *unprojected branch-edited reference*:

| Corpus | α | Natural-function rank 256 | 50:50 branch-aware rank 256 | Paired branch-aware wins / 32 |
|---|---:|---:|---:|---:|
| FineWeb | 0 | .0001252 | **.0001177** | 22 |
| FineWeb | .25 | .0009003 | **.0008541** | 22 |
| FineWeb | .5 | .0031610 | **.0030318** | 22 |
| FineWeb | 1 | .0056920 | **.0053267** | 25 |
| WikiText-103 | 0 | **.0002338** | .0002686 | 13 |
| WikiText-103 | .25 | **.0015369** | .0015666 | 19 |
| WikiText-103 | .5 | .0055068 | **.0054341** | 18 |
| WikiText-103 | 1 | **.0086371** | .0086744 | 21 |
| TinyStories | 0 | **.0000668** | .0000673 | 16 |
| TinyStories | .25 | .0004605 | **.0004322** | 26 |
| TinyStories | .5 | .0025494 | **.0023933** | 20 |
| TinyStories | 1 | .0043234 | **.0040801** | 22 |

For FineWeb, the paired mean KL advantage of the 50:50 cut at `α=.25` is
`4.63e-5`, with a seed-stratified window-bootstrap 95% interval
`[2.37e-5,6.95e-5]` for the *reduction* in KL. At full strength the
advantage is `3.65e-4`, interval `[1.75e-4,5.95e-4]`. TinyStories has a
clear quarter-strength advantage (`2.83e-5`, interval
`[1.51e-5,4.25e-5]`), but its half- and full-strength intervals touch
zero. WikiText-103 intervals include zero at all edited strengths. The
QK2-only fit is not consistently better than 50:50; at full strength it
is worse on WikiText and TinyStories. The usage-SVD control is generally
less faithful than the function-aware cuts, but can match or beat them
on individual windows.

The intervention is substantial even at quarter strength. Mean KL from
the **native unedited** model to the branch-edited reference at
`α=.25/.5/1` is `.121/.584/1.070` on FineWeb, `.101/.396/.787` on
WikiText, and `.078/.427/.847` on TinyStories. At full strength, native
next-token CE rises from `3.26→4.30`, `4.10→4.86`, and `2.76→3.75`
respectively. The branch-aware projection's small KL improvement does
not produce a consistent CE improvement over the natural-function cut.

This dose and KL-to-edited-reference test complements the earlier
small-dose response-MSE panel. It supports **intervention-specific fidelity** on some
corpora, with a measurable natural-domain tradeoff on WikiText. It does
not establish a general causal canonical form, branch-composition transfer,
or a human-legible component. There is one trained fit per objective and
two document-sampling seeds; the bootstrap measures window variability,
not optimization-seed variability. The next legibility stage must hold a
reduced function fixed and compare weight, response, and intervention
priors by selective effects on unseen edits rather than by KL alone.

Reproduce with `flagship_shared_key_branch_full_model.py` using
`--n-docs 16 --seeds 91 93 --length 128` and each of
`--branch-strength 0`, `.25`, `.5`, and `1`; then run
`flagship_shared_key_branch_dose_summary.py`. Per-window logits-derived
metrics and exact spans are in the four
`full_width_results/all_head_shared_key_branch_full_model_n16_s91-93*.json`
files; paired means and bootstrap intervals are in
`full_width_results/all_head_shared_key_branch_dose_summary.json`.
