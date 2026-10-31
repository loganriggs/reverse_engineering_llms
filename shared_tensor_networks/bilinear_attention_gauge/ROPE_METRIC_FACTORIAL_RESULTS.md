# Uniform, corpus, and mixed metrics for pre-/post-RoPE rank-96 cuts

The [matched pre-RoPE result](MATCHED_PREROPE_FIT_RESULTS.md) used a 50/50 uniform-token and FineWeb-frequency pair objective. This factorial varies that weighting while holding the architecture, physical bond rank, sampled positions, WTG truncation procedure, and fit/selection/test document boundaries fixed. The main decision is that **mixed pre-RoPE** best preserves the full 18-layer model on three held-out corpora. Uniform-only and corpus-only cuts still win different pair-level objectives, especially after a QK2 branch edit. Thus reduction metric and intended fidelity target must be stated separately from any later legibility prior.

## Fit and validation design

For each of fit seeds 41, 43, and 47, draw 32,768 uniform-vocabulary and 32,768 FineWeb-training-document cross-token pairs across length-128 causal positions. Exclude relative gaps 32 and 48 from fitting and selection. In each pre- or post-RoPE geometry, fit three rank-96 WTG cuts with normalized per-law weights: **uniform-only**, **half uniform/half corpus**, and **corpus-only**. Uniform sampling uses the checkpoint's token embeddings and a uniform law over vocabulary IDs and cross positions; it uses no natural-text examples, but is a Monte Carlo approximation rather than an exact full-vocabulary contraction. Corpus samples come only from FineWeb documents 0–15,000. Selection uses independent pairs and documents 15,000–17,500; held-out FineWeb tests use documents 17,500–20,000. All six cuts in a seed use matched pair draws. Pre/post uncut-function identity residuals remain below \(3.1\times10^{-16}\); each saved cut is numerically rank 96 and WTG boundary eigenoperator residuals meet \(10^{-7}\).

Three-seed mean selection relative pair-score errors under each evaluation law:

| Fit prior | Post: uniform / corpus | Pre: uniform / corpus |
|---|---:|---:|
| Uniform-only | .07915 / .04847 | **.04532** / .02193 |
| Mixed | .07999 / .03590 | .04640 / .01551 |
| Corpus-only | .09742 / .03571 | .05947 / **.01535** |

The bold entries mark the best pre-RoPE value for each evaluation law, not a single universal winner. All three pre-RoPE cuts improve on their post-RoPE counterpart under both selection laws.

## Independent pair tests and stress cases

On 65,536 new uniform or held-out FineWeb pairs per law, mean relative pair-score MSE across three fit seeds:

| Fit prior / geometry | Uniform natural | Corpus natural | Uniform QK2-constant | Corpus QK2-constant |
|---|---:|---:|---:|---:|
| Uniform / post | .07617 | .03562 | .09339 | .09728 |
| Uniform / pre | **.04501** | .01883 | **.05812** | .06968 |
| Mixed / post | .07651 | .02918 | .09323 | .07793 |
| Mixed / pre | .04520 | **.01337** | .05865 | .04725 |
| Corpus / post | .09465 | .02978 | .11351 | .07103 |
| Corpus / pre | .05750 | .01386 | .07384 | **.04304** |

Uniform pre is best under uniform natural and uniform branch-edited metrics; mixed pre is best under natural corpus weighting; corpus pre is best on the corpus QK2-constant pair edit. Every pre cut beats its same-prior post cut in each bulk metric. The QK2-constant rows are **pair-function** tests with OV and QK1 held fixed, not downstream full-model branch interventions.

The separate held-out anchor panel preserves a sharp exception. At relative gap 48 and key/query positions (16, 64), corpus-key complete-head response error for mixed pre is **.05602** versus **.04994** post; corpus pre is **.05695** versus **.04598** post. Uniform-only pre is slightly better than uniform-only post there (.06082 versus .06457). The bulk test averages over many positions and 1,413 held-out-gap examples; its win should not be read as uniform position robustness.

## Downstream full-model decision

All six cuts from fit seed 41 were propagated through the full 18-layer model, editing layer 0/head 0 only. The main run uses two independent samples of 24 length-128 document windows per corpus: **6,144 next-token positions per corpus**, with paired windows across cuts. An identity pre-RoPE edit matches native logits exactly in this run. Mean original-to-edited KL per token, shown in units of \(10^{-6}\):

| Fit prior / geometry | FineWeb test | WikiText-103 | TinyStories |
|---|---:|---:|---:|
| Uniform / post | 16.83 | 21.71 | 13.05 |
| Uniform / pre | 5.18 | 7.13 | 3.35 |
| Mixed / post | 16.35 | 21.22 | 12.12 |
| **Mixed / pre** | **4.39** | **6.19** | **2.95** |
| Corpus / post | 20.92 | 26.42 | 15.34 |
| Corpus / pre | 5.42 | 8.14 | 3.95 |

Mixed pre has the lowest KL on all three corpora. A smaller cross-seed run (eight documents, one sample seed per corpus) repeats the factorial for fit seeds 43 and 47: mixed pre is again the lowest of the three pre cuts on **all six fit-seed × corpus comparisons**. Pre beats the same-prior post cut in all 36 paired rows across both full-model runs. CE shifts remain tiny and inconsistent; no language-model quality improvement is claimed. The larger KL ranking has one fit seed, while the smaller cross-seed check has fewer documents, so the estimated size of each prior difference still has sampling limits.

This resolves the first stage for this rank/head well enough to proceed: use the **mixed pre-RoPE cut** as the natural-behavior fidelity baseline, retain uniform and corpus cuts as controls, and keep a separate corpus QK2-constant branch-fidelity check. The next legibility-prior comparison should apply factor-weight sparsity, activation sparsity, ordinary-function response grouping, and branch-aware grouping within this **fixed cut**, so gauge changes preserve the exact edited function. Only then test whether groups transfer across positions and unseen documents and yield selective full-model effects. No semantic ontology is established by this metric sweep alone.

Reproduce the fits with `flagship_rope_metric_factorial.py --seed 41` (and 43, 47), the independent pair tests with `flagship_rope_metric_factorial_test.py`, and the full-model runs with `flagship_matched_rope_full_model.py --factorial`. The `head0_rope_metric_factorial_seed*.npz` files contain the six fitted matrices per seed; companion JSON files and the factorial test/full-model JSON retain validation, branch, gap, KL, CE, and fit diagnostics.
