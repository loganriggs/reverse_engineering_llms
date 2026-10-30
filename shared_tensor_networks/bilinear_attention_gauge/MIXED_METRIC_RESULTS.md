# Fixed metric mixtures for the 546M model's first attention head

This tests the first half of the proposed factorial study: how the reduction metric changes a rank-limited QK1 bond cut. It does **not** test a sparsity or human-legibility prior. The target is head 0 in layer 0 of the saved 546M bilinear-attention model. The original physical bond has width 128; we retain rank 32 or 64. All fits use CPU and the cached model weights. No full-model logits, KL, or cross-entropy are measured.

The earlier [context-transfer experiment](FLAGSHIP_CONTEXT_TRANSFER_RESULTS.md) showed that fitting on four-token windows hides a large long-context cross-token error. Accordingly, every mixture here uses length-128 windows. The two fixed measures are:

- **Half:** 50% uniform-token natural QK2 pairs, 50% FineWeb natural QK2 pairs.
- **Thirds:** one third each of uniform-token natural QK2, FineWeb natural QK2, and uniform-token pairs with QK2 replaced by a constant score.

Each component is divided by its *own original response energy* before mixing. Thus the coefficients control relative squared errors, rather than letting the highest-energy condition dominate. The code asserts that the combined sampled-bond operator equals the explicitly weighted component errors on a nontrivial probe. These coefficients were declared before the held-out evaluation; they were not selected by searching the reported conditions.

Both fits use 65,536 pairs per natural sampling law, seed 17, and an independent 65,536-pair validation sample per law. FineWeb fit documents are 0–14,999; validation documents are 15,000–19,999. The constant-QK2 component reuses the *same* uniform pairs with changed QK2 scores. The method is weighted-trace-gauge (WTG) bond truncation; all cuts have the same retained rank.

## Fresh-pair transfer

Relative squared pair-response error at rank 64, averaged across independent pair seeds 23 and 29 (32,768 pairs per condition and seed):

| Fit metric | Uniform length 128 | FineWeb length 128 | Rare tokens | Unseen tokens | Uniform QK2 constant |
|---|---:|---:|---:|---:|---:|
| Uniform only | .160 | .120 | .168 | .199 | .228 |
| FineWeb only | .199 | .097 | .215 | .242 | .270 |
| Half | .161 | **.095** | .171 | .203 | .230 |
| Thirds | **.158** | .102 | **.167** | **.198** | **.225** |

The half mixture largely preserves the uniform cut's rare/unseen coverage and improves FineWeb pair error. The thirds mixture buys only a very small advantage on the intervention it explicitly saw, and gives up some natural FineWeb fidelity. These small pair differences should not be interpreted as discoveries of distinct mechanisms.

## Whole-sequence head outputs

For each of held-out FineWeb, WikiText-103, and TinyStories, the test samples 256 documents per seed at seeds 23 and 29, lengths 4 and 128. This is 3,072 tested sequences across the three corpora and lengths. Each causal pair passes through the original OV projection, then contributions are summed at every query position. The identity-bond path matches a direct PyTorch checkpoint calculation at length 128 to max absolute error `1.42e-14`. The table below reports the mean of the two seed-level relative squared **length-128 head-output** errors.

| Rank-64 fit | FineWeb | WikiText-103 | TinyStories |
|---|---:|---:|---:|
| Uniform only | .0752 | .0703 | .0818 |
| FineWeb only | .0860 | .0940 | .0949 |
| Half | **.0658** | .0635 | **.0691** |
| Thirds | .0667 | **.0627** | .0700 |

The half mixture also improves the corresponding rank-32 natural outputs over either pure measure on all three corpora (FineWeb .184 versus .211/.219; WikiText .174 versus .192/.218; TinyStories .213 versus .237/.246). This is a real gain for this *head-output fidelity* measure, despite modest pair-level differences. Sequence aggregation is therefore an essential evaluation, not an interchangeable proxy for pair error.

The same documents were evaluated under two internal QK2 interventions: replace all QK2 scores by one, and roll the QK2 **key-score columns** by one before reapplying the causal mask. Both edit only this head's second attention-score branch. Rank-64 sequence-head errors are:

| Fit | FineWeb constant / shift | WikiText constant / shift | TinyStories constant / shift |
|---|---:|---:|---:|
| Uniform only | .1484 / .1449 | .1428 / .1613 | .1479 / .1788 |
| FineWeb only | .1354 / .1321 | .1539 / .1562 | .1316 / **.1366** |
| Half | **.1265 / .1262** | **.1280 / .1435** | **.1217** / .1462 |
| Thirds | .1303 / .1304 | .1287 / .1462 | .1266 / .1542 |

The explicit **pair-level** constant-QK2 training term does not improve **sequence-level** constant-QK2 fidelity over the half mixture on any of these corpora; it also worsens the held-out key-shift response. Thus the tested intervention prior does not supply the causal gain we sought. The key shift is an artificial internal edit, not a natural language perturbation. These assays do not establish behavioral preservation after propagating the changed head through the rest of the network.

## What this resolves and what remains

The metric is a substantive modeling choice. A corpus-only cut gives up rare and unseen-token coverage. A fixed uniform–corpus mixture improves natural head outputs on three held-out corpora at matched rank, while retaining much of the uniform cut's coverage. Adding a pair-level branch intervention is insufficient to ensure sequence-level intervention fidelity. This supports moving the causal objective to the *aggregated sequence response*, and eventually to downstream logits/KL, rather than just increasing its weight in the pair metric.

This experiment does not answer which bond directions are human-legible or whether a sparse factorization is stable across equivalent QK factorizations. A [sequence-objective follow-up](SEQUENCE_OBJECTIVE_RESULTS.md) now directly tests the predicted next step: optimizing the summed head output improves natural held-out head fidelity, and adding a *sequence-level* QK2-constant term improves both that intervention and a held-out key shift. The next factorial step should hold the metric and rank fixed while comparing: sparse weights in a declared basis, sparse effective token interactions, structured/group sparsity, sparse activation across a broad token distribution, and intervention-specific support. Each needs a matched fidelity budget and independent semantic/causal validation. A weight-only baseline must also specify a position/input metric; a uniform token law is **data-independent in token frequency** but is still a declared input measure, not a canonical decomposition of the function alone.

Reproduce with `flagship_mixed_metric.py`, `flagship_transfer.py`, `flagship_ood_corpora.py`, and `flagship_sequence_transfer.py --n-docs 256` (the last was run with `taskset -c 2` and single-thread BLAS on this host). Detailed JSON and retained bond matrices are in `full_width_results/head0_l128_mixed_metric.json`, `head0_l128_uniform_corpus_*.npz`, `head0_context_transfer.json`, `head0_ood_corpora.json`, and `head0_sequence_transfer.json`. External model and corpus files are referenced by path, not copied into this repository.
