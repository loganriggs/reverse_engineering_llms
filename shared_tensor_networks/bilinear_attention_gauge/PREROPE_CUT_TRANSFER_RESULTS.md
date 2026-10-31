# Moving an existing QK1 cut before RoPE does not improve a new relative gap

The [position-gauge diagnostic](FLAGSHIP_ROPE_GAUGE_TRANSPORT_RESULTS.md) showed that scalar components of a fixed post-RoPE rank-96 cut change sharply under a global position translation. A natural repair is to move the cut before RoPE, where it defines one position-independent matrix. This experiment tests that **relocation alone**, keeping the same rank, reference-position function, two existing reduction priors, and held-out token panel.

For row rotary matrix \(R_p\), a post-RoPE cut \(X\) at query position 8 corresponds to the pre-RoPE query cut \(Y=R_8 X R_8^{-1}\). At another query position \(p\), evaluate the same \(Y\) through the equivalent post-RoPE matrix \(X_p=R_p^{-1}Y R_p\). Matrix inverses retain exactness with the checkpoint's rounded rotary tables. This is an executable alternative intervention, **not merely a relabeling** of the original fixed \(X\) outside the reference position. Both cuts have rank 96 and identical pair responses at the reference positions (key 7, query 8), by construction.

We compare both interventions against the **uncut, exact first-layer head 0** on the existing 546M checkpoint. The response includes both QK branches and OV, with exact output norm represented in 128 value coordinates. Fit and held-out panels each have 128 uniformly drawn and 128 FineWeb-frequency query IDs, and either 8,192 uniform or 8,192 corpus key draws. Query IDs are disjoint between panels; the corpus key draws come from disjoint FineWeb document ranges. This is sampled pair fidelity, not full-model next-token KL. The two \(X\) cuts were originally fitted under uniform versus corpus weighting for a length-128 objective, **not refitted** for this pre-RoPE comparison.

Held-out query/key panel, relative squared complete-head response error against the uncut head:

| Key law | Cut fitted under | Reference 7/8: either | Translated 31/32: fixed post / relocated pre | New gap 32/64: fixed post / relocated pre |
|---|---|---:|---:|---:|
| Uniform | Uniform | .00851 | **.00586** / .00851 | **.01218** / .07347 |
| Uniform | Corpus | .01066 | **.00703** / .01065 | **.01528** / .09285 |
| Corpus | Uniform | .01649 | .01648 / .01650 | **.04981** / .17879 |
| Corpus | Corpus | .00791 | **.00662** / .00790 | **.03406** / .16308 |

The small 16-query/512-key pilot shows the same strong direction on the new-gap test, but the larger panel above is the main result. The pre-RoPE relocation enforces a position-covariant form and gives the reference error under a same-gap translation, as intended. Yet it is **substantially less faithful on the unseen relative gap**: for example .07347 versus .01218 under uniform keys and the uniform-fitted cut. The fixed post-RoPE cut also improves or stays roughly level under the same-gap translation, so its component instability does not imply poor total-function fidelity.

This rejects the simple repair “take the existing good cut and move it before RoPE.” It does **not** reject learning a pre-RoPE cut from scratch across several relative offsets. The cuts were selected by a different sequence objective; their favorable post-RoPE behavior at new gaps may reflect that objective, and a reference-matched pre-RoPE cut need not inherit it. The [matched multi-offset follow-up](MATCHED_PREROPE_FIT_RESULTS.md) finds that separately fitting the pre-RoPE geometry reverses the broad comparison: it improves held-out pair fidelity and full-model KL, while retaining some specific gap failures. A position-covariant component definition is valuable only if the associated edited function passes these fidelity gates.

Reproduce with `flagship_prerope_cut_transfer.py --n-queries 128 --n-keys 8192`. Per-panel score and response errors, exact reference equality, and all token IDs are in `full_width_results/head0_prerope_cut_transfer_q128_k8192.json`; the small pilot is in `full_width_results/head0_prerope_cut_transfer_q16_k512.json`.
