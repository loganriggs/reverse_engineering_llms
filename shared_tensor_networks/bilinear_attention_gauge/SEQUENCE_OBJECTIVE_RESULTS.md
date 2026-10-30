# Sequence-output optimization after a canonical bond cut

The [fixed-metric mixture](MIXED_METRIC_RESULTS.md) improved first-layer head-output fidelity, but a pair-level QK2-constant term did not improve sequence-level interventions. This follow-up tests whether the mismatch is the *aggregation target*: each causal pair contributes to a sum before the head passes through OV. All candidates retain rank 64 of the original width-128 QK1 bond of head 0 in layer 0 of the 546M bilinear-attention model.

This is a **two-stage reduction**. A fixed half uniform-token/half FineWeb weighted-trace-gauge (WTG) cut supplies a gauge-aware starting point. Direct rank-constrained optimization then changes the cut while preserving rank. The direct stage is not itself a canonical form, and none of these cuts identifies a unique mechanism across QK branch swaps or cross-head redistribution.

For a token sequence, write the original head response after OV as

\[
Y_i(X)=\sum_{j\le i}b_{ij}\,(u_i^{\mathsf T}Xv_j)c_jW_O,
\]

where \(X=I\) is the original width-128 QK1 bond and \(b_{ij}\) is the QK2 score. The direct sequence objective minimizes relative squared \(\|Y(X)-Y(I)\|_F^2\) with \(X=AB^{\mathsf T}\), rank at most 64. It computes the same error exactly in the 128-dimensional value coordinates using \(W_OW_O^{\mathsf T}\), avoiding a 1,152-dimensional output tensor in the training loop. A numerical check against the independent sequence implementation gives maximum absolute difference below `1.3e-11` for both natural and constant-QK2 responses; the quadratic-form error agrees to relative `1.5e-16`. The identity-bond head forward matches direct checkpoint PyTorch at length 128 to max absolute `1.42e-14`.

We fit two variants on 256 FineWeb documents from indices 0–14,999, with 128 disjoint documents from 15,000–17,499 for checkpoint selection:

- **Sequence natural:** only the natural QK2 sequence-output error.
- **Sequence + intervention:** 75% natural sequence error and 25% error after replacing QK2 scores by one.

Each uses the same starting cut, rank, 600 Adam steps, batch size 32, learning rate .001, and seed 41. The saved checkpoint has the lowest *declared validation mixture objective*, including the initial cut. The natural variant selects step 150 and the intervention variant step 200.

An **extra-optimization control** starts from that same WTG cut and directly optimizes its original 50/50 uniform-token/FineWeb *pair-response* metric. It uses 65,536 sampled training pairs per law, 65,536 validation pairs per law, 600 steps, batch size 8,192 per law, learning rate .0001. Its independent validation objective improves from .13373 to .13059, selecting step 100. A larger learning rate was unstable on validation and is not used for the comparison. These are matched-rank, same-start controls, but data volume and objective geometry differ; they do not isolate every optimization confound.

## Disjoint test

The final FineWeb test uses documents 17,500–19,999, which are disjoint from both fit and checkpoint-selection documents. WikiText-103 and TinyStories are separate corpora. Each corpus uses 256 length-128 sequences at each of seeds 23 and 29 (512 per corpus). All four cuts see *identical* test sequences. Values below are pooled relative squared head-output error across both seeds; lower is better.

| Rank-64 cut | FineWeb natural | WikiText natural | TinyStories natural |
|---|---:|---:|---:|
| Half-mixture WTG | .0616 | .0634 | .0691 |
| Pair-refit control | .0529 | .0565 | .0589 |
| Sequence natural | **.0414** | .0488 | **.0408** |
| Sequence + intervention | .0426 | **.0478** | .0419 |

The sequence-targeted cuts improve natural head-output fidelity over both the initial WTG cut and the optimized pair-metric control on all three corpora. The direct pair refit also improves natural output, so some benefit is attributable to further optimization; the remaining gap is consistent with sequence aggregation being the better target for this assay.

We also score two internal edits to the QK2 branch: replace scores by one (**constant**) and roll QK2 key-score columns by one before reapplying the causal mask (**key shift**). Only the constant edit is included in the intervention fit; key shift is held out. Both are artificial circuit edits rather than natural text perturbations.

| Rank-64 cut | FineWeb constant / shift | WikiText constant / shift | TinyStories constant / shift |
|---|---:|---:|---:|
| Half-mixture WTG | .1173 / .1072 | .1280 / .1435 | .1217 / .1462 |
| Pair-refit control | .1199 / .1050 | .1309 / .1423 | .1260 / .1464 |
| Sequence natural | .1191 / .1049 | .1326 / .1442 | .1186 / .1398 |
| Sequence + intervention | **.1098 / .1001** | **.1224 / .1370** | **.1075 / .1312** |

Unlike the previous *pair-level* intervention term, the *sequence-level* intervention term improves the constant-QK2 response over all three alternatives on every corpus. It also improves the held-out key-shift response on every corpus. The natural-versus-intervention tradeoff is small and corpus-dependent: on FineWeb and TinyStories the intervention variant gives up about .001–.002 natural error relative to the natural-only sequence fit.

This is evidence about **head-output and internal-intervention fidelity**, not full-model behavior. No edited head is propagated through layers 1–17, and no logits, KL divergence, cross-entropy, accuracy, semantic labels, or human-legibility measure is evaluated. The study covers one head and one rank. The next high-value test is downstream KL/CE under the saved cuts, followed by a matched-fidelity sparsity sweep over WTG coordinates, effective token interactions, grouped interactions, and intervention-specific support. Those priors need independent rule and causal validation; sparse weights alone are gauge-dependent.

Reproduce using `flagship_sequence_fit.py`, `flagship_pair_refit.py`, and `flagship_sequence_fit_eval.py`. The exact run arguments, split ranges, validation traces, test records, and bond matrices are in `full_width_results/head0_l128_sequence_fit_rank64_cw*.json/.npz`, `head0_l128_pair_refit_rank64.json/.npz`, and `head0_l128_sequence_fit_eval.json`. The experiments used CPU core 2 and single-thread BLAS; the external model and corpora remain referenced rather than copied into this repository.
