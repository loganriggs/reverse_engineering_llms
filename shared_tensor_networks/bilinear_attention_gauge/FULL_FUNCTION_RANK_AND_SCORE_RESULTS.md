# Full-vocabulary function cuts expose the rank cost of flattening attention

The [nonlinear toy](XOR_COUNT_CANONICAL_RESULTS.md) gave an interpretable basis for a complete token-to-head function, while the [546M-model key-side bond test](FLAGSHIP_KEY_SPARSE_RESULTS.md) found no general legibility advantage. This follow-up asks what happens when the **complete first-layer head pair response** is canonicalized on the 546M model, rather than one physical QK1 bond. It compares two function objects and a structure-preserving control.

At fixed query position (i\) and key position (j\), the exact layer-0/head-0 contribution is

\[
T_{q,o,k}^{ij}=S_{q,k}^{ij}C_{k,o},\qquad
S_{q,k}^{ij}=\frac{\langle Q_1(q,i),K_1(k,j)\rangle
                    \langle Q_2(q,i),K_2(k,j)\rangle}{128^2},
\]

where (C_k\) is the **unchanged value/OV output**. The two candidate reductions are:

1. **Flat complete response:** unfold (T\) as (query, output) × key, then take its optimal weighted rank-(r\) SVD. This is the direct analogue of the toy key dictionary.
2. **Effective score:** take the optimal weighted rank-(r\) SVD of the scalar (S\), then multiply by the exact (C_k\). This quotient is invariant to exchanging QK1 and QK2, but does not preserve their individual branch-edit behavior.

For comparison, the previously sequence-fitted **rank-64 QK1 bond cut** replaces the QK1 inner product with (Q_1^{\mathsf T}XK_1\) and keeps QK2 and (C\) exact. These are **different function classes and resource counts**. A rank-64 bond can induce a much higher-rank complete token function because its remaining factors are reused. The comparison tests representation choice, not a matched-rank theorem.

All **50,304 key token IDs** are present in every fit and test. The complete head output is represented in 128 value coordinates using the square root of (O^{\mathsf T}O\), exactly preserving the squared norm in the original 1,152-dimensional output space. The numerical norm check is below (3.8\times10^{-10}\); the effective-score QK-branch swap check is exactly zero. QK RMS normalization and bf16-rounded RoPE tables are included. No labels enter any decomposition.

## Flat complete response: rank 64 is an inadequate starting point for sparsity

The flat fit uses 32 query anchors (16 uniformly sampled vocabulary IDs, 16 sampled from FineWeb training frequencies) at key/query positions 7/8. This anchor mixture is shared across key metrics, isolating *key weighting*; even its uniform-key condition is **not** fully data-free because half its query anchors came from corpus frequencies. The SVD is exact for this declared matrix via a 4,096 × 4,096 row Gram eigensolve. The uniform-key **provably optimal** training residual is **.820 at rank 16, .761 at rank 32, .683 at rank 64, .588 at rank 128, and .476 at rank 256**. The first randomized implementation's rank-64 residual was .695; extra power iterations approached .685, and the exact eigensolve establishes .683. The high error is a property of this unfolding on this domain, not an optimizer failure.

Key frequency weights from 15,000 FineWeb training documents trade uniform response fidelity for corpus-weighted fidelity. With 16 additional held-out-corpus query IDs, all keys, and a separately computed test-document key frequency law, rank-64 relative squared **head pair-response** errors are:

| Flat key-mode fit | Same relative offset 1: uniform / corpus metric | New relative offset 32: uniform / corpus metric |
|---|---:|---:|
| Uniform key weights | .795 / .601 | .911 / .975 |
| Half uniform, half training frequency | .808 / .491 | .922 / .898 |
| 10% uniform, 90% training frequency | .829 / .468 | .941 / .800 |
| Structured rank-64 QK1 bond control | **.198 / .115** | **.257 / .246** |

The bond cut was selected for a different, sequence-level objective, and still has substantial error on arbitrary full-vocabulary pair responses. Yet it is far better on this same pair metric while preserving the QK2/value factorization. Since the flat rank-64 function is already badly unfaithful, rotating it to make components sparse would optimize legibility on the wrong function. This is a stopping criterion for this flat low-rank branch, not a claim that no function-level canonical form exists.

## Effective score: preserving the value factor helps, but position transfer remains hard

The score fit uses 256 query anchors and all key IDs at positions 7/8. A **strict uniform-query panel** draws every fit query uniformly from the vocabulary; its uniform-key fit is corpus-free. A separate mixed panel uses 128 uniform and 128 FineWeb-frequency query anchors. For each panel, uniform, half-mixture, and 10% uniform/90% frequency key measures are compared on the *same* anchors. The pair-response metric weights a key's score error by \(\lVert C_k\rVert^2\); this is exactly the output-space squared error when (C_k\) remains fixed. FineWeb weighting uses only training-document token frequencies. The leading SVD is exact for each declared score table via its row Gram matrix.

On the **mixed** fit panel, the optimal rank-64 training residual under each method's own metric is **.202 uniform, .158 half-mixture, .113 mostly corpus**. These are different weighting measures and therefore not directly comparable as absolute goodness. Rank 128 reduces them to .091, .070, and .049. In a held-out-corpus query panel, rank-64 relative output errors under the **held-out corpus key metric** are:

| Score fit | Offset 1 | Offset 32 |
|---|---:|---:|
| Strict uniform queries + uniform keys | .240 | .731 |
| Strict uniform queries + mostly corpus keys | .192 | .681 |
| Mixed queries + uniform keys | .175 | .695 |
| Mixed queries + mostly corpus keys | .152 | .657 |
| Structured QK1 bond control on mixed query panel | **.086** | **.239** |

Thus frequency weighting improves same-offset transfer even with strictly uniform fit queries, while sampling some corpus queries improves it further. Neither fixes the unseen relative-position gap. Fitting six offsets (1, 2, 4, 8, 16, 32) at query position 64 improves the mixed-query, mostly-corpus rank-64 error from **.151 to .139 at offset 1**, **.609 to .504 at held-out offset 24**, **.657 to .558 at offset 32**, and **.745 to .662 at held-out offset 48**. At rank 128 these become .110, .430, .481, and .594. Multi-offset coverage helps but remains well behind the structured bond on larger offsets. The single-offset fit used query position 8; RoPE is approximately relative-position based, but its rounded tables and the different fit domains are retained rather than assumed identical.

## Actual adjacent token pairs, and why sample size alone is not enough

The fixed-position test above averages over *all* keys and a held-out key-frequency law, rather than actual query–key co-occurrences. As a check, 512 disjoint test documents per corpus supply an actual adjacent pair at positions 7/8, excluding training query-anchor IDs. The flat and score methods are evaluated on the **same pairs**. Relative pair-response MSE:

| Cut | FineWeb | WikiText-103 | TinyStories |
|---|---:|---:|---:|
| Flat uniform-key rank 64 | .497 | .806 | .285 |
| Flat mostly-corpus rank 64 | .448 | .647 | .058 |
| Effective score, strict uniform-query/key rank 64 | .149 | .372 | .037 |
| Effective score, mixed queries + mostly-corpus rank 64 | .138 | .298 | .017 |
| Effective score, mixed queries + six offsets, mostly-corpus rank 64 | .116 | .288 | .015 |
| Structured QK1 bond rank 64 | **.056** | **.121** | .042 |

These are **single-pair head outputs**, not sequence-summed head outputs or full-model logits. Pair energy is highly concentrated: effective sample sizes are **11.4, 20.1, and 3.4** despite 512 documents per corpus. The largest pair carries 19%, 14%, and 50% of total response energy. Paired document bootstrap intervals therefore remain wide, especially for TinyStories. For example, the mixed multi-offset score cut's TinyStories error .015 has a 95% percentile interval **.006–.068**, overlapping the bond cut's **.034–.070**. FineWeb and WikiText favor the bond more consistently. No universal ordering of score versus bond cuts is inferred from the TinyStories sample.

## What this changes about the canonicalization program

The rank difference has a concrete algebraic cause. Expanding the two QK products and value map yields terms proportional to

\[
  [u_r(q)\,\tilde u_s(q)\,O_{o,t}]
  [v_r(k)\,\tilde v_s(k)\,V_t(k)].
\]

A flat key dictionary must absorb combinations of two QK branch coordinates and value coordinates into its key factors. A QK1 bond cut compresses only one factor while reusing the other factors exactly. The effective-score quotient removes the value coordinate from the low-rank object and is therefore much more economical, but still loses position and query generality at these budgets. This explains why ordinary tensor rank of one unfolding is not a good measure of reusable circuit size. The exact measured spectra establish the actual 546M-head limitation on the declared domain; the algebra explains why it is plausible, without claiming a general rank lower bound for every attention model.

The next defensible direction is **structure-preserving, multi-position function canonicalization** with explicit QK multiplication and exact OV, then legibility priors only after a fidelity gate on held-out query tokens, positions, natural sequences, and full-model KL. The function quotient must also declare whether it is intended to predict QK branch interventions. Branch-swap-invariant score factors alone cannot identify which physical branch implements which part. A low-rank flat key basis at rank 64 or 128 should not be sparsified and presented as an explanation of this model.

Reproduce with `flagship_full_function_metric.py`, `flagship_full_function_pairs.py`, `analyze_full_function_pairs.py`, `flagship_effective_score_metric.py` (both `--query-panel mixed` and `uniform`), `flagship_effective_score_positions.py`, `flagship_effective_score_pairs.py`, and `analyze_effective_score_pairs.py`. Source, per-query/per-corpus records, exact basis arrays, and paired bootstrap summaries are stored in `full_width_results/head0_full_function_*` and `head0_effective_score_*`. The external model and corpora are referenced by path rather than copied into this repository.
