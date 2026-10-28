## H742 — Compact operation edits generalize across names, but their directions depend on format

We fitted separate uncentered SVD bases for q2 and k2 binding-to-copying changes using name-pair indices 0–3, then evaluated reciprocal edits on indices 4–7. Fits use either both formats pooled, narrative only, or list only. No coefficient labels enter SVD, but the pairing explicitly uses operation labels; this is a supervised paired-edit diagnostic on an already-inspected panel, not unsupervised discovery or fresh confirmation.

To avoid an artificial coordinate-frame explanation, we first invert the actual bf16-cached rotary transform, including its slight departure from exact orthogonality. We project normalized pre-rotary donor-minus-receiver changes, then restore rotary position at each recipient. Query and key bases are fitted separately: rank four means four directions in each state, not one four-dimensional joint feature. Native factor 1 stays fixed; the target is the exact factor-2 swap from H741.

Both raw projected edits and norm-preserving edits were evaluated, with three random orthogonal subspace controls at each rank and a full-rank identity. Below are norm-preserving held-out-name results. Relative RMS error is error in the coefficient divided by the RMS intended coefficient change, and sign transfer is counted only where native and target signs differ.

| Basis fit / evaluation format | Rank per state | Sign transfer | Relative coefficient-edit RMS error |
|---|---:|---:|---:|
| Pooled / narrative | 1 | 13/16 | 18.5% |
| Pooled / narrative | 2 | 13/16 | 16.2% |
| Pooled / narrative | 4 | 16/16 | 4.1% |
| Pooled / list | 1 | 4/14 | 73.6% |
| Pooled / list | 2 | 14/14 | 12.9% |
| Pooled / list | 4 | 14/14 | 8.6% |
| Narrative / narrative | 2 | 16/16 | 6.7% |
| List / list | 1 | 14/14 | 13.9% |
| Narrative / list | 2 | 7/14 | 62.0% |
| List / narrative | 2 | 0/16 | 85.1% |

Random rank-four controls transfer 0/48 narrative signs and 2/42 list signs across three seeds, with approximately 99% and 96% relative edit error. The pooled rank-four bases capture about 93%/97% of query/key edit energy on narrative contexts and 83%/92% on lists. Higher ranks and both normalization variants are retained in the saved results; there is no monotonicity or convergence claim from this small fit.

The promising finding is low-dimensional paired-edit structure that transfers across held-out names. The restriction is equally important: within-format compactness does not imply the same directions implement the operation across formats. Poor cross-format transfer persists after removing the rotary coordinate-frame difference. Relative source/query distance and contextual states still differ across formats; a pooled basis may simply contain multiple format-specific directions.

Audits: two tests passed for the rotary formula/inverse, projected-edit behavior, norm preservation and full-rank recovery. Runtime rotary round-trip relative error was at most 1.09e-16, basis orthogonality error 1.78e-15 and full-rank coefficient replay error 1.11e-16. All source/data hashes stayed unchanged and coefficients were finite. Zero language-model forwards, new natural documents or original reserved-test evaluations were used.

Next: propagate these projected coefficients through the actual head and downstream model on the held-out-name contexts, retaining the whole rank/control grid. Compare against exact factor-2 swaps and native, zero, and identity controls. Then freeze any candidate before testing genuinely new wording and contexts. This remains a compact operation-edit subspace, not a verified manifold or a universal semantic feature. Sentiment's diagnostic remains outstanding.
