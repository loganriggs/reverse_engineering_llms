# M45: token identity and additional retained-half prediction play different roles

This completes M44’s fixed token-conditional native comparison. All six M42 mappings remain frozen. Token means are fitted only to natural training documents0–31; no task fitting, rank tuning or amplitude selection.

## Compared predictors

For retained write X, original rank32 prediction is mean_Y+(X-mean_X)UB. The token-only predictor is mean_Y(token). The token-centered predictor is mean_Y(token)+(X-mean_X(token))UB. Unseen tokens fall back to global training means, making token-centered and original rank32 predictions identical at those positions. All predictors replace only the missing half; the native retained half and bias remain, and downstream computation is live.

The local covered-token error isolates prediction on token identities present in training. Output KL includes all positions and downstream effects; it is not a covered-position causal attribution. Token-only and token-centered predictions differ on unseen tokens too (mean fallback versus original rank32), so their full-output difference cannot be attributed solely to known-token context. Original versus token-centered differs only at covered positions.

## Results

KL is averaged over inputs and both complementary masks. The last column is token-centered / token-only local squared error on covered positions; below1 is improvement beyond the token table in this comparison.

| Cohort | Layer | Original rank32 KL | Token mean KL | Token-centered KL | Covered local error ratio |
|---|---|---:|---:|---:|---:|
| task train | MLP0 | 0.02568 | 0.02278 | 0.02142 | 0.8989 |
| task train | MLP1 | 0.05247 | 0.04470 | 0.04673 | 0.7132 |
| task train | MLP2 | 0.02380 | 0.02691 | 0.02402 | 0.9152 |
| task evaluation | MLP0 | 0.02643 | 0.02017 | 0.02218 | 0.9045 |
| task evaluation | MLP1 | 0.06308 | 0.04215 | 0.05682 | 0.7641 |
| task evaluation | MLP2 | 0.02788 | 0.02239 | 0.02456 | 0.9225 |
| natural evaluation | MLP0 | 0.02866 | 0.03330 | 0.02543 | 0.8801 |
| natural evaluation | MLP1 | 0.06474 | 0.06590 | 0.05389 | 0.8573 |
| natural evaluation | MLP2 | 0.03338 | 0.03911 | 0.03394 | 0.9435 |

Natural coverage is60.25%; task train58.71% and task evaluation57.03%. Counts weight token occurrences and include both dependent masks. Of1760training token types,1250are singletons. The task train/evaluation labels refer to the existing task split; neither task partition fitted these controls. All material is familiar development data, not new-task discovery.

On natural text, token centering lowers output KL versus original rank32 by11.3% atMLP0 and16.8% atMLP1, while increasing it1.7% atMLP2. The covered-position local correction improves over token means by12.0%,14.3%,5.7% respectively. MLP2’s token-centered local error is nevertheless worse than original rank32. On source-separated task evaluation, the token-only predictor has lower full-output KL than token-centered prediction for all three layers, despite worse local reconstruction.

These gains do not erase M42’s repair failure. MLP1 natural token-centered KL.05389 remains above raw half-deletion KL.03282; MLP2 .03394 likewise exceeds raw.02264. Improved conditional prediction is still not successful functional equivalence.

## Mechanistic interpretation and next discriminator

A pure token-mean description leaves predictable error that the frozen retained-half mapping can reduce, but this result does not identify the source of that improvement. It can reflect context-dependent shared variation, estimation noise in token means, corrective shrinkage, or other correlations. The original maps were trained without token residualization; no claim of an optimal conditional predictor is made. Token means add storage, so this is a diagnostic control rather than a matched-capacity architecture comparison.

The strongest supported update is that token identity matters differently by layer and cannot be ignored when interpreting half-sharing or coactivation. Within-token contextual sharing remains a hypothesis, not a discovered semantic unit. To distinguish actual pairing information from noisy-table correction, the next fixed control should shuffle retained contextual coordinates among occurrences of the same token while holding token means and the coordinate distribution fixed. Restrict and report the subset with multiple distinct contexts; singleton/unshufflable positions stay visible. Use native output consequences if local paired advantage survives. No new token threshold, map fitting or amplitude search.

## Audit

3,648native forwards/3,456intervention rows. All1,152original rank32 rows replay M42 exactly. Unknown-token local errors match between original and token-centered predictors. Full inventories, token coverage, source energies, native task outputs and summaries verified. Eighteen sampled NumPy predictions exactly match; output metrics differ by at most1.19e-8. Covered/uncovered error energies sum to full local error. No independent transformer implementation or semantic-unit validation.

[Protocol](validation/native_half_token_protocol.json) · [Transfer](validation/native_half_token_transfer.json) · [Audit](validation/native_half_token_transfer_audit.json) · [Coverage-stratified comparison](validation/native_half_token_comparison.json)
