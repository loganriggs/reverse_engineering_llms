# Invariance red team of the downstream fingerprint

Absolute singleton-response norms are replaced by within-site z-scores, ranks, or zeros (cosine-only). Causal labels, folds, masks, and evaluation sites are unchanged.

## Locked hypotheses

- **I0_relative_encoding_validates: not supported.**
- **I1_absolute_scale_caused_ood_failure: not supported.**
- **I2_singleton_pair_fingerprint_still_insufficient: supported.**

| Encoding | FineWeb median/q10 | WikiText median/q10 | WikiText minimum |
|---|---:|---:|---:|
| relative_norm | 0.988/0.767 | 0.980/0.413 | -2.585 |
| rank_norm | 0.987/0.767 | 0.980/0.413 | -2.585 |
| cosine_only | 0.990/0.697 | 0.981/0.413 | -2.686 |

The raw fingerprint WikiText 10th percentile was -0.870. Failure after invariant encodings is stronger evidence against singleton norms/cosines than the original corpus-shifted result.
