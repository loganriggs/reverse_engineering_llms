# Supervised-anchor metric red team

This analysis was added after the locked signed-profile comparison. It asks whether the answer changes when head effects are treated as unsigned magnitude rather than promotion versus suppression. The Gram-matched vocabulary permutation is the critical semantic control.

| Objective | Profile | Split r | Closure enrichment / p | Direct-edge enrichment / p |
|---|---|---:|---:|---:|
| semantic | signed_cosine | 0.507 | +0.040 / 0.0817 | +0.033 / 0.1975 |
| semantic | signed_centered | 0.450 | +0.043 / 0.0654 | +0.037 / 0.1589 |
| semantic | absolute_cosine | 0.858 | +0.029 / 0.0001 | +0.034 / 0.0001 |
| semantic | absolute_centered | 0.538 | +0.115 / 0.0004 | +0.177 / 0.0002 |
| semantic | rms_cosine | 0.786 | +0.033 / 0.0001 | +0.035 / 0.0001 |
| semantic | rms_centered | 0.409 | +0.090 / 0.0012 | +0.151 / 0.0003 |
| gram_matched | signed_cosine | 0.182 | +0.019 / 0.2029 | +0.006 / 0.4112 |
| gram_matched | signed_centered | 0.183 | +0.018 / 0.2159 | +0.005 / 0.4283 |
| gram_matched | absolute_cosine | 0.718 | +0.018 / 0.0001 | +0.022 / 0.0001 |
| gram_matched | absolute_centered | 0.485 | +0.130 / 0.0002 | +0.206 / 0.0001 |
| gram_matched | rms_cosine | 0.746 | +0.025 / 0.0001 | +0.028 / 0.0001 |
| gram_matched | rms_centered | 0.337 | +0.111 / 0.0002 | +0.187 / 0.0001 |

## Decision

The magnitude result is real as a statement about generic causal salience: weight-motif members tend to matter on similar examples. It is not semantic validation, because the Gram-matched nonsense objectives reproduce the same effect and can be stronger after centering. The signed semantic effect is larger than its nonsense control, but it did not clear the locked significance or semantic-margin gates.

Therefore the 32 labels should annotate frozen components but should not be folded into cluster discovery or called ground-truth circuits.
