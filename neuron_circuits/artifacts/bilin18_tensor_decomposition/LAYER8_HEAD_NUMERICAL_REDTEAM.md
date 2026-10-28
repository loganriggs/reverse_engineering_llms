# Numerical red team of third-order head localization

The original exact-sum gate used error relative to a cancellation-small third-order residual. This audit scales error by the full exact response and independently defines projection shares against the recomposed sum of head terms.

## Locked hypotheses

- **N0_error_is_cancellation_relative: supported.**
- **N1_head_identity_is_numerically_stable: supported.**
- **N2_localization_strength_is_stable: supported.**
- **N3_original_localization_was_numerical_artifact: not supported.**

Maximum error relative to exact response: `2.977e-05`. All failure top heads unchanged: `True`. Median recomposed-reference top share: `0.923`. Median effective support: `1.17`.

If localization survives while residual-relative identity misses, the discrepancy is float32 cancellation around a tiny residual rather than a different head decomposition.
