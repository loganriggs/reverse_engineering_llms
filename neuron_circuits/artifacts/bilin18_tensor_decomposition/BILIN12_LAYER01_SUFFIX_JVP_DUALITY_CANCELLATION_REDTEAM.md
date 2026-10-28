# Cancellation-robust forward/reverse AD duality red-team

Raw relative error is unstable when the JVP and random output cotangent are nearly orthogonal. This audit also divides the bilinear-form discrepancy by its Cauchy bound, which remains well scaled at a zero dot product.

## Locked hypotheses

- **C0_cauchy_normalized_duality_holds: supported.**
- **C1_large_raw_relative_errors_are_cancellation_cases: supported.**
- **C2_typical_raw_relative_duality_is_precise: not supported.**
- **C3_duality_validation_is_uniform: supported.**

| Layer | Corpus | Maximum Cauchy-normalized error | Raw-relative q99 | Minimum dot alignment |
|---:|---|---:|---:|---:|
| 0 | fineweb | 1.48e-08 | 1.31e-05 | 1.02e-04 |
| 0 | tinystories | 1.84e-08 | 3.71e-05 | 9.56e-05 |
| 1 | fineweb | 2.52e-08 | 2.73e-04 | 1.08e-05 |
| 1 | tinystories | 1.04e-08 | 1.61e-04 | 2.79e-05 |

This does not turn a finite-radius failure into a pass. It determines whether the analytic tangent used for chart routing is the correct derivative of the implemented tensor network.
