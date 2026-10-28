# Scale and support audit of blind-replication failures

This audit separates response reconstruction on fixed supports, discontinuous sparse-program selection, and finite nonlinear propagation for the twelve blind failures and same-site error-matched passing controls.

The first implementation used the route-masked tangent for the own-tangent
check even though finite interventions execute the full suffix. Passing controls
exposed the mismatch. The reported run uses the full analytic suffix Jacobian
for causal comparisons and keeps the route-masked Jacobian only for the declared
pairwise reconstruction. Thresholds, examples, controls, and hypotheses were
not changed.

## Locked hypotheses

- **S0_fixed_support_reconstruction_is_accurate: supported.**
- **S1_selected_programs_disagree_in_the_linear_limit: supported.**
- **S2_finite_nonlinearity_causes_the_failure: not supported.**
- **S3_each_program_is_locally_predicted_by_its_tangent: supported.**
- **S4_failures_are_specific_to_unstable_program_selection: supported.**

## Fixed-support and support-selection results

| Group | Fixed exact support fidelity (median/min) | Fixed reconstructed support fidelity (median/min) | Exact vs reconstructed selected programs, infinitesimal (median/q10) | Support recall (median) |
|---|---:|---:|---:|---:|
| failure | 1.000/0.943 | 1.000/0.950 | -0.030/-2.368 | 0.250 |
| control | 1.000/0.996 | 1.000/0.996 | 0.999/0.990 | 0.938 |

## Scale sweep

| Group | Edit norm | Between-program fidelity median/q10 | Exact support: own-tangent fidelity median/q10 | Reconstructed support: own-tangent fidelity median/q10 |
|---|---:|---:|---:|---:|
| failure | 0.0010 | -0.028/-2.366 | 1.000/1.000 | 1.000/1.000 |
| failure | 0.0025 | -0.028/-2.367 | 1.000/1.000 | 1.000/1.000 |
| failure | 0.0050 | -0.027/-2.370 | 1.000/0.999 | 1.000/0.999 |
| failure | 0.0100 | -0.023/-2.374 | 0.999/0.998 | 0.999/0.997 |
| failure | 0.0200 | -0.017/-2.380 | 0.996/0.991 | 0.997/0.990 |
| failure | 0.0500 | -0.017/-2.397 | 0.980/0.935 | 0.984/0.927 |
| control | 0.0010 | 0.999/0.990 | 1.000/1.000 | 1.000/1.000 |
| control | 0.0025 | 0.999/0.990 | 1.000/1.000 | 1.000/1.000 |
| control | 0.0050 | 0.999/0.990 | 1.000/0.999 | 1.000/0.999 |
| control | 0.0100 | 0.999/0.990 | 0.999/0.994 | 0.999/0.995 |
| control | 0.0200 | 0.999/0.989 | 0.997/0.977 | 0.997/0.978 |
| control | 0.0500 | 0.999/0.987 | 0.982/0.810 | 0.982/0.816 |

A failure here is not automatically a model-computation failure: independently maximizing a non-convex/discrete sparse-program objective can make the explanatory program non-identifiable even when the response tensor is accurately reconstructed.
