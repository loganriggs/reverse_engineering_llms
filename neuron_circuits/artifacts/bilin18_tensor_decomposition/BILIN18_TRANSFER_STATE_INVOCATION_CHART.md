# Invocation-specific read-function charts

This follow-up keeps the same four locked unseen contexts, three target heads, and 128-coordinate budget. The previous readable charts reused one suffix-Fisher residual basis across calls. Here each declared read invocation receives its own candidate-independent residual basis: 4 current sites × rank 32, 8 current/previous sites × rank 16, 16 recent sites × rank 8, or 32 recent sites × rank 4. This tests whether the unexplained joint-field advantage is merely call-specific variable/transform binding.

## Locked hypotheses

- **I1_invocation_specific_current_beats_shared: not supported.**
- **I2_invocation_specific_reads_approach_joint: supported.**
- **I3_invocation_specific_current_beats_random: supported.**
- **I4_current_invocation_subspaces_are_stable: not supported.**
- **I5_joint_retains_unexplained_structure: supported.**
- **I6_noncurrent_reads_are_necessary_somewhere: supported.**
- **attention_jvp_forward_invariant: supported.**

## Causal results

| Target | Boundary | Joint | Shared current | Specific current | Specific current+previous | Specific recent4 | Specific recent8 | Random current |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L0H7 | 6 | 0.883 | 0.551 | 0.614 | 0.671 | 0.691 | 0.687 | 0.030 |
| L0H7 | 10 | 0.810 | 0.592 | 0.749 | 0.726 | 0.669 | 0.597 | 0.032 |
| L0H7 | 14 | 0.724 | 0.561 | 0.725 | 0.621 | 0.564 | 0.497 | 0.060 |
| L0H7 | 17 | 0.690 | 0.531 | 0.692 | 0.590 | 0.528 | 0.464 | 0.033 |
| L4H2 | 6 | 0.817 | 0.433 | 0.507 | 0.534 | 0.530 | 0.511 | 0.015 |
| L4H2 | 10 | 0.749 | 0.507 | 0.648 | 0.568 | 0.519 | 0.445 | 0.033 |
| L4H2 | 14 | 0.618 | 0.434 | 0.602 | 0.469 | 0.408 | 0.349 | 0.038 |
| L4H2 | 17 | 0.575 | 0.389 | 0.572 | 0.443 | 0.378 | 0.316 | 0.051 |
| L5H1 | 6 | 0.822 | 0.380 | 0.450 | 0.487 | 0.520 | 0.541 | 0.015 |
| L5H1 | 10 | 0.749 | 0.504 | 0.667 | 0.597 | 0.526 | 0.466 | 0.037 |
| L5H1 | 14 | 0.676 | 0.499 | 0.674 | 0.540 | 0.463 | 0.397 | 0.049 |
| L5H1 | 17 | 0.643 | 0.469 | 0.647 | 0.516 | 0.440 | 0.368 | 0.018 |

## Stability

| Boundary | Median current-site rank-32 split overlap |
|---:|---:|
| 6 | 0.612 |
| 10 | 0.583 |
| 14 | 0.571 |
| 17 | 0.562 |

## Guardrails

Invocation-specific bases have the same number of runtime coefficients as the shared and joint charts, but higher declaration cost than a reused residual transform. They remain output-metric and context conditioned. A gain supports typed call-specific binding, not human-readable residual variables. A remaining joint advantage can arise from nonseparable combinations spread over nonlocal positions, not necessarily a new semantic concept.
