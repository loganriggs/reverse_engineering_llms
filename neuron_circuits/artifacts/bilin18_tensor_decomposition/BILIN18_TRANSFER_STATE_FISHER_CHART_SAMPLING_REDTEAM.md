# Fisher-chart sampling red team

The base 8-top/8-tail and independent 16-top/16-tail runs use different seeds and chart estimates. This post-run comparison separates stability of literal rank-64 axes from stability of their held-out causal performance.

## Locked hypotheses

- **R1_rank64_behavior_is_sampling_stable: supported.**
- **R2_rank128_behavior_is_sampling_stable: not supported.**
- **R3_more_samples_stabilize_literal_axes: not supported.**
- **R4_behavior_is_more_stable_than_coordinates: supported.**
- **R5_highsample_chart_still_beats_static: supported.**

## Summary

| Rank | Fidelity Spearman | Median / max absolute difference |
|---:|---:|---:|
| 64 | 0.958 | 0.017 / 0.035 |
| 128 | 0.972 | 0.040 / 0.119 |

Mean higher-sample literal rank-64 overlap: `0.496`.

| Boundary | Base overlap | Higher-sample overlap |
|---:|---:|---:|
| 6 | 0.576 | 0.603 |
| 10 | 0.550 | 0.473 |
| 14 | 0.567 | 0.442 |
| 17 | 0.594 | 0.468 |

## Guardrails

Stable scalar fidelity does not mean the two charts predict identical logit-effect directions candidate by candidate; only aggregate fidelity cells are compared here. Conversely, low canonical overlap can be caused by broad or near-degenerate eigenspectra, so it does not by itself invalidate the behaviorally defined subspace family. Rank-128 maximum-difference failure would keep fine-grained high-rank claims provisional.
