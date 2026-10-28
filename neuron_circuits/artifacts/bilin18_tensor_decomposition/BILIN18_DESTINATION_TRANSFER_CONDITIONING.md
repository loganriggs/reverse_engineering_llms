# Conditioning of the destination transfer recurrence

The exact Jacobian has a short compositional program even when its expanded paths are unusable. At each downstream layer the incoming candidate bundle is split into the residual/MLP branch and the dynamic-attention branch, then recombined. `Cancellation condition = (||identity|| + ||attention||) / ||sum||`; one means aligned/non-cancelling and larger values mean the additive path basis is ill-conditioned.

## Locked hypotheses

- **C1_early_transfer_contains_strong_local_cancellations: not supported.**
- **C2_transfer_conditioning_profile_is_split_stable: supported.**
- **C3_attention_is_not_a_small_correction: not supported.**
- **C4_earlier_targets_accumulate_more_conditioning: supported.**
- **C5_same_layers_are_cancellation_bottlenecks: supported.**
- **attention_jvp_forward_invariant: supported.**

## Summary

| Target / half | Strong-cancel layers | Median attention/identity | Sum log condition | Top-3 layers |
|---|---:|---:|---:|---|
| L0H0 / a | 2 | 0.262 | 3.447 | [1, 2, 3] |
| L0H0 / b | 2 | 0.239 | 3.278 | [1, 2, 3] |
| L0H0 split stability / top-3 overlap | 0.995 | — | — | 3 |
| L2H2 / a | 1 | 0.232 | 2.646 | [3, 4, 5] |
| L2H2 / b | 1 | 0.235 | 2.651 | [3, 4, 5] |
| L2H2 split stability / top-3 overlap | 0.996 | — | — | 3 |
| L5H3 / a | 0 | 0.148 | 1.621 | [6, 7, 9] |
| L5H3 / b | 0 | 0.141 | 1.612 | [7, 6, 9] |
| L5H3 split stability / top-3 overlap | 0.979 | — | — | 3 |

## Guardrails

These norms use the checkpoint's residual coordinates and aggregate candidates, tokens, and examples. They diagnose numerical/mechanistic conditioning of one replacement family, not semantic importance. A well-conditioned local split can still participate in a badly conditioned long product; conversely, large local cancellation can be harmless when represented as one composed transfer operator rather than expanded paths.
