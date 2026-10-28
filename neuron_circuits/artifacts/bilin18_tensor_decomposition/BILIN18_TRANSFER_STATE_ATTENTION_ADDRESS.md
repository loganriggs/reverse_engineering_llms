# Attention-derived dynamic read addresses

Native bilinear QK patterns are converted to nonnegative row-normalized routing weights. `direct` sums query-to-source weights across the suffix. `rollout` recursively propagates demand backward through every suffix attention layer with residual carry. Each method selects eight sites, then uses the same candidate-independent local rank-16 Fisher residual transforms as the dynamic selector. Candidate replacement tangents remain held out.

## Locked hypotheses

- **A1_attention_rollout_recovers_fisher_sites: not supported.**
- **A2_attention_addresses_preserve_causal_effect: supported.**
- **A3_attention_addresses_improve_fixed_reads_early: not supported.**
- **A4_rollout_beats_direct_attention: not supported.**
- **A5_attention_addresses_do_not_fully_explain_joint: supported.**
- **attention_jvp_forward_invariant: supported.**

## Address recovery

| Boundary | Direct top-8 Jaccard | Rollout top-8 Jaccard |
|---:|---:|---:|
| 6 | 0.394 | 0.000 |
| 10 | 0.299 | 0.033 |
| 14 | 0.238 | 0.455 |
| 17 | 0.333 | 1.000 |

## Causal results

| Target | Boundary | Joint | Fixed current+previous | Fisher top8 | Direct attention | Attention rollout |
|---|---:|---:|---:|---:|---:|---:|
| L0H7 | 6 | 0.883 | 0.671 | 0.698 | 0.235 | -0.000 |
| L0H7 | 10 | 0.810 | 0.726 | 0.709 | 0.255 | 0.003 |
| L0H7 | 14 | 0.724 | 0.621 | 0.609 | 0.101 | 0.637 |
| L0H7 | 17 | 0.690 | 0.590 | 0.584 | 0.000 | 0.594 |
| L4H2 | 6 | 0.817 | 0.534 | 0.587 | 0.221 | 0.011 |
| L4H2 | 10 | 0.749 | 0.568 | 0.611 | 0.196 | 0.008 |
| L4H2 | 14 | 0.618 | 0.469 | 0.484 | 0.095 | 0.483 |
| L4H2 | 17 | 0.575 | 0.443 | 0.451 | 0.000 | 0.452 |
| L5H1 | 6 | 0.822 | 0.487 | 0.542 | 0.263 | 0.011 |
| L5H1 | 10 | 0.749 | 0.597 | 0.613 | 0.265 | 0.022 |
| L5H1 | 14 | 0.676 | 0.540 | 0.557 | 0.112 | 0.545 |
| L5H1 | 17 | 0.643 | 0.516 | 0.529 | 0.000 | 0.519 |

## Guardrails

This is a deliberately simple unsigned attention rollout. It discards head identity, sign, OV geometry, MLP Jacobians, and cancellations. Failure would reject this short router approximation, not the claim that exact tensor routing predicts demand. Success would identify an address function, not the semantic content of its payload. Local residual bases still use the Fisher metric, so the test isolates address prediction rather than a complete gradient-free interpreter.
