# Exact all-order attention-path recomposition

This audit was locked after the order-6 expansion exploded for early targets. It includes every possible downstream attention order, so its sum must equal the independently propagated full Jacobian. Four already-fixed rows from each prospective half are used; no examples were reselected.

## Locked hypotheses

- **R1_full_order_expansion_recomposes_exact_jvp: supported.**
- **R2_full_order_expansion_recomposes_fisher_response: supported.**
- **R3_low_order_explosion_is_resolved_by_cancellation: supported.**
- **R4_earlier_target_needs_higher_attention_order: supported.**
- **attention_jvp_forward_invariant: supported.**

## Full frontier

### L0H0

| Half | First ≥.90 | Final fidelity / geometry | Order-6 fidelity | State error |
|---|---:|---:|---:|---:|
| a | 12 | 1.000000 / 1.000000 | -282.211 | 8.11e-05 |
| b | 10 | 1.000000 / 1.000000 | -27.841 | 3.74e-05 |

### L2H2

| Half | First ≥.90 | Final fidelity / geometry | Order-6 fidelity | State error |
|---|---:|---:|---:|---:|
| a | 9 | 1.000000 / 1.000000 | -2.927 | 1.23e-05 |
| b | 8 | 1.000000 / 1.000000 | 0.563 | 7.12e-06 |

### L5H3

| Half | First ≥.90 | Final fidelity / geometry | Order-6 fidelity | State error |
|---|---:|---:|---:|---:|
| a | 5 | 1.000000 / 1.000000 | 0.991 | 3.16e-06 |
| b | 5 | 1.000000 / 1.000000 | 0.997 | 2.51e-06 |

## Guardrails

Exact recomposition validates the algebra, not a human-readable ontology. Large individual orders followed by cancellation mean that truncating by apparent path energy can be actively misleading. The first-passing order is a property of this local expansion and corpus panel; it is not the number of discrete reasoning steps taken by the model.
