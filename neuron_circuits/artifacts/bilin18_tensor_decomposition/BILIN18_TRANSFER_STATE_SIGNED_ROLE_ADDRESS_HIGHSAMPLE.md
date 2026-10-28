# Signed suffix-role address programs

The exact probability-weighted score adjoints are propagated backward with selected local Jacobian terms enabled: residual carry, bilinear-MLP VJP, Q1/Q2, K1/K2, and current-value/OV. Each truncated recurrence prospectively ranks eight read sites. `address` tests its selected sites with exact local Fisher payload bases; `self` also requires the truncated program to generate its own local payload bases. Every causal chart has 8 sites × rank 16 = 128 runtime coordinates and sees no candidate operator tangents during construction.

## Locked hypotheses

- **C0_full_role_program_recomposes_exact_suffix: supported.**
- **C1_mlp_only_does_not_recover_early_addresses: supported.**
- **C2_transport_coalitions_approach_exact_address_effects: supported.**
- **C3_no_single_attention_role_is_early_sufficient: not supported.**
- **C4_self_contained_subset_program_is_useful: supported.**
- **C5_late_addressing_simplifies_to_mlp_carry: supported.**
- **C6_role_coalitions_have_nonadditive_value: not supported.**
- **attention_jvp_forward_invariant: supported.**

## Address recovery

| Program | b6 | b10 | b14 | b17 |
|---|---:|---:|---:|---:|
| carry | 0.394 | 0.333 | 0.333 | 1.000 |
| mlp | 0.394 | 0.333 | 0.333 | 1.000 |
| queries | 0.394 | 0.333 | 0.333 | 1.000 |
| keys | 0.527 | 0.600 | 0.778 | 1.000 |
| value | 0.889 | 0.778 | 0.527 | 1.000 |
| query_key | 0.455 | 0.600 | 0.778 | 1.000 |
| query_value | 0.778 | 0.778 | 0.527 | 1.000 |
| key_value | 1.000 | 1.000 | 1.000 | 1.000 |
| attention_no_mlp | 0.689 | 0.689 | 1.000 | 1.000 |
| full | 1.000 | 1.000 | 1.000 | 1.000 |

## Causal address fidelity

| Target | Boundary | Exact | MLP | Queries | Keys | Value | Q+K | Q+V | K+V | Attn no MLP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L0H7 | 6 | 0.680 | 0.607 | 0.607 | 0.600 | 0.688 | 0.614 | 0.673 | 0.683 | 0.646 |
| L0H7 | 10 | 0.714 | 0.678 | 0.678 | 0.675 | 0.715 | 0.676 | 0.715 | 0.713 | 0.711 |
| L0H7 | 14 | 0.628 | 0.627 | 0.627 | 0.628 | 0.628 | 0.628 | 0.628 | 0.628 | 0.628 |
| L0H7 | 17 | 0.603 | 0.603 | 0.603 | 0.603 | 0.603 | 0.603 | 0.603 | 0.603 | 0.603 |
| L4H2 | 6 | 0.573 | 0.487 | 0.487 | 0.496 | 0.571 | 0.495 | 0.572 | 0.567 | 0.567 |
| L4H2 | 10 | 0.600 | 0.572 | 0.572 | 0.568 | 0.607 | 0.570 | 0.607 | 0.600 | 0.593 |
| L4H2 | 14 | 0.485 | 0.481 | 0.481 | 0.485 | 0.484 | 0.488 | 0.484 | 0.485 | 0.485 |
| L4H2 | 17 | 0.435 | 0.435 | 0.435 | 0.435 | 0.435 | 0.435 | 0.435 | 0.435 | 0.435 |
| L5H1 | 6 | 0.536 | 0.491 | 0.491 | 0.461 | 0.544 | 0.453 | 0.546 | 0.539 | 0.513 |
| L5H1 | 10 | 0.610 | 0.594 | 0.594 | 0.588 | 0.607 | 0.593 | 0.607 | 0.611 | 0.606 |
| L5H1 | 14 | 0.551 | 0.549 | 0.549 | 0.551 | 0.550 | 0.551 | 0.550 | 0.551 | 0.551 |
| L5H1 | 17 | 0.529 | 0.529 | 0.529 | 0.529 | 0.529 | 0.529 | 0.529 | 0.529 | 0.529 |

## Self-contained fidelity

| Target | Boundary | MLP | Queries | Keys | Value | Q+K | Q+V | K+V | Attn no MLP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L0H7 | 6 | 0.486 | 0.546 | 0.501 | 0.615 | 0.565 | 0.667 | 0.621 | 0.341 |
| L0H7 | 10 | 0.629 | 0.646 | 0.629 | 0.694 | 0.651 | 0.713 | 0.691 | 0.512 |
| L0H7 | 14 | 0.623 | 0.627 | 0.624 | 0.624 | 0.628 | 0.628 | 0.624 | 0.534 |
| L0H7 | 17 | 0.603 | 0.603 | 0.603 | 0.603 | 0.603 | 0.603 | 0.603 | 0.603 |
| L4H2 | 6 | 0.353 | 0.426 | 0.374 | 0.470 | 0.447 | 0.563 | 0.479 | 0.208 |
| L4H2 | 10 | 0.485 | 0.543 | 0.493 | 0.538 | 0.552 | 0.598 | 0.538 | 0.318 |
| L4H2 | 14 | 0.473 | 0.479 | 0.477 | 0.475 | 0.487 | 0.482 | 0.478 | 0.367 |
| L4H2 | 17 | 0.435 | 0.435 | 0.435 | 0.435 | 0.435 | 0.435 | 0.435 | 0.435 |
| L5H1 | 6 | 0.313 | 0.388 | 0.347 | 0.436 | 0.414 | 0.537 | 0.444 | 0.169 |
| L5H1 | 10 | 0.524 | 0.557 | 0.541 | 0.565 | 0.576 | 0.603 | 0.576 | 0.335 |
| L5H1 | 14 | 0.541 | 0.547 | 0.544 | 0.544 | 0.550 | 0.550 | 0.546 | 0.453 |
| L5H1 | 17 | 0.529 | 0.529 | 0.529 | 0.529 | 0.529 | 0.529 | 0.529 | 0.529 |

## Guardrails

These are ablated recurrent programs, not an additive attribution: removing one role changes every earlier adjoint on which later role terms operate. Pairwise gains therefore establish compositional necessity, not a Shapley percentage. `Address` supplies exact payload geometry after testing only the router; `self` is the stricter surrogate. Score-adjoint sampling is smaller than the high-sample chart, so the exact full-program recomposition is the implementation gate and conclusions concern role families rather than literal axes.
