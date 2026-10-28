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
| keys | 0.527 | 0.689 | 0.889 | 1.000 |
| value | 0.889 | 0.778 | 0.527 | 1.000 |
| query_key | 0.455 | 0.689 | 0.778 | 1.000 |
| query_value | 0.778 | 0.778 | 0.527 | 1.000 |
| key_value | 1.000 | 1.000 | 1.000 | 1.000 |
| attention_no_mlp | 0.689 | 0.600 | 0.778 | 1.000 |
| full | 1.000 | 1.000 | 1.000 | 1.000 |

## Causal address fidelity

| Target | Boundary | Exact | MLP | Queries | Keys | Value | Q+K | Q+V | K+V | Attn no MLP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L0H7 | 6 | 0.650 | 0.590 | 0.590 | 0.582 | 0.670 | 0.591 | 0.654 | 0.650 | 0.631 |
| L0H7 | 10 | 0.726 | 0.696 | 0.696 | 0.695 | 0.728 | 0.694 | 0.728 | 0.726 | 0.695 |
| L0H7 | 14 | 0.654 | 0.653 | 0.653 | 0.653 | 0.654 | 0.653 | 0.654 | 0.653 | 0.654 |
| L0H7 | 17 | 0.622 | 0.622 | 0.622 | 0.622 | 0.622 | 0.622 | 0.622 | 0.622 | 0.622 |
| L4H2 | 6 | 0.564 | 0.471 | 0.471 | 0.485 | 0.567 | 0.476 | 0.567 | 0.564 | 0.560 |
| L4H2 | 10 | 0.607 | 0.576 | 0.576 | 0.576 | 0.608 | 0.576 | 0.608 | 0.607 | 0.579 |
| L4H2 | 14 | 0.519 | 0.514 | 0.514 | 0.518 | 0.517 | 0.518 | 0.517 | 0.518 | 0.515 |
| L4H2 | 17 | 0.481 | 0.481 | 0.481 | 0.481 | 0.481 | 0.481 | 0.481 | 0.481 | 0.481 |
| L5H1 | 6 | 0.492 | 0.472 | 0.472 | 0.440 | 0.526 | 0.430 | 0.527 | 0.492 | 0.494 |
| L5H1 | 10 | 0.612 | 0.593 | 0.593 | 0.592 | 0.609 | 0.591 | 0.609 | 0.612 | 0.598 |
| L5H1 | 14 | 0.580 | 0.579 | 0.579 | 0.580 | 0.581 | 0.580 | 0.581 | 0.580 | 0.580 |
| L5H1 | 17 | 0.557 | 0.557 | 0.557 | 0.557 | 0.557 | 0.557 | 0.557 | 0.557 | 0.557 |

## Self-contained fidelity

| Target | Boundary | MLP | Queries | Keys | Value | Q+K | Q+V | K+V | Attn no MLP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L0H7 | 6 | 0.445 | 0.528 | 0.463 | 0.585 | 0.545 | 0.649 | 0.575 | 0.331 |
| L0H7 | 10 | 0.640 | 0.667 | 0.644 | 0.698 | 0.673 | 0.724 | 0.697 | 0.465 |
| L0H7 | 14 | 0.650 | 0.652 | 0.651 | 0.651 | 0.653 | 0.653 | 0.651 | 0.537 |
| L0H7 | 17 | 0.622 | 0.622 | 0.622 | 0.622 | 0.622 | 0.622 | 0.622 | 0.622 |
| L4H2 | 6 | 0.347 | 0.409 | 0.369 | 0.466 | 0.427 | 0.560 | 0.470 | 0.215 |
| L4H2 | 10 | 0.495 | 0.550 | 0.509 | 0.551 | 0.560 | 0.604 | 0.554 | 0.317 |
| L4H2 | 14 | 0.507 | 0.513 | 0.511 | 0.509 | 0.517 | 0.515 | 0.512 | 0.398 |
| L4H2 | 17 | 0.481 | 0.481 | 0.481 | 0.481 | 0.481 | 0.481 | 0.481 | 0.481 |
| L5H1 | 6 | 0.304 | 0.373 | 0.338 | 0.420 | 0.396 | 0.520 | 0.411 | 0.170 |
| L5H1 | 10 | 0.535 | 0.561 | 0.557 | 0.576 | 0.580 | 0.605 | 0.587 | 0.341 |
| L5H1 | 14 | 0.573 | 0.577 | 0.575 | 0.576 | 0.578 | 0.580 | 0.576 | 0.470 |
| L5H1 | 17 | 0.557 | 0.557 | 0.557 | 0.557 | 0.557 | 0.557 | 0.557 | 0.557 |

## Guardrails

These are ablated recurrent programs, not an additive attribution: removing one role changes every earlier adjoint on which later role terms operate. Pairwise gains therefore establish compositional necessity, not a Shapley percentage. `Address` supplies exact payload geometry after testing only the router; `self` is the stricter surrogate. Score-adjoint sampling is smaller than the high-sample chart, so the exact full-program recomposition is the implementation gate and conclusions concern role families rather than literal axes.
