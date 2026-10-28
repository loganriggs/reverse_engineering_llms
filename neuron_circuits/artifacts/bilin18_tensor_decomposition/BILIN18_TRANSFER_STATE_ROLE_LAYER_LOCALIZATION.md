# Layer localization of signed role programs

At Boundary 6, key+value address programs and query+value payload programs are restricted to the nearest or farthest 2, 4, or 8 downstream attention layers. MLP VJPs and residual carry remain enabled throughout. Address programs receive exact local payload bases; self-contained programs generate both site rankings and local bases. Every causal chart retains the same 128 runtime coordinates.

## Locked hypotheses

- **L0_full_recomposition: supported.**
- **L1_near4_key_value_recovers_addresses: supported.**
- **L2_near8_key_value_is_address_sufficient: not supported.**
- **L3_near_roles_beat_far_roles: supported.**
- **L4_near8_query_value_is_self_contained: supported.**
- **L5_near4_query_value_is_not_uniformly_sufficient: not supported.**
- **L6_role_invocations_are_depth_local: not supported.**
- **attention_jvp_forward_invariant: supported.**

## Address recovery

| Program | Median top-8 Jaccard |
|---|---:|
| full | 1.000 |
| kv_all | 0.778 |
| qv_all | 0.889 |
| kv_near2 | 0.600 |
| kv_far2 | 0.455 |
| qv_near2 | 0.689 |
| qv_far2 | 0.527 |
| kv_near4 | 0.778 |
| kv_far4 | 0.455 |
| qv_near4 | 0.778 |
| qv_far4 | 0.600 |
| kv_near8 | 1.000 |
| kv_far8 | 0.600 |
| qv_near8 | 0.889 |
| qv_far8 | 0.778 |

## Causal fidelity

| Target | Exact address | K+V all address | Q+V all self | Q+V near2 | near4 | near8 | far2 | far4 | far8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L0H7 | 0.697 | 0.678 | 0.680 | 0.599 | 0.663 | 0.686 | 0.498 | 0.486 | 0.638 |
| L4H2 | 0.600 | 0.592 | 0.584 | 0.452 | 0.518 | 0.577 | 0.389 | 0.405 | 0.518 |
| L5H1 | 0.546 | 0.545 | 0.544 | 0.416 | 0.477 | 0.538 | 0.364 | 0.368 | 0.459 |

## Guardrails

Near and far programs are component-removal recurrences, not additive layer attribution. MLP and residual paths remain active everywhere, so a near-eight success means eight attention-role invocations suffice under that interface; it does not mean later layers are behaviorally unused. The test is localized to Boundary 6, where the previous read-program gap was largest.
