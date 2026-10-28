# Direct-read operator transfer

For each direct-read family, discovery examples define only mean QK route coefficients for each frozen head and edited-source role. Those coefficients are frozen; held-out contextual values supply the new argument. The result is compared with native edited edges, a shared-route value-only model, frozen discovery values, and head-permuted routes.

## Locked hypotheses

- **F0_exact_edge_recomposition: supported.** Observed: `{'maximum_coordinate_error': 0.00018310546875}`
- **F1_frozen_routes_transport_new_values: supported.** Observed: `{'fidelities': {'agreement_attractor': 0.9232726573285254, 'ioi': 0.9987258469823209, 'capital_recall': 0.9987102244736107, 'repeated_sequence': 0.8936863536319608}}`
- **F2_heldout_values_are_necessary_for_identity_transport: supported.** Observed: `{'heldout_value_mean': 0.9637074750292974, 'frozen_value_mean': -0.030712893898660682, 'difference': 0.9944203689279582}`
- **F3_value_only_change_is_sufficient: supported.** Observed: `{'separate_route_mean': 0.9535987706041045, 'shared_route_mean': 0.9540412857259399, 'difference': -0.0004425151218354628}`
- **F4_actual_edited_edges_recover_the_program: supported.** Observed: `{'passing_families': ['agreement_attractor', 'ioi', 'capital_recall', 'repeated_sequence']}`
- **F5_head_typed_routes_beat_permuted_routes: supported.** Observed: `{'native_mean': 0.9535987706041045, 'permuted_mean': 0.207814096632299, 'difference': 0.7457846739718055}`

| Family | Site/program | Native edited | Frozen route + new value | Shared route × value change | Native route + frozen value | Permuted route |
|---|---|---:|---:|---:|---:|---:|
| agreement_attractor | L11 H0,H2,H3 | 0.855 | 0.923 | 0.923 | -0.091 | 0.579 |
| ioi | L14 H4,H6,H7 | 0.996 | 0.999 | 0.999 | -0.169 | 0.528 |
| capital_recall | L13 H0,H5,H6 | 1.000 | 0.999 | 0.999 | 0.237 | -0.036 |
| repeated_sequence | L8 H3,H4,H6 | 1.000 | 0.894 | 0.896 | -0.161 | -0.239 |

## Interpretation boundary

Success is evidence for a reusable typed read operator with variable contextual arguments. It is not evidence that the transported names or tokens share one semantic direction. Failure can mean routes depend on context more finely than a role-wise mean, or that non-edited relay positions are part of the operation.
