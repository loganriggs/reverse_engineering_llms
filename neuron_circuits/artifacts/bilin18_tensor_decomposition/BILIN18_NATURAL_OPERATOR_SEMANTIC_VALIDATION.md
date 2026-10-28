# Held-out semantic validation of natural operator invocations

The semantic predicates below were frozen after inspecting the first 512-row FineWeb atlas and before scanning the disjoint held-out rows. They describe broader natural functions, not the names of the synthetic tasks that originally exposed each route template.

## Locked hypotheses

- **V0_agreement_route_predicts_auxiliaries: not supported.** Observed: `{'agreement_rate': 0.09375, 'other_rate': 0.04296875, 'ratio': 2.1818181818181817}`
- **V1_ioi_route_is_local_name_valued_without_echo: not supported.** Observed: `{'profile': {'auxiliary_rate': 0.0, 'local_rate': 0.546875, 'source_echo_rate': 0.0, 'capitalized_output_fraction': 0.7864583333333334, 'copy_rate': 0.015625, 'median_distance': 2.0}}`
- **V2_capital_route_is_source_linked_entity_lexicalization: supported.** Observed: `{'profile': {'auxiliary_rate': 0.046875, 'local_rate': 0.078125, 'source_echo_rate': 0.953125, 'capitalized_output_fraction': 0.7526041666666666, 'copy_rate': 0.046875, 'median_distance': 17.0}}`
- **V3_repetition_route_is_long_range_copy: supported.** Observed: `{'profiles': [{'auxiliary_rate': 0.0625, 'local_rate': 0.171875, 'source_echo_rate': 0.9375, 'capitalized_output_fraction': 0.4427083333333333, 'copy_rate': 0.5, 'median_distance': 19.5}, {'auxiliary_rate': 0.0625, 'local_rate': 0.171875, 'source_echo_rate': 0.9375, 'capitalized_output_fraction': 0.4427083333333333, 'copy_rate': 0.484375, 'median_distance': 20.5}]}`
- **V4_semantic_profiles_transfer: supported.** Observed: `{'cosines': {'agreement_attractor:0': 0.9661973259749915, 'ioi:0': 0.9851547405587701, 'capital_recall:0': 0.9986587360823368, 'repeated_sequence:0': 0.9987086320846793, 'repeated_sequence:1': 0.9993273472939395}}`

| Route role | Aux next | Local d≤2 | Source echo | Capitalized outputs | Copy-next | Median distance | Profile cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| agreement_attractor:0 | 0.094 | 0.578 | 0.000 | 0.062 | 0.000 | 2.0 | 0.966 |
| ioi:0 | 0.000 | 0.547 | 0.000 | 0.786 | 0.016 | 2.0 | 0.985 |
| capital_recall:0 | 0.047 | 0.078 | 0.953 | 0.753 | 0.047 | 17.0 | 0.999 |
| repeated_sequence:0 | 0.062 | 0.172 | 0.938 | 0.443 | 0.500 | 19.5 | 0.999 |
| repeated_sequence:1 | 0.062 | 0.172 | 0.938 | 0.443 | 0.484 | 20.5 | 0.999 |

## Interpretation boundary

These are operational semantic profiles of high-scoring invocations. They do not show that every use of a head has the same meaning, and they do not yet show that the route templates exhaust each natural function. The validated objects are broader operators—agreement-like auxiliary control, local name-valued association, source-linked entity lexicalization, and long-range copy—not one concept per head.
