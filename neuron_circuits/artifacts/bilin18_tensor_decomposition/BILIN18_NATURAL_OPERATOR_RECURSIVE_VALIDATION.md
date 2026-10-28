# Recursive semantic generalization on a third FineWeb panel

The first held-out panel rejected narrow auxiliary and strict-locality labels. Broader labels were then frozen: nearby predicate lexicalization and short-range name-valued association. This third disjoint panel tests those generalizations while replicating the two previously successful functions.

## Locked hypotheses

- **G0_agreement_route_is_nearby_predicate_lexicalization: supported.** Observed: `{'median_distance': 3.0, 'predicate_rate': 0.46875, 'other_rate': 0.0, 'ratio': 4.6875e+29}`
- **G1_ioi_route_is_short_range_name_valued_association: supported.** Observed: `{'profile': {'auxiliary_rate': 0.015625, 'local_rate': 0.625, 'source_echo_rate': 0.0, 'capitalized_output_fraction': 0.8255208333333334, 'copy_rate': 0.0, 'median_distance': 2.0}}`
- **G2_capital_route_remains_source_linked_entity_lexicalization: supported.** Observed: `{'profile': {'auxiliary_rate': 0.0, 'local_rate': 0.125, 'source_echo_rate': 0.9375, 'capitalized_output_fraction': 0.7161458333333334, 'copy_rate': 0.078125, 'median_distance': 17.0}}`
- **G3_repetition_route_remains_long_range_copy: supported.** Observed: `{'profiles': [{'auxiliary_rate': 0.015625, 'local_rate': 0.15625, 'source_echo_rate': 0.96875, 'capitalized_output_fraction': 0.4166666666666667, 'copy_rate': 0.390625, 'median_distance': 18.5}, {'auxiliary_rate': 0.0, 'local_rate': 0.203125, 'source_echo_rate': 0.953125, 'capitalized_output_fraction': 0.4270833333333333, 'copy_rate': 0.375, 'median_distance': 18.5}]}`
- **G4_richer_profiles_remain_stable: supported.** Observed: `{'cosines': {'agreement_attractor:0': 0.9748981319160681, 'ioi:0': 0.9988806563567748, 'capital_recall:0': 0.9987249400781789, 'repeated_sequence:0': 0.9955674479529364, 'repeated_sequence:1': 0.9957733593087613}}`

| Route role | Predicate outputs | Local d≤2 | Echo | Capitalized outputs | Copy-next | Median distance | Profile cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| agreement_attractor:0 | 0.469 | 0.453 | 0.000 | 0.060 | 0.000 | 3.0 | 0.975 |
| ioi:0 | 0.000 | 0.625 | 0.000 | 0.826 | 0.000 | 2.0 | 0.999 |
| capital_recall:0 | 0.000 | 0.125 | 0.938 | 0.716 | 0.078 | 17.0 | 0.999 |
| repeated_sequence:0 | 0.000 | 0.156 | 0.969 | 0.417 | 0.391 | 18.5 | 0.996 |
| repeated_sequence:1 | 0.000 | 0.203 | 0.953 | 0.427 | 0.375 | 18.5 | 0.996 |

## Interpretation boundary

Passing a broader label after a narrow one fails is evidence only because the broader predicate was frozen before this third panel. The result supports recursive semantic organization when related uses share a stable operator profile; it does not license unlimited post-hoc broadening.
