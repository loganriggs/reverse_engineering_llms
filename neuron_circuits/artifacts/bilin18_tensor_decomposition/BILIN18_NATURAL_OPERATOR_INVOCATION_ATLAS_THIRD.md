# Natural FineWeb invocation atlas for typed read operators

Frozen route templates from four controlled behaviors are scanned over unlabeled FineWeb edges. Ranking uses route-template cosine times a weight-aware edge-magnitude proxy. No task labels or answer contrasts enter the search. Every nonidentity permutation of route coefficients across the same three heads is a matched control.

## Locked hypotheses

- **N0_natural_matches_exist: not supported.** Observed: `{'median_cosines': {'agreement_attractor:0': 0.9451021552085876, 'ioi:0': 0.9538741707801819, 'capital_recall:0': 0.8646757304668427, 'repeated_sequence:0': 0.9451881647109985, 'repeated_sequence:1': 0.8592816889286041}}`
- **N1_native_head_typing_beats_permuted_templates: supported.** Observed: `{'families': ['agreement_attractor', 'capital_recall', 'repeated_sequence'], 'ratios': {'agreement_attractor:0': 1.0882662585753702, 'ioi:0': 1.0355201113875825, 'capital_recall:0': 1.290774732650937, 'repeated_sequence:0': 1.123060522691688, 'repeated_sequence:1': 1.1929672438416477}}`
- **N2_natural_invocations_are_not_one_token_artifacts: supported.** Observed: `{'distinct': {'agreement_attractor:0': 56, 'ioi:0': 64, 'capital_recall:0': 61, 'repeated_sequence:0': 64, 'repeated_sequence:1': 64}}`
- **N3_repetition_template_enriches_copy_relations: supported.** Observed: `{'repetition_copy_rate': 0.3828125, 'other_copy_rate': 0.026041666666666668}`
- **N4_examples_are_saved_for_semantic_redteam: supported.** Observed: `{}`

| Template | Site/program | Role | Median cosine | Native/permuted score | Distinct sources | Copy-next rate |
|---|---|---:|---:|---:|---:|---:|
| agreement_attractor | L11 H0,H2,H3 | 0 | 0.945 | 1.088 | 56 | 0.000 |
| ioi | L14 H4,H6,H7 | 0 | 0.954 | 1.036 | 64 | 0.000 |
| capital_recall | L13 H0,H5,H6 | 0 | 0.865 | 1.291 | 61 | 0.078 |
| repeated_sequence | L8 H3,H4,H6 | 0 | 0.945 | 1.123 | 64 | 0.391 |
| repeated_sequence | L8 H3,H4,H6 | 1 | 0.859 | 1.193 | 64 | 0.375 |

## Representative unlabeled matches

### agreement_attractor

- role 0, d=5, cos=0.978: ` stockpiles will remain elevated as constraints on the Seaway pipeline continue to`; source ` constraints` → next ` continue`; direct outputs ` occur`, ` are`, ` indicate`, ` reduce`.
- role 0, d=3, cos=0.949: ` reading to turn pages but the buttons to turn pages are in`; source ` buttons` → next ` are`; direct outputs ` are`, ` aren`, ` were`, ` ARE`.
- role 0, d=11, cos=0.913: ` is completed and long-term contracts for the supply of liquefied natural gas from Russia are signed`; source ` contracts` → next ` are`; direct outputs ` are`, ` aren`, ` were`, ` appear`.
- role 0, d=6, cos=0.983: ` white bass.\ntemperatures just reaching the low 50s, white`; source `atures` → next `,`; direct outputs ` are`, ` were`, ` aren`, ` ARE`.
- role 0, d=4, cos=0.918: ` year. We had a great opportunity in front of us and I`; source ` opportunity` → next ` and`; direct outputs ` to`, `to`, ` To`, ` TO`.

### ioi

- role 0, d=1, cos=0.903: ` garden.(Deglitches vegetables and herbs)`; source ` vegetables` → next ` herbs`; direct outputs ` Bart`, ` Robertson`, ` Nathan`, `Carter`.
- role 0, d=1, cos=0.720: `isive Facilities and Civil Society in Japan and the`; source ` Japan` → next ` the`; direct outputs ` Melanie`, `innamon`, ` Goblin`, ` Leigh`.
- role 0, d=1, cos=0.991: ` followed by “and the advisors and the`; source ` advisors` → next ` the`; direct outputs `Kenn`, ` Cat`, ` brewers`, ` Lucy`.
- role 0, d=1, cos=0.999: ` for her in his will. Brown and Hyn`; source ` Brown` → next ` H`; direct outputs ` hangar`, `Roy`, ` Vi`, `Pool`.
- role 0, d=1, cos=0.724: ` Danish, Portuguese, Finnish, Italian,Norwegian`; source ` Italian` → next `Nor`; direct outputs `Dun`, ` Rams`, ` cosmetics`, ` Mech`.

### capital_recall

- role 0, d=3, cos=0.888: ` of the fishes he knows on Hawaii’s coral reefs`; source ` Hawaii` → next ` coral`; direct outputs ` Hawaii`, ` Hawaiian`, ` Hawai`, ` Honolulu`.
- role 0, d=17, cos=0.956: ` for issues of social conformity in Kafka's Metamorphosis while another concentrates on the relationship between Gregor Samsa and his father`; source ` Kafka` → next ` his`; direct outputs ` Kafka`, ` literary`, ` Hungarian`, ` Berkeley`.
- role 0, d=1, cos=0.864: ` brook trout.\nEast Tennessee and Western North`; source ` Tennessee` → next ` Western`; direct outputs ` Tennessee`, `Tenn`, ` Nashville`, ` TN`.
- role 0, d=2, cos=0.860: ` to catch the eye of a Nebraska Angel\nSc`; source ` Nebraska` → next `Sc`; direct outputs ` Nebraska`, ` Omaha`, ` Iowa`, `braska`.
- role 0, d=5, cos=0.883: ` the ongoing nuclear crisis at the Fukushima Daiichi plant in the town of`; source ` Fukushima` → next ` town`; direct outputs ` Fukushima`, ` Japanese`, ` Japan`, ` reactor`.

### repeated_sequence

- role 0, d=66, cos=0.936: ` Lower East Side and Harlem. MORE\nNov. 20, 2012 …  ad hoc facilities around the New York region. MORE\n`; source ` MORE` → next ` MORE`; direct outputs ` MORE`, ` More`, ` more`, ` less`.
- role 0, d=77, cos=0.953: ` the profile by using the Delprof.exe utilityYou can use …  the User Profile Deletion utility (Delprof.`; source `prof` → next `prof`; direct outputs `prof`, ` prof`, ` Prof`, `Prof`.
- role 0, d=1, cos=0.982: ` 12 non-conference schedules from toughest to easiest\n`; source ` toughest` → next ` easiest`; direct outputs ` toughest`, ` tough`, ` tougher`, ` harder`.
- role 0, d=19, cos=0.926: ` Pumpkin Hair Products\nSkin Obsession Pumpkin Enzyme Mask Peel\nTreat your face to the plumping benefits of the Skin Obsession Pumpkin`; source `ession` → next `ession`; direct outputs `ession`, ` pot`, ` over`, ` professional`.
- role 0, d=9, cos=0.987: `up is not included. Fox Glacier Guiding Building, 44 Main Road, Fox Glacier.`; source ` Glacier` → next ` Glacier`; direct outputs ` Glacier`, ` glacier`, ` gl`, ` glaciers`.
- role 1, d=66, cos=0.831: ` Lower East Side and Harlem. MORE\nNov. 20, 2012 …  ad hoc facilities around the New York region. MORE\n`; source ` MORE` → next ` MORE`; direct outputs ` MORE`, ` More`, ` more`, ` less`.
- role 1, d=1, cos=0.991: ` 12 non-conference schedules from toughest to easiest\n`; source ` toughest` → next ` easiest`; direct outputs ` toughest`, ` tough`, ` tougher`, ` harder`.
- role 1, d=77, cos=0.828: ` the profile by using the Delprof.exe utilityYou can use …  the User Profile Deletion utility (Delprof.`; source `prof` → next `prof`; direct outputs `prof`, ` prof`, ` Prof`, `Prof`.
- role 1, d=2, cos=0.943: `ville, Pa.; Charleston and Spartanburg, S`; source ` Spartan` → next ` S`; direct outputs ` Spartan`, ` S`, ` SC`, `SC`.
- role 1, d=9, cos=0.900: `up is not included. Fox Glacier Guiding Building, 44 Main Road, Fox Glacier.`; source ` Glacier` → next ` Glacier`; direct outputs ` Glacier`, ` glacier`, ` gl`, ` glaciers`.

## Guardrail

A high route-template match establishes recurrent tensor geometry, not the original task semantics. The saved examples are the discovery set for later semantic grouping; any labels inferred from them must be frozen and tested on disjoint corpus rows.
