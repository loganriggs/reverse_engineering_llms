# Natural FineWeb invocation atlas for typed read operators

Frozen route templates from four controlled behaviors are scanned over unlabeled FineWeb edges. Ranking uses route-template cosine times a weight-aware edge-magnitude proxy. No task labels or answer contrasts enter the search. Every nonidentity permutation of route coefficients across the same three heads is a matched control.

## Locked hypotheses

- **N0_natural_matches_exist: not supported.** Observed: `{'median_cosines': {'agreement_attractor:0': 0.947624534368515, 'ioi:0': 0.9423688650131226, 'capital_recall:0': 0.8910095393657684, 'repeated_sequence:0': 0.9551781415939331, 'repeated_sequence:1': 0.8626134693622589}}`
- **N1_native_head_typing_beats_permuted_templates: not supported.** Observed: `{'families': ['ioi', 'capital_recall'], 'ratios': {'agreement_attractor:0': 1.0055282612988683, 'ioi:0': 1.0611717457201717, 'capital_recall:0': 1.2600620216024099, 'repeated_sequence:0': 1.04064009892873, 'repeated_sequence:1': 1.092699025020736}}`
- **N2_natural_invocations_are_not_one_token_artifacts: supported.** Observed: `{'distinct': {'agreement_attractor:0': 49, 'ioi:0': 64, 'capital_recall:0': 61, 'repeated_sequence:0': 62, 'repeated_sequence:1': 62}}`
- **N3_repetition_template_enriches_copy_relations: supported.** Observed: `{'repetition_copy_rate': 0.5078125, 'other_copy_rate': 0.005208333333333333}`
- **N4_examples_are_saved_for_semantic_redteam: supported.** Observed: `{}`

| Template | Site/program | Role | Median cosine | Native/permuted score | Distinct sources | Copy-next rate |
|---|---|---:|---:|---:|---:|---:|
| agreement_attractor | L11 H0,H2,H3 | 0 | 0.948 | 1.006 | 49 | 0.000 |
| ioi | L14 H4,H6,H7 | 0 | 0.942 | 1.061 | 64 | 0.000 |
| capital_recall | L13 H0,H5,H6 | 0 | 0.891 | 1.260 | 61 | 0.016 |
| repeated_sequence | L8 H3,H4,H6 | 0 | 0.955 | 1.041 | 62 | 0.531 |
| repeated_sequence | L8 H3,H4,H6 | 1 | 0.863 | 1.093 | 62 | 0.484 |

## Representative unlabeled matches

### agreement_attractor

- role 0, d=7, cos=0.946: ` Australia began in 1992 when a need to collect and preserve our military history was recognised`; source ` need` → next ` was`; direct outputs ` came`, ` resulted`, ` was`, ` became`.
- role 0, d=7, cos=0.963: ` a correct one.\nThe reason why it is not a correct one is`; source ` reason` → next ` is`; direct outputs ` is`, ` was`, ` lies`, ` isn`.
- role 0, d=3, cos=0.945: `BA) says the air travel market in 2037 will be`; source ` market` → next ` will`; direct outputs ` has`, ` is`, ` remains`, ` continues`.
- role 0, d=5, cos=0.980: ` obtain concrete offers. Here all details and characteristics of the product should be`; source ` details` → next ` should`; direct outputs ` are`, ` were`, ` aren`, ` can`.
- role 0, d=7, cos=0.991: `able Sphynx cat veterinarians in Tucson, Arizona (AZ) are`; source `arians` → next ` are`; direct outputs ` are`, ` were`, ` include`, ` have`.

### ioi

- role 0, d=1, cos=0.917: ` for great healing. Possibly a happier and balanced parenting`; source ` happier` → next ` balanced`; direct outputs ` Pumpkin`, ` pumpkin`, ` Cheng`, `yrus`.
- role 0, d=1, cos=0.900: ` app can send instant messages, photos, audio,`; source ` photos` → next ` audio`; direct outputs ` dispens`, ` Animation`, ` Jolly`, ` hammer`.
- role 0, d=1, cos=0.971: ` and damping, the extremely stable and tors`; source ` stable` → next ` t`; direct outputs ` Dick`, ` Panama`, ` Sergio`, ` ESP`.
- role 0, d=1, cos=0.972: ` out. We also have the awards and speeches plus`; source ` awards` → next ` speeches`; direct outputs ` Mans`, ` Toledo`, ` Shak`, ` Bread`.
- role 0, d=2, cos=0.982: ` pains. Some people even had nausea and severe headaches`; source ` nausea` → next ` headaches`; direct outputs ` Swansea`, ` Bristol`, ` Shak`, ` Norwich`.

### capital_recall

- role 0, d=1, cos=0.960: ` his buns off travelling around Alberta and Saskatchewan in`; source ` Alberta` → next ` Saskatchewan`; direct outputs ` Alberta`, ` Edmonton`, ` Calgary`, ` Saskatchewan`.
- role 0, d=2, cos=0.927: ` around the time of pregnancy, iodine to support thyroid`; source ` iodine` → next ` thyroid`; direct outputs ` iodine`, ` thyroid`, ` iod`, ` radioactive`.
- role 0, d=46, cos=0.847: `\nIn 2004 the makers of Ivory Caps had embarked on a business …  use. Though the idea was simple enough, the research`; source ` Ivory` → next ` the`; direct outputs ` Ivory`, `Iv`, ` African`, ` Iv`.
- role 0, d=61, cos=0.978: ` makes up Haiti is a broadly mountainous one and …  Massif du Nord and the southern string of mountains that`; source ` Haiti` → next ` mountains`; direct outputs ` Haitian`, ` Haiti`, ` Hait`, ` Dominican`.
- role 0, d=38, cos=0.992: ` corruption charges earlier this week, Algerians have continued to take to …  or fair and that the candidates are part of the`; source ` Alger` → next ` the`; direct outputs ` Alger`, ` Algeria`, ` Morocco`, ` Tunis`.

### repeated_sequence

- role 0, d=22, cos=0.945: ` dollari, una carogna\nFrom The Spaghetti Western Database\nUna Colt, cinque dollari, una carogn`; source `ogn` → next `ogn`; direct outputs `ogn`, ` Cogn`, ` Recogn`, ` cogn`.
- role 0, d=4, cos=0.967: ` prior written consent of KMB Trend Products. KMB Trend Products`; source ` Trend` → next ` Trend`; direct outputs ` Trend`, `Trend`, ` trend`, ` Trends`.
- role 0, d=21, cos=0.955: ` a.m.||Keva 1||Diablos||vs.||Madison Fire|\n|9 a.m.||Keva`; source `eva` → next `eva`; direct outputs `eva`, ` Mak`, `ère`, `ev`.
- role 0, d=28, cos=0.888: ` /en/manufactured-home~for-sale~miworth-prince-george/262496687?view=Summary /en/manufactured-home`; source `home` → next `home`; direct outputs ` Home`, ` home`, `home`, `Home`.
- role 0, d=36, cos=0.933: ` Much like my "leaf-lining behavior of Buteo haw …  in native Central American plants avoided by leaf-cutting ants`; source `lining` → next `cutting`; direct outputs `lining`, `line`, `lines`, ` line`.
- role 1, d=22, cos=0.827: ` dollari, una carogna\nFrom The Spaghetti Western Database\nUna Colt, cinque dollari, una carogn`; source `ogn` → next `ogn`; direct outputs `ogn`, ` Cogn`, ` Recogn`, ` cogn`.
- role 1, d=1, cos=0.665: ` 13, 14 (pp. 329 – 415)`; source ` 329` → next ` 415`; direct outputs ` 330`, ` 331`, ` 329`, `330`.
- role 1, d=4, cos=0.863: ` prior written consent of KMB Trend Products. KMB Trend Products`; source ` Trend` → next ` Trend`; direct outputs ` Trend`, `Trend`, ` trend`, ` Trends`.
- role 1, d=21, cos=0.862: ` a.m.||Keva 1||Diablos||vs.||Madison Fire|\n|9 a.m.||Keva`; source `eva` → next `eva`; direct outputs `eva`, ` Mak`, `ère`, `ev`.
- role 1, d=42, cos=0.968: `\nMade in France<|endoftext|>Sat07Feb20159:00 am …  and coffee at Congleton Library.\nSat14Feb`; source `07` → next `14`; direct outputs `07`, ` 07`, `08`, ` 2007`.

## Guardrail

A high route-template match establishes recurrent tensor geometry, not the original task semantics. The saved examples are the discovery set for later semantic grouping; any labels inferred from them must be frozen and tested on disjoint corpus rows.
