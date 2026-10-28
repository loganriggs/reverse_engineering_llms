# End-to-end causality of task-agnostic route representations

Fresh FineWeb sequences are run to four prospectively selected layers. Only the destination's realized nine-head source routes are replaced; the target-layer MLP and all later layers are recomputed. Fidelity is measured relative to the complete finite effect of deleting those destination routes.

## Locked hypotheses

- **E0_native_path_is_exact: supported.** Observed: `{'maximum_logit_error': 5.245208740234375e-05}`
- **E1_continuous_beats_discrete_end_to_end: supported.** Observed: `{'layers': [1, 8, 13, 17]}`
- **E2_semantic_modes_have_finite_causal_coverage: supported.** Observed: `{'by_layer': {'1': 0.8132050913785753, '8': 0.4582063122610769, '13': 0.05939503237427701, '17': 0.24294703156280129}, 'mean': 0.39343836689418266}`
- **E3_semantic_binding_beats_permuted_modes: supported.** Observed: `{'observed': 0.39343836689418266, 'null_means': [0.14538405367198634, 0.0010214140479128386, 0.03843420593985303, 0.13120925633036373, 0.024450588468229534, 0.01168719581172828, -0.009495712307465903, 0.006174069664950627, 0.06606163889624869, 0.07932193810459978, 0.10295785490193343, 0.09257287963908697, 0.05813479736490418, 0.022211342522218958, 0.20085287078236116, 0.01314276082160512, 0.014110398746744662, 0.003224825654004948, 0.01392285967983195, 0.0481085159734492, 0.04367438053030498, 0.0006870909663738434, 0.04566910589385309, 0.01415474194854388, 0.033476711819930866, -0.0010799816256661099, 0.02088956159564151, 0.024785774432582924, 0.040983131098537545, 0.004382469022709501, 0.07828300873087302, 0.08784655434640462]}`
- **E4_local_write_fidelity_predicts_suffix_fidelity: supported.** Observed: `{'correlation': 0.8320796116706631, 'cells': 12}`
- **E5_logit_coverage_is_not_only_noise: not supported.** Observed: `{'layers': [1, 8]}`
- **E6_semantic_modes_recover_signed_examplewise_ce_effects: supported.** Observed: `{'by_layer': {'1': 0.9160183867471965, '8': 0.4923547167571276, '13': 0.1741268760132757, '17': 0.025937056232405387}, 'mean': 0.40210925893750127}`
- **E7_ce_effect_recovery_beats_permuted_modes: supported.** Observed: `{'observed': 0.40210925893750127, 'null_mean': 0.05507369932996556, 'null_p95': 0.18038279171588167, 'null_means': [0.15812002889865245, -1.6693377724058545e-05, 0.0029967533801782964, 0.1467035807962516, -0.009246456737983633, 0.023581655291340625, 0.004265268063973593, 0.041533247037356585, 0.036645400475370904, 0.054226794131245315, 0.20759283515916194, 0.08613132039863589, 0.031775862915583125, 0.050414412094698546, 0.23117750666701728, 0.04455153175199175, 0.09990219624049751, 0.0022722494329014964, 0.010037409690604071, 0.05636822172725245, 0.013424047030806852, 0.016522650289636093, 0.022107267237319927, 0.06101807177774124, 0.012918675626072212, 0.02872256275368329, -0.0008067242221965476, 0.0058251324099626856, 0.06746666358642261, 0.0022551859267135876, 0.12814025317413652, 0.12573146893159365]}`

| Layer | Native/deleted CE | PCA fidelity/CE | Centroid fidelity/CE | Semantic fidelity/CE | Semantic CE recovery | Best permuted fidelity |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.816/4.305 | 0.984/3.834 | 0.896/3.857 | 0.813/3.894 | 0.840 | 0.566 |
| 8 | 3.816/3.881 | 0.804/3.810 | 0.590/3.830 | 0.458/3.840 | 0.630 | 0.341 |
| 13 | 3.816/3.857 | 0.824/3.823 | 0.323/3.836 | 0.059/3.870 | -0.326 | 0.056 |
| 17 | 3.816/3.849 | 0.850/3.820 | 0.584/3.825 | 0.243/3.855 | -0.196 | 0.138 |

### Per-example finite CE effects

| Layer | PCA fidelity | Centroid fidelity | Semantic fidelity/cosine/sign |
|---:|---:|---:|---:|
| 1 | 0.991 | 0.961 | 0.916/0.958/0.828 |
| 8 | 0.923 | 0.839 | 0.492/0.702/0.648 |
| 13 | 0.928 | 0.773 | 0.174/0.427/0.555 |
| 17 | 0.924 | 0.814 | 0.026/0.161/0.578 |

## Guardrails

This is finite end-to-end evidence for one destination-route intervention, not a fraction of total model computation. Full-logit fidelity weights all vocabulary coordinates; CE tests the observed next token and can disagree when the removed native route was locally harmful. The four layers were chosen after the local coverage audit, so conclusions about all layers require another predeclared replication.
