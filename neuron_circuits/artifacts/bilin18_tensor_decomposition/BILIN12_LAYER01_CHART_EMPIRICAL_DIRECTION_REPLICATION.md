# Empirical token-difference direction control

Each natural lexical edit is compared with 8 random directions sampled from actual vocabulary-token differences. The random direction is transported to the natural source embedding and converted into an endpoint with exactly the same spherical chord length. This preserves empirical embedding anisotropy while removing the declared semantic relation. The native-plus-seven-chart router and all downstream measurements are unchanged.

## Locked hypotheses

- **E0_empirical_direction_chord_is_exact: supported.**
- **E1_token_difference_anisotropy_explains_native_alignment: not supported.**
- **E2_semantic_relation_alignment_exceeds_generic_token_directions: not supported.**
- **E3_chart_gain_is_specific_beyond_empirical_directions: not supported.**
- **E4_empirical_directions_also_benefit_from_chart_routing: not supported.**
- **E5_semantic_edits_prefer_native_over_empirical_directions: not supported.**
- **E6_selection_is_strictly_prospective: supported.**

![Empirical direction control](bilin12_layer01_chart_empirical_direction_replication.png)

| Layer | Corpus | Native fidelity semantic / empirical / sphere | Routed gain semantic / empirical / sphere | Empirical−sphere native pair median (95% CI) | Semantic−empirical native pair median (95% CI) | Native choice semantic / empirical / sphere |
|---:|---|---:|---:|---:|---:|---:|
| 0 | fineweb | 0.829 / 0.692 / 0.705 | +0.008 / +0.020 / +0.058 | +0.001 (-0.034, +0.072) | +0.122 (+0.057, +0.201) | 0.451 / 0.350 / 0.422 |
| 0 | tinystories | 0.806 / 0.688 / 0.664 | +0.000 / +0.038 / +0.068 | +0.003 (-0.105, +0.093) | +0.075 (-0.023, +0.118) | 0.471 / 0.346 / 0.333 |
| 1 | fineweb | 0.803 / 0.693 / 0.676 | +0.050 / +0.000 / +0.077 | +0.022 (-0.043, +0.091) | +0.045 (-0.053, +0.141) | 0.333 / 0.350 / 0.343 |
| 1 | tinystories | 0.705 / 0.680 / 0.664 | +0.064 / +0.018 / +0.057 | +0.005 (-0.012, +0.047) | +0.018 (-0.072, +0.129) | 0.294 / 0.321 / 0.284 |

## Example random direction sources

These token pairs determine only the transported direction; neither token is inserted into the evaluated context.

| Semantic pair | Random token difference 1 | Random token difference 2 |
|---|---|---|
| `1049->7818` | ` depending` → ` adul` | ` Drop` → ` married` |
| `1049->7818` | ` NOW` → ` endings` | ` LU` → ` Akin` |
| `1049->7818` | `etc` → ` reload` | ` Detection` → ` fermented` |
| `3621->1612` | ` transparency` → ` Fut` | `hews` → ` apocalyptic` |
| `3621->1612` | ` Wrest` → `Qaida` | `////////////////////////////////` → `389` |
| `3621->1612` | ` Admir` → ` permitted` | ` Storage` → ` Nuclear` |

Maximum relative chord-distance mismatch: **1.66e-07**. Maximum relative sphere-radius mismatch: **1.38e-07**.

This control distinguishes generic anisotropy of learned token-difference directions from specificity to the intended lexical relation. It does not turn the transported endpoint into a real vocabulary embedding, so a remaining semantic advantage could still reflect the discrete token manifold rather than human-interpretable semantics.
