# QK×OV factor decomposition of singular third-order paths

The dominant final head of every causal failure is decomposed into first/second QK routing of current/value-bus codes plus the direct current-value derivative.

## Locked hypotheses

- **F0_factor_decomposition_recomposes: supported.**
- **F1_one_factor_dominates: supported.**
- **F2_routing_derivative_dominates: supported.**
- **F3_direct_current_value_dominates: not supported.**

Maximum recomposition error relative to exact response: `1.729e-05`. Median top-factor share: `0.693`. Effective support: `1.79`. QK-routing mass: `0.999`. Direct current-value share: `0.001`.

| Corpus | Query/row | Triple/head | Top factor/share | Effective support | Routing mass | Current-V share |
|---|---|---|---:|---:|---:|---:|
| fineweb | 3/22 | [9, 11, 13]/H0 | qk2_bus_route/0.579 | 2.42 | 0.997 | 0.003 |
| fineweb | 10/291 | [9, 10, 11]/H6 | qk1_current_route/0.776 | 1.59 | 0.982 | 0.018 |
| wikitext | 1/271 | [9, 10, 14]/H6 | qk2_bus_route/0.498 | 2.11 | 1.000 | 0.000 |
| wikitext | 1/271 | [9, 11, 14]/H6 | qk1_bus_route/0.848 | 1.36 | 1.000 | 0.000 |
| wikitext | 1/271 | [9, 13, 14]/H6 | qk2_bus_route/0.775 | 1.57 | 1.000 | 0.000 |
| wikitext | 1/271 | [10, 12, 14]/H6 | qk2_bus_route/0.693 | 1.79 | 0.999 | 0.001 |
| wikitext | 1/271 | [10, 12, 17]/H2 | qk2_current_route/0.849 | 1.37 | 0.934 | 0.066 |
| wikitext | 1/271 | [10, 13, 14]/H6 | qk2_bus_route/0.786 | 1.53 | 1.000 | 0.000 |
| wikitext | 1/271 | [10, 13, 15]/H6 | qk1_bus_route/0.542 | 2.09 | 0.990 | 0.010 |
| wikitext | 1/271 | [10, 14, 15]/H6 | qk1_bus_route/0.488 | 2.26 | 0.989 | 0.011 |
| wikitext | 1/271 | [11, 12, 14]/H6 | qk2_bus_route/0.645 | 1.91 | 0.999 | 0.001 |
| wikitext | 1/271 | [11, 13, 14]/H6 | qk2_bus_route/0.780 | 1.55 | 1.000 | 0.000 |
| wikitext | 1/271 | [11, 14, 15]/H6 | qk1_bus_route/0.488 | 2.27 | 0.989 | 0.011 |

This is an exact final-factor decomposition. If a QK/value-bus term dominates, the next trace should inspect its source-token routing and the two earlier attention factors in the ordered path.
