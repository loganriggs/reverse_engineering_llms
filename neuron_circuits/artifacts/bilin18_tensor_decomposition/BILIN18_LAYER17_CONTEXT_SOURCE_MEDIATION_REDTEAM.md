# Attention-mediated red team for Layer-17 context sources

Selected-position embedding or carried terms are mean-replaced or cyclically substituted. Final attention is either held native or exactly recomputed with the edited mixed state and native persistent value bus.

## Locked hypotheses

- **M0_native_recomposition_is_exact: supported.** Native context coefficients recompute to relative error at most 1e-6.
- **M1_direct_embedding_mean_replacement_is_small: supported.** Replacing the selected-position direct Layer-17 embedding term by its discovery mean and recomputing attention retains at least 0.98 operator fidelity on every corpus.
- **M2_direct_embedding_cycle_is_small: supported.** Cyclically substituting the selected-position direct embedding term and recomputing attention retains at least 0.95 fidelity on every corpus.
- **M3_embedding_attention_mediation_is_small: supported.** Recomputing rather than holding attention changes embedding-replacement fidelity by at most 0.02 on every corpus.
- **M4_carried_mean_replacement_is_material: supported.** Replacing selected-position carried state by its discovery mean and recomputing attention retains at most 0.80 fidelity on every corpus.
- **M5_carried_cycle_is_material: supported.** Cyclically substituting selected-position carried state and recomputing attention retains at most 0.90 fidelity on every corpus.
- **M6_carried_cost_dominates_embedding_cost: supported.** The operator-fidelity cost of carried mean replacement is at least ten times the direct-embedding mean-replacement cost on every corpus.
- **M7_attention_recomputation_preserves_source_order: supported.** Carried replacement is more damaging than direct-embedding replacement under both held-attention and recomputed-attention executions on every corpus.

## Operator fidelity

| Corpus | embedding mean held/recomputed | embedding cycle held/recomputed | carried mean held/recomputed | carried cycle held/recomputed |
|---|---:|---:|---:|---:|
| fineweb | 0.999997/0.999997 | 0.999996/0.999995 | 0.618/0.537 | 0.514/0.382 |
| wikitext | 0.999998/0.999997 | 0.999997/0.999996 | 0.698/0.621 | 0.626/0.512 |
| tinystories | 0.999999/0.999998 | 0.999997/0.999997 | 0.740/0.700 | 0.641/0.575 |

The intervention changes only the current Layer-17 injection boundary. A negligible direct embedding effect is compatible with strong embedding predictability because the carried state has already compiled token identity through seventeen earlier blocks and embedding injections.
