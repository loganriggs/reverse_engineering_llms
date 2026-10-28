# Held-out semantic validation of unsupervised motifs

Semantic hypotheses were frozen after interpreting the weight-discovered motifs, then evaluated on untouched FineWeb rows 19,200–19,999.

## Locked outcomes

- **S1_semantic_routes_enrich: not supported.**
- **S2_frozen_heads_beat_layer_controls: supported.**
- **S3_semantic_edges_are_causal: supported.**
- **S4_semantic_edges_beat_matched_controls: supported.**

| family | route enrichment | comparison enrichment | causal sites | semantic KL | matched-control KL | ratio | |Δ target log p| |
|---|---:|---:|---:|---:|---:|---:|---:|
| recurrence_reference | 10.22 | 1.78 | 18 | 0.0104 | 0.000203 | 51.22 | 0.0839 |
| local_composition_a | 21.82 | 7.83 | 18 | 0.0133 | 0.000208 | 63.83 | 0.077 |
| local_composition_b | 20.66 | 7.95 | 12 | 0.00679 | 0.000269 | 25.21 | 0.0758 |
| boundary_multiscale | 1.36 | 1.10 | 30 | 0.00151 | 0.0002 | 7.55 | 0.0225 |

Routing enrichment is semantic-edge energy share divided by that edge type's prevalence. The comparison column is the median across all native heads in the same layers. Finite controls remove a nonsemantic source edge at the same head and destination with the closest available local-write magnitude.
