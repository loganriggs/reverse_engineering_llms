# Replication red team of the sparse pair graph

The natural rank-32 basis and weight-only joint rotation are held fixed. Quadratic coordinate pairs are ranked independently on two disjoint 256-example FineWeb panels, frozen, and evaluated on FineWeb plus two OOD corpora. Twenty matched random graphs estimate a null distribution.

## Locked hypotheses

- **P0_top32_graph_is_selection_stable: supported.**
- **P1_both_top32_graphs_are_sufficient: supported.**
- **P2_function_is_more_stable_than_edges: not supported.**
- **P3_selected_graphs_beat_random_null: not supported.**
- **P4_top16_is_sufficient: not supported.**

Pair-set Jaccard at 16/32/64: `1.000/0.939/1.000`.

| Corpus | Full | A/B at 16 | A/B at 32 | A/B at 64 | Random p95 at 32 |
|---|---:|---:|---:|---:|---:|
| fineweb | 0.812 | 0.772/0.772 | 0.809/0.811 | 0.808/0.808 | 0.653 |
| wikitext | 0.746 | 0.605/0.605 | 0.720/0.722 | 0.723/0.723 | 0.465 |
| tinystories | 0.808 | 0.762/0.762 | 0.811/0.811 | 0.793/0.793 | 0.681 |

Stable function with unstable exact edges would identify an equivalence class of short interaction programs rather than one privileged circuit graph. Stable edges and function would justify inspecting individual coordinate pairs semantically. Failure against the random null would retract the sparse-graph claim.
