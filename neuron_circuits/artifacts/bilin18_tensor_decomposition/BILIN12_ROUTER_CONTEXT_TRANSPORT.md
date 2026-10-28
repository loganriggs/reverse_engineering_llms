# Context-gated transport from Layer 6

A frozen linear transport map explained about four-fifths of Layer-6 perturbation effects. This test asks whether the missing variation is a compact bilinear rule: perturbation coordinates multiplied by a low-rank summary of the unperturbed Layer-6 state. Every map is fit on small FineWeb edits and tested on edits four times larger and on OOD corpora.

## Locked hypotheses

- **G0_global_transport_replicates: supported.**
- **G1_low_rank_context_materially_improves: not supported.**
- **G2_context_gating_explains_transport: not supported.**
- **G3_context_gating_predicts_graph_changes: not supported.**
- **G4_position_is_not_the_explanation: supported.**
- **G5_context_rule_transfers_ood: supported.**
- **G6_more_context_does_not_overfit_badly: supported.**

| Corpus | Global | Position | Context r8 | Context r16 | Context r32 | r16 edge fidelity |
|---|---:|---:|---:|---:|---:|---:|
| fineweb | 0.778 | 0.777 | 0.811 | 0.826 | 0.838 | 0.847 |
| wikitext | 0.789 | 0.788 | 0.819 | 0.829 | 0.844 | 0.843 |
| tinystories | 0.801 | 0.800 | 0.827 | 0.846 | 0.855 | 0.865 |

This is a deliberately restrictive conditional model. Success would identify a compact tensor-valued routing rule. Failure would mean that local transport depends on information outside the 32 late coordinates, on richer nonlinear context, or on attention/history variables not represented by the local residual state.
