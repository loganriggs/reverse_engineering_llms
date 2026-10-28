# Finite causal transport of the late sparse interface

The frozen 32-dimensional final-layer basis was perturbed at residual boundaries after Layers 6 and 10. A 32-by-32 transport matrix was fit only on small FineWeb perturbations, then frozen and tested on four-times-larger edits, disjoint FineWeb, two OOD corpora, canonical coordinates, and random rotations of the same subspace.

## Locked hypotheses

- **F0_finite_trace_recomposes: supported.**
- **F1_late_hubs_are_causally_persistent: supported.**
- **F2_midlayer_hubs_are_causally_persistent: supported.**
- **F3_linear_transport_predicts_large_edits: not supported.**
- **F4_transport_predicts_graph_changes: supported.**
- **F5_canonical_hubs_are_privileged_carriers: not supported.**
- **F6_transport_is_stable_across_edit_size: supported.**

| Layer | Corpus | Hub direction cosine | Random direction cosine | Coordinate fidelity | Edge-change fidelity | Cross-scale fidelity |
|---:|---|---:|---:|---:|---:|---:|
| 6 | fineweb | 0.874 | 0.863 | 0.778 | 0.801 | 1.000 |
| 6 | wikitext | 0.882 | 0.873 | 0.789 | 0.800 | 1.000 |
| 6 | tinystories | 0.882 | 0.873 | 0.801 | 0.817 | 1.000 |
| 10 | fineweb | 0.994 | 0.992 | 0.999 | 0.999 | 1.000 |
| 10 | wikitext | 0.993 | 0.992 | 0.999 | 0.999 | 1.000 |
| 10 | tinystories | 0.994 | 0.991 | 0.999 | 0.999 | 1.000 |

The directional cosine asks whether a finite nudge to one proposed coordinate remains pointed along that coordinate downstream. Coordinate and edge fidelities ask whether one small-edit FineWeb transport map predicts substantially larger unseen edits. Random rotations test whether named graph hubs are special carriers, rather than arbitrary coordinates in a generally persistent subspace.
