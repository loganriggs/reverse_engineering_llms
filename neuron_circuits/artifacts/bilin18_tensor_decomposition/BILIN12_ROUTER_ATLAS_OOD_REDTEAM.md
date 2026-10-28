# OOD red team of compact-router semantics

Every axis/edge label and polarity is frozen from the FineWeb atlas, then evaluated without reselection on all sites from WikiText-103 and TinyStories. Random rotations and random edges retain their own FineWeb-selected labels as matched controls.

## Locked hypotheses

- **S0_edge_semantics_transfer_ood: supported.**
- **S1_most_edges_transfer_ood: not supported.**
- **S2_selected_edges_beat_random_edges_ood: supported.**
- **S3_axis_semantics_transfer_ood: supported.**
- **S4_semantic_labels_are_not_all_one_predicate: supported.**

| Corpus | Sites | Axis/random median AUROC | Edge/random median AUROC | Edges >=0.80 |
|---|---:|---:|---:|---:|
| wikitext | 9600 | 0.775/0.905 | 0.931/0.780 | 27/32 |
| tinystories | 9600 | 0.814/0.977 | 0.960/0.787 | 26/29 |

OOD transfer supports a stable descriptive predicate, not a complete mechanistic meaning. If many edges share the same label, the result is one distributed boundary code with multiple output interactions—not 32 independent semantic relations.
