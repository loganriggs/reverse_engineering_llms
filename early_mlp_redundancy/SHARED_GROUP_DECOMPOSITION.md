# M48: separate gate deletion from shared-atom deletion

This returns to the eight frozen M38 shared-dictionary candidates after M41–M47 established stronger native context-sharing controls. No fitting or model selection.

A selected group contributes eight atom memberships. Some atoms are selected only by that gate (exclusive); others remain selected by another active gate (shared). With original coordinates and other gates fixed, removing the gate removes only exclusive atoms. Removing all its atom contributions also removes shared atoms. These are different interventions. Neither equals deleting native MLP channels.

For each group, selected = exclusive disjoint-union shared. In decoded vector space, delta_union = delta_exclusive + delta_shared. Atom-disjoint support does not imply orthogonal vectors; union energy includes twice their inner product. A negligible gate effect can therefore reflect protection by overlapping gates rather than an unimportant represented direction. This protection is an architectural identity, not a discovered native mechanism.

## Verified identities and measured occupancy

All eight candidates ×32groups ×4096natural evaluation positions satisfy exact set identities and gate-removal protection. Independent set-based checks and NumPy decodes cover eight fixed positions for every group/candidate (2048group cases,6144component decodes). Original featurizer supports and codes replay exactly. No native output interventions in this stage.

| Seed | Source objective | Grouping | Shared selected-membership fraction | Shared / union decoded energy | Active cases with no exclusive atoms |
|---|---|---|---:|---:|---:|
| 71 | intact_KL | random | 18.3% | 0.580 | 0 |
| 71 | intact_KL | coactivation | 21.7% | 0.685 | 22 |
| 71 | intact_KL_plus_native_response | random | 18.2% | 0.581 | 0 |
| 71 | intact_KL_plus_native_response | coactivation | 22.0% | 0.704 | 0 |
| 72 | intact_KL | random | 18.0% | 0.597 | 0 |
| 72 | intact_KL | coactivation | 22.0% | 0.715 | 5 |
| 72 | intact_KL_plus_native_response | random | 18.3% | 0.605 | 0 |
| 72 | intact_KL_plus_native_response | coactivation | 23.1% | 0.713 | 5 |

Each model has16384active group-position cases. Shared membership fractions are about18% for random groups and22–23% for learned groups. These counts are summed over active groups: an atom selected twice contributes two memberships, not two decoded copies. Likewise the energy columns sum group-specific intervention vectors and are not an additive partition of full reconstructed energy. Cross terms are explicitly retained in the result JSON.

Higher overlap or shared energy does not establish semantic reuse. The construction favors coactivation, and high-energy atoms can dominate several gates. Native effects, token-conditioned baselines and context-pairing controls are needed.

## Next native comparison

Freeze one group index per original input key using SHA256, identical across candidates; retain inactive selections as no-ops. For each candidate extract exclusive, shared, and combined vectors from its original encoding, then subtract each vector separately from the native MLP1 write with all downstream layers live. No reconstructed baseline replaces the native model. Compare both seeds/source objectives and random/learned groups; report actual energy, active positions, no-ops, task outputs and distributional changes. Group IDs across candidates are index-matched, not semantically aligned.

This measures the consequence of represented directions, not fidelity of an inferred native circuit or reuse across all consumers. Token/context pairing controls remain required before a stronger reuse claim. No gate rerouting, atom-count tuning, seed selection or new model fitting.

[Protocol](validation/shared_group_decomposition_protocol.json) · [All-group audit and measurements](validation/shared_group_decomposition.json)
