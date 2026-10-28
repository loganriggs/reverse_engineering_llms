# Shared sparse descriptive dictionary for Layer-16 loss-path effects

One 128-atom dictionary is trained on FineWeb 32D loss-path vectors with four-sparse OMP codes, selected on disjoint validation, and frozen for held-out FineWeb/WikiText/TinyStories. Codes are fitted after seeing each target, so this tests a shared descriptive language—not an executable predictor or native circuit.

## Locked hypotheses

- **D0_target_collection_is_numerically_exact: supported.**
- **D1_shared_pca_rank8_is_sufficient: not supported.**
- **D2_shared_dictionary_four_atoms_are_sufficient: not supported.**
- **D3_learned_dictionary_beats_random: not supported.**
- **D4_learned_dictionary_beats_prototypes: not supported.**
- **D5_shared_dictionary_transfers_ood: supported.**
- **D6_shared_dictionary_approaches_per_context_span: not supported.**
- **D7_validation_selects_noninitial_dictionary: supported.**

Autograd error: `6.27e-07`; channel-sum error: `4.30e-07`. Best MOD iteration: `3`; validation fidelity `0.691` versus initial `0.662`.

| Corpus | PCA R4 / R8 / R16 | Dictionary S2 / S4 / S8 | Prototype S4 | Random S4 p95 |
|---|---:|---:|---:|---:|
| fineweb | 0.278 / 0.402 / 0.621 | 0.482 / 0.691 / 0.896 | 0.698 | 0.667 |
| wikitext | 0.230 / 0.379 / 0.599 | 0.454 / 0.652 / 0.859 | 0.635 | 0.636 |
| tinystories | 0.314 / 0.435 / 0.637 | 0.512 / 0.698 / 0.887 | 0.719 | 0.653 |

PCA asks for one low-dimensional linear subspace; the overcomplete dictionary asks whether different contexts can select a few atoms from one shared vocabulary. Random and prototype controls are essential because four freely fitted atoms in 32 dimensions can look deceptively strong even without learned structure.
