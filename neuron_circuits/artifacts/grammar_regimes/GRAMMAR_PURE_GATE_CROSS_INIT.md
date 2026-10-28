# Functional alignment of pure-rule gate banks across initialization

All models share one manifest, fact table, tokenizer, and answer codebook. Native final-block channels are represented by their exact answer-space effect on removing the natural Layer-0 MLP state across common prompts. A discovery-only greedy one-to-one functional match is evaluated on disjoint prompts. Functional subspaces are compared directly in the shared prompt×answer effect space.

## Locked hypotheses

- **P0_complete_natural_effect_recurs: not supported.**
- **P1_functional_effect_subspace_recurs: not supported.**
- **P2_individual_native_verbs_recur: not supported.**
- **P3_native_coalition_selector_recurs: not supported.**
- **P4_equivalence_class_is_more_stable_than_channels: not supported.**
- **P5_functional_alignment_tracks_interface_fidelity: not supported.**
- **P6_channel_matching_transfers: supported.**

| Seed | Gate | fixed/transfer | Full effect cos | Subspace 16/32/64/128 | Channel cos disc/eval | Random p95 | Site/fixed top32 overlap |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 101 | pass | 1.000/0.989 | 0.161 | 0.034/0.067/0.127/0.229 | 0.126/0.025 | 0.002 | 0.167/0.188 |
| 102 | fail | 0.999/0.857 | -0.011 | 0.034/0.052/0.104/0.208 | 0.112/0.007 | 0.003 | 0.153/0.125 |
| 103 | pass | 1.000/0.992 | 0.157 | 0.027/0.049/0.112/0.231 | 0.114/0.042 | 0.003 | 0.208/0.219 |

Gate-failing models are retained as exploratory controls, not counted as matched-interface replications. Greedy channel matching is deliberately weaker than an optimal assignment; failure can identify noncanonical native coordinates but cannot by itself falsify a shared distributed effect space.
