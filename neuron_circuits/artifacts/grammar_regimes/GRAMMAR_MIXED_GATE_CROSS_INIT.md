# Functional alignment of mixed gate banks across initialization

All models share one manifest, arbitrary fact table, tokenizer, and answer codebook. Native final-block channels are represented by their exact answer-space effect on removing the natural Layer-0 MLP state across common prompts. A discovery-only greedy one-to-one functional match is evaluated on disjoint prompts. Functional subspaces are compared directly in the shared prompt×answer effect space.

## Locked hypotheses

- **A0_complete_natural_effect_recurs: not supported.**
- **A1_functional_effect_subspace_recurs: not supported.**
- **A2_individual_native_verbs_recur: not supported.**
- **A3_native_coalition_selector_recurs: not supported.**
- **A4_equivalence_class_is_more_stable_than_channels: not supported.**
- **A5_functional_alignment_tracks_interface_fidelity: not supported.**
- **A6_channel_matching_transfers: supported.**

| Seed | Gate | fixed/transfer | Full effect cos | Subspace 16/32/64/128 | Channel cos disc/eval | Random p95 | Site/fixed top32 overlap |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 101 | pass | 0.986/0.961 | 0.019 | 0.009/0.019/0.038/0.077 | 0.074/0.006 | 0.002 | 0.165/0.156 |
| 102 | fail | 0.994/0.917 | 0.059 | 0.009/0.017/0.037/0.074 | 0.076/0.005 | 0.002 | 0.167/0.062 |
| 103 | fail | 0.994/0.894 | 0.005 | 0.010/0.021/0.038/0.075 | 0.074/-0.003 | 0.002 | 0.172/0.219 |
| 104 | pass | 0.974/0.953 | -0.009 | 0.008/0.018/0.038/0.077 | 0.078/-0.007 | 0.002 | 0.168/0.125 |
| 105 | fail | 0.993/0.940 | -0.011 | 0.007/0.016/0.034/0.071 | 0.075/0.001 | 0.002 | 0.175/0.250 |
| 106 | fail | 0.998/0.930 | 0.061 | 0.009/0.020/0.041/0.079 | 0.080/-0.001 | 0.002 | 0.161/0.031 |

Gate-failing models are retained as exploratory controls, not counted as matched-interface replications. Greedy channel matching is deliberately weaker than an optimal assignment; failure can identify noncanonical native coordinates but cannot by itself falsify a shared distributed effect space.
