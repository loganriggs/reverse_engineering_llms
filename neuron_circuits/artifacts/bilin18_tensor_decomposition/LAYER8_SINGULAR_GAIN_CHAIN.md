# Gain chain of the singular ordered QK/value-bus path

The `[11,13,14]` head-6 QK2/value-bus path is propagated directly, avoiding Mobius subtraction at the final measurement. Stage gains compare the singular WikiText context with its matched query-1 control.

## Locked hypotheses

- **G0_upstream_pair_tangent_amplifies: not supported.**
- **G1_qk2_local_gain_amplifies: not supported.**
- **G2_value_bus_routing_gain_amplifies: not supported.**
- **G3_downstream_gain_amplifies: not supported.**
- **G4_direct_ordered_path_matches_mobius: supported.**

| Stage | Failure | Control | Failure/control |
|---|---:|---:|---:|
| Pair tangent arriving L14 | 2.844e-01 | 1.899e-01 | 1.50x |
| QK2 derivative gain | 9.670e-01 | 4.105e-01 | 2.36x |
| Value-bus routing gain | 7.199e+02 | 2.377e+02 | 3.03x |
| Layer-14 MLP gain | 1.017e+00 | 1.021e+00 | 1.00x |
| Downstream suffix gain | 2.229e-04 | 8.049e-05 | 2.77x |
| Final response norm | 4.487e-02 | 1.523e-03 | 29.46x |

Direct-to-Mobius errors: failure `1.399e-05`, control `3.672e-04`. Native head-6 QK2 raw RMS: failure `5.404e+00`, control `5.676e+00`.

A large upstream ratio means the singularity is created before the final head. A large local-gain ratio identifies QK normalization or value-bus readout. A large downstream ratio means a modest route is amplified after layer 14.
