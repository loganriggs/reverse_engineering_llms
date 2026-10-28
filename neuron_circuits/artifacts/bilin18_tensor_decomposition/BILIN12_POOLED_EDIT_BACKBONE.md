# Pooled late edit backbone

One write basis and one native-channel ordering are learned from four FineWeb token replacement operations and frozen before evaluation.

## Locked hypotheses

- **G0_pooled_rank32_matches_target_specific: supported.**
- **G1_pooled_rank64_is_broadly_sufficient: not supported.**
- **G2_common_channel_bank_matches_target_specific: supported.**
- **G3_pooled_backbone_beats_random: supported.**
- **G4_one_shared_native_bank_is_useful: not supported.**

| Target | Corpus | Pooled/target/random rank-32 fidelity | Pooled rank-64 | Pooled/target/random top-128 fidelity | Pooled top-256 |
|---|---|---:|---:|---:|---:|
| `\n` | fineweb | 0.881/0.883/-0.017 | 0.887 | 0.825/0.836/0.016 | 0.850 |
| `\n` | wikitext | 0.817/0.812/-0.007 | 0.849 | 0.771/0.766/0.020 | 0.790 |
| `\n` | tinystories | 0.873/0.870/0.032 | 0.890 | 0.826/0.825/0.017 | 0.851 |
| `.` | fineweb | 0.662/0.661/0.048 | 0.702 | 0.569/0.567/0.014 | 0.600 |
| `.` | wikitext | 0.706/0.713/0.012 | 0.757 | 0.613/0.620/0.018 | 0.657 |
| `.` | tinystories | 0.714/0.711/0.036 | 0.752 | 0.633/0.638/0.012 | 0.661 |
| `,` | fineweb | 0.484/0.487/0.043 | 0.538 | 0.386/0.379/0.019 | 0.416 |
| `,` | wikitext | 0.599/0.603/0.035 | 0.646 | 0.506/0.505/0.021 | 0.551 |
| `,` | tinystories | 0.583/0.561/0.011 | 0.649 | 0.483/0.493/0.021 | 0.516 |
| ` the` | fineweb | 0.379/0.391/0.023 | 0.455 | 0.283/0.283/0.012 | 0.331 |
| ` the` | wikitext | 0.507/0.509/0.004 | 0.583 | 0.394/0.402/0.017 | 0.440 |
| ` the` | tinystories | 0.547/0.525/0.019 | 0.593 | 0.351/0.348/0.020 | 0.400 |

A pooled success identifies a shared actuator space and implementation bank. It does not imply that the operations share upstream evidence or that their coefficient vectors have the same semantics.
