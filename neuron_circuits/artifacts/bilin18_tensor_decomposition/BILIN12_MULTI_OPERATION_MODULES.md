# Multi-operation late write modules

Rank-32 write subspaces and top-128 native channel banks are learned separately for four token replacement operations on FineWeb and frozen before three-corpus evaluation.

## Locked hypotheses

- **M0_compact_write_modules_generalize_across_operations: not supported.**
- **M1_learned_modules_beat_random: supported.**
- **M2_write_modules_are_operation_specific: not supported.**
- **M3_native_channel_banks_are_operation_specific: not supported.**
- **M4_fixed_native_banks_are_broadly_useful: not supported.**

| Target | Corpus | Correct/wrong/random subspace fidelity | Correct/wrong/random channel fidelity |
|---|---|---:|---:|
| `\n` | fineweb | 0.883/0.859/0.015 | 0.836/0.818/0.029 |
| `\n` | wikitext | 0.812/0.805/0.014 | 0.768/0.767/0.026 |
| `\n` | tinystories | 0.870/0.847/0.020 | 0.819/0.814/0.023 |
| `.` | fineweb | 0.661/0.650/-0.002 | 0.571/0.566/0.051 |
| `.` | wikitext | 0.713/0.665/-0.011 | 0.629/0.622/0.047 |
| `.` | tinystories | 0.711/0.707/-0.023 | 0.621/0.608/0.052 |
| `,` | fineweb | 0.487/0.427/0.018 | 0.391/0.385/0.030 |
| `,` | wikitext | 0.603/0.542/0.014 | 0.509/0.507/0.028 |
| `,` | tinystories | 0.561/0.524/0.023 | 0.469/0.468/0.038 |
| ` the` | fineweb | 0.391/0.352/0.014 | 0.278/0.280/0.018 |
| ` the` | wikitext | 0.509/0.492/0.006 | 0.384/0.370/0.027 |
| ` the` | tinystories | 0.525/0.526/0.007 | 0.327/0.299/0.026 |

Mean pairwise subspace overlap: `0.636`; mean top-128 channel Jaccard: `0.751`.

Correct-vs-wrong comparisons test operation specificity. High cross-operation transfer would instead support one generic late edit interface shared by several token changes.
