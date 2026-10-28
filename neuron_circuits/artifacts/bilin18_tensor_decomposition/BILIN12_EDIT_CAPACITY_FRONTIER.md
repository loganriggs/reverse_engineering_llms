# Capacity frontier of the shared late edit backbone

One pooled write basis and one direct native-channel ranking are evaluated from compact to full size for four token edits and three corpora.

## Locked hypotheses

- **F0_rank256_is_universally_sufficient: not supported.**
- **F1_rank512_is_near_exact: not supported.**
- **F2_top1024_native_channels_are_sufficient: not supported.**
- **F3_operation_complexity_order_replicates: supported.**
- **F4_rank_frontier_beats_random_until_half_width: supported.**

| Target | Corpus | Min rank 0.80/0.90 | Min channels 0.80/0.90 | Rank 256/512 fidelity | Channels 1024/2048 fidelity |
|---|---|---:|---:|---:|---:|
| `\n` | fineweb | 32/128 | 128/512 | 0.938/0.980 | 0.939/0.980 |
| `\n` | wikitext | 32/256 | 256/1024 | 0.922/0.973 | 0.916/0.972 |
| `\n` | tinystories | 32/128 | 128/512 | 0.946/0.980 | 0.948/0.983 |
| `.` | fineweb | 256/384 | 1024/2048 | 0.843/0.949 | 0.843/0.949 |
| `.` | wikitext | 128/384 | 1024/2048 | 0.876/0.953 | 0.849/0.944 |
| `.` | tinystories | 256/384 | 512/2048 | 0.869/0.954 | 0.881/0.961 |
| `,` | fineweb | 384/512 | 2048/2048 | 0.779/0.915 | 0.753/0.919 |
| `,` | wikitext | 256/512 | 1024/2048 | 0.820/0.927 | 0.805/0.922 |
| `,` | tinystories | 256/512 | 1024/2048 | 0.823/0.938 | 0.822/0.945 |
| ` the` | fineweb | 384/768 | 2048/2048 | 0.724/0.891 | 0.719/0.900 |
| ` the` | wikitext | 384/512 | 2048/2048 | 0.779/0.912 | 0.733/0.908 |
| ` the` | tinystories | 384/512 | 2048/2048 | 0.793/0.925 | 0.788/0.943 |

The minimum is over the declared coarse grid, not an exact intrinsic dimension. Fidelity is behavior- and corpus-specific; there is no single model-wide number of necessary neurons.
