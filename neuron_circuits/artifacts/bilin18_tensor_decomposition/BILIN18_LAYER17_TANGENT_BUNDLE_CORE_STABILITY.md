# Multilinear core stability inside the tangent-bundle quotient

Each discovery split produces a rank-64 output × tangent × context tensor. Separate HOSVD unfoldings define physical subspaces for the three tensor roles. The comparison is between subspaces, not individual sign/permutation-dependent axes.

## Locked hypotheses

- **C0_role_tensor_has_small_multilinear_rank: supported.** Every split's rank-64 role tensor has 90%-energy rank at most 32 in each of its output, tangent, and context unfoldings.
- **C1_top16_physical_core_is_stable: not supported.** Relative to the head split, every split's top-16 physical output, tangent, and context Tucker subspaces have projector overlap at least 0.70.
- **C2_tucker16_recovers_score_paths: supported.** Every split's Tucker-16 quotient retains at least 0.80 prediction and target fidelity on every external corpus.
- **C3_tucker32_is_score_sufficient: not supported.** Every split's Tucker-32 quotient retains at least 0.90 prediction and target fidelity on every external corpus.
- **C4_tucker32_recovers_margin: not supported.** Every split's Tucker-32 quotient retains at least 0.60 margin fidelity on every external corpus.
- **C5_tucker32_is_split_stable: supported.** Across split choices, every external endpoint's Tucker-32 fidelity range is at most 0.08.
- **C6_tucker_tensor_execution_is_exact: supported.** Direct execution of every split's contracted Tucker-32 tensor matches projected native CP execution to relative error at most 1e-5.
- **C7_tensor_axes_are_not_claimed_canonical: supported.** Results are reported as physical subspaces; no sign, permutation, or individual HOSVD-axis identity claim is made.

Maximum Tucker-32 tensor/native-CP error: `4.04e-07`.

## Spectra and physical-core stability

| Split | R90 output/tangent/context | top-16 overlap output/tangent/context |
|---|---:|---:|
| head | 15/17/16 | 1.000/1.000/1.000 |
| tail | 16/17/16 | 0.921/0.691/0.642 |
| interleaved | 16/17/17 | 0.948/0.777/0.699 |
| random | 15/17/16 | 0.948/0.751/0.692 |

## External Tucker fidelity

Each cell is prediction/target/margin for Tucker ranks 16 and 32.

| Split | Corpus | rank 16 | rank 32 |
|---|---|---:|---:|
| head | fineweb | 0.891/0.860/0.445 | 0.901/0.885/0.556 |
| head | wikitext | 0.921/0.901/0.433 | 0.939/0.925/0.555 |
| head | tinystories | 0.942/0.938/0.726 | 0.954/0.950/0.792 |
| tail | fineweb | 0.881/0.847/0.380 | 0.895/0.879/0.540 |
| tail | wikitext | 0.904/0.884/0.342 | 0.935/0.921/0.542 |
| tail | tinystories | 0.954/0.950/0.712 | 0.961/0.957/0.804 |
| interleaved | fineweb | 0.893/0.865/0.397 | 0.899/0.883/0.503 |
| interleaved | wikitext | 0.911/0.895/0.353 | 0.926/0.916/0.495 |
| interleaved | tinystories | 0.946/0.943/0.724 | 0.954/0.951/0.789 |
| random | fineweb | 0.894/0.862/0.408 | 0.905/0.889/0.521 |
| random | wikitext | 0.916/0.901/0.392 | 0.935/0.923/0.520 |
| random | tinystories | 0.946/0.943/0.738 | 0.955/0.951/0.798 |

A stable physical core would make the rank-64 basis instability a gauge/slack issue. If behavior remains stable but physical core overlap fails, canonical coordinates remain unidentified even though a compact function class exists.
