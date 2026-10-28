# Label-blind weight-circuit atlas

Global-cluster tests and local motifs were frozen from symmetry-invariant random-function evaluations of each native attention head before any token strings or FineWeb contexts were decoded.

## Locked hypotheses

- **U1_pairwise_geometry_replicates: supported.**
- **U2_global_clusters_are_stable: not supported.**
- **U3_local_neighborhoods_replicate: supported.**
- **U4_local_motifs_cross_layers: supported.**
- **U5_route_evidence_is_not_single_head: supported.**
- **U6_reader_evidence_is_concentrated: not supported.**

## Partition

| k | sketch ARI | own-centroid cosine | centroid advantage | minimum size |
|---:|---:|---:|---:|---:|
| 12 | 0.063 | 0.285 | 0.251 | 6 |
| 18 | 0.070 | 0.345 | 0.307 | 4 |
| 24 | 0.113 | 0.393 | 0.353 | 1 |
| 36 | 0.153 | 0.475 | 0.436 | 1 |

No global partition passed stability (best ARI `0.153`). The frozen objects below are connected components of top-3 mutual-neighbor edges recurring in both sketches.

| motif | size | layers | medoid | stable edges | heads in natural evidence | reader support | input-mode token hints |
|---:|---:|---|---|---:|---:|---:|---|
| 0 | 9 | 1,3,4,9,10,15,17 | L1H1 | 8 | 9 | 57.8 | `�`, `ocry`, `cffff`, `ocl`, `о�` |
| 1 | 8 | 4,5,6,7,11,13,14,16 | L4H5 | 7 | 8 | 53.9 | ` economists`, ` experts`, ` regulators`, ` psychiatrists`, ` psychologists` |
| 2 | 6 | 7,13,14,15,16 | L7H0 | 6 | 6 | 33.4 | ` develops`, ` fitted`, ` ac`, ` advanced`, ` OEM` |
| 3 | 6 | 10,11,12,13,16 | L10H8 | 5 | 6 | 37.5 | `zik`, `hani`, ` ADC`, ` Investigator`, ` Alliance` |
| 4 | 5 | 5,8,10,11,16 | L5H5 | 5 | 5 | 31.8 | ` Canaver`, ` --------`, ` 4090`, ` }}`, ` 426` |
| 5 | 5 | 5,6,9,10 | L5H0 | 5 | 5 | 34.1 | `.")`, ` POLITICO`, ` quart`, ` disparate`, `993` |
| 6 | 4 | 7,8,10,15 | L7H3 | 5 | 4 | 24.5 | ` Food`, ` Has`, ` photos`, ` G`, ` Art` |
| 7 | 3 | 0,1,2 | L0H7 | 2 | 3 | 21.4 | ` additionally`, ` likewise`, ` residing`, ` utilized`, ` utilizing` |
| 8 | 3 | 2,3 | L2H5 | 3 | 3 | 22.2 | ` Millennials`, `TYPE`, ` Ivanka`, ` Roku`, ` Daughter` |
| 9 | 3 | 6,8 | L6H5 | 2 | 3 | 20.8 | `SourceFile`, `wcsstore`, `E`, `doesn`, `Dream` |
| 10 | 3 | 8,9,10 | L8H1 | 2 | 3 | 19.4 | ` brilliantly`, ` wonderfully`, ` Fortunately`, ` welcome`, ` hopefully` |
| 11 | 3 | 1,2,3 | L1H3 | 2 | 3 | 23.2 | `ammu`, ` Newfoundland`, ` Trinidad`, ` Dungeons`, ` Founder` |
| 12 | 3 | 3,4 | L3H1 | 2 | 3 | 23.4 | ` constitution`, ` difficulties`, ` domains`, ` expire`, `inese` |
| 13 | 2 | 9,12 | L9H6 | 1 | 2 | 10.0 | `hip`, ` Petro`, ` Weiner`, ` Manziel`, ` statute` |
| 14 | 2 | 1,2 | L1H5 | 1 | 2 | 14.8 | ` There`, ` has`, `There`, ` had`, ` having` |
| 15 | 2 | 11,12 | L11H1 | 1 | 2 | 15.7 | ` deteriorating`, ` destroyed`, ` discontinued`, ` lately`, ` disappointing` |

## Interpretation guardrails

- Token hints come from only the medoid's leading OV mode; they do not name the whole cluster.
- High-energy natural edges show invocation, not task-level necessity. Finite edge removal is still required.
- A recursive-generalization reading must identify a local semantic bridge among apparent branches and predict held-out contexts/readers. A broad label invented after viewing all examples is not a pass.
- Without training checkpoints, branch-growth order cannot be established; positive results are only consistent with recursive generalization.
