# Relative-depth versus head-identity stability

This post-hoc analysis marginalizes the four independent discovery-panel edge scores over either heads or relative block offsets. It asks which axis supplies the stable part of the weak native head scaffold.

| Source | Offset correlation median/min | Head correlation median/min | Offset order | Head order | Panels with offsets {1,2} top-two |
|---:|---:|---:|---|---|---:|
| 0 | 0.850/0.664 | 0.329/-0.065 | 1, 2, 8, 4, 3, 5, 7, 6 | 3, 1, 8, 5, 6, 0, 7, 4, 2 | 1.00 |
| 1 | 0.646/0.209 | 0.353/0.083 | 1, 2, 3, 7, 4, 8, 5, 6 | 3, 8, 7, 6, 5, 1, 0, 2, 4 | 0.75 |
| 2 | 0.559/0.017 | 0.489/0.189 | 1, 2, 3, 6, 7, 4, 5, 8 | 3, 8, 5, 1, 7, 0, 6, 4, 2 | 0.75 |
| 3 | 0.568/0.152 | 0.699/0.283 | 1, 2, 5, 8, 3, 6, 4, 7 | 3, 8, 1, 5, 6, 7, 2, 0, 4 | 0.75 |

## Interpretation

Relative depth is substantially more reproducible than head identity at source Layers 0--2; the first two downstream blocks dominate the averaged ordering. At source Layer 3, head-axis stability rises, but the frozen edge advantage over random half-head programs remains small. This supports a coarse near-source routing grammar rather than a universal list of semantic heads.

This marginal analysis does not establish additive credit: leave-one-out scores can move under redundancy, and averaging hides feature-specific heads. Its role is to identify the shortest stable coordinate system for subsequent causal tests.
