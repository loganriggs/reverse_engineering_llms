# Read-function audit of dynamic sequence fields

Four previously unused FineWeb rows (indices 19000--19999) and three new destination heads were fixed before execution. Every factorized chart uses exactly 128 runtime coefficients: p positional functions times 128/p suffix-Fisher residual directions. The unrestricted joint rank-128 chart and collapsed residual rank-1 chart are matched 128-coordinate references. Charts use native state, weights, and the output Fisher metric but never the candidate replacement tangents.

## Locked hypotheses

- **R1_known_reads_are_early_useful: not supported.**
- **R2_known_reads_beat_random: supported.**
- **R3_learned_positions_beat_random: supported.**
- **R4_position_chart_is_sampling_stable: supported.**
- **R5_unrestricted_joint_has_unexplained_structure: supported.**
- **R6_some_known_read_approaches_joint: not supported.**
- **attention_jvp_forward_invariant: supported.**

## Causal results

| Target | Boundary | Joint | Collapsed | Current | Current+previous | Recent4 | Recent8 | Segment | Prefix | Fisher-pos | Random-pos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L0H7 | 6 | 0.883 | 0.049 | 0.551 | 0.500 | 0.434 | 0.346 | 0.117 | 0.117 | 0.539 | 0.015 |
| L0H7 | 10 | 0.810 | 0.053 | 0.592 | 0.377 | 0.315 | 0.242 | 0.117 | 0.117 | 0.381 | 0.012 |
| L0H7 | 14 | 0.724 | 0.098 | 0.561 | 0.350 | 0.297 | 0.212 | 0.063 | 0.063 | 0.350 | 0.015 |
| L0H7 | 17 | 0.690 | 0.138 | 0.531 | 0.372 | 0.258 | 0.186 | 0.026 | 0.026 | 0.372 | -0.023 |
| L4H2 | 6 | 0.817 | 0.038 | 0.433 | 0.407 | 0.331 | 0.222 | 0.231 | 0.231 | 0.454 | 0.026 |
| L4H2 | 10 | 0.749 | 0.030 | 0.507 | 0.310 | 0.216 | 0.149 | 0.142 | 0.142 | 0.321 | 0.019 |
| L4H2 | 14 | 0.618 | 0.039 | 0.434 | 0.272 | 0.197 | 0.137 | 0.084 | 0.084 | 0.273 | 0.023 |
| L4H2 | 17 | 0.575 | 0.039 | 0.389 | 0.256 | 0.151 | 0.087 | -0.010 | -0.010 | 0.256 | 0.018 |
| L5H1 | 6 | 0.822 | 0.046 | 0.380 | 0.329 | 0.250 | 0.153 | 0.209 | 0.209 | 0.391 | 0.018 |
| L5H1 | 10 | 0.749 | 0.048 | 0.504 | 0.361 | 0.274 | 0.180 | 0.111 | 0.111 | 0.377 | 0.007 |
| L5H1 | 14 | 0.676 | 0.060 | 0.499 | 0.349 | 0.252 | 0.167 | 0.091 | 0.091 | 0.350 | 0.024 |
| L5H1 | 17 | 0.643 | 0.041 | 0.469 | 0.324 | 0.226 | 0.127 | 0.033 | 0.033 | 0.324 | 0.010 |

## Positional generator diagnostics

| Boundary | Median split overlap (rank 8) | Median readable coverage of Fisher position span | Median position rank90 |
|---:|---:|---:|---:|
| 6 | 0.994 | 0.883 | 8 |
| 10 | 0.961 | 0.795 | 4 |
| 14 | 0.923 | 0.757 | 4 |
| 17 | 1.000 | 0.500 | 3 |

## Guardrails

A readable position span paired with opaque Fisher residual directions tests only the read-function part of the ontology; it does not make the transported variable itself human-readable. The Fisher-position chart is a candidate-independent dynamic oracle, not a semantic label. The random control shares rank and residual directions but not causal positional structure. All methods are matched in runtime coefficients, not literal basis-storage cost. Negative results can mean nonseparable position/residual coupling rather than absence of reusable functions.
