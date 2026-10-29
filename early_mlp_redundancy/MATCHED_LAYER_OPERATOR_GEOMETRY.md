# Early MLPs: concentrated changes, broad local operators

M104, 2026-10-29. A matched extension of [M103](NATIVE_COORDINATE_GAUGE_RESULTS.md) to MLP0, MLP1 and MLP2 (zero-based indices).

## Question and theoretical motivation

A particular input difference can be concentrated in a few coordinates even when the function responds independently along many other directions. Conversely, a high-dimensional local operator does not imply all those directions are exercised by a task. We therefore measure the sampled displacement and the operator separately, using identical token-index pairs across layers.

For each layer, fit an orthonormal PCA64 input basis B on the same natural training positions. For a pair of native normalized inputs z and z+delta, let d=BBᵀdelta and h=z+d/2. Since F is bilinear/quadratic, the combined write change is exactly J(h)d. Decompose d into PCA-coordinate displacements d_i. The response contributions J(h)d_i sum exactly to that change.

Coordinate participation measures how many contribution energies materially participate in that one change. The effective dimension of J(h)B instead measures the spread of squared singular values under equal weighting of unit input directions. It is invariant to rotations inside B and does not use the sampled displacement as a direction weighting. A gap between these quantities is expected when the input distribution selects a narrow set of directions from a broader operator.

## Matched data and verification

The capture used 32 native forwards on the existing natural training documents. All 96 layer/document product writes exactly replay their historical cache. MLP1 normalized inputs, weights and mean exactly replay the M77 reference. Training/evaluation document hashes remain disjoint; no held-out documents or task holdouts were accessed.

Each PCA fit uses the same 4,092 unique-prefix positions. Each layer uses the same 64 token-index pairs as M103, but its own observed normalized states. These are arbitrary natural-input differences, not semantically controlled task contrasts. The PCA bases have equal rank but retain different input variance; perturbation norms are not matched across layers. Projected donor states and midpoints are algebraic probes and need not be natural normalized states.

All 192 layer/case conditions pass direct finite-change, additive attribution and interaction-remainder checks. Maximum scaled discrepancy is 2.41e-13. NumPy factor calculations independently check six prespecified conditions (2.69e-15 maximum discrepancy); this is not an independent full native replay. All 192 operator spectra agree between NumPy symmetric eigensolvers and Torch singular values (4.03e-14 maximum scaled discrepancy). MLP1 metrics and spectra reproduce M103 to numerical precision.

## Results

| Measurement | MLP0 | MLP1 | MLP2 |
|---|---:|---:|---:|
| Centered training variance captured by PCA64 | 48.57% | 53.37% | 51.53% |
| Mean input-displacement participation | 16.82 | 11.95 | 13.02 |
| Mean response-coordinate participation | 14.39 | 13.92 | 16.00 |
| Mean local operator effective dimension within PCA64 | 33.43 | 45.84 | 49.03 |
| Local operator dimension range | 13.56–50.19 | 29.05–52.71 | 39.03–51.91 |
| Pooled operator effective dimension | 51.63 | 57.89 | 58.45 |
| Energy remaining outside best 16 shared operator directions | 59.33% | 64.16% | 65.17% |

Participation and effective dimension use the same energy-concentration formula but apply to different objects. Neither is a numerical matrix rank, number of semantic features, fraction understood, or removable-channel count. The pooled operator Gram matrix averages local input-space Gram matrices; pooling can broaden the spectrum when different contexts emphasize different directions.

The measurements describe this panel; pairs share documents and there are no statistical significance or general population estimates. Equal rank is not equal retained variance, and the three layers implement different functions at different states. The local operator comparison does not measure downstream importance.

## Updated conceptual account

All three layers exhibit relatively concentrated sampled changes alongside broadly multidimensional operators inside their selected spans. MLP0 is more locally concentrated by this operator metric on this panel; MLP1 and MLP2 are broader. MLP1's particularly concentrated input differences do not imply a low-dimensional computational core.

This is compatible with, but does not itself establish, context-dependent modulation of broad computations. It does not identify semantic clusters or explain why optimization chose these representations. Differences in local spectra are constraints on an explanation, not an explanation of training history.

Keep the earlier [matched native-channel study](MATCHED_LAYER_MECHANISMS.md) separate: on its task panel, all three layers were net constructive, and generic subset-deletion theory accounted for much of their local deletion curves. MLP2 was less constructive and less consequential downstream in those conditions; it was not generally cancellation-dominated. The current operator experiment changes neither native channels nor that evidence.

Thus a useful description has several axes:

1. Which input changes the data or task actually exercises.
2. How broadly the local function can respond within a specified span.
3. How native channels implement that function and combine their writes.
4. Which deviations the downstream model uses or ignores.

An exclusive label such as 'redundant MLP1' or 'cancelling MLP2' collapses these distinct questions.

## Featurizer implications and stopping decision

A featurizer can exploit concentrated task changes without assuming the entire local MLP operator is low rank. Blocks should be evaluated for the task-conditioned changes they encode and their causal effects, alongside within-block operator spectra. A broad operator can coexist with useful selective groups; these measurements do not justify rejecting sparse featurizers wholesale.

Close this matched-geometry comparison here. The next step returns to the diverse-task/featurizer loop: inspect the existing learned blocks and task-response caches, then identify a specific missing test of whether a block captures reusable task-conditioned changes rather than merely reconstructing common writes. Candidate selection must use training families only; whole-task exclusions remain reserved. No additional rank, layer, or rotation search is warranted by this result alone.

## Evidence

- [Frozen matched protocol](validation/matched_layer_operator_protocol.json)
- [Capture and PCA receipt](validation/matched_layer_operator_inputs.json)
- [Frozen measurement execution](validation/matched_layer_operator_execution.json)
- [All measurements and numerical checks](validation/matched_layer_operators.json)
- [Capture implementation](prepare_matched_layer_inputs.py)
- [Measurement implementation](measure_matched_layer_operators.py)
