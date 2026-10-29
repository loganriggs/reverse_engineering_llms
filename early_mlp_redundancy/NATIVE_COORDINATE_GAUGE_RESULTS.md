# Distributed coordinates versus intrinsically multidimensional computation

M103, 2026-10-29. Follow-up to the [random-subspace comparison](NATIVE_RANDOM_SUBSPACE_RESULTS.md). The question is whether a useful subspace establishes useful individual coordinates, and what survives a change of coordinates.

## Theory and interventions

Fix the Euclidean PCA64 input span of MLP1 and rotate its orthonormal basis B by Q. The projector BBᵀ and complete mixed-interaction candidate stay identical. For a recipient z and donor difference delta, let d = BBᵀ delta and decompose d into coordinate displacements d_i.

Two different quantities matter:

- Additive attribution: c_i = J(z+d/2)d_i. Because the MLP write F is quadratic, the c_i sum exactly to F(z+d)−F(z).
- Individual finite interventions: r_i = F(z+d_i)−F(z) = J(z)d_i+F(d_i). Their sum differs from the combined intervention by F(d)−sum_i F(d_i), the cross-coordinate quadratic interaction remainder.

Rotating coordinates keeps the combined intervention fixed but changes the individual interventions. The invariance of the sum therefore does not imply invariance of coordinate causal roles. These diagnostics concern input coordinates; native product channels are unchanged.

For contributions v_i, participation is `(sum_i ||v_i||²)² / sum_i ||v_i||⁴`: an effective number of contributing coordinates, not a removable-neuron count. It depends on both the displacement and the operator.

## Known-mechanism toys

Twelve random bilinear systems pass independent symmetric-tensor checks. A fixed linear computation can have participation 1 in an aligned basis and 8 in a rotated description. For F(x,y)=xy at the origin and combined displacement (1,1), the standard axes have zero individual finite effects and interaction remainder 1. A diagonal basis puts the entire displacement into one coordinate and the interaction remainder becomes zero. The function and combined intervention are unchanged.

This is why 'distributed' or 'interactive' needs an explicit reference to coordinates, native channels, or invariant function geometry.

## Fixed native test

Use 64 arbitrary natural-input differences selected by a frozen evenly spaced rule from the existing 4,092 unique training positions. These are not semantic matched pairs. Evaluate the PCA axes and three fixed Gaussian-QR rotations (10301–10303), all retained. No task holdouts, fitting, seed selection, or downstream Transformer runs are used.

All 256 conditions pass direct finite-write, additive-sum, interaction-remainder and mixed-function invariance checks. Maximum scaled discrepancy is 1.21e-13. An independently implemented NumPy factor calculation checks eight prespecified conditions, with maximum discrepancy 2.78e-15; it is not a full independent replay of every native condition.

| Measurement, mean across 64 changes unless stated | PCA axes | Three rotated bases |
|---|---:|---:|
| Input-displacement participation | 11.95 | 22.65–23.60 |
| Additive response participation | 13.92 | 21.53–22.85 |
| Individual finite-response participation | 14.00 | 21.39–22.80 |
| Pooled interaction-remainder energy / combined-response energy | 0.4724 | 0.5158–0.5255 |
| Pooled additive net energy / summed contribution energies | 1.6513 | 1.4508–1.4722 |

The net/sum ratios above 1 imply net constructive cross terms in this coordinate attribution, aggregated over these cases. They do not establish native-channel constructiveness or exclude individual cancelling pairs. The interaction remainder ratio is a squared error ratio; it is not a fraction of uniquely assignable computation.

Much of the change in response participation accompanies the change in input participation. This experiment does not isolate a special computational alignment of PCA axes beyond their data-variance alignment.

## A coordinate-invariant positive finding

Measure the local restricted operator J(h)B, where h=z+d/2, with equal weight on unit-length input directions. Its singular values are invariant to B→BQ. The effective operator dimension is `(sum sigma²)² / sum sigma⁴`.

Across these 64 contexts, effective dimension averages **45.84 out of the 64-dimensional span**, ranging from 29.05 to 52.71. Averaging the input Gram matrices across contexts gives effective dimension **57.89**. The best 16 shared input directions of that pooled Gram matrix still leave **64.16%** of its operator energy unexplained; the best four leave 88.83%.

All local spectra are verified against a separate NumPy symmetric eigensolver and recomputed under all three rotations (maximum relative discrepancy 2.16e-14). This is a derivative-energy measurement within the selected span, with isotropic direction weighting. It does not measure the full 1,152-dimensional operator, actual task-change fidelity, or semantic dimension. Pooling contexts can increase effective dimension because different contexts emphasize different directions; it is not proof of one common rank-58 program.

## Interpretation and next decision

The local response to one observed change can look concentrated in roughly 14 coordinates even though the operator supports many more independent response directions. **Displacement sparsity, operator dimension, and native-channel redundancy are separate properties.** Coordinate rotation changes the first description while preserving the operator spectrum and complete candidate computation.

This weakens an inference from concentrated PCA-coordinate effects to a small set of self-contained computational units. Together with M101–M102, a plausible working description is a useful context span modulating broader inputs, with substantial multidimensional sensitivity inside that span. It is not yet a semantic ontology, explanation of optimization history, or evidence that these are the uniquely correct units.

The diagnostic is now available for evaluating featurizers: report invariant within-block operator spectra alongside coordinate participation, then require held-out causal specificity before treating coordinates or blocks as reusable units. Do not keep searching rotations until localization looks attractive.

The next bounded comparison should check whether the same distinction holds across early MLPs, using matched data and metrics and consulting existing MLP0/MLP2 evidence first. This directly addresses whether layers differ in computational organization and prevents attributing a basis effect to a layer-specific strategy. It should produce a layer-by-layer constraint on featurizer design, rather than another unconstrained search for a compact representation.

## Reproducible evidence

- [Frozen native protocol](validation/native_coordinate_gauge_protocol.json)
- [Native measurements and checks](validation/native_coordinate_gauge.json)
- [Independent tensor toys](validation/coordinate_gauge_toys.json)
- [Frozen operator-spectrum protocol](validation/native_gauge_spectrum_protocol.json)
- [Operator spectra and rotation checks](validation/native_gauge_spectrum.json)
