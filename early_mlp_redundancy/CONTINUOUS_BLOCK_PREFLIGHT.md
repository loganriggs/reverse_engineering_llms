# Continuous block shrinkage: verified candidate, unresolved fidelity tradeoff

M111 implements and verifies a continuous alternative to the hard Top8 block gate. It removes the discontinuities demonstrated in M110 while preserving the same parameter count and an at-most-eight-block budget. It has not yet been trained or evaluated for downstream causal fidelity.

## Definition and what it guarantees

For block coordinates a_g, let r_g=||a_g||2 and let tau be the ninth-largest r_g. Output

`z_g = a_g * max(r_g - tau, 0) / r_g`,

with z_g=0 when r_g=0. There are 32 groups of four coordinates and at most eight groups with nonzero output. Blocks tied at the threshold vanish, so fewer than eight may be active. The threshold remains in the autograd graph; detaching it would change the optimization objective.

The block norms are continuous, their order statistic tau is continuous, and radial soft thresholding is continuous jointly in its vector and threshold. At zero, ||z_g||≤||a_g||, so no singularity remains. Thus the entire decoded map is continuous, including selection ties, for fixed finite encoder/decoder weights. Differentiability is not guaranteed at ties, threshold crossings, or zero norms; this is continuity, not smoothness.

A conservative global Lipschitz bound for the coordinate map with G blocks is 1+sqrt(G). At fixed tau, block soft thresholding is nonexpansive. Changing tau moves each block output by at most the threshold change, and the order statistic changes by at most the maximum change of any block norm, itself bounded by the full input difference. The decoder and encoder operator norms multiply this bound. This is not a claim that the learned model has small sensitivity.

The map is positively homogeneous, odd, group-permutation equivariant, equivariant under orthogonal changes within each block, and idempotent. It adds no trainable parameters. The adapter retains all 296,192 parameters and the state-dictionary layout of the hard-block featurizer. The nonzero block budget matches, but amplitudes and gradients differ.

## Important edge cases

- All zero coordinates produce zero output and finite gradients.
- If all 32 block norms tie, every block vanishes. This is a real representational/training limitation, not a numerical workaround.
- Inputs with at most eight nonzero blocks pass through unchanged.
- The threshold block affects selected blocks’ gradients even though its own output is zero. The simple [3,2,1] norm example with k=1 has gradient [1,-1,0] for the summed output.
- The implementation avoids division by zero without adding a smoothing epsilon or a learned gain.

## Verification

Fifteen independent NumPy comparisons cover multiple batch shapes and sparsity budgets; maximum discrepancy4.45e-16. Three double-precision autograd gradchecks pass away from ties. Zero/tied/partly-zero cases have finite gradients. Permutation, rotation, homogeneity, idempotence and sparsity checks pass. The earlier two-block discontinuity toy now has across-boundary change2*sqrt(2)*h, tending to zero with distanceh. Six adapter checks confirm checkpoint compatibility, parameter equality, output shapes, support limits and finite parameter/input gradients.

All117 previously identified real-checkpoint boundary paths were evaluated at the same three shrinking distances. Continuous decoded differences obey the analytic Lipschitz bound; no finite selection jump remains. This is an internal-path implementation check, not semantic manifold validation.

## Cold substitution reveals an amplitude confound

We also applied the map to both endpoints of all56 existing whole-task pairs for each of the four frozen hard-gate-trained models. These448 endpoint calculations were independently checked using a NumPy sorted-norm formula. No fitting or native model forwards occurred.

| Original objective | Seed | Soft / hard code energy | Hard write error | Soft write error | Hard change error | Soft change error |
|---|---:|---:|---:|---:|---:|---:|
| KL | 71 | 0.445 | 0.384 | 0.441 | 1.201 | 0.897 |
| KL | 72 | 0.452 | 0.378 | 0.450 | 1.263 | 0.904 |
| KL + causal | 71 | 0.447 | 0.383 | 0.442 | 1.227 | 0.897 |
| KL + causal | 72 | 0.461 | 0.377 | 0.448 | 1.258 | 0.901 |

Errors are pooled squared error divided by corresponding native squared energy. Write errors include both endpoint vectors; change errors use their difference. Zero change prediction scores1. These are local geometric metrics, not output KL or task accuracy. The sparse shrinkage map retains only44.5–46.1% of the hard code energy: reduced change error can reflect attenuation, as already demonstrated in M65. The worse reconstruction and better difference error cannot establish that the new architecture is better or worse after training.

## Consequence for the conceptual question

Continuity is now an explicit, tested prior on the extracted computation, rather than an arbitrary rule intended only to improve seed agreement. It prevents one demonstrated source of artificial boundaries. It does not guarantee semantic features, useful manifolds, native causal equivalence, or reusable computations. Those require separate experiments.

The next comparison should train hard and continuous block models from identical initial states, with identical data, schedules, causal targets and parameter/block budgets. Reuse the established natural-KL plus encoded-response training runtime and whole-task exclusions; audit gradients through the threshold and causal loss before fitting. Retain all seeds and report reconstruction, amplitude, support cost, full-output fidelity, and held-out intervention-response fidelity separately. A comparison of untrained cold substitution is insufficient.

The existing M69 runner supplies a132-update alternating natural/task schedule and fixed training-only causal-loss calibration. A bounded next experiment can use hard/continuous × seeds71/72 from the same historical block starts, with unchanged calibration and schedule; hard controls must be rerun or demonstrably replayed rather than assuming comparability. This is a proposed next experiment, not a frozen or completed training result.

The map and adapter are continuous_block_shrinkage.py and continuous_write_featurizer.py. Checks and preflight results live in validation/continuous_block_shrinkage_checks.json, continuous_featurizer_adapter_checks.json, and continuous_block_preflight.json. No additional native-data fitting or task-label exposure occurred.
