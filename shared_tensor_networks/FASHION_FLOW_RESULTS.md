# Image-task sharing gradient-flow control

Completed9paired cases (depths2/3/4, three fresh initialization seeds), each with four arms: ordinary tied SGD, multiplicity-corrected tied SGD, explicit occurrence-projected SGD, and free untied SGD. Fashion-MNIST55,000training/5,000validation; width16, RMSNorm, float64, lr.01, no momentum/weightdecay/clipping, one epoch (215minibatches). This is a short dynamics control, not a replacement for longer training or a converged optimizer comparison. Official test unused.

Matched-width models start at exactly the same function. Corrected SGD divides each shared layer gradient by its occurrence count; projected SGD updates separate occurrences then averages same-level parameters. All four receive the same minibatches. Initial gradient measurements use256training examples; finite differences independently verify the predicted loss directional derivatives. No probe-dependent learning-rate selection.

## Results

| Depth | Initial disagreement energy fraction, range | Ordinary tied accuracy, median | Corrected/projected accuracy, median | Free tree accuracy, median |
|---|---:|---:|---:|---:|
| 2 | 36.2%–47.9% | 76.46% | 73.70% | 78.38% |
| 3 | 73.8%–77.6% | 76.70% | 67.02% | 76.52% |
| 4 | 83.8%–92.0% | 75.02% | 60.58% | 75.06% |

Disagreement fraction is ||(I-P)g||²/||g||² for the occurrence-space gradient g of mean cross-entropy. It is the fraction of instantaneous Euclidean gradient-flow loss decrease unavailable to the projected dynamics at that common starting state. It is not the fraction of model computation that is redundant, nor a prediction of generalization.

Across all depths the range is36.3–92.0% (correcting an interim commentary range which omitted depth2). It increases with depth in this small sample, without establishing a universal law.

Corrected tied and explicitly projected trajectories agree to maximum parameter discrepancy3.272e-13 at recorded steps. This extends the earlier synthetic SGD control to an actual image classifier with RMSNorm. Independent NumPy final-checkpoint replay verifies all36models' validation cross-entropy/accuracy and corrected/projected output agreement. All nine finite-difference controls and gradient energy decompositions pass. Code/source/checkpoints/curve records retained. Runtime used genericCPU dispatch pinned tocore0; no crash this run, which does not settle prior runtime failures.

Ordinary tied updates learn much faster than multiplicity-corrected tied updates at the same base step size. Free trees are also faster than the projected control. At depths3/4, ordinary tying and free trees finish near each other despite the large initial disagreement fraction: amplified shared-coordinate steps can offset the lost local directions under this recipe. This is consistent with the metric explanation but is not proof that it completely explains trajectories. Same base step size does not match function-space speed; finite steps and changing gradients matter. No universal convergence advantage or statistically established accuracy ranking is claimed from three seeds and one epoch.

## Next discriminating steps

Compare learning-rate panels fixed independently of validation, including a common initial function-speed normalization, to distinguish base-step choice from anisotropic multiplicity effects. Preserve direct projected/corrected equivalence as a correctness control. Measure disagreement on multiple training probes and along trajectories at common tied states; do not compare tangent kernels at unrelated trained states as if they shared a base point. Maintain full-data longer-training/seed replication and the separate reuse-complexity teacher experiments. The minimum-circuit and general recovery questions remain open.
