# Five-seed SVHN data and capacity confirmation

All30fits and independent audits complete. Five fresh model seeds212001–212005, one fixed training/validation split and minibatch-order schedule. The previous development seed is not pooled. Three depth3 architectures: width32 shared stack, width32 occurrence tree, and width120 shared stack selected solely to approximate tree parameter count. All use identical Adam/RMSNorm/30epoch settings; checkpoint at8010updates retained. No official test use, best-epoch selection, augmentation, or convolution.

| Data | Model | Parameters | Mean final accuracy (range) | Mean final CE | Mean8010-update accuracy |
|---|---|---:|---:|---:|---:|
| standard | stack | 201024 | 82.07% (79.48–84.04) | 0.6216 | 82.07% |
| standard | tree | 792896 | 83.20% (82.48–84.28) | 0.6789 | 83.20% |
| standard | wide_stack | 796080 | 83.05% (82.58–83.66) | 0.8706 | 83.05% |
| extra | stack | 201024 | 87.26% (86.96–87.96) | 0.4454 | 84.38% |
| extra | tree | 792896 | 88.96% (88.52–89.58) | 0.4089 | 86.55% |
| extra | wide_stack | 796080 | 89.74% (89.54–90.02) | 0.4874 | 87.62% |

## Paired differences for every seed

| Seed | Standard tree−stack | Extra tree−stack | Extra tree−wide stack | Data×architecture interaction |
|---|---:|---:|---:|---:|
| 212001 | +3.54 | +0.86 | -0.80 | -2.68 |
| 212002 | +1.34 | +1.46 | -1.50 | +0.12 |
| 212003 | +0.24 | +2.04 | -0.54 | +1.80 |
| 212004 | -0.62 | +1.90 | -1.02 | +2.52 |
| 212005 | +1.14 | +2.22 | -0.04 | +1.08 |

All differences above are percentage points, with final accuracy as the endpoint. The JSON also retains matched-update comparisons, cross-entropies, full per-seed metrics and initialization output/gradient statistics.

## What the confirmation changes

With extra data, the tree beats the narrow shared stack in accuracy in all five seeds (mean +1.70 percentage points), but the parameter-matched wide shared stack beats the tree in all five (mean +0.78 points). The wide stack also beats the tree at the matched 8010-update checkpoint in all five seeds (mean +1.07 points). This does not support an intrinsic accuracy advantage from untying: allocating a comparable parameter budget to shared width is competitive and better here. Width and initial functions differ, so this is a resource-allocation comparison, not a pure causal estimate of tying.

The metric choice matters: the extra-data tree has lower cross-entropy than the wide stack in all five seeds (mean .4089 versus .4874), despite lower accuracy. Cross-entropy also depends on confidence and the severity of mistakes; these numbers alone do not identify miscalibration as the cause. Standard-data tree versus narrow-stack accuracy is positive in four of five seeds; the data-by-architecture interaction is positive in four of five and averages only +.57 points. The earlier single-seed ranking reversal is therefore not a robust description of this panel.

Decision: close this prespecified panel without adding selected seeds or changing hyperparameters. Keep both accuracy and cross-entropy in future optimizer comparisons. The next learning-dynamics experiment should use paired synthetic initial functions and explicit work budgets to separate conditioning from architecture allocation; image accuracy does not establish recovery of reusable computations.

Interpretation limits: the equal-width tree has more parameters; the wide stack controls approximate parameter count but changes width, initial function and possible function class. Standard30epochs has8010updates, extra30epochs70260updates. Extra data changes distribution as well as example count. Five initialization seeds on one split do not establish dataset-level generality or a confidence interval for new tasks. No circuit recovery is inferred from classification accuracy.

Independent checks replay final full-training/validation metrics and checkpoint validation logits using NumPy; verify split/data hashes, parameter budgets, seeded initialization statistics, and equal-width initial functions. Any runtime failures remain in logs and panel status records rather than being treated as optimization failures.
