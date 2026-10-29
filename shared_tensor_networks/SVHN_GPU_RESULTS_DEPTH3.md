# SVHN extra-data GPU comparison

Four depth-3 fits completed on RTX 5080 in float32 with TF32 disabled. All final training/validation metrics and both final and 8,010-update validation predictions replay independently in NumPy. All models use width 32, per-node RMSNorm, Adam .001, batch 256, gradient clipping at 10, and the same model seed. Equal-width stack/tree functions agree at initialization; tree occurrences have independent parameters.

Standard training uses 68,257 examples; added-data training uses 599,388 (including all 531,131 extra examples). Both use the identical 5,000-example validation split and normalization fitted only on standard training. Official test remains unused. No augmentation or convolution.

| Data | Architecture | Parameters | Validation at 8,010 updates | Validation at 30 epochs | Final CE | Final train accuracy |
|---|---|---:|---:|---:|---:|---:|
| standard | stack | 201024 | 84.30% | 84.30% | 0.5464 | 89.55% |
| standard | tree | 792896 | 83.34% | 83.34% | 0.6708 | 93.60% |
| extra | stack | 201024 | 84.28% | 87.34% | 0.4442 | 94.45% |
| extra | tree | 792896 | 86.18% | 89.46% | 0.4063 | 97.03% |

![Learning curves](figures/svhn_gpu_depth3.png)

Standard 30 epochs equal 8,010 optimizer updates and 2,047,710 examples seen. Extra-data 30 epochs equal 70,260 updates and 17,981,640 examples seen. Its 8,010-update snapshot sees 2,050,068 examples (slightly more because incomplete epoch-end batches differ). This controls optimizer-update count, not exactly compute, distinct examples, or data distribution.

Interpretation: the equal-epoch comparison combines more distinct data with more training. The matched-update comparison constrains that confound but extra images also have a different distribution. This is one initialization on a fixed development split, not a seed-robust advantage or a pure sample-size causal estimate. Training accuracy is measured on different data pools and should not be directly compared as if tasks were identical. Accuracy does not establish circuit recovery.

Next: finish paired depth controls and replicate seeds where differences persist. Retain parameter-budget and width-allocation caveats from the previous standard-data studies. Do not infer convergence solely from completing 30 epochs.
