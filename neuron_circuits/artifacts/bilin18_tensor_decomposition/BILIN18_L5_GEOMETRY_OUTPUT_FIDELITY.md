## H765 — Local sensitivity weighting does not consistently improve actual output KL

Frozen H764 Fisher-weighted geometries are tested on64 fixed natural validation documents crossed with four prefix lengths16,32,64,120:256contexts. Each head activation is replaced by its reconstruction at final-position L5H5. Both ridge strengths, fixed/adaptive selection, budgets8–128, write-PCA baselines, three random adaptive bases and three scalar8 models are retained. No metric or model is selected on these outcomes. Observed-token CE is measured separately from KL(native||reconstructed).

| Representation | Mean actual KL at8 | At32 | At64 |
|---|---:|---:|---:|
| Fixed write PCA | 0.001416 | 0.000443 | 0.000187 |
| Fixed Fisher, ridge0.01 | 0.001417 | 0.000526 | 0.000161 |
| Fixed Fisher, ridge0.1 | 0.001404 | 0.000521 | 0.000162 |
| Adaptive write PCA | 0.001183 | 0.000224 | 0.0000251 |
| Adaptive Fisher, ridge0.01 | 0.001147 | 0.000214 | 0.0000369 |
| Adaptive Fisher, ridge0.1 | 0.001460 | 0.000189 | 0.0000349 |

Units are nats. There is no consistent sensitivity-weighting advantage. Fixed Fisher is approximately tied at8, worse at32, and has a lower mean at64. Adaptive results vary with ridge and budget; at64, both Fisher means are higher than write PCA despite its very small absolute KL. The local metric is therefore not validated as a generally better fidelity objective by this experiment.

A paired10,000-draw bootstrap resamples documents, averaging their four contexts within each draw. Most Fisher-minus-write-PCA KL intervals include zero. Fixed rank32 comparisons favor write PCA under these descriptive intervals; adaptive rank64 ridge0.01 also favors write PCA. These are uncorrected multiple comparisons on reused validation data and omit uncertainty in the Fisher estimator; they are not a confirmatory significance claim. Full comparisons and intervals are saved.

Scalar8 mean KL is0.001070,0.000737,0.000787 across the three seeds—lower than the fixed rank8 geometries, with different support flexibility and parameterization. Their observed-token CE changes are+0.00262,-0.00289,+0.00125nats. CE changes can have either sign and do not rank faithfulness in the same way as KL. No CE improvement or generalization claim follows from these small sample means.

Possible explanations for the Fisher result include residual sampling error, context-specific sensitivity averaged into a single matrix, sensitivity/activation correlations ignored by that approximation, or nonlinearity at reconstruction-sized perturbations. This experiment does not discriminate them. Correct gradient implementation alone cannot guarantee the surrogate captures the desired finite intervention.

Audits:12,800 forwards =256native references +12,544 reconstructions/controls. All evaluated documents are in the existing256-document validation split, never the original reserved test. Native/full-rank reconstructions replay logits exactly; cached natural capture error<=3.43e-5; all hashes unchanged and outputs finite. Natural document pool26,112 unchanged.

Next: complete the planned copying causal comparison with both frozen ridge settings, then distinguish context-averaging from finite-step effects if the metric remains inadequate. Do not adopt a metric merely because its construction refers to KL.
