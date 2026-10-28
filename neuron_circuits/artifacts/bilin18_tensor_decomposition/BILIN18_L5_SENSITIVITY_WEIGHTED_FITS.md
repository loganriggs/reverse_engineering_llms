## H771 — Measured training sensitivity gives tentative, inconsistent KL gains

Previous goal turn was progress: H770 rejected the tested activation-only gain predictors. H771 therefore uses measured training-only probe weights rather than extrapolating those failed predictors to the whole corpus.

All matched fits use the same 64 activations from 32 H769 training documents at lengths 32 and 120. The original natural training mean stays fixed. We compare uniform weights, measured scalar sensitivity from the first 12 training probes, and a fixed shuffle of those weights between documents (preserving each document's pair of lengths). For each, the weighted second moment is blended 50/50 with the large natural-training second moment. This shrinkage was fixed before results and shared across treatments.

We repeat the comparison in output-write geometry and the frozen H764 Fisher geometry with ridge .1. Each fitted complete basis is evaluated with fixed-prefix and adaptive TopK coordinate selection at budgets 8, 32, 64 and 128. Large-data original fits are retained as references. All fits are saved and hashed before any validation model output is evaluated. Sensitivity labels from validation contexts are never used to fit these representations.

Validation uses the same 16 H769 validation documents at two lengths, 32 contexts total. Replace the native final-position L5H5 activation by each reconstruction, leaving downstream computation live. Measure actual KL(native||reconstructed), observed-token CE change and write error separately. The table reports mean KL in nats for all nontrivial matched comparisons.

| Geometry | Selection | Budget | Unweighted | Measured weights | Shuffled weights |
|---|---|---:|---:|---:|---:|
| write | fixed | 8 | 0.000708992 | 0.000665267 | 0.000609034 |
| write | adaptive | 8 | 0.000440298 | 0.000342636 | 0.000546693 |
| write | fixed | 32 | 0.000422425 | 0.000353636 | 0.000544465 |
| write | adaptive | 32 | 0.00012323 | 0.00013263 | 8.95444e-05 |
| write | fixed | 64 | 7.07005e-05 | 6.72104e-05 | 6.53716e-05 |
| write | adaptive | 64 | 1.50622e-05 | 1.6094e-05 | 1.73824e-05 |
| fisher | fixed | 8 | 0.000592963 | 0.000593443 | 0.000629599 |
| fisher | adaptive | 8 | 0.000532298 | 0.000431638 | 0.000703085 |
| fisher | fixed | 32 | 0.000541033 | 0.000462378 | 0.000651983 |
| fisher | adaptive | 32 | 0.000185186 | 0.000140964 | 0.000126785 |
| fisher | fixed | 64 | 0.000112033 | 0.000111495 | 0.000120597 |
| fisher | adaptive | 64 | 3.38502e-05 | 2.1394e-05 | 2.72783e-05 |

Measured weights lower mean KL relative to uniform weights in 9 of the 12 nontrivial settings. This is a directional tendency, not 12 independent replications: all settings share data, fits and related subspaces. There is no uniform gain. For example, adaptive write geometry at budget 8 improves from .000440 to .000343, but at budget 32 worsens from .000123 to .000133. Shuffled weights sometimes do as well or better than measured weights, so a generic perturbation of the fitted covariance is a live alternative explanation.

Paired 10,000-draw bootstrap intervals resample the 16 validation documents and average the two lengths within each document. Eleven of twelve measured-minus-unweighted intervals include zero; only fixed write rank 32 has a wholly negative descriptive interval, barely excluding zero. Several measured-minus-shuffled comparisons favor measured weights, but most remain inconclusive. These intervals do not include training-estimator uncertainty and are uncorrected for multiple comparisons. Do not select a winning budget or metric using them.

The measured weights have effective sample size 35.3 contexts out of 64 and 23.2 documents out of 32. This reduction makes fitting uncertainty relevant, although the shared shrinkage limits the influence of the small sample. This experiment tests a fixed-mean weighted second moment; it does not optimize a separately recentered weighted mean or directly train on KL.

Audits: 2,080 whole-model forwards; full-rank logits replay within5.97e-5; cached head activations replay within2.29e-5; inverse basis error<=2.00e-15. Inputs/model hashes and saved pre-evaluation fit hash unchanged; outputs finite. All fitting data are training-only; original reserved test untouched; unique natural document pool remains26,112. No causal transfer or manifold validation has yet been performed for these new fits.

Next: evaluate **all frozen fits** on an additional document panel without refitting or selecting configurations. Use a predetermined document ordering and exclude the H765 and H769 validation documents, while keeping the original reserved test untouched. This is a check on additional research-validation documents, not a claim of pristine test-set confirmation. If an advantage survives, evaluate the same frozen fits on the diverse causal panels. If it does not, reject this weighting scheme as an established improvement rather than extending architecture based on a favorable mean.
