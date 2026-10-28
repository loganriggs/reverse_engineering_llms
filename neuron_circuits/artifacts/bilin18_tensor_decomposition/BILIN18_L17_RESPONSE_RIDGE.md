

## H290 — Ridge slightly rescues wider response fits, without a material width gain

Use H289 saved training/validation sufficient statistics; no native model rerun or new document collection. At widths64/96/128, solve (G + alpha trace(G)/width I)W=R in float64 for alpha0/.0001/.001/.01/.1/1/10/100. The24-point grid is fixed before outcomes. Each width selects alpha by pooled existing-validation response MSE. Best expanded width is compared to the best regularized64-feature fit, not only unregularized64. Coefficient penalty is isotropic in the stored standardized coordinates and therefore not invariant to arbitrary rescaling/reparameterization.

All widths select alpha.1. Unregularized coefficients reproduce H289 eigensolves to relative<1e-8; normal-equation residuals are near machine precision. Exported matrices are float32 and zero-padded to128 rows; objective selection is computed from float64 solutions and saved sufficient statistics. There is no observed numerical instability. The selection minima and zero padding are audited. Prior ridge/least-squares tests cover the algebra; no additional implementation-mirroring test was added. Physical stored sizes are not minimized.

| Width | Selected alpha | Training response error/base | Validation response error/base |
|---|---:|---:|---:|
|64|0.1|0.963913|0.969040|
|96|0.1|0.958209|0.968269|
|128|0.1|0.952884|0.968227|

The selected expanded model isn128_a0.1. Its validation improvement over the best64 is0.0839%; its improvement over the same-width unregularized fit is0.3250%. Both preset2% gates FAIL. The96/128 validation scores differ by only about.0000424 in base-normalized error. No thresholds relaxed.

**Scope.** Existing validation now selects hyperparameters, so its tiny advantage is not confirmatory evidence. No confidence interval or fresh generalization claim is made. These are response-only objectives; intact KL was not remeasured for these selected ridge fits and must not be inferred from H289's unregularized results. No new model is promoted. Original native weights/compiler/base remain unchanged.

**Hypothesis update.** Some H289 degradation was regularization-sensitive: the wider fits become slightly better than the regularized64 baseline. But the tiny gain fails to support a useful missing-high-variance-direction account at these widths. This does not rule out outside information, because the added directions were selected for natural-state variance. It also does not prove regularization is exhausted outside this grid or that the optimum response representation is64-dimensional.

**Next discriminator.** Choose added directions for response prediction rather than natural-state variance. Fit training-only response-informed low-rank linear maps in the orthogonal complement of the existing span, with explicit shrinkage and a fixed rank budget. Compare64+32 and64+64 response-informed directions to the natural-PCA directions using the same response data and output budget. Keep a train/validation separation for direction selection itself; never use validation target covariance to build directions. Track conditioning, rank, all parameter/storage costs and intact KL. Freeze candidates before a new confirmatory panel. Large gains would support direction-selection mismatch; another near-null result would motivate nonlinear/context-dependent features or wider-data estimation. This experiment remains to be run.

All computations terminal.24 fits, full curves and coefficients retained under bilin18_l17_response_ridge*. Goal remains active.
