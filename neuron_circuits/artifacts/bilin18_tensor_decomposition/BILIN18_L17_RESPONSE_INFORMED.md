

## H291 — Residual-response covariance directions fail to improve validation

H290 found negligible gains from natural-variance outside directions. H291 collects full1152-dimensional normalized input-change sufficient statistics using the same768 training/256 validation documents, nine heads, actual/angle payloads and seed. Existing feature maps/native base remain fixed. Direction construction uses only training cross-covariance: subtract the prediction of the fixed H29064-feature alpha.1 readout, project the remaining input-output cross-covariance orthogonal to the existing64 span, and take leading left singular vectors. Reproject/QR for numerical orthogonality, normalize by training natural-state standard deviation, and append32/64 directions. The inherited alpha.1 was chosen using validation in H290; it is fixed here before new direction outcomes. Current validation response covariance is never used to construct directions. Training targets are reused for direction learning and coefficient fitting, so supervised basis overfitting is possible.

This is an unwhitened residual-covariance SVD heuristic, not an optimal reduced-rank regression or exhaustive supervised direction search. Feature coordinates are scaled by natural-state standard deviation; ridge remains coordinate dependent. Joint output-MSE ridge fits use the same eight-alpha grid at widths64/96/128, with all forty candidates retained across five families. The original64 and natural-PCA controls are recomputed from full-input statistics; validation objective replay differs from H290 by at most1.78e-10. Nested maps match exactly, outside-span overlap is below1e-12. A solve residual audit is retained for each candidate. Intact KL is not evaluated in this response-only screening step.

| Family | Selected alpha | Training response error/base | Validation response error/base |
|---|---:|---:|---:|
|base64|0.1|0.963913|0.969040|
|natural96|0.1|0.958209|0.968269|
|natural128|0.1|0.952884|0.968227|
|informed96|1|0.956195|0.970195|
|informed128|1|0.949942|0.970325|

Best informed family: informed96_a1. Validation gain versus best natural expansion is-0.203% and versus regularized64 is-0.119% (both negative, hence worse). Both preset2% gates FAIL. No thresholds relaxed. These are existing-validation selections, not fresh-confirmatory results; no significance claim is made for the small differences.

**What this rules out locally.** Simply selecting outside directions by residual cross-covariance does not solve the validation gap, even though the wider supervised map fits training better. The controls reproduce earlier results, so a statistics-coordinate mismatch is unlikely to explain the failure. This weakens this particular direction-selection recipe, not all outside information or supervised dimensionality reduction. A feature map learned from full-dimensional targets plus a readout learned on the same documents can amplify estimation noise. Unwhitened covariance also need not rank directions by conditional predictive utility.

**Costs.** Both expanded families have the same active output-coefficient counts110,592/147,456 and32/64 extra reads; full direction-map storage and the original base are additional. The informed method uses native response targets for basis construction, unlike the natural PCA comparator. The saved full-input statistics are a research cache, not required at inference. No new semantic or intrinsic-dimensionality claim follows from this negative result.

**Next discriminator.** Separate covariance geometry from sampling noise before another large feature sweep. Using the saved full-input training statistics, compute a regularized reduced-rank regression in the orthogonal complement, residualizing existing reads in the response-design covariance metric. This explicitly accounts for correlated input changes, unlike raw cross-covariance ranking. Fix whitening/shrinkage and rank grids before validation outcomes; compare at matched extra rank/output cost. If it helps, direction geometry was the issue. If it still mainly helps training, use disjoint documents for basis construction and readout fitting or a learning curve to test supervised-basis estimation. A fresh panel is required before promoting any validation winner. No new experiment is claimed beyond the reported covariance-SVD screen.

All jobs terminal. Original native weights/compiler/base unchanged. Full response statistics, frozen maps and all40 fitted readouts saved under bilin18_l17_response_informed*. Goal remains active.
