

## H350 — Radial correction amplifies controls more

Reuse H346 recipients and exact compiler response deltas. At intact residual state s with RMS r, split the normalization derivative into scaling u=δ/r and radial correction v=−s mean(sδ)/r³. Propagate both through the same unembedding, capped-logit derivative and probability covariance. Divide energies by Euclidean response energy, then average sites and realizations equally. Full energy equals scale energy + correction energy + signed cross energy. Also measure Euclidean squared cosine between response and residual state.

|Condition|Radial fraction|Scale energy ×10¹¹|Correction energy ×10¹¹|Cross energy ×10¹¹|Full/scaling energy|Scale/correction cosine|
|---|---:|---:|---:|---:|---:|---:|
|actual|11.40%|5.833|4.971|1.839|2.167|0.171|
|moment|25.12%|7.435|11.754|4.145|3.138|0.222|
|donor|23.44%|7.178|11.428|3.747|3.114|0.207|

Natural responses are less radial, not more. Actual-minus-control radial fractions are −13.71pp [−15.66,−11.84] for moment controls and −12.03pp [−13.73,−10.33] for donors. The natural-more-radial gate FAILS.

The named natural-suppression10pp gate passes numerically: actual-minus-control differences in 1−full/scaling are +0.971 [0.678,1.291] and +0.947 [0.623,1.307]. But all three suppression statistics are negative. This is less amplification for natural responses, not actual suppression. The gate label must not be promoted into a mechanistic conclusion. Cross energies are positive for all conditions, so aggregate cancellation between scaling and correction is not the explanation.

A radial subtraction can reduce Euclidean normalized-state energy while increasing probability-space energy because the downstream metric is anisotropic: removing a component can remove an opposing downstream effect. The scale-only derivative is a counterfactual diagnostic, not the true response to a radial perturbation. Nothing here implies the native model responds to a pure scale change.

Uncertainty:4000 joint document/global realization resamples, pointwise conditional intervals, fixed donor assignments and six realizations; donor dependence not fully reconstructed. Reused panel. The measured effect belongs to the isolated fixed compiler response, not the complete native intervention.

Audit: native full metric replays H346 within 3.39e-16; exact derivative terms and per-site energies saved. Cross energy is defined by full−scale−correction; that identity alone is not an independent validation. Native weights unchanged.

Next: intervene on the radial component of the compiler response at fixed magnitude, and measure both finite KL and local predictions. This can test whether radial geometry explains the control gap beyond an accounting identity. Revisit the broad hypothesis ranking with these negative results. Broad research goal remains active.
