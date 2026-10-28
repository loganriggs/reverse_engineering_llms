

## H289 — Adding natural-variance outside directions does not improve validation responses

H288 found little response-only headroom within64 linear features. H289 retains those64 and appends the leading32 or64 standardized directions from H282's outside-span natural-state PCA map. That map was fit only on the first768 discovery states; no validation targets or responses select directions. Effective outside-direction overlap with the original read span is4.31e-9 in relative Frobenius norm. This is a nested input-span expansion, not a rotation or a comparison of nonlinear features. It tests this particular natural-variance selection rule; response-informed directions remain untested here.

The same joint actual+angle/all-nine-head full-removal response objective is solved exactly using training-only float64 normal equations at widths64/96/128. No intact objective, penalty, fitted intercept, or optimizer changes. Active output coefficients are73,728/110,592/147,456; all saved W arrays are zero-padded to128 rows for common scoring, so physical export size is not minimized. Feature-map storage is additional. All three Gram matrices are full rank; condition numbers883/924/946. Cutoffs1e-12/1e-8/1e-4 produce bitwise-identical exported matrices. Maximum normal-equation residual<5e-15. The64-feature coefficients reproduce H288 to relative1.49e-12 and its pooled response scores match. Intact KL differs only at floating-point rounding scale because matrix dimensions differ. Existing least-squares algebra tests cover this solve; no new test suite was added for the width change.

This is an EXISTING-validation diagnosis, not a fresh confirmation. No model or cutoff was selected using validation. Pooling weights groups by their base response-error energy. Interventions and sampling seeds are identical across widths.

| Width | Training response error/base | Validation response error/base | Validation intact KL | Active output coefficients |
|---|---:|---:|---:|---:|
|64|0.960995|0.969821|0.058870|73,728|
|96|0.955344|0.970354|0.059177|110,592|
|128|0.950112|0.971384|0.059124|147,456|

128 versus64 validation response reduction: -0.161% (negative means worse).128 versus96: -0.106%.128 versus64 intact KL worsening: 0.432%.

Locked diagnostic gates: span_gain: FAIL, incremental_gain: FAIL, state_retention: PASS. State retention is only a<=2% worsening check, not evidence of improvement. No confidence interval or fresh-panel significance claim.

**Hypothesis update.** The expanded span reduces finite-training error monotonically, as nested least-squares classes must, but validation error rises slightly. Thus doubling output coefficients with these variance-selected outside directions does not address the remaining response error. This weakens a simple missing-high-variance-directions account; it keeps estimation and feature-selection mismatch live. Numerical conditioning and iterative optimizer failure are not plausible explanations for these specific fits. The result does not show that outside directions are irrelevant or that64 dimensions suffice: the added basis was chosen for natural-state variance rather than response prediction, and unregularized added coefficients can overfit.

**Next discriminator.** Before choosing a new basis, test a fixed ridge grid for the nested spans using the saved training sufficient statistics. Include the64-feature solution as a baseline and select ridge by document-validation response MSE, with intact KL tracked separately. If shrinkage recovers the added-direction gains, estimation explains this failure; if no expanded model wins, move to response-informed direction selection. Freeze the winning width/regularizer before fresh confirmation. Avoid interpreting a tiny validation winner as proof without that confirmation. The statistics are saved as bilin18_l17_response_span_statistics.npz, so this next test needs no new native collection.

All jobs terminal. Original native weights/compiler/base remain unchanged. New readout coefficients and statistics saved separately. Goal remains active.
