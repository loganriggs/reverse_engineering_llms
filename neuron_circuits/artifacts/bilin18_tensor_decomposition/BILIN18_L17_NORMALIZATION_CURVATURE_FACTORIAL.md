

## H426 — Input normalization versus bilinear curvature

Reuse H4251024 recipients,12 donor/Gaussian draws, H421 donor moments and H410 groups. Same amplitudes0.25 and4. Output scoring is fixed baseline native Fisher quadratic energy throughout. No new documents. The fully linearized response/quadratic-metric rows are hash-verified H425 references; three other cells are newly computed.

Let x=RMS(p), using the native float32 epsilon in double arithmetic. Exact normalized-input change is u_N=RMS(p-a h)-x; the linearized change is u_L=a DRMS(p)[-h]. A bilinear MLP changes by 2B(x,u)+B(u,u). The linearized bilinear cell retains only 2B(x,u). Crossing the two input changes with the two bilinear responses gives four cells. The full/full response is reconstructed in double arithmetic and checked against H425 native-float response scores; the baseline output metric stays unchanged. Numerical checks cover the bilinear identity, linear baseline vector, and pooled native replay.

For each cell, retain S=R_union-R_core-R_added using pooled quadratic errors and effects. I is [S_Gaussian(4)-S_raw(4)]-[S_Gaussian(0.25)-S_raw(0.25)]. Freeze N=I_exactRMS,linearB-I_linearRMS,linearB; B=I_linearRMS,fullB-I_linearRMS,linearB; X=I_exactRMS,fullB-I_exactRMS,linearB-I_linearRMS,fullB+I_linearRMS,linearB. Joint95% max-absolute bands use4000 paired document/draw bootstrap samples conditional on the donor pool. Negative02 requires upper<-0.02; equivalence02 requires the entire interval inside +/-0.02. These signed contributions use cell-specific denominators and are diagnostic accounting, not unique causal mediation fractions.

|Normalized input change|Bilinear response|Distribution-by-amplitude I|
|---|---|---:|
|Exact|Full|-0.048075|
|Exact|Linear|-0.031688|
|Linear|Full|-0.022382|
|Linear|Linear|0.000000|

|Contribution|Estimate|Joint95% interval|Negative02|Equivalence02|
|---|---:|---|---|---|
|input_RMS_only|-0.031688|[-0.081106, 0.017730]|False|False|
|bilinear_only|-0.022382|[-0.071800, 0.027037]|False|False|
|factorial_interaction|0.005994|[-0.043424, 0.055412]|False|False|

Outcome: numerical reconstruction succeeds, but the mechanism split remains statistically unresolved. Input-RMS-only contribution is -0.031688, joint95% interval [-0.081106,0.017730]; bilinear-only is -0.022382, [-0.071800,0.027037]; factorial interaction is +0.005994, [-0.043424,0.055412]. All three negative02 and all three equivalence02 gates fail. Neither component is shown to be negligible, and neither is individually established under this joint band. The point sum recovers the prior MLP contribution; this does not independently replicate H425's statistical result because the panel is reused.

All numerical gates pass. The bilinear product identity error is1.49e-12, the linear baseline vector agrees exactly, and factorial closure is exact. The reconstructed double full-response pooled combination scores differ from H425 native-float scores by at most1.05e-7; pooled full energies differ relatively by at most8.94e-7. Thus arithmetic discrepancies are small under the frozen replay tolerances and do not explain the unresolved attribution.

In plain language: allowing input rescaling to become nonlinear or allowing the multiplication step to retain its extra product term each changes the estimate in the expected negative direction, but these data do not establish which mechanism accounts for the distribution-dependent trend. It would be wrong to label either one the dominant cause. The fixed output metric and the shared native linear baseline make the comparison interpretable, but do not remove sampling uncertainty or turn normalized score differences into unique mediation proportions.

A separate descriptive algebraic diagnostic reconstructs the bilinear-only cell's full energy and three prediction-error curves from its baseline and two endpoints. For linearized normalized-input change, each response vector has form a A+a^2 B. Its Fisher energy is a^2 E2+a^3 E3+a^4 E4. Saved arrays suffice to solve these three coefficients. Endpoint replay error is6.77e-17 relative to the largest saved energy; quartic coefficients are nonnegative and the recovered two-vector Gram matrices satisfy their nonnegative-determinant constraint. Endpoint replay is by construction, not independent validation at new amplitudes. Coefficients and diagnostics are saved separately.

Descriptively, the raw-donor full-energy coefficients [E2,E3,E4] are [0.00234838,0.0000252281,0.0000586574], versus [0.00237735,0.0000867263,0.0000960471] for Gaussian draws. No new confidence intervals or confirmatory claims are attached to these post-primary coefficient comparisons. They motivate testing cubic and quartic contributions directly. At a fixed recipient, their ensemble expectations depend on source moments up to orders2,3,4 respectively; empirical first/second-moment mismatch and recipient assignment still need controls.

Next high-information direction: analyze the polynomial's numerator and denominator terms separately, then predict effects at signed and intermediate edit amplitudes not used to reconstruct the coefficients. Changing the edit sign reverses the cubic energy term while retaining the quadratic and quartic terms. This provides a sharper mechanistic contrast than another unrestricted normalization-versus-curvature estimate. Keep this branch's scope explicit: exact input RMS and exact output KL are outside the polynomial cell, and successful algebraic replay is not a semantic-feature discovery. If an informative coefficient contrast is selected on reused data, confirm it separately.

Updated priorities: H425's evidence that nonlinear MLP response generation contributes remains; the A1-A3 subdivision is open. There is no new reason to discard input normalization, bilinear curvature, or their interaction. The controlled polynomial gives a concrete candidate explanation to test, rather than a justification for asserting a small universal computational unit. The larger goal still requires predictive transfer, causal behavioral coverage, and semantic interpretation where supported.


Numerical gates: {'bilinear_identity': True, 'linear_baseline_vector': True, 'native_pooled_replay': True}. Audits: {'bilinear_identity': 1.4853939785911885e-12, 'linear_baseline_vector': 0.0, 'native_pooled_score_replay': 1.0463202171351327e-07, 'native_pooled_energy_relative_replay': 8.94384483141053e-07, 'factorial_closure': 0.0}. Independent array checks: {'linear_baseline_reference_identical': True, 'pooled_scores_and_joint_interval_replayed': True, 'native_pooled_score_replay': 1.0463202515520464e-07, 'linear_quadratic_row_scaling_max_abs': 0.0}. Completed unique documents remain10944. The broad goal remains open.
