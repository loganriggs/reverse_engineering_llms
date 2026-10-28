

## H402 — Direct prediction of the nonadditive response

Pairs(1,8),(6,8), actual and six full-Gram-preserving rotations, reused H400512-document panel. Test amplitudes a in{0.25,0.5,1,2,4}, with b=+a and b=−a. Predict the raw interaction D=MLP(p+a h+b k)−MLP(p+a h)−MLP(p+b k)+MLP(p). The map includes input RMS. The direct residual is fixed and there is no final RMS or native output KL score in this experiment.

Let N_h(a)=a*n1_h+a²*n2_h, q_h=1+2a*alpha_h+a²*beta_h, q_k analogously, q=q_h+q_k−1+2ab*gamma, and C=2B(h,k)/s0²−2gamma*M0. The mixed Taylor coefficient is m_hk=C−2alpha_h*n1_k−2alpha_k*n1_h. Compare a*b*m_hk and a*b*m_hk/q, with no target fitting.

To avoid cancellation from subtracting four float32 MLP outputs for small interactions, the evaluation target uses the stable exact expression D=(ab/q)[C−(n1_h+a*n2_h)(2alpha_k+b*beta_k+2a*gamma)/q_h−(n1_k+b*n2_k)(2alpha_h+a*beta_h+2b*gamma)/q_k]. Native bilinear coefficients are computed in float32; this expression is assembled in float64. Audit against directly evaluated double-precision finite interactions at four fixed sites per condition/amplitude, explicitly preserving native float32 RMS epsilon. This is an architecture-dependent exact target law in real arithmetic, not a learned target model or a full-panel independent native-double recomputation.

Recovery is1−pooled||predictedD−D||²/pooled||D||², using the common intact-state squared norm for site scaling. It measures prediction of the interaction itself, not fidelity of the much larger joint response or output distribution. Also retain predicted/target interaction-energy ratios, per-site cosines and actual/rotated interaction energy normalized by first-order source energy a²||m1_h||²+b²||m1_k||².

Frozen gates: {'native_double_audit': True, 'quarter99': [True, True], 'unit95': [True, True], 'four90': [False, False]}. Require native-double audit relative error<.001, and per predictor all eight lower recovery bounds above.99 at0.25×,.95 at1×,.90 at4×. Four thousand paired document/rotation-draw resamples give a separate joint8 band per amplitude and predictor; coverage is not simultaneous across all amplitudes/predictors. The actual/rotated normalized-coupling contrast uses a separate joint20 band.

|Pair|Family|Amplitudes|Mixed quadratic recovery|With exact denominator|
|---|---|---|---:|---:|
|[1, 8]|actual|[0.25, 0.25]|0.999676|0.999604|
|[1, 8]|actual|[0.25, -0.25]|0.999603|0.999664|
|[1, 8]|actual|[0.5, 0.5]|0.998693|0.998370|
|[1, 8]|actual|[0.5, -0.5]|0.998412|0.998623|
|[1, 8]|actual|[1, 1]|0.994675|0.993100|
|[1, 8]|actual|[1, -1]|0.993647|0.994220|
|[1, 8]|actual|[2, 2]|0.977919|0.969388|
|[1, 8]|actual|[2, -2]|0.974668|0.974799|
|[1, 8]|actual|[4, 4]|0.906897|0.859471|
|[1, 8]|actual|[4, -4]|0.901958|0.888162|
|[1, 8]|rotated|[0.25, 0.25]|0.999364|0.999433|
|[1, 8]|rotated|[0.25, -0.25]|0.999547|0.999586|
|[1, 8]|rotated|[0.5, 0.5]|0.997475|0.997700|
|[1, 8]|rotated|[0.5, -0.5]|0.998196|0.998314|
|[1, 8]|rotated|[1, 1]|0.990085|0.990572|
|[1, 8]|rotated|[1, -1]|0.992853|0.992994|
|[1, 8]|rotated|[2, 2]|0.962216|0.960835|
|[1, 8]|rotated|[2, -2]|0.971890|0.969821|
|[1, 8]|rotated|[4, 4]|0.868137|0.841158|
|[1, 8]|rotated|[4, -4]|0.889859|0.865341|
|[6, 8]|actual|[0.25, 0.25]|0.997940|0.998582|
|[6, 8]|actual|[0.25, -0.25]|0.998587|0.999018|
|[6, 8]|actual|[0.5, 0.5]|0.991976|0.994310|
|[6, 8]|actual|[0.5, -0.5]|0.994461|0.996035|
|[6, 8]|actual|[1, 1]|0.970077|0.977533|
|[6, 8]|actual|[1, -1]|0.979038|0.984121|
|[6, 8]|actual|[2, 2]|0.901711|0.918274|
|[6, 8]|actual|[2, -2]|0.928944|0.940704|
|[6, 8]|actual|[4, 4]|0.752282|0.764364|
|[6, 8]|actual|[4, -4]|0.814380|0.826742|
|[6, 8]|rotated|[0.25, 0.25]|0.997895|0.998654|
|[6, 8]|rotated|[0.25, -0.25]|0.998803|0.999178|
|[6, 8]|rotated|[0.5, 0.5]|0.992060|0.994718|
|[6, 8]|rotated|[0.5, -0.5]|0.995382|0.996698|
|[6, 8]|rotated|[1, 1]|0.972004|0.979822|
|[6, 8]|rotated|[1, -1]|0.982985|0.986854|
|[6, 8]|rotated|[2, 2]|0.913896|0.927809|
|[6, 8]|rotated|[2, -2]|0.943665|0.950183|
|[6, 8]|rotated|[4, 4]|0.759056|0.765608|
|[6, 8]|rotated|[4, -4]|0.837273|0.838740|

Audits: {'max_relative_stable_interaction_vs_native_double': 1.2408494016013722e-06, 'geometry': [{'seed': 20270724, 'norm_scaled_gram_max_error': 3.810464967172756e-08, 'norm_scaled_residual_dot_max_error': 2.5115461557783615e-08}, {'seed': 20270725, 'norm_scaled_gram_max_error': 3.116312570687793e-08, 'norm_scaled_residual_dot_max_error': 2.2323191496826936e-08}, {'seed': 20270726, 'norm_scaled_gram_max_error': 4.0908587531560625e-08, 'norm_scaled_residual_dot_max_error': 2.6880952128976435e-08}, {'seed': 20270727, 'norm_scaled_gram_max_error': 3.020253555803863e-08, 'norm_scaled_residual_dot_max_error': 2.1853360069183966e-08}, {'seed': 20270728, 'norm_scaled_gram_max_error': 3.6941856236840525e-08, 'norm_scaled_residual_dot_max_error': 2.4686722268322785e-08}, {'seed': 20270729, 'norm_scaled_gram_max_error': 3.396755517584083e-08, 'norm_scaled_residual_dot_max_error': 2.339745315961153e-08}], 'native_audit_sites_per_condition_amplitude': 4}. Independent saved-array checks: {'recovery_max_absolute_error': 1.1102230246251565e-16, 'native_float32_H401_interaction_energy_pooled_relative_differences': [2.8773594956229276e-05, 3.426121792055783e-05, 9.568727701862469e-07, 1.262214005860314e-06]}. The initial runner failed because a residual-dot audit variable overwrote the double output-weight variable. It was fixed before the completed run; the failure log is retained. No inference is drawn from the failed run.

Outcome: native-double audit passes with maximum relative interaction discrepancy1.24e-6 on the fixed audit sites. Both predictors pass the0.25×99% and1×95% gates; both fail the4×90% gate. Mixed-quadratic vector recovery ranges99.789–99.968% at0.25×,99.198–99.869% at0.5×,97.008–99.467% at1×,90.171–97.792% at2×, and75.228–90.690% at4×. Minimum joint lower bounds at0.25× and1× are99.752% and96.523%, respectively. This is prediction of the nonadditive interaction vector itself, so it establishes more than high fidelity of the total response alone.

The exact-denominator variant has minimum recoveries99.858%,99.431%,97.753%,91.827%,76.436% across the same amplitudes. It improves some cases but worsens others: actual heads1/8 at(4,4) drops from90.690% to85.947%. For actual heads6/8 at(4,4), recovery increases only from75.228% to76.436%. Thus the denominator alone cannot restore the omitted higher-order numerator terms in the interaction law. Failures at4× are retained; the known complete rational formula is not treated as evidence of a learned low-complexity feature representation.

Actual/rotated interaction energy divided by first-order source energy is below1 in all20 pair/amplitude contrasts under the joint20 band. Heads1/8 range0.667–0.822, heads6/8 range0.328–0.446. At0.25× same-sign amplitude, the ratios are0.7729,[0.6645,0.8990], and0.3284,[0.2824,0.3820]. These results use a first-order denominator and are distinct from H400's equivalence after adjustment by finite individual-response energies at selected4× conditions. The different denominators encode different questions; neither result cancels the other.

This supports a local explanation for the selected source pairs: learned orientations can have large first-order response energy while mixed curvature is comparatively modest relative to that sensitivity. The mixed Taylor operator predicts the ordinary-amplitude interaction well for actual and rotated directions alike. At larger amplitudes, nonlinear evolution of both individual-response energy and mixed interaction matters. The result does not show that the mixed vector is sparse in native factors, shared across contexts, semantically interpretable or a fundamental computational unit. These are two selected pairs on a reused panel, so the orientation-specific conclusion requires broader coverage.

Hypothesis update: a local derivative-plus-mixed-curvature description is predictively useful for these joint interventions at ordinary scale. A scalar response-gain explanation is incomplete, while the first-order-normalized curvature contrast gives a concrete, testable refinement. Exact-denominator-only resummation is insufficient at large amplitude. Generic Taylor behavior explains the small-amplitude success but does not by itself explain the actual-versus-rotation curvature-to-sensitivity ratios.

Next high-information experiment: map the local mixed-curvature energy relative to first-order source energy across all36 head pairs, using full-Gram-preserving rotations and a fixed simultaneous comparison. At infinitesimal equal amplitudes, the shared amplitude factor cancels from the actual/control ratio, allowing a cheap derivative-only screen. Keep both absolute mixed-curvature energy and first-order energy alongside their ratio. This tests whether the two selected pairs reflect a broad learned orientation pattern or a restricted exception. Any selected pattern should then be confirmed on fresh documents and examined for native-factor support and cross-context transfer, rather than naming a local Taylor coefficient a semantic unit.
