

## H394 — Exact rational response law along a fixed source direction

For intact pre-MLP residual p and direction h=−payload, define s0²=mean(p²)+epsilon, alpha=mean(p h)/s0², beta=mean(h²)/s0², and q(a)=1+2 alpha a+beta a². Let m1,m2 be the first two local Taylor coefficients of the input-normalized bilinear MLP response, excluding the fixed output bias. The derived predictor is

Delta_MLP(a) = [a m1 + a²(m2+2 alpha m1)] / q(a).

Add this predicted MLP-only change to the intact final raw state, then apply final RMS, unembedding, logit cap and softmax exactly. The direct residual p stays fixed. Prediction uses intact context, payload and native weights to calculate local coefficients; no finite target response or amplitude-specific fit enters prediction.

Derivation: the input-normalized homogeneous bilinear map is (M0+a N1+a²N2)/q(a). Its Taylor coefficients imply N1=m1+2 alpha M0 and N2=m2+2 alpha m1+beta M0. Subtracting M0 cancels the baseline numerator and yields the formula above. The input RMS epsilon is included in s0²; the constant MLP output bias cancels in the response and remains in the final state. Thus this is an architecture-dependent real-arithmetic identity, not an empirical low-rank discovery.

Validate on the reused H390512-document panel:39 head/response conditions,2,048 sites, signed amplitudes−4,−1,0.5,1,2,4,8. Negative amplitudes increase the source contribution; large positive amplitudes overshoot removal. Full-panel float32 coefficients and native MLP responses are scored through double-precision unembedding/KL. A separate float64 identity check uses four fixed sites per condition at every amplitude, with the native float32 RMS epsilon explicitly retained.

Frozen gates: {'native_recovery9999': True, 'native_max_kl': True, 'double_identity': True, 'positive_denominator': True}. Require all63 joint lower recoveries >99.99%; maximum per-site native KL <1e-7; float64 relative MLP error <1e-9 on audited sites; and positive denominators throughout.

|Amplitude|Minimum aggregate recovery|Largest aggregate unrecovered fraction|Maximum per-site KL|Minimum denominator|
|---:|---:|---:|---:|---:|
|-4|0.999999999859|1.41e-10|2.15e-11|0.950371|
|-1|0.999999997173|2.83e-09|2.21e-11|0.983641|
|0.5|0.999999987911|1.21e-08|1.48e-11|0.902012|
|1|0.999999996943|3.06e-09|2.21e-11|0.822401|
|2|0.999999999210|7.9e-10|1.5e-11|0.709804|
|4|0.999999999815|1.85e-10|2.33e-11|0.502573|
|8|0.999999999956|4.44e-11|2.61e-11|0.337009|

Audits: {'float64_max_relative_mlp_error': 4.549732104512593e-13, 'float32_max_relative_mlp_error': 0.0047598001547157764, 'float32_relative_mlp_error_quantiles': [8.78037008078536e-06, 0.00012638661864912133, 0.00037227830296615057, 0.0047598001547157764], 'minimum_denominator': 0.3370089530944824, 'maximum_denominator': 3.0691704750061035, 'maximum_native_kl': 2.6104701552110112e-11, 'double_audit_sites_per_condition': 4}. Independent saved-array checks replay earlier baseline scores at the shared amplitudes and recompute denominators from the saved alpha/beta arrays. The actual and angle/norm-matched control scalars are compared explicitly: [{'head': 0, 'alpha_control_minus_actual_max_absolute': 8.381903171539307e-09, 'beta_control_minus_actual_max_absolute': 1.3969838619232178e-09, 'denominator_replay_max_absolute': 0.0}, {'head': 1, 'alpha_control_minus_actual_max_absolute': 3.725290298461914e-08, 'beta_control_minus_actual_max_absolute': 8.381903171539307e-09, 'denominator_replay_max_absolute': 0.0}, {'head': 6, 'alpha_control_minus_actual_max_absolute': 4.470348358154297e-08, 'beta_control_minus_actual_max_absolute': 1.862645149230957e-08, 'denominator_replay_max_absolute': 0.0}]. Coefficient vectors, geometry scalars, native KL scores, denominators, per-site relative MLP errors and bootstrap samples are retained. Independent audit fields are added by this reporting script.

Uncertainty:4,000 paired recipient-document/control-draw bootstrap resamples with one simultaneous63 recovery band across all seven amplitudes and nine groups. Numerical exactness is supported principally by algebra and arithmetic checks; bootstrap recovery is a population-summary diagnostic. Fixed donor assignments, reused documents, same model. Nonzero residual discrepancies reflect finite arithmetic and must not be called bitwise identity.

Interpretation: all four frozen gates pass. Across the nine head/source-family groups and seven signed amplitudes, minimum aggregate recovery is 0.9999999879105654. Maximum per-site native KL is 2.61047e-11. The separate float64 identity audit has maximum relative MLP error 4.54973e-13. Float32 MLP-response relative error can reach 0.00476, particularly when subtracting nearly equal quantities; this does not invalidate the output-distribution agreement, and the two error measures should not be conflated. Denominators remain between 0.337009 and 3.069170 on this panel.

This resolves why second-order Taylor truncation broke down at large amplitudes: the input-normalization denominator generates higher orders that can be summed exactly. Two local response vectors plus two scalar geometry coefficients encode this one context-specific, direction-specific trajectory. The baseline, native weights and output transformation remain necessary. This is neither global rank-two compression nor evidence of two semantic features. The architecture implies the identity; the experiment verifies the implementation and its native-precision consequences.

The next discriminating experiment is simultaneous intervention on two source heads. Compare the sum of separate finite responses, a shared-denominator predictor built from the two single-source laws, and the exact joint law including the mixed bilinear numerator. This separates errors from normalization coupling and cross-source bilinear interaction. Use fixed head pairs and signed amplitude combinations, audit the full joint identity, and measure the behavioral cost of omitting each coupling. A mixed term's formal existence alone does not establish its behavioral importance or irreducibility. Subsequent projection tests can ask whether that mixed response lies in the span of the four single-direction vectors. These hypotheses and experiment choices precede viewing any joint-intervention outcomes.
