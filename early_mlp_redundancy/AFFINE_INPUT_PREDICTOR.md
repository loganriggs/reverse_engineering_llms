# Does one affine computation explain MLP1?

M78 completed. MLP1 means the second transformer block. This experiment predicts its product write directly from its normalized input, rather than encoding a write the native MLP has already computed.

## Question and construction

M77 found that a recipient-specific Jacobian predicts native changes much better than the Jacobian at the training mean. That did not rule out an empirically fitted global affine map. Here we fit exactly one full 1152-to-1152 affine map on 4,092 unique-prefix positions from natural documents 0–31. Centered float64 least squares retains all 1,152 singular directions; there is no rank, regularization, or objective search. All seven task families remain excluded from fitting.

The independent QR calculation agrees with coefficients to relative error 8.64e-15, and the normal-equation residual is 1.26e-15. Thus this is the empirical training MSE optimum to numerical precision. An executable whole-MLP override exactly matches callback logits while a guard prohibits native product computation. It has 1,328,256 effective affine coefficients; comparisons with smaller featurizers are not matched-cost comparisons.

## Results

| Measurement | Result |
|---|---:|
| Training pooled squared write error / native write energy | 0.20088 |
| Held-out pooled squared write error / native write energy | 0.56460 |
| Held-out mean-document relative write error | 0.58032 |
| Natural-text output KL | 0.16322 |
| Natural-text CE increase | 0.16844 |
| Natural-text top-1 agreement with native model | 79.57% |
| Qualified-task contrast accuracy | 95.00% (native 96.25%) |
| Qualified-task top-1 answer accuracy | 36.25% (native 41.25%) |

The training/held-out comparison uses pooled energy in both cases. The large residual on training points establishes that these sampled writes are not exactly affine; the larger held-out residual also leaves generalization and distribution shift relevant. Neither result rules out a different behavioral training objective.

For causal evaluation, replace a recipient write with a donor write and compare the resulting change in final log probabilities. The reported normalized error uses native output probabilities as weights. Error 1 corresponds to predicting no response, not a threshold for adequate understanding. “Source” patches the changed input positions; “final” patches the final position. Each candidate response is centered on its own background.

| Background / supplied change | Source, all 7 | Final, all 7 | Source, qualified 5 | Final, qualified 5 |
|---|---:|---:|---:|---:|
| Native background / affine change | 0.6932 | 1.9224 | 0.7500 | 1.8794 |
| Affine background / native change | 21.7461 | 1.2860 | 0.8913 | 0.8403 |
| Affine background / affine change | 13.4096 | 2.4733 | 1.7141 | 2.4285 |

These are equal-family averages. Pooled errors for both-affine replacement are 5.0676/1.4355 on all families and 0.8581/1.4315 on qualified families (source/final). Retain both aggregations: weak spatial-inverse cases strongly affect the all-family source average. The qualified families are sequence copying, weekday succession, tense cues, continent recall, and object function; role binding and spatial inversion remain reported. Contrast qualification is not free-form mastery. All 56 pairs and 14 formats were historically evaluated, so these are training-excluded tasks, not pristine discovery data.

## What this changes conceptually

A single fitted global affine computation is insufficient for faithful replacement under this protocol. Together with M77's recipient-local advantage, this motivates testing explicit context dependence. It does not show that native neurons are redundant, that their contributions cancel, or that the model contains semantic clusters. Those concern distinct implementation or data-geometry properties. High contrast accuracy alongside poor response fidelity is another concrete reason to keep behavioral success separate from computational equivalence.

## Verification and next discriminating test

All 984 response vectors, 328 geometry rows, 112 independently computed endpoint predictions, and all 32 natural full outputs passed independent audits. Eight source-ineligible assignments remain explicit. Native replay checks cover 112 baselines and 328 teacher targets. There were 1,156 native forwards including four preflight forwards. The transformer itself was not independently reimplemented.

Next: a fixed comparison of local affine charts at one mean anchor, 32 random training anchors, and 32 training-only k-means centroids. Native quadratic structure gives an exact approximation remainder, allowing a theory-backed test of whether input clustering improves reusable computation and whether routing transitions introduce error. These charts use native weights and large full-rank operators: they are geometric reference models, not discovered low-dimensional manifolds or a compression claim. Stop after one frozen comparison; no cluster-count or routing sweep.

Sources: validation/affine_input_predictor_protocol.json, affine_input_fit.json, affine_input_fit_audit.json, affine_input_forward_preflight.json, affine_input_generalization.json, affine_input_natural.json, affine_input_natural_audit.json, affine_input_factorial.json, affine_input_factorial_audit.json, and affine_input_comparison.json. Scripts and small results are in this packet; fitted state and full samples remain in the research artifact directory.

M79 theory preflight now passes 100 random quadratic identity cases and an explicit routing-boundary counterexample. See `check_local_affine_chart_toys.py` and `validation/local_affine_chart_toys.json`. Native chart fitting and evaluation remain pending.
