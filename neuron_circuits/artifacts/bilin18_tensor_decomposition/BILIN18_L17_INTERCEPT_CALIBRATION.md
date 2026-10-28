

## H277 — Discovery-only intercept calibration helps modestly; intact replacement still fails

H276 exposed a gap between conditional response preservation and intact behavior. H277 changes only the constant output vector. Freeze the original compiler and all centered polynomials K. For each variant, fit b=mean_discovery(native−compiler−K), using the first768 discovery documents, four sites each, float64 accumulation and float32 storage. This is the least-squares optimal constant on those discovery states, not a behavior-optimal or test-fitted constant. No gradients, basis selection, coefficient fitting or validation selection are used.

Include zero, exact E(mu), and fitted constants for both actual/mixture polynomials, plus original-compiler and common-term controls. All constants are saved before test evaluation. Fresh128 documents exclude1984 complete-document hashes from nine recent panels; uniqueness, zero overlap and file SHA256 verified. Same corpus/model, not model-training holdout or globally unseen text.512 next-token sites. Native write is ground truth only; approximate inference supplies no native answer.

| Model | Residual fidelity | Mean KL | KL reduction vs compiler | CE increase | Native top1 agreement |
|---|---:|---:|---:|---:|---:|
|compiled|0.00%|0.09226|0.00%|0.11530|84.38%|
|constant_exact|-17.01%|0.09377|-1.64%|0.11543|81.64%|
|compiled_mean|0.62%|0.09146|0.86%|0.11314|84.18%|
|common_exact|34.77%|0.09818|-6.41%|0.12423|82.42%|
|common_cal|41.14%|0.09185|0.44%|0.11806|83.79%|
|actual_zero|44.26%|0.09239|-0.14%|0.11583|84.77%|
|actual_exact|41.78%|0.08511|7.75%|0.10722|81.64%|
|actual_cal|48.18%|0.07720|16.32%|0.09934|84.38%|
|mixture_zero|44.06%|0.09246|-0.21%|0.11314|84.77%|
|mixture_exact|41.63%|0.08536|7.48%|0.10448|82.23%|
|mixture_cal|48.00%|0.07713|16.40%|0.09642|84.18%|

Locked gates: calibration_material: FAIL, bias_dominant: FAIL, repair_actual: FAIL, repair_mixture: FAIL.

- actual, calibrated versus exact E(mu): 9.29% KL reduction, paired document-bootstrap95% [5.53,12.79]%.
- mixture, calibrated versus exact E(mu): 9.64% KL reduction, paired document-bootstrap95% [6.01,13.11]%.

The positive9–10% reductions are below the20% material-calibration and50% bias-dominance gates. Both stronger repair gates fail (>=80% KL recovery, CE increase<=.02 and top1>=.98). On this new panel, actual calibrated recovery is16.3% versus7.7% with E(mu); mixture16.4% versus7.5%. These are within-panel comparisons. H276 used different documents, so its19% exact-anchor recovery must not be compared directly as a treatment difference.

**Response invariance.** Adding a fixed b cancels exactly in [K(z)+b]−[K(x)+b]. The stable evaluator ignores the anchor and is bitwise identical before and after calibration for both programs. Thus all input-response mappings and prior conditional mediation predictions remain the same, even though intact predictions change. This is an explicit degree of freedom left unconstrained by response-only experiments.

**Mean geometry.** Calibrated common/actual/mixture constants differ by only about1e-4 in vector norm (constant norm about2110). This is consistent with the core being natural-covariance principal directions on the same discovery states: centered core coordinates have zero empirical covariance with orthogonal-complement partner coordinates, so the mean cross term vanishes up to numerical error. The basis-dependent couplings can matter for responses even when their discovery mean is zero. The identical constants are not evidence that the state-dependent programs are identical.

**What is and is not ruled out.** A simple MSE-optimal constant accounts for some intact error, but is not enough for faithful behavior. This weakens a dominant mean-bias explanation under that calibration rule. It does NOT rule out every possible constant correction: MSE weights all output directions equally, while final RMS/head/softcap weight directions differently. Missing state-dependent interactions, downstream metric mismatch and optimization/model-class limitations remain open. No semantic unit, unique ontology or intrinsic dimension has been identified.

**Next discriminator.** Optimize only the same1152-dimensional constant for native-distribution KL on discovery states, with a held-out validation split and frozen learning-rate/checkpoint rules. Compare exact E(mu), MSE-optimal and KL-optimized constants on new documents, holding every nonconstant coefficient fixed. This closes the objective-mismatch alternative before claiming state dependence is necessary. Both constant fits must preserve the stable response map exactly. Record absolute KL, CE and top-choice agreement, not only relative improvements, and retain failed gates. If direct behavioral constant optimization also remains inadequate, the evidence for missing state-dependent computation becomes stronger.

**Audits/costs.** Three focused tests pass: least-squares constant optimality identity, response invariance, and anchored/compiled polynomial identities. Stable-response bitwise checks pass on real discovery states. Calibration uses no test statistics. Stored scalar count is unchanged from H276:3,191,040 correction plus504,224 original compiler. Calibrated exports are separate from exact-anchor and response-only files, and remain failed replacement candidates. Full project suite not rerun. All runs complete; native weights and original compiler unchanged; figure inspected. Broad goal remains active.
