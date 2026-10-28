

## H276 — Conditional response success does not survive intact substitution

H275 compiled a frozen cross-core residual rule and found conditional behavioral improvement, but supplied the correct native intact write. H276 removes that baseline. For each normalized input x, approximate the entire final MLP write with original_compiler(x)+E(mu)+K(x), where E(mu) is the exact native-minus-compiler polynomial value at the frozen discovery mean and K is the frozen H275 centered correction. The constant is computed once from weights, not fit to examples. No coefficient, partner basis or mean is refit. Native Down_bias is added once outside both native and approximate writes.

Controls separate original compiler, constant-only correction, constant+common mean-linear/within-core terms, and full actual/mixture partner corrections. Native write is used only as ground truth, never in approximate forward inference. Retain final RMS/head/softcap. This is final-layer substitution with native upstream states, not whole-model compression or multi-layer replacement.

Fresh128 documents exclude1856 unique complete-document hashes from eight recent panels; uniqueness, exclusions and file hash verified. Four sites per document,512 next-token predictions. Same corpus/model; not global or training-set freshness. Float32 forward and float64 KL/CE. Frozen criteria: mean CE increase<=.02 nats AND native top1 agreement>=.98; stronger repair additionally requires>=80% KL reduction versus original compiler.

| Approximation | Residual-error fidelity | Native-write fidelity | Mean KL | KL reduction | CE increase | Native top1 agreement |
|---|---:|---:|---:|---:|---:|---:|
|compiled|0.00%|96.89%|0.10787|0.00%|0.12519|85.94%|
|constant|-15.01%|96.42%|0.10578|1.94%|0.12398|84.18%|
|common|38.72%|98.09%|0.10057|6.76%|0.13352|85.94%|
|actual|46.99%|98.35%|0.08728|19.08%|0.11926|86.91%|
|mixture|46.86%|98.35%|0.08763|18.76%|0.11912|86.52%|

Locked gates: intact_actual: FAIL, repair_actual: FAIL, intact_mixture: FAIL, repair_mixture: FAIL.

- actual: KL recovery95% CI[8.14,28.38]%; CE increase CI[0.0766,0.1652] nats; top1 agreement CI[83.79,90.04]%.
- mixture: KL recovery95% CI[7.75,28.10]%; CE increase CI[0.0759,0.1641] nats; top1 agreement CI[83.20,89.65]%.

Intervals use4,000 document-bootstrap draws. Both corrections significantly reduce KL relative to the compiler in this panel, but are far from replacement thresholds. Neither KL improvement nor a native-write energy fidelity above98% rescues the failed behavioral claim. Common terms recover39% of residual energy but only7% of KL; full corrections recover47% residual energy and19% KL. The original residual is small relative to the native write, yet behaviorally consequential.

Constant-only anchoring worsens residual-energy fidelity to−15% and has no clear KL benefit. E(mu) is the polynomial value at the mean input; it is NOT the mean residual over natural states. Quadratic expectations need not equal evaluation at the mean. This distinction provides a specific alternative explanation for some remaining intact error without affecting the already-measured responses.

Next-token accuracy against corpus labels is38.3% native,38.1% compiler and35.4% for either full correction on512 sites. These are descriptive point estimates, not a preregistered accuracy test or a claim of significant degradation. Native top1 agreement and labeled accuracy measure different things. The CE increases and failed agreement gates already establish inadequate replacement.

**Mechanistic interpretation.** The interacting-subspace rule captures reproducible response structure and some intact residual structure, but it is not an adequate account of the whole local computation. A response-preserving description can omit behaviorally important state information. This strengthens the requirement to separate local energy coverage, intervention fidelity and intact downstream behavior when proposing computational units. It does not establish semantic features, unique coordinates or a minimal representation.

**Next discriminator: intercept versus missing state dependence.** On discovery documents only, estimate the mean remaining residual for each frozen program, changing only its output constant. Such a constant leaves K(z)−K(x), the stable response evaluator and every conditional response prediction exactly unchanged. Freeze the calibrated constant and test intact behavior on independent documents. If this repairs much of the gap, response experiments left a behaviorally important constant unidentified; if not, missing state-dependent interactions remain necessary. Include exact E(mu), discovery-mean and zero-constant controls, with no changes to bases or quadratic coefficients. Do not use a test-set mean as a deployable correction.

**Costs and exports.** Adding the anchor costs1,152 scalars: each standalone correction now3,191,040 scalars, plus504,224 for the original compiler, total3,695,264 (about23.2% of native MLP weights, bias excluded). Dense linear/full-output costs remain. Separate *_intact_program.npz files preserve the earlier response-only exports. intact_forward in the standalone evaluator requires these anchored files. These approximations failed behavioral criteria; exporting them does not imply successful replacement.

**Audits.** Two focused tests pass: arbitrary-input compiled polynomial/response identity and anchor matching at mu without changing responses, including nonzero offsets and compiler biases. Common centered correction is zero at mu; anchor float32 rounding is measured in JSON. Full project suite not rerun. No native answer is used in approximate inference. All jobs complete; native weights/original compiler unchanged; figure inspected. Broad research goal remains active.
