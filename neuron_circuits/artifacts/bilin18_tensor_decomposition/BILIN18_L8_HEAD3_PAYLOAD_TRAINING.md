# Head3 payload-supervision discriminator

Ten fits frozen before test; selected on natural plus even-head validation only. All response fits receive the same number of gradient draws.

| Fit | Selected step | Natural R2 | Even | Actual head3 | Other odd | Shuffled3 | Gaussian3 | Isotropic3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0 | 0.837 | 0.454 | 0.205 | 0.734 | 0.366 | 0.224 | 0.518 |
| even_only_rep0 | 900 | 0.878 | 0.444 | 0.289 | 0.759 | 0.396 | 0.207 | 0.517 |
| actual_rep0 | 1400 | 0.879 | 0.477 | 0.573 | 0.756 | 0.401 | 0.326 | 0.525 |
| shuffled_rep0 | 1600 | 0.876 | 0.447 | 0.283 | 0.760 | 0.461 | 0.237 | 0.523 |
| gaussian_rep0 | 1600 | 0.875 | 0.470 | 0.309 | 0.755 | 0.460 | 0.295 | 0.494 |
| isotropic_rep0 | 1200 | 0.877 | 0.439 | 0.294 | 0.759 | 0.411 | 0.209 | 0.529 |
| even_only_rep1 | 1100 | 0.879 | 0.439 | 0.299 | 0.759 | 0.399 | 0.205 | 0.519 |
| actual_rep1 | 1400 | 0.879 | 0.477 | 0.570 | 0.756 | 0.398 | 0.333 | 0.526 |
| shuffled_rep1 | 1600 | 0.875 | 0.469 | 0.265 | 0.763 | 0.479 | 0.248 | 0.514 |
| gaussian_rep1 | 1400 | 0.877 | 0.476 | 0.353 | 0.757 | 0.474 | 0.266 | 0.515 |
| isotropic_rep1 | 1200 | 0.877 | 0.446 | 0.305 | 0.759 | 0.393 | 0.216 | 0.530 |

Locked gates: {'generic_support': False, 'context_pairing': True, 'generic_regularization': False}.

All payloads are norm matched before RMS recomputation. Shuffled/Gaussian/isotropic test draws are independent from fixed training payloads. Two replicates vary SGD and training payload draws, not the text corpus. Full endpoint scores, histories and conditional row-bootstrap contrasts are in JSON. Four L8 coordinates remain a limited output population.


## H231 — Actual source–context pairing beats raw source-span supervision

**Discriminator.** The H230 source-coverage gain was dominated by head3. To ask
what head3 coverage supplies, ten fits varied five conditions and two SGD/sham
replicates. Each update uses42 samples from each even head plus42 supplemental
samples: another even-head example, actual head3, shuffled head3, Gaussian in
head3's raw output-weight span, or isotropic noise. All supplements match each
site's raw head3 norm before RMS recomputation. Rank64, natural fitting data,
1600 updates, exact linear term and discovery-only intercept are fixed. Selection
uses only natural plus even-head validation; head3 cannot select checkpoints.
The two replicates share text and vary fixed sham realizations plus SGD seeds.
All fits froze before fresh test19304–19431.

| Supplement | Actual head3 fidelity, replicate0 /1 | Shuffled3 fidelity, replicate0 /1 | Natural R2, replicate0 /1 |
|---|---|---|---|
| Even-only control | .289 / .299 | .396 / .399 | .878 / .879 |
| Actual head3 | .573 / .570 | .401 / .398 | .879 / .879 |
| Shuffled head3 | .283 / .265 | .461 / .479 | .876 / .875 |
| Gaussian head3 span | .309 / .353 | .460 / .474 | .875 / .877 |
| Isotropic | .294 / .305 | .411 / .393 | .877 / .877 |

The actual-pairing gate passes in both replicates; generic-support and generic
regularization gates fail. In replicate0 actual supervision's advantage over
control is .284 with conditional row-bootstrap95% [.192,.376]; its advantage over
shuffled is .290 [.191,.360], and over Gaussian .264 [.179,.364]. Shuffled training
learns shuffled responses but fails to transfer that benefit to actual head3.
This is evidence about the joint source–state distribution, not semantic identity.

**Mean-bias red team.** Only8.6% of actual response energy lies in its per-axis
mean. Of actual supervision's .284/.272 gain, .252/.241 comes from reducing
centered errors: roughly89%. Centered response fidelity rises .288/.297→.564/.561.
A source-declared correction fitted to the training mean improves the controls
only to .349/.357, well below actual supervision. That correction is a diagnostic
response offset, not an integrable state decoder. Mean response alone cannot
explain the effect.

**Remaining confound identified and tested next.** Equal raw input norms do not
produce equal response or quadratic target energies. Actual targets have larger
energy on several axes. This prompted an explicitly adaptive energy-weighted
follow-up (H232), rather than declaring actual pairing intrinsically necessary.

**Audit.** Native response replay relative RMS1.98e-6; raw norm mismatch <1.2e-7;
head reconstruction <4e-7. Fourteen algebra/control tests pass, including exact
source quotas, actual-payload replay, finite integrability and gradients. One
combined pytest attempt segfaulted during matplotlib import; an unchanged rerun
passed14 tests. No experimental run failed. Reports, frozen factors, raw training
payloads, fresh test inputs, row errors, training histories and bias/energy audits
are retained. Exact Q is a teacher/evaluation oracle, not a decoder input.
