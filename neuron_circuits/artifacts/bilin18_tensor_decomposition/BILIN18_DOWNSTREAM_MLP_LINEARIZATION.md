## H879 — Context dependence survives local MLP linearization

H878 showed that freezing downstream MLP responses largely removes early-reader context dependence under controlled readout. H879 distinguishes different local transformations from nonlinear changes over the finite intervention. Replacing each downstream MLP map with its recipient-native first-order approximation preserves almost all measured mismatch. This supports context-specific local transformations as a sufficient explanation at the tested scales, rather than requiring large finite nonlinear transitions.

### Design and implementation

All 192 H873 contexts, L8H3 common headwrite, amplitudes 0.25, 0.5, 1 and 2 times the previous amplitude. Compare live MLPs, recipient-native linearizations, and frozen native MLP updates, for every MLP L8–17. Attention stays live. Each condition includes positive, negative and identity arms. Readouts are native, shared-fixed, and shared-fixed without cap. Matched-context scales and pair-conditioned headwrites retain their prior oracle-diagnostic interpretation.

The linearized map is the complete raw-pre-MLP map F(x)=Down[(Left RMSNorm(x))*(Right RMSNorm(x))]+bias. At each layer use F(x_native)+J_native(x−x_native). The native offset is cached directly from the float32 model; the analytic Jacobian-vector product is evaluated in double precision and cast back. The RMS derivative includes its radial subtraction and the actual float32 epsilon. It is not a linearization only of the bilinear map after normalization. Reconstruct raw pre-MLP states from the block's residual/anchor mixture plus its attention update, following the native recurrence.

The approximation preserves the local derivative of each recipient context. It is not a shared Jacobian, task-blind feature basis, globally linear model, or approximation around the perturbed state. Attention remains nonlinear, so this experiment isolates finite nonlinearities in MLP maps, not every nonlinearity in the model.

### Competent format C, shared-scale uncapped readout

|Output|Amplitude multiplier|Live mismatch|Linearized mismatch|Frozen mismatch|Largest linearized-vs-live canonical response error across contexts|
|---|---:|---:|---:|---:|---:|
|country|0.25|0.3586|0.3587|0.0212|0.0001|
|country|0.5|0.3585|0.3586|0.0211|0.0004|
|country|1|0.3578|0.3584|0.0211|0.0014|
|country|2|0.3553|0.3577|0.0209|0.0056|
|capital|0.25|0.2718|0.2718|0.0312|0.0001|
|capital|0.5|0.2717|0.2718|0.0312|0.0004|
|capital|1|0.2713|0.2718|0.0311|0.0015|
|capital|2|0.2696|0.2717|0.0309|0.0059|

At original amplitude, linearized canonical responses differ from live responses by only about 0.1–0.2%, while copy/capital context mismatch remains about 27–36%. At twice the amplitude the canonical approximation errors remain below 0.6%. Reducing amplitude to one quarter does not remove the mismatch. Thus the response difference is already present locally.

Centered positive/negative responses can cancel even-order effects. To avoid relying solely on that cancellation, the analysis also compares each signed endpoint response with its own identity baseline, without averaging pairs, orders or signs. These checks are summarized below.

|Amplitude multiplier|Largest aggregate error across output/family cells, unaveraged one-sided responses in C|
|---|---:|
|0.25|0.0194|
|0.5|0.0388|
|1|0.0771|
|2|0.1513|

At original amplitude, these aggregate errors across individual signed responses are 4.7–7.7%; at twice the amplitude they rise to 9.3–15.1%, and at one quarter fall to 1.2–1.9%. These are norm errors over all signed examples within each cell, not maximum per-example errors. They qualify the centered-response conclusion: local linearization need not exactly reproduce every finite signed intervention. Full results include native readout and separate A/B and failed-capital D cohorts. No failed-competence cohort is promoted to semantic transfer evidence.

### Interpretation and next discriminating test

The evidence favors context-dependent local MLP transformations acting on reusable source directions. This does not yet identify why the Jacobians differ: normalization geometry, bilinear partner activations, and upstream differences entering each MLP can all contribute. It also does not make the Jacobian itself a fundamental feature or demonstrate a low-dimensional manifold.

Next, interchange the native Jacobians between matched copy/capital contexts while retaining each recipient's own native offset. Compare native-context Jacobians, copy-context Jacobians in both contexts, capital-context Jacobians in both, and swapped Jacobians. Attention remains live. This tests whether sharing the transformation itself makes the response reusable, rather than merely removing it. Audit recipient identity and self-Jacobian replay, and track absolute response strength. Subsequent factorization can separate RMS geometry from bilinear partner gating if the swap is informative. These remain causal diagnostics on the reused panel; fresh diverse task transfer and stable unsupervised units are still required for the broader goal.

### Verification

Completed 7,104 full-model forwards in the final audited run. All identity final states and original-amplitude live states replay exactly. Analytic Jacobian-vector products checked on all ten layers agree with independent autodiff within relative 3.58e-16 and central finite differences within 1.01e-08. Double formula versus native MLP update relative error is 1.19e-06; maximum coordinate discrepancy 0.0277 is retained because updates are large. Selected readout error is at most 9.91e-06. H878 live/frozen response errors replay within 0. An initial complete run was repeated after adding the explicit relative formula-reconstruction audit. All source hashes unchanged, all outputs finite; no reserved test evaluated, natural-document pool unchanged at 26,112. Broad goal active.
