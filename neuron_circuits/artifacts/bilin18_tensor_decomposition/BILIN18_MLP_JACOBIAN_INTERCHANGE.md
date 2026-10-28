## H880 — Interchanging local MLP transformations transfers the context-dependent response gap

H879 established that context dependence survives recipient-native MLP linearization. H880 actively interchanges those local transformations across matched copying and capital-recall contexts. Under shared final scale and uncapped readout, giving both contexts the same MLP Jacobians reduces mismatch from 27–36% to 6–9%. Swapping the Jacobians reverses the projection of the response gap onto its original direction. This is stronger causal evidence that the local MLP transformations carry much of this context dependence, rather than merely showing that eliminating MLP responses helps.

### Intervention

Use all 192 H873 contexts and the L8H3 common headwrite at original and quarter amplitudes. Capture native raw pre-MLP states and outputs at L8–17. For recipient c and matched donor d, replace the final-token MLP map with native_output(c)+J(d)*(current_input(c)−native_input(c)). The donor supplies only the derivative; recipient baselines and offsets are preserved. Each Jacobian includes the raw-input RMS normalization and bilinear MLP. Attention remains live.

Five conditions: live model, recipient-native linearization, copy-context Jacobians in both contexts, capital-context Jacobians in both, and Jacobians swapped between contexts. Donors match the same country pair, prompt format and endpoint order. Every condition includes identity and positive/negative interventions. These are posthoc oracle operations on reused contexts, not a learned featurizer, naturally realized model state, or fresh transfer confirmation.

### Competent format C, original amplitude, shared-scale uncapped readout

|Output|Jacobian condition|Relative mismatch|Absolute mismatch RMSE|Copy response norm/live|Capital-context response norm/live|Gap projection on live gap|
|---|---|---:|---:|---:|---:|---:|
|country|live|0.358|0.650|1.000|1.000|1.000|
|country|linearized|0.358|0.651|0.999|0.999|1.001|
|country|both_copy|0.058|0.131|0.999|1.247|-0.040|
|country|both_capital|0.081|0.146|0.795|0.999|0.020|
|country|swapped|0.314|0.712|0.795|1.247|-1.021|
|capital|live|0.271|0.452|1.000|1.000|1.000|
|capital|linearized|0.272|0.453|0.999|0.999|1.001|
|capital|both_copy|0.066|0.110|0.999|1.001|0.038|
|capital|both_capital|0.086|0.143|1.008|0.999|-0.011|
|capital|swapped|0.299|0.500|1.008|1.001|-0.974|

Gap projection is dot(intervention gap, live gap)/squared_norm(live gap), across 12 canonically oriented pair responses. A value near −1 means the component aligned with the original gap reverses with similar magnitude; it does not establish exact vector negation or reversal for every example. Swapped country and capital gaps have projections −1.021 and −0.974. Common-Jacobian conditions reduce the aligned component close to zero while retaining response norm ratios of approximately 0.80–1.25. Thus the reduction is not response collapse.

The quarter-amplitude results are nearly unchanged: common-Jacobian errors 5.8–8.6%, swapped gap projections −1.019 and −0.973. Familiar A/B formats also show reductions (quarter-amplitude errors 2.9–5.1%, versus live 28.4–32.9%). Failed-capital D remains separately saved as a diagnostic.

With the original native readout at original amplitude, common-Jacobian mismatches are larger: country 19.4–24.1% versus live 28.0%; capital 13.4–19.6% versus live 36.1%. Output normalization and saturation still contribute. The strong 6–9% result applies to the controlled readout and should not be described as universal invariance in the native model.

### Interpretation

A reusable source direction can have different output effects because it passes through different local MLP transformations. Those transformations are causally interchangeable to a substantial extent in this matched lexical panel. They do not account for every residual difference, and attention remains context dependent and live. Sharing all ten downstream MLP Jacobians does not tell us which individual layer or parameter factor is necessary, nor prove low rank, unique fundamental units, meaningful semantic manifolds, or unsupervised discoverability.

The intervention requires a matched donor and cached native recipient states. It is evidence about mechanism, not a practical data-based canonical representation. The tested output contrasts and tasks remain narrow compared with the broader research goal.

### Next information-gaining factorization

For raw input x, let r be its RMS and u=x/r. The MLP Jacobian factors as the bilinear derivative at u, followed by (I−u*u^T/d)/r. Context can therefore change three separable factors: scalar inverse RMS, the radial correction from normalization, and bilinear partner activities. Run a factorial interchange of those factors while retaining the recipient native offset and input subtraction. Use both copy and capital donor directions, self and full-swap controls, and predeclare all factor combinations rather than selecting the best one after observing results. Audit the all-native and all-donor endpoints against H880 and the analytical factorization against the independent derivative checks.

This can distinguish a normalization-driven explanation from genuinely different bilinear partner gating before asking a featurizer to learn context-dependent computational units. Broader competent tasks and held-out combinations remain essential after this diagnostic stage.

### Verification

5,952 full-model forwards. Every identity final state, live replay, and self-Jacobian replay is exact. Independently verified 1,920 donor assignments against pair, format, order and intended family. Jacobian checks cover all ten layers: analytic versus autodiff relative error 3.58e-16, versus central finite differences 1.01e-08; double formula versus native update relative error 1.19e-06. Selected readout matches native logits within 7.9e-06. H879 response errors replay within 0. Source hashes unchanged and all outputs finite. Pair-bootstrap intervals (4,096 resamples, uncorrected) saved separately by cohort. No reserved test evaluated; natural document pool unchanged at 26,112. Broad goal active.
