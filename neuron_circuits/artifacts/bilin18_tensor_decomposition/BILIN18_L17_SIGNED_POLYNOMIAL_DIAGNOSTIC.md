## H427 — Signed polynomial predictions and numerator/denominator accounting

Post-result diagnostic on reused H425/H426 coefficients. Predictions, not new forward evaluations or fresh confirmation. Fixed Fisher metric and linearized input RMS only.

Combination score S = R_union - R_core - R_added. With pooled full energy D and error combination N, S(a)=(N2+a*N3+a^2*N4)/(D2+a*D3+a^2*D4). The common a^2 cancels. Cubic terms determine sign asymmetry, but both numerator and denominator matter.

| Edit amplitude | Predicted raw S | Predicted Gaussian S |
|---|---:|---:|
|-4|0.167964|0.168925|
|-2|0.234178|0.240042|
|-1|0.256964|0.264085|
|-0.25|0.265117|0.269694|
|0.25|0.265602|0.266647|
|1|0.259015|0.253016|
|2|0.238839|0.223574|
|4|0.179155|0.157818|

| Diagnostic contrast | Estimate | Exploratory joint 95% interval |
|---|---:|---|
|positive_amplitude_interaction|-0.022382|[-0.070239, 0.025476]|
|negative_amplitude_interaction|-0.003616|[-0.051474, 0.044242]|
|odd_part|-0.009383|[-0.057240, 0.038475]|
|even_part|-0.012999|[-0.060856, 0.034859]|

Symmetric numerator/denominator accounting of the Gaussian-minus-raw positive amplitude interaction: numerator +0.017129; denominator -0.039510. This averages the two orders of changing numerator and denominator. It is descriptive ratio accounting, not unique causal mediation.
Accounting closure error 2.8e-17; odd rational identity error 6.1e-17. Neither is an independent model evaluation. Bootstrap resamples recipient documents and paired source draws conditional on the fixed donor pool; bands are exploratory and do not replace H426 gates.

Next: evaluate new signed/intermediate amplitudes directly in the controlled model cell, first checking polynomial implementation, then confirm any selected scientific contrast on independent recipients/draws. Exact input normalization and exact KL require separate tests. Total distinct documents remains 10944. Broad goal open.
