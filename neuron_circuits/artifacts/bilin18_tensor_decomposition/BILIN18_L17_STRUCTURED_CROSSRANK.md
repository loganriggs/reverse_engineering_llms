## H490 — Cross-output rank at fixed four-source structure

Hold the four calibration source axes,1024 training receivers,32 training donors, exact source quadratic core, scalar RMS denominator and native baseline write fixed. Compare shared cross-output ranks32/64/128, obtained as nested truncations of the same training contribution SVD. No new input/source dimensions or learned coefficient regression. Validation on256 receivers/64 donors chooses the smallest rank with full-source KL recovery>.95 at both signs, otherwise128 with a failure flag. Rank64 meets validation at95.14/95.65% and is selected before confirmation collection.

Use512 new complete-document-hash-disjoint receivers. Independent audit rehashes17856 excluded documents and512 new documents, verifies all129-token windows, and finds zero overlap. Total tracked documents18368. Source family/calibration stays fixed; freshness is within this continuation, not a model-training holdout.

| Cross-output rank | Fresh full-source KL recovery −/+ | Raw recovery −/+ | Conditional model-array scalars |
|---|---|---|---:|
|32|93.007% / 93.476%|99.231% / 99.224%|207360|
|64|94.138% / 94.508%|99.320% / 99.313%|391680|
|128|95.392% / 95.691%|99.414% / 99.406%|760320|

Scalar counts include four1152xr read maps, shared rx1152 write,1152x4 source basis and4x4x1152 quadratic core. They exclude native baseline computation, bias subtraction and a fixed epsilon constant. The archive holds all candidates, not just the selected model. No runtime speedup or complete-model compression claim. With an uncompressed cross operator, the same source4 model reaches98.30/98.14% on this panel, so both source truncation and cross compression still contribute to error.

| Predeclared primary statistic for selected rank64 | Estimate | Joint95% interval | Threshold | Pass |
|---|---:|---|---:|---|
|absolute_negative|0.9414|[0.9254, 0.9574]|0.95|False|
|cross_gain_negative|0.0113|[0.0062, 0.0164]|0|True|
|absolute_positive|0.9451|[0.9336, 0.9566]|0.95|False|
|cross_gain_positive|0.0103|[0.0063, 0.0143]|0|True|

Both selected-rank gain gates pass: rank64−rank32 gives1.13/1.03pp with joint intervals[.62,1.64]/[.63,1.43]pp. Both absolute>.95 gates fail: selected rank64 reaches94.14/94.51%, with lower bounds92.54/93.36%. Rank128 has95.39/95.69% point recovery but was not selected. It is a predeclared candidate evaluated descriptively, not a post-confirmation replacement for the chosen model. No threshold guarantee for rank128 is claimed.

Raw recovery is already99.32/99.31% at selected rank64, yet probability recovery remains~94%. More raw-energy-ranked directions help, but the cost rises from207360 array scalars at32 to391680 at64 and760320 at128. This motivates testing the subspace-selection metric at fixed rank rather than assuming further energy-ranked width is the most informative next step. Per-case errors can be concentrated; the absolute bands reflect receiver-sampling uncertainty conditional on fixed training/source choices.

Four joint standardized bootstrap bands use4000 paired within64-donor resamples, eight receivers each, shared across ranks and signs. Training-sample, basis-calibration and donor-pool uncertainty are not bootstrapped. Validation has adaptive history; the fresh confirmation is the transfer test.

Audits: cross32 validation metrics exactly replay H489 source4/cross32; source basis orthogonality and exact projected-source rational delta tests pass; all arrays finite; selections and compiled factors saved before confirmation collection; independent document/window audit passes. Candidate read maps, writes, source bases, quadratic cores, per-case errors and native baseline inputs archived. No process remains active.

Interpretation: source geometry and normalization-aware structure provide a useful conditional representation, but a Euclidean data-weighted subspace is not necessarily a probability-efficient one. The higher-rank gains do not establish unique feature axes or full-layer replacement. Native baseline write remains required, as before.

Next fix source4 and cross-output64 and optimize the output subspace for actual finite KL rather than training cross-term energy. Initialize at the current data SVD, retain an orthonormal output basis, and compile reads directly from the known cross operators after each basis update. Only basis orientation changes; scalar normalization and quadratic terms remain exact. Use training-only line search, validation stopping including the initialization, and a newly reserved confirmation panel. Unlike prior generic coefficient refinement, this asks which subspace of a known structured cross computation to retain, while coefficient reads remain exact for that subspace. It separates basis metric from additional rank; no convergence or success is presumed. Observed-token CE, native-baseline removal and cross-head/layer transfer remain open.
