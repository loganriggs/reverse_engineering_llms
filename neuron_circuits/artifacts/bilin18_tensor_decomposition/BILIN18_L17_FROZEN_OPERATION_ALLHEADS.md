## H459 — Cross-head boundary of the frozen operation

Reuse H458128 receiving contexts. Draw12 new natural donor documents from the fixed pool and use each donor’s writes from all nine attention heads (108 head-major sources), at amplitudes−1/+1. There are no Gaussian controls in this experiment. Freeze the head6 source-coordinate transform, source16 rational maps/quadratic writes, joint256 input maps and retained128 factors; no head-specific refitting or norm matching. Probabilities enter evaluation only, not prediction. Total12992 corpus documents, no new receiving-document confirmation.

The exact retained response uses each original head signal, while the omitted predictor reconstructs it through the same head6 source coordinates. Compare dense-source16, compact-source16 and retained-only responses against native full interventions. A failure can therefore diagnose scope limits of this specific operation, not absence of a useful operation for another head.

| Head | Source reconstruction error fraction | −1 dense / compact / retained KL recovery | +1 dense / compact / retained KL recovery | +1 dense raw recovery |
|---|---:|---|---|---:|
|0|0.9965|15.65% / 16.86% / 17.51%|24.05% / 24.74% / 24.76%|44.53%|
|1|0.9687|55.51% / 57.23% / 60.25%|50.98% / 52.43% / 55.59%|91.61%|
|2|0.9771|33.52% / 32.82% / 31.22%|36.98% / 36.26% / 34.11%|62.56%|
|3|0.9957|51.05% / 51.51% / 50.16%|49.18% / 49.69% / 48.54%|67.69%|
|4|0.9829|32.90% / 31.17% / 30.30%|32.96% / 31.33% / 30.19%|50.48%|
|5|0.9945|31.73% / 31.10% / 30.12%|29.92% / 29.48% / 28.10%|33.80%|
|6|0.0092|99.49% / 97.77% / 78.54%|99.54% / 98.10% / 79.51%|99.94%|
|7|0.9902|43.29% / 43.50% / 43.54%|42.33% / 42.51% / 42.38%|70.28%|
|8|0.9713|77.42% / 77.36% / 76.88%|78.07% / 78.04% / 77.71%|85.47%|

Source error fraction is pooled squared error of hhat versus h divided by source energy. It is not variance explained by an orthogonal projection: the frozen whitened source-coordinate reconstruction can be oblique. Raw response recovery measures native MLP-write changes; finite KL recovery weights token probability effects.

| Frozen off-head mean minus head6 contrast | Estimate | Joint95% interval | Lower > −.05 |
|---|---:|---|---|
|nonhead6_minus_head6_dense_negative|-0.568608|[-0.631723, -0.505494]|False|
|nonhead6_minus_head6_compact_negative|-0.550808|[-0.613922, -0.487693]|False|
|nonhead6_minus_head6_dense_positive|-0.564812|[-0.627926, -0.501697]|False|
|nonhead6_minus_head6_compact_positive|-0.550378|[-0.613493, -0.487264]|False|

Primary compares the unweighted mean of eight non-head6 per-head recovery scores to head6 at each sign for dense/compact methods, with a5pp tolerance. This avoids letting high-effect heads dominate pooled fidelity.4000 paired donor/receiving-context bootstrap replicates preserve donor pairing across heads; heads themselves are the fixed target set and are not resampled. Joint95% max-absolute-deviation intervals cover all four contrasts. Per-head patterns and correlations are descriptive.

Audits: rational formula versus direct source-restricted double intervention relative error7.21e-13; native float32 source-restricted error9.18e-05. All numerical gates pass, all arrays finite, frozen hashes checked and no head-dependent exclusions. These audits confirm implementation for hhat, not adequacy of hhat for the original source.

The frozen operation fails cross-head transfer decisively. The mean non-head6 minus head6 recovery gap is−56.86pp dense/−55.08pp compact at−1 and−56.48pp/−55.04pp at+1. Joint95% intervals are respectively[−63.17,−50.55], [−61.39,−48.77], [−62.79,−50.17], [−61.35,−48.73]pp. All5pp tolerance gates fail by wide margins; this is stronger evidence of a scope limit than earlier near-threshold compactness failures.

Head6 retains99.49%/99.54% dense and97.77%/98.10% compact recovery at−1/+1. Other-head positive dense recovery ranges24.05% (head0) to78.07% (head8). Their source-reconstruction squared-error fractions range96.87–99.65%, versus0.916% for head6. The frozen head6 source space therefore misses nearly all raw source information from other heads. This aligns with, but does not by itself causally isolate, the response-transfer failure: retained factor groups are also head6-selected and other source properties differ.

For non-head6 signals, the correction often adds little over retained-only. Positive examples: head0 dense24.05% versus retained24.76%, head1 dense50.98% versus55.59%, head7 dense42.33% versus42.38%, head8 dense78.07% versus77.71%. Some other heads receive modest gains; no universal benefit is claimed. Compact sometimes scores above dense because changing an inaccurate source-conditioned response can accidentally reduce error; that does not make compression the main problem. Both retain the same inadequate source reconstruction.

The appropriate interpretation is a source-conditional operation, not a universal late-MLP unit. High prior recovery and a compact rational formula did not establish architectural-source coverage or semantic generality. This result materially narrows the candidate ontology: source coordinates and their scope must be included in the description of the computational object.

Next distinguish head6-specific low dimensionality from a family of small source-specific operations. Fit rank16 source directions separately for each head using disjoint calibration donor writes only, freeze before receiving-intervention evaluation, and compare source reconstruction plus exact rational omitted prediction against the current fixed-head6 space. Start with the uncompressed context-map control so a head6-trained input projection cannot obscure source coverage. Include a shared equal-total-budget source basis as a control only with explicit storage accounting. If own-head rank16 restores prediction broadly, a family of source-conditional units is supported; if several heads require much larger source rank, head6 was unusually simple. Treat this as testing an architectural factorization, not labeling the head spaces semantically. Follow with context-input compression only after source sufficiency is established.

The source reconstruction, native retained factors and baseline omitted write are structured parts of the predictor. Head labels are architectural source labels, not semantic feature labels. No inference speedup, universal computational basis or whole-model compression is claimed.
