## H431 — Frozen source-by-recipient cross with finite interventions

H430 was progress: it identified recipient normalization gain and source concentration. This post-selection diagnostic applies exactly the same48 frozen vectors to the selected tail site797 and16 ordinary recipients at token position63. The vectors are the12 raw-donor and12 Gaussian draws assigned to the tail in H429, each with its reflection around the fixed mean. Controls are the16 sites closest to median baseline statistics, selected without intervention-response scores. Thus source selection is conditioned on the tail case, and these are not random population controls or fresh confirmation. No new documents; total11456.

Compare original source norms with all48 vectors rescaled to one common median norm. Cross exact native input-RMS/MLP responses with the linearized-input-RMS/full-bilinear response. For each response measure raw vector squared norm, baseline Fisher energy and exact finite output KL(target || intact). The direct residual path remains fixed. Finite native responses use native float arithmetic; the linearized-input path is double and casts response increments to float for native final RMS and exact KL.

| Source norms | Edit size | Raw response squared-norm ratio | Fisher-energy ratio | Exact-KL ratio |
|---|---:|---:|---:|---:|
|Original|0.25|1.535|62.704|53.503|
|Original|1|1.583|65.466|29.883|
|Original|4|1.489|32.760|8.358|
|Common norm|0.25|1.443|42.115|41.512|
|Common norm|1|1.474|45.755|40.948|
|Common norm|4|1.483|50.881|21.240|

Ratios are mean tail effect divided by mean effect across the16 controls and the same48 source vectors. They are not mean per-source ratios. At ordinary amplitude1 with original norms, absolute mean KL is0.057616 at the tail versus0.001928 in controls. With common norms it is0.042154 versus0.001029. A large relative difference should be read alongside these absolute effects.

The original source quartic-energy tail/control ratio is2660.2; common-norm is1725.2. Nevertheless, the native finite raw-response ratio stays near1.5 across amplitudes. At amplitude4, the linearized-input model gives Fisher/KL ratios542.77/73.63, while exact native input normalization gives32.76/8.36. Therefore the polynomial tail substantially exaggerates large finite responses in this selected comparison, while a finite probability-distribution sensitivity difference remains.

Output normalization is material but not the only output-side factor. Holding output derivative projectors and directions fixed and algebraically removing its scalar inverse-RMS gain reduces the ordinary-amplitude original-norm Fisher ratio from65.47 to7.04; for common-norm sources it reduces45.75 to4.91. This is a derivative-metric diagnostic, not exact-KL mediation. Remaining differences can arise from response direction, unembedding alignment, radial projection, logit-cap derivatives and probability weighting. Entropy alone does not determine that geometry.

Numerical audit: all48 original source quartic coefficients at the tail replay H429 with maximum relative discrepancy2.2e-14, below frozen1e-7 tolerance. Arrays have shape48 sources x2 norm conditions x2 response types x3 amplitudes x17 recipients x3 metrics and are finite. The GPU run saved all results, then JSON serialization failed on a numpy boolean; export was fixed and results were reconstructed from saved arrays without rerunning the experiment.

Interpretation: this selected context is behaviorally sensitive to a fixed collection of signals, even after source norms are equalized. Raw MLP response magnitude alone is a poor proxy for its output effect. Both finite input normalization and context-specific output geometry materially change how a proposed computational unit appears. These findings do not establish semantic features, the population prevalence of this behavior, or unique causal fractions for each mechanism. H429 statistical gates remain failed, and no high-sensitivity documents are removed.

Next discriminating direction: compare the same response vectors under swapped recipient output metrics, and compare equal-length response directions. That separates receiving-state output geometry from special MLP response directions more directly than another source-distribution comparison. Inspect which token probabilities account for the KL differences, then test predeclared predictions on other contexts. Return to factor-group recovery under finite KL after this metric/sensitivity control; a compact unit should transfer behaviorally rather than merely capture response energy.
