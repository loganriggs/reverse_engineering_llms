

## H287 — All-head supervision in the same fixed readout

H286 showed response gains follow the supervised head group. H287 supervises all nine heads without adding input features or readout coefficients. The same64 reads, H280 initialization, intact training minibatches, payload seeds, eight strength/rate fits and400 updates are retained. Exact response statistics sum actual and angle full-removal errors over all heads, normalized by total base error. This weights examples by raw response-error energy, not equally by head or group. Validation eligibility requires intactKL<=1.02 H280 AND response error<=base separately in all four actual/angle x heads0–5/6–8 groups. Lowest intact-KL eligible checkpoint wins, otherwise the predeclared joint fallback applies.

Five checkpoints qualify; selected lambda10/rate.1/step300, matching both split specialists' settings. Zero-strength checkpoints are bitwise identical to H285, and normalization arrays match exactly. No new loss implementation is introduced: the prior sufficient-statistic loss/gradient test covers the algebra, while grouped validation is checked in the saved selection records. The original model/compiler/base remain frozen. This is not a global optimizer or capacity bound.

A fresh common128-document panel excludes3264 prior unique complete-document hashes. Independent full-document/window/file SHA256 and exact source-window checks pass. Both specialists, all-head fit, H280, zero-strength control and base are evaluated on this panel. Model selection uses only discovery validation. Intervals use document-cluster bootstrap, conditional on the frozen fits; no simultaneous-family confidence claim. Response metrics measure predicted changes with a native intact-output anchor; intact KL is measured separately without that anchor.

| Response group | All-head vs base MSE reduction | All-head vs matching specialist | All-head vs H280 |
|---|---:|---:|---:|
|actual_heads0_5|3.36%|-0.67%|9.17%|
|actual_heads6_8|5.07%|-0.67%|13.41%|
|angle_heads0_5|0.16%|-0.57%|2.42%|
|angle_heads6_8|0.63%|-0.31%|3.81%|

Matching specialist means forward fit on heads0–5 and reverse fit on heads6–8, for both actual and angle. Negative reduction denotes worse response error.

- actual_heads0_5 vs specialist: paired95% [-0.83,-0.51]%.
- actual_heads6_8 vs specialist: paired95% [-0.83,-0.50]%.
- angle_heads0_5 vs specialist: paired95% [-0.65,-0.49]%.
- angle_heads6_8 vs specialist: paired95% [-0.45,-0.18]%.

| Model | Fresh intact KL |
|---|---:|
|base|0.055159|
|h280|0.051628|
|optimization_control|0.050723|
|forward|0.050274|
|reverse|0.050916|
|selected|0.050461|

Locked gates: state_retention: PASS, base_retention: PASS, specialist_retention: PASS, response_gain: PASS. State retention requires lower95% KL reduction>=−2% vs zero-strength control. Base/specialist retention require lower95% response-error reduction>=−2% in ALL four groups. Response gain requires>=5% actual-error reduction vs H280 with positive lower95% in BOTH groups. No thresholds relaxed.

**Result and interpretation.** All four locked gates pass. Actual-response MSE is3.36% and5.07% lower than base on heads0–5 and6–8, respectively; improvements versus H280 are9.17% and13.41%. Compared with the matching specialists, actual-response error is about.67% higher in both groups; angle penalties are.57%/.31%. Every retention interval stays inside the2% tolerance. Intact KL versus the zero-strength control changes by a relative reduction.52% with95% interval[−.89,1.90]%, satisfying retention but not establishing superiority. Broader supervision therefore suffices to largely combine these split-specific gains within one fixed64-feature readout. Strong interference is not required to explain the earlier split failure at this effect scale. This does not establish complete response coverage or semantic units. Retention relative to each specialist is stronger than merely beating the original adapter. A small loss relative to specialists need not mean irreducible interference: energy-weighted loss aggregation and intact/response weighting may explain it. Likewise, success would show coverage suffices for these gains, not that these64 coordinates are semantic units or a complete computational basis.

**Next discriminator.** Use the saved response statistics to compute the exact response-only least-squares optimum within the same64-feature linear class, separately for each head group and jointly. Evaluate training and document-validation response errors before a new fresh test. Compare these analytic optima to the state-constrained fits to separate optimization/regularization tradeoffs from linear-class interference. Report rank and conditioning; use a stated pseudoinverse cutoff and sensitivity checks. These are executable fitted readouts, but response-only gains may damage intact predictions and do not identify semantics. This analysis is proposed, not already run.

All jobs terminal; checkpoint arrays, curves, grouped metrics and per-document response arrays retained. Broad research goal remains active.
