

## H357 — Native context pairing across all nine heads

Reuse H356 documents and confidence neighbors. For each head, native MLP responses to actual ablation, six moment controls and six empirical donor controls are evaluated under own and matched-context directional Fisher metrics. Payload seeds, recipient norm/context-dot constraints and output-Rayleigh statistic are fixed. Head6 uses verified cached H356 rows; other heads computed anew. No compiler output projection or metric-trace normalization.

Primary statistic is geometric mean of per-site mean-control/actual output-Rayleigh ratios, followed by own/matched ratio. Use4000 joint document/control-realization/neighbor-rank resamples. Form one95% maxabsolute-deviation band on log ratios across all18 head×family entries. These bands address selection across this family, conditional on fixed matching and donor assignments.

|Head|Moment ratio [joint95% band]|Donor ratio [joint95% band]|
|---|---:|---:|
|0|0.887 [0.795, 0.989]|0.896 [0.803, 0.999]|
|1|1.102 [0.988, 1.228]|1.060 [0.951, 1.182]|
|2|0.992 [0.890, 1.106]|0.992 [0.889, 1.106]|
|3|0.887 [0.796, 0.989]|0.910 [0.816, 1.014]|
|4|1.079 [0.968, 1.203]|1.036 [0.929, 1.155]|
|5|0.860 [0.772, 0.959]|0.947 [0.849, 1.056]|
|6|1.365 [1.224, 1.522]|1.306 [1.172, 1.457]|
|7|0.997 [0.894, 1.111]|1.011 [0.907, 1.127]|
|8|0.989 [0.887, 1.103]|1.008 [0.904, 1.124]|

Heads whose lower bands exceed1 for both families: [6]. Heads whose lower bands exceed1.2 for both: []. Frozen broad gates require at least5 heads respectively. Outcomes: {'broad_pairing': False, 'broad_strong_pairing': False}.

The broad and strong-broad gates both fail. Head6 is the only head with both lower bands above1; head0 has both upper bands below1 (moment0.989, donor0.999), an opposite-direction effect. Moment controls also show upper bounds below1 at heads3 and5, but donor bands do not. This rejects a broad positive-pattern claim at the chosen criterion and prioritizes head heterogeneity. The near-boundary head0 donor result remains sensitive to uncertainty assumptions.

Audit: all cached head6 Rayleigh values replay exactly; native response and geometry assertions pass per newly computed head; source hashes verified. Per-head response vectors and metrics, aggregate Rayleigh array and joint bootstrap saved. All output energies positive. No native weight changes.

Limits: reused H356 panel; this is cross-head generalization within the final layer, not a new independent sample or depth generalization. Head-specific fixed moments and shared matching structure remain. Bootstrap does not rebuild neighbors or donor dependencies; nonlinear finite-control averaging can shift its center. A failure of a band threshold is not evidence of no effect. These directional output tests do not identify semantic feature units.

Next: use the cross-head pattern to choose a discriminating intervention, while keeping head selection separate from confirmation. Compare natural context pairing with controlled context changes that preserve confidence and residual geometry, then evaluate whether any transferable low-dimensional computation predicts the native response. Broad research goal remains active.
