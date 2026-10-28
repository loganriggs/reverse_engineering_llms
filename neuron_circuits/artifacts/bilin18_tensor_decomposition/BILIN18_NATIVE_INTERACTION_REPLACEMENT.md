## H892 — Fixed compact interaction replacement preserves much native behavior and causal response

Replaced the native L12 centered cross component with H891's fixed discovery-derived approximations, retaining the exact context-only and squared terms. Rank1 plus the mean preserves every binary agreement preference on this selected panel, with native margin RMSE0.062–0.088 logits and mean full-vocabulary KL0.000213–0.000401. It also approximates fixed-dose coordinate effects with about7–20% individual error. This is stronger than an intervention-only fit, but does not establish a whole-model or task-blind replacement.

### Native replacement, not a cached-offset intervention

For the frozen centered coordinate t and context vector w, replace F(w)+t*C(w)+t²*Q by F(w)+t*C_hat(w)+t²*Q. At dose0 this changes the native layer computation directly. For doses0.25,0.5,1 toward the opposite-number centroid, use t' in both the approximated cross term and exact squared term, keeping w fixed. Compare exact, mean, rank1, rank7, random7, wrong-family mean and cross-component removal. All fitted quantities are loaded unchanged from H891; no rank or coefficient is refitted.

The approximation uses analytically compiled affine context encoders, rather than requiring the full interaction vector to calculate its coefficients. The full vector is still computed by this diagnostic runner for subtraction and auditing; this is not a deployment speed benchmark. Context-only and squared computations remain exact, and the rest of the model remains live.

Both frozen axes and all120 previously selected task/copy endpoints are evaluated. The family-specific fit is selected using the known task/copy label. New nouns were excluded from the original discovery fit, but have already been used for localization and intervention analysis. Therefore the evidence is neither task-blind nor an untouched external confirmation set.

### Native agreement fidelity, dose0

|Cohort|Axis|Rank1 native KL|Rank1 margin RMSE|Rank7 native KL|Rank7 margin RMSE|Mean margin RMSE|Random7 margin RMSE|
|---|---|---:|---:|---:|---:|---:|---:|
|irregular|A|0.000242|0.077|0.000194|0.037|0.272|0.268|
|irregular|B|0.000213|0.062|0.000187|0.032|0.424|0.420|
|new_regular|A|0.000222|0.071|0.000191|0.031|0.425|0.419|
|new_regular|B|0.000401|0.088|0.000372|0.059|0.480|0.476|

Rank1 native binary is/are accuracy stays100%. Exact target top-1 remains75% on regular tasks but falls from75% to71.4% on irregular tasks, for either axis. The approximation is useful but not exact. Rank1 native KL is approximately5.9–24.8% of the error produced by removing the cross component entirely; that ratio is a comparison of output-distribution errors, not a percentage of semantics captured.

Random7 barely improves on the mean, whereas the learned rank1 direction materially reduces margin error. Rank7 improves it further, but vocabulary-wide gains over rank1 are modest in the weakest cell (new regular axisB). Different metrics need not rank approximations identically.

Copying native KL is smaller (roughly0.000008–0.000103 for rank1), with unchanged binary and exact top-1 correctness. This alone is weak evidence for copying computation fidelity: the number cross-component effect is relatively small there, and earlier copying intervention approximations remained poor. A small native error for a small component is not proof that the copying mechanism has been compressed.

### Causal effect fidelity after replacement

Compare each approximate model's dose-induced margin change relative to its own dose0 baseline with the corresponding exact model's dose-induced change. This separates native replacement bias from the causal effect being approximated.

|Cohort|Axis|Rank1 error, quarter dose|Half dose|Full dose|Rank7 full-dose error|
|---|---|---:|---:|---:|---:|
|irregular|A|0.103|0.103|0.070|0.034|
|irregular|B|0.117|0.116|0.086|0.044|
|new_regular|A|0.088|0.099|0.068|0.031|
|new_regular|B|0.195|0.203|0.181|0.104|

At full dose, mean errors span23–104%, random7 errors23–103%, and cross-removal errors89–102%, compared with rank1's7–18% and rank7's3–10%. Full output-distribution errors at every dose and polarity-separated margin errors are also retained. These are individual-effect metrics, not just agreement of cohort averages.

### Research implication and next generalization boundary

The candidate has progressed from a labeled direction to a compact approximation of an actual native bilinear component that also preserves much of its causal behavior. Its structure is a number coordinate controlling a fixed output vector and multiplying an affine context coefficient controlling another output direction. The remaining context-only and quadratic terms are not compressed. This does not demonstrate that every model feature has this form, that the selected axes are unique, or that a BSF extension would automatically recover them.

The next boundary is selection and context breadth. Freeze a single agreement-trained unit and apply it without switching fits by known task labels on a fresh, predeclared mix of grammatical, copying and natural-text contexts. Measure native distribution fidelity and component effect strength separately, including contexts where number decoding or task competence is weak. This checks whether the current success depends on manual family selection. Only after that should a learned task-blind gate or richer featurizer be credited with recovering the component. The outstanding copying lexical-coverage learning curve remains part of the broader unresolved objective.

### Verification

6,840 full-model forwards, matching the declared count. Native and exact dose0 identity logits replay exactly. Bilinear decomposition relative error 1.98e-15; compiled rank encoders agree with explicit PCA projection within relative 1.36e-15; double formula versus float module relative error 8.24e-07. Exact full-dose effects reproduce H889 within 0. All source/fit hashes unchanged; all fidelity KL values finite. No reserved test evaluated; natural-document pool unchanged at26,112. Broad research goal active.
