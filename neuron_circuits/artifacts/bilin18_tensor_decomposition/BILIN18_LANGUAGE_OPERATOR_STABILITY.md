## H911 — Disjoint country sets recover similar language operators, but not an identical canonical map

The previous turn made progress: H910 demonstrated frozen L13 context-map transfer to competent new-country examples, with weaker L14 results and wrong-context controls. H911 asks whether disjoint discovery countries recover comparable operators.

First discovery uses the original four old country pairs; second uses the other four old country pairs. Each contributes16 endpoints across templates A/B, separately for task and copying. First means remain the exact H908 saved values; second means are fitted only to old countries, never the new-country outcomes. The newly fitted maps are frozen before evaluation. Evaluate all96 new-country rows from H909 at L13/14, including failed-gate C; this panel is now reused diagnostic data, not fresh confirmation.

The explicit linear operator is J(m)=Down[diag(Right m)Left+diag(Left m)Right]. Compare mean-vector cosine, full-matrix Frobenius cosine/difference, operator actions on new-country paired changes, and actual downstream effects. Every map is independently applied to task and copy rows; no runtime map selection is learned.

### Operator recovery across disjoint discovery sets

Relative differences use the first fit as denominator and are asymmetric. High cosine does not imply small relative error.

|Layer|Context family|Mean cosine|Operator cosine|Relative operator difference|Relative action difference on matching-family inputs|
|---|---|---:|---:|---:|---:|
|13|task|0.988|0.959|0.292|0.307|
|13|copy|0.993|0.931|0.381|0.391|
|14|task|0.985|0.953|0.314|0.340|
|14|copy|0.992|0.933|0.379|0.390|

Task means have cosine~0.985–0.988, but their operators differ29–31% in Frobenius norm and their actions differ31–34% on language changes. Copy means are even closer geometrically, yet their operator/action differences are larger (~38–39%). This directly shows why mean-vector stability alone is insufficient.

### Causal fidelity on competent new-country task cells A/B/D

Ratios are relative to doing nothing: target-margin effect RMSE and KL(exact-edited || approximation). Lower is better. These cells contain36 task endpoints with repeated countries/templates, not36 independent discoveries.

|Layer|Discovery set|Task-map margin-error ratio|Task-map KL ratio|Copy-map margin-error ratio|Copy-map KL ratio|
|---|---|---:|---:|---:|---:|
|13|first|0.346|0.172|0.761|0.814|
|13|second|0.412|0.221|0.756|0.798|
|14|first|0.792|0.454|0.877|1.021|
|14|second|0.849|0.643|0.845|1.095|

At L13 both task maps substantially outperform the copying-context controls. Their intervention-effect profiles correlate0.987, with between-fit effect RMSE0.100 logits, but the second map is less faithful (margin ratio0.412 vs0.346, KL ratio0.221 vs0.172). This supports a recurring approximate operation without establishing uniqueness or perfect recovery.

L14 remains weak on the new countries: task-map margin ratios0.792/0.849 despite effect-profile correlation0.977. Reproducibility is not the same as fidelity. Copying maps also vary: L13 copying margin ratios0.541/0.781 and KL ratios0.421/0.558. Language maps remain worse than no change on copying, preserving the context-specific limitation.

All new-country C results remain in the analysis as failed-gate diagnostics. No discovery set, layer or map is selected based on these outcomes. The first maps reproduce H910 margins exactly.

### Next discriminator

Full-operator cosine can be dominated by processing common to task and copying. Next compare the task-minus-copy operator contrast across the two disjoint discovery sets, including its spectrum and its actions on held-out changes. This asks whether the task-specific change in processing is stable, rather than merely a shared background map. Low-rank contrast approximations would then require live intervention verification; a matrix spectrum alone cannot validate a fundamental unit or computational DAG.

The current method retains original model weights and full lexical changes, with supervised context definitions. It does not recover unsupervised task labels or prove a low-dimensional manifold. Binding pathways and broader shared-dictionary/featurizer work remain open.

### Verification and execution correction

The first execution completed both layer evaluations but failed in final result aggregation due to a variable-scope bug. The aggregation code was fixed and the unchanged experimental design rerun fully. The retained run has1,344 full-model forwards; including the initial completed evaluations, this turn used2,688. The failure log is retained locally.

Identity, baseline-logit replay and normalized-activation replay are exactly zero error. Original discovery means reconstructed from the newer captures agree within relative1.60e−7; original saved means, rather than reconstructed values, were used for first-set evaluation. Original map target margins reproduce H910 exactly. All source/model/mean hashes unchanged within the successful run; newly fitted means frozen across evaluation; KL finite. No natural or reserved-test evaluation; document pool remains26,112. Goal active.
