## H933 — Causal prediction can be stable while prototype partitions are not unique

H932 showed partial template transfer but did not test sensitivity to the lexical fit sample. H933 partitions the original four discovery pairs per domain into two disjoint two-pair subsets. Each fit uses24 examples from only one template; all models are evaluated on the same96 last-four-pair examples across both templates. Run both template directions,1/4/8 prototypes and all three seeds. All conditions and failures remain included.

This is a deliberately small-sample stress test. It changes both lexical coverage and sample size relative to earlier fits; instability here does not isolate sample identity from limited data. The evaluation panel is familiar from earlier research, not an untouched holdout.

### Partition stability

Adjusted Rand index (ARI) compares assignments on the common unedited evaluation queries, ignoring arbitrary cluster numbering. Each range covers all nine seed combinations across the disjoint training subsets;1 indicates identical partitions. Prototype counts4 and8 are shown;1 is trivially stable.

|Fit template|Reader|Prototypes|Cross-sample ARI range|Mean ARI|
|---|---|---:|---:|---:|
|A|L8H3|4|0.599–1.000|0.803|
|A|L8H3|8|0.824–0.956|0.905|
|A|L13H0|4|0.399–0.614|0.473|
|A|L13H0|8|0.714–0.843|0.774|
|A|L14H4|4|0.489–0.599|0.526|
|A|L14H4|8|0.630–0.952|0.826|
|B|L8H3|4|0.393–0.816|0.606|
|B|L8H3|8|0.706–0.845|0.790|
|B|L13H0|4|0.223–0.965|0.457|
|B|L13H0|8|0.814–0.944|0.896|
|B|L14H4|4|0.333–0.599|0.475|
|B|L14H4|8|0.629–0.961|0.824|

The clusters are only partly reproducible across samples. Eight prototypes are generally more reproducible than four in this setup, but none of the highlighted nontrivial settings establishes a unique partition. Earlier identical seed partitions at L13H0 do not establish sampling robustness; that earlier fit also used more examples and both templates, so this is not a pure one-factor comparison.

### Causal prediction stability

For each same-seed pair of disjoint fits, compare predicted donor-value answer-margin effects on the common evaluation panel. Normalize disagreement by the native-routing reference effect RMS. The error range compares each of the six fitted models with that reference. These metrics pool both evaluation templates; per-template endpoint results remain available in the run files.

|Fit template|Domain/family|Cross-fit prediction disagreement ratio|Effect-error range, eight prototypes|
|---|---|---:|---:|
|A|agreement task|0.841–0.858|0.870–0.976|
|A|agreement copy|0.247–0.271|0.277–0.326|
|A|language task|0.061–0.067|0.105–0.169|
|A|language copy|0.172–0.709|0.320–0.726|
|A|binding task|0.033–0.033|0.083–0.098|
|A|binding copy|0.105–0.226|0.114–0.320|
|B|agreement task|2.645–2.645|2.555–5.098|
|B|agreement copy|0.163–0.302|0.315–0.429|
|B|language task|0.095–0.100|0.139–0.237|
|B|language copy|0.252–0.899|0.456–0.867|
|B|binding task|0.033–0.052|0.082–0.130|
|B|binding copy|0.132–0.179|0.114–0.269|

Binding task predictions disagree by only3.3% of reference effect under A fits, and3.3–5.2% under B fits, despite imperfect partition agreement. Their effect errors remain about8.2–13.0%. Language task predictions disagree by6.1–10.0%, with errors10.5–23.7%. Language copying is much less robust: disagreements can approach90% and effect errors87%. Agreement task ratios are again dominated by a very small reference effect at these readers; absolute metrics should be consulted before interpreting those ratios.

This distinguishes reproducibility of a cluster ontology from reproducibility of an operation. Different partitions can support similar causal predictions. Stable predictions do not identify a unique feature grouping, and unstable grouping alone does not establish that the approximated computation is wrong. Both must be measured.

### Research decision and next experiment

Do not label one prototype partition as the model's settled semantic units. The evidence supports a family of useful context discretizations for parts of language and binding, with weaker copying fidelity. This matches the broader possibility that functionally equivalent data-based representations can differ in internal grouping.

Next compare frozen prototype means with a small continuous query-to-coefficient model and a prototype-plus-continuous-correction model, using the same discovery/evaluation separation and fixed regularization choices. Preserve task-blind fitting and report every predetermined setting. Evaluate native KL, donor-intervention KL and effect fidelity, alongside cross-sample stability. A correction that reduces baseline drift but worsens causal response is not a clear improvement. If continuous-only prediction matches the grouped model, discrete groups may be an explanatory convenience rather than necessary computational structure. This comparison should precede a manifold or hierarchy claim.

### Verification

11,136 forwards across four runs (2,784 each). Every run passes identity replay, native replay, finite-output and unchanged-source checks. The two24-example training subsets are explicitly disjoint for each template, and both are disjoint from the common96-example evaluation panel. Live gates and all seed/count outcomes are retained. Cross-sample assignments are evaluated on common native queries; causal comparisons use live intervention rollouts. Configs, fitted models, gate logs, partition comparisons and endpoint metrics saved. No reserved test or added natural documents; pool remains26,112. Broad goal active.
