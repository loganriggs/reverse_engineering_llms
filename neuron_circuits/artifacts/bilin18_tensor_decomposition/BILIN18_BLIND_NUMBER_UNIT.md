## H893 — One fixed unit causes little mixed-context disruption, but natural-text fidelity gains are limited and heterogeneous

Applied the same agreement-trained axis-A component everywhere, without task/family/template-based fit selection. Rank1 replacement changes no top token across192 contexts. It preserves substantially more of the component's behavior than removal in grammar contexts, but the advantage is modest in copying and natural text. On56.25% of natural documents, rank1 is actually less faithful than removing the component. Low absolute KL therefore does not establish broad recovery: this component is usually weak at the sampled natural positions.

### Frozen mixed-context test

The fixed unit is H891 axisA, task-family fit, always active at the final-query L12 site. No label-based gate or family switching is used. Compare exact, mean, rank1, rank7, random7 and component removal, all with previously fitted coefficients and no refit.

Eight new declared noun pairs: pilot(s), nurse(s), singer(s), soldier(s), lawyer(s), chef(s), scientist(s), engineer(s). Use new grammar templates “behind the fence” (G) and “beside the window” (H), plus matched word/item repetition. All pairs pass answer tokenization eligibility; none are filtered by correctness. These give32 grammar and32 copying contexts.

Add the first128 documents from the existing frozen natural confirmation panel, each truncated at its previously locked token position. Document windows are verified against the original manifest. These documents were used in earlier broader research, but not to fit or evaluate this number unit before this experiment; they are not globally untouched or pretraining-disjoint data. The original reserved test is not evaluated. Each document contributes one local replacement, not simultaneous edits throughout all sequence positions.

AxisA was frozen for this mixed test after earlier comparisons of the axes, so this is not an unbiased initial axis-selection experiment. Cohort labels are used only for reporting, not runtime selection.

### Native output fidelity and component influence

|Context group|Rank1 mean KL|Removal mean KL|Rank1/removal KL ratio|Rank7/removal ratio|Rank1 worse than removal|Rank1 top-token changes|
|---|---:|---:|---:|---:|---:|---:|
|grammar|0.000346|0.001391|0.249|0.224|0.031|0.000|
|copy|0.000356|0.000431|0.826|0.821|0.281|0.000|
|natural|0.000246|0.000344|0.716|0.675|0.562|0.000|

Grammar rank1 mean KL is0.000346 versus mean-vector0.000799 and random7 0.000773. The learned contextual direction helps beyond a mean or arbitrary dimensions. Copying's corresponding improvement is small: rank1 0.000356 versus mean0.000399. Natural text gives rank1 0.000246 versus mean0.000291 and random7 0.000291. The low removal KL of0.000344 on natural text shows that low approximation KL partly reflects weak component influence.

Natural rank1 median KL is0.000131,95th percentile0.000873 and maximum0.002786; no sampled top token changes. Its mean document-bootstrap95% interval is approximately0.000189–0.000314. The ratio of summed errors favors rank1 over removal, but72/128 individual documents favor removal. This heterogeneity rules out a uniformly beneficial native-fidelity claim. No deployment gate is fitted from these outcomes.

For124 documents with an available next token, mean CE change is−0.00097 nats under rank1 and−0.00084 under component removal. These tiny changes are descriptive and do not establish a language-model improvement; CE and fidelity are distinct criteria, and changing toward the realized token can reduce CE while departing from the native model.

### Behavioral competence remains separate

Both new grammar templates have100% binary is/are preference, but exact full-vocabulary top-1 is56.25% in G and0% in H. Thus only G passes the prior task gate. Copying passes in both formats. All contexts were still evaluated and retained, so the pooled grammar result includes a failed exact-answer template and is not wholly competent-task transfer evidence.

Separately, competent G has rank1 KL0.000313 versus removal0.000877; failed-gate H has0.000378 versus0.001904. Neither improves nor loses binary target correctness after replacement, and no full-vocabulary top token changes. A small perturbation in a failed-gate context must not be promoted to a validated semantic transfer result.

### Interpretation and next stability/generalization test

The fixed unit can be used without runtime task labels with limited disruption in this panel, but its relative computational fidelity is strongest where it was semantically motivated. Its natural-text advantage is limited and nonuniform. This is compatible with a context-specific computation and does not establish a universal feature or task-blind featurizer.

Before fitting a gate to hide weak domains, test recovery stability: derive analogous axes and compact interactions from disjoint noun discovery sets, then compare recovered geometry, compiled operators and causal effects on fixed evaluation contexts. Use the remaining locked natural documents as a prospective unit-evaluation tranche, keeping them out of fitting and selection. This would distinguish a reproducible conditional operation from an accidental solution to a small supervised task panel. The broader copying lexical-coverage learning curve and alternatives-to-BSF ontology remain unresolved.

### Verification

1,344 full-model forwards across 192 contexts. Exact component replacement is an identity to full logit precision. Compiled encoders agree with explicit projections within relative 1.74e-15. All128 natural document-window hashes verified against the locked manifest; source/model/fit hashes unchanged. No output-dependent exclusions, all KL values finite. Synthetic confidence intervals in the analysis are endpoint resamples and are not treated as independent-pair confirmation; the natural interval resamples documents. No reserved test evaluated, natural-document pool unchanged at26,112. Broad research goal active.
