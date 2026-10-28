## H915 — Query-dependent processing survives matched surfaces, but is not format-independent

The previous turn made progress: H914 established jointly competent zero-/two-demonstration matched-query conditions while retaining the one-demonstration failure. H915 fits context means on only the first four old discovery pairs, separately for language/country queries and N0/N1/N2. Each mean uses eight endpoints. Means are frozen before interventions on the other four old and six new country pairs (120 rows). L13/14 remain screening-derived sites.

Within each condition, task and copy prompts have identical prefixes, lengths and final tokens and differ by one query token. Both context maps, their average, and earlier unmatched-format task/copy maps are independently applied to every recipient. Choosing the demonstration condition's mean is an explicit diagnostic context label; no automatic selector or task classifier is learned. Exact full donor, identity, native-linear and quadratic-only controls are retained. N1 remains failed-gate data.

### New-country transfer in the competent conditions

Values below compare the midpoint approximation J(mean)d with the exact full single-MLP source-change effect. Margin error and KL are normalized by doing nothing; lower is better. Means are not fitted to any new-country outcomes.

|Layer|Condition|Recipient task|Language-map margin error|Country-map margin error|Language-map KL ratio|Country-map KL ratio|
|---|---|---|---:|---:|---:|---:|
|13|N0|task|0.321|0.499|0.228|0.738|
|13|N0|copy|0.582|0.327|0.900|0.298|
|13|N2|task|0.444|0.540|0.226|0.490|
|13|N2|copy|0.534|0.363|0.507|0.267|
|14|N0|task|0.774|0.805|0.319|0.778|
|14|N0|copy|0.778|0.509|1.052|0.343|
|14|N2|task|0.802|0.787|0.457|0.604|
|14|N2|copy|0.768|0.545|0.728|0.409|

At L13, matched language means reproduce language changes better than country means, and country means reproduce copying changes better than language means. The average map is intermediate: new-country language margin/KL ratios0.403/0.394 for N0 and0.482/0.316 for N2, compared with language-map0.321/0.228 and0.444/0.226. This context difference is therefore not explained solely by the previously unmatched final token or prompt length.

The full language intervention RMS effect is0.686 logits for N0 and0.559 for N2 at L13. New-country language baseline exact-answer accuracy is only50% in each condition, at the gate threshold, so this remains a limited-competence panel rather than uniformly reliable task execution. L14 language margin fidelity is weak (ratios~0.77–0.80), and the wrong/average map can be slightly better on that metric. Distributional fidelity still favors the language map over the copying map. Do not generalize the stronger L13 specificity to every metric and layer.

N1's language task fails baseline competence. Its L13 matched-language margin/KL ratios0.359/0.133 are retained as diagnostics, not a reason to override the gate. All old-transfer, new-transfer, failure and control results remain in the analysis.

### Operator contrast after surface matching

|Layer|Condition|Task/copy operator cosine|Contrast norm / task-map norm|Contrast norm / earlier unmatched contrast|Cosine with earlier unmatched contrast|
|---|---|---:|---:|---:|---:|
|13|N0|0.637|0.804|0.769|0.565|
|13|N1|0.778|0.640|0.541|0.543|
|13|N2|0.788|0.619|0.479|0.545|
|14|N0|0.584|0.851|0.783|0.582|
|14|N1|0.773|0.646|0.530|0.559|
|14|N2|0.786|0.620|0.468|0.556|

Matched contrasts remain substantial: their norm is62–85% of the language-map norm. But they are smaller than the earlier unmatched contrast and align with it only at cosine0.54–0.58. The earlier contrast was therefore not a demonstrated format-independent semantic operator. These comparisons change the broader prompt family and demonstrations, so they do not assign a precise fraction of the original contrast to any one surface feature.

Legacy language maps can preserve a target margin reasonably while giving worse distributional fidelity. For new-country L13 N0, the legacy margin ratio0.329 is close to matched0.321, but KL ratio0.581 is much worse than0.228. For N2, legacy KL ratio0.858 versus matched0.226. This again argues against defining faithfulness from target margins alone.

### Interpretation and next discriminator

There is controlled evidence for different reusable processing under two one-token queries with otherwise matched surfaces. However, “Language” and “Country” are still different lexical tokens. The experiment does not distinguish response to those token identities from an abstract relation request independent of its wording, nor does it identify a complete circuit or a compact manifold.

Next use a counterbalanced label-to-task assignment: the same query label should request language in one context and country copying in another, with shared demonstrations/instructions defining the assignment. Check competence and token matching before interpreting operators. Retain both mappings and all failures; do not tune the mapping from intervention outcomes. This would separate query-token identity from the computation specified by context. Binding pathways and broader shared-dictionary/featurizer validation remain open.

### Verification

2,400 full-model forwards. Identity patches, normalized activation replay and baseline-logit replay against H914 are exact. Source/model/earlier-map hashes unchanged; newly fitted means saved before evaluation and unchanged afterward. All KL finite. Discovery/transfer pairs remain disjoint, all three conditions retained, and no intervention-based exclusions. No natural or reserved-test evaluation; document pool remains26,112. Goal active.
