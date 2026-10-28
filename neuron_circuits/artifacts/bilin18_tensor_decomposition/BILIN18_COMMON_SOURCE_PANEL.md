## H868 — Exact common-source query panel fails native competence

H867 completed all-reader bus erasure localization. The next proposed control held lexical sources and source positions fixed while varying task queries. H868 implements that control, but the model is not reliably competent on it. Only1/24 cells passes the predefined gate, and no task/format/query passes both discovery and transfer. Therefore this panel cannot support a competent cross-task reader-reuse claim.

Sixteen country/capital pairs were declared before outputs, first8 discovery and last8 transfer. Common source prefixes use A/B labels with colon or equals formatting. Queries ask to repeat the first/second country, retrieve the country labeled A/B, or retrieve its capital. Each pair has both country orders.12 lexical pairs survive common single-token source/answer eligibility;4 transfer pairs are excluded for multi-token capital answers.288 contexts remain.

Gate per task/format/query/split: at least4 eligible pairs, >=75% of pairs correct at both endpoints under binary answer comparison, and >=50% full-vocabulary intended-token top1. Individual outcomes were not filtered. Only capital/equals/queryA/transfer passes; the corresponding discovery cell fails. The original first/second-country copy task is an ordinal lookup, not the previously competent repetition task, and A/B instructions are a new demand. This failure cannot be assigned to absence of a representation or inability of the frozen readers to operate on competent tasks.

|Task|Format|Query index|Split|Pairs|Both correct|Full-vocab correct|Pass|
|---|---|---:|---|---:|---:|---:|---|
|copy|colon|0|discovery|8|0.0|0.375|False|
|copy|colon|0|transfer|4|0.25|0.25|False|
|copy|colon|1|discovery|8|0.125|0.125|False|
|copy|colon|1|transfer|4|0.25|0.25|False|
|copy|equals|0|discovery|8|0.75|0.25|False|
|copy|equals|0|transfer|4|1.0|0.25|False|
|copy|equals|1|discovery|8|0.0|0.0|False|
|copy|equals|1|transfer|4|0.0|0.0|False|
|binding|colon|0|discovery|8|0.625|0.0|False|
|binding|colon|0|transfer|4|0.5|0.0|False|
|binding|colon|1|discovery|8|0.0|0.0|False|
|binding|colon|1|transfer|4|0.0|0.0|False|
|binding|equals|0|discovery|8|0.5|0.0|False|
|binding|equals|0|transfer|4|0.5|0.0|False|
|binding|equals|1|discovery|8|0.0|0.0|False|
|binding|equals|1|transfer|4|0.0|0.0|False|
|capital|colon|0|discovery|8|1.0|0.25|False|
|capital|colon|0|transfer|4|0.75|0.0|False|
|capital|colon|1|discovery|8|0.0|0.0|False|
|capital|colon|1|transfer|4|0.0|0.0|False|
|capital|equals|0|discovery|8|1.0|0.4375|False|
|capital|equals|0|transfer|4|0.75|0.5|True|
|capital|equals|1|discovery|8|0.0|0.0|False|
|capital|equals|1|transfer|4|0.0|0.0|False|

Native routing and mixed values were captured at frozenL13H0,L14H4,L8H3 without interventions. The signed attention readout AV reconstructs the captured head output with maximum relative error6.25e-7. Same-token source bus vectors agree across task contexts within1.53e-5 raw units, verifying the content-control implementation. Native forward replay is exact, source positions are identical across query tasks, and swapping source order changes only the two declared source tokens. These are implementation checks, not evidence of task success.

Next use familiar continuation-style tasks with the same country contrasts: repeated token sequences, diplomatic entity binding, and direct capital queries. Relax identical prefix/position rather than claim an instruction-following ability the model does not exhibit. Record all changed/source positions, retain the same eligibility and competence gates, and explicitly account for routing differences in later interventions. This is an adaptive redesign informed by the failed panel, so a subsequent screen is exploratory rather than an independent preregistered confirmation. Preserve H868 as a negative task-competence result; do not tune prompts until they pass and then present only successes.

300 whole-model forwards including replay, all inputs unchanged and outputs finite; no reserved-test use and natural-document pool26,112 unchanged. No causal reader-reuse conclusion drawn. Broad goal remains active.
