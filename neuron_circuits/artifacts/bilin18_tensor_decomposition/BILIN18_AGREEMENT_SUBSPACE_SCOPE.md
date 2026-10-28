## H585 — The number-associated direction transfers partially across wording and is selective among tested contrasts

Freeze H584's basis without refitting. Test all four H582 agreement wording cells, both lexical splits, plus the H583 selected formats for recall, copying, binding and sentiment. Use opposite-answer donor differences at Attention5's final-position write. Compare full swap, zero, ranks 1/4/8, and nuisance plus five random rank-one projections norm-matched to PCA rank one. This is 2,772 interventions on already evaluated prompts, not a new corpus or untouched confirmatory panel. The basis was fixed before these projection interventions.

Agreement conditions: 00 beside/window; 01 beside/window/usually (basis-training wording); 10 near/window; 11 near/window/usually. For other tasks, inherit H582/H583 selections and competence limits. Projection coefficients still depend on the actual donor write difference.

| Lexical-transfer cell | Competence gate | Full donor movement | Rank1 donor movement | Rank1 relative margin error | Rank1 relative reference KL | Rank8 relative margin error | Rank8 relative reference KL |
|---|---|---:|---:|---:|---:|---:|---:|
|agreement_00_transfer|pass|0.6339|0.5900|0.1457|0.3435|0.1223|0.2397|
|agreement_01_transfer|pass|2.1581|2.0870|0.0821|0.0796|0.0568|0.0436|
|agreement_10_transfer|pass|0.6661|0.6078|0.1317|0.4664|0.1225|0.3112|
|agreement_11_transfer|pass|2.5753|2.4214|0.1457|0.0989|0.1151|0.0551|
|capital_recall_10_transfer|pass|0.0254|-0.0004|1.1065|1.1525|1.2486|1.4885|
|entity_binding_00_transfer|fail|0.3372|0.0041|0.9878|0.9763|0.8931|0.7769|
|sentiment_01_transfer|fail|0.6538|-0.0010|1.0015|1.0031|1.0228|0.9829|
|sequence_copy_00_transfer|pass|0.0395|-0.0005|0.9993|1.0068|0.9787|1.0146|

The direction carries a substantial agreement effect beyond its fitting wording. For near/window without “usually”, rank one gives donor movement 0.608 versus full 0.666. But relative margin error is 0.132 and relative reference KL 0.466, failing the H584 <=10% descriptive criterion. With near/window/usually, rank one gives 2.421 versus 2.575, with 0.146 margin error and 0.099 relative KL; this also fails the joint criterion. Rank eight improves some metrics but does not meet the margin threshold in these transfer cells. Mean effects can appear close while individual errors remain appreciable.

Thus the direction's behavioral usefulness transfers better than strict full-swap equivalence. The learned one-dimensional approximation is context scoped. Wording changes jointly alter context, position and sometimes final token; this experiment does not isolate which factor causes the fidelity loss. Internal error also grows, but that alone cannot predict which omitted directions matter behaviorally.

On recall and copying, the same basis captures almost none of the relevant donor contrast, despite those tasks passing competence gates. Binding and sentiment show the same descriptive pattern, but their transfer competence fails. This is consistent with selectivity for number-related contrasts, not a universal answer-change direction. It does NOT show that a number direction cannot be reused in other tasks where number matters: these control contrasts change names, countries or sentiment, not number. Small full effects in recall/copying make relative metrics especially sensitive; absolute donor effects and KL are retained.

Neither off-target task failure nor rank-one compression establishes abstract grammatical number. Every fitted subject pair was a regular singular/plural noun pair. The direction could encode suffix morphology, lexical plurality, subject number, or a correlated agreement preparation signal. Its downstream construction and reader remain unidentified.

Next discriminating test: freeze the direction and cross subject number with an intervening noun's number, using both regular and irregular subject plurals. Include at least two grammatical frames and retain every competence failure. Measure whether the coefficient and intervention follow the grammatical subject, the nearest noun, or surface plural morphology. Contrast subject-changing and attractor-only swaps at an identical final token. This directly separates a reusable subject-number representation from local plurality and lexical shortcuts. Keep full-reference KL, margin and internal fidelity separate; use new lexical/frame combinations before claiming a general unit.

Audits: 352 shared interventions replay H584 within 1.63e-5 margin; native logits replay prior panels within 3.87e-5; zero margin <=1.01e-5; all finite; source and frozen basis unchanged. No new prompts or natural-text documents; FineWeb total remains 26,112. The full feature/computation objective remains active and incomplete.
