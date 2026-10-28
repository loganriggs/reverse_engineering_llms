## H601 — Context coverage dominates signal-metric choice at fixed rank

H600's cross-task failure left two explanations: the context span was too narrow, or its metric was adapted to number signals. This factorial separates them. Keep rank at 64. Compare the old agreement-trained basis with a broad basis built from 80 same-answer nuisance context differences across five tasks and two templates. Fit only item IDs 0–3; evaluate only IDs 4–7. Within each basis compare Euclidean coefficients, a response metric trained on number signals, and a response metric trained equally by case on number/task/random signals. Include each basis's target-informed oracle and the historical H598 map as diagnostics.

Number metrics use 160 signed discovery responses; multi-signal metrics use 480. No evaluation-item vectors enter basis or metric fitting. Freeze the fitted maps before 5,280 live downstream interventions on 480 held-out signed responses. Templates and some vocabulary are shared across the item split. This is an adaptive experiment on previously observed H600 data, not independent confirmation or complete lexical isolation. The broad basis retains 99.957% of discovery context energy; that training statistic alone does not establish transfer.

Recompute competence on the four held-out pairs in each task/template. Seven cells pass; agreement1, binding1 and sentiment1 fail and are retained. Capital recall1 now passes on this subset, although it failed on the full H600 cell. This subset distinction is reported rather than used to select cases.

**Broadening context coverage causes the large improvement.** With the number-response metric held fixed in kind, internal relative errors fall from 0.858–0.979 in the old basis to 0.151–0.295 in the broad basis. Relative output KL falls from 0.637–1.102 to 0.018–0.076. Broadening the signal mixture while keeping a basis fixed changes these results much less. Even ordinary Euclidean projection in the broad basis gives relative KL 0.020–0.100 across all 30 task/template/signal cells.

All errors compare an approximation with the exact local context switch, normalized by the discrepancy from leaving context unchanged. They are not fractions of total task error or total model computation. The local context-switch operation, actual paired context difference and incoming perturbation remain supplied. No full-model compression or context prediction is demonstrated.

| Held-out cell | Signal | Gate | Old number-metric KL | Broad number-metric KL | Broad multi-signal KL | Broad number-metric margin error |
|---|---|---|---:|---:|---:|---:|
|agreement_0|number|pass|0.6909|0.0677|0.0699|0.1647|
|agreement_0|task|pass|0.7852|0.0727|0.0728|0.1622|
|agreement_0|random|pass|0.7107|0.0681|0.0671|0.1787|
|agreement_1|number|fail|0.7316|0.0500|0.0573|0.2116|
|agreement_1|task|fail|0.7624|0.0436|0.0487|0.1641|
|agreement_1|random|fail|0.8859|0.0504|0.0495|0.2302|
|capital_recall_0|number|pass|0.7839|0.0473|0.0486|0.2774|
|capital_recall_0|task|pass|0.6749|0.0389|0.0391|0.1856|
|capital_recall_0|random|pass|0.8727|0.0513|0.0492|0.1585|
|capital_recall_1|number|pass|0.6371|0.0698|0.0712|0.1534|
|capital_recall_1|task|pass|0.6606|0.0447|0.0444|0.1988|
|capital_recall_1|random|pass|0.8137|0.0451|0.0446|0.1246|
|entity_binding_0|number|pass|0.8647|0.0338|0.0341|0.2248|
|entity_binding_0|task|pass|0.9636|0.0317|0.0312|0.2762|
|entity_binding_0|random|pass|0.8048|0.0378|0.0366|0.1499|
|entity_binding_1|number|fail|0.6407|0.0231|0.0232|0.2794|
|entity_binding_1|task|fail|0.9621|0.0259|0.0264|0.1066|
|entity_binding_1|random|fail|0.9920|0.0476|0.0459|0.3485|
|sentiment_0|number|pass|0.7631|0.0321|0.0325|0.1054|
|sentiment_0|task|pass|0.8870|0.0185|0.0176|0.2883|
|sentiment_0|random|pass|0.8196|0.0277|0.0265|0.3722|
|sentiment_1|number|fail|0.9219|0.0360|0.0352|0.1133|
|sentiment_1|task|fail|0.7963|0.0531|0.0520|0.3064|
|sentiment_1|random|fail|0.8471|0.0269|0.0263|0.1607|
|sequence_copy_0|number|pass|0.8639|0.0608|0.0613|0.1610|
|sequence_copy_0|task|pass|0.9120|0.0758|0.0765|0.2686|
|sequence_copy_0|random|pass|1.1023|0.0741|0.0718|0.2766|
|sequence_copy_1|number|pass|0.7758|0.0221|0.0207|0.0560|
|sequence_copy_1|task|pass|0.6567|0.0265|0.0271|0.1059|
|sequence_copy_1|random|pass|0.8951|0.0342|0.0327|0.1142|

Better KL does not establish fidelity for every answer statistic. Broad number-metric margin error ranges 0.056–0.372, and relative target-token CE error ranges 0.065–0.588. Broad multi-signal CE error reaches 0.653. The distribution can be close while a selected token's loss or a small contrast is less faithfully preserved. CE here is fidelity to the full intervention's intended-token loss, not a demonstrated improvement in task accuracy.

The new basis's response oracle errors remain 0.144–0.281 internally, much smaller than the old oracle's errors but nonzero. The new basis therefore captures genuinely more of the tested response map, rather than merely choosing better coefficients in the old span. Internal oracle residuals still do not give a lower bound on output KL or CE.

Updated interpretation: H600's weak transfer was chiefly inadequate context coverage on this panel, not evidence that each task requires unrelated computational units. A shared fixed-size span can support these five families' local context-switch responses. Metric choice remains relevant to particular output contrasts, but adding signal diversity has much smaller effects than replacing the narrow span. Because the metric uses natural e magnitudes, equal training case counts are not equal response-energy weighting; this study does not exhaust all possible metric designs.

Remaining explanations include (a) reusable local computational coordinates, (b) a shared subspace for these few nuisance edits and prompt templates, and (c) wider rank requirements once contexts vary more. The current result cannot distinguish them. Next freeze the broad basis and both maps, then test unseen wording and different same-answer context edits across all five tasks. Use matched task-changing context differences as an additional scope challenge, clearly separated from nuisance changes. Keep competence failures, new-case target-informed oracles and separate KL/margin/CE reporting. Fresh wording/context transfer has higher information value than refitting this panel again.

Audits: 80 discovery contexts, 160 number/480 multi discovery responses, zero item-ID overlap with 480 evaluation responses; all fitted objects and source files unchanged during evaluation; exact native logit replay; independent forward/adjoint checks for all four fitted maps <=8.32e-8; normal-system residual <=2.50e-8; coefficient projector identity <=1.79e-7; system conditions 2.46–6.16; oracle normal-equation error <=3.53e-15; all metrics finite. No new natural-text documents; FineWeb remains 26,112. Broad goal remains active and incomplete.
