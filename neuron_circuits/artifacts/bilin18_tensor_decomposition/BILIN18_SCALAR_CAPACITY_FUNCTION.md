## H736 — Larger active budgets restore much of the missing cross-task computation

Every H735 scalar Top-K model was frozen and tested on the complete H732 panel, retaining all three seeds and all competence-failure labels. Dictionary width and parameter count are fixed; K=8,16,32,64 models were trained separately. The task intervention adds F(donor)-F(receiver) to the native receiver head. A separate native-head replacement measures output KL and intended-answer-set NLL change. No model or checkpoint was chosen from these functional results.

Mean retained exact-head task effect in competent, non-negligible-effect strata:

| Head and task / format | K=8 | K=16 | K=32 | K=64 |
|---|---:|---:|---:|---:|
| Agreement head, agreement A | 88.5% | 95.6% | 97.0% | 98.2% |
| Agreement head, agreement B | 94.3% | 96.0% | 98.5% | 99.5% |
| Recall head, capital recall B | 65.1% | 74.8% | 84.3% | 96.5% |
| Recall head, entity binding A | 37.5% | 52.6% | 73.1% | 90.4% |
| Recall head, entity binding B | 35.1% | 51.5% | 69.5% | 92.2% |
| Recall head, sequence copying A | 41.0% | 58.8% | 78.0% | 92.4% |
| Recall head, sequence copying B | 43.6% | 59.1% | 77.0% | 93.4% |

These ratios summarize margin effects, not accuracy. At K=64, per-example relative RMS effect error is 3.3%/1.9% for agreement, 4.0% for capital recall, 12.5%/12.4% for binding, and 8.9%/8.1% for copying. Thus the larger budget is substantially more faithful but does not fully explain every effect; binding retains meaningful residual error even where average retention exceeds 90%.

Native output KL also decreases. For the recall head, K=8 to K=64 lowers mean KL from 0.0357 to 0.0026 on competent capital recall, from 0.0215/0.0158 to approximately 0.0007/0.0008 on binding, and from approximately 0.0040 to 0.0004 on copying. Agreement-head KL falls from 0.0074/0.0060 to approximately 0.0007/0.0005 on agreement. These task-context results remain distinct from natural-corpus fidelity and intended-answer-set NLL, which is archived separately.

All failed-competence strata and low-effect cross-task controls are retained in the summary JSON. Retention ratios remain suppressed where the absolute mean exact-head effect is below 0.25; absolute effect errors are used there. No success claim is based on sentiment or capital Format A, which failed the native-competence gate.

Interpretation: the very sparse active budget was a substantial source of missing computation. A higher-capacity adaptive representation now provides a stronger baseline for interpreting the recall head. However, H735 showed that individual atoms become less reproducible across seeds as K rises. Faithfulness and identifiable computational units remain separate achievements. Higher-K success alone may partly reflect generic high-dimensional approximation rather than discovered model structure.

Audits: 12,012 full-model example forwards completed. All input/source hashes remained unchanged and task/KL values were finite. K=8 task results exactly reproduced H733; identity task replay maximum error was 6.68e-6 and identity native reconstruction KL was zero. H735's implementation and budget tests passed. No new natural documents or original reserved-test evaluations were used.

Next: evaluate a matched-active-budget random orthogonal dictionary in the same write coordinates, using signed adaptive Top-K and the same K values/seeds, alongside adaptive PCA coordinates. This separates improvement from simply retaining more coordinates from improvement attributable to learned directions. The null requires no semantic labels or training; it is an approximation control, not a parameter-matched learned model. If it is comparably faithful at K=64, do not interpret the learned atoms as meaningful units merely because they reconstruct well. Then examine stable subspaces, causal-effect patterns and selective shared-component interventions, with the random-coordinate controls carried forward. Sentiment's format/content diagnostic remains outstanding.
