## H718 — Full early residual swaps also fail the task-transfer test

Previous goal turn classification: progress. H717 found no positive selective group candidate across five competence-checked task families. H718 tests whether the broader state at the same intervention site can provide a positive control.

For every eligible H717pair, retain the same task swaps in both directions and nuisance-to-base swap. At block1final-token output, intervene with (1)the full donor-minus-receiver residual, (2)its orthogonal projection into the256Dfeaturizer span, or(3)the orthogonal complement. Evaluate each at natural full strength and capped at5%receiver residual norm, without amplification. Preserve the persistent value bus, other positions and receiver input pathway. Include zero-edit replays. No group selection or new competence filtering.

The transfer split contains8pairs per family,16directed task swaps and8nuisance swaps. Positive task gain moves the predefined answer margin toward the donor's answer. Specificity subtracts mean absolute nuisance margin change, as inH717.

| Family | Full natural task gain | Full natural nuisance change | Full natural specificity | Mean full task-edit norm | Donor answer Top1after full swap |
|---|---:|---:|---:|---:|---:|
|agreement|-0.00427|0.50626|-0.51052|12.7%|0.0%|
|capital_recall|0.01796|0.04876|-0.03080|10.2%|0.0%|
|entity_binding|-0.00031|0.07333|-0.07363|7.1%|0.0%|
|sentiment|-0.01603|0.23891|-0.25494|9.0%|0.0%|
|sequence_copy|-0.05321|0.87846|-0.93167|19.8%|0.0%|

No full natural early-state swap produces a donor Top1answer on the80directed transfer examples. All five specificity means are negative, with pair-bootstrap95%intervals below zero. Capital recall has a small positive task margin shift, but nuisance sensitivity is larger. The other task gains are near zero or negative. Full natural task-edit norms average7.1–19.8%of receiver residual norm across families; failure therefore is not explained solely by the prior5%cap.

The retained span and complement also produce zero donor Top1answers on all transfer tasks at either dose. Their specificity means are negative in every family; several small-dose confidence intervals span zero. Neither increasing the intervention from a group to the entire retained span nor restoring omitted residual directions supplies the missing positive control. This does not show that those directions carry no task information: sufficiency of this patch interface and presence of information are different questions.

The task scores themselves support the conclusion independently of the nuisance penalty: natural full-swap donor-directed gains are-0.0043agreement,+0.0180capital recall,-0.0003entity binding,-0.0160sentiment,-0.0532copy. The nuisance criterion remains conservative and natural nuisance differences can be larger/different than task differences. Do not use the negative score to claim semantic irrelevance of every component.

Full residual replacement here is only at the final position afterblock1. It is not donor-context replacement. Later layers can still read receiver states at other positions, its persistent value bus and its input pathway. Task-relevant information may not yet be integrated at this site, or the receiver's remaining pathways may overwrite the transplanted signal. These are more immediate explanations to test than declaring the featurizer invalid.

Next map full last-position residual swaps across model depth on the same fixed competence-checked suite. Include nuisance donors and retain discovery/transfer separation. A swap after the final block must reproduce the donor's logits, providing an exact architectural positive control; audit this before interpreting intermediate-layer effects. Determine where answer transfer emerges, then choose representation/manifold tests at sites with demonstrated causal relevance. If early multi-position or value-bus interventions are necessary, test them explicitly rather than silently broadening what this negative final-position result means.

Audits:1680whole-model prefix evaluations (1440nonzero,240zero). Zero-edit maxKL4.63e-12 and maxanswer-margin discrepancy8.59e-6. Full natural swap produces the saved donor block1state with maxrelative error8.36e-7. Retained/complement sum error7.79e-14in norm and orthogonality cosine7.05e-16. All source/model/data hashes unchanged, finite metrics. Original corpus reserved test untouched, no new unique research documents, pool26112. Broad goal active.
