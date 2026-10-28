## H1027 — Restoring the cached target edge improves answers, but only partly explains distraction

Previous turn was progress: H1026 completed the wording/lexical controls and exposed both routing effects and full-vocabulary task failures. This turn moves from score correlation to a finite behavioral intervention.

Assay registered before evaluation: all128 neutral-mention examples from H1026, four original discovery country pairs, two endpoints, sixteen distractors. No example selection. Restore only the final-token contribution of the target source's cached head0 value at attention13:

Δwrite = (P_baseline − P_context) λ13 W_O13,0 v1,0(source).

The baseline coefficient comes from the matching no-prefix country prompt. It is a counterfactual oracle, not a trained routing predictor. This changes only the cached portion of one edge; all other attention edges, current-layer values and downstream computation remain live. It is not a whole-query replacement. Correct-answer identities are used only for evaluation; wrong-country control intentionally uses the opposite pair endpoint.

Six arms: native; restoration; sign-reversed restoration; two fixed seeded Gaussian head-space directions projected through the same output map and scaled to the restoration's output norm; opposite-country cached payload, likewise norm matched. No fit or tuning.

| Distractor class | Native top1 | Restored top1 | Reversed top1 | Restored paired-margin gain | Restored correct-answer log-probability gain |
|---|---:|---:|---:|---:|---:|
| Country |59.4%|75.0%|53.1%|+0.346|+0.100 nats|
| City |84.4%|87.5%|81.3%|+0.234|+0.017 nats|
| Person |93.8%|93.8%|93.8%|+0.101|+0.004 nats|
| Abstract noun |93.8%|93.8%|93.8%|+0.093|+0.001 nats|

Restoration improves the paired correct-minus-opposite-language margin on all128 rows; reversed and wrong-country writes worsen that margin on every row. Both random controls leave top1 unchanged in every class and produce small mean paired-margin changes (+0.001 to+0.021). This establishes direction-sensitive behavioral relevance of the cached target edge on this panel. It is not merely an effect of adding a vector of that size. However, small log-probability improvements for person/abstract conditions do not consistently exceed both random controls, and the wrong-country control slightly increases country-class top1 to62.5% while worsening every paired margin. These mismatches show why one must report full-vocabulary metrics alongside the paired contrast.

Scope of recovery: mean no-prefix-minus-context paired-margin gaps are1.124(country),0.948(city),1.075(person),0.676(abstract). Restoration's aligned gap recovery, sum(gain*gap)/sum(gap²), is16.6%,8.8%,7.8%,11.6%, respectively. This is not the ratio of mean gains to mean gaps (for country that different statistic is30.8%), nor a percentage of all model computation explained. Most prefix-associated changes remain outside the restored edge, and template removal changes more than the isolated coefficient.

Interpretation: the source-dependent payload and context-dependent edge strength have separable, causally relevant roles. The existing lexical operator is therefore a useful candidate computational unit, with routing treated as another component. This does not yet prove that the same unit is reused across tasks or identify a feature manifold. The intervention depends on native prefix values and a target-specific baseline coefficient; it is not an autonomous replacement program.

Limits:128 correlated rows represent only four target pairs and sixteen distractor words. This is an intervention on previously opened discovery contexts, not held-out generalization. Two fixed random directions are limited controls, not a distributional null estimate. Strengthening an answer-supporting edge could improve answers without explaining why its native coefficient changed; no optimality or model-error-repair claim follows.

Audits: native margins reproduce H1026 exactly. Independent installations before versus after the output projection agree in margin/log-probability within1.91e-6 on eight registered rows and preserve top1. Control output norms match within1.53e-7relative. All six arms present for every row; saved individual outputs and independent rescue/break counts.128prefix and776suffix sequences, zero fits, zero reservedBSFtests; natural-document pool26112unchanged. No active process. Broad goal remains open.

Next action: move to distinct-task reuse rather than extending this language-only gate. Predeclare country→capital prompts using the same source countries, establish native competence without outcome filtering, and test the frozen lexical value operator with task-specific native routing as an explicitly charged diagnostic. Compare its isolated cached13.0 path coverage with language and with another head control. This asks whether the operator transports reusable country content to distinct readers or whether its apparent meaning is language-specific; native routing alone must not count as a learned generalizing gate. After this bounded task comparison, feed the evidence into the shared-dictionary versus gated-operation featurizer comparison.
