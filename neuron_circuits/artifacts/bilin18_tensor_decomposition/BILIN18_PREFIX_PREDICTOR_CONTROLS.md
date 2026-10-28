## H606 — Within-format prediction works much better; earlier state does not broadly fix transfer

H605 held out task and wording simultaneously. This control separates them and tests predictor location. Capture 320 native prompt records covering all eight items and both wordings for the five tasks, with/without the prefix. Fit only item IDs 0–3 and evaluate only IDs 4–7 under new wording. For each target task, compare four training regimes: same task/new wording, same task/old wording, other tasks/new wording, other tasks/old wording. These labels refer to task-specific template versions; they do not impose one common sentence shell across different tasks.

For each regime fit a fixed mean and three otherwise identical ridge predictors from final-position MLP11 normalized input, MLP5 normalized input, or Attention5 output. The attention write is used as observed; ridge regularization scales with each input's Gram trace. The predictors never receive the evaluation edited-context difference. All use observed incoming e to execute the predicted bilinear response. Training uses eight contexts in same-task regimes and 32 in other-task regimes, so training-set size is not matched; the other-task regime has more data. No model or hyperparameter selection occurs on evaluation.

There are 240 identical evaluation signal cases across 18 modes, giving 4,320 interventions. This is an adaptive test on familiar evaluation prompts with newly captured training items/states, not independent confirmation. Item IDs are disjoint, but lexical overlap is possible. In same-task/new-wording capital recall, two evaluation endpoint prompt strings exactly duplicate training strings through neighboring country pairs. That condition is not a clean lexical holdout and must not support a generalization claim. No exact prompt overlap occurs in the other regime/family combinations.

**Prediction is much better when both task and wording are represented in training.** For competent copying, native-MLP11 prediction gives relative KL 0.073–0.094 on held-out items with matching wording, versus 0.762–0.921 from old wording. For competent agreement the corresponding ranges are 0.103–0.130 versus 0.343–0.658. The within-format constant mean already gives copying relative KL 0.083–0.103, so a large part of that predictable effect is shared within the format; ridge adds a modest improvement. Agreement benefits more from conditioning on native input.

| Task | Signal | Same task/new wording MLP11 KL | Same task/old wording MLP11 KL | Other tasks/new wording MLP11 KL | Other tasks/old wording MLP11 KL |
|---|---|---:|---:|---:|---:|
|agreement|number|0.1029|0.3426|0.7066|0.5610|
|agreement|task|0.1078|0.3433|0.7220|0.5915|
|agreement|random|0.1295|0.6576|0.8057|0.6908|
|capital_recall|number|0.0247|0.5345|0.8458|1.0174|
|capital_recall|task|0.0296|0.3667|0.7646|0.8822|
|capital_recall|random|0.0299|0.3693|0.8411|0.8060|
|entity_binding|number|0.4073|0.7980|1.1861|1.0817|
|entity_binding|task|0.1672|0.5309|0.9769|1.1178|
|entity_binding|random|0.3217|0.5884|1.7192|1.7400|
|sentiment|number|0.0354|0.1355|1.2608|1.1053|
|sentiment|task|0.0530|0.1603|2.1152|1.6236|
|sentiment|random|0.0424|0.1518|1.4122|1.3306|
|sequence_copy|number|0.0725|0.9208|1.6426|1.5379|
|sequence_copy|task|0.0938|0.9079|0.9900|1.2454|
|sequence_copy|random|0.0867|0.7616|3.2087|2.8642|

Earlier-state predictors do not consistently rescue the wording shift. In copying, same-task/old-wording MLP5 KL is 0.881–1.485 and Attention5 KL 0.838–1.357, versus MLP11 0.762–0.921. In agreement the earlier inputs likewise worsen transfer. Within matching wording, MLP5 is slightly better than MLP11 for agreement and similar for several other cases, so this is not a claim that later state is universally superior. It specifically weakens the hypothesis that simply replacing final-position MLP11 input with one of these earlier local vectors solves the transfer problem. Full-sequence state and richer predictors remain untested.

All numbers are fidelity errors relative to the full local context switch versus leaving its contribution unchanged, not task accuracy or shares of total computation. Internal response, answer-margin and intended-token CE errors are reported separately in the JSON. Good relative KL does not imply faithful reconstruction of all internal components or all answer statistics.

Competence remains as in H603: copying and agreement pass fully for recipients and prefixed donors; binding recipients pass but prefixed donors are weak; sentiment reaches the recipient threshold but prefixed donors are weak; new-wording recall recipients fail. The clearest task-level inference therefore comes from copying and agreement. All cases remain in the result.

Interpretation: the difficult part is transfer between context/format regimes, not basic inability to predict the edit within a regime. A nearly constant format-conditioned component plus item-dependent variation is a plausible account for copying. The current evidence does not establish whether format dependence is caused by sentence length, lexical construction, token positions, routing, or semantics. Task and surface form still covary across families. Nor does a family of format-specific predictors constitute a canonical form or identify unique units.

Next isolate those structural variables with matched edits that preserve task competence: hold the copied content fixed while changing sentence length, verb construction and repetition/position separately. Measure which changes alter prefix response geometry and which changes alter downstream use of otherwise similar responses. This can turn the current descriptive conditioning into a mechanistic account instead of fitting another more flexible predictor.

Audits: actual saved training/test metadata checked for every regime; zero item-ID overlap; exact prompt overlap reported above; frozen source/runner/manifest/fitted objects unchanged; 320 native capture records, 240 cases, 4,320 interventions; native replay max logit error 3.87e-5, matched prior native capture difference <=2.82e-5; ridge normal-equation residual <=1.91e-15; all metrics finite. Source response algebra was independently audited in H603. No new natural-text documents; FineWeb remains 26,112. Broad goal remains active and incomplete.
