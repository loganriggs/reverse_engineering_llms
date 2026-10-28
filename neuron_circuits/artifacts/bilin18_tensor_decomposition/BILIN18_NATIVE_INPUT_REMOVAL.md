## H675 — Finite removal confirms input support, without revealing a task solver

We removed half or all of the native layer14 input-token attention write from head4 alone or all nine heads, including both local and carried values, then ran full downstream computation. A zero-edit replay was included. Same128 previously inspected contexts,16 input identities, four domains, copy/task instructions and four forms. The intervention removes only the specified input-position contribution, not the entire head or its attention to other positions.

| Family | Context | Removal | Input logit change | Task-answer logit change | Correct-target CE change | Native KL | Native top1 | Edited top1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
|plural|copy|head4|-0.9589|-0.9244|0.8585|0.0414|0.9375|0.6875|
|plural|copy|all|-1.8445|-1.5284|1.7074|0.1284|0.9375|0.2500|
|plural|task|head4|-0.9128|-0.8586|0.7240|0.0494|0.0000|0.0000|
|plural|task|all|-1.7268|-1.4147|1.2193|0.1521|0.0000|0.0000|
|antonym|copy|head4|-0.4093|-0.0327|0.3606|0.0100|0.7500|0.6250|
|antonym|copy|all|-1.6541|-0.6214|1.5447|0.0856|0.7500|0.2500|
|antonym|task|head4|-0.4390|-0.0295|-0.0389|0.0147|0.0000|0.0000|
|antonym|task|all|-1.6995|-0.6462|0.4871|0.1207|0.0000|0.0000|
|capital|copy|head4|-0.2196|-0.1250|0.1866|0.0102|0.8750|0.7500|
|capital|copy|all|-0.7564|-0.4694|0.6225|0.0469|0.8750|0.5625|
|capital|task|head4|-0.1620|-0.0796|0.0624|0.0047|0.0000|0.0000|
|capital|task|all|-0.4887|-0.2646|0.1924|0.0223|0.0000|0.0000|
|successor|copy|head4|-0.2877|0.0621|0.2486|0.0074|1.0000|1.0000|
|successor|copy|all|-1.4500|-1.0067|1.2676|0.1129|1.0000|0.4375|
|successor|task|head4|0.0707|-0.0206|0.0353|0.0014|0.0000|0.0000|
|successor|task|all|-0.3968|-0.4619|0.2839|0.0219|0.0000|0.0000|

Positive CE change is worse prediction of the prompt's correct target; for copying that target is the input, whereas the task-answer column always refers to the transformed answer. KL is native-distribution-to-edited-distribution fidelity, not correctness. Top1 uses the full vocabulary, not a restricted candidate set. Table entries average16 contexts per cell; these are not16 independent identities.

Removing head4 lowers the input logit under both instructions for nouns, antonyms and capitals, and under copying for months. Month task prompts instead show a small input-logit increase, matching the direct-path sign reversal inferred in H674. Removing all input-position writes lowers input logits in all eight family/context means. Thus the principal direct-path sign pattern survives full downstream computation and finite removal.

The contributions support useful copying: all-head removal reduces copying top1 from93.75% to25% for nouns,75% to25% for adjectives,87.5% to56.25% for countries and100% to43.75% for months. Head4-only removal produces smaller losses, with month copying still100% despite worse CE. This illustrates why accuracy alone can hide weakened predictions.

Task behavior does not improve into successful transformation. All native and edited task top1 rates are0% on this held-out subset. Removing all input writes worsens correct-target CE in every task family; removing head4 modestly improves antonym CE (−0.0389), but worsens the other three. Suppressing the current input is therefore not sufficient for computing the requested answer; these writes can support related answers as well as copying. Native failure also limits how much these prompts can reveal about successful task algorithms.

This supports signed context-dependent control of input-associated content, while arguing against treating removal of copying as a general solution to transformation tasks. A next high-information step is to repeat the gate/value test on independently specified prompts where the native model actually performs the intended transformation, with matched copying controls and a predeclared competence criterion. Keep all attempted prompt families in the report, including failures, to avoid selecting a flattering mechanism after looking at interventions. The diverse-domain mechanism and task-solver claims must remain separate.

Audits:640 whole-model forwards (128×5 conditions), native replay maximum absolute logit error1.907e-5, all tracked hashes unchanged and outputs finite. Half-strength conditions retained in JSON/captures; the table reports full removal. This is retrospective finite activation removal, potentially off manifold, not a unique decomposition or a globally necessary circuit. FineWeb remains26,112 unique documents. Previous goal turn was progress; broad goal active and incomplete.
