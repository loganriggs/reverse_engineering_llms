## H604 — Prefix responses transfer better within task than across held-out tasks

Can a shared prefix component predict the previously observed failures? For each of five tasks, fit only old-wording prefix context differences from the other four tasks: 32 endpoint records. Evaluate the held-out task under new wording: eight endpoints and six signed signals, 48 cases per fold. Use the H603 data and no new task examples. This is adaptive cross-validation, not independent confirmation. Task and wording are excluded from the main fits; lexical items and other surface properties are not guaranteed disjoint.

Compare a fixed mean context difference, projection onto its direction, uncentered shared rank8 and full-training-span projections, matched random spans, and per-case response oracles. Also fit a small span from the same task's old wording as a separately labeled diagnostic. That last condition is not a task holdout; it tests wording transfer with familiar task/items. It uses up to eight directions and eight training contexts, whereas the pooled other-task fit uses 32 contexts. Per-case active rank8 can be compared, but maintaining separate task dictionaries is not equal total storage to one shared dictionary.

The fixed mean is the only mode that predicts the context difference without observing each evaluation dc. All projection modes still observe actual dc; all modes use actual incoming perturbation e to evaluate T(dc,e). No response metric is fitted here: this isolates the availability of shared context components. Per-case oracles see the full target response and diagnose the available span.

An initial rank assertion caught exact dependencies from repeated capital prompts before intervention results existed. We excluded singular values <=1e-6 times the largest before continuing. Full shared spans have rank29 in four folds and rank32 in the capital-recall fold; same-task spans have rank8 except recall rank5. Random controls match each effective shared rank. No arbitrary null directions are used. All fitted folds were saved and hashed before 2,640 interventions.

**Neither a universal mean nor the other-task spans predicts this prefix well.** The fixed mean is frequently worse than leaving the context contribution unchanged. The full shared span's internal errors remain about 0.845–0.985; its relative KL ranges 0.663–1.134. The same-task old-wording span has much smaller internal errors, about 0.347–0.775, and relative KL 0.107–0.657, despite fewer active directions. It improves both internal error and relative KL over the full other-task span in all 15 task/signal cells.

| Held-out task | Signal | Fixed mean KL | Other-task rank8 KL | Other-task full-span KL | Matched random KL | Same-task old-wording KL | Other-task oracle internal error |
|---|---|---:|---:|---:|---:|---:|---:|
|agreement|number|0.5847|0.7192|0.7021|0.8753|0.2898|0.7934|
|agreement|task|0.6139|0.7344|0.7180|0.8747|0.3160|0.7958|
|agreement|random|0.7106|0.6901|0.6632|0.9257|0.5055|0.8405|
|capital_recall|number|1.0116|0.8733|0.8902|1.0120|0.4553|0.8671|
|capital_recall|task|0.9060|0.8385|0.8122|0.9470|0.3535|0.8480|
|capital_recall|random|0.7914|0.7464|0.7547|0.8733|0.3322|0.8624|
|entity_binding|number|1.1387|0.9473|0.8756|1.0569|0.3346|0.9329|
|entity_binding|task|1.1352|0.9407|0.8420|0.9689|0.2702|0.9185|
|entity_binding|random|1.8933|1.1966|1.1341|0.9458|0.6565|0.9285|
|sentiment|number|1.0061|0.9017|0.8950|0.9579|0.1373|0.9340|
|sentiment|task|1.6098|1.0397|0.9962|1.0063|0.1066|0.9539|
|sentiment|random|1.4562|0.9814|0.9593|0.9483|0.1360|0.9487|
|sequence_copy|number|1.7070|0.7505|0.7507|1.0121|0.4871|0.8603|
|sequence_copy|task|1.1447|0.7708|0.6810|0.9532|0.3222|0.8655|
|sequence_copy|random|2.8737|0.9771|0.8711|0.9417|0.5562|0.8588|

Relative KL is fidelity to the full local context switch divided by the full-versus-unchanged discrepancy. Values above one are worse than omitting that context-switch contribution. They are not task error rates. Margin and intended-token CE remain separately available in the JSON and can behave differently from KL. No overall accuracy improvement is claimed.

Copying and agreement have fully competent recipients and prefixed donors. Binding's recipients pass, but prefixed donors are weak; sentiment's recipients only reach the threshold and prefixed donors are weak. Recall's new-wording recipients fail full-vocabulary competence. The cleanest functional claims therefore come from copying and agreement; other cells characterize local response geometry and remain included rather than filtered away.

For competent copying, same-task old-wording KL is 0.322–0.556, versus 0.681–0.871 from other tasks. For agreement it is 0.290–0.506, versus 0.663–0.718. Thus the benefit does not depend solely on the weak or failed cells. Large errors remain even in the better condition; this is partial transfer, not faithful reconstruction.

The other-task response oracle still leaves internal errors about 0.793–0.954. A better coefficient rule within those spans cannot erase that missing local component. That statement does not lower-bound output KL or CE. Shared components do exist—many learned projections beat random controls—but they are insufficient for the full prefix response.

Interpretation: "the same textual edit" is not a universal additive feature at this MLP input. Its effect depends strongly on the recipient context. A context-conditioned family of components is currently more plausible than one reusable prefix direction. This does not establish task-specific semantic units: task identity is entangled with sentence structure, length, final-token patterns and other surface factors. Same-task transfer across wording might exploit these shared properties. Nor does this exclude a compact nonlinear or bilinear parameterization of the dependence.

Next discriminate task/semantic dependence from structural dependence. Cross a common terminal format and matched prefix placement with varied task content, or predict prefix responses from a small set of native-context coordinates and test unseen wording/tasks. Compare a shared interaction model with task-specific dictionaries under explicit total-capacity accounting. The useful target is a rule that predicts interventions from context, not just another span that reconstructs observed dc.

Audits: actual saved training/evaluation IDs checked against source metadata for every fold; main training excludes evaluation task and wording; same-task diagnostic exception explicit; orthonormal learned spans checked; source, runner and fitted folds unchanged; 240 cases/2,640 interventions; native replay max logit error 3.87e-5 across batching; oracle normal-equation error <=1.20e-15; finite metrics. Exact context-switch algebra was independently verified for these source cases in H603. FineWeb remains 26,112 documents; broad goal active and incomplete.
