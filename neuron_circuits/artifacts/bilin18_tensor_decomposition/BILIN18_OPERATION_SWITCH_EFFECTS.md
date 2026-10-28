## H656 — Copy-to-task local transformations transfer partially across wording

H655 tested how task-conditioned processing responds to verb/suffix changes. H656 instead changes the requested operation: copy the first input word/number → perform the task. Within each contrast, input, instruction verb and answer suffix stay fixed. Copy prompts are identical in the original and paraphrase panels. Target task prompts use the second wording. All six families, four forms and four held-out items remain, giving96 contrasts and288 probe evaluations.

Fit copy-to-task normalized MLP-context shift means on four original-wording discovery items. Evaluate these shifts on the second-wording held-out items. Compare: a mean fit on second-wording discovery items; original-wording copy-to-constant-yes means; original-wording copy-to-task means pooled over the other five families; and that pooled mean rescaled to the original same-task discovery mean norm. Norm matching uses no held-out target information. The norm-matched control tests whether advantages over pooled other tasks are simply differences in shift magnitude, though it does not control every family-specific nuisance.

For each contrast, retain the native copying receiver's attention, normalization derivatives and readout. Replace bilinear MLP operating contexts10–17 with native context plus the selected shift. Exact target task contexts provide the reference. Use the same seed654 three unit directions injected directly after block9. No output fitting or finite activation intervention occurs. This is a conditional change of local derivative operators, not a complete copying→task model replacement.

Relative effect error compares predicted versus exact context-replacement logit derivatives under receiver-probability-weighted centered logits, divided by the exact effect norm. No-change error1; lower is better.

| Family | New-wording mean | Original same-task mean | Constant control | Other-task mean | Norm-matched other |
|---|---:|---:|---:|---:|---:|
|addition|0.210|0.455|0.965|0.703|0.739|
|agreement|0.273|0.649|1.036|0.856|1.037|
|antonym|0.369|0.818|1.042|0.846|0.869|
|capital|0.270|0.518|0.913|0.761|0.849|
|plural|0.306|0.646|0.857|0.889|0.828|
|successor|0.289|0.496|0.812|0.763|0.796|

The original same-task mean beats every control in all six family means: constant in284/288 individual cases, pooled other tasks in268/288, and norm-matched pooled other tasks in273/288. These repetitions share templates/items and are not independent trials. Antonym transfer is weak:0.818 versus other tasks0.846, with some individual errors above1 (maximum1.235). Thus average transfer does not imply uniform transfer.

New-wording means are better in all six families, with mean errors0.210–0.369. Cross-wording same-task errors0.455–0.818 show substantial residual wording dependence. Norm matching does not erase the same-task advantage. Exact context effects have magnitude0.186–0.569 times the native response, median0.290, so the relative-error pattern is not driven solely by near-zero effect denominators. These ratios are not causal shares.

This provides stronger evidence for task-conditioned transformations than the previous verb/suffix assays: a discovery-only copy-to-task shift retains useful directional information across a second wording and new items. However, other-task controls also differ in input/answer vocabulary, the paraphrases share words, and four families change token length. The transformation remains conditional on known family identity and native receiver states. It does not isolate a semantic primitive or establish a unique feature system.

Competence remains a separate limitation. Task top1 next-token accuracy is0% except successor9.375% in the second-wording panel; candidate accuracy ranges from12.5% to100%. Copying instructions are more reliably followed. Predicting the task prompt's local derivative does not imply executing that task or improving answers. H656 measures neither actual output distribution replacement nor finite intervention efficacy.

The next informative step is to test finite activation edits using these frozen mean shifts on copying prompts, allowing the rest of the network to run. Compare the resulting output distribution with the natural task-prompt distribution using output KL, while separately reporting target loss, candidate choice, and exact next-token accuracy. Include unchanged copying, exact target-context replacement, same-task means, and other-task controls across a fixed amplitude grid; do not choose amplitudes on held-out performance. Exact target-context replacement may itself fail to reproduce the task output because attention and earlier computation remain receiver-specific. That failure is informative about the limits of treating this MLP-context transformation as a computational unit. Finite effects must be measured rather than inferred from derivative fidelity.

Audits: no new whole-model forwards;768 cached attention evaluations;1344 batched JVP calls. Independent native whole-block derivative replay maximum relative error1.46e-6; reconstructed contexts match captured float32 values within2.98e-6. All hashes unchanged, finite metrics, explicit original/new item/answer/split equality checked. All96 contrasts retained. FineWeb remains26,112 unique documents. Previous goal turn was progress. Broad research goal remains active and incomplete.
