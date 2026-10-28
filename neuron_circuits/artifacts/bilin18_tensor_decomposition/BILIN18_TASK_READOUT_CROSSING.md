## H535 — Changing the readout weakens the general-feature interpretation

Freeze the H534 number and sentiment MLP-write changes. Test 104 newly constructed prompt pairs (208 prompts) across six agreement formats and seven sentiment formats. Items and wording cross output formats; this is not a random population sample or model-training holdout. The original eight answer-token sets remain unchanged for anchor formats. New formats include pronouns, reflexives, other verbs, ratings, and yes/no questions with reversed semantic mappings. The full manifest and controls are fixed before inference; every format is reported.

Compare shared changes with their reversed signs, same-norm nuisance changes, four pre-existing random vectors, and a same-norm output-bias control built directly from the difference of original answer-class mean unembedding rows. Also project the shared change off the span of all original answer-token rows and rescale to its original norm. Test all ten modes at MLP12; test shared and output-bias modes at the final residual stream as well. The unembedding is the model's linear map from final hidden state to token scores. These controls match norm, not covariance or task performance, and are not optimized on these examples.

Each task format passes only if at least 75% of its pairs prefer both intended answers and at least 50% of endpoints produce an intended full-vocabulary top-1 token. Candidate-only success is distinct from operational competence. The modal invariance format requires the same verb form after “can” for both subject numbers and is reported separately.

| Task / format | Both endpoints choose intended contrast | Full-vocabulary top-1 | Pass | Shared MLP change | Output-bias MLP control | Readout-projected shared change |
|---|---:|---:|---|---:|---:|---:|
|agreement/original_verbs|100.0%|62.5%|yes|1.4238|2.1476|0.1347|
|agreement/pronouns|100.0%|62.5%|yes|-0.0172|0.2229|-0.2168|
|agreement/reflexives|100.0%|56.2%|yes|0.2395|0.7057|-0.2580|
|agreement/seems_seem|100.0%|0.0%|no|1.3582|1.7779|0.3113|
|agreement/works_work|100.0%|0.0%|no|1.2252|1.6653|0.2459|
|sentiment/avoid_yes_no|0.0%|6.2%|no|-0.3811|-0.3782|-0.2386|
|sentiment/enjoyment|0.0%|0.0%|no|0.7691|0.8378|0.4591|
|sentiment/opposite_labels|0.0%|0.0%|no|-1.2201|-2.6063|-0.3056|
|sentiment/original_labels|100.0%|56.2%|yes|1.0668|2.4750|0.3856|
|sentiment/rating|0.0%|37.5%|no|0.7699|0.2646|0.6781|
|sentiment/recommend_yes_no|0.0%|50.0%|no|0.4835|0.4142|0.3343|
|sentiment/recommendation|0.0%|50.0%|no|0.5939|0.6682|0.4107|

Effects are signed changes in the two-answer log-odds toward the counterfactual answer, not accuracy changes. Values on failed formats measure contrasts only and cannot establish successful semantic transfer or its absence.

The original agreement format retains a strong shared-vector effect (1.424), but the competent pronoun format has essentially zero effect (−0.017); the competent reflexive format has a smaller effect (0.239). The simple output-bias control has larger effects on all three (2.148, 0.223, 0.706). Positive transfer to new verb forms is candidate-only evidence: both verb formats have zero literal full-vocabulary top-1 under the narrow declared output sets. This does not mean the model cannot perform grammatical agreement; other valid verbs may be its preferred completions.

Removing the original answer-row span leaves 79.2% of agreement vector norm before rescaling, but reduces the original-format effect from 1.424 to 0.135. Pronoun/reflexive effects become negative. For sentiment the retained norm is 93.1%, and the original-format effect falls from 1.067 to 0.386; the output-bias control produces 2.475. Direct final-residual insertion already moves the anchor contrasts by 0.776 for agreement and 0.417 for sentiment. Later model computation amplifies the effects of L12 insertion, but amplification is not proof of a semantic operation.

No alternative sentiment format passes both competence criteria. In particular, reversed yes/no and opposite-sentiment formats fail their native behavior tests, so they cannot adjudicate semantic reversal. Recommendation prompts sometimes emit an intended token but do not distinguish both polarities correctly. The original sentiment transport result survives, but its interpretation remains unresolved beyond an output-associated causal change.

The modal invariance control always prefers “work” over “works” natively, but neither is its top prediction. The shared vector still changes that contrast by mean absolute 0.672 (output-bias control 1.196), versus 0.074 after removing the original answer-row span. This is suggestive of grammatical-form bias insensitive to this construction, not a valid full-task failure claim.

Conclusion: downgrade the claim that these selected late writes are general number/polarity features. A write aligned with answer vocabulary, or an answer-amplifying operation, is a serious explanation; isotropic random controls were too weak to discriminate it. The evidence does not establish that all semantic information is absent, that output directions are the only explanation, or that earlier computation lacks reusable features. Strongly correlated grammatical readouts may share genuine number information, and the direct readout subspace is not a uniquely semantic/nonsemantic partition.

Hypothesis update: H533 alternatives 5 (answer amplification) and 12 (answer-token encoding) rise in priority for these late writes. Alternative 1 (reusable semantic features) requires a different location or stronger readout-general evidence. Alternative 2 (conditional operations) remains viable but needs a semantic prediction beyond operand algebra. Alternatives 3/4 (routing and source/reader relations) motivate the next binding/copying experiment.

Audits: 2,496 intervention records; no excluded prompts; all measurements finite; manifest and frozen source unchanged; self interventions reproduce margins within 8.6e-6 and KL within 3.4e-7. Negative KL at this numerical scale is floating-point noise, not a divergence claim. Shared vectors, output controls, complete-distribution KL/entropy changes and all per-prompt predictions are archived. Float32 computation uses bfloat16-rounded weights from the shared loader. No new FineWeb documents; tracked total 26,112.
