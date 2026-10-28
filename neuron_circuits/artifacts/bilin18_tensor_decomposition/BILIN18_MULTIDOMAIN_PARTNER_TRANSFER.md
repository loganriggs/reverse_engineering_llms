## H883 — Partner transfer generalizes across tasks, but is directional and does not establish universal interchangeability

Executed the H882 frozen plan on all 192 competent agreement, language-recall and human-name-binding contexts at the fixed L8H3 reader. Task-context partner sharing reduces mismatch in all three domains. Copy-context partner sharing is weaker and worsens binding mismatch. The result extends the mechanism beyond capital recall while exposing a substantial limit: the donor context matters, and sharing partner activations is not universally beneficial.

### Frozen design and scope

Eight conditions: live, recipient-native linearization, and normalization-only / partner-only / full-Jacobian sharing, each using copying or task contexts as donors. Replace final-token L8–17 MLP responses around recipient-native offsets. Attention remains live. Each domain has eight pairs, two formats, task/copy contexts and both source endpoints; every condition includes identity and both intervention signs. The reader and conditions were fixed before intervention outcomes.

Common amplitude is the geometric mean of absolute native source-routing coefficients in the matched contexts. The perturbation is the same source-bus contrast projected through L8H3 in each context. Minimum amplitude is 0.00945; none are zero or below the declared tiny-amplitude flag of 1e-6. No pairs or low-effect examples were removed. Selection depended on H882 behavioral competence, so this is prospective mechanistic transfer conditional on competence, not an unconditional task sample or an independent natural-language test set.

### Task-answer contrast, shared-scale uncapped readout

|Domain|Live mismatch|Normalization only, copy donor|Normalization only, task donor|Partners, copy donor|Partners, task donor|Full, copy donor|Full, task donor|
|---|---:|---:|---:|---:|---:|---:|---:|
|agreement|1.907|1.908|2.319|0.660|0.423|0.798|0.342|
|language|0.300|0.321|0.252|0.251|0.088|0.221|0.130|
|binding|0.216|0.224|0.167|0.343|0.035|0.299|0.070|

These are canonical pair-average response mismatches, not task errors or accuracy losses. Agreement's live task-context response RMS is only 0.0754, versus 1.766 for language and 2.197 for binding. Its relative mismatch exceeding 100% reflects this weak denominator. Absolute task-answer mismatch falls from 0.144 to 0.032 with task-partner sharing, and from 0.144 to 0.090 with copy-partner sharing. That is evidence about a weak reader response, not evidence that L8H3 is a major agreement mechanism.

Language task-partner sharing reduces absolute mismatch 0.530→0.155, while copy-partner sharing reduces it to 0.423. Binding task-partner sharing reduces 0.474→0.076, but copy-partner sharing increases it to 0.925. Thus the directional result is not only a relative-denominator effect. Binding's source-token and task-answer contrasts use the same two names in reversed order; their equal error values are not independent evidence.

Both templates preserve the key directions. Language task-partner errors are 15.2% and 9.6% versus live 37.1% and 25.1%. Binding task-partner errors are 5.4% and 4.2% versus live 22.1% and 22.6%, while copy-partner errors worsen to 33.4% and 35.4%. Agreement remains weaker and more template-dependent.

### Individual interventions and vocabulary-wide effects

Averaging conceals some heterogeneity. For task-partner sharing, errors across individual signed interventions are 31.8% for agreement, 18.1% for language and 12.6% for binding, versus canonical pair errors 42.3%, 8.8% and 3.5%. These retain endpoint orders, formats and both perturbation signs before computing the norm error. The stronger averaged result is therefore not a claim of per-example equivalence.

Measured native-distribution KL is KL(unedited native output || edited native output), over the full vocabulary. This is perturbation magnitude, not cross-context KL, learned-featurizer fidelity, or a model-quality improvement. In task contexts, mean KL for live versus copy-partner sharing is approximately 0.00003→0.00011 for agreement, 0.00208→0.00137 for language, and 0.00507→0.01902 for binding. Binding top-token change rate increases 4.7%→9.4% in that adverse condition. In copying contexts, task-partner sharing raises mean KL from 0.00145 to 0.00597 for binding while improving the selected-margin match. Better margin transfer therefore does not imply less vocabulary-wide disturbance.

All native and controlled-readout summaries, separate templates, absolute response norms, per-condition KL and top-token changes are retained. Conclusions above explicitly concern the controlled readout; native normalization and saturation remain part of the original model.

### Interpretation and next discriminating test

The cross-domain result supports local computation conditioned by partner activations. The directional asymmetry argues against treating any donor partner pattern as an interchangeable task-independent unit. Possible explanations include donor-specific interactions with live attention, lexical content in the partner vector, and different distances from recipient-native computation.

The next useful distinction is exact lexical donor versus reusable context transformation. Keep the fixed reader and evaluate partner patterns estimated from other lexical pairs, with a frozen discovery/transfer split within each domain. Compare exact matched donors, context-mean partner patterns built only from discovery pairs, and incorrect-lexical same-context donors. Keep the recipient normalization and offset fixed, use the same forward interventions, and report response strength, individual effects and vocabulary-wide changes. This tests whether a lower-dimensional context component generalizes before training a flexible conditional featurizer. Exact-pair oracle transfer alone is insufficient to establish a meaningful manifold or unsupervised computational unit.

### Verification

4,800 full-model forwards. All identity final states replay exactly. Native logits replay H882 within 2.1e-05; selected readout within 8.43e-06. All ten layer Jacobian implementations checked against autodiff (relative 3.36e-16), central finite differences (1.08e-08) and factorization (7.46e-16). Independently checked 1,536 donor assignments. Identity KL max absolute 8.73e-12; all KL values and selected outputs finite. Hashes unchanged. Pair bootstrap intervals use4,096 uncorrected resamples over8pairs per domain. No reserved test evaluated; natural-document pool unchanged at26,112. Broad goal remains active.
