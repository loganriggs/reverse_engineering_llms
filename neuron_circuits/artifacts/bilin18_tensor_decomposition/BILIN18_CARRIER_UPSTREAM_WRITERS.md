## H580 — Upstream value restoration and downstream influence rank writers differently

The previous goal turn made progress: H579 localized the carrier positions and supplied the fixed targets for this experiment. Test L4H0's period and pronoun values and L7H8's verb value. At the carrier token, replace every single earlier attention or MLP output with its aligned base-prompt output. This changes the model normally thereafter. Separately, take only the resulting local value difference and inject its contribution through native producer routing at the final destination; keep the persistent bus and all other producer reads native. These are different interventions, not an additive mediation decomposition.

All 128 edited copying prompts are retained: two templates, four phrases, two delimiters, four name pairs and both variants. Include self controls and direct base-value controls. There are 4,608 paired evaluations (9,216 forward passes in the intervention stage, including controls). Direct base-value controls have only a meaningful isolated-path intervention; their upstream pass is a native capture. No new prompts or natural-text documents.

Measure three distinct quantities: local value distance from base divided by native distance; downstream L14H4 gate distance from phase-only divided by native distance; and full-vocabulary native-to-intervened KL plus intended-answer CE change. KL measures prediction-distribution change; CE measures answer loss. Candidate-margin magnitude is retained separately. None is automatically a semantic explanation or a percentage of independent computation.

Select the upstream writer giving the lowest mean value-distance ratio over old-template delimiters, without using the new template for selection. The selected operations are MLP3 for both L4H0 carriers and MLP6 for L7H8's verb. New-template evaluation is within an already explored panel, not an untouched confirmatory sample.

| Carrier | Selected writer | Old parentheses value ratio | Old commas | New parentheses | New commas |
|---|---|---:|---:|---:|---:|
|L4H0_pos1|mlp_3|0.575|0.539|0.535|0.516|
|L4H0_pos2|mlp_3|0.521|0.495|0.566|0.527|
|L7H8_pos3|mlp_6|0.611|0.657|0.673|0.647|

The selected writer leaves roughly half to two-thirds of the internal difference. These ratios do not imply that the remainder is an independent component. Single-writer clamps change later operations and can introduce counterfactual states.

Crucially, the geometric winner need not have the largest downstream gate effect. For L7H8's verb on old-template commas, MLP6 yields value ratio 0.657 and gate ratio 0.971. Attention5 restores less of the value (0.748), yet has a larger gate movement toward phase-only (0.899). On new-template commas, Attention5 yields gate ratio 1.006, while MLP6 gives 0.981. Thus even the directional influence remains context dependent. For L4H0's period, MLP3 roughly halves the internal difference but leaves the old-template gate nearly unchanged (0.997/1.015). Large internal changes can lie in directions with weak effect on the measured downstream read.

Whole-write and isolated-path behavioral effects also differ. For new-template commas, the verb's MLP6 clamp gives mean absolute candidate-margin change 0.0774 in the full run versus 0.0402 through the isolated value path; native-to-intervened KL is 0.00059 versus 0.00018 (rounded), and mean answer CE change is +0.0290 versus +0.0150. These are not additive shares. Other altered paths and downstream nonlinearities can contribute or cancel.

Audits: self full and isolated-path margins match exactly; direct base-value replay agrees with H579 within 1.53e-5 margin and 7.43e-7 gate coefficient; all measured fields finite; source hash unchanged. Computation is float32 after loading bf16-rounded weights. Tracked FineWeb documents remains 26,112.

Research implication: causal response directions are a more promising unit to test than raw value-vector similarity alone. This is a hypothesis, not proof of a canonical basis. The nearest MLP contributes strongly to contextual state, while a different upstream operation can dominate a selected readout. We should not compress away directions merely because their variance is small, or identify a feature solely from a large norm restoration.

Next priority is a fresh diverse panel, before further narrowing this copying circuit: copying, person binding, agreement, factual recall, and sentiment, each with at least two templates, answer-changing counterfactuals and matched answer-preserving context edits. Lock MLP3/MLP6 and Attention5 as candidate writer operations and include neighboring controls. Check whole-family competence without dropping individual failures. Compare held-out internal value restoration against full-output KL, answer CE, and donor-directed effects. Test whether response-based grouping transfers better than raw-state grouping; use whole-task holdouts as well as template holdouts. Carrier positions must be declared from syntax before interventions; failure to find an analogous position is itself a limitation, not permission to select the best token after seeing effects.

The objective remains active and incomplete: these tests establish conditional routes on a small panel, not the model's general feature ontology.
