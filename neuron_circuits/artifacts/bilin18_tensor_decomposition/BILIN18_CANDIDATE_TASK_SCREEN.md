## H840 — Natural-text stability does not ensure task availability or reproducible block contributions

We froze all16 H839 candidates and examined their final-position preMLP activations on eight task families. Five families pass the declared competence gate in both formats: agreement, entity binding, repeated-sequence copying, capital recall and sentiment. This gives a diverse usable causal panel, but the feature screen weakens a direct leap from natural-text stability to task-general units.

**Design and scope.** Reused H533 suite:16 pairs/family, two format/lexicon splits, answer-changing endpoints and a same-answer nuisance prompt. All128 pairs token-valid,384 contexts. Entity-binding prompts received the previously motivated `Yesterday, ` prefix for name tokenization. This is not fresh task confirmation. No candidates were replaced or ranked using these results. Both H836 seeds retained, training-token means and scales frozen. Capture is the normalized MLP input, matching the shape coordinates used by these featurizers. The reserved test split remains untouched; natural-document pool stays26,112.

Competence requires at least4 valid pairs/cell, both endpoints binary-correct on at least75% of pairs and intended semantic-group full-vocabulary top1 on at least50% of endpoints. Semantic groups include fixed agreement alternatives and sentiment synonyms; this is not exact-token accuracy for those families. All pairs retained within qualifying cells, not just individually successful pairs.

|Family|Split|Pairs|Both endpoints binary-correct|Full-vocab intended group|Pass|
|---|---|---:|---:|---:|---|
|agreement|discovery|8|1.000|0.562|True|
|agreement|transfer|8|1.000|0.562|True|
|capital_recall|discovery|8|1.000|0.875|True|
|capital_recall|transfer|8|0.875|0.875|True|
|context_association|discovery|8|0.750|0.000|False|
|context_association|transfer|8|0.875|0.000|False|
|entity_binding|discovery|8|1.000|1.000|True|
|entity_binding|transfer|8|1.000|1.000|True|
|negation|discovery|8|1.000|0.375|False|
|negation|transfer|8|1.000|0.312|False|
|relational_order|discovery|8|0.000|0.000|False|
|relational_order|transfer|8|0.000|0.000|False|
|sentiment|discovery|8|1.000|0.625|True|
|sentiment|transfer|8|1.000|0.688|True|
|sequence_copy|discovery|8|1.000|0.812|True|
|sequence_copy|transfer|8|1.000|0.562|True|

**Frozen candidate availability.** Below, active counts are seed8351/8352; joint support counts both seeds active over240 contexts in the ten competent cells, including nuisance contexts. The cosine compares each decoded component contribution at the same context, conditional on both being active.

|Site|Architecture|Matched units|Active counts|Joint active|Decoded contribution cosine|
|---|---|---|---|---|---:|
|1|scalar48|439→189|24/28|21/240|0.950|
|1|scalar48|83→409|33/81|33/240|0.952|
|1|scalar48|497→397|77/69|52/240|0.931|
|1|scalar48|57→472|103/111|98/240|0.939|
|1|block8|46→43|54/68|48/240|0.136|
|1|block8|47→45|33/5|5/240|0.680|
|1|block8|60→56|66/32|4/240|0.457|
|1|block8|29→57|31/13|5/240|0.357|
|2|scalar48|176→401|1/2|0/240|undefined|
|2|scalar48|432→111|15/15|8/240|0.960|
|2|scalar48|415→494|27/24|21/240|0.913|
|2|scalar48|140→57|1/2|0/240|undefined|
|2|block8|16→46|38/47|35/240|0.505|
|2|block8|39→59|6/9|1/240|0.652|
|2|block8|59→23|157/212|134/240|0.469|
|2|block8|0→45|128/65|65/240|0.205|

Scalar site2 candidates176→401 and140→57 have no jointly active contexts in the competent panel. Others are sparse or concentrated in particular formats. This is useful negative evidence about this shortlist's availability, not evidence that their directions are meaningless elsewhere.

For scalar components, the high conditional decoded cosine mostly restates their frozen decoder alignment plus coefficient-sign agreement: it is not an independent demonstration of semantic reproducibility. For blocks, the actual within-subspace coordinates matter. Site2 block16→46 had strong prior subspace overlap but its task-panel decoded contribution cosine is only about0.505; site1 block46→43 is about0.136 despite48 jointly active contexts. Gate coactivation alone is insufficient to establish a reusable computation.

**Contrast caveat.** The script retains paired answer-changing and nuisance component distances. Nuisance changes are often larger, but agreement/copy nuisance prompts change the final token, and some others change length or format. Training-token-mean subtraction does not make those confounds disappear. Do not interpret these raw distances as causal selectivity or semantic specificity. Same-final-token, position-matched controls are needed in the intervention analysis. Repeated-sequence copying is not distinct-value binding. Association probes also have ambiguous intended completions. Sentiment now passes this specific suite; older failures on different templates remain valid and must not be silently revised.

**Interpretation.** Current evidence favors reproducible directions for subsets of natural inputs, while task availability, activation support, and the coordinate vector within a block remain separate empirical questions. It weakens the idea that selecting globally stable blocks directly produces general-purpose task concepts. It does not yet discriminate semantic directions from lexical/context geometry, nor establish causal necessity or sufficiency.

**Next experiment.** Keep all16 candidates and both seeds. On the five competent task families, replace only the candidate contribution at the normalized MLP input with its donor counterpart, preserving the rest of the native input and residual stream. Compare bidirectional answer-changing donors, same-answer donors, matched-norm direction controls, and identity interventions. Record exact final-token/position matching, intervention dose, task-margin transfer and output KL separately. Preserve inactive/no-effect candidates. This tests whether decoded coordinates carry causal task information before investing in more elaborate manifold labels; an MLP-input-only patch is a controlled module intervention, not a natural residual-state replacement.

**Audits.**384 native contexts plus12 duplicate reference forwards; native logits exactly match the established runner. All16 candidates retained, captures finite, source/checkpoint hashes unchanged, zero reserved-test evaluations. Runtime outputs include full example-level predictions, normalized inputs, decoded candidate vectors and support masks for reproducible intervention construction.
