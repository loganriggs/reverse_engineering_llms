## H655 — Same-operation wording transfer is partial, with substantial lexical sensitivity

H654 showed that task-conditioned context shifts predict local output effects better than generic controls. H655 changes the task instruction wording while preserving the inputs, answers, discovery/transfer item splits, Give/Provide verbs, Answer:/Result: suffixes, and copying/constant control prompts. All576 prompts are evaluated; no output-based filtering occurs.

| Family | Original phrase | Second phrase | Token-length change |
|---|---|---|---:|
|addition|the sum|the total|0|
|agreement|the present-tense form of be that agrees with this subject|the correct present-tense conjugation of be for this subject|+1|
|antonym|the opposite|the antonym|+2|
|capital|the capital|the capital city|+1|
|plural|the plural|the plural noun|+1|
|successor|the following month|the next month|0|

The enclosing instruction remains Give/Provide [phrase] for the input. These are intended synonymous instructions, not proof that the model treats them as equivalent. Most phrasings retain substantial lexical overlap. Only addition and successor preserve token length; other families retain length confounds. This is one alternative wording per family, not a general paraphrase benchmark.

For each of96 held-out task contrasts, use the same three seed654 unit probes directly after block9. Compare the frozen original-wording same-task mean with a new-wording mean fit on its four discovery items, unchanged copying/constant control means, and frozen original-wording means pooled across the other five tasks. Receiver attention, normalization, and readout remain native; shifts replace MLP operating contexts10–17. No output fitting or probe changes. All source/target item/answer/split fields are verified equal across panels; unchanged control logits reproduce exactly.

Relative effect error compares predicted versus exact context-replacement logit derivatives, normalized by the exact effect norm. Logits are centered and weighted by native receiver probabilities. No-change error1. This is a local derivative prediction metric, not answer accuracy or finite-intervention KL.

**Verb change:**

| Family | New-wording mean | Original same-task mean | Copy | Constant | Other tasks |
|---|---:|---:|---:|---:|---:|
|addition|0.210|0.553|0.858|0.874|0.773|
|agreement|0.231|0.601|0.754|0.791|0.840|
|antonym|0.409|0.817|0.800|0.848|0.777|
|capital|0.174|0.376|0.829|0.891|0.800|
|plural|0.285|0.642|0.671|0.819|0.947|
|successor|0.267|0.545|0.816|0.809|0.741|

**Suffix change:**

| Family | New-wording mean | Original same-task mean | Copy | Constant | Other tasks |
|---|---:|---:|---:|---:|---:|
|addition|0.251|0.336|0.605|0.484|0.622|
|agreement|0.253|0.379|0.499|0.570|0.540|
|antonym|0.419|0.517|0.577|0.574|0.521|
|capital|0.258|0.343|0.591|0.486|0.580|
|plural|0.257|0.383|0.453|0.464|0.598|
|successor|0.456|0.469|0.710|0.545|0.767|

The original same-task mean beats all three controls in11/12 cell means. Antonym verb changes are the exception: original-wording error0.817 is slightly worse than copying0.800 and other-task0.777. The antonym suffix advantage over other tasks is tiny (0.517 versus0.521), so it should not carry a strong mechanistic interpretation. All cross-wording cell means beat no change.

New-wording means are better than original-wording means in every cell. The degradation is often large for verb changes, including equal-length addition and successor. Thus token-count changes alone cannot explain all the loss. The result favors partial shared instruction-conditioned structure plus meaningful wording sensitivity, rather than either perfect semantic invariance or purely unrelated lexical responses. Lexical overlap, answer distributions and remaining formatting similarity prevent a clean semantic interpretation.

**Competence remains separate:** task candidate/top1 accuracy is addition12.5%/0%, agreement50%/0%, antonyms93.75%/0%, capitals100%/0%, plurals100%/0%, successor21.875%/9.375%. The successor wording improves these metrics relative to H653, but competence was not used for selection or fitting. Candidate success is not full-response success. The functional transfer results are compatible with instruction processing even when requested operations are poorly performed.

The next experiment should directly change the requested operation at fixed input and suffix, rather than only measuring how an instruction-conditioned state responds to a verb or suffix change. Use copying→task context changes from the existing paired panels; fit original-wording discovery items and evaluate new-wording transfer items. Compare same-operation, pooled other-operation, and new-wording own means, preserving generic formatting controls and native competence reporting. This reuses the crossed design to test a closer candidate for an operation transformation. It still will not isolate semantics from all instruction differences, but failure would sharply limit the current operation-unit interpretation.

Audits:576 whole-model forwards for the second panel;768 cached attention evaluations and1728 batched JVP calls for functional evaluation. Independent whole-block native derivative replay errors below1.31e-6; reconstructed MLP contexts within3.39e-6. All runtime hashes unchanged and finite metrics. Explicit item/answer/split matching verified; unchanged control logits identical. No excluded families, items or probes. FineWeb remains26,112 unique documents. Previous goal turn was progress. Broad research goal active and incomplete.
