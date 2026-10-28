## H543 — Operation-linked L8H3 behavior survives crossing prose and list format

The previous goal turn made progress by matching source/final tokens and positions and separating query/key contextual contributions. A prose/list heuristic remained plausible. Keep the four heads and identity-map signs fixed, and cross intended operation with surface format using the same eight name pairs and final “to”.

Primary cells are narrative binding, list binding/retrieval, narrative copying and list copying. Narrative copying uses “He spoke to A. He spoke to A. He spoke to”; list binding uses “from A, to B; from B, to A; from G, to”. Tokenizer-selected “Note” padding matches source and final positions across operations within each primary format. Also test reversed demonstration order in binding lists and a longer temporal copying narrative. All interventions target only the last changed name source, with one head and final attention destination affected.

All six cells pass native competence: both intended contrasts are correct on every pair; full-vocabulary top-1 is 87.5% for the repeated narrative-binding anchor and 100% elsewhere. The run contains 96 prompts, including 16 repeated narrative-binding prompts and 80 newly composed prompts. New prompts reuse known names; this is format/operation transfer, not unseen model-training data.

| Operation / format | Reader | Positive routing coefficient | Mean causal third-token identity effect | Frozen sign predictions |
|---|---|---:|---:|---:|
|entity_binding/list|L14H4|16/16|-1.8720|16/16|
|entity_binding/list|L8H3|16/16|-0.6870|16/16|
|entity_binding/list_reversed|L14H4|16/16|-1.9476|16/16|
|entity_binding/list_reversed|L8H3|16/16|-0.7301|16/16|
|entity_binding/narrative|L14H4|16/16|-3.0854|16/16|
|entity_binding/narrative|L8H3|16/16|-0.2830|16/16|
|sequence_copy/list|L14H4|11/16|-0.1163|16/16|
|sequence_copy/list|L8H3|0/16|0.5770|16/16|
|sequence_copy/narrative|L14H4|7/16|0.0532|16/16|
|sequence_copy/narrative|L8H3|0/16|0.2010|16/16|
|sequence_copy/narrative_temporal|L14H4|16/16|-0.4213|16/16|
|sequence_copy/narrative_temporal|L8H3|0/16|0.1645|16/16|

L8H3 suppresses the queried identity in both binding formats and copies the repeated identity in both copying formats, including the temporal narrative. A pure prose-versus-list rule cannot explain these observed signs. Matching source/final positions within the primary format comparisons also rules out those absolute positions alone. The locked coefficient-conditioned rule predicts all 192 leading-head third-token effects.

L14H4 is not a uniform copying reader: its identity effect is small/mixed in the simple copying cells and suppressive in every temporal-copying prompt despite perfect native competence. Its conditional identity transformation remains predictable, but the head's sign is not determined simply by the intended copying operation. This further narrows its task-level interpretation.

The evidence for L8H3 is stronger than a lexical/format correlation, but it does not establish an abstract binding variable. Local words around the source and query still differ. More importantly, with only two names, suppressing the query identity can favor the other answer without selecting an association. The new list task is mapping retrieval; it is not assumed identical to narrative entity binding at the algorithmic level.

Next: compare identity mappings A→A/B→B with swap mappings A→B/B→A under identical local query syntax and balanced name frequencies. Evaluate the query position, the mapped-value position and the competing value separately. A fixed query suppressor should remain suppressive even when the query is the correct answer; a correct-value selector should distinguish the demonstrated values. Screen competence before interpreting task success or failure.

Audits: exact source/final token and position matches verified for all 32 primary matched directions (two formats × eight pairs × two query identities). 1,248 interventions and 384 third-token comparisons; self margins reproduce exactly, token-value replay error <=1.72e-5, coefficient factorization relative error <=1.8e-7. All measurements finite and source/manifest hashes unchanged. No new FineWeb documents; tracked total 26,112. Native caches and the complete prompt manifest are archived.
