## H544 — Mapping tests reject a fixed query suppressor and do not reveal a selective value reader

H543 strengthened the operation-linked interpretation of L8H3 across prose/list format, but two-name swap tasks can be solved partly by suppressing the query identity. This experiment compares identity mappings A→A/B→B with swaps A→B/B→A, in both demonstration orders. The local query syntax, names, name frequencies and positional layout are fixed across mapping type; only demonstrated target assignments change.

Keep all four readers fixed. For each prompt, intervene separately on the last query-name position, the value demonstrated for that query, and the other demonstrated value. Replace one source-to-head bus contribution at the final destination, allowing the rest of the model to respond. Two absent-token replacements measure relative copying/suppression; replacing the source value by the midpoint of A/B bus vectors neutralizes that source's A-versus-B identity distinction. The midpoint intervention is local and does not remove all information about the source from the model.

The run uses 32 prompt pairs (64 prompts). The 32 swap prompts repeat the two list conditions from H543; 32 identity-mapping prompts are newly composed. All examples and heads/sites are retained without selecting successful individual cases.

### Native competence is a material limitation

| Mapping | Demonstration order | Pairs with both intended contrasts correct | Full-vocabulary intended top-1 | Gate |
|---|---|---:|---:|---|
|identity|ab|37.5%|68.8%|fail|
|identity|ba|0.0%|50.0%|fail|
|swap|ab|100.0%|100.0%|pass|
|swap|ba|100.0%|100.0%|pass|

Swap mappings remain fully competent. Identity mappings fail the operational gate in both orders, despite some correct individual outputs. Consequently these results cannot establish the mechanism of successful general identity-map lookup. They can still test what the selected readers do and how their local effects relate to correct-answer evidence on this explicitly specified task.

### Source-specific identity effects and neutralization

Positive third-token identity effects mean copying; negative effects mean suppression. Positive midpoint changes improve the correct-versus-other answer margin, while negative changes weaken it. Values average 32 prompt cases per mapping/site, including both demonstration orders and query names.

| Reader | Mapping | Source | Third-token identity effect | Midpoint change in correct-answer margin |
|---|---|---|---:|---:|
|L14H4|identity|query|-0.0984|0.0603|
|L14H4|identity|mapped_value|0.3610|-0.1719|
|L14H4|identity|other_value|0.3889|0.1845|
|L14H4|swap|query|-1.9098|-0.9422|
|L14H4|swap|mapped_value|1.8797|-0.9028|
|L14H4|swap|other_value|1.7934|0.8384|
|L8H3|identity|query|0.3669|-0.1773|
|L8H3|identity|mapped_value|0.6371|-0.3059|
|L8H3|identity|other_value|0.6068|0.2952|
|L8H3|swap|query|-0.7085|-0.3380|
|L8H3|swap|mapped_value|1.1773|-0.5487|
|L8H3|swap|other_value|1.1925|0.5283|

The fixed-query-suppressor hypothesis is not supported for L8H3. It suppresses the query in all 32 swap cases, but copies it in 25/32 identity cases. Neutralizing the query distinction therefore usually harms correct-answer evidence in both mapping types, contrary to the preregistered prediction that removing query suppression would help identity mappings. L14H4's identity-query effect is weak and mixed. This prevents attributing the model's identity-map failures simply to an unchanging query suppression rule.

Both readers copy demonstrated values, but they do not show a clear preference for the value belonging to the queried key. For L8H3, the mean correct-value minus other-value identity effect is +0.030 for identity mappings and −0.015 for swaps; the correct value wins only 17/32 and 16/32 paired comparisons. L14H4 differences are −0.028 and +0.086, with wins in 13/32 and 16/32. This does not support identifying either reader as a selective lookup mechanism on this panel.

The midpoint tests confirm the practical distinction: for L8H3, neutralizing correct-value evidence weakens the correct answer in all 32 cases per mapping type, while neutralizing competing-value evidence strengthens it in all 32. Both value paths carry behaviorally relevant identity information, including information favoring the wrong answer. Their coexistence is compatible with a pool of value evidence plus query-dependent correction, or with selection distributed elsewhere. It is not a complete explanation of natural task performance.

### Updated hypotheses and discriminating next step

The stable conditional identity transformation remains supported. Its task-level interpretation must narrow: it is neither a fixed query suppressor nor, from these measurements, a uniquely selective reader of the requested value. Local query syntax and name frequency do not determine its sign alone, since demonstrated mapping changes alter query behavior while those quantities stay fixed. Identity-map competence failure means the abstract “binding” interpretation is still unproved.

Leading alternatives now include: (1) multiple value identities are read broadly and query-dependent corrections disambiguate easy two-name cases; (2) true association selection occurs in other heads or upstream representations, with these readers supplying partial evidence; (3) template/recency effects rather than a general mapping algorithm explain some success; and (4) the tested mappings exceed the reliable capability of this model in certain formats. Keep all four live.

Next H545: introduce three-name permutations, including cycles and fixed points, in competent formats. With three candidates, suppressing the query no longer determines the answer. Freeze the current heads, score full-vocabulary correctness and all three candidate answers, and measure query, correct-value and competing-value paths separately. Compare selective association evidence with broad copying and signed query correction; report competence failures rather than selecting a passing subset. This is a stronger falsification test than another round of two-name sign confirmation.

### Verification and scope

768 geometry records, 2,368 interventions and 768 third-token contrasts. Source positions and token identities are asserted for each role; all values finite, self interventions produce exactly zero margin change, manifest/source hashes unchanged. These are correlated structured samples with hybrid local interventions, not population estimates or whole-circuit necessity tests. No new FineWeb documents; tracked total 26,112. The broad research goal remains active and incomplete.
