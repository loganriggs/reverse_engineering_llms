# V10 fact-bearing discourse acts: true propositions, limited form gain

The main goal is a realistic, variable **sampled** language DGP with known rules, so that weight-based interpretability methods can be tested across generator settings. V10 tests whether adding distinct discourse acts to the V8 event world improves form while retaining causal ground truth. The four arms separate the effects of typed lexical variation (V9), discourse acts, and both together.

After a sampled update, the planner may add one of four kinds of true statements: a current-state observation, a before/after contrast, a comparison of two objects, or the history of two reports about the same object attribute. Each act is sampled with rate .8 in this pilot and realized in one of three forms. Claims use the live world state at that point; they do not update it. The act plan is fixed across surface forms. The scorer reconstructs answers from visible text after removing these redundant statements. No model is trained with rule labels.

An independent sentence parser extracts the act's objects, attributes, values, and sources, then checks them against a separate replay of the event ledger. On two 300-document task-rich panels it validates **2,571** fact acts (including 139 source-history acts). All **2,767** answer sites and matched upstream edits pass, both without and with typed lexical variation. Removing the added acts leaves the same answer sequence. [Semantic audit](results/v10_fact_acts_semantic_audit_300x2.json). This establishes truth and task correctness for these forms; it does not establish realistic discourse or an exact marginal next-token oracle.

The following paired 64×100-word panels use the same sampled worlds and eligibility for V8, lexical-only, acts-only, and both. The panels show word types, then top-ten word share, then first-40 repeated-word fraction.

| Prevalence | Seed | V8 | Lexical only | Acts only | Both |
| --- | ---: | --- | --- | --- | --- |
| Reference | 85119 | 231 / .320 / .243 | 466 / .312 / .242 | 232 / .312 / .243 | 461 / .311 / .242 |
| Reference | 85120 | 235 / .316 / .248 | 476 / .300 / .245 | 239 / .312 / .248 | 483 / .304 / .245 |
| Task-rich | 85219 | 233 / .314 / .234 | 472 / .310 / .231 | 236 / .312 / .234 | 472 / .311 / .231 |
| Task-rich | 85220 | 211 / .328 / .257 | 456 / .319 / .257 | 224 / .317 / .257 | 466 / .315 / .257 |

Each eligible V10 document has about 4.2–4.4 added acts. Yet the acts barely affect the first 40 words, because V8 still opens with three full object introductions. Top-ten share improves only modestly. Both combined remains far from the saved independent FineWeb panels (2,358/2,435 word types, .207/.217 top-ten share, .160/.189 early repetition). [Reference-prevalence form screen](results/v10_fact_acts_paired_reference_n64x2.json); [task-rich screen](results/v10_fact_acts_paired_source50_n64x2.json).

An actual V10 excerpt reads: “The ledger listed the owner of boot as Wendy. In Rosa's report, Hannah relocated the shirt to the tray. Before Rosa's report, the shirt had counter as its place; afterward it had tray.” The claims are consistent, but wording such as “had counter as its place” is stiff, and several forms explicitly say *report* or *entry*. This remains a controlled synthetic distribution. More appended sentences are not solving the opening-form problem.

V10 is useful as a truth-checked discourse-structure stress test. The next version should express **initial world facts through a sampled sequence of acts**, rather than always emitting one complete introduction per object and then appending variation. That requires a visible-text parser that can accumulate partial facts, aliases, and updates. It should also sample genre-specific document purposes and naturalized realizations, then repeat the semantic audit and independent FineWeb screens. Only after the DGP is substantially broader should new model training be interpreted as evidence about realistic-language circuits; for every rule, behavioral competence remains the gate before inspecting weights.
