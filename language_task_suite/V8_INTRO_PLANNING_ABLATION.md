# V8 introduction timing: correct, variable, but no realism gain

The main objective remains a varied, language-like **sampled** distribution with known rules for testing weight-based interpretation across DGP settings. This experiment asks a narrow generator question: does moving an object's initial description to the point where it first matters improve document form? It does not test model capability or interpretability, so no model was trained on this setting.

`intro_interleave_rate` schedules each object's introduction anywhere before its first event or recap. At zero, the original V8 renderer and random stream are retained. At one, every object has a sampled introduction slot and the renderer brings it forward if a reference would otherwise precede it. The world ledger, event order, recap plan, and answers remain paired across settings. This is a real change in discourse order, not a lexical substitution or a new task.

On two 300-document task-rich panels, all 2,868 visible-prefix answer checks and matched upstream edits pass; every document changes between syntax settings. The paired planning screen compares the **same sampled worlds** at timing rates zero and one, requiring both realizations to have at least 100 words. All 64 documents change in each panel.

| Source-query prevalence | Seed | Intro timing | Word types | Top-ten share | First-40 repeated-word fraction | Mean first answer token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reference | 73381 | 0 / 1 | 220 / 224 | .311 / .317 | .246 / .286 | 78.8 / 51.7 |
| Reference | 73382 | 0 / 1 | 230 / 237 | .325 / .325 | .250 / .274 | 83.0 / 54.2 |
| Task-rich | 73481 | 0 / 1 | 219 / 203 | .314 / .324 | .242 / .297 | 78.8 / 49.2 |
| Task-rich | 73482 | 0 / 1 | 215 / 218 | .315 / .319 | .242 / .272 | 78.4 / 48.2 |

The current saved independent FineWeb reference panels have 2,358/2,435 word types, top-ten share .207/.217, and first-40 repetition .160/.189. Eligibility and seeds differ, so those figures are a scale reference, not a paired causal comparison with natural text. Introduction timing reduces the distance to the first answer, making task sites more front-loaded, but worsens early repetition in every paired generated panel and barely changes lexical diversity. It is a useful structural control and a **negative realism result**, not a replacement DGP.

The unchanged rule behavior is verified in [the semantic audit](results/v8_intro_interleave_source50_semantic_audit_300x2.json). The [reference-prevalence](results/v8_intro_planning_paired_reference_n64x2.json) and [task-rich](results/v8_intro_planning_paired_source50_n64x2.json) panels contain the aggregate form measurements. No FineWeb text or generated document bodies are saved in those results.

This rules out introduction order alone as the next training condition. The larger gap comes from a small content lexicon and recurring clause frames. The next generator version needs sampled discourse acts that express distinct propositions and more varied, typed lexical realization while preserving a visible-text rule audit. Those two changes should be tested together and separately, since a larger lexicon alone previously left top-word concentration almost unchanged in V17. Only after a broader distribution is verified should we train a new model grid and compare weight methods on behaviorally learned rules.
