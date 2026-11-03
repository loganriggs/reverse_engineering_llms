# Fact-clause variation preserves rules but remains synthetic

V8 is a controlled structural change to the V5 world-document generator. It samples the same event ledgers, sources, aliases, recap plans, and answers, but expands fact-bearing clause choices from three to five introduction forms and four to eight attributed event forms. It also optionally states when a later same-key report supersedes an earlier source's account. Every new clause is normalized from visible text before the independent V5 answer parser runs. This version has no full-token next-word oracle and is not a claim of natural-language realism.

For the same RNG seed, 300 V5 and V8 documents have exactly equal latent worlds and `(rule family, answer)` sequences. The V8 audit checks 300 documents at each of two additional seeds. All 1,417 answer sites pass visible-prefix recovery; all 1,417 upstream edits change the intended answer and are recovered from the edited visible text. At both seeds, changing syntax alters all 300 documents while leaving answer sequences fixed. The optional supersession clause changes 245/300 and 216/300 documents respectively; it is inserted only when the same object–attribute key has an earlier report. This checks semantic preservation, not whether a trained model learns the rule.

Two untouched FineWeb sample-10BT offsets provide 64 first-100-word windows each. V5 and V8 windows use the *same* sampled worlds and must both be 100-word eligible. The V8-no-context ablation holds V8 clause choices but removes supersession clauses.

| Offset | Panel | Word types | Top-ten share | First-40 repeated-word fraction |
| --- | --- | ---: | ---: | ---: |
| 36000 | V5 | 225 | .353 | .300 |
| 36000 | V8, no context | 237 | .333 | .248 |
| 36000 | V8 | 240 | .331 | .248 |
| 36000 | FineWeb | 2259 | .218 | .196 |
| 37000 | V5 | 221 | .334 | .295 |
| 37000 | V8, no context | 224 | .333 | .266 |
| 37000 | V8 | 226 | .332 | .266 |
| 37000 | FineWeb | 2415 | .214 | .163 |

The syntactic clause changes help repetition; the supersession clause contributes very little to these first-window form metrics. V8 still repeats a tiny set of entities, facts, and sentence purposes. One generated document begins: “At the shelf, Theo had a red badge, known as Pine. … Emma said: Maya painted the Cedar blue. … The update came from Leo: Theo painted the badge blue.” The first-window lexical diversity remains roughly one tenth of FineWeb's, and the prose is easy to identify as generated. **V8 does not pass the realism gate.**

The result also clarifies what to build next. More synonyms, asides, or clause templates alone are unlikely to make the documents language-like. The next generator needs a probabilistic *document plan* in which facts serve several discourse purposes—describing, contrasting, explaining, attributing, correcting, summarizing, and querying—with variable entity recurrence and compositional syntax. Those choices should be independently parameterized from task-rule probabilities. A visible-only parser and supported counterfactual edits remain required for each rule. Form audits should include lexical diversity, repetition, sentence/layout frequencies, and qualitative review, not a single scalar score.

No V8 model was trained. Training and weight interpretation should follow only after a broader document-plan generator is checked, with a behavioral gate for each learned rule. V8 remains a paired structural control for isolating the effect of fact-clause variation on future models.

Reproduce with [`audit_document_world_v8_discourse.py`](audit_document_world_v8_discourse.py) and [`compare_document_world_v8_fineweb_form.py`](compare_document_world_v8_fineweb_form.py). Aggregate JSON panels are in `results/document_world_v8_discourse_audit_300x2.json`, `results/document_world_v8_fineweb_skip36000_n64.json`, and `results/document_world_v8_fineweb_skip37000_n64.json`. No FineWeb source passages are saved.
