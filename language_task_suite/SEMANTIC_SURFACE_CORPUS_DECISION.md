# Grounded prose resources: ToTTo slot closure and DART source screen

The aim remains a sampled, varied, rule-based document distribution for bilinear-model interpretability. The prior [ToTTo form-reuse screen](TOTTO_FORM_REUSE_FEASIBILITY.md) found many recurring table schemas, but simple value swapping left source-specific names. We tightened that test before treating any human sentence as a reusable realization.

The [slot-closure auditor](audit_totto_slot_closure.py) draws a deterministic reservoir of 3,000 from the 21,165 strict training examples, binds unique visible row values plus full/last page-title references, and uses English NER and a number scanner to flag unbound entities and numbers. It found 1,940 sampled examples whose ordered header schema occurs on another page; 264 of those had no *detected* unbound entity or number, representing 220 distinct delexicalized forms across 119 schemas. The full-train extraction count reproduced the prior 21,165. These are upper-bound candidates, not verified interchangeable forms: NER can miss references, and a sentence's predicate may depend on facts absent from the row.

A manual cross-page substitution check confirms that limitation even in the apparently closed subset. A source sentence about playing a role in an *episode* could be transferred to a row about a multi-year television series; another labeled the target as a *film*; another asserted a *lead character* or *recurring role* without support from the target cells. The names and dates were bound, yet the relation and discourse claim changed. **Stop naive ToTTo same-schema substitution as a training generator.** Its human sentences remain useful form references and table-to-text behavioral controls, but a valid sampler needs an explicit relation ontology and truth-conditional validation beyond slot closure.

I then screened the official [DART data-to-text corpus](https://github.com/Yale-LILY/dart), which supplies human text paired with semantic triple sets. The local v1.1.1 train JSON contains 30,526 triple sets and 62,659 annotations. The [aggregate auditor](audit_dart_semantic_bank.py) sampled 512 annotations per source pool (fixed seed) and measured literal endpoint visibility and cross-document repeated-trigram occurrence share:

| Source pool | Train annotations | Triple predicates | All triple endpoints literal | Repeated trigrams, 512 sentences |
| --- | ---: | ---: | ---: | ---: |
| WikiTableQuestions MTurk | 2,047 | 1,363 | .420 | .077 |
| WikiTableQuestions Lily | 4,700 | 1,883 | .334 | .079 |
| WikiSQL declarative | 3,608 | 1,934 | .320 | .152 |
| WikiSQL Lily | 720 | 472 | .294 | .230 |
| WebNLG | 18,102 | 241 | .334 | .462 |
| E2E | 33,482 | 7 | .309 | .660 |

The WikiTableQuestions pools offer more varied single-sentence form than E2E/WebNLG on this diagnostic and retain many typed predicates. Literal endpoint visibility is deliberately strict; paraphrases, changed morphology, and inferred/omitted details can lower it. It is **not** an estimate of annotation correctness. Manual inspection found both useful multi-relation descriptions and noisy or incomplete examples. These source pools are not realistic documents or a full next-token symbolic oracle. The next DGP step should start with a small, manually reviewed relation subset from the lower-repetition pools, specify which triples each candidate sentence actually expresses, and build a typed plan that samples multiple linked facts and independently checks each realization. Evaluate whole-document and rule-window form against fresh natural panels and a true predictive-context benefit before model training. If the candidate bank cannot support multiple compatible discourse acts, do not merely concatenate unrelated annotations or relax the truth gate.

Aggregates: [ToTTo closure](results/totto_slot_closure_n3000.json) and [DART source screen](results/dart_semantic_bank_v111_train.json). The official corpus files are kept locally outside the Git repository. No new model or circuit claim follows from these screens.
