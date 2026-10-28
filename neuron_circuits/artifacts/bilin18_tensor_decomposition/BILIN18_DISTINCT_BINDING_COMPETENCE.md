## H808 — Competing-answer retrieval panels fail competence gates

The repeated-value panels used in H797–H807 cannot establish key selection: when both stored values are identical, the answer does not depend on which key is queried. H808 tests a stricter condition before interpreting cross-task interventions. Three families (car color, person location, giver/recipient) each have prose and structured surfaces, sixteen lexical pairs, two value assignments, both fact orders, and both queried entities. Every prompt contains two different answers. Expected labels follow the entity/value assignment.

The gates were written before model execution: at least eight complete aligned items, candidate accuracy at least .75 separately for queries to each displayed fact, full-vocabulary accuracy at least .50, and at least half of matched query pairs both correct. No correctness-based item removal is allowed. The future mechanism stays L2H5 first-source value perturbation at second source, then L6 query MLP response crossing.

| Family | Surface | Items | Candidate accuracy | Full-vocabulary accuracy | Both query choices correct |
|---|---|---:|---:|---:|---:|
| Color | Prose | 16 | .492 | .031 | .016 |
| Color | Structured | 8 | .453 | .000 | .031 |
| Location | Prose | 16 | .477 | .023 | .016 |
| Location | Structured | 8 | .453 | .266 | .000 |
| Recipient | Prose | 16 | .547 | .523 | .141 |
| Recipient | Structured | 8 | .656 | .594 | .344 |

All six fail. Positional preferences are strong: prose color candidate accuracy is .906 when the answer is in the first fact but .078 when it is in the second; prose location is .156/.797. Recipient completion has higher full-vocabulary accuracy but still fails the paired discrimination test. Structured surfaces have identical key/value syntax across families; they are not three independent computational operations.

This is evidence of a task-competence limitation, not evidence that the frozen internal mechanism fails on a task the model solves. No causal intervention was run on these failed panels. The result also narrows interpretation of prior high repeated-value accuracy: it supports repeated-content completion, not demonstrated arbitrary entity binding.

576 forwards completed. Prose retains all sixteen items per family. Structured surfaces exclude items 5,8,9,10,11,12,14,15 due to paired-query length/position alignment, leaving eight. All exclusions are saved. Hashes unchanged, finite margins, and first-source mixed-value invariance across future query changes exactly zero. Reserved test untouched; document pool 26,112.

Next test a separately specified standard indirect-object completion, motivated by earlier task evidence rather than outcome-based filtering of these failed families. Retain both initial name orders and both giver choices, use the same competence gates, and keep the selected intervention head/layer fixed. The broader distinct-task transfer claim remains unproven.
