# Form at the rule site remains the main generator bottleneck

The earlier first-100-word audit found a large V14/FineWeb form gap. The [`rule-window audit`](audit_rule_window_form.py) now measures 41-word contexts centered on actual synthetic prediction sites. It samples at most one site of each family per independent 256-token stream, avoiding repeated windows from one document. A frozen 480-row, 512-GPT-2-token FineWeb panel supplies randomly located 41-word windows, sampled separately at the same count for each comparison. Its raw text stays local; the [aggregate JSON](results/rule_window_form_v14_v2_n128.json) stores a SHA-256 of the frozen token panel.

| Rule window | Synthetic windows per seed | Repeated cross-document trigram share, seed 0 / 1 | Matched-count FineWeb, seed 0 / 1 |
|---|---:|---:|---:|
| V14 mixed verb | 128 / 128 | .484 / .468 | .007 / .012 |
| V14 mixed object | 128 / 128 | .503 / .486 | .010 / .014 |
| V14 attribute object | 96 / 82 | .488 / .461 | .005 / .005 |
| V14 bracket closer | 63 / 62 | .268 / .293 | .007 / .002 |
| V14 source answer | 23 / 16 | .447 / .386 | .002 / .003 |
| V14 conditional action | 53 / 38 | .380 / .306 | .006 / .001 |
| V2 mixed verb | 128 / 128 | .780 / .783 | .011 / .018 |
| V2 mixed object | 128 / 128 | .798 / .791 | .010 / .011 |

The lexical head also differs at the high-density V14 sites: mixed-verb top-ten word share is .373/.372 versus .217/.215 in the matched FineWeb windows; only 323/320 word types appear versus 2,071/1,969. The source-answer comparison has only 23/16 independent windows and should be read as a qualitative signal, not a precise population estimate. No natural window is known to require the matched synthetic rule. The FineWeb panel consists of token windows, not complete documents, so this result specifically compares local language form. Regex sentence boundaries and word tokens are shared across arms but are not linguistic parses.

This shows that improving only document openings would not solve the benchmark problem. The most useful rule-bearing contexts in V14 are themselves highly templated. V2's larger lexicon and five ways to phrase one event do not solve that; they worsen the site-context metric. Further isolated event-frame additions stop here. The next generator should broaden *what is being said* and how acts interact in a document, using typed semantics and independently checked realizations. Candidate task windows must pass the same audit before a small bilinear model is trained for weight-method comparison. Behavioral and causal rule gates still decide which learned rules are valid interpretability targets.
