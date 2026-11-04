# Rule-site form gap by task family

The [scored-clause composition pilots](SCORED_CLAUSE_COMPOSITION_AND_RELATIVE_RESULT.md) failed to close the rule-window form gap. Before building a new clause grammar, this audit asks which rules account for that gap and whether explicit “Question? Answer:” formatting is the main cause.

The [aggregate-only script](audit_proposition_site_form_by_family.py) independently sampled 512 threaded proposition V3 plans for each of two development seeds. It renders typed lexical surfaces and checks every scored answer against the visible-prefix parser. For each family in each document, it selects at most one 41-word window centered on a marked answer. Natural reference windows come from different FineWeb source rows in the same local preview range, sampled at random positions. They are **not matched by task, topic, or syntax**. Cross-document repeated-trigram share is a finite-panel statistic and depends on the number of windows; compare each family only with its size-matched natural reference, not with other families as if rates were directly comparable.

| Rule-site family | Synthetic windows, 130k / 140k | Repeated-trigram share, synthetic | Size-matched natural random windows | Question-form windows, 130k / 140k |
|---|---:|---:|---:|---:|
| Comparison | 225 / 228 | .605 / .625 | .017 / .014 | 0 / 0 |
| Fact recall | 201 / 200 | .567 / .562 | .015 / .012 | 76 / 74 |
| Numbered step | 69 / 68 | .537 / .548 | .009 / .002 | 0 / 0 |
| Source attribution | 50 / 38 | .400 / .389 | .008 / .000 | 17 / 9 |
| Update recall | 71 / 52 | .391 / .343 | .005 / .002 | 20 / 13 |
| Measure recall | 29 / 28 | .363 / .329 | .002 / .002 | 10 / 10 |

The small source, measure, and record subgroups are descriptive, not precise estimates. The stable comparison/fact/step result is the actionable one. Comparison and numbered-step sites have no `Answer:` wording at all, yet remain very repetitive. Within fact recall, both question and prose forms repeat heavily; their window counts differ, so the raw within-family fractions are not a controlled effect of question wording. The current two comparison realizations reuse phrases such as “the color of” and a small set of binary relation words, while numbered steps use an even smaller action/ordinal vocabulary. The [earlier dependency audit](SEMANTIC_SYNTAX_OPPORTUNITY_AUDIT.md) found the same structural constraint at the document level.

This changes the next implementation priority. Build a typed clause AST for **comparison and fact/update recall first**, while preserving source, step, measure, and record as jointly sampled rules. A single semantic relation should admit multiple predicate-centered constructions, participant orders, referring expressions, clause embeddings, and truth-checked negation. Sampling those choices must be part of the DGP, and the visible-text checker must recover the relation and answer without access to the sampled derivation. Merely changing document openings, dropping explicit questions, or adding modifiers around the existing fixed comparison phrase will leave the dominant repeated trigrams. Validate both complete-document form and these family-specific task windows on a new natural panel before bilinear training and weight-method evaluation.

The exact counts, including all site-family frequencies and question/prose splits, are in [130k aggregate](results/proposition_site_form_by_family_fineweb130000_n512.json) and [140k aggregate](results/proposition_site_form_by_family_fineweb140000_n512.json). No source or generated passages are committed.
