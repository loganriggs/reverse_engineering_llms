# Corpus construction inventory for the next sampled-language renderer

The [V2 proposition renderer](DOCUMENT_PROPOSITION_SURFACE_V2.md) has
correct visible-state answers and causal edits, but its repeated clause
frames fail the natural-form screen. This audit asks a narrower design
question: which lexical constructions actually appear around attribution,
change, comparison, and quantities in an independent FineWeb panel, and how
do the existing synthetic clauses differ? It does **not** label natural
sentences with ground-truth semantic tasks.

The [auditor](audit_discourse_constructions.py) scanned 2,000 FineWeb
`sample-10BT` documents at offsets 90,000–91,999. It compared their first
100 alphabetic words with 2,000 independently sampled, eleven-move threaded
V2 proposition documents using typed lexical realization. The panels have
similar mean length: 96.8 versus 93.5 words per document. A second natural
screen used the first 500 words from the same documents to check that the
dominant forms were not peculiar to the opening. Only aggregate counts were
saved; no source prose is committed. All patterns were fixed before the
first run and are listed in the script.

| First ~100 words, 2,000 documents | FineWeb | V2 |
| --- | ---: | ---: |
| Screened sentences under eight words | 31.8% | 48.2% |
| Documents with attribution pattern | 299 | 827 |
| Attribution-pattern sentences / 1,000 words | 2.05 | 8.55 |
| Documents with change pattern | 183 | 1,119 |
| Change-pattern sentences / 1,000 words | 1.07 | 14.67 |
| Documents with comparison pattern | 273 | 0 |
| Documents with quantity pattern | 125 | 0 |

The natural attribution screen most often sees `said` (188 sentences),
then `announced` (55) and `according to` (52). V2 has no `said`; it uses
`reported` (1,591), `according to` (498), and `reported by` (490). Natural
comparison-like wording includes `while` (171) and `both` (112), whereas
V2's comparison sentences say that attribute “values were same/different.”
Natural quantity-like wording includes `more than` (70), `number of` (47),
and `there were` (12); V2 favors `file records`, `file lists`, and `Count N`.
The [500-word natural screen](results/discourse_constructions_fineweb90000_n2000.json)
also ranks `said`, `while`, `both`, `more than`, and `number of` highly, so
the direction is not confined to the first sentence or two.

The absence of screened V2 comparison and quantity patterns is a **surface
coverage** finding, not an absence of those rules. Its sampled semantic
plans contain comparison moves in 1,032 documents and measure moves in
1,261 documents. This is exactly why the next renderer should change how
already-defined propositions are said, not merely add more task labels.
Attribution should express the source and the proposition together instead
of “reported update 1”; comparisons should express a relationship between
the two entities; counts should appear in ordinary quantitative sentences.
The separate long tail of lexical choices and document planning still need
work. Replacing a few phrases will not by itself close the earlier repeated
trigram gap.

These regexes are intentionally broad: `while` can be temporal, `now` may
not denote a state update, and `said` does not prove a source-binding task.
Sentence splitting is approximate, and the corpus and synthetic samples
have different topics and genre mixtures. The numbers are surface
diagnostics, **not** natural task priors or a license to reduce task exposure
until models cannot learn. Keep task prevalence and surface-form entropy as
independent DGP controls. A revised renderer must still pass visible-text
semantic replay and supported edits, then be compared on lexical diversity,
repeated trigrams, layout, and actual sample coherence before training
interpretability models.

Aggregate records: [length-matched panel](results/discourse_constructions_fineweb90000_n2000_w100.json)
and [longer natural panel](results/discourse_constructions_fineweb90000_n2000.json).

## Controlled V3 renderer test: lexical alignment is insufficient

I added an experimental `surface_version=3` to the existing proposition
renderer. It keeps the same semantic plans and rule sites, but has sources
state the changed proposition (using `said` or `according to`), expresses
comparisons as `match/mismatch`, and uses `there were` or `the number of`
for recorded quantities. A visible-text parser checks those clauses without
reading the plan. The [semantic audit](results/document_proposition_surface_v3_semantic_form_audit.json)
passed 2,800 rendered documents, 9,432 original scored prefixes, 678
targeted update flips, 504 source flips, 1,070 comparison flips, 392
measure flips, and 626 deliberately false comparison rejections. The
original V2 regression audit reproduced its prior semantic totals.

On the same length-matched [V3 construction panel](results/discourse_constructions_v3_fineweb90000_n2000_w100.json),
the short-sentence share falls from 48.2% to 37.4%, closer to FineWeb's
31.8%. But V3 overshoots the screened construction rates: attribution
9.36 versus 2.05, change 16.47 versus 1.07, comparison 4.31 versus 1.56,
and quantity 10.84 versus 0.74 sentences per 1,000 words. The parser still
passes because semantic correctness and realistic occurrence rates are
different tests. On the earlier two 64-document first-100-word form panels,
V3's typed cross-document repeated-trigram shares are .460 and .459,
compared with V2's .449 and .451. A separate [fresh FineWeb reference](results/document_proposition_surface_v3_fresh_form_reference.json)
at offset 92,000 has .00717 on 64 documents cropped to 100 words; the
earlier FineWeb development references were .0067 and .0104. The altered
clauses **do not improve the main form gap**.
Manual samples still read as unrelated listings and numbered updates.

**Decision:** retain V3 only as a negative surface control; do not train
the next interpretability-model comparison on it. The next generator must
change document purpose, event-link structure, and amount of varied
supporting prose, while retaining enough causal sites for learning. A
natural-looking trigger word is not itself evidence of realistic language
or a useful ground-truth rule. Audit task-site prevalence separately from
surface construction frequency so the renderer cannot hide an unlearnable
task behind better aggregate statistics.
