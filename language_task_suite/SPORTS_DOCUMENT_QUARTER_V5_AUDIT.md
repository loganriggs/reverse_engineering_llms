# V5 quarter-event prose: spreading six acts does not close the form gap

The [V4 quarter fixture](SPORTS_DOCUMENT_QUARTER_V4_AUDIT.md) passed five linked semantic rules but rendered one rigid four-row table and two final recaps. V5 tests a narrow alternative: express the four period scores through six typed prose forms, replace the yes/no comeback answer with the more natural **lead/deficit** state of the eventual winner, and distribute Q1, Q2, halftime answer, Q3, Q4, and winner-state answer among the existing report sentences. It preserves the V4 sampled world and edit operations. This is a controlled renderer test, **not** a corpus-fitted discourse model.

[`sports_document_quarter_v5.py`](sports_document_quarter_v5.py) keeps the base score and player sentences, samples quarter-sentence forms, and chooses either one common insertion slot (`act_spread=0`) or six ordered slots (`act_spread=1`). The latter respects the visibility constraint: the halftime answer follows Q1–Q2; the winner-state answer follows Q1–Q4. A text-only parser recognizes the quarter forms and computes each answer without the hidden plan. Separate random streams make the fixed-versus-spread arms use **the same worlds, sentences, answer forms, and paragraph choices**; only placement changes. On every sampled spread-world, the sentence multiset matches its fixed counterpart and the text order changes.

The [audit](audit_sports_document_quarter_v5.py) sampled 100 worlds in each RotoWire/SBNation × train/held-out × placement arm: **800 original and 3,200 edited documents**, each with five visible-answer checks (20,000 site checks). It verified quarter totals, weekday consistency, grammatical singulars, and four intervention patterns. Team-name swaps change winner and halftime-leader names; a Q4 +1 score edit changes margin; a cross-half score transfer flips halftime leader and winner's lead/deficit state without changing final totals; a player-point edit changes only top player. All checks passed. Source and quarter donor game keys remain in the requested game partition.

The form result is negative. On 64 matched worlds per arm, first-55-word repeated-cross-document trigram share is:

| Source/group | V4 reference | V5 fixed | V5 spread | Winner-state window fixed → spread |
| --- | ---: | ---: | ---: | ---: |
| RotoWire train | 52.2% | 60.4% | 64.9% | 68.5% → 66.9% |
| RotoWire held-out | 54.1% | 58.4% | 60.9% | 67.6% → 65.8% |
| SBNation train | 62.6% | 65.0% | 69.8% | 69.9% → 68.8% |
| SBNation held-out | 59.1% | 64.2% | 68.4% | 68.4% → 69.6% |

Replacing the table with prose and spreading the same acts generally **increases whole-crop repetition**. It sometimes reduces the new answer-window repetition by one or two points, but the window remains near 66–70%; SBNation held-out moves the other way. The V4 reference is matched to each arm's underlying world, while fixed and spread also share worlds with each other. The fixed 55-word crop is a partial document view; the task windows prevent treating a favorable crop as proof of linguistic realism.

One generated spread example begins with a result, inserts first- and second-quarter evidence around two player-stat sentences, gives the halftime leader, then interleaves the remaining periods with the original winner, margin, and top-player recaps. The ordering is more coherent than a single table block, but six period forms and seven state forms still repeat conspicuously, and the base player/recap grammar is unchanged. The [aggregate](results/sports_document_quarter_v5_semantic_form_n100.json) includes generated examples and all rule-window measures; no raw corpus reports are committed.

**Decision:** retain V5 as a versioned five-rule behavior and placement fixture. Do not use it as the main realistic training distribution or spend another turn adding a handful of sports paraphrases. This controlled failure says that shuffled typed sentences are not enough. The next generator should make discourse acts and evidence selection genuinely stochastic at document scale, with a much broader, semantically checked realization source or an induced probabilistic grammar; test complete documents and each scored window against held-out natural text before model training. Only behaviorally learned rules should enter the later weight/canonicalization/sparsity comparison.
