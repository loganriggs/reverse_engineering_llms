# Full sports reports as a document and task-language source

The [structured-sentence screen](STRUCTURED_REALISATION_BANK_AUDIT.md)
found too little discourse in DART/ToTTo mosaics. The next source screen
tests *full*, human-written reports with structured facts. The
[Harvard box-score release](https://nlp.seas.harvard.edu/boxscore-data/)
aligns RotoWire and SBNation basketball summaries to game and player
statistics. Its authors explicitly warn that games cross their supplied
train/validation/test splits and recommend the improved
[SportSett:Basketball resource](https://github.com/nlgcat/sport_sett_basketball).
Raw archives remain local, outside Git.

The [document audit](audit_rotowire_document_source.py) read both official
archives. On separate random panels of 64 reports cropped to 100 words:

| Source | Train reports | Mean report tokens | Word types | Top-ten word share | Repeated-cross-document trigram share | Reports with exact final-score pair | Reports with a player-name/points surface match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RotoWire | 3,398 | 334 | 1,245 | 25.7% | 18.5% | 2,548/3,398 | 3,141/3,398 |
| SBNation | 7,633 | 807 | 1,611 | 25.4% | 6.7% | 3,236/7,633 | 3,739/7,633 |
| Independent broad FineWeb reference | — | — | 2,397 | 21.3% | 0.73% | — | — |

The broad FineWeb panel is **not** a sports-genre-matched target. Still,
both report corpora are far more varied than the linked synthetic report
DGP's 86.9–95% repeated trigrams. RotoWire is more tightly aligned with
explicit statistics; SBNation is longer and more variable but often
contains context absent from a box score. Exact numeric/name matches are
only surface opportunities, not validated entailments. Results are in
[RotoWire](results/rotowire_document_source_audit.json) and
[SBNation](results/sbnation_document_source_audit.json) aggregate JSON.

## The split and cross-style opportunity

The supplied splits are unsuitable for a strict generalization test.
Grouping by date and both team names, RotoWire has 235 game keys shared
between train/validation and 222 between train/test; SBNation has 538
and 508. The [paired-style audit](audit_boxscore_paired_styles.py) found
2,944 date/team keys in both corpora. Of these, 2,746 have at least one
pair with the *same* home/away final scores; 198 have score conflicts and
must not be treated as same-fact paraphrases. Among score-consistent pairs,
1,404 express the exact final-score pair in both summaries. First-100-word
type Jaccard is 0.157 for matched games versus 0.104 after shuffling games.
This is an unusually useful natural style-transfer control: two reports
can discuss the same verified result with substantially different wording.
But the reports are not exact paraphrases and may discuss different events.
All records for a date/team game key must be assigned to the same split
across both corpora, with scores checked for any paired comparison. The
[aggregate paired result](results/boxscore_paired_styles_audit.json) saves
no source prose.

## Reversible score-sentence extraction is not semantic validation

The [extractor](corpus_score_sentence_renderer.py) keeps only 8–42-token
sentences with both full team names exactly once, the winning score first
in an exact score pair, no other number, and no unrelated capitalized name.
It replaces teams and scores with typed slots and asserts that reinserting
the original values reconstructs the source sentence exactly. On the
supplied training splits this yields 102 distinct RotoWire templates and
214 SBNation templates. A [held-out-game rendering audit](audit_corpus_score_templates.py)
draws new teams/scores from 1,735 game keys absent from *both* corpora's
training splits and checks 128 substitutions per source. All numeric edits
change the intended score token and leave the other tokens intact.

That test establishes slot mechanics only. Manual review of generated
sentences revealed phrases such as “first game of the season,” “in overtime,”
and “16th straight home game,” which are not licensed by a plan containing
only teams and final scores. Some evaluative language contradicted the
sampled score margin. RotoWire has 102 and SBNation 214 *candidate* forms,
not 316 safe forms. The aggregate result and caveat are in
[the template audit](results/corpus_score_sentence_templates_audit.json).

A first [semantic guard](score_template_guard.py) checks winner/loser verb
order, host/visitor roles, margin-dependent adjectives, and excludes many
claims not represented in the sampled score plan. At an 8-point margin it
keeps 51 RotoWire templates and only 3 SBNation templates. On 128 novel-game
sentence draws per source, guarded cross-sentence repeated-trigram shares
are 81.4% and 89.8%. Manual red teaming still found that the guard needed
to reject source-specific rivalry and season-context phrases. It is a
candidate filter, **not** a certified faithful or variable renderer; a
strict score-only world collapses much of the corpus's useful variation.

## Decision for the sampled DGP

Use these corpora to learn a **document-plan prior**: result, prominent
players, game trajectory, and outlook can occur in different orders and
styles. Use score sentences only under semantic guards. The first candidate
guard requires a supported win/loss verb with the correct team role,
conditions some evaluative words on the score margin, and rejects
streak/overtime claims until the sampled world contains those variables.
It needs a separate manual validation sample before training. The same
treatment can later add player-stat clauses. This turns unsupported
modifiers into explicit candidate rules to model, instead of silently
accepting them as prose.

The next concrete generator experiment is a sports-document world with
sampled team scores, home/away roles, a few player records and a variable
act sequence. Fit act/style frequencies on one group-split corpus portion;
keep other games for independent natural-form and factual-alignment
checks. Vary style mix, score-margin dependence, player-stat mention rate,
and discourse order as DGP hyperparameters. Include the existing
synthetic linked-report rules in the broader document mixture so this
does not become a single sports-task detour. Before weight interpretation,
verify each learned rule by held-out behavior and supported edits. The
current work is a source audit and renderer component; no new model has
been trained or weight method validated on it.
