# Where the sports generator still repeats itself

The corpus-derived player-stat predicates improved surface diversity, but the
complete sports document remains synthetic. This audit isolates the source of
that repetition before changing the document generator again. It samples the
same 128 training-partition worlds per source for fixed and empirical stat
forms. The score sentence, answer sentences, semantic rules, and placement
controls are held fixed. It counts a trigram occurrence as repeated when the
same trigram appears in another sampled document **within the same sentence
kind**. This full-document, category-local measure is not numerically
identical to the earlier first-55-word form screen.

| Source and sentence kind | Sentences | Fixed repeated trigram share | Empirical repeated trigram share |
| --- | ---: | ---: | ---: |
| RotoWire score | 128 | .690 | .690 |
| RotoWire player facts | 640 | .800 | .593 |
| RotoWire answer recaps | 384 | .776 | .786 |
| SBNation score | 128 | .848 | .848 |
| SBNation player facts | 640 | .798 | .579 |
| SBNation answer recaps | 384 | .792 | .786 |

All 1,152 sentences in each 128-document panel are one of these three kinds:
one score, five player-stat clauses, and three explicit answer recaps per
document. There are **zero** other discourse acts. This is a stronger and more
direct diagnosis than simply observing a high top-ten word share. The new
player predicate bank does its local job, but cannot change the rigid
score→stat→answer document plan. The three answer recaps remain repetitive,
and the score opener is especially repetitive for SBNation-derived forms.

The next generator should sample which supported facts to present and why,
with genuinely different acts such as event sequence, comparison, attribution,
and explanation. Each act needs a typed semantic contract and visible-text
checker. It should not merely add more synonyms or shuffle the current nine
sentences. The acceptance screen should compare full documents and the
task-bearing answer windows against held-out human reports, and separately
check task coverage. Only rules the model then learns behaviorally and under
selective edits qualify for weight-circuit interpretation.

Reproduce with `audit_sports_form_bottleneck.py`; the aggregate counts are in
`results/sports_form_bottleneck_n128.json`. The script saves no source prose.
