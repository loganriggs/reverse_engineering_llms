# Can the score-copy query come from the report itself?

The current score×bracket benchmark inserts a score lead and a later answer sentence around natural RotoWire prose. Those insertions raise first-100-word cross-document repeated trigrams from .182 to .212 on a matched train-game panel and from .089 to .125 heldout; the score-answer windows are still more repetitive. I screened the corpus for naturally occurring later mentions of the winning score that might replace the inserted query.

Among reports with a conservatively validated final-result lead, the upper-bound opportunity counts are:

| Split | Valid result leads | Later exact winning-score token | Also lacks losing-score token | Winning score followed closely by “points” | Also has a bracket somewhere |
|---|---:|---:|---:|---:|---:|
| Train | 1,506 | 51 | 42 | 29 | 27 |
| Heldout | 395 | 12 | 12 | 9 | 9 |

These are **not accepted task sites**. Manual inspection of 16 candidates found several different meanings: a team scored the total, an opponent was *allowed* the total, or another quantity was described as a part of the team's total. A simple numeric match would mislabel some reports, and a +1 counterfactual would need to update every dependent claim. The 29/9 train/heldout counts are therefore generous upper bounds; requiring the bracket and semantically valid score reference leaves fewer. This corpus slice cannot supply enough unmodified natural score-copy queries for the multi-rule model by itself.

The useful next step is to broaden source genres and document structures or build a semantically typed renderer with a larger, validated bank of score-reference constructions. Adding a few more query paraphrases to the same fixed report suffix would not address the underlying support limit. Keep the current corpus-anchored benchmark for controlled weight-method tests, with its form limitation explicit.

Reproduce with `audit_natural_score_copy_mentions.py`. Only aggregate counts are saved in `results/natural_score_copy_mentions_screen.json`; source excerpts are printed locally for review and are not committed.
