# Can final-score arithmetic share a natural report with other rules?

The previous three-rule model exposed a weak surname task and a capitalization target dominated by `The/the`. A final-score margin could provide a different, deterministic computation: the visible score `high–low` determines a later winning-margin token. `sports_report_carrier_dgp.py` already supplies a conservative score-semantic screen and a human-report carrier with potentially contradictory score sentences removed. I intersected those eligible games with a bracket pair and a full-name-to-later-surname pair **inside the retained carrier**.

| Carrier target words | Split | Score eligible | Score + bracket | Score + alias | All three |
|---:|---|---:|---:|---:|---:|
| 50 | train | 823 | 205 | 36 | 9 |
| 50 | heldout | 204 | 52 | 8 | 3 |
| 100 | train | 554 | 185 | 92 | 34 |
| 100 | heldout | 123 | 39 | 23 | 6 |
| 150 | train | 387 | 171 | 95 | 43 |
| 150 | heldout | 89 | 35 | 19 | 6 |
| 200 | train | 276 | 137 | 84 | 41 |
| 200 | heldout | 58 | 27 | 18 | 8 |

These are unique-game opportunity counts, not BPE/context alignment counts or successful learned tasks. The full score+alias+bracket intersection is too small for a convincing model comparison: at 100 words it has only six heldout games, and varying carrier length does not fix this. A score+bracket two-rule pilot is more plausible, with 205/52 games at 50 words or 185/39 at 100. That pilot would need to audit the score edit's semantics, BPE alignment, full and rule-window form, then behaviorally gate both rules before comparing weight methods. The score query is still an inserted natural-style sentence; it is not evidence that a corpus-derived text distribution naturally asks this arithmetic question at the required rate.

This screen is a course correction, not a rejection of the broader goal. The task suite needs multiple competent computations and realistic varied text, so the next generator should either combine score+bracket in this corpus-anchored pilot or broaden the source/discourse process enough to support a true three-rule sample without recycling a tiny intersection.

Reproduce with `audit_score_rule_overlap.py`; aggregate counts are in `results/score_rule_overlap_by_carrier_words.json`. No source prose is saved.
