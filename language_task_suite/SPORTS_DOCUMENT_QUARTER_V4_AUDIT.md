# Quarter-trajectory V4: five linked rules pass; language form still fails

The [discourse and quarter audit](SPORTS_DISCOURSE_ACT_AND_QUARTER_AUDIT.md) found that about 94% of source rows have four quarter scores reconstructing the final result, and halftime reversals are common. This versioned [V4 arm](sports_document_quarter_v4.py) tests whether a trajectory can be added to the existing winner, margin, and top-player rules with clear, selective causal semantics. It is a **semantic fixture**, not an accepted realistic training distribution.

The sampler chooses an actual, sum-consistent four-quarter score vector from the same source style and game partition, restricts to 100–130 winning points, a 1–25 final margin, and a non-tied halftime margin of at most eight. It attaches that trajectory to a synthetic game with a guarded score sentence and five sampled player records. It inserts four visible quarter rows into the report and adds **halftime leader** and **whether the eventual winner trailed at halftime** as two new scored sites. Any donor score vector must reconstruct the sampled final totals. The score-template weekday is changed to the sampled base game's true weekday; the visible-only score parser treats weekday as a surface slot. The V4 arm also corrects singular “1 assist/rebound” text without changing legacy V3 outputs.

The [audit](audit_sports_document_quarter_v4.py) sampled 100 plans in each RotoWire/SBNation × train/held-out game group. Every one of the **400 original documents and 1,600 edited documents** passed all five site spans and visible-prefix answers (10,000 site checks); every visible quarter table summed back to the final score. Four edits tested the dependency graph:

| Edited fact | Answers that must change | Answers held fixed |
| --- | --- | --- |
| Swap team names | winner; halftime leader | margin, top player, comeback relation |
| Increase winning score by one in Q4 | margin | winner, top player, halftime leader, comeback |
| Move quarter points across halftime without changing finals | halftime leader; comeback | winner, margin, top player |
| Make another listed player the highest scorer | top player | winner, margin, halftime leader, comeback |

The observed comeback-answer shares ranged from 31% to 42% across the four arms. Both source and quarter-donor game keys stay in the requested group; no game crosses the training/held-out source partition in this audit. These checks establish the **local marked rules**, not a full next-token oracle for the text.

The form screen rejects V4 as the main language distribution. On 64 paired worlds per arm, cropped to the first 55 words, repeated-cross-document trigram share moves from V3-like base to V4 as follows:

| Source/group | Matched base | V4 quarter arm | Halftime answer window | Comeback answer window |
| --- | ---: | ---: | ---: | ---: |
| RotoWire train | 61.0% | 52.8% | 73.0% | 70.9% |
| RotoWire held-out | 60.3% | 52.7% | 71.9% | 71.1% |
| SBNation train | 63.1% | 59.2% | 73.8% | 70.9% |
| SBNation held-out | 64.8% | 57.3% | 71.9% | 72.7% |

The lower *whole-crop* repetition is partly mechanical: the numeric quarter table displaces repeated player and recap prose from the fixed crop. The rule-bearing windows are more repetitive, not more natural. A sampled example still reads as result → Q1–Q4 table → player clauses → recaps, with only a few final question forms. Natural RotoWire's previously measured 64-document 55-word panel repeated about 16% of trigrams; panels are not content-matched, but V4's synthetic phrasing gap is clear.

**Decision:** retain V4 as a ground-truth multi-rule and intervention fixture. Do not spend the main training or weight-method sweep on it or claim realism from the crop improvement. The next DGP change must alter the mention/discourse process and the language of the **task-bearing spans**—for example, sampling several evidence-bearing quarter events and temporal comparisons within a report rather than placing one rigid score table before a recap block. Re-run the paired form and visible-rule audits before model training. The [aggregate audit](results/sports_document_quarter_v4_semantic_form_n100.json) includes per-arm samples and full window statistics; raw source reports stay local.
