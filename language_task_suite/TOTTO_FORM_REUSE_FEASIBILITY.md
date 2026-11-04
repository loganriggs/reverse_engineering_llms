# Corpus-attested realization reuse: a ToTTo feasibility screen

The current language-DGP bottleneck is varied prose that remains true to a sampled semantic world. This screen tests a different resource from finite clause templates or unconstrained LM paraphrases: human-corrected ToTTo descriptions paired with visible table rows. It does **not** approve a generated dataset or retrain a model.

The [audit](audit_totto_form_reuse.py) reads all 120,761 official training records in the local archive. It applies the already corrected strict extractor: a complete aligned header row, one highlighted data row, at least two distinct highlighted cell values appearing exactly once in the human sentence, and independent source-row checks. This yields 21,165 examples. It replaces only these verified value spans with typed header slots, then measures whether a form could even be considered for another page with the same ordered headers. The exported [aggregate](results/totto_form_reuse_screen.json) contains no source prose.

| Measure | Count |
| --- | ---: |
| Distinct ordered header schemas | 6,728 |
| Schemas on at least two pages | 1,044 |
| Strict examples with another page of the same schema | 13,558 |
| Distinct delexicalized sentence forms | 20,219 |
| Exact forms used on multiple pages | 108 |
| Forms retaining the source page's final name word outside value slots | 12,305 |
| Forms retaining a number outside value slots | 5,616 |

The first three numbers show enough cross-page schema recurrence to make a *controlled* realization-bank experiment feasible. The last two expose why straightforward slot substitution is invalid. A manual sample of cross-page swaps included a filmography sentence that retained the original actor's surname after replacing its Year and Role values, and another that retained the original performer and film title. A title-word match is only a coarse leakage heuristic: it can fire on a common last word and miss aliases or unsupported facts. The number count likewise does not prove an unbound number is wrong. These are risk screens, not validated semantic-error rates.

This rules out promoting same-header value swapping as the next training DGP. The useful next test is more stringent: derive a typed relation with **all** entities and quantities bound to the source row or page, replace those bindings jointly, then use a separate semantic parser and targeted human review on source-disjoint pages to reject unsupported assertions. If enough valid realizations remain, sample document-level event/discourse plans and draw from that verified bank; measure full-document and task-window form against independent natural panels before training. If coverage collapses, move to a corpus with richer explicit meaning/text annotation instead of relaxing the semantic gate. The earlier depth-2 ToTTo model's failed year-value intervention remains a separate behavioral negative result; this screen does not change its eligibility for weight interpretation.
