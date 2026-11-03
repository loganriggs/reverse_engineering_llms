# Existing document-renderer controls do not close the form gap

The user-facing research target is a family of variable, language-like *sampled* distributions with known rules. We need to test each trained model behaviorally on the rules it actually learned before asking whether weight-based methods recover them. To decide whether the current document generator can be tuned into that role, this sweep changes one V5 renderer control at a time while holding its sampled latent worlds fixed.

`sweep_document_world_v5_form_controls.py` draws 300 documents at each of two seeds. Within a seed, all settings use exactly the same event ledgers, source assignments, aliases, recap plans, and answer targets; only the realization changes. Every answer site is checked against the visible-prefix transducer. The form screen uses the first 100 alphabetic words of each eligible document and measures distinct word types, top-ten word share, and repeated-word fraction in the first 40 words. Eligibility counts are reported because some settings make documents shorter. No generated document bodies are saved.

| Setting | 100-word eligible, seed 7723 / 8823 | Types | Top-ten share | First-40 repeat |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 262 / 267 | 272 / 268 | .334 / .342 | .283 / .283 |
| Syntax entropy 0 | 260 / 272 | 264 / 263 | .346 / .353 | .364 / .366 |
| Syntax entropy 1 | 261 / 266 | 271 / 269 | .334 / .337 | .275 / .267 |
| Aside rate 0 | 145 / 164 | 119 / 119 | .360 / .366 | .292 / .303 |
| Aside rate 1 | 300 / 300 | 270 / 270 | .359 / .362 | .290 / .280 |
| Header rate 0 | 258 / 262 | 274 / 265 | .337 / .343 | .296 / .295 |
| Header rate 1 | 263 / 266 | 277 / 272 | .331 / .339 | .269 / .267 |
| Paragraph rate 0 | 262 / 267 | 272 / 268 | .334 / .342 | .283 / .283 |
| Paragraph rate 1 | 262 / 267 | 272 / 268 | .334 / .342 | .283 / .283 |

The full [aggregate JSON](results/document_world_v5_form_controls_300x2.json) also includes intermediate settings and newline, question, colon, quote, parenthesis, and digit incidence. The baseline parameters are syntax entropy .7, aside rate .3, paragraph rate .22, header rate .5, and delayed source rate 1. The four controls are meaningful: syntax entropy notably changes repeated-word fraction, header and paragraph rates change layout, and asides change document length. None creates enough lexical or discourse variety. A previous *jointly eligible* V5/FineWeb screen found V5 top-ten share .349 versus FineWeb .229; that is a different panel and should not be read as a paired estimate for the sweep.

This is a structural limit of the current renderer. A high aside rate adds more copies of a few incidental frames and dilutes answer-site density; it does not create diverse ways to express facts. Header and paragraph changes mainly alter typography. Even maximal syntax entropy selects among only three introduction forms and four event/recap forms. The next generator should therefore sample **document acts and fact-bearing clause structures**, not just their surface rates: e.g. introduce a fact, update it, attribute it, contrast sources, refer back, summarize it, or ask about it, with different orders and optional dependent clauses. The acts must retain independently checkable visible-text answers and supported counterfactual edits. Lexical frequency and genre should be separate knobs, so later model and weight-method comparisons can vary them without silently changing the underlying rule.

No model was trained on this sweep. It selects the next DGP change; it is not evidence that any model learned or represents a rule.
