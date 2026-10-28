## H571 — Parenthesis-trained adjustment also transfers to comma-delimited binding

Pair parentheses and commas around the exact same four phrase contents from H570. Assert matched total length, source token/position and final token/position for every delimiter pair. Use both binding templates, four name pairs and two endpoints. For copying, change the continuation to “and then spoke/wrote to” so a closing comma does not immediately precede a period. Preserve frozen H568 query edits and all predictor/intervention modes. Predictions are locked before edited-prompt evaluation. There are 288 panel prompts and 2,880 interventions.

| Family/head | Delimiter | Phase-only coefficient MAE | Same-task pooled MAE |
|---|---|---:|---:|
|binding_old/L8H3|paren|0.01027|0.01301|
|binding_old/L8H3|comma|0.01006|0.01273|
|binding_old/L14H4|paren|0.23090|0.07114|
|binding_old/L14H4|comma|0.25336|0.04497|
|binding_new/L8H3|paren|0.01158|0.01855|
|binding_new/L8H3|comma|0.01854|0.01947|
|binding_new/L14H4|paren|0.20253|0.06505|
|binding_new/L14H4|comma|0.26806|0.03893|
|copying_old/L8H3|paren|0.01220|0.00883|
|copying_old/L8H3|comma|0.01053|0.01750|
|copying_old/L14H4|paren|0.06855|0.08632|
|copying_old/L14H4|comma|0.03969|0.05008|
|copying_new/L8H3|paren|0.02310|0.01174|
|copying_new/L8H3|comma|0.01201|0.00943|
|copying_new/L14H4|paren|0.07550|0.09171|
|copying_new/L14H4|comma|0.04160|0.04864|

The strongest binding result survives and improves under comma delimiting. L14H4's pooled correction reduces error by about 82% in the old binding template and 85% in the new one, versus 69% and 68% with parentheses. A detector specific to parentheses cannot alone explain this transfer. A broader insertion/clause-structure response remains possible. L8H3 binding still does not show the same general benefit.

The copying construction exposes a competence limitation. Candidate accuracy is 100%, but base full-vocabulary correctness is only 3/8 in each copying template; old-template “as she explained” with parentheses is also 3/8. These cells fail the >=50% full-vocabulary gate. Non-name top tokens across copying failures include “the”, “her”, and “me”; some may be plausible continuations, so failure of this evaluation gate is not necessarily incorrect language behavior. We retain all results but do not treat the entire copying panel as established name-repetition competence.

Binding cells pass the gate (candidate 7–8/8, full-vocabulary 5–8/8). Delimiter changes can alter clause interpretation despite matching positions and words, so this is a punctuation-format test, not proof that all semantics are fixed. Query source keys remain numerically unchanged because edits occur after the selected source.

Audits: source-key max error 3.06e−5, self-margin max error 1.82e−5; matched lengths/positions verified; donor edits, predictions, manifest and source unchanged; finite outputs. A source-string quoting error was fixed before any inference; no model outcomes informed that fix. Next H572 tests copying with “yesterday” before the sentence boundary, retaining grammatical comma forms and the original repetition structure. H571 failures remain separately recorded.
