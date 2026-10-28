## H561 — Prospective names and colors confirm limited metric benefit

Freeze H560 bases, metrics and all rank budgets. Construct four new name pairs (Roger/Janet, Donald/Martha, Kenneth/Paula, Raymond/Teresa) and four token-valid color pairs (bronze/copper, cream/navy, charcoal/golden, peach/salmon). Tokens were checked before model inference. Use unchanged prose-copying, record-copying and binding templates. All 40 prompts survive; no exclusions or refitting. This expands the previously inadequate fresh-color panel.

Every family has four pairs correct at both candidate endpoints. Full-vocabulary correctness is 8/8 for color prose and both name-copying formats, 7/8 for color record, and 6/8 for binding. All pass the existing competence gate. Fresh means constructed prompt instances, not guaranteed absence of vocabulary from training or earlier unrelated experiments.

| Family, +1 | PCA rank 32: response error / KL ratio | Weighted rank 32: response error / KL ratio |
|---|---|---|
|binding|18.0% / 0.034|18.0% / 0.034|
|color_prose|59.6% / 0.322|59.6% / 0.323|
|color_record|68.1% / 0.348|68.1% / 0.348|
|name_prose|41.6% / 0.318|41.6% / 0.316|
|name_record|60.2% / 0.665|60.2% / 0.664|

The new panel confirms the small difference between metrics: weighted and ordinary projections remain nearly identical, with copying response errors around 42–68% at rank 32. Binding remains easier internally and behaviorally. Absolute KL and margin data are retained in JSON; KL ratios are to the local zero-response baseline, not the entire model's computation. Exact recipient displacement, quadratic term and downstream model remain available to both methods, so neither is a compact replacement for the full circuit.

Audits: no refitting; source/basis hashes unchanged; exact replay max margin error 1.24e−5 and max KL 2.78e−11; all finite. Saved new native inputs for diagnostic follow-up. 880 interventions and 40 new constructed prompts. No new FineWeb documents, tracked total 26,112. Broad goal active.

Next H562: use all 39 training directions and solve for oracle coordinates given the target response. This deliberately leaks the answer to measure a representational lower bound; it is not a predictor. If even this oracle fails, coefficient fitting within this span cannot remove the remaining error.
