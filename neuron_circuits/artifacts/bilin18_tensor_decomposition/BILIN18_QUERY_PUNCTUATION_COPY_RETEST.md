## H572 — Competent copying reveals a counterexample to task-specific query-edit labels

Follow H571's copying gate failures with a predeclared grammar change: “He spoke to X. He spoke to X [insertion] yesterday. He spoke to” and its “She wrote to” counterpart. Both comma and parenthesis insertions remain grammatical, and matched delimiter pairs retain identical source/final positions. Keep the frozen discovery means, name pairs, phrases and intervention modes. Binding cells replay H571 anchors. This is a separate follow-up, not replacement of unfavorable H571 results.

All copying candidate endpoints are correct. Full-vocabulary correctness is 8/8 in every copying cell except old-template “as I recall” with parentheses (7/8). Thus every copying cell now passes the competence gate. Binding competence remains as in H571.

| Copying template/head | Delimiter | Phase-only coefficient MAE | Copying pool | Binding pool |
|---|---|---:|---:|---:|
|copying_old/L8H3|paren|0.01809|0.01310|0.03723|
|copying_old/L8H3|comma|0.01700|0.01141|0.03512|
|copying_old/L14H4|paren|0.06848|0.07413|0.01570|
|copying_old/L14H4|comma|0.02494|0.03134|0.05649|
|copying_new/L8H3|paren|0.05857|0.04337|0.06965|
|copying_new/L8H3|comma|0.04517|0.02997|0.05626|
|copying_new/L14H4|paren|0.05101|0.06217|0.04066|
|copying_new/L14H4|comma|0.02878|0.02759|0.09181|

Copying L8H3 generally benefits from its own pooled correction in both punctuation formats. At L14H4, however, the task-label account breaks: in the old copying template with parentheses, a binding-derived pooled correction reduces coefficient MAE from 0.06848 to 0.01570 (about 77%), while the copying-derived correction worsens it to 0.07413. With commas, binding-derived correction instead worsens error. The new copying template shows a weaker but similar contrast: binding-derived correction helps parentheses modestly and harms commas strongly.

This is a concrete counterexample to interpreting the learned offsets as pure “binding” and “copying” features. The same competent task changes which offset helps when punctuation/continuation structure changes. The result supports reader-specific structural/contextual adjustment, but does not yet identify its source or semantic content. We should trace how the query is constructed rather than naming its direction from task labels.

Next H573: capture final-position attention and MLP writes before L14, reconstruct the base-to-edited pre-attention residual difference using the model's residual multipliers, and separately account for input normalization when mapping to the query. Then test selected upstream write clamps with live downstream computation. Keep binding and the copying parenthesis/comma counterexample, and distinguish descriptive write decomposition from causal mediation. This seeks the construction of the reusable adjustment, not further rank tuning.

Audits: source-key error <=3.06e−5; self replay <=2.01e−5; equal-length delimiter pairs verified; source, manifest, predictions and frozen edit file unchanged; all finite. H571–H572 complete 5,760 interventions. Each has 288 panel prompts; H572 repeats 144 binding anchors and adds 144 revised copying prompts. Relative to H570, the two experiments add 352 prompt combinations. No new FineWeb documents, tracked total 26,112. Broad goal active and incomplete.
