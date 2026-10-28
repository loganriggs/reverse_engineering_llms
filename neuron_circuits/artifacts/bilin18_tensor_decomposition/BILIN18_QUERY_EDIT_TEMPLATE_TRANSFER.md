## H570 — Generic query adjustment transfers selectively across phrases and templates

Freeze H568 discovery query means. New phrases are “(of course)” and “(for example)” (four tokens), plus “(as I recall)” and “(as she explained)” (five tokens). None has a matched fitted mean. Cross these with the original templates and changed templates: binding uses a museum visit followed by handing a book; copying uses repeated “She wrote to” instead of “He spoke to.” Keep the H568 evaluation name pairs. Predict using pooled means from the same task and length, same task across lengths, or other task at matched length. Predictions are saved from base prompts before evaluating edited prompts; no refitting.

The panel has 160 prompts: original/new templates × two tasks × four name pairs × two endpoints × base/four edits. The 128 phrase-edit prompts are new combinations; 16 original-template base prompts reuse H568, while 16 new-template bases are added. This is a prospective phrase/template test with previously used evaluation names, not a fully independent lexical test. All candidate endpoints are correct. Full-vocabulary correctness is 8/8 for copying and 5–8/8 for binding across cells; every cell passes the existing competence gate.

| Task/template/head | Phase-only coefficient MAE | Same-task, length pooled | Same-task, all pooled | Other-task, length pooled |
|---|---:|---:|---:|---:|
|binding_old/L8H3|0.01027|0.01255|0.01301|0.01526|
|binding_old/L14H4|0.23090|0.07114|0.07114|0.21192|
|binding_new/L8H3|0.01158|0.01833|0.01855|0.01969|
|binding_new/L14H4|0.20253|0.06516|0.06505|0.18115|
|copying_old/L8H3|0.01445|0.01232|0.01323|0.01355|
|copying_old/L14H4|0.02018|0.02440|0.02002|0.10874|
|copying_new/L8H3|0.07339|0.04027|0.03905|0.06742|
|copying_new/L14H4|0.03008|0.03534|0.03534|0.09937|

At L14H4 binding, the pooled same-task correction reduces error by about 69% in the old template and 68% in the new one. Length-specific and all-length pooling are almost identical there. This supports a reusable contextual adjustment broader than the exact discovery phrases. Cross-task pooling loses most of the benefit.

The result is not universal. L8H3 binding gets worse under the pooled corrections. Copying L8H3 improves modestly in the old template and substantially in the new one; L14H4 copying does not consistently improve. These counterexamples limit the claim to particular reader/context regimes. Knowing that the first query factor carries a correction does not establish that its correction is constant or transferable everywhere.

All models still receive the recipient's base query, source key and edited rotary position. The learned object is only a pooled raw-query offset, and interventions change one source-edge coefficient while the rest of the edited model stays native. This is not a replacement model or a prediction of the entire sentence-edit effect.

A remaining confound is now central: every discovery edit is a parenthetical insertion. The transferable adjustment may represent punctuation, clause structure or interruption, rather than the phrases' meanings or an abstract task. Next H571 should compare parenthetical and comma-delimited forms with matched tokens/positions where possible, checking competence and using frozen predictors. Include grammatical copying controls rather than inserting an extra comma before a period. Separate punctuation-format transfer from semantic substitution, and retain phase-only and cross-task baselines. This targets what the reusable query adjustment represents.

Audits: fixed query-edit file and predictions unchanged; source keys match within 3.06e−5; self replay <=1.82e−5; equal-length groups verified; source/manifest unchanged and all finite. H569–H570 total 3,136 interventions. H570's 160-prompt panel adds 144 prompt combinations relative to H568. No new FineWeb documents; tracked total 26,112. Broad goal active and incomplete.
