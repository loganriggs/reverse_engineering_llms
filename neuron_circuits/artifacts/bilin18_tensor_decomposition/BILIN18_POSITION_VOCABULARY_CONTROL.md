## H554 — Vocabulary and context both influence downstream copying responses

Cross the eight binding name pairs into the same prose-code and record-list copying templates used for colors in H553. Keep L8H3, source-selection rule, virtual phase positions and all four clamp modes fixed. The 80-prompt panel contains 32 newly constructed name-copying prompts and 48 reused color/binding anchors. All 32 name/color comparisons within copying formats match source position, total length and final token/position; vocabulary and resulting context states change. Binding versus copying still differs in syntax, distance and source role, so this does not isolate an abstract task variable.

Name copying is competent: 16/16 candidate and full-vocabulary correct in each format. Binding is 16/16 candidate and 14/16 full-vocabulary. Color prose is 16/16 for both; color record is 15/16 candidate, 13/16 full-vocabulary, with 7/8 complete pairs correct. No cases are filtered on outcome. There are 400 locked frozen-readout predictions and 1,600 interventions.

| Family | Clamp | Direction agreement | Prediction MAE |
|---|---|---:|---:|
|binding|live|100.0% (64)|0.01200|
|binding|single_8|100.0% (64)|0.01042|
|binding|freeze_mlp|100.0% (64)|0.00589|
|binding|freeze_both|100.0% (64)|0.00000|
|color_prose|live|0.0% (64)|0.14114|
|color_prose|single_8|6.2% (64)|0.08268|
|color_prose|freeze_mlp|100.0% (64)|0.00955|
|color_prose|freeze_both|100.0% (64)|0.00001|
|color_record|live|0.0% (64)|0.20256|
|color_record|single_8|6.2% (64)|0.13733|
|color_record|freeze_mlp|100.0% (64)|0.02225|
|color_record|freeze_both|100.0% (64)|0.00000|
|name_prose|live|51.6% (64)|0.08183|
|name_prose|single_8|93.8% (64)|0.05004|
|name_prose|freeze_mlp|100.0% (64)|0.01017|
|name_prose|freeze_both|100.0% (64)|0.00000|
|name_record|live|39.1% (64)|0.22958|
|name_record|single_8|96.9% (64)|0.13463|
|name_record|freeze_mlp|100.0% (64)|0.02901|
|name_record|freeze_both|100.0% (64)|0.00000|

Name-copying live directions agree with the frozen prediction on 33/64 prose and 25/64 record comparisons (58/128 combined), versus 0/128 for color copying and 64/64 for name binding. Thus neither a pure vocabulary explanation nor a vocabulary-independent copying reversal fits these observations. Holding L8's MLP write fixed restores 60/64 and 62/64 name-copying directions, substantially more than the 4/64 per format seen for colors. All downstream MLP clamps restore every direction in each family. The same intervention has token- and context-dependent downstream consequences.

This is a response-level result, not a semantic feature identification. It does not imply all names share a universal operation, or establish why colors behave differently. The head's bus write and the contextual downstream states both change with vocabulary. Paired content-transfer interventions would be needed to distinguish those two sources of dependence.

Audits: native-position replay within 2.77e−5, fully frozen analytic replay within 1.82e−5; source, manifest and predictions unchanged; all finite. Native cache is saved for follow-ups. No new FineWeb documents, tracked total 26,112. Broad goal active. Next H555 tests positive/negative amplitudes approaching zero to separate locally persistent response reversals from large-excursion artifacts.
