## H564 — Expanded discovery spans lower the oracle floor, but do not resolve the mechanism

Use the same 40 H561 evaluation prompts at +1/16 amplitude. For each discovery coverage level, allow all centered context directions: 39, 79 and 159. Solve for response-optimal coefficients given the correct target response. This is an oracle representational diagnostic, not a learned predictor or downstream behavioral evaluation. Both data coverage and available span dimension increase here; H563 separately held rank fixed.

| Family | Oracle error, 40 contexts | Oracle error, 80 contexts | Oracle error, 160 contexts |
|---|---:|---:|---:|
|binding|16.0%|13.1%|10.4%|
|color_prose|46.7%|34.9%|23.3%|
|color_record|54.2%|40.3%|28.4%|
|name_prose|31.6%|24.5%|17.4%|
|name_record|44.9%|35.4%|25.5%|

The copying oracle floor drops from 32–54% to 17–28% mean relative vector error; binding drops from 16% to 10%. Ordinary full-span projection also improves. This supports the diagnosis that the original context set omitted response-relevant directions. It does not establish the intrinsic dimension, show that still more data will reach a desired fidelity, or validate an oracle as an approximation usable without the target.

The compactness branch now has a bounded conclusion: within this intervention class, a simple context mean/PCA account preserves some behavioral statistics while missing substantial internal computation; metric reweighting within the old span did little, and broader coverage helps. No unique feature basis or canonical computational unit has been identified. We should avoid spending further rounds on rank sweeps without testing whether the local account predicts natural changes in behavior.

Next H565 returns to the outstanding physical-position question. Use matched copying and binding examples with a fixed intervention head, varying actual source–destination separation while checking native competence. Compare the actual gate change with a position-only prediction from the original query/key content, then separate remaining query/key-context contributions. This tests whether the virtual phase mechanism predicts natural prompt changes. Keep template, vocabulary and padding confounds explicit; do not claim that a physical edit changes position alone. Include fresh lexical pairs and no per-example fitting.

Audits: all 120 oracle fits no worse than their corresponding full-span projection; relative normal-equation residual <=6.14e−16; all finite and source unchanged. These are algebraic diagnostics, not live causal interventions. H563–H564 total 640 causal interventions, 120 algebraic diagnostics and 80 new constructed prompts. No new FineWeb documents; tracked total 26,112. Broad research goal active and incomplete.
