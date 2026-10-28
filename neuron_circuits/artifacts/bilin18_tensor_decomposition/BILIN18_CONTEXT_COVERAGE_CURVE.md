## H563 — Broader context coverage improves fixed-rank copying reconstruction

Keep H561's 40 evaluation prompts and rank budgets 16/32 fixed. Fit ordinary context PCA on three nested discovery sets: original 40 contexts; all 80 original H554 contexts; then 160 after adding 80 newly constructed contexts. The new contexts use eight additional name pairs and eight common-word code pairs in the existing templates. They expand both sample count and lexical diversity; this is not an iid sample-size scaling law. All discovery candidate tokens are disjoint from evaluation candidate tokens. Evaluation prompts remain excluded from fitting, although their earlier results are already known.

New discovery competence: every family is candidate-correct on all 16 endpoints; full-vocabulary correctness is 16/16 for both name and common-word copying formats, 15/16 for binding. No outcome-based selection. Common-word copying is intentionally broader than the original color category. This could change which contexts PCA prioritizes, rather than simply reducing sampling noise.

Use the same local operator replacement as H558, preserving exact recipient displacement, quadratic response and later computation. At each of two amplitudes, compare six coverage/rank settings, exact and zero-response controls: 640 interventions.

| Family, rank 32 and +1 | 40 contexts: response error / KL ratio | 80 contexts | 160 contexts |
|---|---|---|---|
|binding|18.0% / 0.034|17.4% / 0.034|17.9% / 0.035|
|color_prose|59.6% / 0.322|51.8% / 0.257|48.6% / 0.245|
|color_record|68.1% / 0.348|57.5% / 0.279|56.4% / 0.288|
|name_prose|41.6% / 0.318|37.7% / 0.247|37.3% / 0.238|
|name_record|60.2% / 0.665|54.6% / 0.479|53.8% / 0.461|

At fixed rank 32, broader discovery coverage reduces copying response error from 42–68% to 37–56%, with substantial remaining error. Output KL generally improves, but is not monotone in every cell: color-record KL slightly worsens from 80 to 160 contexts despite lower internal error. Binding changes little. Thus additional coverage helps more than the tested metric reweighting, without establishing a faithful compact representation or a universal fidelity tradeoff.

KL ratios are relative to omitting L8's response while retaining the attention perturbation, not to the entire model. This is a fixed-panel diagnostic coverage curve; new generalization claims still require prospective evaluation. Rank and coverage are independently controlled here, unlike the following full-span oracle diagnostic.

Audits: fit/evaluation IDs and candidate tokens disjoint; source and saved bases unchanged; exact-response replay max margin error 1.24e−5 and max KL 2.78e−11; finite KL. 80 newly constructed prompts, no new FineWeb documents; tracked total 26,112. Broad goal active. Next H564 checks whether the larger full spans lower the attainable local-response error, using target-informed oracle coordinates only as a diagnostic.
