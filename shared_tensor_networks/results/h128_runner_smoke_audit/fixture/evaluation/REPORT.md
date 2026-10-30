# d4/h2 coverage results

SMOKE FIXTURE ONLY — not scientific scaling evidence.

Transfer MSE uses whole-task holdouts and the fixed primary adaptation budget. Medians are across teacher means; random starts are averaged within teacher. Full recovery requires every matched feature cosine to exceed.99. Finite-budget outcomes are not convergence certificates.

| Inputs | Tasks | Start | Transfer MSE | Mean features, teacher median | Full recovery |
|---:|---:|---|---:|---:|---:|
|128|1|random|0.794783|0|0/2|
|128|1|teacher|0.000391816|2|1/1|
|128|3|random|0.792993|0|0/2|
|128|3|teacher|0.000370964|2|1/1|
|512|1|random|0.789578|0|0/2|
|512|1|teacher|0.000265221|2|1/1|
|512|3|random|0.789135|0|0/2|
|512|3|teacher|0.00048822|2|1/1|

No equal-label comparison: the tiny smoke grid does not have matching label budgets.
8/8random-start endpoints have lower-loss same-condition teacher-start witnesses. This does not certify strict local minima.
Total training time across independent model runs: 0.000hours, excluding preparation and analysis.

Audit scope: input/checkpoint integrity, data regeneration, literal empirical predictions, Gaussian feature matching, exact population metrics through the shared verified Wick kernel, and independent SVD transfer solves. It does not replay every optimizer step. This panel does not establish depth3–5 recovery or a general scaling law.
