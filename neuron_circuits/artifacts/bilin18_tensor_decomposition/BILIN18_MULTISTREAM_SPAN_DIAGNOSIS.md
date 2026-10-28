## H1032 — Coordinate refitting helps little; learned decoder spans remain weak

Previous turn H1031 completed six multi-stream sparse fits. This turn freezes their parameters and tests exact coordinate refitting on their selected supports, full decoder-span projection, and training-only PCA16/PCA128 per stream. Same38historically opened development endpoints and same downstream interventions. PCA uses identical discovery token/family weights and frozen centers/scales. No rank sweep or validation fitting. PCA16 uses48total coordinates/token; PCA128 and full learned spans use up to384, and are capacity diagnostics, not budget-matched sparse competitors.

| Representation | Mean whole-output KL (family macro) | Paired relative KL (family macro) |
|---|---:|---:|
| PCA16 per stream |0.920|0.889|
| PCA128 per stream |0.413|0.261|
| Scalar selected-support LS |0.883|0.765|
| Scalar full decoder span |0.681|0.534|
| Block selected-support LS |1.376|0.835|
| Block full decoder span |0.843|0.576|
| Shared selected-support LS |1.599|0.834|
| Shared full decoder span |0.887|0.575|

Two seeds averaged for learned models, all five families retained including capital's limited competence/length-filtered diagnostic cohort. Original scalar paired macro error was approximately0.773, so exact coefficients on unchanged support yield only a small gain. Full spans help more. PCA128 substantially outperforms the full learned spans, so poor dictionary learning/coverage is a real issue under the fixed small-data/400-step setup, alongside sparsity. Exact LS minimizes Euclidean error, not downstream KL; it does not rule out a better output-sensitive encoder. Active-support LS uses only permitted input activations and frozen gates, so it is an implementable encoder alternative, not a privileged downstream-output oracle. Full-span reconstruction is an increased-budget diagnostic.

PCA128 paired relative KL by task:agreement0.013,pronoun0.011,sentiment0.088,capital0.529,binding0.662. Larger geometry supports most grammatical/affective task edits but still loses much lexical content on these development prompts. PCA16 has better mean normalized state reconstruction than scalar but worse paired macro fidelity, reiterating that reconstruction is not sufficient. No manifold, new-task-discovery or whole-model-understanding percentage follows.

Audits: all six checkpoints unchanged; active-support LS never increases squared reconstruction error; current native-logit replay exactly0;1068suffix sequences; zero original task-holdout reads. Fourteen saved-logit samples independently reproduce whole and paired KL within1e-12. PCA bases and all scalar measurements saved. No process remains active; natural-document pool26112unchanged.

Next: use the train-only PCA128 geometry as a stronger reference for a finite-effect-supervised representation, retaining matched-budget scalar and group controls; resolve lexical-content coverage with explicit port-preserving controls rather than adding opaque group gates. This remains an outstanding experiment, not an achieved improvement. User then asked where results live and about repository publication; inspect storage and repository destinations before publishing. Broad goal remains active.
