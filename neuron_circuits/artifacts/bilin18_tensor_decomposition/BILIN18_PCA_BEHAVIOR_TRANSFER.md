## H991 — Training-only PCA preserves finite intervention behavior much better than the sparse pilots

Previous turn was progress: H990 separated learned decoder coverage, selected support and encoding errors, and found a strong train-only PCA activation baseline. This turn tests that baseline in the model so Euclidean improvement does not stand in for behavioral fidelity.

Use the exact H989 six-original-family non-discovery context pairs plus all old copying/spatial task-holdout pairs.312pairs/624directed edits, including native-failing diagnostic cohorts. Fit centered PCA on original48training slots or expanded144training slots only. Freeze bases before interventions. Compare32components and128requested components: small-data PCA128 has numerical rank40, expanded PCA128 has128. Include allfour expanded6 pair-change-trained sparse models (block/shared, two seeds); no selection or retraining. The predicted difference is obtained by separately projecting donor and recipient writes, then added to the native recipient MLP12 output. Reference is full donor-write replacement; identity is no edit.

Metric KL(full donor edit||candidate edit) / KL(full donor edit||native), pooled across directed edges within each family/context/split and macro-averaged across native-passing cohorts. Context aggregate uses50cohorts, task-holdout aggregate fourcohorts. Mean reference KL0.02383 and0.01592 respectively; all absolute KL and signed margin effects are saved. Zero is perfect intervention-distribution fidelity, one is no change; this is not percent whole-model computation or task accuracy.

| Predictor | New-context KL ratio | Whole-task holdout KL ratio |
|---|---:|---:|
| PCA32, original small6 |0.190|0.912|
| PCA40 (requested128), original small6 |0.185|0.881|
| PCA32, expanded6 |0.162|0.920|
| PCA128, expanded6 |0.091|0.792|
| Block pair-change, expanded6, two-seed mean |0.703|1.057|
| Shared pair-change, expanded6, two-seed mean |0.738|1.065|
| Identity |1.000|1.000|

PCA's lower activation error corresponds to a large behavioral-fidelity advantage on new contexts, so the previous sparse-pilot failures cannot be attributed entirely to an uninformative Euclidean metric. Broader training contexts improve PCA32 modestly on familiar-family context transfer. The larger PCA128 model improves more, but the small-versus-expanded maximum-rank comparison changes both available rank and training data. Do not attribute the full0.185→0.091 gain solely to sample count. Fixed-rank32 whole-task transfer does not improve; even expanded128 leaves most whole-task discrepancy.

PCA is a dense linear representation with no recovered semantic groups, no unsupervised task labels and no evidence of unique computational atoms. It receives the actual MLP write as input, like the sparse reconstruction models; it is not a compact predictor that bypasses the original MLP. This establishes a behaviorally useful representation baseline and a target for structured featurizers, not completion of the bootstrap discovery objective. Data and prompts remain reused development material; fresh prospective validation is still required before broad claims.

Audits:6,240/6,240 full-model forward examples,624references plus624identity and4,992candidate edits. Identity logits replay within2.48e-5. Difference-of-projections matches direct linear projection within1.53e-12. Independent summary aggregation exact. Forty-eight overlapping whole-task reference edits replay H985 within7.49e-7 KL and9.54e-6 task-margin change. All input/model hashes unchanged, finite outputs, saved frozen PCA bases, no sparse-model training or reserved-test access. Natural-document pool26,112.

Next controlled structural comparison: preserve the verified PCA decoder span and exact projection coordinates, then impose scalar/block/shared support choices at matched actual coordinate budgets. This separates sparsity/grouping cost from the previous poorly learned span and encoder. Include dense PCA32/PCA128, compare explicit support inference and decoder initialization before another unconstrained training sweep, and retain finite-intervention fidelity as the acceptance criterion. Only groups that preserve behavior and show stable selective responses should be candidates for unlabeled task discovery and retraining. Whole-task transfer remains weak; the broad research goal stays open. H961 context-bridge predictor evaluation remains pending.
