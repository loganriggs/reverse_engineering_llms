# M27: response supervision has a clear training effect but only uneven transfer beyond KL-only

2026-10-28. Construct eight full-data KL-only controls from the saved M24 gradients: half scheduled natural K plus half scheduled task K. Contrast with the existing eight joint K+lambdaR aggregate candidates. Same starts, data weighting, Euclidean gradient construction, two fixed raw displacement norms and decoder normalization. No fresh gradients, optimizer tuning or candidate selection. This closes the objective-coverage/preconditioning mismatch of the earlier natural-minibatch comparison.

## What adding response supervision buys

Counts of cases with lower error for joint than matched KL-only. Eight native cases are correlated model/region comparisons; counts are descriptive, not significance tests.

| Displacement | Training R /4 | Native margin /8 | Native log-probability /8 | Equal-family margin /8 | Joint lexical/context margin /8 |
|---|---:|---:|---:|---:|---:|
|1.0|4|4|5|6|7|
|0.1|4|4|6|5|5|

All eight training comparisons favor response supervision. Full-step training response errors:

| Start | Full-step KL-only R | Full-step joint R | Tenth-step KL-only R | Tenth-step joint R |
|---|---:|---:|---:|---:|
|scalar71|0.003977|0.002115|0.004288|0.003886|
|scalar72|0.002055|0.001358|0.002300|0.002054|
|block71|0.002791|0.001354|0.002478|0.002265|
|block72|0.007621|0.003791|0.007871|0.006894|

The matched control therefore does not explain away the training-response gain. Transfer remains uneven: at either displacement only4/8 native margin comparisons favor joint supervision; log-probability response favors it5/8 at full size and6/8 at tenth size. This is more limited than the7/8 and8/8 improvements versus the original checkpoints in M26, because KL-only updates also help. Neither comparison proves a reusable semantic decomposition or a generally reliable multi-step learning rule.

## Ordinary prediction costs

Lower KL means closer ordinary predictions to the native model. Full-step comparisons:

| Start | Full-step natural KL: control → joint | Full-step joint lexical/context KL: control → joint |
|---|---:|---:|
|scalar71|0.117705 → 0.118550|0.074501 → 0.073694|
|scalar72|0.114128 → 0.113949|0.055460 → 0.055284|
|block71|0.146934 → 0.147388|0.084582 → 0.084492|
|block72|0.155973 → 0.155345|0.068579 → 0.079576|

One material regression is block72 at full size: joint lexical/context KL is .079576 versus .068579 for KL-only. Conversely, joint supervision wins7/8 full-size margin comparisons in the small jointly withheld causal panel. These differing outcomes prevent a single overall fidelity verdict.

Full JSON includes both displacement sizes, all three nontraining lexical/template quadrants, unprefixed lexical evaluation, accuracy and local reconstruction metrics. Training gains, output fidelity and causal response fidelity remain separate outcomes. All data are familiar development sources; the jointly withheld native panel contains only six pairs. Whole-task exclusions remain intact. No new-task discovery or manifold validity claim follows.

## Information-gathering next step

Stop optimizer and displacement comparisons. The current transfer test changes both inputs and perturbation directions: training uses selected M18 groups, while the existing native benchmark uses M11 directions on other inputs. Next make one bounded same-input, unseen-direction probe to locate a failure before another training run. For each scheduled training endpoint/region, choose one deterministic group never supervised on that input for that starting featurizer, using its original M18 dictionary. Exclude every group supervised on that input across both regions; retain inactive groups, duplicates and all task families. Do not select by native effect or competence after inspection. Freeze the rule before collecting outcomes. Evaluate all joint/KL-only candidates at both norms and initial models, with identical deltas applied to native and reconstructed writes. Compare against existing supervised-direction scores on the same inputs, with raw errors, native effect energies, zero counts and family-balanced summaries. If transfer fails already on these same inputs, direction coverage is a bottleneck; success would shift attention toward context/input or basis transfer without proving either alone. This diagnostic feeds the featurizer loop and is not a new optimizer search.

## Validation and runtime history

All eight new parameter states independently reconstructed in NumPy (maximum element discrepancy9.69e-8); raw displacement norms match M25 exactly. Post-normalization norm differences versus joint candidates are below .008%, and are reported rather than silently rematched. Gradient caches and source checkpoints are hash-verified. Four initial checkpoints reused from M26.

Native benchmark:3004forwards,2112response records,704identical directions;72 sampled NumPy metrics/decodes and all336 summaries pass. Training-target scores:1646forwards,960records,16 sampled objective replays and88 summaries pass. Output inventory:14976forwards,13824records; independent decoder/output audit, full inventories/aggregates and initial-output replay pass. Audits do not independently reimplement the transformer.

The first training-score audit exited139; the unchanged audit passed with BLAS/OMP threads1. The first output run exited139 inside native CastedLinear F.linear. Retry used Torch/BLAS threads1, fault tracing and atomic128-input progress checkpoints. Initial-output replay verifies numerical compatibility of the completed run. Causes of the runtime failures remain undetermined; partial runs are not used as scientific results.

Output audit covers 84 samples; maximum decoder relative error 2.09e-07, metric error 5.42e-08, near-tie support rows 0.

[Protocol](validation/aggregate_kl_protocol.json), [parameter audit](validation/aggregate_kl_models_audit.json), [native audit](validation/aggregate_kl_native_fidelity_audit.json), [training audit](validation/aggregate_kl_training_fit_audit.json), [output audit](validation/aggregate_kl_outputs_audit.json), [all paired comparisons](validation/aggregate_kl_comparison.json), [runtime history](validation/aggregate_kl_run_history.json). Large tensors remain local in `artifacts/bilin18_tensor_decomposition/emlp_m27_*`, hash-linked from the results.
