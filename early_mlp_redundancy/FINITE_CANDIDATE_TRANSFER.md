# M26: one-step joint improvements transfer unevenly across causal metrics

2026-10-28. Evaluate all16 prespecified live-support M25 candidates plus four initials. Four starts (scalar/block, seeds71/72), aggregate versus first-natural-minibatch direction, full versus tenth displacement. No candidate omitted or chosen based on training performance; no new fitting. Same704 M18 interventions defined by M11 basis checkpoints, same native effects, same output inventory and lexical/template splits. Training perturbations use M18 initial directions and therefore differ from the benchmark directions.

## Causal transfer

Counts of model/region cases with lower error than their own initial model. These are eight correlated descriptive comparisons, not eight independent replications. Lower normalized response error is better; a zero-response predictor has error1.

| Candidate | Margin /8 | Log-probability /8 | Equal-family margin /8 | Joint lexical/context margin /8 |
|---|---:|---:|---:|---:|
|minibatch_adam:1.0:live|6|5|4|3|
|minibatch_adam:0.1:live|4|1|5|4|
|aggregate_gradient:1.0:live|5|7|7|4|
|aggregate_gradient:0.1:live|5|8|5|5|

Aggregate full-step log-probability error improves7/8, with changes ranging from a substantial scalar72-prefix improvement (1.63480→1.16820) to a block71-prefix regression (1.08988→1.20956). The tenth aggregate step improves8/8 log-probability comparisons, but some gains are small (block71-task1.03814→1.03669). Both scales improve only5/8 margin comparisons. Neither sign counts nor one favorable metric establish uniformly faithful task computation.

The natural-only minibatch direction also improves several cases. Direct aggregate-versus-minibatch comparisons at matched displacement are:

| Aggregate vs same-norm minibatch | Margin /8 | Log-probability /8 | Equal-family margin /8 | Joint margin /8 |
|---|---:|---:|---:|---:|
|1.0|3|6|6|4|
|0.1|4|8|6|5|

Because directions differ in objective coverage, data averaging and preconditioning, these comparisons do not isolate the effect of response supervision. This is exactly why a full-data KL-only control is needed before attributing gains to causal targets. No claim that blocks beat scalar features follows from these two seeds.

## Ordinary output fidelity

All1152 existing natural/original/expanded inputs per checkpoint are retained, including the three nontraining lexical/template quadrants and failed native competence cohorts. Aggregate scores below use all examples, not only passing cohorts. Full analysis JSON also contains tenth-step and minibatch controls, last-token KL, task accuracy and local reconstruction error.

| Start | Natural KL initial → full aggregate | Joint lexical/context KL initial → full aggregate |
|---|---:|---:|
|scalar71|0.118066 → 0.118550|0.074757 → 0.073694|
|scalar72|0.114626 → 0.113949|0.058955 → 0.055284|
|block71|0.147658 → 0.147388|0.085105 → 0.084492|
|block72|0.156909 → 0.155345|0.078256 → 0.079576|

At full displacement, natural KL and joint lexical/template KL each improve in3/4 starts, with regressions in different starts. At tenth displacement, both improve inall4. These descriptive outcomes do not select a winning step size or establish a robust training procedure.

These are familiar development sources, not pristine holdouts. The jointly withheld causal panel contains only six pairs. Whole-task exclusions remain unchanged. A one-step fidelity gain is neither identification of semantic units nor validated manifold structure or new-task discovery.

## What to test next

Close the broad optimizer diagnosis. The next bounded causal-supervision comparison should construct full-data intact-KL-only directions from the already saved M24 natural K and task K gradients, preserving their half/half schedule weighting. Use both existing raw displacement norms and the same decoder normalization across all four starts. Contrast against existing aggregate K+lambdaR candidates; no new gradient targets, optimizer or loss-weight tuning, no seed selection. Replay training scores and evaluate the same native/output inventories. This removes the objective-coverage mismatch of the natural-only control while matching averaging and Euclidean direction construction. Remaining parameter-direction and geometry effects are part of the intended intervention. If no consistent response advantage survives this control, preserve that result and do not label the improvement a success of causal supervision.

## Reconstruction and audit

Strict replay initially rejected re-adding stored realized parameter differences: it produced a1.10e-5 maximum logit discrepancy. Reapplying the original raw displacement followed by decoder normalization exactly reproduces every saved realized parameter difference and all40 saved intact/intervened output tensors for the20 checkpoints. The failed reconstruction artifacts and receipt are retained. This repair changes numerical reconstruction, not the experiment or targets.

Native evaluation completed4572forwards,3520 responses,880 baselines. Independent NumPy audit checks120 sampled decoder/response calculations and all560 summaries; all704 prior native effects replay,96 inactive directions retained. Output evaluation completed24192forwards and23040 records. Independent audit checks all20 checkpoint hashes, complete inventories/aggregates, sampled NumPy decoder/output metrics, and exact initial inventory reproduction. It does not independently reimplement the transformer. Parameter reconstruction replay covers one fixed training endpoint per checkpoint, not every training endpoint.

Output audit: 140 samples, maximum decoder relative error 2.08e-07, maximum metric absolute error 5.42e-08, near-tie rows 0.

[Protocol](validation/finite_candidate_protocol.json), [candidate replay](validation/finite_candidate_models_audit.json), [native audit](validation/finite_candidate_native_fidelity_audit.json), [output audit](validation/finite_candidate_outputs_audit.json), [all comparisons](validation/finite_candidate_transfer_analysis.json). Checkpoint and sample tensors remain local under `artifacts/bilin18_tensor_decomposition/emlp_m26_*`, with paths and hashes in result JSONs.
