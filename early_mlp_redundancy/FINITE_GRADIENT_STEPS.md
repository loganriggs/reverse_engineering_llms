# M25: a finite aggregate update improves training responses without sacrificing intact predictions

2026-10-28. Four unchanged M18 starts; compare the first scheduled natural-minibatch Adam displacement with a negative full-schedule gradient displacement. Match raw parameter norm, then normalize decoder columns as in training. Test the frozen full norm and one tenth of it, each with live selections and original selections held fixed. Include initial baselines. No learned update is retained in the source checkpoints. All scheduled training endpoints and natural-document multiplicities enter scoring.

## One aggregate step realizes the favorable local direction

Full-norm aggregate direction, live supports:

| Start | Initial training response R | After step R | R reduction | Full schedule objective reduction | Families with lower R /10 |
|---|---:|---:|---:|---:|---:|
|scalar71|.004298|.002115|50.8%|28.4%|9|
|block71|.002505|.001354|46.0%|23.0%|10|
|scalar72|.002317|.001358|41.4%|26.9%|10|
|block72|.007959|.003791|52.4%|21.6%|10|

Both intact task KL and natural-training KL improve in all four starts. This is not a response gain purchased by sacrificing these measured intact predictions. The smaller aggregate displacement also improves total objective and response error in all four; family R improves8/10,9/10,10/10,10/10 respectively. These are training scores on familiar development data, not generalization or new-task discovery.

With original feature selections held fixed, the full aggregate step still improves the objective by .008119,.009627,.008464,.008296 respectively. Allowing selections to change modifies those improvements by −.000015,+.000276,−.000047,+.000385 in objective units (positive means worse). Thus most of the benefit can occur through continuous parameter changes with the same active coordinates. This is not proof that selection changes never matter: they determine whether scalar72's full minibatch step helps or harms the overall objective.

## Direction and finite displacement both matter

The full first-minibatch Adam step worsens the overall objective in both scalar starts and modestly improves it in both block starts. At one tenth its displacement, all four overall objectives improve. The first-order schedule prediction is negative for every direction, even the harmful full minibatch steps. For scalar71 the harmful full minibatch result persists with fixed supports; for scalar72 it becomes beneficial with fixed supports. Smooth finite-step effects and selection changes can therefore produce distinct failures.

Post-normalization full displacement norms are approximately .16307–.16311; paired aggregate/minibatch norms differ by less than .009%. Raw norms were exactly matched by construction. At one tenth scale, norms are approximately .01631. Smaller-scale controls are diagnostics, not a learning-rate selection procedure.

Crucial limitation: the first minibatch contains only natural KL, whereas the aggregate direction includes every scheduled natural/task/response objective and uses raw gradient descent rather than Adam preconditioning. This does NOT identify averaging, objective inclusion or preconditioning separately, and full aggregation costs substantially more gradient computation. It is a constructive demonstration that the existing architecture admits a nearby joint improvement, not an equal-compute training comparison or a proof that repeated aggregate steps will work.

## Next: return to causal evaluation of the representation

Stop local optimizer diagnosis here. Freeze evaluation of all four prespecified live-support candidates per start (aggregate/minibatch × full/tenth norm), plus initials, on the unchanged704-direction M18/M11-basis native benchmark and the existing natural/original/expanded output inventory. Reconstruct candidates exactly from saved deltas and original tensors; hash them and verify training replay before evaluating. Preserve every candidate, both seeds, both architectures, all three nontraining lexical/template quadrants and whole-task exclusions. No choice of a winner based on the training improvement, no further displacement tuning. This directly tests whether the joint training improvement feeds causal featurizer transfer. Still no claim of reusable semantic units or manifold validity from a one-step improvement.

## Execution and validation

First attempt session18113 terminated with exit139 (segmentation fault) before writing a complete result. Cause undetermined. The scientifically unchanged retry enabled Python fault tracing and per-condition progress receipts; session81912 completed8433forwards, reproducing earlier console values. Preserve the failure history rather than treating the partial run as evidence.

There are4320 target-condition records,36 complete condition summaries, and36 saved output samples. Independent NumPy audit replays sampled K/R (maximum error3.41e-16), first-step Adam formula, all saved parameter norms and first-order predictions, identical live/fixed parameters, all family and exact schedule aggregations, inactive targets and initial target scores. Forward count is derived from actual unique scheduled documents and cached inputs, not an assumption that every seed uses every natural document. All source parameters are restored exactly. The auditor does not independently reproduce transformer internals or support-mask computation.

[Protocol](validation/finite_gradient_protocol.json), [results](validation/finite_gradient_steps.json), [audit and contrasts](validation/finite_gradient_audit.json), [run history](validation/finite_gradient_run_history.json). Local sample tensors and actual parameter deltas: `artifacts/bilin18_tensor_decomposition/emlp_m25_finite_gradient_samples.pt`, hash-linked from results. Scripts: `probe_finite_gradient_steps.py`, `audit_finite_gradient_steps.py`.
