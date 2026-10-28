# M24: initial family gradients largely admit joint local improvement

2026-10-28. Following the failed momentum ablation, measure gradient compatibility rather than fit another optimizer variant. Four unchanged M18 starts, all480 scheduled response targets, ten families, frozen response weights and actual scheduled natural-training documents. No evaluation data enters the diagnostic. Counts preserve repeated endpoints and inactive targets. No model parameters are retained after perturbation checks.

For each family, compute mean intact-output KL gradient K and mean response-error gradient R through both downstream branches. Compute natural K with its actual document multiplicities. The schedule gradient is half natural K plus half (task K + lambda task R), matching the equal natural/task update counts. All gradients use the raw Euclidean parameter coordinates, not Adam preconditioning. A positive dot product between a family's R gradient and the schedule gradient means infinitesimal descent on the latter decreases that family's R.

| Start | Conflicting R-family pairs /45 | Cosine(task K, weighted R) | Cosine(natural K, weighted R) | Families locally harmed by schedule descent /10 |
|---|---:|---:|---:|---:|
|scalar71|24|.053|−.004|1|
|block71|10|.110|.019|0|
|scalar72|19|.122|.012|0|
|block72|12|.077|.012|0|

The exception is scalar71 pronoun_gender (cosine−.0276). Thus pairwise conflicts exist but do not force broad immediate degradation:39/40 family response losses have a favorable local direction under the actual aggregate objective. Pair counts include arbitrarily small negative alignments and are not a severity measure. Full matrices and gradient norms are retained.

Weighted response gradient norms are .124,.206,.198,.266; corresponding task K norms .103,.113,.118,.119 and natural K norms .0216,.0248,.0225,.0218 (table order). Matching initial scalar loss contributions did not match gradient magnitudes. These norms alone do not explain training failure.

## What this changes

The evidence weakens an explanation based on immediate aggregate conflict at the starts. It does not rule out conflict later in training, interference between individual examples hidden by family averages, coordinate/preconditioner effects, finite-step overshoot, or representational limitations away from these points. It does not prove joint optimum feasibility. Better local derivatives are not better semantic features or causal transfer.

Stop the gradient map here. Next freeze one finite-displacement diagnostic across all four starts: compare descent along the aggregate gradient with the first scheduled minibatch Adam direction, at a common displacement norm derived from that existing first-step rule, and at one tenth that norm as a prespecified local control. Apply the normal decoder-column normalization. Evaluate the complete frozen training objective and each family's response loss, with actual supports and fixed original supports to separate smooth versus support effects. Restore starts exactly between conditions. No optimizer/loss-weight search or fitted final model. This will test whether full-data local compatibility survives a finite update and whether minibatch direction or support transitions explain the discrepancy. It remains a bounded diagnostic feeding causal featurizer training, not an alternative research objective.

## Validation

Run completed1073 native forwards. All480 initial K/R values replay the cached targets. Four parameter-space combined-objective directional checks use both epsilon .001 and .0003; maximum relative discrepancy .001543 across all eight differences. Every model restored exactly after finite differences. Native model remains frozen.

Independent NumPy audit recomputes all four Gram matrices, cosines, norms, per-family combined gradients and exact schedule aggregation from saved gradient tensors. Maximum Gram discrepancy5.78e-15. It does not independently reproduce transformer backpropagation; finite differences sample one fixed endpoint per start rather than every family. Existing M19 checks additionally cover both downstream response branches.

[Protocol](validation/family_gradient_protocol.json), [complete map](validation/family_gradient_compatibility.json), [audit and summaries](validation/family_gradient_audit.json). Gradient tensors are local at `artifacts/bilin18_tensor_decomposition/emlp_m24_family_gradients.pt`, hash-linked from results. Reproduce with `probe_family_gradient_compatibility.py` then `audit_family_gradient_compatibility.py` in the documented model environment. Existing results must be preserved before rerunning.
