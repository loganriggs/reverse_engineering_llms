# M30: paired wrapper changes can erase a response-fidelity advantage, but not uniformly

2026-10-28. Use all240 task-region scheduled rows from M28, with their already-frozen unused groups. For each, retain the exact source task tokens and original M18 task-span perturbation. Apply the identical float32 task-span delta under the reference training wrapper and both evaluation wrappers, shifting it to the task-token positions and leaving prefix positions zero. All20 candidate/initial models remain fixed; no fitting or sample selection.720 context-directions produce3600 candidate responses.

## Paired context results

R is weighted squared log-probability response discrepancy. Joint/KL-only ratios below1 favor causal supervision. Each ratio compares the candidates on identical native interventions within that context. Context changes native effect magnitudes, so raw errors across wrappers are not the primary comparison.

| Start | Size | Reference joint/KL-only R | Passage wrapper | Sample wrapper |
|---|---:|---:|---:|---:|
|scalar71|1.0|0.8570|0.8614|0.9393|
|scalar71|0.1|0.9920|0.9767|0.9897|
|scalar72|1.0|0.9942|0.9651|0.9559|
|scalar72|0.1|0.9827|0.9865|1.0017|
|block71|1.0|0.8619|0.9623|1.0123|
|block71|0.1|0.9948|1.0089|0.9957|
|block72|1.0|0.9222|0.9531|1.0044|
|block72|0.1|0.9903|0.9891|0.9926|

At full size, joint supervision wins4/4 reference aggregates,4/4 passage aggregates and2/4 sample aggregates. Both block models lose their advantage under sample: block71 goes from13.8% lower error to1.2% higher; block72 from7.8% lower to.4% higher. Scalar72 improves its advantage under either new wrapper. At tenth size, counts are4/4,3/4,3/4, with small reversals. Therefore context sensitivity exists in this paired test, but a universal collapse would overstate the evidence.

Margin wins are3/4 in all three full-step contexts, and3/4,4/4,2/4 at tenth size. Family-level log-probability wins are26/40,26/40,21/40 full-size and23/40,25/40,22/40 tenth-size. The classification-related metric and finer task-family accounting do not support an unqualified statement that every task's computation is improved or degraded.

## Competence and concentration checks

Native binary answer success is100% on reference and passage inputs and98.3% on sample inputs for each schedule. Native full-vocabulary top1 success is83.3–85% on reference,76.7–85% on passage and80–88.3% on sample. None of these cases were filtered from the primary result.

Post-result robustness accounting (not a replacement validation set) retains endpoints native-binary-correct in all three wrappers:59/60 per model. Both full-step block reversals persist. Paired leave-one-family-out analyses also retain each full-step block sample regression in9/10 omissions; these correlated subsets are not significance tests or proof against concentration within a family.

The stricter all-context native-top1 subset retains45/60 seed71 and43/60 seed72 endpoints and changes conclusions. Full-step block71 has ratios .8674/.9342/.9463, retaining an advantage in all contexts. Full-step block72 has1.0834/1.0262/1.1312, losing the advantage even at reference. Thus which competent behaviors receive weight matters, and the aggregate block crossover should not be claimed as a universal mechanistic law. Scalar71 full-step sample is essentially tied on this stricter subset (1.0004). All subset tables and original scores remain available.

## What was actually isolated

Task content, candidate parameters and the task-span perturbation were held fixed. Wrappers change surrounding text, absolute task positions, native activations and downstream sensitivity. This is not a separation of lexical wrapper meaning from positional effects. A fixed vector carried from another context may also be less appropriate than removing the current context's own group component. The experiment concerns task-span interventions and cannot explain prefix-region failures. It does not establish semantic units or manifold validity.

## Next: fixed vector versus context-adapted group component

Stop wrapper selection here. On the exact same720 context rows and20 candidates, decode the same original M18 group using each context's own native write instead of transporting the reference component. Keep group IDs, task masks, sources, templates, candidates and norms fixed. Record direction angle, magnitude and active/inactive changes relative to the transported delta; retain zeros. Replay reference responses and all intact baselines. Compare joint/KL-only fidelity within each intervention scheme, retaining native effect energies rather than conflating changed directions with improved models.

If reencoding the group in the target context restores fidelity, the useful candidate unit may be a context-dependent component rather than a fixed intervention vector. If it does not, neither this representational adjustment nor the generator checkpoint explains the remaining errors. Either result is a representation-level diagnostic, not a reason to restart optimizer/step-size searches. All existing whole-task exclusions and the wider task/featurizer/manifold/discovery loop remain in force.

## Validation

Probe74708 completed6820forwards; audit36982 passed. All720 exact-token pairings,3600 target/context/arm records and660 summaries checked. All1200 reference rows reproduce M28 native effects and reconstructed responses.31 reference directions are inactive and remain inactive under every wrapper; none excluded.60 independent NumPy sample audits verify response/competence metrics and map task deltas exactly to the independently audited M28 deltas, with zero prefix values. Task-delta hashes are identical across wrappers for every scheduled intervention. Maximum metric error9.58e-16. Native transformer internals are not independently reimplemented. Familiar development data; no new tasks or pristine holdouts.

[Protocol](validation/wrapper_crossover_protocol.json), [results](validation/wrapper_crossover_results.json), [audit](validation/wrapper_crossover_audit.json), [paired analysis](validation/wrapper_crossover_analysis.json), [post-result robustness accounting](validation/wrapper_concentration.json). Large tensors/progress remain in `artifacts/bilin18_tensor_decomposition/emlp_m30_*`, hash-linked from results.
