# M31: context-reencoded components do not rescue the remaining block crossover

2026-10-28. Use exactly the same720 source/context rows and20 checkpoints as M30. Instead of transporting the reference task-span vector, encode each context's own native MLP1 write with the original M18 featurizer and decode the same group. Keep prefix intervention zero. Candidate weights, group IDs, task tokens and wrappers do not change. No fitting or candidate selection. Native effects may change because the intervention changes.

## Joint causal supervision versus matched KL-only

Ratios below1 favor joint supervision on weighted log-probability response error. Each ratio compares models under the same intervention scheme. Comparing ratios across schemes is not evidence that a fixed model itself improved: the causal target also changed.

| Start | Size | Passage fixed → recomputed | Sample fixed → recomputed |
|---|---:|---:|---:|
|scalar71|1.0|0.861387 → 0.855028|0.939315 → 0.898672|
|scalar71|0.1|0.976662 → 0.975457|0.989711 → 0.989645|
|scalar72|1.0|0.965080 → 0.935321|0.955945 → 0.962228|
|scalar72|0.1|0.986479 → 0.992322|1.001691 → 1.000001|
|block71|1.0|0.962263 → 0.943787|1.012264 → 1.007412|
|block71|0.1|1.008931 → 1.006179|0.995687 → 0.994042|
|block72|1.0|0.953075 → 0.948291|1.004395 → 1.008090|
|block72|0.1|0.989101 → 0.988653|0.992626 → 0.992740|

Reencoding improves the full-step joint/control ratio for all four passage cases, but only two of four sample cases. Full-step sample wins stay2/4: both block models remain slightly worse than KL-only, by .74% and .81%. Tenth-step counts also remain3/4 for both passage and sample. Scalar72's recomputed tenth-step sample ratio is1.000000818, effectively a tie at this descriptive precision; strict win counts are not significance tests.

The full-step sample margin win count changes3/4→2/4, while other context/scale margin counts stay unchanged. Thus reencoding the current group is not a uniform fix, and causal fidelity cannot be summarized by a single winning label. M30's competence/concentration qualifications remain relevant; these are small development panels with correlated repeated rows.

## Representation geometry

Cosines condition on both perturbations being nonzero. Norm ratios include every nonzero transported direction, including ones that become inactive. Support changes count examples with any selected-group token gate change; they are not counts of independent features.

| Start | Wrapper | Both nonzero /60 | Median cosine | Median norm ratio | Group-support changes /60 |
|---|---|---:|---:|---:|---:|
|scalar71|passage|58|0.9734|0.9417|46|
|scalar71|sample|60|0.9762|1.0037|42|
|scalar72|passage|60|0.9577|0.9365|49|
|scalar72|sample|60|0.9635|0.9955|49|
|block71|passage|38|0.9774|0.9505|18|
|block71|sample|40|0.9940|1.0177|10|
|block72|passage|47|0.9961|0.9539|16|
|block72|sample|49|0.9971|1.0006|7|

For the two block models under sample, median cosines are .9940 and .9971, with median norm ratios1.0177 and1.0006. Most of the active components therefore remain close geometrically even though their model-level causal-fidelity advantage is not preserved. This weakens a wholesale transported-direction mismatch explanation. It does not rule out consequential outliers, nonlinear downstream responses or effects of the rest of the reconstructed write. Different active fractions prevent treating support-change counts as an unbiased architecture ranking.

Inactive counts change from31 in every fixed-vector context to31/35/30 in reference/passage/sample after reencoding. All zeros remain. No angle or magnitude matching was used to improve a result.

## Interpretation and next bounded information gain

The tested representation adjustment did not resolve the block crossover. Geometry alone is not enough to predict these causal-fidelity comparisons. This is not a proof that context-dependent features or manifolds are impossible; a legitimate feature's downstream role may itself depend on context, and reconstruction error in other directions may matter.

Stop intervention-construction variants here. Next replay only the native model's fixed M30 interventions on the same240 directions across the three wrappers, collecting full final-logit response vectors and verifying every previous native margin/energy/competence score. Use the reference intact probability distribution as a common metric across each paired triple. Decompose squared differences of log-probability response vectors into a weighted centered component and a constant component, and verify that centered log-probability response equals centered logit response. Report angles/norm changes, zero cases, family contributions and the exact additive identity. This separates changes in relative-logit effects from changes in the common log-probability shift; it does not automatically attribute that shift to a unique internal mechanism.

No student fitting or new prompts are required. This returns the analysis to the native computation: whether a geometrically stable intervention has a context-sensitive functional effect, and whether the response metric itself emphasizes that variation. Use the result to guide functional manifold validation or broader task coverage, not another optimizer search. Prefix-region failures, reusable semantic labels and new-task discovery remain open.

## Validation

Probe52462 completed6802forwards and3600 responses. Audit46213 verifies720 exact-token pairings, all candidate hashes, all3600 intact metric replays against M30,1200 reference response replays, all660 summaries and forward accounting.60 independent NumPy samples recompute target-context M18 coordinates, group selection, decoder deltas, direction geometry, native/featurizer response metrics and competence. Maximum decoder relative error1.78e-7, metric error9.58e-16, zero near-tie mismatches. Reference group supports match the hash-verified M19 cache; transported vectors match independently audited M28 samples. Native transformer internals are not independently implemented. Familiar development data, no fresh whole-task holdouts.

[Protocol](validation/adaptive_wrapper_protocol.json), [results](validation/adaptive_wrapper_results.json), [audit](validation/adaptive_wrapper_audit.json), [paired and geometric analysis](validation/adaptive_wrapper_analysis.json). Local tensors/progress: `artifacts/bilin18_tensor_decomposition/emlp_m31_*`, hash-linked from results.
