# M29: changing the intervention generator modestly helps, but does not resolve transfer

2026-10-28. Keep the same22 evaluation pairs, endpoints, regions, group indices and20 initial/joint/KL-only candidates. Replace M11 intervention generators with the original M18 starting featurizers used to define training perturbations. This changes encoder activations, selected groups and decoder directions; it is NOT a pure rotation or gauge transformation. No fitting or ordinary-output reevaluation.

## Joint versus KL-only within each generator

Counts indicate lower causal error for the joint objective. Compare objectives within a generator, where native effects and denominators match. Do not treat raw cross-generator errors as measurements on identical interventions.

| Step size | Generator | Margin wins /8 | Log-probability wins /8 | Equal-family margin wins /8 |
|---|---|---:|---:|---:|
|1.0|M11|4|5|6|
|1.0|M18|5|6|8|
|0.1|M11|4|6|5|
|0.1|M18|4|7|4|

At full size, switching generators changes just one of eight margin winners and one of eight log-probability winners. At tenth size, no margin winner changes and one log-probability winner changes. The generator mismatch contributes, but is not a sufficient explanation of the mixed transfer. Equal-family margin weighting shows a stronger full-step benefit (8/8), reminding us that weighting changes the conclusion.

The input strata remain unequal. With M18 generators, the lexical-evaluation/training-template quadrant favors joint supervision in6/8 full-step and7/8 tenth-step log-probability comparisons. The training-lexical/evaluation-template quadrant favors it in5/8 and3/8 respectively. These panels contain different pairs and family coverage; their difference is not a randomized estimate of a context effect. Jointly withheld margin wins remain7/8 full and5/8 tenth, but that panel has only six pairs.

## The interventions actually changed

Across704 directions,24 switch active/inactive status. Total inactive directions change96→94. Native response energy changes by model; for block71 it rises2.3784→2.9948, while for scalar72 it falls1.3680→1.1652. Neither rescaling nor clipping was used to force agreement. All880 intact candidate baselines reproduce their previous task metrics and KL, so these changes concern interventions rather than changed reconstructed models or inputs.

This provides a limited negative result: adopting the training generator does not make the evaluated task computations uniformly faithful. M28 nevertheless showed benefits on unused groups at training inputs. Together these findings motivate a paired input-context test, not another optimizer or generator search. They do not establish semantic units, manifolds, or new-task discovery.

## Next experiment: fixed task-span intervention across wrappers

Freeze a paired crossover using all M28 task-region scheduled rows, both existing evaluation wrappers, all20 checkpoints and both fixed update scales. Retain the same source task tokens. Decode the original unseen M18 group on its training-wrapper input, then transplant exactly that task-span delta onto the corresponding task-token positions under each evaluation wrapper; set prefix delta to zero. Require exact source-token equality and task-span length, native baseline competence reporting, and replay of reference M28 responses. Keep inactive groups, repeats, all families and native failures; no outcome-based replacement.

This holds source content, candidate parameters and task-span perturbation fixed while changing the wrapper (including its positional/contextual effects). Compare the within-wrapper joint/KL-only contrast, native effect energies and margins. Different native effect magnitudes remain explicit; do not compare raw errors alone. It is a targeted task-span diagnostic because those tokens can be aligned exactly. It does not explain prefix-region failures or prove a context-invariant semantic unit. Stop after this crossover; use the hypothesis review to choose a representation-level next step from the evidence.

## Validation

Session4239 completed4584forwards,3520responses and880baselines. Independent audit20786 passed120 sampled NumPy decoder/delta/response checks and all560 summaries, all704 selected intervention keys and within-generator invariants. Every candidate intact baseline replays; benchmark fields match the old protocol except initial basis models and explicit provenance metadata. Maximum decoder relative error1.52e-7, metric error1.78e-15, no near-tie mismatches. Native transformer internals are not independently reimplemented. All sources are familiar development data; whole-task exclusions remain unchanged.

[Protocol](validation/basis_transfer_protocol.json), [generator protocol](validation/m18_basis_benchmark_protocol.json), [results](validation/basis_transfer_native_fidelity.json), [audit](validation/basis_transfer_native_fidelity_audit.json), [paired analysis](validation/basis_transfer_analysis.json), [updated hypotheses](CAUSAL_TRANSFER_HYPOTHESES.md). Local samples/progress are under `artifacts/bilin18_tensor_decomposition/emlp_m29_*`, hash-linked from results.
