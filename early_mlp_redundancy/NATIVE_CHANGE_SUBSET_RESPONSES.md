# M67: downstream task effects can survive inaccurate intermediate changes

The fixed native intervention test is complete: all84 pairs, MLP0–2, source/final/identity sites, both endpoints, and four predetermined half-channel masks (two random masks and their complements). No mask or task was chosen by its result. All six ineligible source conditions remain explicit.

**Doubling a retained half worsens local write fidelity in all12 layer/site/split aggregates, but improves pooled output-response fidelity at source positions in all6 layer/split comparisons. At final positions it worsens output fidelity in all6.** This is a direct demonstration that local Euclidean similarity and preservation of downstream computation can rank interventions differently.

## Exactly what was intervened on

Start with the original recipient state and add only a subset of the native donor-minus-recipient channel changes, or twice that subset. The recipient's existing channel outputs remain present. The teacher is the complete native donor-write minus recipient-write intervention, and responses are centered on the intact recipient baseline. This is change-subset substitution, not neuron deletion, model pruning, or proof that a half-width model implements the same function.

The four masks are reused across conditions and include complements; they are paired controls, not four independent trials. Both directions of each prompt pair are retained and also are not independent observations.

## Results

Relative squared error uses the relevant teacher energy as denominator. For output responses, zero response scores1; for local writes, zero change scores1. Family means weight each available family equally; pooling weights by target response energy.

| Split | MLP | Site | Local half | Local doubled | Response half: pooled | Response doubled: pooled | Response half: family | Response doubled: family |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| train | 0 | source | 0.435 | 0.741 | 0.495 | 0.353 | 0.421 | 0.470 |
| train | 0 | final | 0.447 | 0.789 | 0.433 | 0.744 | 0.533 | 1.144 |
| train | 1 | source | 0.399 | 0.596 | 0.799 | 0.104 | 0.436 | 0.291 |
| train | 1 | final | 0.438 | 0.753 | 0.330 | 0.405 | 0.451 | 0.819 |
| train | 2 | source | 0.415 | 0.659 | 0.450 | 0.171 | 0.388 | 0.315 |
| train | 2 | final | 0.447 | 0.788 | 0.336 | 0.378 | 0.387 | 0.558 |
| source_eval | 0 | source | 0.437 | 0.747 | 0.512 | 0.429 | 0.587 | 1.375 |
| source_eval | 0 | final | 0.466 | 0.866 | 0.428 | 0.706 | 0.458 | 0.826 |
| source_eval | 1 | source | 0.391 | 0.565 | 0.481 | 0.167 | 0.429 | 0.319 |
| source_eval | 1 | final | 0.448 | 0.792 | 0.400 | 0.641 | 0.551 | 1.231 |
| source_eval | 2 | source | 0.429 | 0.718 | 0.387 | 0.350 | 0.398 | 0.462 |
| source_eval | 2 | final | 0.460 | 0.839 | 0.505 | 0.998 | 0.564 | 1.251 |

MLP1 source evaluation illustrates the reversal: local error rises0.391→0.565, while pooled output-response error falls0.481→0.167, equal-family error falls0.429→0.319, and answer-margin error falls0.483→0.092. Training source responses show the same direction, but remain training evidence.

The result is not uniform across tasks or metrics. Equal-family source error improves in only3/6 layer/split aggregates, despite pooled improvement6/6. MLP0 evaluation source error worsens0.587→1.375 under equal-family weighting even as pooled error improves. Source answer-margin error improves5/6. Final-position response and margin errors worsen in all6 comparisons. All mask-specific and family-specific summaries remain available.

## What this says about distributed, redundant and cancelling computation

M66 established that these changes are broadly distributed and net constructive across native channels. M67 adds evidence that, particularly for MLP1 source changes, a scaled partial change can preserve much more of the downstream task effect than its intermediate-vector accuracy suggests. It supports partial functional substitutability *under this particular change intervention and output metric*. It does not support literal removal redundancy or unchanged computation.

This also explains why a label such as “distributed redundant MLP” is too coarse. Distribution describes where the contributions live. Coordination describes how their vectors combine. Substitutability describes what can replace them while preserving a specified effect. The downstream reader determines which errors matter. Source and final positions in the same MLP can differ on the last property.

For a locally linear reader J, the same subset-error theory applies to transformed contributions Jv_j, with `S_J = ||sum Jv_j||²` and `C_J = sum ||Jv_j||²`. Euclidean C/S need not equal C_J/S_J. A change can therefore be poorly reconstructed locally but accurate in directions the reader uses. Our finite interventions also include nonlinear downstream effects, so this result does not identify a linear projection or prove all discarded error lies in a nullspace.

No new evidence here makes MLP2 generally cancellation-dominated. Nor do these measurements establish semantic clustering, a clean manifold, or reusable computational DAG nodes. They refine the causal metric that an explanation must satisfy.

## Return to featurizers

The direct target should be the downstream effect of the change a featurizer *actually encodes*, not only the response to an exact native change supplied externally. M63 exposed that objective mismatch, M64–M65 isolated a candidate boundary problem, and M66–M67 show why forcing local vector fidelity alone is insufficient.

The next bounded step is an implementation and gradient preflight for an encoded-change causal loss using only the existing training pairs. Keep the same six starts, fixed schedule, natural-text retention objective and training-only energy scales as M60/M61; do not add a new architecture or tune on these evaluation results. A later matched fit can compare this objective against existing KL-only and supplied-direction controls. A successful loss still would not establish semantic units or new-task discovery.

## Verification

9186 native forwards; all1338 previously saved full-change teacher results and all168 native baseline final vectors replay exactly. Independent NumPy audit verified12048 candidate final-response vectors,1506 teacher vectors, all applied masked changes and local errors, and six retained ineligible conditions. Maximum applied-vector relative squared discrepancy is3.10e-14; maximum response-metric absolute discrepancy3.56e-14. Identity controls remain zero. Full product-difference versus write-difference rounding discrepancy is retained separately in teacher rows; the original native write difference is always the teacher.

The native subset comparison is closed. The next fixed preflight is `validation/encoded_change_objective_preflight_protocol.json`. No amplitude, mask, layer or task sweep follows this result.
