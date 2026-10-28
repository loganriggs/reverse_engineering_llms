# Curvature and wrong-target red-team of the prospective tangent router

The base JVP is checked against continuous embedding edits of increasing dose. A midpoint JVP tests whether one edit-conditioned relinearization rescues the finite circuit. A cyclic wrong target tests whether support prediction is target-specific. All causal scores patch the actual correct-target finite channel deltas, so a wrong-target router cannot win by changing the intervention being scored.

## Locked hypotheses

- **R0_tangent_is_correct_in_the_small_edit_limit: supported.**
- **R1_full_edit_failure_is_curvature: supported.**
- **R2_midpoint_tangent_rescues_full_direction: supported.**
- **R3_midpoint_router_is_causally_strong: supported.**
- **R4_target_identity_is_causally_specific: not supported.**
- **R5_target_identity_predicts_native_support: not supported.**
- **R6_base_tangent_beats_base_activity_under_pair_bootstrap: not supported.**
- **R7_midpoint_router_beats_base_activity: supported.**
- **R8_direction_accuracy_predicts_causal_fidelity: not supported.**

| Layer | Corpus | Dose cosine .1/.25/.5/1 | Product cosine base/mid/wrong | Fidelity base-active/base/mid/wrong | Support Jaccard base/mid/wrong | Base tangent - activity pair median (95% CI) | Cosine↔fidelity Spearman |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | fineweb | 0.995/0.964/0.848/0.492 | 0.492/0.995/-0.137 | 0.440/0.408/0.741/0.285 | 0.164/0.862/0.115 | -0.025 (-0.185, +0.068) | 0.174 |
| 0 | tinystories | 0.995/0.961/0.830/0.451 | 0.451/0.996/-0.205 | 0.387/0.356/0.737/0.323 | 0.151/0.869/0.120 | +0.029 (-0.115, +0.240) | 0.218 |
| 1 | fineweb | 0.995/0.970/0.872/0.547 | 0.547/0.976/-0.015 | 0.283/0.370/0.679/0.278 | 0.185/0.712/0.113 | +0.137 (-0.077, +0.215) | 0.405 |
| 1 | tinystories | 0.995/0.969/0.862/0.495 | 0.495/0.981/-0.055 | 0.387/0.405/0.739/0.301 | 0.174/0.736/0.108 | +0.019 (-0.295, +0.195) | 0.448 |

The midpoint router is still prospective relative to the endpoint effect, but it evaluates the network at a soft embedding halfway to the requested token. It is therefore an executable nonlinear approximation, not a static feature dictionary. Pair bootstraps treat lexical pairs, not contexts or neurons, as replication units.
