# Random-subset red team of native-head sparsity

The frozen leave-one-out top-8 and top-16 Q+V invocation sets are compared with sixteen independently sampled native-head subsets at each rank. Every subset is evaluated on the same two held-out contexts and must prospectively generate its own sites and local payload bases before candidate-operator causal testing.

## Locked hypotheses

- **P1_top8_beats_random_distribution: not supported.**
- **P2_top16_beats_random_distribution: not supported.**
- **P3_top8_has_lower_adjoint_error: supported.**
- **P4_top16_has_lower_adjoint_error: supported.**
- **P5_specific_native_head_sparsity_survives_redteam: not supported.**
- **attention_jvp_forward_invariant: supported.**

## Permutation summary

| Rank | Top fidelity | Random fidelity q05/median/q95 | Fidelity percentile | Top adjoint error | Random error q05/median/q95 | Error percentile |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.441 | 0.395/0.416/0.443 | 0.938 | 0.286 | 0.334/0.397/0.411 | 0.000 |
| 16 | 0.460 | 0.438/0.452/0.465 | 0.750 | 0.221 | 0.252/0.290/0.328 | 0.000 |

## Guardrails

Sixteen permutations give only a coarse empirical tail probability (resolution 1/16). The ranking is conditional leave-one-out importance and can miss cooperative subsets. Failure means the selected native identities are not distinguished from random subsets under these held-out behavioral and adjoint metrics; it does not mean the four-layer Q+V role program itself is unstructured.
