# Pairwise Mobius reconstruction of three-layer suffix routes

Complete singleton and pair softmax-Fisher response tensors reconstruct each triple by second-order inclusion-exclusion. Exact and reconstructed responses independently select 16-mode programs, which are compared by finite full-model interventions.

## Locked hypotheses

- **M0_pair_mobius_reconstructs_response: supported.**
- **M1_pair_mobius_selects_equivalent_program: supported.**
- **M2_pair_mobius_recovers_support: supported.**
- **M3_true_third_order_propagation_is_required: not supported.**

| Corpus | Response cosine median/q10 | Relative error median | Finite fidelity median/q10 | Support recall median | Objective efficiency median |
|---|---:|---:|---:|---:|---:|
| fineweb | 1.000/1.000 | 0.001 | 1.000/1.000 | 1.000 | 1.000 |
| wikitext | 1.000/1.000 | 0.001 | 1.000/1.000 | 1.000 | 1.000 |

This tests how a chosen route composes, not how the model or an interpreter chooses that route. Pairwise reconstruction can pass while the address problem remains unsolved.
