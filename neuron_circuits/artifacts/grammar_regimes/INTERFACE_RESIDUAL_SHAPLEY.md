# Shapley split of generic answer verb and semantic residual

The exact semantic activation patch is split into the discovery-Fisher-scaled answer ray and its residual; all four finite coalitions are evaluated on held-out identities/template 6.

## Locked hypotheses

- **C0_shapley_recomposes_semantic_effect: supported.**
- **C1_final_effect_is_answer_verb_dominated: supported.**
- **C2_rule_specific_residual_matters_upstream: not supported.**
- **C3_interaction_is_more_important_upstream: supported.**
- **C4_residual_alone_is_not_the_rule: supported.**

Maximum vector recomposition error: `1.91e-06`.

| Layer | Answer/residual target projection | Residual absolute share | Interaction relative norm | Semantic/answer/residual accuracy | Scale |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.931/0.068 | 0.126 | 0.265 | 0.722/0.588/0.022 | 1.88 |
| 2 | 0.931/0.068 | 0.155 | 0.179 | 0.601/0.570/0.001 | 1.16 |
| 3 | 0.949/0.051 | 0.085 | 0.108 | 0.843/0.734/0.003 | 1.18 |

Shapley allocation is exact for this declared two-component activation game. Changing the answer ray or calibration changes the residual and therefore the allocation; causal dominance is conditional, not a unique storage percentage.
