# Simple explanations of mixed-model causal charts

Bounded ridge models are fit on unselected training facts in one template, selected on disjoint unselected facts in a held-out template, and evaluated on correctly generalized held-out pairs. The target is the activation-covariance-weighted correct-versus-rival margin operator. Semantic, output-role, token/address, and combined feature vocabularies are fixed before fitting.

## Locked hypotheses

- **P0_semantic_features_explain_heldout_charts: not supported.**
- **P1_address_features_beat_semantics: not supported.**
- **P2_short_additive_description_is_sufficient: not supported.**
- **P3_feature_models_beat_permutation_null: not supported.**
- **P4_training_fit_transfers_to_new_pairs: supported.**

At early layers P4 is a vacuous closeness result because both validation and
held-out R2 are near zero. At the final layer, where R2 is high, the transfer is
substantive.

| Seed | Layer | Semantic R2 | Output R2 | Address R2 | Combined R2 | Combined validation R2 | Null p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0.002 | 0.006 | -0.001 | 0.005 | 0.011 | 0.000 |
| 0 | 1 | 0.023 | 0.041 | 0.001 | 0.021 | 0.020 | 0.000 |
| 0 | 2 | 0.071 | 0.176 | 0.004 | 0.126 | 0.122 | 0.000 |
| 0 | 3 | 0.382 | 0.936 | 0.017 | 0.899 | 0.908 | 0.002 |
| 1 | 0 | 0.005 | 0.013 | 0.005 | 0.014 | 0.013 | 0.000 |
| 1 | 1 | 0.032 | 0.053 | 0.006 | 0.030 | 0.032 | 0.000 |
| 1 | 2 | 0.093 | 0.177 | 0.006 | 0.148 | 0.135 | 0.000 |
| 1 | 3 | 0.360 | 0.929 | 0.019 | 0.883 | 0.910 | 0.002 |
| 2 | 0 | 0.002 | 0.004 | 0.000 | 0.001 | 0.001 | 0.001 |
| 2 | 1 | 0.021 | 0.042 | 0.001 | 0.020 | 0.022 | 0.000 |
| 2 | 2 | 0.071 | 0.167 | 0.005 | 0.134 | 0.120 | 0.001 |
| 2 | 3 | 0.371 | 0.932 | 0.014 | 0.884 | 0.896 | 0.000 |

A failed bounded explanation does not prove irreducibility: the model may use a compact nonlinear or tensor program outside this feature class. It does falsify the claim that the observed chart family is already explained by these simple additive semantic/address variables.
