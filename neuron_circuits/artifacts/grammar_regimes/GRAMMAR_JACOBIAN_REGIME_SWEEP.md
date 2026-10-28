# Local Jacobian charts across rule and memorization regimes

For balanced, correctly learned facts, the centered answer-logit Jacobian with respect to the final-position post-product MLP channels is measured at every layer. Discovery examples define a PCA dictionary; disjoint unseen-template examples test reconstruction. One-sided activation edits validate that the Jacobians predict finite effects. Rho controls the fraction of fixed arbitrary assignments.

## Locked hypotheses

- **G0_jacobians_pass_finite_validation: supported.**
- **G1_reusable_rules_have_lower_early_chart_rank: supported.**
- **G2_rules_have_a_more_compact_early_dictionary: supported.**
- **G3_late_answer_interfaces_converge: not supported.**
- **G4_randomized_facts_increase_within_model_chart_variation: not supported.**
- **G5_rule_charts_have_stronger_spectral_outliers: supported.**
- **G6_depth_profiles_replicate_across_seeds: not supported.**

| rho | Layer | Effective rank | Leading/null | Mean/r16/r32 fidelity | Finite 1%/5% fidelity |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 0 | 20.9 | 3.28 | 0.101/0.524/0.670 | 1.000/0.999 |
| 0.0 | 1 | 21.8 | 3.54 | 0.310/0.811/0.897 | 1.000/1.000 |
| 0.0 | 2 | 18.8 | 3.17 | 0.559/0.909/0.956 | 1.000/1.000 |
| 0.0 | 3 | 11.1 | 3.34 | 0.841/0.990/0.996 | 1.000/1.000 |
| 0.5 | 0 | 82.3 | 1.88 | 0.013/0.139/0.229 | 1.000/0.998 |
| 0.5 | 1 | 73.4 | 1.98 | 0.043/0.250/0.361 | 1.000/0.999 |
| 0.5 | 2 | 43.5 | 2.62 | 0.129/0.580/0.732 | 1.000/0.999 |
| 0.5 | 3 | 17.0 | 5.10 | 0.935/0.982/0.998 | 1.000/1.000 |
| 1.0 | 0 | 77.7 | 2.06 | 0.002/0.133/0.236 | 1.000/0.997 |
| 1.0 | 1 | 74.6 | 2.03 | 0.023/0.223/0.334 | 1.000/0.998 |
| 1.0 | 2 | 38.6 | 2.66 | 0.089/0.618/0.767 | 1.000/0.999 |
| 1.0 | 3 | 14.7 | 5.57 | 0.935/0.982/0.997 | 1.000/1.000 |

A lower-rank rule regime would support the hypothesis that reuse compresses the family of local computations. Convergence only at the final layer would instead separate upstream rule/address organization from a shared answer interface. Null or reversed load effects would falsify Jacobian rank as a simple memorization axis.
