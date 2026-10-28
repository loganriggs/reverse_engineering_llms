# Prospective tensor-path predictor of destination compatibility

Three previously untested destinations were fixed before running. Candidate operator differences are computed from weights and the native target router, then propagated through the exact analytic checkpoint JVP. Finite 1% replacements are used only as held-out outcomes. The readable approximation contains the attention-free path plus every path with exactly one downstream attention insertion; it omits all paths containing two or more downstream attention operations.

## Locked hypotheses

- **P1_analytic_tensor_jvp_predicts_finite_candidate_geometry: supported.**
- **P2_first_order_attention_path_program_is_sufficient: not supported.**
- **P3_identity_only_is_not_sufficient: supported.**
- **P4_path_importance_is_split_stable: supported.**
- **P5_tensor_predictor_beats_raw_operator_geometry: supported.**
- **P6_prediction_is_not_one_target_specific: not supported.**
- **attention_jvp_forward_invariant: supported.**

## Results

| Target / half | Analytic→finite geometry / diagonal | Analytic→KL | Raw→KL | Identity fidelity | First-order fidelity / geometry |
|---|---:|---:|---:|---:|---:|
| L0H0 / a | 1.000 / 1.000 | 1.000 | 0.098 | 0.036 | -2.553 / 0.872 |
| L0H0 / b | 1.000 / 1.000 | 1.000 | -0.080 | 0.158 | -0.350 / 0.839 |
| L0H0 path-energy split stability | 0.880 | — | — | — | — |
| L2H2 / a | 1.000 / 0.999 | 0.999 | -0.005 | 0.113 | 0.121 / 0.842 |
| L2H2 / b | 1.000 / 1.000 | 1.000 | -0.085 | -0.040 | 0.285 / 0.879 |
| L2H2 path-energy split stability | 0.950 | — | — | — | — |
| L5H3 / a | 1.000 / 1.000 | 1.000 | -0.546 | 0.451 | 0.816 / 0.896 |
| L5H3 / b | 1.000 / 1.000 | 1.000 | -0.103 | 0.545 | 0.787 / 0.905 |
| L5H3 path-energy split stability | 0.973 | — | — | — | — |

## Interpretation guardrails

The exact analytic JVP is a local tensor-network calculation, not a semantic explanation. The first-order path program is interpretable at the level of downstream attention use, but success would not name the variables transported by those paths. Failure would specifically imply that compositions containing two or more downstream attention operations are behaviorally necessary; it would not invalidate the exact analytic predictor. Candidate replacements retain the target QK router and differ only in the value-side operator. Results therefore concern typed operator substitution, not whole-head interchangeability.
