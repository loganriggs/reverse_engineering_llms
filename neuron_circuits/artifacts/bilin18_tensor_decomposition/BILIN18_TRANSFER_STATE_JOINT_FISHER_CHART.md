# Joint sequence-residual suffix-Fisher charts

Three new target heads were fixed before this experiment. The prior chart collapses each suffix score gradient over positions into a residual-axis covariance, allowing separate coefficients at every position. The joint chart instead diagonalizes complete sequence×residual gradient fields, preserving structured cross-position relationships. Both charts are generated from the same native context and Fisher sketch and see no target replacement tangents.

## Locked hypotheses

- **J1_joint_chart_beats_position_collapsed_chart: not supported.**
- **J2_joint_rank64_is_causally_useful: not supported.**
- **J3_joint_rank128_is_high_fidelity: not supported.**
- **J4_joint_chart_is_sampling_stable: not supported.**
- **J5_cross_position_structure_generalizes_to_new_targets: not supported.**
- **attention_jvp_forward_invariant: supported.**

## Prospective new-target results

| Target | Boundary | Joint overlap / rank90 | Collapsed r64/r128 | Joint r64/r128 |
|---|---:|---:|---:|---:|
| L1H6 | 6 | 0.535 / 54 | 0.856 / 0.917 | 0.860 / 0.904 |
| L1H6 | 10 | 0.440 / 68 | 0.800 / 0.872 | 0.754 / 0.818 |
| L1H6 | 14 | 0.407 / 73 | 0.722 / 0.810 | 0.655 / 0.742 |
| L1H6 | 17 | 0.441 / 70 | 0.639 / 0.752 | 0.629 / 0.707 |
| L3H4 | 6 | 0.535 / 54 | 0.792 / 0.881 | 0.805 / 0.870 |
| L3H4 | 10 | 0.440 / 68 | 0.746 / 0.843 | 0.697 / 0.779 |
| L3H4 | 14 | 0.407 / 73 | 0.700 / 0.788 | 0.604 / 0.713 |
| L3H4 | 17 | 0.441 / 70 | 0.658 / 0.759 | 0.600 / 0.682 |
| L5H7 | 6 | 0.535 / 54 | 0.688 / 0.808 | 0.758 / 0.836 |
| L5H7 | 10 | 0.440 / 68 | 0.721 / 0.816 | 0.682 / 0.766 |
| L5H7 | 14 | 0.407 / 73 | 0.676 / 0.768 | 0.611 / 0.709 |
| L5H7 | 17 | 0.441 / 70 | 0.614 / 0.736 | 0.584 / 0.683 |

## Guardrails

The joint chart has rank coordinates over the complete sequence field, while the collapsed residual chart has rank residual directions with independent coefficients at every token. They therefore have different runtime degrees of freedom despite sharing the same nominal rank. A joint-chart win supports structured routing but is not a parameter-count-matched compression result. Tail-sketch overlap and exact suffix fidelity are reported separately; unstable literal axes should not be assigned individual semantic names.
