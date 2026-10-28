# Finite transplant of a candidate shared output verb

L4H2 and L5H7 are the strongest bus-aware output-direction pair and remain aligned after global whitening. At each target layer, the native scalar coefficient on the leading direction is preserved while that direction is replaced by the partner direction. Eight controls are random combinations of learned head modes, constructed to have exactly the same cosine to the native direction. The full nonlinear suffix is recomputed on fresh FineWeb.

## Locked hypotheses

- **V1_controls_are_angle_matched: supported.**
- **V2_partner_is_privileged_on_both_transplants: not supported.**
- **V3_partner_beats_the_control_tail: supported.**
- **V4_partner_is_better_than_deleting_the_verb: not supported.**
- **V5_transplant_advantage_is_split_stable: supported.**
- **native_branch_matches_checkpoint: supported.**

Maximum native-branch checkpoint error: `2.86e-05`.

| Transplant / split | Native↔partner cosine | Partner KL | Control median [p10] | Delete KL | Partner/control | Coeff RMS / energy fraction |
|---|---:|---:|---:|---:|---:|---:|
| L4H2_to_L5H7 / a | 0.705 | 5.660e-04 | 4.365e-04 [3.714e-04] | 6.948e-04 | 1.297 | 1354.415 / 0.471 |
| L4H2_to_L5H7 / b | 0.705 | 5.170e-04 | 3.767e-04 [3.462e-04] | 6.109e-04 | 1.372 | 1320.360 / 0.445 |
| L5H7_to_L4H2 / a | 0.705 | 2.230e+00 | 2.748e+00 [2.241e+00] | 1.009e+00 | 0.811 | 6544.829 / 0.991 |
| L5H7_to_L4H2 / b | 0.705 | 2.209e+00 | 2.734e+00 [2.212e+00] | 1.012e+00 | 0.808 | 6530.387 / 0.991 |

## Guardrails

This tests interchangeability of one output direction, not the complete QK/value function. Controls match angular distance and are drawn from the learned output-mode span, but they are not matched for every downstream curvature direction. The pair was selected after the atlas and matched-null analyses; the evaluation corpus is fresh, but a positive result remains a prospective validation of this pair rather than a population prevalence estimate.
