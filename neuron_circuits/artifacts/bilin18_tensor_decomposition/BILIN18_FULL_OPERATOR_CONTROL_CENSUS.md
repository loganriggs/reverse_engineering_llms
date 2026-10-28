# Complete control census for directional operator compatibility

Every learned operator other than the target and partner supplies one exact norm-and-angle-matched control, giving 52 controls per direction. Lower percentile is better. `L3H8→L2H5` names the target first: L2H5 is transplanted into L3H8.

## Locked hypotheses

- **C1_earlier_operator_is_top_decile_forward_compatible: supported.**
- **C2_later_operator_is_not_reverse_compatible: not supported.**
- **C3_directional_asymmetry_is_large: supported.**
- **C4_both_transplants_beat_deletion: supported.**
- **C5_complete_census_is_split_stable: supported.**
- **factorization_and_native_invariants: supported.**

| Target ← partner / split | Partner KL | Control p10 / median / p90 | Percentile | Delete KL |
|---|---:|---:|---:|---:|
| L2H5_to_L3H8 / a | 1.246e-02 | 1.107e-02 / 1.252e-02 / 1.529e-02 | 0.462 | 2.091e-02 |
| L2H5_to_L3H8 / b | 1.187e-02 | 9.876e-03 / 1.119e-02 / 1.388e-02 | 0.596 | 2.264e-02 |
| L3H8_to_L2H5 / a | 2.092e-03 | 3.074e-03 / 3.488e-03 / 4.849e-03 | 0.000 | 5.121e-03 |
| L3H8_to_L2H5 / b | 1.278e-03 | 1.617e-03 / 1.904e-03 / 2.846e-03 | 0.000 | 2.265e-03 |

## Guardrails

The census removes sampling uncertainty from the control bank but reuses the finite evaluation panel from the eight-control result, so it is a post-result red team. All controls match raw Frobenius operator geometry; they do not match downstream Fisher curvature. Directionality can therefore reflect layer-specific destination sensitivity rather than a semantic refinement relation, which is exactly the distinction tested next.
