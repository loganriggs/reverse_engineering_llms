# Source-support, normalization, and scale factorial

Discriminates T01/T02/T03 from normalization, large-edit, and denominator alternatives T04/T05/T06. Same-head shams preserve support and per-site raw-write norm, not semantic meaning. Frozen RMS is an explicit local counterfactual. All source charts frozen previously; no fitting or semantic inspection on this panel. Fidelity, effect RMS/natural coefficient SD, and normalized MSE are reported separately.

Locked gates: `{'same_support_benefit': True, 'support_specificity': True, 'normalization_primary': False, 'small_edit_recovery': False, 'exact': True}`. Native-change replay relative RMS: `2.64e-06`.

## Scale 1 controls

| Payload | RMS | Effect RMS / natural SD | Global256 fidelity | Correct hybrid | Wrong hybrid | Global normalized MSE |
|---|---|---:|---:|---:|---:|---:|
| actual | recompute | 0.097 | 0.812 | 0.939 | 0.683 | 0.001750 |
| actual | frozen | 0.108 | 0.841 | 0.944 | 0.726 | 0.001847 |
| shuffled_same_head | recompute | 0.077 | 0.789 | 0.920 | 0.663 | 0.001249 |
| shuffled_same_head | frozen | 0.073 | 0.776 | 0.915 | 0.643 | 0.001202 |
| gaussian_same_head | recompute | 0.071 | 0.800 | 0.915 | 0.614 | 0.000999 |
| gaussian_same_head | frozen | 0.074 | 0.809 | 0.920 | 0.630 | 0.001059 |
| isotropic_residual | recompute | 0.053 | 0.763 | 0.670 | 0.638 | 0.000675 |
| isotropic_residual | frozen | 0.055 | 0.775 | 0.688 | 0.661 | 0.000686 |

## Actual-head magnitude curve with recomputed RMS

| Scale | Global256 | Correct hybrid | Wrong hybrid |
|---|---:|---:|---:|
| 0.05 | 0.781 | 0.946 | 0.645 |
| 0.25 | 0.786 | 0.945 | 0.651 |
| 0.5 | 0.793 | 0.943 | 0.659 |
| 1.0 | 0.812 | 0.939 | 0.683 |

Same-head shams are norm matched but not semantically matched. Frozen normalization is an artificial local counterfactual. Bootstrap intervals in JSON resample complete text rows; no result establishes model-wide semantic coverage. Small absolute errors relative to natural variation do not imply behavioral irrelevance.
