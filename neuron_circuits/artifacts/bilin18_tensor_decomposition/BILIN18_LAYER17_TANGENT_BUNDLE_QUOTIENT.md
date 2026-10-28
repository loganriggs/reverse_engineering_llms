# Role-separated tangent-bundle quotient of the final MLP

The natural state, incoming Layer-16 tangent, and outgoing final-MLP write are treated as distinct tensor roles. Uncentered PCA bases are fit on 192 FineWeb discovery sites, frozen, and tested on a 64-site discovery holdout plus disjoint FineWeb, WikiText, and TinyStories. All effects use the exact original factors and the untouched endpoint adjoints.

## Locked hypotheses

- **T0_analytic_jvp_recomposes_native_path: supported.** The analytic final-MLP JVP recomposes the saved native backward-path effect to relative error at most 1e-5.
- **T1_output_write_basis_is_compact: not supported.** A discovery-frozen rank-64 natural-write basis alone retains at least 0.95 of every endpoint on held-out FineWeb and both OOD corpora.
- **T2_role_separated_rank64_recovers_scores: supported.** The rank-64 context/tangent/output quotient retains at least 0.80 of prediction and target effects on every corpus.
- **T3_role_separated_rank64_recovers_margin: not supported.** The rank-64 role-separated quotient retains at least 0.70 of the precision-sensitive margin effect on every corpus.
- **T4_role_separated_rank128_is_sufficient: not supported.** The rank-128 role-separated quotient retains at least 0.90 of prediction and target and at least 0.80 of margin on every corpus.
- **T5_role_separation_beats_shared_pca: supported.** At rank 64, the role-separated quotient exceeds the shared-state-PCA quotient by at least 0.20 for prediction and target and 0.30 for margin on every corpus.
- **T6_role_assignment_matters: supported.** At rank 64, the correctly assigned quotient exceeds the context/tangent-swapped quotient by at least 0.10 in mean fidelity across endpoints on every corpus.
- **T7_role_quotient_beats_random: supported.** At rank 64, every endpoint exceeds the matched random-quotient 95th percentile by at least 0.20 on every corpus.
- **T8_role_quotient_transfers_ood: supported.** At rank 64, no OOD endpoint falls more than 0.15 below held-out FineWeb.
- **T9_contracted_tensor_is_exact: supported.** The contracted rank-64 context-by-tangent-by-output tensor matches the projected native CP computation to relative error at most 1e-5.
- **T10_validation_split_is_not_fit: supported.** The bases are fit on the first 192 discovery sites and all selection diagnostics are also reported on the disjoint final 64 discovery sites.

Maximum analytic-JVP/native-path error: `7.75e-07`.
Maximum contracted-tensor/projected-CP error: `6.43e-07`.
Rank-64 tensor output-mode energy R90: `15`; top share: `0.291`.

## Quotient fidelity

Each cell is prediction/target/margin fidelity relative to the complete native final-MLP direct path.

| Corpus | Condition | rank 32 | rank 64 | rank 128 |
|---|---|---:|---:|---:|
| fit | output_only | 0.971/0.955/0.543 | 0.977/0.963/0.651 | 0.985/0.975/0.765 |
| fit | context_only | 0.967/0.958/0.784 | 0.985/0.979/0.889 | 0.998/0.996/0.976 |
| fit | tangent_only | 0.938/0.924/0.606 | 0.953/0.942/0.674 | 0.968/0.960/0.780 |
| fit | role_separated | 0.907/0.888/0.429 | 0.932/0.916/0.526 | 0.959/0.945/0.658 |
| fit | shared_state_pca | 0.159/0.094/-0.718 | 0.307/0.272/-0.399 | 0.468/0.445/-0.027 |
| fit | swapped_roles | -0.293/-0.305/0.096 | -0.030/-0.054/0.147 | 0.245/0.215/0.237 |
| validation | output_only | 0.954/0.917/0.400 | 0.962/0.929/0.438 | 0.971/0.944/0.507 |
| validation | context_only | 0.937/0.907/0.473 | 0.942/0.918/0.558 | 0.960/0.943/0.638 |
| validation | tangent_only | 0.930/0.901/0.481 | 0.947/0.924/0.538 | 0.955/0.936/0.601 |
| validation | role_separated | 0.880/0.834/0.291 | 0.890/0.846/0.295 | 0.912/0.872/0.314 |
| validation | shared_state_pca | 0.081/0.041/-0.900 | 0.214/0.150/-0.997 | 0.400/0.332/-1.237 |
| validation | swapped_roles | -0.230/-0.228/0.024 | 0.053/0.049/0.081 | 0.306/0.287/0.159 |
| fineweb | output_only | 0.970/0.958/0.695 | 0.974/0.964/0.717 | 0.978/0.968/0.755 |
| fineweb | context_only | 0.919/0.911/0.645 | 0.928/0.920/0.671 | 0.942/0.935/0.708 |
| fineweb | tangent_only | 0.947/0.936/0.694 | 0.953/0.942/0.723 | 0.960/0.951/0.750 |
| fineweb | role_separated | 0.881/0.867/0.589 | 0.899/0.886/0.603 | 0.915/0.903/0.626 |
| fineweb | shared_state_pca | 0.076/0.107/-0.147 | 0.217/0.238/-0.039 | 0.334/0.356/-0.118 |
| fineweb | swapped_roles | -0.535/-0.471/0.011 | -0.261/-0.213/0.088 | 0.070/0.097/0.234 |
| wikitext | output_only | 0.983/0.971/0.669 | 0.986/0.974/0.702 | 0.988/0.977/0.743 |
| wikitext | context_only | 0.970/0.956/0.642 | 0.975/0.962/0.670 | 0.978/0.966/0.703 |
| wikitext | tangent_only | 0.953/0.942/0.707 | 0.960/0.949/0.743 | 0.966/0.957/0.776 |
| wikitext | role_separated | 0.931/0.921/0.583 | 0.941/0.930/0.604 | 0.948/0.937/0.634 |
| wikitext | shared_state_pca | 0.168/0.174/-0.963 | 0.310/0.288/-0.869 | 0.445/0.427/-0.694 |
| wikitext | swapped_roles | -0.370/-0.334/0.137 | -0.158/-0.130/0.212 | 0.127/0.140/0.321 |
| tinystories | output_only | 0.990/0.985/0.870 | 0.992/0.988/0.894 | 0.994/0.990/0.915 |
| tinystories | context_only | 0.980/0.979/0.840 | 0.983/0.981/0.866 | 0.987/0.986/0.887 |
| tinystories | tangent_only | 0.970/0.966/0.819 | 0.977/0.974/0.840 | 0.982/0.978/0.862 |
| tinystories | role_separated | 0.944/0.942/0.785 | 0.955/0.951/0.809 | 0.963/0.960/0.838 |
| tinystories | shared_state_pca | 0.129/0.312/-0.209 | 0.267/0.436/-0.334 | 0.366/0.522/-0.270 |
| tinystories | swapped_roles | -0.109/0.063/0.539 | 0.096/0.239/0.584 | 0.318/0.418/0.630 |

## Random rank-64 controls

| Corpus | Endpoint | median | p95 | maximum |
|---|---|---:|---:|---:|
| fit | prediction | 0.000 | 0.001 | 0.001 |
| fit | target | 0.001 | 0.001 | 0.001 |
| fit | margin | 0.000 | 0.002 | 0.003 |
| validation | prediction | 0.000 | 0.001 | 0.001 |
| validation | target | 0.000 | 0.002 | 0.002 |
| validation | margin | -0.000 | 0.003 | 0.004 |
| fineweb | prediction | 0.000 | 0.002 | 0.002 |
| fineweb | target | -0.000 | 0.002 | 0.002 |
| fineweb | margin | -0.000 | 0.004 | 0.006 |
| wikitext | prediction | 0.000 | 0.001 | 0.001 |
| wikitext | target | 0.000 | 0.001 | 0.001 |
| wikitext | margin | -0.000 | 0.003 | 0.003 |
| tinystories | prediction | 0.000 | 0.001 | 0.001 |
| tinystories | target | -0.000 | 0.001 | 0.001 |
| tinystories | margin | -0.002 | 0.007 | 0.009 |

## Tensor modes

| Corpus | Endpoint | exact rank-64 | modes 8/16/32/64 |
|---|---|---:|---:|
| fit | prediction | 0.932 | 0.888/0.919/0.928/0.932 |
| fit | target | 0.916 | 0.864/0.892/0.912/0.916 |
| fit | margin | 0.526 | 0.230/0.352/0.435/0.526 |
| validation | prediction | 0.890 | 0.876/0.873/0.882/0.890 |
| validation | target | 0.846 | 0.826/0.825/0.836/0.846 |
| validation | margin | 0.295 | 0.214/0.272/0.305/0.295 |
| fineweb | prediction | 0.899 | 0.876/0.891/0.899/0.899 |
| fineweb | target | 0.886 | 0.838/0.862/0.885/0.886 |
| fineweb | margin | 0.603 | 0.297/0.470/0.591/0.603 |
| wikitext | prediction | 0.941 | 0.910/0.934/0.942/0.941 |
| wikitext | target | 0.930 | 0.884/0.914/0.930/0.930 |
| wikitext | margin | 0.604 | 0.278/0.501/0.590/0.604 |
| tinystories | prediction | 0.955 | 0.940/0.948/0.955/0.955 |
| tinystories | target | 0.951 | 0.935/0.940/0.951/0.951 |
| tinystories | margin | 0.809 | 0.685/0.729/0.800/0.809 |

The role-separated quotient is an existence and transfer test, not yet a semantic interpretation. Natural-write and tangent bases use causal path data on the discovery split. If they transfer, the next question is whether weight-only or cheaply observable proxies recover the same tensor roles.
