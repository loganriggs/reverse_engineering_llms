# Finite validation of candidate read relations

Each candidate edge is paired with a non-candidate edge matched on destination position, log-distance bin, source frequency, and destination frequency. The native attention edge is then removed exactly, and both the immediate normalized operator sketch and all downstream layers are recomputed. Results use two corpus splits and 256 candidate/control edges per pair per split.

Maximum unmodified-branch logit error: `4.77e-05`.

## Locked results

- **F0_native_branch_is_exact: supported.**
- **F1_boundary_has_finite_local_effect: supported.**
- **F2_boundary_reaches_output: supported.**
- **F3_L7H8_function_read_is_finite: supported.**
- **F4_L9H7_function_read_is_finite: supported.**
- **F5_relations_are_head_specific: supported.**
- **F6_failed_name_relation_stays_negative: supported.**

| Split | Relation / head | Role | Local AUC / ratio | KL AUC / ratio | CE delta candidate/control |
|---:|---|---|---:|---:|---:|
| 0 | boundary_source / L9H8 | predicted | 0.776 / 4.51 | 0.767 / 2.50 | +0.0009876/+7.082e-05 |
| 0 | boundary_source / L8H1 | head_control | 0.527 / 0.35 | 0.521 / 0.52 | -0.0001107/+0.0002175 |
| 0 | adjacent_function_word / L7H8 | strongest_predicted | 0.798 / 2.67 | 0.794 / 3.05 | +0.006588/+0.0004824 |
| 0 | adjacent_function_word / L9H7 | second_predicted | 0.610 / 1.45 | 0.617 / 1.20 | +0.004713/+0.006416 |
| 0 | adjacent_function_word / L8H1 | head_control | 0.396 / 0.29 | 0.433 / 0.26 | -0.001996/+0.002543 |
| 0 | adjacent_name_piece / L8H1 | failed_relation_control | 0.535 / 0.95 | 0.552 / 1.72 | -0.001146/-0.001181 |
| 1 | boundary_source / L9H8 | predicted | 0.767 / 12.47 | 0.745 / 8.59 | -0.0008952/-4.699e-06 |
| 1 | boundary_source / L8H1 | head_control | 0.502 / 0.79 | 0.487 / 0.41 | +5.332e-05/-9.356e-05 |
| 1 | adjacent_function_word / L7H8 | strongest_predicted | 0.761 / 2.23 | 0.759 / 2.24 | +0.01076/+0.001829 |
| 1 | adjacent_function_word / L9H7 | second_predicted | 0.652 / 1.69 | 0.672 / 2.45 | +0.003987/+0.00658 |
| 1 | adjacent_function_word / L8H1 | head_control | 0.379 / 0.32 | 0.423 / 0.23 | +0.0003956/-0.001733 |
| 1 | adjacent_name_piece / L8H1 | failed_relation_control | 0.510 / 1.26 | 0.516 / 2.73 | -0.002272/+0.001765 |

## Guardrails

This establishes finite causal importance of a relation-conditioned edge population, not a complete semantic description of a head. Candidate and control edges come from different examples, so matching removes declared address/frequency confounds but cannot equalize every latent context. KL measures output-distribution change regardless of whether the removed edge helped the observed target. The local probe vocabulary is weight-derived and fixed before this test. Reusing the prospective corpus for the finite follow-up makes this an intervention validation of frozen candidates, not a second independent semantic discovery split.
