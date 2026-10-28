# Symmetric tensor quotient of the final-MLP backward path

The native Left/Right/Down CP factors are contracted into a symmetric output-adjoint × input × input tensor on 64D discovery-frozen input and output quotients. SVD modes are executed through native RMS normalization and the exact 32D upstream behavioral tangent.

## Locked hypotheses

- **Q0_symmetric_tensor_matches_projected_native_cp: supported.**
- **Q1_rank64_input_output_quotient_is_sufficient: not supported.**
- **Q2_sixteen_tensor_modes_are_useful: not supported.**
- **Q3_thirtytwo_tensor_modes_are_sufficient: not supported.**
- **Q4_symmetric_tensor_has_energy_rank32: not supported.**
- **Q5_symmetric_tensor_transfers_ood: not supported.**
- **Q6_tensor_modes_beat_native_margin_channels: not supported.**

Projected-CP/tensor error: `8.91e-07`. Tensor energy R90: `38`; top share: `0.323`.

| Corpus | Component | Full quotient | Native fidelity K4/K8/K16/K32/K64 | Within-quotient K4/K8/K16/K32/K64 |
|---|---|---:|---:|---:|
| fineweb | prediction | 0.197 | 0.143/0.170/0.191/0.192/0.197 | 0.724/0.929/0.962/0.987/1.000 |
| fineweb | target | 0.217 | 0.110/0.180/0.208/0.210/0.217 | 0.682/0.937/0.968/0.987/1.000 |
| fineweb | margin | -0.033 | -0.555/-0.205/-0.152/-0.112/-0.033 | 0.156/0.827/0.893/0.961/1.000 |
| wikitext | prediction | 0.309 | 0.234/0.276/0.287/0.296/0.309 | 0.817/0.947/0.966/0.988/1.000 |
| wikitext | target | 0.283 | 0.193/0.243/0.252/0.263/0.283 | 0.759/0.921/0.951/0.981/1.000 |
| wikitext | margin | -0.670 | -1.057/-0.768/-0.647/-0.664/-0.670 | 0.435/0.856/0.914/0.965/1.000 |
| tinystories | prediction | 0.273 | 0.073/0.194/0.243/0.241/0.273 | 0.503/0.899/0.949/0.974/1.000 |
| tinystories | target | 0.437 | 0.093/0.315/0.348/0.362/0.437 | 0.441/0.888/0.894/0.943/1.000 |
| tinystories | margin | -0.248 | -0.453/-0.031/0.049/-0.049/-0.248 | 0.452/0.866/0.878/0.937/1.000 |

Full-quotient fidelity separates basis truncation from tensor-mode truncation. High within-quotient but low native fidelity means the symmetric factorization is compact only after choosing a lossy observer quotient.
