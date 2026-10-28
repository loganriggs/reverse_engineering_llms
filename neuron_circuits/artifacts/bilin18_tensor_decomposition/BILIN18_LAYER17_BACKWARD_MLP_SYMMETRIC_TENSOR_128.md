# Symmetric tensor quotient of the final-MLP backward path

The native Left/Right/Down CP factors are contracted into a symmetric output-adjoint × input × input tensor on 128D discovery-frozen input and output quotients. SVD modes are executed through native RMS normalization and the exact 32D upstream behavioral tangent.

## Locked hypotheses

- **Q0_symmetric_tensor_matches_projected_native_cp: supported.**
- **Q1_rank128_quotient_improves_over_rank64: not supported.**
- **Q2_rank128_input_output_quotient_is_useful: not supported.**
- **Q3_sixtyfour_tensor_modes_are_useful: not supported.**
- **Q4_sixtyfour_modes_preserve_the_quotient: not supported.**
- **Q5_symmetric_tensor_has_energy_rank64: not supported.**
- **Q6_rank128_tensor_transfers_ood: not supported.**

Projected-CP/tensor error: `1.21e-06`. Tensor energy R90: `92`; top share: `0.207`.

| Corpus | Component | Full quotient | Native fidelity K8/K16/K32/K64/K96/K128 | Within-quotient K8/K16/K32/K64/K96/K128 |
|---|---|---:|---:|---:|
| fineweb | prediction | 0.315 | 0.296/0.310/0.307/0.309/0.312/0.315 | 0.928/0.953/0.968/0.980/0.991/1.000 |
| fineweb | target | 0.336 | 0.307/0.317/0.319/0.326/0.330/0.336 | 0.927/0.950/0.970/0.980/0.992/1.000 |
| fineweb | margin | -0.056 | -0.215/-0.148/-0.175/-0.154/-0.124/-0.056 | 0.839/0.876/0.910/0.940/0.971/1.000 |
| wikitext | prediction | 0.429 | 0.384/0.393/0.403/0.412/0.420/0.429 | 0.941/0.954/0.968/0.982/0.992/1.000 |
| wikitext | target | 0.418 | 0.368/0.374/0.382/0.395/0.405/0.418 | 0.914/0.932/0.950/0.972/0.987/1.000 |
| wikitext | margin | -0.826 | -0.826/-0.757/-0.764/-0.794/-0.804/-0.826 | 0.857/0.894/0.912/0.943/0.975/1.000 |
| tinystories | prediction | 0.372 | 0.293/0.329/0.318/0.332/0.349/0.372 | 0.906/0.927/0.926/0.962/0.984/1.000 |
| tinystories | target | 0.524 | 0.399/0.412/0.389/0.431/0.469/0.524 | 0.881/0.861/0.835/0.917/0.963/1.000 |
| tinystories | margin | -0.103 | 0.077/0.110/0.068/0.068/0.053/-0.103 | 0.813/0.779/0.766/0.890/0.950/1.000 |

Full-quotient fidelity separates basis truncation from tensor-mode truncation. High within-quotient but low native fidelity means the symmetric factorization is compact only after choosing a lossy observer quotient.
