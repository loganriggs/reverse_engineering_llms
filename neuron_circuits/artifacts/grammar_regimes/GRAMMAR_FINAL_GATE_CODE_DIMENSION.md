# Causal dimension of the final bilinear gate code

PCA and coordinate subspaces are fit on discovery-template native post-product gate activations. Held-out gates are projected around the discovery mean, passed through the original down projection, and executed through final RMSNorm/unembedding.

## Locked hypotheses

- **G0_full_gate_reconstruction_is_exact: supported.**
- **G1_pure_gate_code_is_compact: supported.**
- **G2_arbitrary_gate_code_is_not_rank32_compact: not supported.**
- **G3_arbitrary_gate_code_is_rank64_usable: supported.**
- **G4_dense_gate_axes_beat_native_coordinates: supported.**
- **G5_arbitrary_gate_variation_exceeds_rule_variation: not supported.**
- **G6_gate_pca_beats_random_subspaces: supported.**
- **G7_template_transfer_is_stable: supported.**
- **G8_signal_gate_passes: not supported.**

| Regime | Seed | Status | PCA r16/r32/r64/r96 | coordinate r32/r64 | random64 mean/p95 |
|---|---:|---|---:|---:|---:|
| pure | 0 | all | 0.999/1.000/1.000/1.000 | 0.961/1.000 | 0.598/0.666 |
| pure | 101 | all | 0.996/0.999/1.000/1.000 | 0.898/0.974 | 0.603/0.661 |
| pure | 103 | all | 0.997/0.999/1.000/1.000 | 0.887/0.980 | 0.575/0.624 |
| mixed | 0 | unselected | 0.965/0.981/0.992/0.997 | 0.516/0.759 | 0.634/0.667 |
| mixed | 0 | selected | 0.968/0.978/0.992/0.996 | 0.498/0.758 | 0.608/0.641 |
| mixed | 101 | unselected | 0.963/0.977/0.990/0.996 | 0.536/0.790 | 0.634/0.666 |
| mixed | 101 | selected | 0.965/0.979/0.990/0.996 | 0.502/0.768 | 0.605/0.636 |
| mixed | 104 | unselected | 0.946/0.970/0.989/0.996 | 0.538/0.766 | 0.636/0.694 |
| mixed | 104 | selected | 0.956/0.974/0.990/0.996 | 0.523/0.757 | 0.598/0.643 |

This is an affine variation dimension, not a count of sufficient neurons: all methods retain the discovery mean gate. The coordinate comparison asks whether native channel axes describe variations more efficiently than a dense learned basis.
