# Spectral and state-predictable atlas of dense late-MLP operators

The complete local bilinear derivative at each site is flattened as a 32×1,152 operator. Uncentered spectral modes are learned only on FineWeb. `Oracle` projects the already-known target into those modes and therefore measures coordinate compression, not prediction. `State` uses a validation-selected ridge map from the local normalized hidden state to the frozen spectral coordinates, then reconstructs the operator. This is the executable test.

## Locked hypotheses

- **P0_train_spectral_energy_is_accounted: supported.**
- **P1_rank64_operator_atlas_is_sufficient: not supported.**
- **P2_rank128_operator_atlas_is_high_fidelity: not supported.**
- **P3_rank128_state_predictor_is_sufficient: not supported.**
- **P4_state_prediction_nearly_matches_oracle_coordinates: supported.**
- **P5_state_predictor_transfers_ood: supported.**
- **P6_spectral_rank64_beats_512_native_channels: not supported.**
- **P7_spectrum_has_nonrandom_outliers: supported.**
- **P8_layer16_is_more_native_than_spectral_special: not supported.**

| Corpus | Layer | Oracle R64 / R128 / R256 | State R64 / R128 / R256 | Native512 | Spectrum R90 | Top/null share |
|---|---:|---:|---:|---:|---:|---:|
| fineweb | 13 | 0.308 / 0.412 / 0.544 | 0.302 / 0.400 / 0.524 | 0.249 | 249 | 0.0700/0.0109 |
| fineweb | 14 | 0.267 / 0.369 / 0.505 | 0.261 / 0.358 / 0.484 | 0.220 | 257 | 0.0508/0.0045 |
| fineweb | 15 | 0.439 / 0.519 / 0.622 | 0.433 / 0.509 / 0.604 | 0.396 | 224 | 0.2195/0.0277 |
| fineweb | 16 | 0.862 / 0.883 / 0.908 | 0.860 / 0.879 / 0.902 | 0.840 | 56 | 0.7266/0.0935 |
| wikitext | 13 | 0.314 / 0.415 / 0.548 | 0.308 / 0.403 / 0.528 | 0.260 | 249 | 0.0700/0.0109 |
| wikitext | 14 | 0.270 / 0.370 / 0.508 | 0.262 / 0.357 / 0.486 | 0.224 | 257 | 0.0508/0.0045 |
| wikitext | 15 | 0.446 / 0.525 / 0.628 | 0.439 / 0.515 / 0.609 | 0.404 | 224 | 0.2195/0.0277 |
| wikitext | 16 | 0.877 / 0.895 / 0.918 | 0.873 / 0.891 / 0.912 | 0.856 | 56 | 0.7266/0.0935 |
| tinystories | 13 | 0.349 / 0.449 / 0.574 | 0.343 / 0.438 / 0.555 | 0.265 | 249 | 0.0700/0.0109 |
| tinystories | 14 | 0.303 / 0.404 / 0.534 | 0.297 / 0.393 / 0.514 | 0.233 | 257 | 0.0508/0.0045 |
| tinystories | 15 | 0.484 / 0.566 / 0.663 | 0.479 / 0.557 / 0.648 | 0.421 | 224 | 0.2195/0.0277 |
| tinystories | 16 | 0.886 / 0.904 / 0.926 | 0.884 / 0.901 / 0.921 | 0.857 | 56 | 0.7266/0.0935 |

Oracle/state gaps are central: a small oracle atlas can merely rename each already-observed operator, whereas state prediction tests whether those coordinates form a reusable intermediate representation. Native-channel fidelity is copied from the separately frozen native-bank artifact and answers a different question: architectural execution sparsity.
