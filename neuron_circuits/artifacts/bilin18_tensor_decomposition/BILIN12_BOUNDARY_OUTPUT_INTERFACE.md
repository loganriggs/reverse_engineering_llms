# Natural boundary computation versus generic output interface

The exact per-example Layer-11 MLP write effect of changing the current token to newline is predicted by a paired mean semantic edit, an independently estimated natural newline prototype, and a calibrated period-token control.

## Locked hypotheses

- **N0_mean_semantic_write_is_reusable: not supported.**
- **N1_generic_newline_interface_transfers: not supported.**
- **N2_generic_interface_matches_paired_semantics: supported.**
- **N3_newline_semantics_beats_punctuation_control: supported.**
- **N4_scalar_calibration_generalizes_ood: not supported.**

Natural prototype counts: newline `308`, period `623`. Semantic/generic write cosine: `0.986`. Generic scale/fidelity on calibration: `0.707` / `0.642`; period scale/fidelity: `0.354` / `0.043`.

| Corpus | Examples | Semantic cosine/fidelity | Generic newline cosine/fidelity | Period control cosine/fidelity | Oracle Fisher energy |
|---|---:|---:|---:|---:|---:|
| fineweb | 252 | 0.878/0.734 | 0.869/0.714 | 0.273/0.053 | 2.830e+00 |
| wikitext | 253 | 0.787/0.412 | 0.777/0.376 | 0.183/0.030 | 1.676e+00 |
| tinystories | 240 | 0.875/0.721 | 0.863/0.711 | 0.278/0.052 | 2.233e+00 |

The oracle is only the final-MLP write path of the token edit, realized on the original recipient state. No claim is made that this path reproduces the edit's upstream residual or attention changes.
