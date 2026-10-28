# Finite test of residual-direction input decoders

Four fixed output axes per layer; intact orthogonal write remainder for primary causal tests. Report projected-only full-MLP coverage separately. Dynamic-effect denominator replaces coefficients with discovery means. Zero-component denominator also reported. No target labels enter decoder. Includes axes with weak individual recurrence: L8 axes2/3 and L12 axes1/2; do not describe all axes as independently stable.

Locked gates: `{'replay': True, 'finite_dynamic': True, 'nonlinear_value': True, 'binding': True, 'beats_fitted_affine': True}`; maximum replay error `4.86e-05`.

Primary dynamic fidelity compares native logits with the mean-coefficient intervention. The native orthogonal remainder is retained; these numbers describe the four-axis component, not reconstruction of the full MLP.

| Layer | Coefficient R2 rank64 | Dynamic logit linear/16/64 | Dynamic CE rank64 | Permuted64 dynamic logit | Four-axis-only full-MLP logit fidelity |
|---|---:|---:|---:|---:|---:|
| 1 | 0.901 | 0.426/0.816/0.929 | 0.940 | -0.723 | -0.296 |
| 8 | 0.818 | 0.051/0.676/0.841 | 0.834 | -0.403 | 0.080 |
| 12 | 0.950 | 0.461/0.828/0.934 | 0.950 | -0.967 | -0.041 |
| 17 | 1.000 | 0.811/0.999/1.000 | 1.000 | -1.058 | 0.958 |

## Fitted affine alternative

This control was added after the initial finite run. Ridge regularization was selected on row-grouped discovery validation; no finite-evaluation rows entered fitting.

| Layer | Affine coefficient R2 | Affine dynamic logit fidelity | Rank64 advantage |
|---|---:|---:|---:|
| 1 | 0.822 | 0.848 | +0.081 |
| 8 | 0.655 | 0.768 | +0.073 |
| 12 | 0.840 | 0.814 | +0.120 |
| 17 | 0.920 | 0.935 | +0.065 |

The decoder reads normalized pre-MLP state; explaining how upstream layers generate those read coordinates remains open. Each scalar uses up to rank+1 dense input covectors, and four separate decoders are not a shared rank-64 module. The single row permutation is a binding diagnostic, not a permutation-tail significance estimate.
