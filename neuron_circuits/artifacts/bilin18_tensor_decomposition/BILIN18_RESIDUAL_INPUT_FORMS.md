# Residual-direction input quadratic pilot

Axes are frozen from previous semantic-residual discovery; the predicted target is the FULL native write projected onto those directions, not a label-subtracted residual. No target-token labels enter prediction. Panels are disjoint from axis discovery and each other, but have been used in unrelated older studies. This is local function compression, not end-to-end causal validation.

A scalar write is exactly xᵀQx for normalized pre-MLP x. Both spectra preserve the exact mean-state linear term and restore the discovery mean of omitted quadratic terms. Covariance coordinates use discovery inputs only. R2 is centered using the discovery scalar mean; it is not raw mean-dominated fidelity.

Locked gates: `{'exact': True, 'compact': False, 'nonlinear': True, 'specific': True}`

| Layer | Kind/axis | Linear R2 | Raw16 R2 | Cov16 R2 | Cov64 R2 | Cov16 quadratic-only R2 |
|---|---|---:|---:|---:|---:|---:|
| 1 | residual/0 | 0.188 | 0.683 | 0.878 | 0.960 | 0.849 |
| 1 | residual/1 | 0.320 | 0.552 | 0.780 | 0.936 | 0.677 |
| 1 | residual/2 | 0.118 | 0.850 | 0.952 | 0.985 | 0.946 |
| 1 | residual/3 | 0.451 | 0.699 | 0.836 | 0.945 | 0.701 |
| 1 | random/0 | 0.112 | 0.290 | 0.556 | 0.791 | 0.500 |
| 1 | random/1 | 0.218 | 0.378 | 0.566 | 0.793 | 0.445 |
| 1 | random/2 | 0.172 | 0.301 | 0.624 | 0.807 | 0.546 |
| 1 | random/3 | 0.245 | 0.395 | 0.645 | 0.841 | 0.529 |
| 8 | residual/0 | 0.067 | 0.600 | 0.665 | 0.812 | 0.641 |
| 8 | residual/1 | 0.035 | 0.644 | 0.775 | 0.896 | 0.767 |
| 8 | residual/2 | -0.012 | 0.531 | 0.641 | 0.855 | 0.645 |
| 8 | residual/3 | -0.214 | 0.280 | 0.405 | 0.713 | 0.510 |
| 8 | random/0 | -0.059 | 0.117 | 0.248 | 0.585 | 0.290 |
| 8 | random/1 | -0.057 | 0.112 | 0.255 | 0.545 | 0.295 |
| 8 | random/2 | -0.059 | 0.097 | 0.289 | 0.581 | 0.329 |
| 8 | random/3 | -0.003 | 0.199 | 0.375 | 0.618 | 0.377 |
| 12 | residual/0 | 0.522 | 0.808 | 0.858 | 0.941 | 0.703 |
| 12 | residual/1 | 0.010 | 0.586 | 0.738 | 0.888 | 0.736 |
| 12 | residual/2 | 0.026 | 0.467 | 0.629 | 0.830 | 0.620 |
| 12 | residual/3 | 0.336 | 0.749 | 0.812 | 0.910 | 0.717 |
| 12 | random/0 | 0.036 | 0.232 | 0.361 | 0.590 | 0.337 |
| 12 | random/1 | 0.039 | 0.147 | 0.309 | 0.543 | 0.281 |
| 12 | random/2 | 0.058 | 0.151 | 0.330 | 0.554 | 0.289 |
| 12 | random/3 | -0.052 | 0.087 | 0.274 | 0.558 | 0.310 |
| 17 | residual/0 | 0.906 | 0.995 | 0.999 | 1.000 | 0.990 |
| 17 | residual/1 | 0.832 | 0.999 | 0.999 | 1.000 | 0.997 |
| 17 | residual/2 | 0.864 | 0.997 | 0.998 | 1.000 | 0.989 |
| 17 | residual/3 | -0.577 | 0.991 | 0.996 | 0.999 | 0.998 |
| 17 | random/0 | 0.078 | 0.469 | 0.900 | 0.940 | 0.892 |
| 17 | random/1 | 0.021 | 0.262 | 0.934 | 0.963 | 0.933 |
| 17 | random/2 | 0.591 | 0.962 | 0.980 | 0.988 | 0.952 |
| 17 | random/3 | 0.835 | 0.926 | 0.970 | 0.984 | 0.821 |

This pilot does not establish semantic meaning, necessity, finite suffix fidelity, or compact upstream generation of the input coordinates. Four random axes are a descriptive control, not a calibrated tail-probability test. The linear term is a dense input covector and is explicitly charged as one additional read direction.
