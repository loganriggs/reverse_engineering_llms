# Fitted linear alternative to residual input quadratics

Regularization was selected on row-grouped discovery data. This is a post-pilot alternative-explanation test on the same held-out panel; no held-out rows entered fitting.

| Layer | Fitted affine R2 | Quadratic16 R2 | Difference |
|---|---:|---:|---:|
| 1 | 0.892 | 0.867 | -0.025 |
| 8 | 0.690 | 0.669 | -0.021 |
| 12 | 0.831 | 0.826 | -0.006 |
| 17 | 0.890 | 0.999 | +0.109 |

Locked two-early/middle-layer +0.10 advantage gate: False.

The exact tangent at the mean is not the best affine predictor. A large advantage over that tangent alone does not establish nonlinear indispensability. The separate finite tests compare rank64 quadratics against these frozen affine fits.
