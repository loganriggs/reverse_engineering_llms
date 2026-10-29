# Exact linear-readout elimination: verified development result

Three fresh d6/h3 teachers, six source heads, 256 Gaussian inputs, noise SD .05, same initial features and optimally fitted initial readout. Joint and reduced TRF each receive a 100-function-evaluation cap. No restarts; all cases retained. Equal caps are not equal compute. These are development cases, not a confirmation study.

| Seed | Joint exact population MSE | Reduced exact population MSE |
|---|---:|---:|
| 231101 | .08772935 | .06303162 |
| 231102 | .0001577421 | .0001577420 |
| 231103 | .05891156 | .0003132322 |

The reduced method substantially helps one case, modestly helps another, and essentially ties the third. Both methods exhaust the cap on the first case; joint also exhausts it on the third. No broad superiority or circuit-recovery claim follows. Whole-task transfer is not measured in this pilot.

## Mathematics and correctness

Write Z=[Phi(theta);sqrt(lambda)I], Yaug=[Y;0], U=argmin ||ZU-Yaug||², and r=ZU-Yaug. When Z has full column rank, U is unique and smooth locally. Differentiating Zᵀr=0 gives

(ZᵀZ)dU = -dZᵀr-Zᵀ(dZ)U,

dr = (dZ)U+Z dU.

Since Zᵀr=0, the reduced objective derivative is rᵀdr=rᵀ(dZ)U: the envelope gradient can freeze the fitted readout, but the residual Jacobian generally cannot. Omitting Z dU produces the wrong reduced residual Jacobian for least-squares optimization. This is elementary variable-projection calculus, not a novelty claim.

Implementation solves readouts by augmented least squares and derivative systems through QR triangular factors. Six d3/h2, d6/h3, d8/h4 fixtures with penalties0/.001 agree with independent differentiable normal-equation solves to maximum relative Jacobian error2.81e-14; central finite differences agree within4.78e-10. Envelope gradients agree within9.24e-14. These tests establish the implementation in full-rank fixtures, not uniform stability near rank loss. Rank deficiency raises an error.

Independent checkpoint audit reconstructs polynomial coefficients and exact Gaussian population losses for all six fits. Reduced fitted predictions agree with independent QR projection to relative error below1.25e-13. Training-loss coefficient replay passes. The original first attempt failed an absolute normal-equation tolerance; it is retained in aborted_absolute_normal_check. The rerun changed only the diagnostic to a scale-normalized residual, not training. Largest reduced absolute normal residual .00833 corresponds to relative5.55e-15; QR prediction replay independently confirms optimal fitted predictions.

## A new issue to control before scaling

With lambda=0, replacing Phi by Phi T for any invertible matrix T preserves its column span and hence the reduced objective. Bilinear feature rescaling induces such a transformation. Therefore readout elimination leaves flat scale directions; it does not select balanced hidden weights. In this pilot maximum absolute hidden parameters reach6.03,39.59,459.11 in the three reduced fits. Scale growth is observed; this alone does not establish which optimizer mechanism caused it.

Adding a penalty solely on U does not fix this: under Phi->c Phi, U->U/c, predictions stay fixed while lambda||U||² shrinks. Hidden quadratic-product features can realize positive common scaling by rescaling their factors. Thus readout-only ridge has a scale escape if hidden scale is unconstrained. A fixed normalized-feature parameterization or joint scale-controlling penalty is needed before interpreting ridge as comparable regularization. Raw feature ridge must not be silently added as a numerical fix.

Next: define a smooth unit-factor/normalized-feature control away from zero, independently verify its reduced derivative, then freeze a paired joint/reduced comparison with whole-task holdouts and exact population evaluation. Preserve the failed/capped cases. Do not infer recovered circuits from improved source prediction.
