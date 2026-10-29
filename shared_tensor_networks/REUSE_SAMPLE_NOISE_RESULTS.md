# Finite-data reuse, noise, and conditioning

Completed162factorized fits plus54polynomial least-squares baselines. Fresh3teacher seeds, teacher/student groups1/2/4, nested Gaussian sample counts32/128/512, observation noise standard deviation0/.1. Quartic degree and3input dimensions remain fixed. Targets have exact Gaussian RMS1; independent student factors have frozen initial Gaussian RMS1.1000full-batch multiplicity-corrected SGD steps atlr.03, no clipping/regularization. Exact population evaluation uses noiseless teacher functions; noise is in training labels only. Factor matching uses the previously declared.99cosine/5%amplitude criterion.

Every fit completed with finite metrics. Independent coefficient Gaussian moments reproduce162population losses to maximum relative discrepancy1.732e-15. Saved-data training losses replay, all54least-squares normal equations pass, and initial/teacher normalizations pass. No discarded runs, GPU, or native crash in this batch. Three seeds are a small panel, not a definitive statistical comparison.

## Exact population MSE, median over3seeds

| Teacher factors | Noise SD | N | Student1 | Student2 | Student4 | Polynomial OLS |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 32 | 0.36398 | 0.61217 | 0.90692 | 6.5792e-30 |
| 1 | 0 | 128 | 0.0052281 | 0.016235 | 0.04276 | 3.6701e-30 |
| 1 | 0 | 512 | 0.00032517 | 0.0063418 | 0.016665 | 6.3337e-31 |
| 1 | 0.1 | 32 | 0.32479 | 0.5889 | 0.92969 | 0.26285 |
| 1 | 0.1 | 128 | 0.0093028 | 0.015258 | 0.038194 | 0.0082209 |
| 1 | 0.1 | 512 | 0.00042079 | 0.0058914 | 0.015915 | 0.00062432 |
| 2 | 0 | 32 | 0.11508 | 0.55989 | 0.8124 | 2.2341e-30 |
| 2 | 0 | 128 | 0.063945 | 0.53236 | 0.070149 | 4.011e-30 |
| 2 | 0 | 512 | 0.34481 | 0.058661 | 0.023937 | 1.2961e-30 |
| 2 | 0.1 | 32 | 0.13817 | 0.54722 | 0.81051 | 0.26285 |
| 2 | 0.1 | 128 | 0.064135 | 0.55384 | 0.074871 | 0.0082209 |
| 2 | 0.1 | 512 | 0.33618 | 0.05882 | 0.023647 | 0.00062432 |
| 4 | 0 | 32 | 1.0382 | 1.0259 | 0.79869 | 1.2342e-29 |
| 4 | 0 | 128 | 0.97071 | 1.0014 | 0.18701 | 6.6851e-30 |
| 4 | 0 | 512 | 0.98267 | 1.0027 | 0.046721 | 2.0121e-30 |
| 4 | 0.1 | 32 | 1.0278 | 1.0209 | 0.78084 | 0.26285 |
| 4 | 0.1 | 128 | 0.97024 | 1.0013 | 0.18701 | 0.0082209 |
| 4 | 0.1 | 512 | 0.98144 | 1.0026 | 0.045805 | 0.00062432 |

OLS recovers every noiseless quartic to numerical precision even atN32. Thus the remaining factorized-model errors cannot be explained by noiseless nonidentifiability from these samples alone. Optimization and empirical-design conditioning remain relevant; “enough samples for exact identifiability” is weaker than “enough for stable noisy recovery.”

For noisy one-factor targets atN512, the correctly shared one-factor student has medianMSE.000421 versusOLS.000624; this is a modest conditional result, not a broad sharing advantage. For two/four-factor teachers the nonlinear optimizer remains much worse thanOLS atN512. In the correctly specified families, strict factor recovery counts across3seeds are0/2/3 for one-factor teachers asNincreases, for both noise levels; two- and four-factor teachers have0/3 at everyNunder this training recipe. Do not substitute successful earlier population-training results for these fresh finite-data failures.

## Conditioning theorem and measured bounds

Let phi(x) contain the15homogeneous quartic monomials, Phi be the N-by15design matrix, A=Phi^T Phi/N, and G=E[phi(X)phi(X)^T] for standard GaussianX. For any quartic residual coefficient vector d,

empiricalMSE=d^T A d, populationMSE=d^T G d.

When A is positive definite, the generalized eigenvalues of (G,A) bound the ratio for every residual:

lambda_min*empiricalMSE <= populationMSE <= lambda_max*empiricalMSE.

This follows by substituting u=A^(1/2)d into the Rayleigh quotient. Full rank holds almost surely forN>=15Gaussian inputs: linearly independent polynomial evaluation functionals admit an invertible15point evaluation matrix, so its determinant is a nonzero polynomial whose zero set has measurezero. This is a noiseless identifiability statement, not a numerical or noise guarantee.

Measured worst-case population/empirical amplification ranges:

- N32:944–5888.
- N128:19.2–34.6.
- N512:4.08–6.10.

Generalized eigenvalue bounds independently match singular values of the population-whitened design. All81noiseless fitted residuals satisfy both bounds. Small empirical loss atN32can therefore leave substantial population error, even though exact noiseless interpolation identifies the true polynomial.

## Conditional noise risk

For unregularized full-rank OLS with independent labelnoise variance sigma², conditional onPhi,

E_noise[population error | Phi]=sigma²*trace[G(Phi^T Phi)^-1].

Proof: coefficient error=(Phi^T Phi)^-1 Phi^T epsilon, covariance=sigma²(Phi^T Phi)^-1, and expected squared function norm is the trace withG. This does not include noise in the test target.

AtnoiseSD.1, these conditional expected risks are.505–2.096 atN32,.00502–.01011 atN128,.000483–.000617 atN512. The observed median from one noise realization perseed differs from those conditional expectations; they are not interchangeable. Because the samePhiandnoise vector are paired acrossteacher families, OLS coefficient error and population error are mathematically the same acrossteachers. The matching observed errors are expected, not duplicated experiments.

## Decision

This batch does not establish broad statistical superiority of sharing. It reveals a practical optimizer bottleneck and quantifies the conditioning cost of few samples. Next bounded control: alternating exact least-squares updates for the fully untied product (each individual factor is linear with others fixed), using the same data and initialization, to test whether optimization explains its residual errors. Shared repeated-factor blocks are generally nonlinear and must not be mislabeled as linear least-squares subproblems. Keep exact coefficient recovery as a positive control, preserve noisy/generalization and factor-recovery metrics separately, and avoid an unbounded hyperparameter search.
