# Homogeneous evaluation and depth2–5 feasibility

The previous confirmation is complete. This stage validates an evaluator and tests numerical feasibility at larger input dimension and depth. It reveals a rare-direction bottleneck; it does not establish a deep-model learning advantage.

## Exact radial integration

Let f and g be homogeneous polynomials of the same degree p, and let L(x)=||f(x)-g(x)||², so L has degree2p. Write a standard Gaussian input as X=RU, with independent chi_d radius R and uniform unit direction U. With V=sqrt(d)U,

E L(X) = c(d,p) E L(V),
c(d,p) = 2^p Gamma(d/2+p) / (d^p Gamma(d/2)).

Thus c times an average of sphere losses is an unbiased Gaussian-MSE estimator. The same identity holds for a fixed linear covariance transform T: apply it to L(Tx). Biases, input-dependent normalization, unequal degrees and nonhomogeneous activations invalidate this argument. Fixed scalar/per-feature calibration constants preserve it.

This is conditional averaging of the radius. If A=L(V), then the ordinary Gaussian loss coefficient of variation satisfies

CV²[L(X)] = rho(d,p) (1+CV²[A]) - 1,
rho(d,p) = Gamma(d/2+2p) Gamma(d/2) / Gamma(d/2+p)².

The radial-integrated estimator has CV²[A]. Consequently an IID Gaussian sample mean has squared relative RMS error at least (rho-1)/N, regardless of angular structure, provided the mean loss is nonzero. This is a variance statement, not a high-probability bound. It follows directly by factoring independent first and second moments; no novelty claim is made.

At d16, rho for degrees4/8/16/32 is approximately4.14/76.2/153825/1.03e13. At d256 it is1.13/1.60/5.98/641.5. Naive Gaussian evaluation is intrinsically inefficient for deep homogeneous networks even before considering concentration over directions.

## Quartic validation

check_homogeneous_evaluation.py checks exact chi moments and coordinate-monomial identities for five dimensions and five degrees. It then evaluates three saved trained quartic students against their teacher, using16 independent batches of8192 samples, both identity and shifted covariance. Direct network factors are evaluated; exact population MSE comes from saved polynomial coefficients and independently constructed factorial Gaussian moments.

All six integrated means agree with exact values within the prespecified diagnostic tolerance (max of6 empirical standard errors and2.5% relative error). Observed batch SD is2.02–4.25 times smaller than direct Gaussian sampling. This finite check does not prove reliable confidence-interval coverage for deeper polynomials. Source, protocol and outputs are in results/homogeneous_evaluation.

## Deeper models and smoke runs

DeepMultiTask contains depth-1 bilinear trunk layers followed by a complete quadratic-product linear readout, giving degree2**depth. Shared trunks serve all tasks; separate trunks have per-task parameters. Shared and separate students start at identical functions. Independent teachers use independently seeded task trunks. Each layer is calibrated once on4096 independent sphere inputs with fixed per-feature scales; top outputs are calibrated to unit sample RMS. These constants remain fixed during training, preserving polynomial structure.

For depths2,3,4,5 at d16/h8 with6 source heads, checks cover independent NumPy forwards, exact homogeneity, matched initial functions, and cached/recomputed forward-gradient equivalence. Sixteen CPU fits span two teacher regimes and two student architectures at every depth. Each uses256 sphere training examples,100 Adam steps, lr.001, gradient clipping10, and8192 independent sphere evaluation inputs. All16 completed with finite training. This is one-seed, short-training feasibility, not a transfer study or fair optimized comparison. Sphere training changes Gaussian radial weighting and must not be silently pooled with earlier Gaussian training.

The normalized population Gaussian MSE equals the ratio of sphere population error to sphere target energy because c cancels. The reported sample ratio is not itself an unbiased estimator, and its reliability depends on concentration. Here it becomes unreliable:

|Depth|Degree|Observed error effective sample size out of8192|
|---|---:|---:|
|2|4|1556–3484|
|3|8|219–454|
|4|16|4.36–57.9|
|5|32|1.10–5.89|

Effective sample size means (sum loss)²/sum(loss²), a concentration diagnostic rather than a guaranteed precision measure. The depth5 separate-student/independent-teacher error ratio is about50, while other depth5 ratios are near1; none warrants a population performance conclusion. Calibration energy can itself miss rare directions. Numerical stability is not statistical reliability.

All saved aggregate metrics were recomputed from per-example error/target arrays. That audit is not an independent trained-model replay; only initialization/recomputation algebra has the independent forward checks. Artifacts are in results/deep_smoke. No GPU work ran.

## Next experiment

Before a larger sweep, validate directional importance sampling. Fit a proposal only on independent pilot inputs and teacher/initial-model geometry, retain a uniform-mixture component, and use exact density ratios on fresh draws. One tractable proposal is an angular central Gaussian: normalize N(0,Sigma); relative to uniform direction its density is det(Sigma)^(-1/2) (u^T Sigma^(-1)u)^(-d/2). A mixture of rank-one covariance stretches around pilot-discovered high-energy directions can target rare angular regions. Validate normalization and weighted moments against exact quartic cases before deep use; report importance-weight/error concentration and replicate variability. This proposal is not implemented or established by the present results.

If evaluation becomes reliable, freeze a multi-seed grid across input dimension, hidden width and depth, with shared/independent teachers, width/budget controls, whole-task holdouts and fixed transfer rules. Retain failures and limits. The full research goal remains active.
