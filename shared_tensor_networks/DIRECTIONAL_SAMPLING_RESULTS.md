# Directional importance sampling: validated, with a calibration warning

This stage makes deeper experiments more feasible. It does not yet compare trained deep networks or establish scalability of the sharing advantage.

## Proposal and unbiased estimator

For a unit direction u and positive-definite covariance Sigma, normalizing a Gaussian N(0,Sigma) gives density relative to uniform sphere measure

q_Sigma(u)=det(Sigma)^(-1/2) (u^T Sigma^(-1)u)^(-d/2).

This follows by integrating r^(d-1) exp(-r² u^T Sigma^(-1)u/2) over positive r. We use Sigma=I+tau vv^T with unit axis v. Determinant=1+tau and inverse=I-tau/(1+tau) vv^T. Sampling stretches a standard Gaussian by Sigma^(1/2), then normalizes it. A20% uniform component plus an equally weighted mixture of these proposals ensures q>=.2 and weights1/q<=5.

For any integrable angular loss A, E_q[A/q]=E_uniform[A]. Directions are sampled from the actual mixture, and the full mixture density is used for every sample. The implementation does not divide by the sampled component's density and does not self-normalize weights. Axis optimization uses separate pilot points and is frozen before evaluation. The estimator is conditionally unbiased given the pilot. Bounded importance weights do not guarantee low variance when the loss itself is sharply concentrated.

The pilot objective is log(teacher energy + INITIAL student squared error), optimized over directions. This is an evaluation proposal, not training supervision or a recovered computation graph. Future trained students need their own independently fitted proposals or a proposal covering their residuals.

## Validation

Analytic controls use unit-sphere x1^(2p), with known expectation Gamma(p+.5)Gamma(d/2)/(Gamma(.5)Gamma(d/2+p)). Twelve batches of8192 samples:

|Dimension, degree|Weighted mean / exact|Uniform mean / exact|
|---|---:|---:|
|8,4|0.99946|0.99398|
|16,16|1.00493|1.08149|
|16,32|1.01313|0.15803|

Normalization E_q[1/q] was checked. An independent full-matrix determinant/inverse implementation verifies the optimized rank-one density formula; maximum relative discrepancy for the refined mixture was4.05e-15.

An additional actual depth5, degree32 network in d2 admits an exact circle oracle: squared outputs/errors have maximum Fourier frequency64, so equally spaced N>64 quadrature is exact in exact arithmetic. N256 and N512 agree to1e-12. Exact teacher energy/error are1.0/1.959547; importance estimates over16batches are1.004726/1.965953. This checks multilayer code against a deep exact oracle, not only monomials. It does not prove finite-sample precision in d16.

## d16 depth4/5 refinement

Fixed shared teachers and initial students use h8 and6 heads, matching the previous smoke-study initialization. Initial proposal:16 optimized pilot axes,80 steps, stretches16/64,12batches of8192 evaluation directions. Depth5 error batch CV remained54%, so a second protocol was recorded before new outputs:64 fresh pilot starts,200 steps, stretches32/128,12 independent batches of32768 directions. Pilot and evaluation seeds differ from development; teachers are the same. This is methodological refinement, not independent teacher replication.

|Depth|Teacher sphere energy|Initial error|Energy batch CV|Error batch CV|Median error ESS|
|---|---:|---:|---:|---:|---:|
|4|8.6873|10.9518|2.38%|3.25%|572|
|5|233.9699|264.1762|3.96%|4.28%|570|

Both meet the descriptive predeclared CV<10% gate. This gate measures observed repeatability, not guaranteed accuracy or confidence coverage. Weight means ranged.9896–1.0124. Saved per-example weighted values independently reproduce all24 batch aggregates. Exact radial factors can convert these sphere energies to Gaussian energies; they cancel in population relative-error ratios, but ratios of estimates are not exactly unbiased.

## Calibration changes the next experiment

Teachers had been normalized to unit output energy on4096 uniform sphere calibration points. The refined estimates show true sphere energy around8.7 atdepth4 and234 atdepth5 for these fixed teachers. Thus finite, stable activations and apparently unit calibration RMS did not ensure comparable task scale. This affects gradients, clipping, and any across-depth performance comparison.

Before scaling, estimate each teacher's and student's initialization energy from independent importance-weighted calibration data, freeze the resulting scalar/head scales, and evaluate using fresh draws. Do not calibrate using the evaluation estimates in this report. Preserve uniform/sphere/Gaussian training distinctions: changing training sampling requires explicit loss weights if the population objective is meant to stay fixed. Rare-direction training coverage is itself a likely learning bottleneck and should be measured rather than mistaken for optimization failure.

## Next bounded stage

Build independent importance-weighted calibration and a reusable chunked evaluator around frozen proposals. Run paired depth2–5 training controls, including ordinary sphere training and importance-sampled training with exact weights, to separate missing informative examples from optimizer difficulty. Retain independent teacher regimes, matched-width/budget controls and whole-task holdouts for the subsequent scaling grid. Save full trained states and data seeds for independent replay; the earlier smoke archive only supported aggregate replay. Use CPU first; no GPU work has begun. The complete toy-research goal remains active.

Artifacts: results/directional_sampling, results/directional_refinement. Scripts: directional_sampling.py, check_directional_sampling.py, refine_directional_sampling.py, audit_directional_sampling.py. All producers and audits are terminal.
