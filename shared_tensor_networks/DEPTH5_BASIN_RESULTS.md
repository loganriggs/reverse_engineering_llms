# Depth5 controls: local learnability, poor initial alignment

Nine CPU fits distinguish smaller updates from initialization near a known solution. The positive control learns locally; lowering the learning rate alone does not recover useful prediction from the original random starts. This narrows the failure mode without proving a general optimization barrier.

## Frozen comparisons

Reuse the exact three depth5 teachers, normalized initial states and1024 importance-weighted training datasets from the coverage experiment. All models are d16,h8,6heads, shared trunks, degree32 in inputs. Train600Adam steps, clip10. Random-start controls use lr1e-4 or1e-5, compared with the prior1e-3 result. Teacher-near starts copy teacher parameters/buffers and add independent Gaussian noise with standard deviation1% of each parameter tensor's RMS, then use lr1e-4. This deliberately privileged positive control tests local optimization, not discovery of the teacher from data.

Fresh final evaluation proposals use64pilot axes,200steps, six16384-point importance batches. No test selection. All training inputs and their hashes are unchanged; full initial/final states and reproducible evaluation RNGs are archived.

## Results

|Teacher index|Random lr1e-4|Random lr1e-5|Teacher-near lr1e-4|
|---|---:|---:|---:|
|0|1.000142*|1.001679|0.000258|
|1|1.000189|1.003566|0.000672*|
|2|1.000162|1.002474|0.001686|

Values are relative population-error estimates;1 denotes zero prediction. Asterisk marks failure of the descriptive absolute-energy/error batchCV<10% gate. All9 fits complete600steps without nonfinite failure. The starred teacher-near error has12.2% batchCV; it supports a much smaller error than1 but not a highly precise estimate. No best-seed reporting.

Teacher-near empirical training losses drop from.00766/.0150/.01026 to4.75e-6/6.04e-6/.000414. Random-start losses converge near the weighted target energy. The final population alignments are approximately-.000009 to.000005 for random starts and.99924–.99987 for teacher-near starts.

Smaller updates did not rescue these random starts in600steps. Longer runs, different optimizers, initialization or curriculum remain possible; no impossibility theorem is implied. The positive control proves that this architecture/optimizer can refine a solution locally, not that generic initialization should find one.

## What the traces rule out and suggest

Original random training-set function cosines are only about-3.2e-6,-2.4e-6,+3.3e-7. There is very little initial predictive alignment despite matched target/output energy. In the smaller-rate controls, readout W norms fall to.276–.619 of their initial values, while trunk norms mostly stay within about1% of their starts. Prediction energy collapses much more than those norms suggest.

Therefore the previously derived simultaneous parameter-scaling path is not a literal description of these trajectories. It remains a valid flatness example. Actual behavior is consistent with suppressing initially unhelpful predictions while failing to align their narrow input response regions. Finite-data gradients, internal scale conditioning, feature interactions and initialization remain confounded; these traces do not establish a unique causal mechanism.

## Exact illustrative theory: degree suppresses alignment

Let unit vectors a,b have cosine rho, X~N(0,I), and define f=(a·X)^p, g=(b·X)^p. Their normalized Gaussian L2 inner product is

C_p(rho) = sum over k with p-k even of [choose(p,k)^2 k! ((p-k-1)!!)^2 rho^k] / (2p-1)!!,

with(-1)!!=1. Count k cross pairs in the Gaussian moment expansion; the remaining pairings are internal to each factor. The same normalized inner product holds on a uniform sphere by radial cancellation. This is a direct pairing calculation; no novelty claim.

|Input degree p|rho0|rho.25|rho.5|rho.9|
|---|---:|---:|---:|---:|
|4|.085714|.129464|.271429|.791109|
|8|.005439|.018425|.086014|.645554|
|16|2.14e-5|.000426|.008645|.428638|
|32|3.28e-10|2.32e-7|8.68e-5|.188732|

Thus moderate overlap of linear directions need not give appreciable function overlap at high degree. This is an exact rank-one-power example, not a formula for arbitrary deep bilinear networks. It supplies a falsifiable explanation for why energy normalization and better coverage alone can fail to initialize useful features. Exact rational pairing coefficients sum to1 at rho1. Independent256-point circle quadrature verifies all20 table values within1e-12; degree32 products have Fourier frequency at most64.

## Verification and next boundary

Independent NumPy replay regenerates every final evaluation from saved proposal axes and RNG seeds, checks initial/final training losses and verifies all frozen-data hashes. Maximum relative forward discrepancy9.49e-16. Teacher-near perturbations are reproduced exactly. All9 audited fits are terminal; no GPU work ran.

Next use a bounded initialization/curriculum control to manipulate early feature alignment, rather than continuing an unbounded learning-rate sweep. Keep privileged teacher access explicit. A useful outcome is a recipe that makes depth4/5 learn reliably enough to test sharing and held-out-task transfer; a negative result should become a documented depth limitation. Broader independent-teacher and parameter-budget scaling still remain. Preserve original exact-width/budget and whole-task holdout requirements when returning to those comparisons.

Artifacts: results/depth5_controls. Scripts: run_depth5_controls.py, audit_depth5_controls.py, check_power_alignment.py. The overall goal remains active.
