

## H297 — Equal-family loss and worst-family selection retain the original proxy

Repeat H296 rank32 partner fitting with identical core96, native cross coefficients, initializations, rates,400-update budget, minibatch seed, train/validation documents and payloads. Divide each sample error by its own actual/angle training-family mean target energy. Uniform sampling from equal-sized family pools gives equal expected weight to separately normalized family errors. Actual mean target energy is87,262.93; angle123,304.99, so H296 pooled energy loss gave angle about1.41 times actual weight. Select checkpoints by minimum(actual validation fidelity,angle validation fidelity), rather than pooled fidelity. This changes both loss and selection; it is a combined procedure comparison, not pure causal isolation of normalization.

| Run | Selected step | Worst-family validation cross fidelity |
|---|---:|---:|
|weighted_lr0.003|0|15.31%|
|weighted_lr0.01|0|15.31%|
|random0_lr0.003|400|13.28%|
|random0_lr0.01|100|10.52%|

Both weighted-start runs select step0. The winning weighted/rate.003 candidate is therefore the original proxy, up to QR rounding. Neither tested random-start candidate exceeds it. The selected projection matrix matches the proxy to relative<1e-5. This is a rejection of updates under the declared validation rule, not successful optimization or a certified minimax solution. One random run selects the final step, so convergence is not established.

Evaluate frozen partners in float64 on the reused H293 diagnostic panel, with the same source/angle payload seed. This confirms output behavior rather than selecting a model. The prior projection-gradient test covers unchanged partner algebra. No new gradient test is claimed for this scalar normalization change; the normalized loss and equal expected sampling weights were inspected. No fresh-panel or intact-behavior claim.

| Payload | Proxy | Pooled fit (H296) | Balanced selection |
|---|---:|---:|---:|
|actual|17.40%|14.89%|17.40%|
|angle|15.19%|17.40%|15.19%|

- Balanced selection versus H296 pooled fit, actual: 2.51pp fidelity change; paired95% [2.16,2.87]pp.
- Balanced selection versus H296 pooled fit, angle: -2.21pp fidelity change; paired95% [-2.95,-1.33]pp.

Locked gates: balanced_gain: FAIL, compact80: FAIL, retention: PASS, pooled_tradeoff_repair: FAIL. Balanced gain requires5pp versus proxy in both families; retention requires lower95%>=−1pp versus proxy; tradeoff repair requires>=2pp actual gain versus pooled fit while angle lower95%>=−1pp; compactness requires80% in both. Retaining step0 can pass retention without progress in fidelity. No thresholds relaxed.

**Hypothesis update.** The tested balanced procedure prevents adopting a harmful worst-family update but does not reveal a better compact partner subspace. Thus pooled energy weighting is not by itself a sufficient explanation for weak joint coverage under this optimizer/grid. This does not establish irreducible objective conflict: finite steps, nonconvex parameterization and gradient magnitudes could matter. Nor does it establish broad context dependence as a theorem. Dense compiled rank32 cross cost remains3,538,944 output coefficients for low fidelity.

**Next discriminator.** Audit the two separately normalized loss gradients at the fixed proxy. Accumulate training gradients in chunks, project to the partner-subspace tangent space, and measure norms and cosine. Construct a common-descent candidate when one exists, then perform a fixed small retracted line-search grid, reporting training and validation changes separately. Aligned or non-opposing gradients with an improving small step would implicate step-size/optimizer behavior. Nearly opposing gradients and no shared local descent would support local objective conflict, still not a global impossibility. This local diagnostic should precede another broad training sweep or higher rank.

All jobs terminal. Original model/compiler/base unchanged. Curves, normalizer energies, frozen partners, selections and diagnostic arrays retained under bilin18_l17_conditional_balanced_fit* and _eval*. Goal remains active.
