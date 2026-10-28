

## H295 — The96-span cross response is broad under proxy-weighted partner selection

H294 found large span/complement interactions in the conditional96 surrogate's remaining error. Hold that96-dimensional core and discovery mean fixed. Using first768 discovery documents, build the existing core_metric proxy from actual and angle full removals across nine heads. It weights core sums/differences by squared outside payload norms; it does not retain all empirical input/output cross correlations. Contract native-minus-original-compiler bilinear tensors with these weighted core directions, form the partner Gram in the core's orthogonal complement, and retain leading128 eigenvectors. Nested ranks8/16/32/64/128 plus two fixed random orthonormal partner bases are evaluated.

The refined residual base has its quadratic read span inside the96-core and the adapter is linear, so their span/complement quadratic cross terms vanish. This permits the native-minus-original-compiler cross helper here. Cross target energy replays H294's full-remaining-error cross energy within1.34e-15 relative across every head/payload. Partner overlap and orthogonality errors are below1e-10. Basis construction uses discovery only; evaluation reuses H293/H294128 documents and identical payloads. This is diagnostic, not fresh confirmation or intact replacement. Exact native cross coefficients are used for each projected partner span; no learned output readout. All-partner oracle cross fidelity is1 by definition, not a model score.

| Partner rank | Actual cross fidelity | Angle cross fidelity | Dense bilinear output coefficients |
|---|---:|---:|---:|
|8|8.45%|7.05%|884,736|
|16|12.19%|10.35%|1,769,472|
|32|17.40%|15.19%|3,538,944|
|64|25.50%|22.27%|7,077,888|
|128|37.17%|32.75%|14,155,776|

- actual, rank32: paired document-bootstrap95% [16.19,18.75]% cross fidelity.
- actual, rank128: paired document-bootstrap95% [36.11,38.30]% cross fidelity.
- angle, rank32: paired document-bootstrap95% [14.12,16.35]% cross fidelity.
- angle, rank128: paired document-bootstrap95% [31.92,33.67]% cross fidelity.

Locked gates: compact32: FAIL, compact128: FAIL, specificity32: PASS. Both compactness gates require80% for both payload families. Specificity requires weighted rank32 to exceed both random rank32 controls by5 percentage points. It passes; random rank32 fidelities are about3%. No thresholds relaxed.

**Cost and interpretation.** Rank32 already implies3,538,944 dense output coefficients; rank128 implies14,155,776, before core/partner reads, base, linear/within terms and any other correction. The latter is close to the native MLP's15,925,248 weight scalars while recovering only about a third of the isolated cross response. These counts are for a straightforward compiled96xrank bilinear expansion, not a proven minimum storage cost; factorization could change costs. Nothing is compiled or promoted in this experiment.

Proxy-selected directions are better than random, so the cross interaction is structured, but they do not furnish a small high-fidelity partner set. Proxy spectral mass is only12.0% at32 and28.9% at128; it is not identical to empirical response fidelity and should not be presented as a distribution-wide lower bound. Poor compactness may reflect broad context dependence, proxy mismatch, or both. The result does not prove that every partner basis of these ranks fails, nor identify independent semantic units. Complement-only remaining error from H294 is unresolved even if cross compression succeeds.

**Next discriminator.** Optimize the empirical cross-response objective directly at a fixed modest partner rank, starting from both proxy and random bases, while holding core96 fixed. Use discovery train/validation documents, include actual and angle payloads, and preserve held-out evaluation. A large improvement over the proxy at equal rank would implicate selection mismatch; failure across well-behaved starts would strengthen—but not prove—the distributed-dependence account. Compare training and validation gaps and actual/angle tradeoffs. Do not increase rank merely to pass a threshold, and keep dense/factorized parameter costs explicit. A compact candidate would still require fresh and intact behavioral evaluation before promotion.

All jobs terminal. Frozen partner bases, proxy spectra, raw document errors and intervals retained under bilin18_l17_conditional_partner_rank*. Native model/compiler/base unchanged. Goal remains active.
