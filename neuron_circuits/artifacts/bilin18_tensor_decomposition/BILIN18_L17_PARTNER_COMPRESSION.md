

## H271 — Exact cross-core partner compression

H270 established that cross-core interactions reverse natural/Jacobian basis rankings, including fresh-document replication. H271 asks whether these cross-core terms admit a small shared partner span. Freeze both core32 subspaces from H269, orthonormalizing without changing their spans. Compare tensor-selected partners, discovery-data-weighted tensor partners and three nested Gaussian random-complement controls at widths16/32/64. Exact native-minus-compiled coefficients are used throughout; no readout is fitted.

Tensor selection diagonalizes the restricted Gram sum_i M(c_i)'M(c_i), M(c)v=B_E(c,v). Data weighting uses the first768 discovery documents, all nine actual full source edits. Write c_a,c_b for core coordinates and r_a,r_b for complementary vectors. The core metric is sum ||r_b−r_a||²(c_a+c_b)(c_a+c_b)' + ||r_a+r_b||²(c_b−c_a)(c_b−c_a)'. Factoring this metric gives weighted core directions whose tensor contractions define the partner Gram. This is an isotropic-partner sensitivity surrogate: it omits partner anisotropy and correlation between the two response terms. It is NOT exact response-loss minimization.

Evaluation uses the128 H270 replication documents, now reused, four sites and nine sources at scales0.1/1. Only actual source edits are evaluated here; no new angle-control or distribution-transfer claim. The main denominator is the full cross-core response energy. Separately report total residual response fidelity with the full mean-linear and within-core terms retained. These metrics differ because errors can correlate with omitted outside-core terms.

| Core | Method | Cross fidelity at16 /32 /64 | Total fidelity at32 |
|---|---|---:|---:|
|natural|tensor|35.27% / 42.83% / 50.47%|35.07%|
|natural|weighted|36.73% / 44.33% / 52.14%|35.59%|
|natural|random0|2.14% / 3.58% / 8.09%|18.17%|
|natural|random1|1.57% / 3.20% / 6.69%|18.07%|
|natural|random2|1.92% / 3.77% / 7.96%|18.30%|
|jacobian|tensor|21.23% / 31.04% / 40.14%|32.64%|
|jacobian|weighted|23.29% / 32.44% / 42.14%|32.62%|
|jacobian|random0|1.67% / 3.33% / 5.75%|24.67%|
|jacobian|random1|1.40% / 3.16% / 6.00%|24.64%|
|jacobian|random2|0.95% / 2.52% / 5.87%|24.53%|

Locked gates: compact32: FAIL, weighted_gain: FAIL, weighted_specificity: PASS.

Compact32 requires80% cross-response fidelity at both scales for either selected method/core. Weighted gain requires>=5 points versus tensor, positive paired lower CI, both cores/scales at32. Specificity requires the same against EACH random control. All outcomes are retained.

- s0.1, natural: weighted−tensor 1.40 points [1.04, 1.77] paired document-bootstrap95%.
- s0.1, jacobian: weighted−tensor 1.73 points [1.39, 2.09] paired document-bootstrap95%.
- s1.0, natural: weighted−tensor 1.50 points [1.15, 1.86] paired document-bootstrap95%.
- s1.0, jacobian: weighted−tensor 1.40 points [1.07, 1.74] paired document-bootstrap95%.

**Results.** At32 partners, weighted selection captures44.33% of the natural-core cross response versus42.83% for tensor and3.20–3.77% for randoms. For the Jacobian core the corresponding values are32.44%,31.04%,2.52–3.33%. At64 partners weighted coverage rises to52.14% and42.14%. Thus specific partner structure is strong, but neither tested selection achieves the80% compactness criterion. Data weighting gives small positive gains, below its5-point gate.

Weighted32 total residual fidelity is35.59% for natural and32.62% for Jacobian, versus their all-partner56.18% and51.80%. For Jacobian32, improved cross-term fidelity does not improve total fidelity (tensor32 total32.64%): correlated omitted terms matter. This tiny total-score difference is descriptive, not a significance claim. Do not substitute cross-component coverage for whole-response accuracy.

**Interpretation and next discriminator.** These selectors test a specific route to shared partners, not all possible low-dimensional factorizations. Spectral concentration in a tensor or surrogate metric is not necessarily response fidelity. If the selected spans remain weak, the next useful distinction is whether actual partner anisotropy/correlation defeats the surrogate or whether a shared span itself is inadequate. A direct empirical cross-response projection objective, fit on discovery data and checked against random controls and frozen test data, would address the former. Context-conditioned partner spans would test the latter, but must include routing/read costs. Avoid attributing a failed surrogate to fundamental irreducibility.

The evidence still concerns subspace interactions of one residual polynomial, not semantic units. Exact coefficient access and full mean-linear terms are diagnostic costs. At a fixed partner width the methods have the same number of core-partner product features, but basis construction differs and total implementation cost has not been optimized. Fresh-document confirmation is needed after selecting a useful method; the current panel cannot be called fresh again.

**Audits.** Maximum centered polynomial replay error 8.4e-12. Three targeted tests pass: full-complement partner response equals the cross term, core-metric covariance under rotations, exact quadratic partition and tensor contraction. Maximum basis overlap/orthogonality errors below3e-15. Full project suite not rerun. All runs complete; native weights and original compiler unchanged. Figure inspected. Broad research goal remains active.
