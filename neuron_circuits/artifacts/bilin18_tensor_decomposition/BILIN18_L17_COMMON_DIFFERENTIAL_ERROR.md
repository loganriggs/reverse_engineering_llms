## H524 — Shared and fit-dependent errors; averaging versus pooling

For every frozen H523 half-fit pair, define reconstruction errors e_common=(e_left+e_right)/2 and e_diff=(e_left−e_right)/2. Use the full-vector derivative through the residual bypass, MLP and final probability readout at each actual receiver. With Q(e)=one-half e^T F e and cross=e_common^T F e_diff, left error is Q(common)+Q(diff)+cross and right error is Q(common)+Q(diff)−cross. Their average is Q(common)+Q(diff). Pairwise local prediction KL is4Q(diff). The midpoint reconstruction has error e_common.

This is an algebraic decomposition conditional on the two fits, not a population bias/variance decomposition or identification of semantic features. The common error need not be irreducible; different data, model families or objectives may change it. We evaluate actual finite prediction KL for midpoint reconstructions on the same two reused panels, and compare with archived individual fits and H522's pooled3,072-source rank-k fit. No maps are refitted.

| Method | Rank | Differential fraction of local error | Finite midpoint error reduction | Pooled rank-k error / midpoint error | Finite midpoint gain / local prediction |
|---|---:|---:|---:|---:|---:|
|PCA|36|2.23%–2.62%|2.26%–2.66%|0.9972–1.0089|1.0049–1.0100|
|PCA|144|8.03%–8.60%|8.08%–8.62%|1.0064–1.0247|0.9994–1.0053|
|PCA|576|25.03%–25.47%|25.04%–25.45%|1.0642–1.0906|1.0001–1.0014|
|Fisher|36|2.22%–2.42%|2.24%–2.44%|0.9996–1.0066|0.9998–1.0133|
|Fisher|144|8.26%–9.05%|8.25%–9.10%|1.0137–1.0388|0.9994–1.0063|
|Fisher|576|23.71%–25.68%|23.73%–25.69%|1.0569–1.0836|1.0002–1.0028|

Ranges cover all four partitions and both panels, not confidence intervals. At rank36 only about2.2–2.6% of average local error is differential; at144 about8.0–9.0%; at576 about23.7–25.7%. Thus H523's near100% pairwise-disagreement/error ratio at576 corresponds to roughly25% removable-by-averaging error, not100%. About three quarters remains common to each pair.

Finite midpoint gains closely follow the local differential-energy prediction: their ratio is about0.999–1.013 across all tested cases. Finite symmetric pair KL is about0.986–1.002 times four local differential energy. This supports a largely local explanation for these fit-to-fit differences at the tested perturbations; it is not a global linearity claim.

Pooling data at the original rank achieves almost the midpoint's fidelity. Pooled rank36 error ranges from about0.3% lower to0.9% higher than midpoint error; at144 it is0.6–3.9% higher; at576 it is5.7–9.1% higher. The midpoint can use up to min(2k,1023) dimensions:72,288,1023 respectively. These are upper bounds, not measured numerical ranks. The pooled comparator retains36,144,576. Consequently, midpoint gains are not free compression or a matched-rank advantage, and the small difference cannot establish equivalence. Both procedures use the same total source pool through different constructions.

The findings support a shared large computation plus sample-sensitive corrections within this approximation family. They do not identify a stable semantic core or unique fundamental units. Better calibration and averaging both reduce some error, but the common remainder remains substantial and depends on the objective and rank.

Audits: frozen inputs hash checked; direct local derivative evaluations for left/right/common/differential errors obey both reconstruction identities to at most5.29e−6 relative per-case tolerance (float32 derivative arithmetic with double probability-weighted contractions). All arrays are finite, and saved rows reproduce summary statistics. Full residual vectors are used, including components outside the fixed supported basis. Native half-fit and pooled finite values come from hash-checked H523/H522 artifacts. No new documents:24,064. No CE or task-loss claim is made.

Next H525 shifts to the user's objective hierarchy. At the frozen pooled3,072-source Fisher rank36 map, introduce36 multiplicative coordinate gains initialized to1, preserving the read/write directions and rank budget. Compute mean gradients of raw source reconstruction error, full-path prediction KL, and observed-token CE with respect to those same gains on H512 and H521 separately. Report cosine/sign alignment between objectives and cross-panel reproducibility, with split-document uncertainty where feasible. Test prescribed small steps along normalized negative-gradient directions, using gains=1−alpha*sqrt(36)*g/||g|| at alpha0.01/0.03, on both panels; report all directions and scales with no selection. A direction derived on one panel evaluated on the other tests local transfer, but both panels remain reused diagnostics. Verify finite differences against gradient predictions and preserve the frozen-model endpoint. The purpose is to distinguish a reproducible objective conflict from noisy CE estimates before attempting CE optimization. Coordinate gains need not preserve idempotence, which must be disclosed; the source representation remains rank at most36. No claim of a new canonical form, task improvement, or fresh generalization follows without subsequent fitting and confirmation.
