

## H320 — Readout-weighted covered/uncovered interaction components

Same H319 reference spans and reused H315 queries. For feature matrix F and native-supervised labels Y=FW, covered output is kK⁺Y=(phi P_span)W, where K=FFᵀ. Uncovered output is target minus covered. Thus feature-orthogonal components can overlap or cancel after the output map W. Compute exact-kernel interpolation at relative eigen cutoffs1e-10/1e-6 and compare ridge.01/10 predictions. No output coefficients are newly optimized and no query-based selection.

| Target | Span | Covered energy / target | Uncovered energy / target |2×inner product / target | Output component cosine | Projection fidelity |
|---|---|---:|---:|---:|---:|---:|
|actual|specialist|0.554|0.564|-0.118|-0.105|43.62%|
|actual|mixed|0.316|1.045|-0.361|-0.314|-4.51%|
|angle|specialist|0.404|0.739|-0.144|-0.131|26.08%|
|angle|mixed|0.253|1.035|-0.288|-0.282|-3.55%|

| Target | Span | Projection fidelity cutoff1e-6 | Ridge.01 fidelity | Ridge10 fidelity |
|---|---|---:|---:|---:|
|actual|specialist|43.62%|43.11%|15.08%|
|actual|mixed|-4.51%|-1.79%|1.99%|
|angle|specialist|26.07%|25.20%|6.76%|
|angle|mixed|-3.55%|-2.22%|1.19%|

Locked gates: mixed_cancellation: PASS, mixed_uncovered_dominates: PASS. Cancellation requires mixed covered/uncovered cosine<=−.1 in both families; uncovered dominance requires mixed uncovered energy>=90% of target in both. These are diagnostic thresholds, not improved-model performance.

Mixed-span covered output energy is31.58% actual/25.27% angle of target energy, while uncovered energy is104.51%/103.55%. Their negative cross terms (−36.09%/−28.82%) bring the total back to100%. Consequently, retaining only the Euclidean-covered component yields−4.51%/−3.55% fidelity. Component cosines−.314/−.282 pass the cancellation gate. Matching specialists show milder cancellation and43.62%/26.08% projection fidelity, close to their ridge.01 performance. This directly demonstrates that omitted features supply important canceling contributions, not merely additional independent signal.

**Interpretation limits.** A low output projection fidelity despite moderate Euclidean feature coverage means the omitted feature component remains important under the readout. Negative output inner products indicate that covered/uncovered contributions cancel; this is not contradictory to their orthogonality in input-feature space. Ridge can reduce large canceling covered contributions, but cannot recreate missing feature directions. The decomposition distinguishes support geometry from output weighting; it does not identify semantic units or prove globally irreducible rank.

The identity kK⁺Y assumes exact linear labels in the cross-product feature representation. Here native response collection used float32 and kernel algebra float64; labels are effectively noiseless model supervision but contain numerical error. Cutoff sensitivity and ridge replay constrain, rather than eliminate, this concern. A truncated pseudoinverse describes the retained numerical subspace, not all exact-arithmetic directions. Output energy decomposition is conditional on the frozen approximate core/partner geometry and existing reference selection.

**Audit.** Energy identity E_target=E_covered+E_uncovered+2dot replays within relative1e-10; this verifies accounting but is partly algebraic because uncovered is defined by subtraction. More substantively, independently reconstructed ridge.01 prediction errors replay both H315 specialist and H317 mixed saved rows within relative1e-6. Source/reference labels and mixed archive hashes checked before use. Training interpolation residuals/ranks saved for19 spans. All jobs terminal; native model/compiler/base unchanged.

**Next discriminator.** Increase balanced reference coverage with nested samples while measuring BOTH feature coverage and readout-weighted residual energy. This tests whether additional directions repair the missing output map, rather than merely improving a geometric proxy. Use shared references and fixed normalization/regularization policy; include an equal-total-label specialist control when affordable. If gains require too many dual references, seek a structured primal tensor factorization. Fresh confirmation and downstream behavior remain necessary for any promoted model. Broad research goal active.
