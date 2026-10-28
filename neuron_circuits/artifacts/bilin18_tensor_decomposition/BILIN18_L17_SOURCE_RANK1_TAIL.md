## H485 — Native source rank1 versus orthogonal tail

Before outcomes, refresh20 competing hypotheses and prioritize source dimensionality, source-tail sensitivity, interactions, receiver dependence and RMS/bilinear nonlinearity. The source basis is frozen from H463's calibration donors, disjoint from the64 evaluated donors. For each centered source h, decompose h=a b+t using the calibration's leading unit direction b and its Euclidean-orthogonal tail t. Preserve original amplitudes; no norm matching. Apply full h, rank1 a b and tail t separately to the native layer17 MLP, with both signed unit changes, on H484's already-used512 receiving contexts. Direct residual path remains fixed. Compare their native write changes and the sum of separate rank1/tail changes against the full native response. This is exploratory on reused receivers, without a learned canonical predictor or fresh confirmation.

The leading direction captures 97.625% of centered calibration source energy and 97.617% of the64 evaluated sources' energy. This geometric fact is now paired with a native behavioral test.

| Intervention/predicted response | KL recovery of full −/+ | Raw recovery of full −/+ | Own KL effect / full effect −/+ |
|---|---|---|---|
|rank1|95.989% / 95.048%|98.771% / 98.709%|101.588% / 98.604%|
|tail|0.539% / 0.933%|0.043% / -0.517%|4.104% / 4.944%|
|additive_response|99.931% / 99.945%|99.979% / 99.979%|101.690% / 99.498%|

KL recovery means1−KL(full native output || approximation)/KL(full native output || baseline), pooled over receivers. Own effect is KL(each intervention output || baseline) relative to full effect; it is a magnitude comparison, not recovery and not an additive share. Raw metrics use squared write changes. Since probability changes can cancel, own-effect ratios near or above100% do not imply more than full fidelity.

Rank1 alone recovers95.99%/95.05% of full probability effects and98.77%/98.71% of raw responses. Tail-only own KL effect is4.10%/4.94% of full, while tail-only recovery of full is only.54%/.93%. The latter prevents confusing a small effect with a faithful substitute. Adding the two separate write changes recovers99.931%/99.945% in KL and99.979% in raw response. Non-additivity is small in this pooled diagnostic at the tested amplitudes, not identically zero or proved small for each receiver.

Hypothesis update: strong descriptive support for source rank1 as a behavioral approximation on this pool; weaken source-tail dominance and large rank1/tail interaction as explanations of the difficult response prediction. The tail still matters at the few-percent probability-effect scale. The result favors studying how receivers transform the dominant source scalar into different response directions. It does not show that rank1 outputs themselves are rank1 across contexts, that the source has a particular semantic meaning, or that95% recovery is universal. It also does not directly measure what fraction of the learned predictor's error originates in the source tail; that would require a matched learned-prediction comparison.

The earlier~2% omitted output component (H481) and this~2.4% source tail are different objects. Their similar norm fractions must not be conflated. Most native response behavior is induced by the dominant source direction, yet a low-dimensional global output predictor can still struggle with receiver-dependent output geometry.

Numerical audits: source decomposition sums correctly and is orthogonal within tolerance; full native effect KL/raw arrays replay H484; all arrays finite. Save full, rank1 and tail source vectors, per-case own effects, errors to full, raw inner products and non-additivity. No new documents or donor calibration; total16832 tracked documents. No confirmatory intervals or gates introduced after inspection.

Next exploit the algebra of a bilinear MLP with RMS-normalized input. For a fixed source direction b and scalar amplitude a, the bias-free write is (N0+a N1+a² N2)/(d0+a d1+a² d2), where N0=D[(Lp)⊙(Rp)], N1=−D[(Lp)⊙(Rb)+(Lb)⊙(Rp)], N2=D[(Lb)⊙(Rb)], d0=mean(p²)+eps, d1=−2mean(p b), d2=mean(b²), with matrix orientation adapted to code. Subtracting the intact write leaves a combination of two receiver-dependent output vectors, with scalar rational coefficients. N2 is global for b; the response vectors also include the intact write through normalization. This is an algebraic prediction, not yet a numerically verified experimental result.

Next verify that exact two-vector formula against native rank1 interventions, then compare its linear tangent and normalization-frozen variants. This discriminates receiver-dependent cross-products from normalization/finite-amplitude curvature without another generic compression fit. Use the existing panel for diagnosis and retain fresh confirmation for learned claims. The broad canonical-form objective, observed-token CE comparison, coefficient learning and broader-head/layer transfer remain open.
