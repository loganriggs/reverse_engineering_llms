## H429 — Fresh centered-reflection source-moment test

Fresh512 recipient documents and12 fresh paired source draws, conditional on the frozen512-document H413 donor pool, H421 Gaussian moments, and H410 factor partition. Full-document hashes were independently checked against10944 prior documents; overlap0. The old sampling region contained only130 unused eligible documents, so eligibility was extended to the entire same cached corpus before model outcomes. The new panel has31 documents from the old region and481 from the extension. This sampling change limits direct comparisons with older aggregate estimates. Total completed unique documents11456.

For each natural or Gaussian source h, also evaluate2*mu-h. Normalized-input change is linearized, the MLP retains its full bilinear product, and output scoring uses the fixed baseline Fisher metric. Directly compute quadratic, cubic and quartic energy coefficients for full response and core/added/union errors. No input norm/angle matching. Reflection symmetrizes centered odd moments but preserves second/fourth moments about the frozen mean. It can change empirical covariance about the empirical mean; moment diagnostics record that change. Gaussian vectors here are formed in double precision, unlike native-float Gaussian vectors in H425/H426; this is not a byte-identical replay.

Let N be core error + added error - union error - full energy, so S=N/E. Frozen four contrasts are Gaussian-minus-natural reflection-odd cubic N and E (times4), and reflection-even quartic N and E (times16), all divided by one common original-source mean quadratic energy. This common scale avoids cell-specific ratio denominators; it is not a mediation decomposition. Joint95% maxabs bands use4000 paired recipient-document/source-draw resamples.

| Contrast | Estimate | Joint95% interval | Sign resolved | Equivalent within±.02 |
|---|---:|---|---|---|
|odd_cubic_numerator|-0.002832|[-0.568106, +0.562441]|False|False|
|odd_cubic_energy|+0.075434|[-0.489840, +0.640707]|False|False|
|even_quartic_numerator|+0.007645|[-0.557629, +0.572918]|False|False|
|even_quartic_energy|+0.021064|[-0.544209, +0.586337]|False|False|

Numerical gates: {'reflection_center': True, 'coefficient_PSD': True}. Reflection-center relative error 5.41e-17; coefficient Gram determinant relative violation 0. Independent array aggregation reproduces all four points and joint intervals.

Interpretation limits: reflection-odd cubic terms contain both linear and cubic powers of the centered deviation when the source mean is nonzero. A significant reflection contrast would therefore not uniquely identify skewness. Reflection-even quartic terms contain constant, quadratic and quartic centered contributions; these do not isolate kurtosis. Finite empirical moment mismatch, recipient-dependent operators and the fixed donor pool remain alternatives. Statistical failures are preserved; this test cannot establish semantic units.

Next discriminating operation: directly evaluate centered sources z=h-mu. Because the cubic and quartic energy coefficients are homogeneous in h, C3(z) and C4(z) directly isolate their centered third/fourth contributions; source-mean and recipient-assignment controls remain necessary. A complementary algebraic check is to evaluate centered deviation scales lambda=1 and1/2 with both signs, around the same frozen mean. For a cubic energy coefficient, the reflection-odd part is lambda*A+lambda^3*B; O(1)-2*O(1/2)=3B/4 isolates the centered cubic contribution and removes the linear term. Analogous even differences at sufficient scales separate centered second/fourth contributions. This requires new computations, not interpreting current reflection contrasts as pure higher moments. Assess joint uncertainty before any targeted confirmation.

### Precision finding and revised next step

All four joint intervals include zero and extend beyond equivalence bounds. A post-primary precision diagnostic finds that the even quartic full-energy contrast sets the maximum bootstrap deviation in3105/4000 samples (77.6%). One recipient, document index199, contributes18.87% of pooled reflection-averaged natural quartic energy and24.55% of Gaussian quartic energy. This is a share of quartic energy, not a fraction of total prediction error or variance. No recipient is removed and no gate is changed.

Before a larger repetition or pure centered-moment follow-up, inspect this recipient's token sites, input/output normalization scales, baseline probability concentration and per-source sensitivity. Compare its per-site quartic coefficients with ordinary recipients and the previous panel. This distinguishes broad distributional effects from a small set of highly sensitive receiving contexts and directly bears on whether computational units must be context-dependent. The sample-region extension and possible recipient/source-assignment interactions remain live explanations; current data do not identify the cause.
