

## H269 — Core coverage grows gradually; natural PCs outperform Jacobian cores with all partners

H268 located most missing response outside the selected16-direction core. H269 separates core size and selection from partner selection and fitted-coefficient error. For each nested core8/16/32/64/128, retain the exact full mean-linear response and every quadratic term touching the core. The omitted term is precisely the quadratic change entirely outside that core. This is an oracle structural diagnostic using native weights, not a new executable compact predictor.

Natural PCs are the frozen H259 basis. Residual-Jacobian directions recreate H266's covariance from eight Gaussian output probes per site on the first768 discovery documents only; its first32-dimensional span agrees with the previous basis to numerical precision (normalized overlap0.999999999996). All widths fixed before outcomes. Evaluate the same reused128 H259 documents, four sites, nine source heads, scales0.1/1 and norm/angle-matched same-head controls. No validation or test observations select the bases. Full signed outside/retained overlaps and per-document energies are saved. No fitted readout is involved.

| Core width | Natural PCs, small/full actual | Jacobian, small/full actual | Natural/Jacobian full angle |
|---|---:|---:|---:|
|8|22.94% / 22.55%|23.22% / 24.69%|27.20% / 29.29%|
|16|39.34% / 38.96%|33.37% / 34.29%|37.71% / 36.10%|
|32|57.20% / 56.90%|51.26% / 51.17%|50.61% / 44.75%|
|64|69.67% / 69.45%|62.69% / 62.57%|60.69% / 54.23%|
|128|79.50% / 79.27%|72.27% / 72.13%|70.12% / 64.66%|

**Both locked gates fail.** Neither family reaches80% at width32; both families already exceed50% by width64. The result occupies the middle between the proposed alternatives: substantial but incomplete, gradually increasing coverage. Do not reinterpret failure of both gates as evidence for either endpoint or lower thresholds after seeing results.

Natural-minus-Jacobian full-edit gains at widths16/32/64/128 are4.67/5.74/6.88/7.14 percentage points. Paired document-bootstrap95% intervals are[1.14,8.76], [3.53,8.14], [5.41,8.40], [6.03,8.31]. These basis contrasts were summarized after outcomes; they are descriptive uncertainty estimates, not additional preregistered gates. Width8 difference is−2.14[-6.12,1.86]. Full and small edits broadly agree. Angle controls show the same basis ordering at widths>=16, with lower coverage than actual responses at larger widths.

**Why this changes the next question.** H266's fitted within32 products favored Jacobian directions (41.65% versus37.87%). Exact all-partner accounting now favors natural32 (56.90% versus51.17%). These comparisons differ in retained interactions AND coefficient fitting, so the reversal cannot yet be attributed solely to either. It does show that a basis ranking from one feature family need not transfer to another. The next useful factorial should hold exact coefficients and test population fixed while crossing core family with within-core-only versus all-partner interactions. That directly tests whether natural PCs have useful couplings to directions outside the core. If the ranking reversal is absent under exact within-core coefficients, coefficient estimation/correlations become the next target.

**Limits and cost.** All-partner terms still read the full1152-dimensional input; a width128 core is not a128-read implementation or a parameter-matched fit. Exact truncation fidelity is not an upper bound on fitted readouts. The curves do not prove intrinsic dimension, a neuron ontology, semantic units, or downstream behavioral repair. They are conditional on the frozen mean, two discovery-derived basis families, source perturbations and reused corpus panel. A fresh document-disjoint replication is still required for any selected representation, followed by downstream and distribution-transfer checks.

**Audits.** Exact centered polynomial replay relative error9.06e-12. Three targeted tests pass: core-complement identity/full-core endpoint, six-term replay with offsets, and residual pullback. Full project suite not rerun. Figure inspected. All jobs terminal; native weights and original compiler unchanged. Saved bases, config, raw document arrays and report permit reproduction. Broad research goal remains active.
