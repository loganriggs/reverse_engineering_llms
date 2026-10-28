

## H321 — Nested shared reference coverage,3072→6144 examples

Start with exact H317 mixed3072-reference set. Add one distinct group per existing reference site: second group=(original+5)mod18. All768 documents and3072 input positions stay fixed; each now supplies two different head/payload examples. Added assignment is a deterministic locked cyclic shift, not a random-seed sweep. Fit the same shared interaction kernel with original3072 mean-diagonal scale held fixed. Ridge coefficient10×N/3072 is10 then20, preserving the regularization coefficient in average-loss primal form. No new query tuning. H315 panel reused diagnostically.

| Target | References | Feature coverage | Ridge fidelity | Unregularized projection fidelity | Output component cosine |
|---|---:|---:|---:|---:|---:|
|actual|3072|35.34%|1.99%|-4.51%|-0.314|
|actual|6144|51.43%|2.59%|6.09%|-0.284|
|angle|3072|33.39%|1.19%|-3.55%|-0.282|
|angle|6144|47.55%|1.68%|5.34%|-0.244|

Locked gates: coverage_gain10: PASS, response_gain5: FAIL, response10: FAIL. Coverage gain requires>=10pp in both families; response gain>=5pp in both; response10 requires>=10%6144-reference ridge fidelity in both. No fresh-document confirmation or calibrated uncertainty is claimed.

**What this isolates.** More labeled interventions at the same underlying sites expand a nested feature span, while fixed geometry and average-loss penalty constrain the comparison. It is not a new-document learning curve and does not separately identify reference count, head coverage and local intervention diversity. One cyclic assignment cannot establish a universal scaling law. Feature coverage is geometric; output fidelity depends on the native readout and cancellation, so they must remain separate endpoints.

**Cost.** Dual output coefficients double3,538,944→7,077,888. Stored four reference descriptors double7,667,712→15,335,424 ambient-coordinate scalars, before core/partner/mean and any baseline program. Even one6144-reference model is not a compact native-MLP replacement when descriptors are counted. This experiment is meant to diagnose coverage and response weighting before choosing a structured primal factorization.

**Audit.**3072-reference feature coverage replays H319 and ridge10 fidelity replays H320 within1e-8. Both regularized linear-system residuals<1e-8. Every saved document/group has nondecreasing feature-projection energy under the expanded span within numerical tolerance. The second group differs at every site; assignment and6144 selected readout/descriptors are saved. Reference hashes checked; all jobs terminal, native weights unchanged.

**Next discriminator.** Compare stronger and weaker regularization at the larger fixed span using discovery calibration, and measure exact projected-output cancellation as coverage expands. Do not interpret weak ridge performance as failure of the expanded representation if unregularized projection improves. If the larger span gives little output benefit despite coverage, change the selection metric or introduce a primal tensor factorization rather than growing dual memory indefinitely. Any promoted model needs fresh and downstream validation. Broad research goal active.
