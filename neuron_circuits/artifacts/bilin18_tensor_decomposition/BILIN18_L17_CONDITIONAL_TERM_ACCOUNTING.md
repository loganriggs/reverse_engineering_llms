

## H294 — Span/complement interactions dominate what a linear96 readout misses

Decompose the remaining error of native MLP minus original compiler minus refined residual base minus the frozen conditional96 adapter. All tensors and the96-dimensional read span are frozen before outcomes. Center at the discovery natural-state mean, orthonormalize the stored read map, and split into five exact polynomial terms: mean-linear derivative along the span, mean-linear derivative along its complement, within-span quadratic, span/complement quadratic cross, and complement-only quadratic. The refined base's polynomial terms are subtracted too; this is not merely native-minus-original-compiler accounting.

Use float64 native/compiled/base coefficients with stable bilinear differences. The sum is compared with direct subtraction of the full remaining-error function on intact and intervened inputs. Same H293128 documents, four positions, nine heads, actual/angle full-removal payloads and seed: this is already-used-panel mechanism diagnosis, not new confirmation. No fitting. All quantities use native weights and are oracle polynomial components, not a deployable compressed model or output of the learned adapter. The denominator is remaining response error after the full conditional96 surrogate, distinct from earlier compiler residual denominators.

Relative full reconstruction error is9.17e-13; normalized Gram entries sum to1 within1e-10, all saved values finite. Cross-correlations and signed target alignments are retained. The sum of isolated term energies is1.088 times target energy for actual and1.032 for angle, illustrating non-additivity. Mean-linear complement energy is about1e-15 of target: the inherited base already contains the full mean-linear derivative and its quadratic read span lies inside the current96-span. This is a property of this surrogate/decomposition, not a claim that native computation ignores outside inputs.

| Exact terms retained | Actual remaining-error fidelity | Angle remaining-error fidelity |
|---|---:|---:|
|linear|-7.24%|-2.05%|
|linear_within|9.92%|3.68%|
|linear_cross|51.66%|44.71%|
|linear_within_cross|67.97%|49.40%|
|linear_outside|28.17%|50.21%|
|full|100.00%|100.00%|

- actual: adding exact cross terms raises fidelity58.06 percentage points; omitting complement-only terms leaves32.03% of target error energy.
- angle: adding exact cross terms raises fidelity45.73 percentage points; omitting complement-only terms leaves50.60% of target error energy.

Locked diagnostic gates: within_sufficient: FAIL, cross_gain: PASS, outside_needed: PASS, replay: PASS. Within sufficiency required80%; cross gain required10pp; complement-only necessity required20% residual, each for both payload families. Replay required<1e-8. No thresholds relaxed. Bootstrap intervals in JSON cluster documents and condition on this frozen decomposition.

**Interpretation.** Exact linear-plus-within96 terms account for only9.9% of actual remaining error. Cross interactions raise that to68.0%; complement-only interactions supply the rest. Angle controls show3.7%→49.4%, leaving50.6%. The within-only polynomial family is therefore not sufficient under this exact native-weight truncation. Negative linear-only fidelity reflects cancellation with quadratic terms; it is not evidence that all fitted linear maps must worsen error. Exact native terms are not necessarily the best regression approximations within a restricted feature family, so these truncations are not statistical capacity bounds.

This helps explain why repeated linear readout improvements were small: significant residual dependence involves products of read coordinates with outside context, plus outside-only products. It motivates relational/bilinear features over merely another static linear output map. It does not establish semantic binding or unique fundamental units. The decomposition is invariant to rotations within the chosen span but depends on the span and centering; a different span could redistribute the terms.

**Next discriminator.** Measure how many outside partner directions are required to reproduce the96-span/complement cross term. Use discovery-weighted cross-response covariance to select nested partner ranks, then evaluate exact projected cross contributions on these diagnostic documents and subsequently a new panel if a compact candidate emerges. Compare actual and angle responses, and retain all-partner cross as an oracle reference. Report storage and arithmetic cost;96xpartner-rank bilinear features can be far more expensive than the linear adapter. If few partners capture the cross response, compile and test that relational extension with intact behavior; if broad rank is required, the evidence favors distributed context dependence rather than a small added feature set. Complement-only residual remains a separate unresolved target.

All jobs terminal. Original model/compiler/base unchanged. Exact term Grams, alignments and reconstruction errors retained under bilin18_l17_conditional_term_accounting*. Research goal remains active.
