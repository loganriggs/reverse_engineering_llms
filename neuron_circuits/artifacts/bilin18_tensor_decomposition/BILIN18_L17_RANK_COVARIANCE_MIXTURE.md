

## H370 — Cancellation survives exact removal of between-context covariance

Use the same H369 probability distributions and post-RMS response vectors. For each recipient, compare two donor pools: six fixed confidence-matched documents and all127 other documents at the same position. For each pool apply the law of total covariance: covariance under the mean probability distribution = mean within-donor covariance + covariance of donor expected logit effects. Compute this separately for top2 self-energy, complementary self-energy, their signed cross term and total energy. Cap derivative remains omitted; probabilities come from the intact capped model.

Expected unembedding vectors E_p[W] allow exact between-donor accounting without fitting. H369 provides the matched mean-within metric and all127 mixture metric. Add the between term to the former and subtract it from the latter. A direct probability-mixture calculation on16 sites for each of3 actual-head responses independently checks the matched reconstruction.

|Head|Family|Pool|Within C/E|Mixture C/E|Between fraction of mixture energy|
|---|---|---|---:|---:|---:|
|0|actual|matched6|-0.584|-0.604|0.370|
|0|actual|other127|-0.628|-0.733|0.484|
|0|moment|matched6|-0.472|-0.557|0.366|
|0|moment|other127|-0.445|-0.611|0.413|
|0|donor|matched6|-0.396|-0.460|0.373|
|0|donor|other127|-0.383|-0.524|0.409|
|1|actual|matched6|-1.030|-1.176|0.492|
|1|actual|other127|-0.957|-1.199|0.507|
|1|moment|matched6|-0.900|-1.010|0.496|
|1|moment|other127|-0.865|-1.059|0.509|
|1|donor|matched6|-0.994|-1.116|0.499|
|1|donor|other127|-0.917|-1.136|0.508|
|6|actual|matched6|-0.872|-0.614|0.660|
|6|actual|other127|-0.689|-0.918|0.500|
|6|moment|matched6|-0.370|-0.279|0.627|
|6|moment|other127|-0.378|-0.406|0.538|
|6|donor|matched6|-0.363|-0.278|0.639|
|6|donor|other127|-0.361|-0.386|0.535|

Frozen gates: {'within_cancellation': True, 'averaging_equivalent': False, 'between_energy_majority': False}. All18 mean-within C/E joint upper bounds are negative. Thus cancellation is present without the covariance-of-mixture contribution. The ±0.1 equivalence gate for mixture-minus-within C/E fails, but every corresponding joint interval also includes zero: neither equivalence nor a nonzero change is established by these bands. The universal >50% between-energy gate fails; point shares range36.6–66.0%, and every joint interval includes50%. All lower bounds exceed zero, supporting a nonzero between-context contribution under this conditional analysis.

The cross/total ratio is nonlinear: within and between C/E values cannot simply be added. Energies and signed cross terms add exactly; the mixture C/E is their total-energy-weighted average. Depending on the donor pool, the between term can strengthen or weaken normalized cancellation. For head6 actual responses it weakens cancellation in the matched pool but strengthens it in the all127 pool. That pattern is exploratory.

Audits: {'matched_direct_mixture_relative_error': 3.971271695109971e-15, 'decomposition_relative_error': 1.1306537045093925e-17, 'minimum_self_energy': 1.1217619472837418e-11}. The direct matched-mixture subset replay is within4e-15 relative error; covariance decomposition is within2e-17 relative error. Self-energies are nonnegative and all entries finite. Input artifacts verified by SHA256. Save site-level within/between/mixture energies, bootstrap samples, and frozen position-mean probabilities plus all512 expected unembedding vectors for an independent future panel.

Uncertainty: bootstrap4,000 recipient-document and control-draw resamples, with donor pools held fixed. There is no donor-rank resampling here, unlike H369, and no rebuilding of probability distributions. The negative-band result must not be read as a correction or reversal of H369’s failed broader criterion; the estimand and conditional uncertainty differ. Bands are simultaneous across18 groups separately for within C/E, difference and energy fraction. Reused-panel accounting, not new-document confirmation.

Hypothesis update: reject the strong claim that cancellation is solely an artifact of averaging probability distributions. Shared within-context covariance itself supports interference. Between-context effects remain substantial and donor-pool dependent. Nothing here identifies two semantic features or compresses the native MLP computation.

Next: freeze these discovery probability summaries and evaluate their mean-within and mixture metrics on new documents with new native responses and control draws. Compare against each new context’s own metric. This tests whether the shared geometry transfers independently, and whether head6-specific attenuation repeats, without fitting to the new response outcomes. Broader goal remains active.
