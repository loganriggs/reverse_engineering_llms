## H494 — Alignment of less stable corrections in probability space

Compare the complementary cross responses from H493's paired train-half models, retaining all four splits and both signs. Use the same512 reused H491 receivers and a common native intact baseline probability metric. Source4 calibration, exact operators, normalization, quadratic core and principal-core threshold remain frozen. This is a post-hoc alignment diagnostic, not new model fitting or fresh confirmation.

For each side, the correction is its cross projection onto the complement of the principal-cosine>=.95 core. Compute three pooled alignments: (1) raw vector cosine, (2) Fisher cosine of the baseline-centered readout tangent J delta, weighted by baseline probabilities, and (3) cosine of the actual finite log-probability change log p_full−log p_core, centered and weighted by the same baseline probabilities. The third metric includes finite nonlinear changes but is not KL or a cosine of probability vectors. Also replay the actual KL(full-model||core-model) changes from H493.

Pool inner products and energies over receivers before computing cosine. These are energy-weighted alignments, not the mean per-example cosine; large events can dominate. Disagreement/mean energy is (E_left+E_right−2 inner)/[(E_left+E_right)/2]. Both sides use the same weighting distribution, so apparent alignment cannot arise merely from choosing a different metric for each side.

| Split/sign | Raw pooled cosine | Fisher pooled cosine | Finite log-ratio cosine | Fisher disagreement / mean correction energy |
|---|---:|---:|---:|---:|
|0/negative|0.692|0.738|0.745|0.523|
|0/positive|0.689|0.720|0.713|0.560|
|1/negative|0.676|0.730|0.734|0.541|
|1/positive|0.674|0.723|0.721|0.555|
|2/negative|0.685|0.729|0.735|0.545|
|2/positive|0.691|0.720|0.715|0.564|
|3/negative|0.704|0.756|0.758|0.489|
|3/positive|0.704|0.745|0.741|0.509|

Across pairs/signs, raw alignment ranges 0.674–0.704, Fisher alignment 0.720–0.756, and finite log-ratio alignment 0.713–0.758. Fisher disagreement is 48.9%–56.4% of mean correction energy, but only 0.62%–1.21% of native intervention Fisher energy. These denominators answer different questions.

The corrections have meaningful common behavior and somewhat higher alignment in the probability-sensitive metric, but output sensitivity does not collapse their differences into functional identity. They are small relative to the dominant native effect while still appreciably different relative to one another. Similar total fidelity across half-data fits therefore reflects a stable dominant computation plus partly reproducible corrections; it does not establish that the entire unstable geometric tail is merely an irrelevant coordinate rotation.

This result also distinguishes three objects that should not be conflated: subspace overlap (H492), individual axis alignment after sign/permutation matching (H492), and input-dependent correction outputs under a probability metric (H494). Weak subspace/axis overlap can coexist with stronger output alignment because the dataset weights how directions are used. Conversely, raw-output overlap does not guarantee behavioral equivalence. These observations support an explicitly distribution- and metric-dependent canonical approximation, with a fidelity tolerance rather than a claim of unique universal feature axes.

Audits: finite full-to-core KL replays H493, all saved inner products finite, pooled cosines obey Cauchy–Schwarz and disagreement energies are nonnegative within numerical tolerance. Save per-receiver raw/Fisher/finite-logratio energies and cross inner products, native effect denominators and finite omission KL. No new documents or calibration; total18880 tracked documents unchanged. Four splits share the same training pool and receiving panel, so they are not independent replications. No confirmatory confidence interval, new threshold or global equivalence claim.

Next assess source-distribution transfer before assigning semantic interpretations. Freeze the structured source4/cross64 program and old calibration. Generate a new covariance-matched Gaussian source pool from centered calibration head6 writes, without per-sample norm matching; use a single frozen global scale if needed to match expected energy. Compare natural held-out source vectors and a head-output-span isotropic control with the same expected norm. Measure native full-source effects, exact source4-projection recovery and compressed structured recovery on already collected receivers. Separating exact source projection from cross compression identifies where transfer fails. This revisits the earlier covariance controls under the new structured computation, which lacks the earlier retained-output correction. Matching covariance would support a geometry/statistics explanation for this conditional representation, not prove the absence of semantic computation elsewhere. Keep this exploratory and audit source-energy differences before stronger claims. Observed-token CE, semantic interpretation, native-baseline removal and cross-head/layer transfer remain open.
