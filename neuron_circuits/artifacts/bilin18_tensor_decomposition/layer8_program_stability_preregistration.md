# Preregistration: stability boundaries of sparse program extraction

Second-order Mobius responses are nearly exact in bulk, but rare triples select
causally different mode sets. This experiment tests whether those exceptions
are ill-conditioned sparse-selection boundaries rather than broadly missing
third-order energy.

For every frozen triple recompute the exact and second-order response Grams and
record:

- the minimum relative gap between the chosen and runner-up mode at all 16
  greedy selection steps;
- relative Frobenius error after whitening each Gram coordinate by its mode's
  input-amplitude cost;
- unweighted Gram error and response cosine;
- the already frozen exact finite-effect fidelity.

Failure is fidelity below 0.8. Report AUROC for each diagnostic. FineWeb chooses
a two-dimensional margin/error certificate by exhaustive threshold search,
maximizing coverage subject to at least 99% precision. That certificate is then
frozen for WikiText. Certification passes only if it covers at least 90% of
triples at 99% WikiText precision.

With only two FineWeb and eleven WikiText failures, AUROC and certification are
descriptive and uncertainty is substantial. The frozen OOD certificate is the
stronger test. Failure does not invalidate bulk second-order reconstruction.
