## H688 — Missing coactivation constraints explain synthetic transfer failure

Audit H687 known-membership shared dictionaries analytically. For each group support q, the learned linear map is D diag(q) E plus decoder/encoder biases. With the orthonormal synthetic teacher B and independent unit Gaussian active coordinates, expected relative error is computed exactly from the squared Frobenius norm of (D diag(q) E−I) B diag(q), plus bias error, divided by active atom count. A fresh Monte Carlo check corroborates the formula.

Five unordered teacher-atom pairs never coactivate in training: (1,5),(1,6),(1,7),(2,7),(3,7). Encoder couplings between these coordinates are unconstrained by reconstruction on the observed supports. As a teacher-informed diagnostic, zero only those entries of E B, leaving the rest of the encoder and decoder unchanged.

| Seed | Training analytic error | Held-support analytic error | Training after diagnostic | Held-support after diagnostic |
|---|---:|---:|---:|---:|
|11|8.81e-10|0.0535|8.81e-10|4.18e-10|
|12|5.04e-13|0.0232|5.05e-13|3.06e-13|
|13|1.12e-12|0.0274|1.12e-12|7.99e-13|

The diagnostic removes essentially all held-combination error while preserving training reconstruction. This identifies an unconstrained extension of the learned encoder outside the observed coactivation patterns, rather than incorrect union decoding. It does not imply exact teacher atom recovery; decoder alignments are imperfect, and multiple equivalent decompositions may remain.

Teacher-coordinate zeroing is unavailable in ordinary unsupervised LLM analysis and is not proposed as a practical repair. The actionable next test is training support coverage: include all pair combinations while continuing to hold out triples and all-group combinations. This distinguishes missing pair constraints from a general inability to compose known groups.

Audits:all source hashes unchanged; analytic and fresh Monte Carlo all-group errors differ by at most0.000776 absolute. Finite results, no model forwards or new documents. This is a conditional known-membership diagnosis, not unsupervised recovery. Broad goal and alternative-featurizer validation remain active.
