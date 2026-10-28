# M2: empirical function dimensionality differs from native channel implementation

2026-10-28. Returns the completed toy-grid diagnostics to native MLP0/1/2.160 forwards on the
same80 opened training pairs, with no replacement, pruning, learned featurizer or new-task exposure.
Capture native product writes excluding Down bias. Analyze both all1,278 token occurrences and
705 unique causal token prefixes. Repeated prefixes produce exactly identical captured writes in
all three layers, so deduplication removes repeated states rather than inventing a new state.

For each sampling, analyze raw writes and writes centered by their corresponding sample mean.
Center the channel activation columns consistently when measuring centered contribution energy.
Centering is an observational decomposition; no modified model is executed.

## How many directions carry the observed output energy?

r90/r99 count singular directions needed for90%/99% of squared output energy. Nuclear effective
rank is(sum singular values)²/(sum squared singular values); it is a different spectral statistic,
not numerical rank or a count of semantic features.

| Layer | Sampling | Centered? | r90 | r99 | Nuclear effective rank |
|---|---|---|---:|---:|---:|
|MLP0|Occurrences|No|24|102|51.1|
|MLP0|Occurrences|Yes|46|136|84.5|
|MLP0|Unique prefixes|No|22|122|55.4|
|MLP0|Unique prefixes|Yes|48|163|97.7|
|MLP1|Occurrences|No|36|110|67.2|
|MLP1|Occurrences|Yes|41|119|77.0|
|MLP1|Unique prefixes|No|52|160|103.1|
|MLP1|Unique prefixes|Yes|59|172|117.4|
|MLP2|Occurrences|No|51|150|100.1|
|MLP2|Occurrences|Yes|56|160|109.7|
|MLP2|Unique prefixes|No|56|196|120.7|
|MLP2|Unique prefixes|Yes|61|206|131.4|

MLP1's apparent dimensionality depends strongly on repeated-prefix weighting. Its occurrence-weighted
centered spectral participation rank is6.39, rising to24.38 after deduplication; r90 rises41→59.
Both samplings answer legitimate but different questions: frequency-weighted behavior in this
panel versus equal weight for each distinct prefix. Neither is a representative language distribution.

The mean accounts for53.3%,18.5%,13.7% of unique-prefix raw energy in MLP0/1/2 respectively.
Thus a leading direction can partly reflect a common write, not a distinct semantic feature.
After centering/deduplication, MLP1 and MLP2 r90 are similar, although MLP2 has a larger r99 tail.
This does not support naming qualitatively different computational strategies from one rank number.

## Relating the spectra to the toy's energy bound

For the output matrix Y over measured prefixes, each native product channel contributes a rank-one
matrix C_i. The nuclear-norm inequality from T2 implies:

    sum_i ||C_i||_F² / ||Y||_F² >= ||Y||_*² / (4608 * ||Y||_F²).

Unique-prefix values:

| Layer | Centered? | Actual channel/net energy | Free-factorization bound | Actual / bound |
|---|---|---:|---:|---:|
|MLP0|No|.2233|.0120|18.59|
|MLP0|Yes|.3972|.0212|18.74|
|MLP1|No|.3924|.0224|17.53|
|MLP1|Yes|.4366|.0255|17.14|
|MLP2|No|.5133|.0262|19.60|
|MLP2|Yes|.5503|.0285|19.30|

These native decompositions are not near the free matrix-factorization energy minimum on this
sample, unlike the tightly rank-limited penalized toys. All ratios remain below1, consistent with
net constructive aggregate interference even after centering/deduplication. This does not exclude
opposing channel pairs or cancellation on other inputs.

## What this does NOT establish

- The705 distinct prefixes impose a rank ceiling by themselves(at most704 after centering).
  They are short authored task inputs, not a corpus-wide view of the layer's function.
- The free-factorization bound allows arbitrary activation values over those rows. A real bilinear
  channel must have activation(Lz)(Rz), so a matrix refactorization need not be implementable with
  the same product architecture. Extra constraints or other inputs can require very different costs.
- A spectral truncation can discard low-energy directions with large downstream effects. No
  truncation was executed; these energy fractions are not causal fidelity or CE recovery.
- A large gap above the bound is not proof of waste, redundant neurons, or an achievable17-fold
  compression. Nor does it identify the optimizer's reason for the learned decomposition.

The useful conclusion is narrower: sample function dimensionality and native implementation width
are distinct, and sampling/centering change their relationship materially. The high-capacity toy
analogy is a hypothesis to test, not a description already established for the LLM.

## Next bounded interpretation check

Before training another representation, separate current-token-conditioned variation from
within-token context variation on these same observed writes. Use both occurrence and unique-prefix
views; report token counts, singleton groups, and exact within-sample energy accounting. A token
mean is allowed to re-cluster token geometry, so do not equate it with copying MLP0. This is an
observational decomposition, not held-out table fidelity or a causal context-removal intervention.
It will determine whether these few dominant output directions mostly encode token identity, a
common mean, or context-dependent changes, and which target a causal featurizer would need to preserve.
Do not reopen rank/seed sweeps or consume whole-task holdouts. Natural-text generalization remains
required before assigning these profiles to the layers generally.

## Reproduction and audit

```bash
python probe_empirical_spectra.py --output validation/empirical_spectra.json
OPENBLAS_NUM_THREADS=4 python audit_empirical_spectra.py
```

All twelve spectra independently reproduced with NumPy(max relative spectral discrepancy2.71e-13;
scalar discrepancy1.78e-11). Prefix membership/dedup indices checked against the source token IDs.
Raw occurrence contribution-energy ratios replay M1 within5.56e-17. Full native activation/write
matrices are generated at `/tmp/emlp_m2_matrices.pt` for the audit, not bundled. No model weights
or training data added beyond the already-shared panel. [Results](validation/empirical_spectra.json),
[audit](validation/empirical_spectra_audit.json).
