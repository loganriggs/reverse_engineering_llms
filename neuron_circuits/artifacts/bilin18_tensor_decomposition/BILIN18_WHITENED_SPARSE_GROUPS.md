## H704 — Whitening repairs nonorthogonal synthetic group recovery

Keep H703 teacher structures, data, noise, initializations, step budget, support extractor and gate costs fixed. Add training-only uncentered second-moment whitening before sparse rotation and unwhiten the decoder afterward. With training SVD X=U S Vᵀ, use encoder span V diag(1/std) and decoder span V diag(std), where std=S/sqrt(N). Rotate both by the same orthogonal matrix. Normalize decoder columns and inversely rescale their codes, preserving reconstruction. The whitened encoder and original-coordinate decoder are dual bases; original decoder columns need not be orthogonal.

This uses the synthetic distribution's zero-centered population assumption rather than subtracting the empirical mean. All fitted statistics come from training examples. Teacher labels enter posthoc metrics only. Group extraction and interventions use the resulting original-coordinate, normalized-decoder representation; rotation selection minimizes whitened validation coordinate magnitude. Both gate costs are retained as diagnostics, not selected using heldout recovery.

| Teacher | Condition | Membership agreement | Mean group-subspace overlap | Held clean relative MSE, cost0.01 | Mean block-removal error, cost0.01 |
|---|---|---:|---:|---:|---:|
|7031|clean|100.00%|0.999463|0.000277362|0.00181371|
|7031|noise_5pct|100.00%|0.999643|0.000999492|0.00200664|
|7031|nonorthogonal|100.00%|0.999492|0.000287175|0.00167751|
|7032|clean|100.00%|0.999715|0.000119201|0.00146994|
|7032|noise_5pct|100.00%|0.999707|0.000824553|0.00203468|
|7032|nonorthogonal|100.00%|0.999699|0.000113834|0.00143592|
|7033|clean|100.00%|0.999878|0.000529521|0.00455928|
|7033|noise_5pct|100.00%|0.999839|0.00113959|0.00560662|
|7033|nonorthogonal|100.00%|0.999897|0.000535112|0.00468724|

All nine fits recover the teacher membership pattern after alignment. In the previously failing nonorthogonal conditions, mean group-subspace overlap rises from0.749–0.754to0.999492–0.999897; held reconstruction error falls from20.2–27.3%to0.0114–0.0535%. Mean coactivation-block removal error falls from0.358–0.462to0.00144–0.00469. This supports whitening/unwhitening as a useful repair for this full-column-rank linear-mixture setting, not as a generic guarantee for real activations.

There is a cost: some originally orthogonal cases have worse block attribution after whitening than H703, despite good reconstruction and exact memberships. Finite-sample second moments and optimization may introduce cross-block mixing; the present comparison does not isolate those causes. At the small1e-5gate penalty, nonorthogonal reconstruction reaches1.97e-7–5.33e-7but exact support recovery is only40.9–83.8%. Numerical/basis leakage and weak components make exact support more fragile than energy-weighted component effects. Low reconstruction error is still not sufficient evidence for the intended units.

The block metric aggregates directions with identical teacher membership patterns, accounting for legitimate within-block rotational ambiguity. It does not erase reconstruction/gating errors, and it is available here only because the synthetic teacher is known. In real activations, candidate blocks will need independent evidence from repeatability, data geometry and interventions rather than teacher matching.

Implementation audits explicitly check training whitening, encoder–decoder duality, and equality of the full reconstruction operator with the fitted PCA projector. Maximum dual-basis error9.54e-7; all checks below1e-5. Source hashes unchanged, finite results. Original-coordinate observed/clean reconstruction and component effects are evaluated after unwhitening. No original-model forwards/new documents; reserved LLM test split untouched.

The pipeline now merits a bounded early-layer pilot alongside the existing BSF baseline. First measure retained variance and numerical conditioning on the existing training activations, then fit sparse rotations in a tractable retained span and assess stability/coactivation using validation only. Report projection loss separately from sparsification loss, and compare rotations within the same retained span before comparing with overcomplete BSFs. Do not impose the synthetic four-group/three-atom ontology on LLM data. Full-dimensional or overcomplete discovery, efficient group inference, matched resource frontiers, manifold validation and diverse-task causal tests remain outstanding. This pilot is an information-gathering step, not certification of a BSF replacement. Broad goal active.
