

## H355 — Removing compiler-based metric normalization

Reuse H354 exact directional energies. Primary statistic: for each site and assigned metric divide the mean control output Rayleigh by actual output Rayleigh, then average logs equally across sites and matched ranks and exponentiate. A scalar multiplier of the metric cancels before averaging. Report own-context geometric ratio divided by matched-context geometric ratio. Secondary statistic: restore the reference trace and form unnormalized pooled ratios. These aggregation rules estimate different quantities; they are not interchangeable.

|Response|Aggregation|Moment own/matched [95% interval]|Donor own/matched [95% interval]|
|---|---|---:|---:|
|compiled|invariant|1.434 [1.288, 1.524]|1.399 [1.256, 1.495]|
|compiled|raw_pooled|1.401 [1.156, 1.669]|1.404 [1.160, 1.686]|
|native_mlp|invariant|1.383 [1.260, 1.453]|1.332 [1.207, 1.402]|
|native_mlp|raw_pooled|1.310 [1.111, 1.520]|1.308 [1.108, 1.527]|

The native invariant gate passes: both lower95% bounds exceed1.2. The native raw pooled gate fails: lower bounds are1.111 and1.108. Thus a compiler-based scalar metric normalization is not necessary for the pairing effect, but the stronger threshold is not robust across aggregation choices. Both raw pooled pointwise intervals remain above1; that weaker observation is distinct from the failed frozen threshold.

The invariant statistic still uses native MLP responses rather than full source interventions, and retains H353 confidence-matched neighbors. It tests geometric output alignment, not semantic units. No fresh documents are introduced.

Audit: invariant estimates agree with and without reference-trace multiplication to absolute error 4.44e-16; all saved Rayleigh values positive; source hash verified. No model rerun or native weight change.

Uncertainty:4000 shared document resamples, control-realization resamples and matched-rank resamples. Ratios recomputed within each draw. Pointwise conditional intervals; fixed neighbors and donor dependence not rebuilt. Nonlinear averaging across a finite number of control realizations can affect bootstrap centering.

Next: freeze the native per-site geometric statistic and confirm on fresh documents, computing confidence neighbors within the new panel. Keep the raw pooled diagnostic and report disagreements. Then test other heads to distinguish a head6-specific effect from a broader organization principle. Broad research goal remains active.
