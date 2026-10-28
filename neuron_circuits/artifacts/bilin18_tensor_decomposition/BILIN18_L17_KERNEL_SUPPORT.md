

## H319 — Input interaction-feature support differs sharply between specialist and mixed references

No output fitting. For each query feature phi=u_z⊗v_z−u_x⊗v_x, compute its orthogonal projection onto the row span of training interaction features. If K is the training Gram matrix and k the query-to-training kernel row, captured squared norm is k K⁺ kᵀ; divide by k(query,query). Compare3072 matching-specialist references with the same-size H317 mixed references. Query panel is H315 reused diagnostically. Native response labels are not used in these projection scores, although the script reconstructs them in its inherited collection setup. Core96/partner64 geometry remains native-informed.

| Target | Reference span | Feature-energy coverage | Equal-site coverage |
|---|---|---:|---:|
|actual|specialist|73.81%|68.99%|
|actual|mixed|35.34%|34.31%|
|angle|specialist|61.62%|60.20%|
|angle|mixed|33.39%|31.54%|

Locked mixed50 gate FAILS: mixed coverage does not reach50% in both families. Specialist-gap20 gate PASSES: matching-specialist coverage exceeds mixed by38.47pp actual/28.24pp angle. Primary relative eigenvalue cutoff1e-10;1e-8 gives identical reported scores, and1e-6 changes pooled coverages by at most.011pp. The large coverage difference is not explained by tested numerical cutoff choices.

**Inference.** At the same total reference count, dispersing examples across18 groups covers much less of each target interaction-feature population than a dedicated specialist. This directly supports a sample-support component of shared prediction failure, independent of output optimizer choice. It does not demonstrate intrinsic head-specific polynomials: the model's exact cross rule is still shared, and mixed supervision supplies far fewer examples per group. The specialist comparison uses a different matching reference span for every target group, not one equally sized global span.

Feature coverage is NOT response fidelity or its upper bound. Native output coefficients can strongly weight poorly covered directions, or cause cancellation between covered and uncovered contributions. In particular, mixed feature coverage~33–35% is much higher than calibrated shared response fidelity~1–2%; a readout-weighted analysis is needed before attributing the full failure to Euclidean support. These norms use the natural orthogonal core/complement product coordinates, not an arbitrary learned whitening. General nonorthogonal reparameterizations could change this geometric metric.

**Audit.** Gram projection formula matches explicit row-span projection on a small deterministic random matrix within1e-10. All per-query fractions are in[0,1] up to1e-6 tolerance;19 source spectra and retained ranks are saved for three cutoffs. Independent aggregation reproduces summary coverage within1e-12. Previous H314 explicit outer-product kernel identity remains applicable. Specialist source hashes checked by inherited setup; the mixed archive was read from the frozen H317 artifact (hash recorded in provenance below). All jobs terminal; no native weights or readouts updated.

**Next discriminator.** Apply the native linear readout of interaction features to the covered and uncovered feature components, using exact kernel interpolation from noiseless training responses. Measure their energies, alignment/cancellation and reconstruction error relative to ridge predictions. This determines whether Euclidean feature projection discards disproportionately important output directions or whether regularization is the main extra loss. Keep this oracle decomposition separate from any efficient predictor claim. Data-coverage scaling and a compact shared factorization remain subsequent tests. Broad research goal active.
