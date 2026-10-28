

## H322 — Regularization at fixed expanded reference coverage

Freeze H321’s6144 references, descriptors, labels and kernel scale. Test ridge coefficients0.0002,0.002,0.02,0.2,2,20,200; these match H318’s average-loss penalty grid. Select one common coefficient maximizing the minimum family fidelity on discovery documents768:896; score documents896:1024. These documents have been used in earlier diagnostics; this is not fresh confirmation. Reference documents0:768 remain fixed.

Selected ridge: **0.002**.

| Target | Ridge20 fidelity | Selected fidelity | Gain (pp) | Conditional paired95% interval (pp) |
|---|---:|---:|---:|---:|
|actual|2.601%|4.944%|2.344|[0.904, 3.764]|
|angle|1.661%|2.839%|1.179|[-0.662, 2.835]|

Locked gates: shared10: FAIL, gain5: FAIL. shared10 requires>=10% selected fidelity in both families; gain5 requires>=5pp gain over ridge20 in both. Bootstrap resamples128 evaluation documents jointly within each family, conditional on fitted references and selection; it does not include training or adaptive research uncertainty.

**Audit.** Frozen reference/response hashes verified. Ridge20 coefficients reproduce H321 with relative discrepancy0; minimum Gram eigenvalue1.75e-05. Labels use the original site index modulo3072 for the second3072 examples. Layer17, payload seed and four positions match the cached response construction.

**Interpretation limits.** This isolates regularization within a single reference span. Neither a failure nor a success identifies semantic features, a compact implementation, or downstream usefulness. The stored dual readout still has7,077,888 coefficients plus15,335,424 descriptor scalars. Native weights remain unchanged.

**Next discriminator.** Compare the selected expanded-reference model against the3072-reference selected model on a new document panel with paired uncertainty, or change to a structured primal interaction model if gains remain small. Distinguish transferable interaction structure from source-specific sampling support. Broad research goal remains active.
