## H820 — Frozen representation transfers partially to 31 fresh prefixed IOI pairs, with a capacity tradeoff

Prespecify the new surface “Yesterday, NAME and NAME went to the market. GIVER gave a parcel to” before native evaluation. The fixed prefix gives initial names a leading word boundary, reducing the tokenization mismatch responsible for H818's seven-item panel. Both initial name orders and giver choices remain. This is a new surface condition; H818's original eligibility failure is preserved, not retroactively changed.

### Native competence and frozen implementation

31 of 32 planned name pairs pass all source/answer token and paired-query alignment checks. Only pair27 (Roger/Pamela) fails single-token validity, with all four affected contexts recorded. The 124 native contexts have 100% candidate accuracy separately for both expected-recipient positions and 100% paired-giver correctness; full-vocabulary accuracy is 95.97%. The panel passes the frozen competence/item gates.

Names are disjoint from the original response-fitting panel. The H815 dictionary, shared membership, private bases, and geometry gate are loaded unchanged and checked against the pre-native lock hash. Frozen uniform flat5 and flat8 bases are included without refitting. Source perturbations remain L2H5 at the second initial name, using the first initial name's projected mixed value, at 0/.25/.50. Capture and cross the L6 raw baseline/input response as before. The prefix changes absolute positions and potentially computation; this tests transfer under both lexical and surface change, not a pure lexical-only comparison.

### Full-output KL

All divergences compare the full uncompressed intervention distribution with the candidate representation under the same perturbation and response arm.

| Representation/components | Mean KL | Ratio to zero-module KL |
|---|---:|---:|
| Zero module response | .00002751 | 1.000 |
| Shared + selected private | .00001261 | .458 |
| Shared only | .00002045 | .743 |
| Selected private only | .00002424 | .881 |
| Shared + other private | .00002029 | .737 |
| Frozen flat5 | .00001473 | .535 |
| Frozen flat8 | .00001171 | .426 |

The shared/private representation remains useful but imperfect. Removing either component worsens KL in both selected groups: group0 both/shared-only/private-only ratios are .289/.527/.616, and group1 .467/.754/.895. The shared component's operational usefulness therefore extends beyond the small original IOI internal holdout.

Selected private blocks outperform the alternatives on both groups. Aggregate other-minus-selected item-mean KL is +7.68e-6 with descriptive item-bootstrap interval [6.30e-6,9.21e-6]. The other block wins on only 10.3% of individual correlated variants. This differs from the uncertain copying-group1 anomaly and argues against globally reversing the geometry gate.

### Architecture tradeoff rather than a universal winner

Shared/private uses eight stored directions and five active coordinates. Flat5 uses five stored/five active and has higher KL; its paired excess is +2.12e-6 [1.61e-6,2.65e-6]. Flat8 uses eight stored/eight active and has lower KL; its paired difference is -.896e-6 [-1.477e-6,-.408e-6]. These comparisons show a storage/active-dimension/fidelity tradeoff, not dominance at matched total resources.

Mean total variation is .00171 for shared/private, .00183 for flat5, and .00167 for flat8, agreeing with the KL ordering. Absolute errors are small, and the tested MLP is only one downstream path. KL ratios must not be interpreted as percentages of model computation explained. Full-intervention and shared/private top-1 accuracy are both 94.96%; flat5/flat8/zero are 95.16%. These tiny accuracy differences do not replace distributional fidelity measurements.

### Verification

6324 total forwards: 124 native competence, 1860 response capture/reference, and 4340 representation evaluation. Native source value invariance across giver changes is exact. Response capture also verifies exact L2 source-state invariance, exact RMS input reconstruction, float64 bilinear expansion relative error 3.98e-14, endpoint-scaled module reconstruction error 2.26e-6, exact zero-delta full-logit replay, and own-response full-logit replay within 1.05e-5.

Final uncompressed log-probability effects replay fresh references within 5.49e-6. Component decomposition error is 1.78e-15, basis orthogonality error 3.11e-15, dictionary hash matches the pre-native lock, names are disjoint from fit data, and all tracked hashes unchanged. KL values are finite/nonnegative and full-reference self-KL exactly zero. Full logits are saved. Reserved test untouched; natural-document pool 26,112.

Group0 contains 52 correlated variants across 20 name pairs; group1 contains 444 variants across all31. Item bootstrap accounts for lexical grouping, but intervals remain descriptive without multiplicity correction. The word “group” refers to a learned response-geometry selector, not a proven semantic concept.

### Scope and next step

We now have fresh evidence that the frozen shared/private representation carries useful response information on copying and IOI, while remaining substantially imperfect. It describes finite L6 input changes under one selected upstream perturbation family, not all native activations, all computations, or a complete activation featurizer. Hierarchy, semantic manifold identities, and a computational DAG remain unvalidated.

Complete the corresponding frozen flat5/flat8 controls on the fresh 32-item copying panel before comparing architecture tradeoffs across tasks. Reuse saved full-reference distributions and the fixed bases; do not refit or choose gates from fresh outcomes. Then assess whether any shared/private advantage justifies extending the architecture beyond this response-specific benchmark.
