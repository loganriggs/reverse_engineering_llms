## H830 — Native bilinear state explains more local MLP response variation than incoming direction

H829 showed that equalizing the MLP response strongly aligns the first downstream block's outgoing perturbation, but did not explain why MLP responses differed. H830 crosses the raw MLP baseline and arriving perturbation separately for all13 groups and both block/random directions.

### Exact local operator and natural-reference crossings

For raw MLP baseline b, incoming direction v, and r=sqrt(mean(b^2)+float32 epsilon), normalized state u=b/r and normalized direction q=v/r-b*mean(b*v)/r^3. For the ungated bilinear MLP, its derivative is

D[(L u) elementwise (R q) + (L q) elementwise (R u)].

Compute in float64 using the model's fixed weights. Bias does not contribute to the derivative. The incoming v is H828's captured central derivative at the raw MLP input. This is the exact analytic derivative at b for that v, not an exact derivative of the entire upstream finite-intervention path.

Cross each of four NATURAL baselines with all four incoming directions within a group. The diagonal gives each recipient's own pair. For common-baseline comparisons, hold each natural baseline fixed in turn while varying four inputs; for common-input comparisons, hold each natural input fixed in turn while varying baselines. Average all reference choices, without choosing a favorable reference or creating an averaged baseline. Full16-cell response arrays are saved.

### Results

Numbers are equal-weight group means of within-group response cosines, not the medians used in H827–829; do not compare them numerically as identical statistics.

| Injection site | Direction | Own baseline/input | Common baseline | Common incoming direction |
|---|---|---:|---:|---:|
| after block0 | block |.387|.728|.497|
| after block0 | random |.385|.943|.407|
| after block1 | block |.455|.847|.489|
| after block1 | random |.386|.941|.400|

Holding baseline fixed produces higher alignment than holding input fixed in all13 groups, for both direction types. Incoming perturbation variation still matters, particularly for block edits after block0; the fixed-baseline alignment is not universally near1. The evidence supports baseline-dependent local MLP sensitivity as a larger source of angular variation in this panel than incoming perturbation direction alone. It does not provide additive percentages of explained variation or naturally mediated causal shares.

### Separate normalization from bilinear multiplication

At each fixed raw input direction, vary baseline states. Then separately hold normalized bilinear state u fixed while varying RMS-derived q, or hold q fixed while varying u. Average all four natural reference choices for each factor.

| Injection site | Direction | Vary normalization state only | Vary bilinear state only |
|---|---|---:|---:|
| after block0 | block |.974|.510|
| after block0 | random |.994|.413|
| after block1 | block |.979|.495|
| after block1 | random |.997|.401|

The bilinear-state-only patterns closely resemble the full varying-baseline/common-input result; normalization-state-only responses remain highly aligned. This points mainly to multiplication by the native bilinear state, rather than RMS normalization alone, as the source of directional variation under this decomposition. Norm effects and interactions remain, and this artificial factor crossing is not a natural network intervention.

State-dependent derivatives are expected from the bilinear architecture. The new empirical finding is their relative importance compared with upstream perturbation variation and normalization for these actual captured states. It would be circular to call the formula alone discovery of the model's feature ontology. Nor is this specific to good BSF blocks: random-direction controls show the same baseline dependence.

### Verification and scope

Native normalized-state replay relative error is7.64e-8. Analytic own derivatives match captured natural MLP central responses with median relative error.000217 and maximum.001102. An independent centered finite difference of the float64 RMS-plus-bilinear map at step1e-4 matches the analytic expression within1.02e-9 relative error. All tracked input hashes unchanged and metrics finite. Zero whole-model forwards, no test evaluations or new documents; natural-document pool26,112. The original model is loaded only to evaluate local weight maps.

The result identifies a local mechanism for context-dependent response variation. It does not show that changing the baseline-dependent operator is behaviorally necessary, recover semantic features, or establish smooth computational manifolds. Subsequent layers also introduce context dependence, as H829 demonstrated. A useful representation may pair reusable directions with state-conditioned operators, but the appropriate compact operator coordinates are still unknown.

### Hypothesis priorities after the latest tests

The H824 twenty-hypothesis set remains open. Its strongest updated branch is recipient-state-dependent sensitivity (H824 hypothesis14), with specific bilinear-state evidence here. Generic direction magnitude (15) and gate-only geometry (4) no longer suffice for the controlled findings. Context-free portability (13 interpreted strongly) is weakened, including the block56 output-filtering exception. Flat subspaces (19) remain serious alternatives; shared/hierarchical/DAG descriptions (20) still require selective reuse and transfer evidence. Smooth geometry (1) versus contextual mixtures (2) remains unresolved: neither is necessary to explain the local bilinear result.

Next perform an outcome-controlled downstream experiment replacing only the MLP response with crossed-baseline versus crossed-input derivatives, preserving recipient skip/attention paths and original native MLP baseline. Use all natural references rather than selecting one. Validate own-response replay, then ask whether the baseline-driven local angular changes predict full-output differences. In parallel with that causal question, the research target should become a compact state-conditioned operator that predicts responses on new recipients, compared with a fixed operator and flat/BSF representations. Reconstruction alone remains insufficient.
