

## H351 — Direct intervention on compiler-response radial geometry

Reuse H346 recipients and cached compiler response d at intact final residual s. Define α=(s·d)/||s||² and t=d−αs. Compare finite KL(p(s)||p(s+δ)) with δ=d, t, t||d||/||t||, and t/(1+α). These are output-space diagnostic edits, not changes to native head payloads. Full-model interventions remain a separate object.

For positive1+α, s+d=(1+α)[s+t/(1+α)]. Thus RMS-normalized states should match for original and effective tangent, up to numerical precision and RMS epsilon. This is an algebraic positive control, not evidence for learned structure. Simple radial removal leaves t unchanged, while norm restoration increases its magnitude.

|Response|Actual mean KL|Moment mean KL|Donor mean KL|Moment/actual [95% interval]|Donor/actual [95% interval]|
|---|---:|---:|---:|---:|---:|
|original|0.00133398|0.00492942|0.00566851|3.695 [2.800, 4.935]|4.249 [3.218, 5.595]|
|tangent|0.00135902|0.00400093|0.00450612|2.944 [2.379, 3.674]|3.316 [2.674, 4.118]|
|normmatched|0.00171963|0.00932129|0.010221|5.421 [4.104, 7.253]|5.944 [4.505, 7.836]|
|effective|0.00133398|0.00492942|0.00566851|3.695 [2.800, 4.935]|4.249 [3.218, 5.595]|

Tangent/original control-to-actual ratios are0.797 [0.732,0.871] for moment controls and0.780 [0.726,0.847] for donors. Removing radial components reduces the ratio by20–22%, but does not halve it. The remaining tangent gap exceeds2 for both families; restoring response norm enlarges it. These reductions describe ratios, not causal fractions of KL excess.

Locked gates: tangent_gap2: PASS, radial_removal_halves_gap: FAIL. Radial finite scaling matters, but it cannot account for most of the observed control/native ratio under this test. Tangential response geometry remains a strong lead.

Uncertainty:4000 paired document/global realization resamples; pointwise intervals conditional on this reused panel, six realizations and fixed donor assignments. Donor dependence not reconstructed. These tests do not establish semantic roles or stable shared directions.

Audit: original finite KL replay error 3.75e-06; norm-restoration error 4.29e-16; tangent orthogonality error 8.13e-16; original/effective normalized-state relative difference 7.91e-08. Every1+α positive. Per-site α and four finite KL outcomes saved.

Next: refresh the hypothesis tournament around the surviving tangential alignment effect, contrasting stable output directions, context-specific geometry, token-confidence effects and learned coordination. Choose experiments that distinguish these alternatives rather than adding another decomposition of the same panel. Broad research goal remains active.
