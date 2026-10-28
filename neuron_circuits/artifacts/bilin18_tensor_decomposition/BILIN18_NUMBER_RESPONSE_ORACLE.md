## H595 — Response-optimal coordinates improve causal fidelity sharply, despite a large irreducible local-response residual

The previous goal turn made progress by showing that retaining all discovery PCA directions did not rescue projection transfer, while explicitly leaving the coefficient objective unresolved. Freeze the same 32-dimensional context span. For each actual incoming perturbation e, form M with columns T(a_j,e), where a_j are the context basis vectors. Solve min_c ||M c - T(dc,e)||_2 using float64 SVD with rcond=1e-10. Then replace the MLP11 context-switch response by M c in the live model. Compare full target, unchanged context, Euclidean context projection and this oracle: 2,048 interventions over 512 signed-dose cases.

The oracle uses each full target local response. It is a diagnostic lower bound for Euclidean response error in this span, not a learned predictor, deployable representation, or validation of generalization. Incoming perturbations are not compressed. It does not optimize output KL, answer margin or answer loss.

| Small-dose cell | Projection local error | Oracle local error | Projection relative margin error | Oracle relative margin error | Oracle relative reference KL |
|---|---:|---:|---:|---:|---:|
|regular/fronted_preposition|0.0000|0.0000|0.0002|0.0002|0.0000|
|regular/relative_clause|0.8300|0.7889|0.7948|0.1650|0.1814|
|irregular/fronted_preposition|0.6020|0.5523|0.6355|0.1432|0.1616|
|irregular/relative_clause|0.8824|0.8431|1.1264|0.3099|0.3023|

Changing only the coefficients improves behavior much more than internal reconstruction. Irregular fronted margin error drops 0.636 to 0.143 while local response error drops only 0.602 to 0.552. Irregular relative-clause margin error drops 1.126 to 0.310 while local response error drops 0.882 to 0.843. Relative reference KL improves from 0.510 to 0.162 and 0.765 to 0.302 respectively. The half-dose results show the same pattern. These are ratios relative to the full context-switch versus unchanged-context discrepancy, not total task performance or explained-computation percentages.

This establishes that Euclidean context projection was a poor objective for the downstream effect. It also establishes a distinct internal-fidelity limit: under the fixed incoming perturbation, even the response-optimal combination within this context span leaves substantial local-response error outside its column space. Better internal fidelity requires additional directions or a different span. That does not establish a lower bound on output KL or answer loss: a behavior-optimized combination could do better while preserving less of the internal vector.

The result is not driven by an unstable inverse or huge coefficients. Every response matrix has retained rank 32, the maximum condition number is 7.46, and oracle coefficient norm / actual context-difference norm ranges 0.413–1.000. The maximum normalized normal-equation residual is 7.38e-15. Reconstructing the coefficient-derived context and running the bilinear map directly agrees with the oracle response within 6.56e-7 relative error. No oracle case is worse than ordinary projection in the optimized local metric.

Next separate coverage from rank and objective. Expand discovery contexts from regular/fronted only to regular/fronted plus regular/relative-clause contexts, keeping irregular nouns outside fitting. Compare old rank32, new rank32 (fixed capacity), and new full rank64 spans, each with both Euclidean projection and response-optimal coefficients. Keep full and unchanged references, all signed doses, and all irregular cases. This tests whether broader context coverage improves the attainable response space, rather than attributing every failure to intrinsically complex computation. All existing prompts are already explored; subsequent confirmatory transfer will require new lexical/frame combinations.

Audits: sum/full and unchanged references replay H591 within 1.44e-5 margin; context-switch identity <=1.87e-6; finite metrics and frozen source/bases; float64 KL/group-NLL. The inherited matched-input-norm audit field is zero because no new random matching is done here. No new prompts or natural-text documents; FineWeb remains 26,112. Broad goal active and incomplete.
