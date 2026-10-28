## H715 — Downstream response geometry is context-dependent and only weakly favors coactivation groups

Previous turn classification: progress. H714 completed512causal evaluations and found no reliable group advantage over matched substitute neighborhoods. H715 changes the target: do members of a group influence similar output probabilities locally, rather than making the entire distribution more like a donor?

Freeze16hash-selected H712documents, excluding all64H713swap documents. Evaluate position63in every document with no activation or output selection. Test the union of134directions appearing in the eight real groups and their24frozen H714substitute neighborhoods. Each decoder column has unit residual-space norm. Apply symmetric positive/negative perturbations of1%of residual norm at block1output, preserving the value bus. Compute centered finite-difference logit derivatives; first-context half-dose checks test numerical stability.

For native probability p and logit derivative v, the response signature is sqrt(p)*(v-E_p[v]). Its squared norm is the local Fisher/KL sensitivity, and its direction describes which probabilities change. Compare squared cosine between anchor and neighbor signatures: this removes arbitrary dictionary signs. Also remove the projection onto sqrt(p)*(native_logits-E_p[native_logits]), the local temperature direction. This is a local one-dimensional confidence control, not removal of every confidence-related effect.

The frozen contrast averages the seven real anchor-neighbor squared cosines, subtracts the average over the three substitute neighborhoods, then averages equally over all eight anchors and16contexts. Bootstrap intervals resample16documents, retaining all anchors within each document;10000resamples. They do not cover uncertainty over group discovery or control matching.

| Response metric | Real groups | Substitute groups | Difference | Context bootstrap95%interval |
|---|---:|---:|---:|---|
|fisher|0.30880|0.28893|0.01987|[0.00149,0.04016]|
|confidence_removed|0.27866|0.26375|0.01491|[-0.00314,0.03540]|

The full response metric gives a modest positive advantage; after temperature projection the interval includes zero. Only9of16contexts have a positive difference for either metric. Group effects vary, so this is not evidence that all eight groups share a common computational role. Temperature accounts for18.6%of anchor response energy on average.

A material matching limitation remains: mean squared decoder cosine is0.06428in the real groups versus0.04186in controls, a static difference0.02242. Thus the control procedure did not fully match geometric alignment. The response advantage cannot be confidently attributed to coactivation-based organization beyond decoder geometry. The two cosine metrics are not interchangeable, and subtracting their differences would not be a valid causal correction.

### What the metric audit adds

After removing the temperature component, all134tested directions' output responses still concentrate in few effective directions within each context. The participation rank (sum eigenvalues squared divided by sum squared eigenvalues) ranges1.80–7.32, median3.51; the leading three response axes carry55.6–94.4%of response energy. This is a local response-space result, not the dimension of a representation manifold. Concentrated native token probabilities can contribute to low Fisher rank; native probability participation varies1.00–124.3across contexts. An anisotropic downstream map may make many unrelated decoder directions appear functionally aligned.

Across different contexts, the same anchor's probability-weighted response vectors have near-zero average signed cosine. Those vectors use different probability weights, so this alone cannot establish unstable computation. A posthoc audit uses one pooled probability vector for all16contexts and recomputes centering and temperature projection. Mean signed cross-context cosine remains low,0.0006–0.0315across anchors; mean squared cosine0.0207–0.0354. A fixed token-output direction is therefore not demonstrated, although reusable context-dependent operations could still exist.

The pooled-weight within-context real-minus-control similarity difference is0.02881on average, positive15of16contexts. This is a sensitivity analysis, not replacement of the frozen primary metric: it changes the weighting objective and retains the decoder-geometry mismatch. Results depend materially on which output differences count as important, reinforcing the distinction between activation reconstruction, local distributional faithfulness and task effects.

### Research implication and next test

These data do not verify the current coactivation neighborhoods as computational units. They do reveal a measurable local functional geometry, potentially closer to the user's data-based canonical-form idea: many activation directions can have redundant or weak effects on the outputs relevant to a context. Its meaning and persistence across contexts still need testing.

The next informative control should preserve decoder geometry exactly. Construct rotations within the measured decoder span that fix the anchor while preserving the full neighborhood Gram matrix, and compare their predicted downstream similarities against real groups. The saved directional derivatives permit this as a local linear-response test; verify a subset with actual finite interventions. This addresses the known geometry mismatch and tests whether coactivation groups align with the downstream map beyond merely pointing in similar residual directions. Such rotations do not preserve natural activation frequencies, so retain the present statistical controls as a complementary comparison. If geometry-corrected advantages survive, proceed to competence-checked diverse tasks and held combinations, not semantic naming from cosine alone.

Audits:4572whole-model prefix evaluations (16native,4288symmetric edits,268half-dose edits). Prefix-state replay maxrelative1.39e-6. First-context derivative relative error from halving dose:median0.00177,95thpercentile0.00471,max0.01033; confidence-removed response cosine with half-dose minimum0.999981. Maxchange in any pair cosine0.00631. Independent CPUfloat64 metric reconstruction from savedfloat32derivatives agrees within2.53e-7. All source/model hashes unchanged, metrics finite, fresh contexts disjoint from prior swap panel. No original reserved-test evaluation or new unique research documents; pool26112. Broad goal active.
