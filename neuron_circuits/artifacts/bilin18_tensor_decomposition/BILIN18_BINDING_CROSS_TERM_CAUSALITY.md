## H926 — The missing finite-change interaction is local to attention, with little downstream nonadditivity

H925 found that adding separate routing and value effects badly predicts a full donor-head swap. H926 tests the two competing explanations directly at exploratory heads L14H4, L14H6 and L16H3, retaining all128 transfer rows and both contrasts.

Let native attention routing and values be A,V and donor counterparts B,W. The exact head-output change is R+U+C, with R=(B-A)V, U=A(W-V), and C=(B-A)(W-V), summed over source positions. Intervene separately on R,U,C, jointly on R+U, and jointly on all three. Compare these with the cached full donor-minus-recipient head write. All remaining computation runs live.

This attention product identity is already known from the architecture; the empirical question is which term explains the observed failure of separate causal effects to add.

### Results

All numbers are RMS changes/errors in the recipient second-event-name margin. Downstream RU nonadditivity compares the simultaneous R+U edit with the sum of separate R and U logit effects. Cross increment compares full replacement with simultaneous R+U. Cross-context nonadditivity compares that increment with the cross-only effect.

|Contrast|Head|Full effect|Separate-sum error|Downstream RU nonadditivity|Cross increment|Cross-context nonadditivity|
|---|---|---:|---:|---:|---:|---:|
|event_swap|L14H4|0.2465|0.3808|0.00140|0.3802|0.00353|
|event_swap|L14H6|0.1029|0.1966|0.00027|0.1965|0.00092|
|event_swap|L16H3|0.0802|0.1781|0.00010|0.1781|0.00038|
|same_fact_reword|L14H4|0.2439|0.3927|0.00150|0.3921|0.00386|
|same_fact_reword|L14H6|0.1011|0.2006|0.00030|0.2005|0.00101|
|same_fact_reword|L16H3|0.0794|0.1798|0.00010|0.1798|0.00039|

At L14H4, event-swap separate-sum error is0.38079. The explicit cross increment is0.38022, whereas downstream RU nonadditivity is0.00140. The cross-only effect is0.38166 RMS and differs from its contextual increment by0.00353 RMS. All three heads and both contrasts show the same ordering. These are comparisons of magnitudes, not additive percentages of explained variance.

The evidence therefore attributes the large gap mainly to the omitted local finite-change cross term, not downstream nonlinear processing of the two individual writes. The full effect can be smaller than the cross term because components cancel. Signed means also hide this: cross-only mean effects are about-0.0005 to-0.0014 despite much larger RMS effects. Per-endpoint fidelity is necessary.

Adding all three terms reproduces full donor replacement: margin RMS mismatch about1.5–1.8e-6, with full-vocabulary maximum logit mismatch9.06e-6. This is an implementation check, not independent proof of a discovered feature.

### Interpretation caveat and next test

This cross term is relative to the recipient-centered, position-aligned decomposition. It is not a coordinate-invariant new unit of computation. The same bilinear change can be written exactly as delta-A times midpoint-V plus midpoint-A times delta-V, without a separate cross term. Moreover, both experimental contrasts move names between event positions. Aligning a donor coefficient with the value at the same position may mismatch identities that moved.

Next compare position alignment with name-identity alignment. Permute donor routing and values together at the two event-name positions so that each name aligns with its recipient occurrence. Such joint reindexing leaves the full donor head output unchanged. Recompute the component geometry and causal effects to test whether the large cross term shrinks when name identity is aligned. If it does, much of the interaction reflects coefficient/value reassignment under a token swap; if it persists, investigate context-dependent routing beyond that reassignment. Retain non-name positions and both contrasts, verify permutation invariance, and distinguish this analysis reindexing from changing the model's positional computation.

This is directly relevant to a data-based canonical representation: the representation should track stable relationships and preserve behavior, rather than declare every large coordinate-dependent cross term a separate feature. The experiment has not yet identified a canonical form or verified a manifold.

### Verification

5,504 forwards. Identity replay exactly zero; prior H925 routing/value/full margin effects replay exactly. Native-cache max logit difference2.05e-5. Exact local algebra error7.11e-15. All-terms versus full replacement max logit error9.06e-6. Source hashes unchanged and all outputs finite. Every endpoint/control and diagnostic saved. No reserved test or added natural documents; pool remains26,112. Broad goal active.
