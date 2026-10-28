## H936 — Frozen live tests: continuous routing prediction nearly matches grouped alternatives

Evaluate all21 H935 head-models with their discovery-selected penalties, without further fitting or selection. Jointly replace only the persistent-value component of L8H3/L13H0/L14H4, using each live query to predict signed routing coefficients; keep current-value contributions live. Run native and opposite-endpoint donor values, subtract each approximation's own native output for intervention effects, and compare full output distributions. Existing96 lexical-transfer contexts cover agreement, language recall, name binding and matched copying. This is repeatedly used research-transfer data, not a fresh or reserved test.

| Task | Continuous effect error | Learned-group range | Random-group range |
|---|---:|---:|---:|
| Binding |6.36%|5.65–6.22%|6.03–6.81%|
| Language recall |14.36%|13.04–14.18%|13.51–14.24%|
| Name copying |9.43%|6.23–8.44%|7.86–11.91%|
| Country copying |32.24%|33.76–35.11%|31.49–42.07%|
| Agreement-word copying |27.57%|22.53–27.09%|22.88–25.99%|

Effect error is RMS discrepancy in donor-induced answer-logit margin change divided by reference-effect RMS, pooled within task. Ranges include all three seeds, not confidence intervals. Only four transfer lexical pairs/domain underlie these values; correlated endpoints/templates are not independent replicates. Agreement task itself has a tiny reference effect for these attention sites and relative errors over100%; no positive agreement claim.

Binding native KL is .00307 continuous, .00238–.00311 learned groups, .00215–.00336 random groups. Binding edited KL is .00269 continuous, .00212–.00253 learned, .00223–.00379 random. Language native KL is .02453 continuous, .02116–.02518 learned, .02187–.02411 random. Thus both whole-distribution and margin comparisons show close competition, not evidence that discrete group membership is required. Name copying benefits from learned groups here, but country copying does not. No universal architecture winner.

The important change from H934 is the continuous baseline: discovery-only CV chose weaker penalties, reducing binding effect error from about14% under alpha .01 to6.36%. The prior apparent need for grouped baselines was partly regularization-dependent. Random geometric groups can contain context; this is not proof of semantic absence or a perfectly capacity-matched null. The continuous model also consumes native query features, so it does not explain how upstream computation constructs those features or replace the whole model.

Verification:2208 expected/actual model forwards; four baseline modes reproduce H934 exactly across nativeKL, editedKL and effecterror. Native full-logit replay max2.48e-5, no-edit identity exact, all inputs/frozen models unchanged, fit/evaluation indices disjoint, all finite, reserved-test evaluations zero. Frozen model hash matches H935. All models/seeds retained. No new natural documents; pool26,112.

Research decision: do not promote these query clusters to discovered semantic manifolds. Keep continuous prediction as the essential baseline and grouped variants as modest task-dependent alternatives. A future generalization test must freeze this comparison and introduce new lexical/template/context combinations, with competence gates and no outcome-driven model selection. Before expanding this branch, consolidate the broader featurizer/manifold acceptance ledger so geometry, reproducibility, causal specificity and reusable computation remain explicit separate requirements. Hierarchy/DAG meaning is not established by this routing surrogate.
