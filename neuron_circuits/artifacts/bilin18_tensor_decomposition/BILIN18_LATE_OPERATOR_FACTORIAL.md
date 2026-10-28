## H639 — Later MLP context explains much of wording transfer, while item transfer often requires coordinated attention

H638 was progress: a validated block9/input–downstream–readout factorial localized substantial prompt-dependent variation to the composition of incoming responses and later operators. H639 decomposes those later operators mechanistically.

At every block10–17, independently choose receiver or matched-donor values for three factors: C, bilinear MLP operating context; N, MLP RMS-normalization derivative; A, the normalized-attention Jacobian. A includes attention's own normalization, so N refers only to MLP normalization. Run all eight combinations, crossed with receiver/donor incoming block9 tangents and receiver/donor final readouts, for both donor maps and the three fixed probes. Factor exchanges are applied consistently across the eight later blocks. Residual skip weights are unchanged and shared across contexts. No fitting, prompt changes or output selection occurs.

The metric remains the norm of the remaining donor–receiver full-logit derivative gap under receiver probabilities. Lower is better;1 means no improvement relative to native receiver. These are local derivative counterfactuals, not finite task interventions. An all-donor operator is a validated endpoint, not an independently learned mechanism. Analytical factorizations reproduce both native and donor whole-block chains from H638 within1.35e-6 relative residual error.

**For wording changes with incoming tangent and readout already matched to the donor, later bilinear context is the strongest individual factor.** It reduces the gap from0.510–0.704 to0.192–0.353 across task families. Normalization alone offers little benefit; attention alone offers smaller or inconsistent benefit. Additional normalization or attention improves over context alone, and all three recover the donor to numerical tolerance.

| Cross-wording family; donor input/readout | Native later operators | C | N | A | C+N | C+A | N+A | C+N+A |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|copy|0.704|0.353|0.729|0.605|0.268|0.257|0.625|0.000|
|fact|0.637|0.241|0.614|0.643|0.101|0.178|0.620|0.000|
|plural|0.619|0.270|0.646|0.592|0.209|0.152|0.600|0.000|
|agreement|0.572|0.192|0.574|0.536|0.140|0.118|0.539|0.000|
|addition|0.510|0.293|0.535|0.478|0.218|0.200|0.488|0.000|

This is a conditional mechanistic result: it concerns which later factors reproduce the donor transformation when the donor incoming response and donor readout are available. It does not mean context replacement alone would reproduce the donor response starting from the receiver. With receiver incoming tangent and receiver readout, context-only remaining gaps are0.854–0.992; even complete donor later operators leave0.783–0.917, reproducing H638. The strong improvement in the first table depends on the incoming response and readout.

**Within-wording item changes often require attention and MLP context together.** The same factorial gives a qualitatively different pattern:

| Within-wording family; donor input/readout | Native later operators | C | N | A | C+N | C+A | N+A | C+N+A |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|copy|0.748|0.547|0.765|0.566|0.538|0.135|0.577|0.000|
|fact|1.132|1.208|1.074|0.932|1.346|0.401|0.864|0.000|
|plural|0.954|1.008|0.928|0.590|1.092|0.206|0.578|0.000|
|agreement|0.650|0.292|0.653|0.628|0.134|0.229|0.638|0.000|
|addition|0.755|0.441|0.771|0.527|0.422|0.146|0.544|0.001|

For facts, C alone worsens the gap1.132→1.208; C+N worsens it further to1.346. A alone lowers it to0.932, while C+A lowers it to0.401. Plurals show a similar conditional interaction: C alone worsens0.954→1.008, whereas C+A reaches0.206. Copying and addition also benefit substantially from coordinated C+A. Agreement differs: C alone already helps, and C+N is better than C+A. Thus a universal ranking of attention, MLP context and normalization is not supported.

These findings are consistent with a distinction between construction-related and item-related operator changes, but do not establish a strict division of labor. The within controls change lexical item and, for facts, target identity; cross controls change wording, position, tokenization and sometimes continuation competence simultaneously. Probability-weighted gap norms are relative to each pair's gap, so within/cross numerical ratios are not direct comparisons of absolute effect size. Full context independence is not assumed for any component.

The current most useful model of the evidence is a coordinated sequence of context-dependent transformations. Bilinear MLP operating contexts carry substantial information needed to transfer late computation across wording, conditional on matching the incoming response. Attention and MLP context often need to change together for item transfer. This goes beyond the observation that derivatives differ, but it still uses native donor operators rather than a compact predictive representation or semantic algorithm. The fundamental-unit question remains open.

Refreshed20-hypothesis map, with priorities based on current evidence:

1. Bilinear operating contexts capture construction-related variation in late transformations. Strong current support conditionally; prospectively test matched new wordings.
2. Coordinated attention and MLP context carry item-specific transformation differences. Supported for several within-template contrasts; test novel items and more donor pairings.
3. Incoming-response compatibility is essential to interpreting a later operator. Strong support from H638–639 factorials.
4. Readout effects depend on the incoming/internal response. Strong support; retain readout as an explicit factor.
5. A small fixed output-direction basis explains all responses. Weakened by held-out projection gaps.
6. Residual passthrough explains the apparent need for many units. Explains part of early held-out-direction error, insufficient later.
7. Task labels identify invariant operators. Weakened by wording and item effects.
8. Construction labels identify invariant operators. Plausible approximation, limited by agreement and within-template effects.
9. Normalization is negligible everywhere. Rejected as a general rule: it improves matched combinations and matters for agreement/item transfer.
10. Normalization alone drives the observed changes. Weak on these panels.
11. Attention alone accounts for item-specific transfer. Insufficient for facts/plurals; coordination matters.
12. Later MLP context alone accounts for item-specific transfer. Weakened by worsening on facts/plurals.
13. Independent layer/component effects add into a sufficient explanation. Weakened by factorial interactions and joint versus additive results.
14. Finite-dose curvature is the main source of mismatch. Weak for the small doses already validated; not settled at larger doses.
15. The original selected vector is uniquely special. Weakened by control probes, though it sometimes transfers better.
16. A shared internal low-dimensional basis rescues universality. Still possible with richer structured discovery, but tested spans remain insufficient.
17. A low-dimensional basis of *operator changes* transfers better than response bases. Open and promising after prospective replication; do not infer it from response spectra.
18. Apparent semantic structure comes from generic context mismatch. Still viable; same-wording controls and new minimally different prompts are needed.
19. Poor competence creates the whole phenomenon. Insufficient for competent copying/facts, but material for plural/addition interpretations.
20. Positional or tokenization changes masquerade as construction effects. Open: current template swaps are not controlled minimal pairs.

Next prioritize1,2,3,18,20 with a frozen prospective replication on copying and capital facts. Cross new items with matched paraphrases sharing an answer-introducing ending, include same-wording donor exchanges, and retain exact competence/format checks. Freeze the C/N/A contrasts and input/readout choices before evaluating results. This will test whether the wording-versus-item distinction survives improved controls rather than continuing to explain the same80 prompts. Only after replication should an operator-change basis be fitted and evaluated on held-out contexts/directions.

Audits:640 cached attention-stage evaluations;5376 batched JVP calls; no new whole-model forwards; native/full-donor factorization matches H638 below1.35e-6; all source/runner hashes unchanged; finite metrics. All five families, two donor conditions and three probes retained. Existing competence limitations and retrospective probe history carry over. No new natural-text documents; FineWeb remains26,112. Broad goal active and incomplete.
