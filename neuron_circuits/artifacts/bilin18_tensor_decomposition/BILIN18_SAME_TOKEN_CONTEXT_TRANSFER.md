## H846 — Same-token context transfer identifies a later, cleaner representation target

**Main result.** Holding token identity and position fixed removes H845's donor lexical-mean mismatch. Agreement has a small but clear contextual effect at preMLP2; frozen scalar featurizers preserve it partially, and current block/shared alternatives preserve less. Much stronger same-token, nonfinal-position transfer appears at preMLP4 for agreement and sentiment. This is a better site for studying contextual computations than the weak early prediction-position site.

**Design.** All76 equal-length competent H840/H843 pairs across five families, both directions. At every18 preMLP sites, swap donor raw state at every token-identical/index-identical position, jointly; separately exclude final position. Different-token positions are never patched. Recipient original embeddings and persistent layer0 values remain native. At sites1/2 additionally restore recipient radius with exact donor normalizedshape, or with recipient shape plus frozen decoded donor-minus-recipient difference for all five H836 models/both seeds. Token means cancel exactly at edited sites, even when the common token uses the global fallback. This removes lexical identity changes at those sites; it does not remove all distribution shift or training-coverage effects on encoding.

**Early agreement result at site2.** Percentages are aggregate donor-directed answer-margin change divided by native donor-recipient gap. Error compares individual directed intervention effects with exact shape swaps, retaining both seeds. It is not activation reconstruction error or output KL.

|Representation|Gap transferred|Effect relative RMSE to exact shape|
|---|---:|---:|
|raw|4.93%|0.018|
|shape|4.96%|0.000|
|scalar48|3.09%|0.465|
|scalar64|3.11%|0.492|
|block8|1.89%|0.714|
|shared_fixed|2.53%|0.685|
|shared_learned|1.00%|0.870|

Exact-shape agreement transfer is4.96%, with descriptive paired16-item bootstrap interval[3.97%,5.95%]. Scalar48 transfers3.09%, scalar64 3.11%, block8 1.89%, fixed shared2.53%, learned shared1.00%. Scalar48's individual-effect error is0.465 vs scalar64 .492; this is no universal architecture ranking, especially with unmatched effective support and shared parameters. The effect is modest, so confirmation and a fixed-subspace baseline remain needed.

Sentiment has negative mean exact same-token transfer at sites1/2 (-2.16%/-2.47% with recipient radius). Some learned reconstructions flip the sign. A more positive answer effect is not greater faithfulness: the intended reference is the exact intervention, including negative effects. Capital/binding/copy early same-token effects are tiny or uncertain; relative-error ratios there should not be used as meaningful rankings in isolation.

**Depth/position result.** At preMLP4, all-same-token transfer is42.2% for agreement and38.4% for sentiment; excluding the final position still gives40.9% and39.8%. Hence a substantial intermediate contextual route is available outside the prediction token. By comparison capital/binding/copy nonfinal same-token transfer stays weak over most depths, consistent with their persistent-value routes. Later final-position transfers are large, but proximity to the output makes them weaker evidence for a reusable intermediate unit. H847 localizes every eligible same-token position atL4, without selecting by effect beforehand.

**Updated hypothesis tournament (20 live alternatives).**

1. Sparse scalar coordinates capture a portion of early agreement context — supported byL2 causal preservation; requires fresh replication.
2. Learned groups are universal task units — weakened by support/coordinate instability and current causal errors.
3. Fixed shared groups outperform learned memberships because optimization collapses coverage — still plausible; not proof of meaningful groups.
4. BSF-like blocks encode context within a subspace rather than fixed axes — possible, but current task fidelity does not establish it.
5. Dictionary reconstruction misses low-variance causal differences — plausible; compare fixed/response-weighted subspaces on same-token contrasts.
6. Apparent contextual failure was entirely unseen-token leakage — weakened forL2 agreement, important forH845 pooled effects.
7. Token means are biased by their sampled contexts — unresolved even for seen tokens; same-token subtraction reduces donor-mean mismatch.
8. Early task differences are mostly lexical storage — supported for changed-token routes, insufficient forL4 same-token effects.
9. Context is relayed into a few unchanged intermediate tokens — now plausible; single-position intervention next.
10. Contextual effects require many positions jointly — also plausible; compare singles and joint swaps.
11. Agreement and sentiment share a reusable contextual relay computation — not yet established; localization may reveal different positions or operators.
12. Persistent value content carries recall/binding/copy donor differences — supported byH844 and earlierH537/H739.
13. Changing routing is necessary for those donor effects — weakened by prior fixed-reader controls, while native routing still matters.
14. A fixed value/output map plus contextual coefficient is a useful computation unit — supported for tested earlier heads, generalization remains open.
15. Gate behavior reflects format/position rather than abstract operations — still live, requires crossed controls.
16. Radius is the primary causal variable — weakened at current early same-token sites by raw/shape agreement.
17. Final-position early negative findings show no early computation — contradicted by source andL4 nonfinal context routes.
18. Reconstruction residual carries useful task information omitted by learned coordinates — compatible with errors; isolate residual interventions.
19. A computational graph should represent content sources, readers and transformations rather than only nested semantic labels — plausible design direction, not yet learned/recovered evidence.
20. Fresh cross-format lexical generalization will distinguish a reusable unit from panel-specific correlations — necessary next confirmation after exploratory localization.

**Priority.** Localize L4 contextual transfer, then test low-dimensional/featurized explanations at that actual site with natural-data fits and independently frozen validation. Do not applyL1/L2 dictionaries atL4 or relabel block-output representations as preMLP representations. Keep useful scalar baseline and alternative architectures, but require causal-effect fidelity before semantic manifold claims.

**Audits.**9,272 full forwards; native runner, all-layer identity edits and states before first changed token match exactly. Every edited token/index identity asserted. All hashes unchanged, metrics finite; no reserved-test use; natural-document pool26,112 unchanged. Previous turn was progress: it exposed the lexical-mean confound that this experiment directly controls. Broad goal remains active.
