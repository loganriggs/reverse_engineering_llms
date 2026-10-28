## H615 — Joint coverage supports a shared output span without a wording-specific router

H614's old-wording span transferred poorly. Fit token-balanced output spaces on discovery items0–3 in both wordings, and evaluate items4–7 separately in each. Compare a single shared rankB basis with two wording-specific rankB/2 bases, for total stored-direction budgets B=8/16/32/64. Give each wording equal total weight in the joint PCA fit, despite different body lengths. Include single-wording rank32 cross-transfer diagnostics and a seeded random64 control. All choices are fixed before downstream evaluation.

The comparison matches stored vector count, not active rank or routing cost: the shared model uses B coefficients per token, while the conditional model uses B/2 and an explicitly provided wording label. Each projection observes the actual interaction write. Neither model predicts it from text or removes the cost of computing the original interaction. The evaluation panel is familiar and adaptive; fitting excludes evaluation names, but this is not independent confirmation on an unseen format.

**Joint coverage repairs most of the former transfer failure.** Shared rank32 gives held-out write error 0.114/0.097 and MLP11 effect error 0.127/0.114 in the two wordings. Shared rank64 gives target-effect error 0.090/0.084. Wording-specific coordinates are therefore not required for this level of approximation on these two distributions.

| Held-out wording | Total directions | Model | Write error | MLP11 effect error | Relative KL | Relative margin error | Relative token CE error |
|---|---:|---|---:|---:|---:|---:|---:|
|0|8|joint|0.3616|0.2799|0.0317|0.1294|0.1115|
|0|8|conditional|0.4194|0.2945|0.0385|0.1284|0.1161|
|0|16|joint|0.2397|0.2157|0.0261|0.1257|0.1434|
|0|16|conditional|0.2413|0.2394|0.0336|0.1787|0.1635|
|0|32|joint|0.1141|0.1266|0.0283|0.1834|0.2169|
|0|32|conditional|0.1189|0.1376|0.0294|0.1769|0.2275|
|0|64|joint|0.0722|0.0900|0.0147|0.1286|0.1367|
|0|64|conditional|0.0806|0.0989|0.0144|0.1462|0.1287|
|1|8|joint|0.3917|0.3214|0.1433|0.1610|0.4679|
|1|8|conditional|0.4620|0.4501|1.3202|0.7623|2.0177|
|1|16|joint|0.2267|0.1711|0.0763|0.1372|0.4017|
|1|16|conditional|0.2707|0.2120|0.2865|0.3572|0.9275|
|1|32|joint|0.0971|0.1144|0.0550|0.1276|0.3077|
|1|32|conditional|0.1079|0.1188|0.0783|0.1814|0.4081|
|1|64|joint|0.0670|0.0839|0.0233|0.1079|0.2181|
|1|64|conditional|0.0743|0.0940|0.0204|0.1108|0.1935|

The shared basis generally matches or improves on conditional bases at equal total stored directions, especially at small budgets. For new wording, joint rank8 relative KL is 0.143 versus 1.320 for the conditional rank4 used within its total8 budget. This does not prove an inherent advantage of shared coordinates: the shared representation has twice as many active directions per token, and PCA optimizes reconstruction under its weighting rather than downstream KL. At total64, the two approaches have similar fidelity, with conditional KL/CE slightly better in some cells.

The separate spaces are not identical. Their mean squared principal-angle cosine is about 0.302 at rank4, 0.321 at rank8, 0.273 at rank16 and 0.288 at rank32. This is a geometric overlap measure, not a percentage of shared semantic features. Old-only rank32 has new-wording write/target errors 0.693/0.621; new-only rank32 has old-wording errors 0.644/0.636. Yet a single span fitted to both covers held-out names well. Different local directions do not force disjoint computational units.

All fidelity metrics refer to the full contextual-interaction removal. They are not task error rates or causal shares. All interventions retain correct copied-name top1. Rank64 shared output KL errors are about 0.015/0.023, but intended-token CE errors remain 0.137/0.218, so no universal faithfulness threshold is met across all objectives. A random rank64 span remains near omission in target-effect fidelity (0.947/0.923).

A refreshed hypothesis set connects these local results to the broader goal:

1. The Attention5 interaction is distributed across heads — supported on this panel.
2. A single dominant head is sufficient — weakened by H613.
3. High write variance guarantees output fidelity — contradicted by H614's weighting comparison.
4. Token weighting materially changes useful coordinates — supported locally.
5. Shared output coordinates can cover both known wordings — supported at finite error.
6. Separate wording-specific units are necessary here — weakened by the joint-basis result.
7. The two separately learned spans are nearly identical — not supported by overlap geometry.
8. Different wording spans encode the same abstract operation in different coordinates — open; geometric approximation alone cannot establish semantics.
9. A universal additive prefix direction predicts all contexts — weakened by H604–H605.
10. A common shift plus normalization explains the dependence — weakened by H605.
11. Boundary-token changes explain most Attention0 interaction — supported by H609's control.
12. Tokenization explains the entire later interaction — weakened by substantial remaining interaction.
13. A large MLP0 cross term drives the later target — weakened by H610's body removals.
14. Later non-final writers materially influence the target — supported by H611.
15. Attention5's direct persistent-value interaction dominates — weakened by H612.
16. Persistent values contribute upstream to local routing/value formation — still open.
17. Routing alone is the relevant computational unit — weakened by factor removals.
18. Coordinated contextual values and routing form the relevant local object — supported, with baseline-dependent decomposition.
19. Preserving this interaction is necessary for copied-name accuracy — not demonstrated; top1 survives interventions.
20. The shared span transfers to unseen constructions without growing rank — open and the highest-value next test.

Most likely current account: context-dependent computations can share a compact, data-dependent output representation when discovery covers the relevant regimes. Architectural units and fitted coordinates are different descriptions. The evidence supports neither a unique canonical basis nor universally stable semantic directions. The important next question is how required dimension grows as genuinely new constructions enter the distribution, not whether we can fit the two familiar wordings more closely.

Next freeze the shared rank32/rank64 bases for new lexical items crossed with familiar and previously unseen constructions, with token-boundary alignment and competence checks retained. Do not refit after viewing failures. This separates lexical generalization from construction generalization and tests the strongest remaining hypothesis directly.

Audits: sixteen discovery endpoints across both wordings, item IDs0–3 verified from locked artifact; evaluation IDs4–7 excluded; source/head vectors/runner/manifest/bases unchanged; orthonormality <=2.39e-7; native replay <=2.48e-5; 416 interventions plus32 native baseline forwards; finite metrics and correct copying top1 throughout. No new natural-text documents; FineWeb remains 26,112. Broad goal active and incomplete.
