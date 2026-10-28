## H614 — Combined-write compactness depends on weighting, wording and fidelity objective

H613 found distributed head contributions. This does not imply a high-dimensional combined write. Fit uncentered output-space PCA to Attention5's full contextual interaction using only old-wording items0–3 (eight endpoints). Compare raw token-row PCA with PCA after normalizing each training token row to unit norm, so large-magnitude positions do not dominate. Include one seeded orthonormal random basis and the H613 four-head baseline. Freeze bases and ranks1/4/8/16/32 before downstream evaluation.

Project each observed body-position interaction write onto the chosen output directions, then subtract that approximation on the combined prompt. Coefficients still require the actual four-condition interaction; this is conditional reconstruction, not prediction from raw text or elimination of the original model's computation. Per-token balancing changes the fitting measure, not a learned causal objective. The reference is full contextual-interaction removal. Head pruning and direction projection have different parameter/computation costs; this is a fidelity comparison, not a matched compute-budget benchmark.

There are 152 discovery token rows, eight same-wording holdout endpoints and sixteen new-wording endpoints. The familiar panel makes this adaptive transfer evaluation rather than independent confirmation. 576 interventions plus32 native baseline forwards complete.


**Within familiar wording, a few shared output directions can be useful even though no single head suffices.** On held-out items, token-balanced rank4 gives relative output KL 0.0385 versus 0.9704 for raw rank4. Yet its write reconstruction is worse (0.419 versus 0.369 error). MLP11 effect fidelity also improves (0.295 versus 0.495 error), but is far from exact. Thus retaining more write variance is not the same as preserving the downstream behavior of this intervention.

| Evaluation | Approximation | Write relative error | MLP11 effect relative error | Relative output KL | Relative margin error | Relative token CE error |
|---|---|---:|---:|---:|---:|---:|
|same_wording_holdout|head4|0.5165|0.5239|0.1771|0.3661|0.3082|
|same_wording_holdout|raw1|0.6450|0.6049|0.5529|0.7161|1.0960|
|same_wording_holdout|raw4|0.3692|0.4946|0.9704|1.0118|1.5320|
|same_wording_holdout|balanced4|0.4194|0.2945|0.0385|0.1284|0.1161|
|same_wording_holdout|raw8|0.2305|0.2707|0.0522|0.1903|0.2319|
|same_wording_holdout|balanced8|0.2413|0.2394|0.0336|0.1787|0.1635|
|same_wording_holdout|raw16|0.1150|0.1483|0.0341|0.1997|0.2509|
|same_wording_holdout|balanced16|0.1189|0.1376|0.0294|0.1769|0.2275|
|same_wording_holdout|raw32|0.0798|0.1001|0.0150|0.1433|0.1323|
|same_wording_holdout|balanced32|0.0806|0.0989|0.0144|0.1462|0.1287|
|same_wording_holdout|random32|0.9817|0.9812|0.9789|0.9317|0.9561|
|new_wording|head4|0.6076|0.6132|0.3642|0.3530|0.7096|
|new_wording|raw1|0.8441|0.8399|0.6790|0.5386|0.8546|
|new_wording|raw4|0.7709|0.8170|0.8543|0.5478|1.2975|
|new_wording|balanced4|0.7958|0.7710|0.6048|0.6161|0.7805|
|new_wording|raw8|0.7438|0.7143|0.4644|0.5955|0.6805|
|new_wording|balanced8|0.7423|0.6969|0.4290|0.5681|0.6886|
|new_wording|raw16|0.7229|0.6709|0.5004|0.7060|1.1375|
|new_wording|balanced16|0.7224|0.6698|0.4822|0.6839|1.1052|
|new_wording|raw32|0.6928|0.6209|0.2654|0.3866|0.4516|
|new_wording|balanced32|0.6923|0.6220|0.2879|0.4327|0.4796|
|new_wording|random32|0.9849|0.9816|0.8876|0.9070|0.9346|

All errors concern fidelity to full contextual-interaction removal. MLP11 effect error compares target shifts, and relative KL divides approximation-to-full KL by unchanged-to-full KL. These are not task error rates. Full copied-name top1 remains correct in every intervention. Margin and token CE are separately reported because low KL does not guarantee a selected answer statistic is equally faithful.

Rank32 gives strong familiar-wording approximation: write error about 0.080, MLP11 effect error about 0.099–0.100, and relative KL 0.014–0.015. But frozen rank32 fails to transfer nearly as well to new wording: write error about 0.692, target-effect error about 0.621–0.622, and relative KL 0.265–0.288. Random32 remains near omission in target fidelity, supporting learned structure rather than arbitrary low-dimensional compression. This is evidence for within-distribution compactness, not a global canonical form.

The two training objectives also behave differently as rank increases. Raw rank4 retains 86.3% of discovery squared write norm yet has poor held-out KL; rank8 reaches 95.1% and substantially improves output fidelity. Balanced PCA's corresponding normalized-row fractions are 84.2% and 93.6%, which are fractions under a different measure and cannot be compared directly as retained raw energy. Output KL need not decrease monotonically with reconstruction rank because downstream computation is nonlinear and errors can cancel.

Interpretation: several architectural heads can jointly implement an approximately low-dimensional operation on a specified data distribution. Which directions count as important depends on token weighting and on whether fidelity means write reconstruction, downstream target, output KL or answer loss. This directly supports studying data-based equivalence with an explicit distribution and objective, while the wording failure warns against calling the fitted directions universal features. No semantic meaning or uniqueness of these basis vectors is established.

Next compare shared versus wording-conditioned output spans at fixed total direction count, fitting both only on discovery items and evaluating held-out names. A joint basis versus two smaller conditional bases can distinguish insufficient coverage from intrinsically different local coordinates without simply increasing capacity. Report principal-angle/overlap geometry alongside interventions, and preserve the distinction between representation fidelity and a predictor that avoids observing the full response.

Audits: eight old-wording discovery endpoints verified from locked artifact; source/head vectors/runner/manifest/bases unchanged; orthonormality error <=2.39e-7; projected head sum replay <=6.47e-8; native logit replay <=2.48e-5; all 576 interventions finite and copying top1 correct. No new natural-text documents; FineWeb remains 26,112. Broad goal active and incomplete.
