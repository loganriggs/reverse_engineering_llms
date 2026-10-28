## H1017 — Number and valence use separate causal read subspaces; a joint four-coordinate dictionary preserves both

Previous turn was progress: H1016validated a four-dimensional number-response projection across intervention constructions. This turn returns to multitask featurization by adding plain adjective valence at the same preL11residual boundary, instead of continuing only number-specific fits.

Sentiment material reuses the H966panel with negation absent: first4adjective pairs in formatAprovide8discovery states; last4pairs provide8historical evaluation states in each of formatsA/B. No individual outcome filtering. Native positive-versus-negative logit contrast is correct100%discovery,100%testAand75%testB. Exact lowercase positive/negative labels are top1on0%of states in every cohort. Therefore this is a competent two-choice valence contrast, not demonstrated free-generation classification. Other aliases/output tokens may rank higher; no alias claim is needed or established here. H966negation failures are not silently converted into successes.

Learn sentiment4from the8discovery gradients of positive-minus-negative at the final-position preL11residual. Number4is frozen from H1016. Joint4is SVDof both task gradient matrices after each is divided by its Frobenius norm (equal total task weight). Union8is QRof both task4bases and has twice the active dimension. Balanced activationPCA4uses centered discovery raw states with equal task Frobenius weighting; random4is seeded. Bases freeze before projected-edit evaluation. Number validation reuses32historical recipient states and natural-token-derived edits from H1016. Sentiment edits are opposite-valence donor-minus-recipient last-position residuals at their native magnitude, with all other positions/recipient context fixed; donors and recipients can have different prefix lengths, but only their1152-dimensional final state is transferred.

| Fixed projection | Number test error | Sentiment testAerror | Sentiment testBerror |
|---|---:|---:|---:|
| Number4 |7.2%|98.8%|99.3%|
| Sentiment4 |100.2%|8.2%|13.7%|
| Balanced joint4 |6.3%|9.7%|13.7%|
| Union8 |7.3%|8.3%|13.8%|
| Balanced activationPCA4 |21.4%|10.0%|47.9%|
| Random4 |99.8%|100.0%|99.9%|

Relative L2error compares projected and full-edit target-margin changes. Task-specific4with known task identity is a supervised selection control, not an unsupervised router. Joint4preserves both without a supplied task gate at projection time. Union8does not show a point advantage, but budgets differ; this small experiment is not a rank-optimality proof. Native suffix still generates effects during validation, so this is causal coordinate preservation, not standalone prediction of output behavior.

The task4subspaces are nearly orthogonal: canonical cosines0.0973,0.0685,0.0255,0.0041. Cross-task projection loses almost the entire target effect. Joint4has approximately2.007dimensions of squared overlap with number4and2.003with sentiment4. Discovery gradient energy retained: joint4number90.4%,sentiment99.3%; each task4retains92.4%and99.8%respectively. This supports packing mostly distinct task coordinates into a small dictionary rather than sharing the same atom across these two tasks. Numerical rotations may mix task axes inside joint4without creating shared computation.

Collateral is reported, not silently discarded. On sentimenttestA, full-edit can/willRMS0.558 versus joint4RMS0.141; is/are0.251versus0.113. On number, full positive/negativeRMS0.193 versus joint4RMS0.105; can/will0.104versus0.047. These projections preserve chosen targets while changing other effects, so they are task-oriented approximations, not full-model faithful reconstructions. Three fixed contrasts do not establish general selectivity.

Uncertainty: evaluation has only4adjective pairs, repeated across two formats and both directions; they are not16independent lexical observations. Historical evaluation states were already opened; the new evidence is the projected intervention. Number and sentiment supervision use different row counts, tasks and perturbation magnitudes; equal Frobenius weighting controls total gradient weight but not all task difficulty. This validates neither a nonlinear manifold nor a learned BSF, overlapping-group advantage, hierarchy, or unsupervised task discovery.

Audits: native number full-edit replay1.67e-5absolute; independent Torch metric replay exact; orthonormality1.89e-15; basis hash unchanged;8sentiment discovery backwards, zero evaluation gradients;finite.56prefix sequences,448suffix sequences. Native sentiment competence measured from the same suffix with exact earlier number replay control; no reservedBSFtest data or new natural documents. Pool26112unchanged; no remaining job; H961pending;broad goal active.

Next use these two validated task-oriented coordinates as supervised anchors in a shared-dictionary featurizer, with learned gates versus scalar-coordinate selection at equal active dimension. Do not impose overlapping groups when these task directions are separate. Add a third independently competent task to test actual coordinate reuse before architecture complexity; keep reconstruction, target causal fidelity, full-output collateral and task-generalization separate. Preserve existing failed H1007/H1008group-selection baselines: richer grouping must beat matched scalar supports, not merely reconstruct labels.
