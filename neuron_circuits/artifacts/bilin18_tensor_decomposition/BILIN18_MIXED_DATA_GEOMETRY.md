## H1004 — Broader geometry makes the supervised anchor useful, with an old-task tradeoff

Previous turn was progress: H1003 demonstrated that a causal contrast addition improved state reconstruction but worsened new-control output KL in the task-only span. This turn compares task-only, natural-only and mixed discovery geometry at exactly128dimensions, with optional inclusion of the same frozen nounrank1 anchor. All fits freeze before model evaluation;8320forwards complete.

Task data:144historical discovery slots/132unique prompts from original six tasks. Natural data:768discovery states from192documents, separate from64natural confirmation documents. Mixed PCA minimizes the average of the two domain mean squared errors, using their equally weighted mean as center. It equalizes sample-count contributions, not domain activation energies. Anchored arms reserve one direction for q and use127leading directions from the respective centered training data after projecting orthogonal to q. This is a constrained rank128dictionary, not adding a129thdirection. Two center-swap diagnostics preserve natural/mixed bases but use the original task mean.

Evaluation matches H1003:896states including diagnostic failures, with primary summaries on400old task states/50passing cohorts,40new control states/fivepassing cells, and256natural states/64documents. Whole-state MLP12 reconstruction; KL(native || reconstructed output). These are reused development evaluations, not final untouched tests. Top1 agreement measures native-output fidelity.

| Rank128 representation | Old-task KL | New-control KL | Natural KL |
|---|---:|---:|---:|
| Task-only |0.003093|0.033368|0.033326|
| Task + forced anchor |0.003096|0.033766|0.033124|
| Natural-only |0.025075|0.032691|0.026562|
| Natural + forced anchor |0.025622|0.025634|0.026567|
| Mixed |0.003373|0.034291|0.028645|
| Mixed + forced anchor |0.003371|0.026423|0.028605|

The mixed anchor reduces new-control KL22.95% relative to mixed without anchor, with negligible old-task/natural changes (-0.063%/-0.140%). Natural anchored improves new-control KL21.6% relative to natural alone but retains the natural-only representation's large old-task deficit. In the task-only geometry, anchoring still fails to improve new-control KL. Thus the anchor's utility depends on the surrounding representation; a fixed contrast is not an independently sufficient unit of good reconstruction.

Relative to oldtask128, mixed anchored improves new-control KL20.8% and natural KL14.2%, but increases old-task KL9.0% (absolute increase~0.000278nats). Top1 agreement old/new/natural is97.25%/95.0%/90.23% versus97.5%/95.0%/86.72% for task-only. This is a tradeoff, not uniform dominance. Mixed without anchor reduces new-control top1 agreement to87.5%; the forced anchor restores95%. With only40eligible new states, two templates/limited subjects and reused vocabulary, avoid broad generalization or significance claims.

Center diagnostics: natural128 with old task mean has KL0.015972/0.047043/0.030249 on old/new/natural, compared with0.025075/0.032691/0.026562 using its own mean. Center choice contributes materially to natural-only domain tradeoffs. Mixed basis with old center is almost unchanged (0.003376/0.034307/0.028631 versus0.003373/0.034291/0.028645), so its behavior is not primarily a mean correction. Reconstruction ratios use the same old-task-centered denominator across arms for comparability.

This provides a concrete example where a controlled supervised direction helps a broader data-trained dictionary at fixed rank. It does not yet establish learned block structure, a meaningful manifold, an improved sparse featurizer, or a generic noun feature. Constrained direction versus unconstrained rank128 is tested; matched random forced directions in mixed geometry remain a needed control before attributing the effect uniquely to this supervision. Discovery basis fit and held-out causal transfer are distinct questions.

Next freeze the mixed dictionary and compare the real constraint against seeded random constraints at the same rank on whole-state and paired-edit fidelity. Then test sparse coordinate/group allocation within the supported geometry, carrying the narrow anchor label and original task holdouts forward. Do not return to an extended noun mechanism investigation. The central research loop now has a promising joint representation candidate, but needs random-constraint, causal fidelity and sparse-allocation validation before adoption.

Audits:8320/8320forwards,8064unique state/arm results. Identity logit maximum error3.96e-5; natural prefix capture relative error3.78e-6. Original task projector replay error5.21e-15. Independent discovery-only fit replay error0 for all six fitted bases; task training/evaluation prompt overlap0, no new control prompts fitted, natural train/confirmation document overlap0. Independent54saved-logit sample metrics/reconstruction errors0 and all summaries/macros replayed. Sample audit is not a full-logit rerun. Source and frozen basis hashes unchanged; no reserved tests. Natural-document pool26,112 unchanged. H961 pending; broad goal active.
