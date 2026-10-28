## H1033 — Actual finite causal supervision yields modest, inconsistent selector gains

Previous turn was progress: published the verified H1032snapshot to the user-selected repository. No experiment ran during publication. This turn returns to the substantive representation objective and completes actual downstream-loss training, closing the gap explicitly left by H1031's activation-only pair supervision.

Geometry frozen: training-only PCA128 for each of the three native boundary streams. Select up to16coordinates per stream per token (48total maximum). Every architecture learns384log-importance coefficients weighting its current squared PCA coordinates. Scalar picks16individual coordinates; block picks4of32disjoint four-atom groups; shared picks2of32fixed eight-atom groups with degree2overlap and decodes their union once. Block memberships are seeded random partitions of PCA atoms, distinct from H1031's learned dictionary blocks. This tests structured support, not optimized semantic memberships or a full operation DAG. Actual shared support ranges below48; overall minimum40. Learned parameter count and native information access are equal; active budget is a matched maximum rather than exact union match.

Training: the same120discovery endpoint occurrences and60pairs, balanced family round-robin over80pair steps; each pair list cycles deterministically. Six fits (three architectures × activation/finite-causal loss), plus each untrained selector. Seed10331, Adam0.03, fixed final step, log weights clamped[-2,2],1e-4mean-square regularization. Hard exact support in forward with softmax-based straight-through surrogate gradients. These are not derivatives of the discrete top-k selection; no convergence claim.

Activation objective: normalized whole-port reconstruction plus0.1normalized paired-state difference error. Finite-causal objective: average whole-output KL plus average paired-intervention KL, evaluated in both directions through the frozen native suffix. The paired candidate is recipient state plus decoded donor-minus-recipient states at every port; target is native full-donor output. All target distributions are discovery-only, and model weights are frozen. The native suffix is used for training/evaluation only, not selector inference. Coordinate values are ordinary PCA projections; the selector learns which to retain. This is genuine finite downstream supervision, unlike gradient-only proxies or activation-difference labels.

Development retains38historically opened lexical endpoints/19pairs, fivefamilies, all frozen before fitting, including the restricted capital diagnostic cohort. Original whole-task holdouts remain unread by this experiment. No model, step, temperature, budget or threshold selected from development outcomes.

| Selector | Objective | Whole-output KL, family macro | Paired relative KL, family macro |
|---|---|---:|---:|
| Scalar |Untrained|0.6754|0.5401|
| Scalar |Activation|0.6897|0.5349|
| Scalar |Finite causal|0.6928|0.5208|
| Block |Untrained|1.0060|0.8354|
| Block |Activation|0.9843|0.7931|
| Block |Finite causal|0.9533|0.7680|
| Shared |Untrained|1.9197|0.8623|
| Shared |Activation|2.0192|0.8139|
| Shared |Finite causal|1.4215|0.8793|

Lower is better. Paired relative KL uses full-donor-to-predicted-patch divergence divided by full-donor-to-native divergence within family. Whole KL compares native output with reconstruction of all ports. These are different objectives; the shared model's whole-state improvement does not imply improved causal edits.

Finite causal versus activation paired gains are2.6%scalar and3.2%block, while shared worsens8.0%. Scalar remains strongest overall. Task behavior is heterogeneous: scalar agreement paired error0.151→0.075 and sentiment0.576→0.518, but pronoun0.298→0.345 and binding0.816→0.831. Shared sentiment worsens0.796→1.172 (worse than no-edit reference), despite improved whole-state KL. Native-answer top1 after reconstruction remains poor: family macro32.5%scalar,7.5%block,2.5%shared for finite-causal fits. No reliable group advantage or useful autonomous replacement established.

Limits: one seed/partition, tiny discovery and development panels,80steps, fixed diagonal importance model, approximate hard-gate gradients. A small difference cannot establish statistically reliable causal-supervision superiority. The low-dimensional frozen geometry also cannot recover directions absent from its span. No new-task discovery, manifold validation or unseen-task transfer has occurred here.

Audits: sixfits completed;960student suffix sequences during causal training,120native training-reference suffixes and684development suffixes,1764total. All outputs/gradients finite, maximum gradient9.66; exact hard supports and budget<=48checked. Training-source/PCA hashes unchanged; saved weights satisfy clamp and group degree invariants. Independent reduction of all summaries exact. No new corpus or original whole-task-holdout reads; natural-document pool26112unchanged. Checkpoints, all development rows, training histories and schedule saved. No process running. H1033 is local and is not part of the already pushed H1032snapshot.

Next: freeze these candidates and evaluate the pre-existing task-training-excluded copying/spatial-inversion families with the same boundary/port contract, first checking length alignment and native competence. These are historical evaluation families, not globally untouched tasks or genuine discoveries. Do not tune candidates on them. This tests whether finite supervision transfers beyond trained task families before any further selector optimization. Preserve scalar untrained/activation/causal and full-span references; retain all failures. Broad goal remains active.
