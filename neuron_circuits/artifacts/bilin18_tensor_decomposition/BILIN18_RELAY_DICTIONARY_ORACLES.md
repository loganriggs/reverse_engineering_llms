## H854 — Sparse selection and reconstruction objectives both limit causal fidelity

**Main result.** Frozen dictionary spans can represent substantially more of the relay effect than the learned sparse encodings express. But correcting coefficients or minimizing contrast reconstruction error on the active span does not reliably improve causal fidelity. Thus the current shortfall cannot be reduced to a single encoder bug, and reconstruction-optimal coefficients are not automatically computation-preserving coefficients.

**Four diagnostics.** All10 H852 models, all78 H849 pairs, both directions, all gap variants remain fixed. Each edit uses the same native-shape plus reconstructed contrast intervention as H853, restoring recipient radius.

1. **Learned:** original donor-minus-recipient decoder reconstruction.
2. **State LS:** independently least-squares reconstruct each input on its own frozen active atoms, retaining decoder bias, then subtract the two reconstructions. This needs no donor pair or task label to encode each state. It is a plausible inference refinement, not a new trained model.
3. **Union oracle:** least-squares project the true donor contrast into the union of both native active supports. This uses both inputs/supports jointly and is a diagnostic pairwise oracle, not a deployable single-state encoder.
4. **Full oracle:** project the contrast into all512 dictionary atoms. This has much higher rank/cost and is not a matched-capacity improvement.

For learnedshared we additionally test the101 atoms covered by any membership. This distinguishes accessible atoms from the remaining, often untrained/inaccessible dictionary columns. Every projection uses float64 SVD with a relative singular-value cutoff; ranks and conditioning retained.

**Adjacent-prompt causal preservation.** Entries are percentages of the exact-shape relay effect, seed8351/8352 separately; they are not native task accuracy or fractions of unique mediation.

|Task|Model|Learned|State LS|Union oracle|Full512 oracle|
|---|---|---:|---:|---:|---:|
|agreement|scalar48|58/73|65/82|50/61|92/92|
|agreement|scalar64|57/73|61/73|50/52|91/91|
|agreement|block8|57/72|60/77|57/75|92/89|
|agreement|shared_fixed|19/31|28/42|30/54|93/91|
|agreement|shared_learned|21/17|23/19|33/16|85/86|
|sentiment|scalar48|74/55|76/58|60/55|89/86|
|sentiment|scalar64|57/56|57/55|47/52|90/87|
|sentiment|block8|66/53|62/49|62/58|87/90|
|sentiment|shared_fixed|52/42|40/54|47/49|83/82|
|sentiment|shared_learned|17/6|17/9|20/10|75/78|

Full dictionary effects are about89–93% for scalar/block agreement and86–90% for scalar/block sentiment. However, full rank is512 versus state supports40–64 and active unions48–106. Higher rank is a major confound; this is evidence of available capacity, not proof that a512-dimensional code is a good feature explanation. The supervised one-dimensional task contrast remains a different, task-informed comparator.

**Reconstruction and causal fidelity can disagree.** Every union-oracle contrast is at least as accurate geometrically as the learned contrast, as required by least squares and confirmed numerically. Yet scalar48 seed8351 agreement preservation falls57.6%→50.4%, while geometric relativeRMSE improves .897→.754; task-effect error worsens .447→.509. Sentiment for that same model falls74.4%→60.5% preservation while geometric error improves .808→.701. Therefore a reconstruction-optimal contrast can discard or weaken directions that matter to the downstream task. These are counterexamples to using reconstruction error alone as causal faithfulness.

Independent state LS improves per-state reconstruction for every input, but can worsen contrast reconstruction and downstream effects. For example sentiment block8 seed8351 effect preservation declines66.4%→62.5%; other cases improve. This does not justify replacing all encoders with least squares without a broader behavior check.

**Shared-group diagnosis.** Learnedshared covered spans have rank101, compared with40–57 per-state active ranks and48–71 pair-union ranks. The covered-span oracle preserves about63–64% agreement and33–36% sentiment adjacent effects, versus17–22% and6–17% for learned encodings. All512 columns improve further, but unused atoms make that an inaccessible artificial capacity expansion. Coverage collapse and support/coordinate inference both contribute; neither alone explains all errors. Fixedshared has full512 coverage yet weak active behavior, so simply enforcing coverage is unlikely to solve the problem.

**What we can and cannot infer.** The total dictionary often contains useful directions spread across atoms. That does not establish an interpretable atom/group, nor show that current sparse gates select the right computational subspace. Euclidean reconstruction is insensitive to which errors downstream readers amplify. MoreoverH850 already showed equal scalar coordinates can yield different outputs under different off-axis interventions, so even a task-weighted scalar loss is not a complete fidelity criterion.

**Next discriminating experiment.** On the same frozen active spans, compare ordinary projection with a constrained minimum-error projection that preserves the independently frozen causal coordinate, using full raw-state contrasts and an exact raw-coordinate audit. This deliberately task-informed oracle asks whether the span could carry the salient scalar at reasonable edit norm. It is not an unsupervised solution and must not be advertised as one. If it restores causal behavior, test task-blind response-weighted fitting/inference on natural data with independent tasks; if it does not, investigate off-axis interactions and sparse-span limits. Track intervention norm, rank and output effects to avoid claiming success by large off-manifold edits. Revisit the20-hypothesis tournament after this bottleneck test.

**Audits.**6,864 full-model forwards. All bases orthogonal within4.56e-15; no per-state LS reconstruction-error increase and no nesting/projection-error violation. Identity logits and H853 learned-effect replay exact. All sources/checkpoints unchanged, all effects finite; no reserved-test evaluation; natural-document pool26,112 unchanged. Rank/condition metadata and all oracle edits are saved. Previous turn was progress: correct-site task-blind featurizers created the benchmark. This turn distinguishes available dictionary capacity, selected support and a substantive objective mismatch. Broad goal remains active.
