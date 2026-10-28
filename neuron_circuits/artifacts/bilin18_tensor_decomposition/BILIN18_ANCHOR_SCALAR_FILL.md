## H997 — Scalar residual fill removes most anchor-allocation cost; oracle advantage is modest

Previous turn was progress: H996 implemented and evaluated an actual activation-only shared-group router, finding weak generalization and worse fidelity than scalar selection. This turn holds the learned routes fixed and changes only residual allocation, separating coarse background blocks from anchor selection.

For every known-class anchor choice, lock its eight coordinates and select the24largest-energy remaining singleton coordinates. Exactly32unique atoms, union counted once, with identical fixedPCA128 coordinates/decoder. This is oneanchor group plus24singleton gates, not the same fourgroup computational budget as H996; it fixes atom budget while relaxing the background grouping constraint. H996 router predictions are loaded unchanged; no classifier training. Controls: pure scalar32, dense128, anchor-energy choice, one seeded random anchor per state, task-label oracle. Unknown tasks in the label oracle fall back to pure scalar32. Random anchors are an index-seeded negative control, not a deployable activation function.

Same624directed finiteMLP12 edits. Native-passing50context/fourholdout cohort macro KL ratios;zero=full donor-reference match,one=noedit.

| Policy | Context KL ratio | Whole-task KL ratio |
|---|---:|---:|
|dense128|0.091|0.792|
|scalar32|0.140|0.973|
|learned_9961|0.144|0.932|
|learned_9962|0.142|0.957|
|energy_anchor|0.140|0.987|
|label_oracle_known|0.132|0.973|
|random_anchor|0.143|0.982|

Learned-anchor context ratios improve from H9960.192/0.199 to0.144/0.142, even though the poor46% compatible-label routing is unchanged. Relaxing background blocks AND using the now-full32atom budget explains this intervention; it is not evidence that the router improved. H996's matched-count scalar controls already showed atomcount alone was insufficient, but this experiment does not quantitatively separate count from blockshape effects.

Known-label oracle plus scalar fill reaches0.132 versus scalar32's0.140, a modest~5%relative reduction in the averaged KL ratio. This is an empirical known-label reference, not a mathematical upper bound over all possible routers or anchors. Learned and random anchors are close to scalar selection; this panel supplies little justification for an extended classifier-tuning program. Holdout scores remain weak (~0.93–0.99); the learned-policy differences do not constitute discovery of copying/spatial mechanisms, especially since those states are forced into known valence anchors.

Interpretation: the shared dictionary can retain a supervised anchor alongside flexible unlabeled residual capacity with only a small familiar-context fidelity cost. It is still not an ideal or uniquely identified featurizer. Task recognition, semantic grouping and novel-task discovery remain separate unproven properties. We should not keep optimizing small changes on the same reused development panel and call that progress toward discovery.

Audits:5,616/5,616forwards; identity logits2.60e-5; source/router hashes unchanged. Independent checks confirm exactly32unique atoms for every sparse state, all eight locked anchor atoms retained, no residual double-counting, and learned routes bit-identical to H996. Scalar/dense control KL replays within1.09e-6; all cohort ratios independently reaggregated. No refitting, no reserved-test access. Natural-document pool26,112.

Research decision: end this narrow routing/allocation sweep. Freeze scalar32 and the anchored+singleton alternatives as faithful-enough development baselines for exploratory interpretation, without claiming semantic validation or selecting a seed by these results. Next inspect existing natural-text discovery-data manifests, identify a permitted unlabeled discovery sample disjoint from reserved tests, and capture the common-site activations needed to inspect residual structure. Generate candidate task hypotheses from reproducible activation/context patterns and require fresh competence and causal tests before adding labels or retraining. Keep known whole-task holdouts excluded from all fitting; do not relabel evaluation examples as new discoveries. The actual bootstrap loop now needs a bounded discovery pilot, not another anchor-classification variant. H961 context-bridge predictor evaluation remains pending; broad goal stays active.
