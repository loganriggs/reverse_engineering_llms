## H996 — Actual activation-only anchor routing fails to improve the faithful scalar baseline

Previous turn was progress: H995 validated supplied supervised subspaces, while explicitly leaving online routing unproven. This turn implements actual gates over a shared dictionary, keeping unlabeled background capacity, and evaluates the resulting featurizer without evaluation labels selecting its anchor.

FrozenPCA128 dictionary/coordinates. Six H994 supervised eight-atom anchor groups plus16PCA-ordered eight-atom background groups cover everyatom. An activation-only linear classifier chooses one anchor; three background groups then fill the budget greedily by uncovered coordinate energy. Exactlyfour activegroups, union decoded once, actual activeatoms<=32. Two classifier seeds9961/9962 trained600Adam steps on144discovery states with training-only per-coordinate scaling. Label loss treats sentiment/negation as compatible anchors to avoid contradictory supervision on duplicated positive prompts. The rest of the dictionary/coordinate encoder is frozen.

Controls: background-only fourgroups, energy-based selection over all22groups, explicit task-label oracle plus threebackground groups, dense128 and scalarTop32. Every group policy has a scalar control matched to its EXACT per-state union cardinality. Task-label oracle falls back to background-only on copying/spatial and other unknown families; do not interpret that fallback as a correct-task oracle. The learned router has no unknown class and is forced to select a known anchor on holdouts.

Same H991validation panel and reference intervention. Native-passing50context/fourholdout cohorts, independent per-state prediction. Table lowerisbetter;one=noedit. Atom counts shown for competent-context endpoints.

| Policy | Mean active atoms | Context KL ratio | Whole-task KL ratio |
|---|---:|---:|---:|
|dense128|128.00|0.091|0.792|
|scalar32|32.00|0.140|0.973|
|background_only|32.00|0.172|0.949|
|energy_all|30.93|0.174|0.961|
|learned_9961|27.41|0.192|0.995|
|learned_9962|27.25|0.199|0.977|
|label_oracle_known|28.59|0.166|0.949|
|matched_background_only|32.00|0.140|0.973|
|matched_energy_all|30.93|0.142|0.995|
|matched_learned_9961|27.41|0.146|0.976|
|matched_learned_9962|27.25|0.146|0.980|
|matched_label_oracle_known|28.59|0.145|0.973|

Compatible task-label accuracy on400native-passing validation endpoints is46.0%/45.5%. The supervised router mostly misroutes holdouts into valence anchors: seed9961 assigns26/32negation and6/32sentiment; seed9962 assigns31/32negation and1/32sentiment. These are forced known-class labels, not discoveries or evidence that holdout mechanisms are valence-like. A constant valence-compatible prediction would already match the40% sentiment/negation fraction of this validation set, so the headline46% is weak. Training accuracies and single-state replay errors are saved in config; no validation tuning occurs.

The actual routed featurizer has been implemented and tested, but it does not improve the scalar baseline. Matched-count controls confirm the deficit is not solely fewer active atoms. Even explicit task-label oracle+background groups reaches0.166, worse than scalarTop32 at0.140; better task classification alone therefore cannot be assumed sufficient. Fourgroup allocation, background-block constraints and task-margin-selected anchors versus full-distribution fidelity may all contribute. This is a specific fixed-anchor/linear-router architecture, not a general impossibility result for overlapping groups.

Unlabeled capacity remains16fixed background groups, not discovered task categories. Nothing here implements a successful novelty detector, hierarchy or computational DAG. This turn closes the routing implementation gap without converting its failure into a discovery claim.

Audits:8,736/8,736full-model forwards. Identity logits2.58e-5, source hashes unchanged and finite outputs. Exact fourgroup gates, complete dictionary coverage, shared union-once decoding and per-state scalar budget matching independently replayed. All144training indices are discovery-only/non-holdout with no exact prompt overlap to validation; classification loss uses only these labels. Router single-example versus batch inference checks passed at1e-4 logits. Dense/scalar controls replay prior KL within7.49e-7 and all cohort ratios independently reaggregated. No reserved-test access; natural-document pool26,112.

Next decisive control: hold anchor choices fixed and fill the remaining atom budget with individual high-energy coordinates rather than coarse background blocks, always using exactly32unique atoms. Compare learned, energy and task-label oracle anchors against scalar32. If even oracle anchors fail then improving classification is the wrong next target; if oracle anchors help but learned gates fail, routing is the actionable bottleneck. This separates group usefulness from budget/allocation cost before another training sweep. Preserve whole-task holdouts and candidate shared valence semantics. Bootstrap unlabeled discovery/retraining remains outstanding; H961 context-bridge predictor evaluation remains pending.
