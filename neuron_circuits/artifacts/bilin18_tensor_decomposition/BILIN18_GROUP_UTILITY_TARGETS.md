## H1008 — Discovery-only group-choice oracle has modest, nonspecific gains

Previous turn was progress: H1007 found real groups better than shuffled groups for old causal contrasts but worse than scalar selection. H1008 collects downstream utility targets on discovery only:132deduplicated old training prompts,8noun discovery endpoints and768natural discovery states from192documents,908total. No evaluation/confirmation examples used.22sparse candidates consist of scalarTop32 plus seven real group-plus-fill options and seven options in each of two shuffled panels. Every candidate uses exactly32coordinates. Identity and dense128 add two references.22560forwards complete (768native references plus908x24state/arm evaluations).

For each proposed group, force its coordinates and fill remaining slots by largest absolute coordinate, independently for each state. Shared dictionary and center stay frozen. An oracle chooses lowest native-output KL among scalar and seven panel options. Because scalar is included, oracle improvement is guaranteed nonnegative and is not deployable performance. Compare matched-choice shuffled panels, not zero gain alone.

| Discovery domain | Scalar32 KL | Real oracle KL/reduction | Shuffled oracle reductions |
|---|---:|---|---|
| Old tasks,132states |0.0002701|0.0002598 /3.81%|4.22%,4.83%|
| Noun,8states |0.0211888|0.0209665 /1.05%|2.26%,2.78%|
| Natural,768states |0.0216078|0.0198862 /7.97%|8.16%,8.35%|

Real groups do not beat shuffled choices even with downstream oracle selection. Scalar32 is the best single fixed real-panel action in every domain. Natural improvement is skewed: median per-state gain0.000189nats versus mean0.001722, maximum0.15631. This supports modest instance-dependent support-selection opportunity, but not a group-specific benefit. Discovery old-task KL is much lower than validation because the dictionary used these data; no generalization claim follows.

Decision: do not train a complicated semantic group gate on this evidence. Retain utility targets for future output-aware support prediction, but label current proposed groups unsupported as superior sparse units. The user meanwhile asked how to quantify early-layer understanding; inspect H683 and H981/H982 and distinguish early activation reconstruction from local causal-channel prediction and full-prefix reverse engineering. Current layer12 metrics cannot stand in for first-layer coverage. Further work should define a broad layer-local prediction/fidelity benchmark before claiming a percentage of understanding, while retaining the user's multi-task featurizer objective.

Audits:discovery source membership independently verified,132old prompts unique, every support independently replayed at exactly32coordinates, all oracle choices independently replayed.72saved-logit samples reproduce KL exactly. Source/input hashes unchanged, finite outputs, no evaluation states used and no router training. Natural-document pool26,112unchanged. H961pending; broad goal active.
