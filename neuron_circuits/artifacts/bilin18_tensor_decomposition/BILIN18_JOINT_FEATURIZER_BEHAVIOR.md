## H985 — Frozen joint featurizers show modest familiar-task behavioral fidelity and little whole-task transfer

Previous turn was concrete progress: H984 completed the matched pair-change comparison and independent audit. This turn applies the actual frozen H983/H984 model predictions as finite MLP edits, resolving whether their activation-space errors hide useful downstream fidelity. No retraining, model selection or extra optimization.

At last-token MLP12, add decoded donor-minus-recipient write differences to the native recipient write. Reference replaces that write with the complete donor write. This preserves the native recipient residual outside the edited contribution. Evaluate96 non-discovery pairs/192 directed edits across eight families, all three development splits. Twenty frozen candidates: four H983 real-answer-label models and sixteen H984 reconstruction/pair-change/shuffled-change/task-mean-trained models. Both architectures and both seeds retained. Also evaluate identity and direct task-mean discovery deltas for the six trained families. No direct task-mean control is fitted for copying/spatial holdouts.

Metric KL(reference full donor edit || candidate edit) / KL(reference full donor edit || native). Zero is perfect reference reproduction; one is no change. Ratios sum KL across endpoints within each family/split, then the table averages equally over native-competent cohorts and both seeds. Fourteen trained-task cohorts and two holdout cohorts pass the prior native gate; failed cohorts are retained separately in raw results. Mean reference KL across these cohorts is0.01963 for trained families and0.01607 for holdouts. This measures fidelity to a particular MLP intervention, not whole-model task accuracy, recovered percentage of computation or semantic validity of individual groups.

| Architecture / training objective | Trained-task validation KL ratio | Whole-task holdout KL ratio |
|---|---:|---:|
|block / reconstruction|0.987|1.053|
|block / true_labels|0.939|1.063|
|block / paired_changes|0.798|0.981|
|block / shuffled_changes|0.761|0.990|
|block / task_mean_changes|0.763|0.971|
|shared_fixed / reconstruction|1.560|1.136|
|shared_fixed / true_labels|1.091|1.075|
|shared_fixed / paired_changes|0.901|0.990|
|shared_fixed / shuffled_changes|0.958|1.048|
|shared_fixed / task_mean_changes|0.987|1.005|
| Direct task-mean edit, using family and endpoint labels |0.610|Not fitted|
| No change |1.000|1.000|

Task-change supervision improves familiar-task behavior compared with reconstruction alone. For blocks, correct pairs, shuffled within-family pairs and task-mean targets have comparable aggregate performance, so correct example pairing has not established a special advantage. Shared paired-change models improve substantially over their reconstruction baseline, but still leave most reference discrepancy. Copying/spatial transfer stays close to no change, despite modest familiar-family gains.

The direct task-mean edit is stronger than all learned-model aggregates here. It uses explicit task identity and change orientation, whereas the frozen featurizer reconstructs each state independently; it is a labeled diagnostic, not a fair unsupervised discovery algorithm. Its advantage demonstrates available coarse task structure that the tiny-data representation has not reliably captured, rather than establishing that task means are fundamental computational units. A common layer is also a limitation: MLP12 is not the strongest site for every family.

Audits:4,368/4,368 full-model forward examples (192 complete donor references,192 identity,144 direct task means,3,840 frozen candidate edits). Identity logits agree with saved native outputs within3.05e-5. Independent replay against96 prior H963 full-donor edits gives maximum task-margin discrepancy1.12e-5 and KL discrepancy3.09e-7. All cohort ratios independently reaggregated exactly. All source/model/checkpoint hashes unchanged, finite outputs, no training or reserved-test access. Natural-document pool remains26,112.

Research decision: stop extending this48-state objective sweep. It has now been tested in activation and output spaces, and lacks useful whole-task transfer. Next broaden task/context coverage while preserving the joint-featurizer endpoint: design additional task families and richer combinations; screen native competence at cohort level; record causal responses and common-site availability; use eligible non-holdout cohorts for training and keep existing whole-task holdouts excluded. New task labels must feed a concrete training/validation dataset rather than another unbounded single-task mechanism detour. Keep existing frozen candidates and direct task-mean baseline as references. This follows the user's explicit approval of diverse supporting tasks.

Unlabeled-group discovery, fresh validation of inferred tasks, and retraining remain outstanding. Do not equate this bounded pilot with the intended bootstrap loop. H961 context-bridge predictor evaluation remains pending. Broad research goal stays active.
