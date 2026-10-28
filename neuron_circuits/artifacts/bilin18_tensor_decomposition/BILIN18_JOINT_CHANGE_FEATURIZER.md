## H984 — Joint pair-change supervision improves some scores but does not establish context-sensitive transfer

Previous turn made workflow progress by creating the requested two-hourly research-direction reminder. This turn resumed the joint-featurizer benchmark rather than extending the scalar sentiment branch. The user's permission to explore diverse supporting tasks remains in force; causal targets and broader coverage should feed back into the joint comparison.

H983 data, common last-token MLP12 output site, architectures, initialization, two seeds and600-step budget remain fixed. Six training tasks,48 discovery endpoint states; copying and spatial inversion remain entirely outside fitting, centering, scaling and target construction. Reconstruct all256 cached endpoint writes. Matched block and fixed-overlap architectures each register337,574 parameters. Their actual active dimensions still differ (32 versus roughly30), and neither has Grassmannian orthogonality. This is a tiny development benchmark, not a general architecture ranking.

New loss: normalized activation reconstruction plus0.1 times pair-difference prediction error. Each training family's pair error is divided by that family's mean true delta squared norm, then averaged equally across24 pairs. Target is the actual endpoint1-minus-endpoint0 write difference. All128 atoms receive this supervision. This differs from H983's answer labels, whose heads read only64 atoms; the supervision support and objective both differ, so this is not a pure target-information ablation. Neither loss is downstream causal supervision. The unused label head remains serialized to preserve initialization and registered parameter count.

Controls: reconstruction only; true paired changes; within-family cyclically deranged targets; task-mean targets. The derangement removes correct pair matching but retains a mean target/true-delta cosine0.547. It therefore is not a no-semantic-information null. Task-mean control was added after observing that result and the initial score pattern; it uses only the four discovery changes in each family, never evaluation data. Final16 fits include allfour conditions and bothseeds. Twelve initial fits preceded the expanded rerun, so28 fits executed total; final comparison retains16 checkpoints. This is an explicitly adaptive development analysis.

Metric is per-cohort squared error of reconstructed pair changes divided by true change energy. One equals predicting no change. Average ratios equally across native-competent non-discovery cohorts and seeds; these are not pooled-energy ratios, output KL or task accuracy.

| Architecture / objective | Trained-task validation delta error | Whole-task holdout delta error |
|---|---:|---:|
|block / reconstruction|1.471|1.057|
|block / paired_changes|1.253|1.043|
|block / shuffled_changes|0.973|1.039|
|block / task_mean_changes|1.132|1.050|
|shared_fixed / reconstruction|2.124|1.116|
|shared_fixed / paired_changes|1.398|1.041|
|shared_fixed / shuffled_changes|1.213|1.168|
|shared_fixed / task_mean_changes|1.387|1.017|

True pair-change supervision improves over reconstruction alone in aggregate, but does not establish an advantage over within-task shuffled targets. Held-out-task changes remain poorly reconstructed. The shuffled control's retained task direction is important: its success cannot establish that arbitrary labels are equally useful, nor can it establish per-example computational recovery. The task-mean arm matches or improves the true-pair aggregate: block1.132 versus1.253, shared1.387 versus1.398 on trained-task validation. Whole-task error remains near1. This further weakens the case that the benefit requires context-specific pair targets, although two seeds and these reused development cohorts do not establish equivalence. All16 fits and all512 cohort records are retained, including native-failing diagnostics. No best-seed filtering or validation checkpoint selection.

Independent audit: all512 delta-error metrics replay from saved predictions and cached source writes within1e-5. Repeated reconstruction-only predictions match H983 exactly, isolating the implementation change from initialization drift. Training means/scales checked independently,48 unique training states per fit, no whole-task holdout fitting. Exact union masks, finite outputs and source hashes verified. Zero full-model forwards and no reserved-test access. Natural-document pool26,112.

Next concrete integration: freeze these models and evaluate their reconstructed donor-write changes through the model, comparing full donor changes, no change and task-mean controls. This tests whether poor activation-space differences hide useful behavioral fidelity. Use this bounded validation to select what supervision is missing, then expand task/context training coverage rather than escalating tiny-data objective variants indefinitely. A broader task bank is authorized when it produces training anchors and held-out validation for the joint featurizer. Genuine unlabeled-task discovery, prospective validation and retraining remain outstanding. H961 context-bridge predictor evaluation remains pending.
