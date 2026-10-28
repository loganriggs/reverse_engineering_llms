## H993 — Task-change and coactivation grouping do not outperform simpler faithful baselines

Previous turn was progress: H992 demonstrated strong familiar-context fidelity with fixed PCA geometry and scalar/PCA-ordered sparse inference. This turn fits memberships from discovery data without altering that basis or projection coordinates. It tests the missing claim that task information or coactivation produces useful groups.

Fit disjoint four-atom groups from144eligible expanded6 training states/72pairs only. Task signatures summarize each atom's squared pair-change energy within each of six task families, normalized by family mean total pair-change energy, then cosine-normalize each atom's six-dimensional signature. Coactivation profiles are squared single-state coordinates over the same training states, cosine-normalized across atoms. These are activation statistics, not actual downstream causal-effect profiles. A shuffled control permutes task-family labels across the72pairs, preserving every family count.

Deterministic balanced greedy agglomeration first pairs128atoms, then pairs the64resulting groups to obtain32blocks ofsize4. Cluster similarity is mean pairwise cosine; every atom belongs to exactlyoneblock. Evaluation selects the eight highest-energy groups independently perstate, yielding exactly32active coordinates. Retain dense32/128, scalarTop32 and PCA-ordered blocks as controls. No evaluation labels, gradients or membership selection enter fitting. This is a specific heuristic, not an optimization proof for all possible groupings.

Same H991/H992624directed MLP12 edits with full-donor reference. Native-passing context/whole-task cohort counts50/4. KL ratiozero=reference agreement,one=noedit.

| Policy | Context KL ratio | Whole-task KL ratio |
|---|---:|---:|
| Dense128 |0.091|0.792|
| Dense32 |0.162|0.920|
| ScalarTop32 |0.140|0.973|
| PCA-ordered blocks |0.162|0.956|
| Task-signature groups |0.218|0.929|
| Coactivation groups |0.199|0.933|
| Shuffled-task groups |0.179|0.978|

True task labels do not establish a benefit for this grouping rule; shuffled-label groups perform better on the context aggregate. Task/coactivation groups show modest holdout improvements relative to scalar32 but still remain close to noedit and are inferior to dense128. This cannot establish reusable semantic groups. Normalizing signatures also removes amplitude information, while PCA ordering groups similar variance scales; that is a plausible competing explanation, not a reason to tune many variants on the same validation panel.

Conditional membership stability: fit each grouping again on source0–1 versus2–3 training subsets, holding the PCA basis fixed. Jaccard over unordered atom pairs assigned to the same block is0.0696 for task signatures and0.0726 for coactivation, compared with mean0.0118 from32random fixed-size partitions. Full-fit versus either half ranges0.139–0.167 task and0.129–0.185 coactivation. This is above that descriptive null but low absolute agreement. It is neither independent training-seed reproducibility nor evidence of unique causal units; the common PCA gauge is fixed throughout.

Implementation verification: synthetic known four-atom groups recovered exactly, recovered after atom permutation, invariant to positive per-atom profile rescaling. All training indices are discovery-eligible and non-holdout; shuffled label counts exactly preserved. Group column-degree1/row-size4, support count32 and union masks independently checked.5,616/5,616 full-model forward examples, identity logits2.60e-5, finite outputs and unchanged source hashes. Existing controls independently replay prior KL within1e-4 and all cohort ratios reaggregated exactly. No reserved-test access; natural-document pool26,112.

Research decision: retain scalar/PCA-ordered faithful representations as baselines rather than promote these unstable heuristic blocks as manifolds or task labels. The next information-gathering step is a bounded per-coordinate finite-intervention atlas on discovery tasks, measuring actual output-distribution and answer-margin effects instead of inferring causal grouping from activation energy alone. Compare selective/shared response profiles across held-out contexts before deciding whether to group coordinates or treat larger aggregate subspaces as the unit. Do not launch another long single-task tracing detour. Unsupervised task discovery and the bootstrap retraining loop remain outstanding; H961 context-bridge predictor evaluation remains pending.
