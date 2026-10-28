## H992 — Preserved PCA geometry supports sparse behavior; arbitrary grouping adds a measurable cost

Previous turn was progress: H991 validated train-only PCA's behavioral advantage. This turn holds that verified geometry and exact projection coordinates fixed while varying support inference. It is a controlled group-structure comparison rather than another learned-dictionary optimization sweep.

Use the exact frozen expanded6 PCA128 orthonormal basis and center from H991. Single-state coordinates are z=(x-mu)B. Decode masked coordinates once with B. Conditions: dense128; fixed first32 PCA directions; scalar Top32 by squared coordinate magnitude; four-coordinate disjoint blocks selected Top8 by total energy; two random block partitions; two overlapping group graphs formed by two balanced random partitions of128atoms into16eight-atom groups (32groups total, eachatom degree2). Select four overlapping groups greedily by marginal uncovered coordinate energy, masking already-selected groups and decoding the resulting union exactly once. The greedy union is not claimed globally optimal.

Overlapping supports use at most32atoms, often fewer. For each overlap seed a scalar control uses EXACTLY that example's resulting atom count, not merely the same mean budget. This matched-count control inherits a count chosen by the group policy; it is a diagnostic comparison, not an independently budgeted encoder. Scalar TopN is optimal for single-state Euclidean reconstruction in this orthonormal basis; that does not guarantee pair-change or output-KL optimality. Contiguous blocks are grouped by PCA eigenvalue order, not semantic labels. All memberships fixed and no group-learning occurs.

Same H991312pairs/624directed edits and finite MLP12 intervention comparison. Table macro-averages native-passing50context cohorts and fourwhole-task cohorts. Atom counts shown for evaluated competent-context endpoints; holdout counts saved separately. KL ratios zero=full donor-edit match, one=no edit.

| Support policy | Mean context atoms | Context KL ratio | Whole-task KL ratio |
|---|---:|---:|---:|
|dense128|128.00|0.091|0.792|
|dense32|32.00|0.162|0.920|
|scalar32|32.00|0.140|0.973|
|block_contiguous4|32.00|0.162|0.956|
|block_random4_9921|32.00|0.221|0.969|
|shared_greedy8_9921|30.62|0.247|1.070|
|scalar_matched_9921|30.62|0.142|0.987|
|block_random4_9922|32.00|0.253|0.965|
|shared_greedy8_9922|30.90|0.305|1.071|
|scalar_matched_9922|30.90|0.143|1.002|

Sparse scalar and PCA-ordered block reconstruction now retain much of the useful behavioral fidelity, despite only32active coordinates. This contrasts with expanded learned block/shared pilots at~0.70–0.74 on the identical context benchmark. It supports preserving a good decoder geometry and reliable coordinates while investigating structure. It does not prove that PCA is the unique correct gauge or that its coordinate blocks are meaningful manifolds.

Random block membership worsens fidelity relative to PCA-ordered blocks. Overlapping graphs are worse than scalar controls matched at every example's actual atom count, so their deficit cannot be explained solely by lower union cardinality. But these are random fixed groups with greedy inference, not learned semantic memberships; they do not refute useful overlap. PCA ordering also groups similar variance scales, so its advantage need not reflect semantic computation.

Whole-task transfer remains poor: scalar/block ratios around0.92–0.97 and overlap~1.07, versus dense128~0.79. Sparse inference discards directions useful for copying/spatial counterfactuals, and original training-span coverage was already weak. Preserving familiar-task fidelity is therefore a prerequisite, not a demonstration of new-task discovery or reuse.

Audits:7,488/7,488 full-model forward examples. Identity logit replay2.48e-5. Orthonormal basis verified, hard group budgets and exact union decoding independently replayed, overlap degree2 checked, scalar per-state counts exactly match unions, scalar residual optimality relative to each shared mask checked. Both dense controls independently replay H991 within2.54e-7 KL and1.34e-5 margin. Cohort ratios independently recomputed. All hashes unchanged, finite outputs, no training or reserved-test access. Natural-document pool26,112.

Next bounded experiment: use discovery-only per-coordinate task-change signatures and/or coactivation statistics to propose groups within this fixed verified span. Compare against PCA-order and random memberships with identical budgets, then test held-out context fidelity, selective group interventions and stable reuse. The supervision must improve grouping rather than merely memorizing task labels; keep original whole-task holdouts excluded. Only validated unlabeled structure should generate fresh task hypotheses for the bootstrap retraining loop. H961 context-bridge predictor evaluation remains pending; broad research objective stays open.
