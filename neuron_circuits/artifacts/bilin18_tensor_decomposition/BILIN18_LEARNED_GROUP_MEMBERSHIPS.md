## H730 — Learning memberships improves agreement reconstruction, but does not close the scalar gap

We replaced the correlation heuristic with discrete optimization of captured coefficient energy. All six scalar dictionaries remain frozen. Each of 128 groups keeps its anchor atom and selects three other atoms; two groups activate per example, with shared atoms decoded once. The training objective maximizes the sum of squared encoder coefficients covered by their union. This is a support-energy proxy, not exact reconstruction loss for a nonorthogonal dictionary.

Training alternates group assignments with sequential membership updates. Given assignments, each group selects the three neighbors with greatest uncovered energy, accounting for what the other selected group already covers. Assignment inference uses a greedy start plus four alternating best-response rounds. We use exactly this inference for every validation condition, including initial memberships. Ten training rounds are run, retaining the highest training-objective snapshot including the initial state. There is no validation checkpoint selection.

Every dictionary is tested from raw-correlation and random anchored initializations. Each learned grouping has three degree-preserving rewires. All conditions have 128 groups, four memberships per group and two active gates; scalar Top-8 remains the unrestricted comparator. Training uses 2,048 documents, validation 256, all 16 sampled positions. No task labels are used.

| Memberships | Agreement validation write MSE | Capital validation write MSE |
|---|---:|---:|
| Scalar Top-8 | 0.3465 | 0.4420 |
| Initial correlation groups | 0.4636 | 0.5427 |
| Learned from correlation groups | 0.4409 | 0.5425 |
| Rewires of learned correlation groups | 0.4658 | 0.5703 |
| Initial random groups | 0.4922 | 0.5748 |
| Learned from random groups | 0.4454 | 0.5704 |
| Rewires of learned random groups | 0.4587 | 0.5744 |

Values average all three dictionary seeds; rewires average all nine seed/control combinations. The agreement improvement transfers from training to validation and appears from both initializations, although the final inventories differ. Training changes about 38% of correlation-initialized agreement groups and 73% of random-initialized groups. For capitals it changes only 3% and 14%, respectively; gains are correspondingly small. Local optimization and forced anchors may matter, and these runs do not prove global convergence.

The learned groups preserve a modest advantage over degree-matched rewires, but still reconstruct worse than scalar Top-8. Selected unions average approximately 7.96–7.98 unique atoms. Better memberships are therefore part of the explanation for agreement, while this particular objective and optimization provide little progress for capitals. No causal or manifold conclusion follows from reconstruction alone.

Correctness: four tests pass: union counting, assignment nondecrease, membership nondecrease with anchor/size constraints, and independent exhaustive enumeration of the fixed-assignment membership objective on a small example. Every training membership sweep was nondecreasing for its fixed assignments. Best snapshots are selected by training score only because restarting assignments can find different local optima. Input/source hashes stayed unchanged; all results were finite and every saved group had four atoms. Zero model forwards, new documents or reserved-test evaluations were used.

Next: evaluate all learned and initial groupings, both initializations and all seeds, on the frozen causal-head assay with matched rewires and scalar controls. Compare task effects, nuisance sensitivity, output KL and CE separately. Any promising candidates then need fresh diverse-task and selective shared-component tests before manifold or hierarchy interpretation.
