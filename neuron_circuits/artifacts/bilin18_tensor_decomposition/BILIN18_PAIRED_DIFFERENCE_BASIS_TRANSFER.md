## H1037 — Training-change geometry fits well but fails to cover excluded tasks

Previous turn was progress: H1036 tested broader natural training and found no aggregate causal-transfer improvement. This turn tested the alternative that activation variance hides reusable task-change directions.

Fit: rank128 orthonormal residual basis from the uncentered second moment of donor-minus-recipient states across the original60discovery pairs and fivefamilies. Weights balance families, pairs within family, and tokens within pair. This is equivalent to covariance over both signed orientations, avoiding arbitrary direction centering. The difference covariance has effective rank248 at relative eigenvalue threshold1e-10. Both candidate bases use the same activation-training center for whole-state reconstruction. Centers cancel for paired differences. No downstream loss or holdout input enters the fit.

Compare against frozen activation PCA128 at the same residual dimension, same native other streams, and same competent copying/spatial historical panel (16pairs/32endpoints). Paired edits preserve recipient residual context and add projected donor-minus-recipient residual; other streams receive exact donor differences. This is a residual representation diagnostic with native assistance, not an autonomous sparse model. These holdouts are already opened, and no fresh generalization claim is made.

| Basis | Weighted training-change energy retained | Held-out change energy retained | Whole KL | Paired relative KL | Whole-reconstruction top1 |
|---|---:|---:|---:|---:|---:|
| Activation PCA128 |90.62%|21.14%|2.0246|0.4787|3.1%|
| Paired-difference basis128 |99.67%|20.55%|1.8271|0.5006|15.6%|

Training energy uses the weighted ratio of total squared norms. Held-out energy is the mean of per-pair ratios; the distributions and weighting differ, so the gap is descriptive rather than a calibrated generalization statistic. Paired KL worsens on both excluded tasks: copying0.3402→0.3519; spatial0.6173→0.6494. Whole KL improves in aggregate, mainly spatial3.0320→2.3105, while copying worsens1.0173→1.3436. Whole-state answer improvements do not imply faithful causal edits.

The representation strongly captures observed training changes but does not extend their geometric coverage to excluded tasks. This weakens the hypothesis that merely switching from activation covariance to task-difference covariance supplies broadly reusable directions. It does not establish that task changes are inherently non-reusable: the supervised task inventory is small, context dependence is possible, and an adequate basis may require richer training families. Combined with H1036, the evidence favors testing supervised task coverage rather than another small selector or covariance variation.

Verification:128suffix executions, no new prefix captures, no holdout fit rows, exact prompt disjointness, unchanged input hashes. Double-precision PCA orthogonality error1.78e-15. Activation baseline agrees with H1035 within7.37e-7. Independently replayed KL from one saved endpoint per arm (two samples), error3.74e-14. No multi-seed uncertainty estimate or independent full execution rerun claimed.

Next stopping decision: stop further basis variants on this excluded-task panel. Evaluate these two frozen bases on existing same-family lexical development examples, reporting training-change coverage and development causal fidelity separately. If same-family causal transfer works while whole-task transfer fails, expand the competent training-task inventory and introduce separately frozen task holdouts for the next joint featurizer fit. If same-family transfer also fails, inspect fitting/conditioning and local contextual variation before attributing the failure to missing tasks. This is a bounded two-arm check, not another selector sweep. The resulting diverse causal task data must feed scalar/block/shared-dictionary comparison; no unrelated mechanistic detour is opened.
