# How many edit types fix a useful three-variable gauge?

We used ten already-trained, three-head bilinear-attention models with three count-sign variables and a learned quadratic parity output. A rare A=0/B=1/C=0 query combination was absent from their supervised training. Their exact combined `6×8` token-function tables were reduced to rank three either by uniform weight-only SVD or unlabeled training-frequency weighted fitting. Both frozen reductions already retain 100% full and rare four-output accuracy; this experiment changes only the coordinates within each reduced function.

The prior [all-pair study](THREE_COUNT_SCRATCH_EDIT_GAUGE_RESULTS.md) optimized sparse differences over all 28 unordered key-token swaps. Here the four within-pair bit flips `(0,1), (2,3), (4,5), (6,7)` are **never used to fit the gauge**. The 24 remaining cross-pair swaps form the fitting pool. We first fit all 24, then build five deterministic, connected edge orders per reduction and fit prefixes of 7, 10, 14, 18, or 24 edges. Each prefix covers all eight tokens. This creates 50 gauge fits per edge count and reduction (five graph orders × ten trained model seeds); the 50 fits are **not** independent model seeds.

| Fit edges | Uniform: mean variable alignment | Uniform: perfect full/rare two-node runs | Frequency: mean variable alignment | Frequency: perfect full/rare two-node runs |
|---:|---:|---:|---:|---:|
| 7 | .961 | 50/50 | .961 | 50/50 |
| 10 | .990 | 49/50 | .9998 | 50/50 |
| 14 | **.9998** | **50/50** | **.9998** | **50/50** |
| 18 | .9998 | 50/50 | .9998 | 50/50 |
| 24 | .9998 | 50/50 | .9998 | 50/50 |

At seven edits, 40/50 fits per reduction exceed .99 variable alignment; the worst is about .805. All still preserve full and rare task accuracy after retaining only the two strongest quadratic product nodes. Mean L1 on the withheld flips is already 1.131/1.134 under uniform/frequency reductions, versus 1.560/1.551 for SVD. At fourteen edits, all 100 fits across the two reductions exceed .999 alignment and pass both task panels; held-out flip L1 falls to 1.068/1.070. The trained task can therefore be robust even when the inferred variable axes are not fully stable; variable recovery needs a stronger gate than task score alone.

Using all 24 cross-pair edits, the mean L1 response on the **four held-out within-pair flips** is 1.0665 (uniform) or 1.0671 (frequency). The plain SVD bases score 1.5595 and 1.5514 respectively. All 20 fits recover the known variables at mean alignment .9998 and retain 100% four-output accuracy after two-node pruning on independent 30,000-sentence test sets and the rare whole-combination subset. The all-28-pair fit is only slightly lower on held-out flip L1 (1.0626/1.0633), as expected because it directly includes those edges. The known variable labels and task outcomes enter only after gauge fitting and node selection.

This is a **held-out edit-edge family**, not unseen tokens or a discovered task. The split itself uses the known token pairing to define the test family. Moreover, the 24-edge graph is connected: each withheld difference is algebraically a sum of observed differences. Success shows that the sparsity orientation transfers to edits omitted from its objective under adequate graph coverage; it does not prove that unseen intervention effects were inferred from no information. A [finite-edit follow-up](THREE_COUNT_FINITE_EDIT_TRANSFER_RESULTS.md) now applies these omitted token substitutions to the trained models on fresh and shifted contexts and compares their actual logit changes with gauge-attributed reduced-function changes. Context-dependent or out-of-vocabulary edits where effects cannot be reconstructed from a small token table remain the stronger gate.

Reproduce with `taskset -c 2-5 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/loganriggs/Coding/neuron_circuits/.venv/bin/python three_count_edit_family_holdout.py`. Per-seed gauges, edge partitions, and all 500 coverage fits are saved in `three_count_edit_family_holdout.json`.
