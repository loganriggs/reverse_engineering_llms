## H689 — Pair-complete synthetic training tests composition without teacher repair

Use the same synthetic teacher as H687/H688, now training on all singleton and pair group supports while holding every triple and the all-group support out. All atom pairs coactivate somewhere in training. All variants receive the true group gates; this remains conditional representation learning. No teacher-coordinate edits are applied during training or evaluation. Nine fits:three seeds×known membership, learned membership, separate blocks; same4096training samples,2048heldout samples,3000Adam steps and batch128.

| Variant | Mean train relative MSE | Mean held-combination relative MSE | Parameters | Mean active heldout coordinates |
|---|---:|---:|---:|---:|
|shared_known_membership|3.97646e-08|1.43145e-09|380|7.190|
|shared_learned_membership|0.0918202|0.109418|380|6.460|
|separate_blocks|0.0545288|0.13654|412|9.571|

Pair coverage resolves the known-membership transfer failure in all three seeds: held-combination relative MSE is 4.29e-9, 1.32e-13, and 2.38e-13. Learning membership is still fragile: seed 11 reaches 8.24e-9 held error and exact membership recovery after atom alignment, while seeds 12 and 13 have held errors 0.111 and 0.217, with membership agreement 0.9375 and 0.875. The mean hides one successful recovery and two failed fits. Known-membership atom cosines remain 0.918–0.942 despite excellent reconstruction, reinforcing that functional recovery need not mean unique atom recovery.

This table is not a parameter- or active-dimension-matched superiority claim. Higher-order test supports differ from the H687 test mixture, so compare mechanisms and per-support diagnostics rather than treating aggregate errors as an identical-distribution benchmark.

Teacher atom matching is also recorded via maximum one-to-one absolute cosine assignment, with membership agreement after that alignment. Reconstruction and atom recovery are different criteria; neither implies uniquely identified semantic factors. Actual results are retained for every seed, including any failed learned-membership fit.

The next required stage is genuinely unsupervised gate/membership discovery and a trained nested-structure benchmark, with equal-budget frontiers and selective reuse interventions. The broader research plan remains to return validated alternatives to early LLM activations, assess geometry/stability and causal usefulness, and derive candidate computational regime labels. Perfect synthetic reconstruction alone is not that goal.

Audits:all atom-pair coverage asserted, all source/implementation hashes unchanged and errors finite. No original-model forwards or new documents. Known gates remain supplied; graph topology discovery and matched LLM comparisons are outstanding. Broad goal active and incomplete.
