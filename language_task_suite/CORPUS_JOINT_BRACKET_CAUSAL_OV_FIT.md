# Does explicit edit-effect supervision improve a rank-limited OV circuit?

The [mixed FineWeb rule model](CORPUS_SURNAME_JOINT_MODEL_AND_OV.md) has a strong two-choice response to an edited bracket opener, while its surname-copy response remains too weak for a circuit claim. This experiment tests the bracket rule at a fixed **rank four per head** in both attention layers of the depth-two, width-64, seed-46451 checkpoint. QK, MLP, embedding, and unembedding weights remain frozen. It compares weight-only SVD, training-activation-weighted SVD, a fit to teacher predictions at both bracket-closer sites, and a fit to the **change in the two-candidate logit margin caused by the opener edit**. Both fitted arms start from the same weight-SVD factors and include 0.03 times sampled background teacher KL. The edit-effect arm also includes 0.1 times site KL. Each has 300 Adam updates; omitted OV channels are checked to remain exactly zero.

The paired fit sources are sampled from the 10,000 training-range FineWeb previews. The two evaluation ranges, 130,000–131,999 and 140,000–141,999, are source-hash disjoint and were not used by the optimizer. They were previously inspected for other model comparisons, so these are **exploratory monitored panels**, not a final untouched confirmation. Each row below evaluates 200 source pairs sampled from the eligible records in that range. The full teacher's paired two-choice accuracy is .930 on 130k and .945 on 140k; full-vocabulary paired top-1 is .200/.225.

| Fit pairs | Eval range | OV method | Paired two-choice | Paired full-vocab top-1 | Edit-effect MAE vs teacher | Site KL vs teacher | Background KL vs teacher |
|---:|---|---|---:|---:|---:|---:|---:|
| 128 | 140k | Weight SVD | .945 | .225 | 1.269 | .0847 | .0409 |
| 128 | 140k | Activation SVD | .945 | .235 | 1.200 | .0675 | .0359 |
| 128 | 140k | Site-KL fit | .945 | .205 | 1.252 | .0683 | .0441 |
| 128 | 140k | Edit-effect fit | .940 | .215 | 1.306 | .0869 | .0439 |
| 512 | 140k | Site-KL fit | .945 | .210 | 1.189 | .0652 | .0426 |
| 512 | 140k | Edit-effect fit | .940 | .200 | 1.215 | .0834 | .0445 |
| 512 | 130k | Weight SVD | .950 | .195 | 1.165 | .0774 | .0401 |
| 512 | 130k | Activation SVD | .950 | .195 | 1.119 | .0700 | .0373 |
| 512 | 130k | Site-KL fit | .955 | .190 | 1.067 | .0592 | .0421 |
| 512 | 130k | Edit-effect fit | .945 | .175 | 1.157 | .0780 | .0446 |

The 128-pair edit-effect fit lowers its last-50-step *training* effect MSE to .285 versus .870 for the site fit, yet has worse heldout edit-effect MAE (1.306 versus 1.252). At 512 fit pairs, the site-KL fit edges out activation SVD on heldout effect MAE (1.189 versus 1.200 on 140k; 1.067 versus 1.119 on 130k) but pays higher background KL. Explicitly minimizing teacher edit-effect MSE still transfers worse than activation SVD on both panels and lowers full-vocabulary paired top-1. Thus the supplied causal objective is not automatically the best prior for a reusable low-rank circuit. It may overfit particular edited contexts or trade away broad teacher fidelity; these data do not identify which explanation dominates.

The outcome is narrower than interpretability: all methods are local OV reductions of the **same frozen model**, not a canonical decomposition of full attention. Two-choice bracket accuracy saturates across the methods, so edit-effect error and KL give more discrimination here. The teacher itself has low paired full-vocabulary top-1; a rank cut cannot certify a fully learned natural-language bracket algorithm. No component was shown to be human-legible, necessary, or sufficient. This completes the bounded bracket-only method check. The next generator experiment should return to a more varied, fully sampled semantic DGP with multiple rule families, independent rule checks, and whole-combination holdouts before expanding method optimization.

Reproduce with [the fit script](fit_corpus_joint_bracket_causal_ov.py). The aggregate, text-free outputs are [128-pair/140k](results/corpus_joint_bracket_causal_ov_rank4_seed46482.json), [512-pair/140k](results/corpus_joint_bracket_causal_ov_rank4_fit512_seed46482.json), and [512-pair/130k](results/corpus_joint_bracket_causal_ov_rank4_fit512_fresh130k_seed46482.json). Source previews, token pairs, and model checkpoints remain local.
