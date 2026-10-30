# d128/h64 task and input scaling

All 60 scientific fits have completed evaluation and audit. Five fresh teachers, two random starts and a privileged teacher-start control per condition. Primary transfer uses 8,192 adaptation labels for 2,080 hidden-product features; 2,048-label diagnostics remain in the raw records.

Errors below are medians across five teacher means, averaging two random starts and four whole-task holdouts within teacher. Teacher starts remain separate. Full recovery requires every matched quadratic-feature cosine to exceed .99. These are finite-budget outcomes after 3,000 Adam steps, not convergence certificates.

| Inputs | Tasks | Initialization | Source MSE | Transfer MSE | All 64 features |
|---:|---:|---|---:|---:|---:|
| 8192 | 520 | random | 1.6115 | 1.43575 | 0/10 |
| 8192 | 520 | teacher | 0.00105312 | 0.00106435 | 5/5 |
| 8192 | 2080 | random | 0.035626 | 0.0320563 | 1/10 |
| 8192 | 2080 | teacher | 0.00105428 | 0.00104946 | 5/5 |
| 32768 | 520 | random | 0.0128254 | 0.0160153 | 4/10 |
| 32768 | 520 | teacher | 0.00018545 | 0.00103836 | 5/5 |
| 32768 | 2080 | random | 0.0157393 | 0.0237113 | 3/10 |
| 32768 | 2080 | teacher | 0.000186862 | 0.00104716 | 5/5 |

## Paired budget comparisons

At equal 17,039,360 scalar labels, 520 tasks × 32,768 inputs beats 2,080 tasks × 8,192 inputs on 4/5 teachers. Per-teacher differences, ranges and all conditions are retained in summary.json.
At 8,192 inputs, more tasks improves transfer on 5/5 teachers; it also adds labels and readout parameters.
At 32,768 inputs, more tasks improves transfer on 3/5 teachers; it also adds labels and readout parameters.

## Optimization witnesses

38/40 random-start endpoints have empirical loss more than 1e-8 above their same-condition trained teacher-start checkpoint. This is a numerical feasible lower-loss witness, not a certificate of strict local minima or proof that optimization explains all generalization error. Teacher starts are privileged.

## Resources and audit scope

Twenty independent three-model groups used 1.767 GPU-training wall hours in total, excluding dataset preparation, serialization and CPU analysis. Maximum torch-allocated memory was 6.619 GiB. Do not multiply each group time by its three models.

The audit checks all conditions, training/data/code/checkpoint hashes, inactive weights, initial pairing, logged curves, regenerated targets, literal empirical predictions, quadratic-coefficient feature matching and explicit SVD adaptation. Exact population errors use the previously verified Wick kernel, shared between evaluation and audit. It does not replay every optimizer step or independently measure GPU memory. Training is float32 with TF32 off; evaluation is float64.

There are 120 learned transfer fits and 10 shared oracle fits. Repeating oracle results across student records yields 240 reported transfer entries, not 240 independent fits. Raw scientific artifacts preserve all starts and controls; no result-based seed selection or optimizer rescue was performed.

## Scope

This is larger-width depth-two shared-feature recovery under known Gaussian normalization. It does not complete depth3–5 recovery, unprivileged graph discovery or prove a general scaling law. Compare the whole table and paired teacher outcomes before choosing a subsequent experiment; the broader research goal remains active.
