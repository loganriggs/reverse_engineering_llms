# d64/h32: crossed task and input coverage

All 60 fits and 240 adaptation fits are complete and audited. Five fresh teachers, two random starts plus a privileged teacher start per condition, 3,000 full-batch Adam steps. Source tasks132/528 cross inputs2048/8192. Primary adaptation uses2048labels for528hidden-product features;512label adaptation is retained as an underdetermined diagnostic. See TASK_COVERAGE_H32_PLAN.md.

Errors below are medians across five teacher means, averaging both random starts and four held-out tasks within teacher. Teacher starts remain separate. Unit Gaussian target second moment makes MSE1 the zero-predictor reference. Component recovery requires all32matched quadratic features to exceed absolute Gaussian cosine.99.

| Inputs | Tasks | Initialization | Source MSE | Transfer MSE | All32components |
|---:|---:|---|---:|---:|---:|
| 2048 | 132 | random | 1.9488 | 1.5357 | 0/10 |
| 2048 | 132 | teacher | 0.00141942 | 0.00142927 | 5/5 |
| 2048 | 528 | random | 1.71588 | 1.52167 | 0/10 |
| 2048 | 528 | teacher | 0.00131352 | 0.00133583 | 5/5 |
| 8192 | 132 | random | 0.0312251 | 0.048311 | 3/10 |
| 8192 | 132 | teacher | 0.000217652 | 0.00130016 | 5/5 |
| 8192 | 528 | random | 0.0198697 | 0.0267005 | 3/10 |
| 8192 | 528 | teacher | 0.000252041 | 0.00128401 | 5/5 |

## Equal-label and fixed-input comparisons

At equal1,081,344scalar labels,132tasks×8192inputs beats528tasks×2048inputs on 5/5teacher blocks. Their median transfer errors are 0.048311 and 1.52167, respectively.

At fixed input counts,528tasks beats132tasks on 4/5teachers with2048inputs and 4/5with8192inputs. This supplies four times as many labels and more readout parameters; it does not isolate task diversity from resources.

## Optimization witnesses

36/40random fits have empirical loss more than1e-8above their same-condition trained teacher-start checkpoint. Saved checkpoints and independently replayed losses give explicit numerical lower-loss witnesses. They rule out global empirical optimality of those candidate endpoints, but do not certify strict local minima or explain all population error. Teacher initialization is privileged, not a practical recovery algorithm.

## Resource use and validation

Training each12-model teacher batch took 99.31–99.40seconds, excluding CPU dataset construction and exact evaluation. Maximum allocated GPU memory 1.882GiB; maximum reserved 2.035GiB. Do not sum the repeated per-case batch times. GPU was idle before launch; no other compute process was observed then.

Active parameter counts are73,792/282,880. Inactive readout slots remain byte-identical to initialization. Independent checks replay literal target/noise generation, exact Wick source/transfer errors, SVD adaptation, quadratic-coefficient Gaussian feature matching, all initial pairing and logged curve coverage. Source datasets are stored in four chunks per teacher plus teacher/adaptation metadata. Training is CUDAfloat32 withTF32off; population analysis is CPUfloat64. No nonfinite fits are silently removed.

## Scope and next decision

This extends width/input scaling within a fixed depth-two polynomial family, not depth3–5 recovery or unprivileged circuit discovery. The task/input/width/seed and adaptation counts differ from preceding widths; cross-panel differences cannot be assigned to one factor. Close this fixed grid without optimizer rescue. Use the observed data-coverage and optimization limits to choose the next scaling step; keep the broad research goal active until its remaining requirements are audited.
