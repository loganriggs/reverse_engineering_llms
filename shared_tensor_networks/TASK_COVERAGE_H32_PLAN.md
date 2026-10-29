# d64/h32 crossed task/input scaling

Frozen before running. Five fresh teacher seeds251101–251105, independent random starts252001/252002 plus privileged teacher start. Input dimension64, hidden width32,528unordered hidden products. Source heads132/528 crossed with2048/8192Gaussian inputs. The equal-label pair132×8192=528×2048=1,081,344is retained. Four whole-task holdouts are excluded from training. Sixty total fits.

Same verified batched training module:12independent models/teacher,3000full-batch Adamsteps,lr.003,per-model gradient clip10,CUDAfloat32,TF32off. Known Gaussian teacher and random-initial head RMS normalization. Source/adaptation noiseSD.05. No schedule, restarts or rescue. Report active parameters4096+tasks×528; unused readout slots must stay unchanged. CPUfloat64exact Wick population errors and Gaussian quadratic matching.

Adaptation512/2048labels;2048is primary for528features,512is an underdetermined secondary diagnostic. Same adaptation data across all source conditions. Teacher starts remain diagnostic, excluded from primary random-start aggregates. Compare teacher-level means across random starts and whole-task holdouts. Report optimization witnesses only from same-condition lower empirical-loss checkpoints.

GPU checked idle before launch (RTX5080,806MiBdesktop memory,no compute processes). Record training-only batch time and allocated/reserved GPU memory. Save source data as four2048-row chunks plus teacher/adaptation metadata so each artifact stays below20MiB. Independent audit covers all60source/240transfer fits, data/noise, initial pairing, inactive heads, component matching and resources/curve coverage. Larger input/width may expose poor finite-budget optimization; retain it without expanding optimizer tuning.
