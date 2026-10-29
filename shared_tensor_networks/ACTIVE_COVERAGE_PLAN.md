# Memory-bounded training preparation for d128/h64

Previous turn completed and audited d64/h32. This turn verifies infrastructure before a larger scientific sweep. No training outcome is selected here.

Replace one padded12-model batch by four independent3-model groups, one source input/task condition each. Evaluate only active input rows and readout heads, while retaining the full readout parameter tensor for auditing inactive coordinates. Keep per-model gradient clipping, Adam hyperparameters and update counts unchanged. This changes execution/storage, not the mathematical full-batch objective. Independent optimizer states are essential; do not clip across different models.

Correctness: compare losses, gradients and five successive Adam updates in CPUfloat64 against the existing masked/padded12-model implementation for four unequal input/task conditions. Check inactive weights remain exactly unchanged. Separately compare CPU/CUDAfloat32 gradients. Tolerances1e-12forCPUupdates and1e-5forrelativeCUDAgradients.

Resource probe after correctness passes: RTX5080, d128/h64,2080products,3models/group, input counts8192/32768, task counts520/2080, full readout2080×2080. Fixed random inputs and zero targets; random factors/readouts scaled by square-root widths. Six Adamsteps per shape, first step warmup and five timed;lr.003,clip10,TF32off. Record timings, allocated/reserved memory and finite losses. This is a capacity probe, NOT scientific teacher/student training. Retain OOM failures rather than silently changing group size or input count. Check no external GPU compute process before starting.

Before the full larger sweep, deterministic regeneration or chunked storage must also be verified, because full source/noise arrays and readout checkpoints now become large. A successful resource probe alone does not satisfy that requirement or establish recovery at this width.
