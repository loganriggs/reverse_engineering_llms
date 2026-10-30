# d256/h128 task and input coverage — candidate protocol

Status: design recorded before scientific outcomes; NOT yet launch-approved by the technical gates below. No scientific h128 run has started. The objective remains broader than this panel, including deeper-network recovery and theory.

## Question and comparisons

Does the conditional value of task diversity persist at128 shared bilinear features and100,000 input examples? Cross source inputs25,000/100,000 with task heads2,064/8,256. The unordered product dictionary has8,256 coordinates. The equal scalar-label comparison is25,000×8,256 versus100,000×2,064, both206,400,000 labels. Adding tasks at fixed inputs also adds labels and readout parameters; it does not isolate diversity from resources.

Use five fresh teachers266101–266105, two random starts267001/267002, and one privileged teacher start per condition:60 fits. Gaussian inputs, Gaussian source/adaptation noise SD.05, exact Gaussian RMS target normalization, and four whole-task holdouts. Primary transfer uses32,768 adaptation examples;8,192 is a secondary underdetermined diagnostic. Neither held-out tasks nor adaptation errors select training settings, seeds or checkpoints.

Compare teacher-level means after averaging two random starts and four held-out tasks; privileged controls remain separate. Report paired teacher differences, median errors, all-component cosine>.99 recovery and partial recovery distributions. Preserve every condition and failure. Low empirical-loss teacher-start witnesses do not certify strict local minima or explain all population error. Width/input/task/seed changes across panels do not isolate a width effect.

## Training and persistence

Retain3,000 full-batch Adam steps, learning rate.003, per-model gradient clipping10, CUDAfloat32 and TF32off. Accumulate gradients in512-row input chunks weighted by chunk length/total rows; clip and update only after the full dataset. No stochastic minibatch optimizer steps. Run one model at a time unless a separately verified batched implementation is adopted before protocol freeze. Source prefixes and initial functions match across task/input arms; initial dictionaries and all source readout rows are common within each random seed.

Generate reproducible256-row data blocks with full source-head noise before task-prefix slicing. Store generator versions, seeds, stream IDs and block hashes; retain teacher factors/readouts and initial arrays. Source seed is teacher+5000, stream0; independent adaptation stream1. Chunked checkpoints retain dtype and SHA256 hashes with each file below20MiB. Record times, peak memory, loss and gradient curves, clipping counts, and terminal status. Do not silently discard nonfinite runs or restart from favorable seeds.

Known exact Gaussian normalization is a simulation privilege. All-source readout rank must be checked; factor independence and task-span claims must have numerical diagnostics rather than assumptions from dimensionality alone. Exact identifiability results concern noiseless function equality under their specified assumptions, not guaranteed finite-budget recovery.

## Technical gates before freeze and launch

1. Streamed versus full-batch losses, gradients and Adam updates: numerical checks passed on CPUfloat64/CUDAfloat32; first process shutdown failure retained, cleanup retry terminalexit0.
2. Full-size training memory and sustained-throughput check for all four shapes: pending terminal inspection. The initial two-step smoke passed at100,000 inputs, including full task count.
3. Full8,256-product population Gram plus adaptation solves and independently explicit SVD at8,192/32,768 inputs: running. Require terminal success and acceptable resident memory.
4. End-to-end tiny scientific smoke: target regeneration, normalization, paired initialization, complete training/checkpoint reload, evaluator and independent audit. Not yet implemented for this generalized runner.
5. Reviewer-readable resource estimate and frozen source/protocol hashes before scientific launch. Resume only matching completed cases; preserve interrupted artifacts and never duplicate a live worker.

## Resource decision

Full task-count models contain68,227,072 active parameters; lower-task models17,105,920. These heads dominate parameter count. The full grid may require tens of GPU hours plus substantial CPU analysis; two-step timings alone do not establish sustained throughput. Evaluate that cost explicitly rather than treating memory feasibility as sufficient. Source100,000×8,256 labels consume about3.08GiB in float32, before model/optimizer/temporary storage. Running an independent analysis process concurrently can increase host-memory pressure.

The broad goal remains active regardless of this panel's result. This experiment scales depth-two recovery; it does not establish depth3–5 circuit recovery or a general sharing advantage.
