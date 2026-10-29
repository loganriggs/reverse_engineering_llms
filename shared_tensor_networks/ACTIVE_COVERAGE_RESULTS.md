# d128/h64 training and storage preparation

All numerical correctness, GPU resource and regeneration/storage checks passed. This is verified infrastructure, not a completed scientific recovery experiment at the larger width.

Four separate three-model groups reproduce the original padded twelve-model computation while evaluating only each condition’s active inputs and heads. Five CPUfloat64 Adam updates match within2.22e-16; losses/gradients also match, and inactive weights remain exactly unchanged. CPU/CUDAfloat32 gradient relative discrepancy is3.63e-7. This equivalence assumes independent optimizer states, equal steps/hyperparameters and per-model clipping.

## Resource probe

Three d128/h64 models per group,2080hidden products and full readout parameters. Random inputs/zero targets; one warmup plus five timed Adamsteps, TF32off. These targets test resource use and finiteness, not task learnability.

| Inputs | Active tasks | Seconds per step | Peak allocated GiB | Peak reserved GiB |
|---:|---:|---:|---:|---:|
| 8192 | 520 | 0.0293 | 1.247 | 1.338 |
| 8192 | 2080 | 0.0587 | 1.820 | 1.984 |
| 32768 | 520 | 0.1057 | 4.193 | 5.051 |
| 32768 | 2080 | 0.2236 | 6.619 | 7.330 |

All four shapes fit the RTX5080 without OOM. Timings are short single-run warmed measurements, not a benchmark distribution or a guaranteed full-sweep duration. Full scientific data generation, normalization, exact population evaluation and audits add CPU cost.

## Deterministic data and lossless checkpoints

Fixed256-row generators use separate seeded streams for inputs/noise and source/adaptation. They always draw full blocks before truncation, preserving input-count prefixes. Task prefixes are sliced after generation at a fixed teacher width. Repeated generation and requests with different final lengths agree byte-for-byte in this environment. Save the complete recipe, NumPy version, teacher parameters and per-block hashes in each scientific experiment; do not assume floating-point target hashes are portable across every BLAS platform.

Independent literal product reconstruction passes at h2,h4,h64, with largest discrepancy6.78e-15. At h64, float64teacher readout chunks are at most8,163,563bytes; the2080×2080float32final readout is16,067,642bytes. Both round-trip exactly and stay below20MiB. Saving actual trained float32weights as float32 introduces no extra quantization; exact population evaluation can cast them to float64 after loading. No scientific dataset has been discarded or replaced by these probes.

## Retained process failure

The initial correctness process completed its assertions and wrote results, then aborted during shutdown with exit134 and “terminate called without an active exception.” This failure is retained in initial_shutdown_failure.json. After adding explicit GPU synchronization/object cleanup, the same check reran with identical numerical results and exit0. The resource probe and regeneration check also exited0. This is a successful replay, not a diagnosis or general fix of the underlying runtime issue.

## Next experiment

Use the verified grouping/storage path for the d128/h64 crossed grid:520/2080tasks and8192/32768inputs, retaining an equal-label pair, teacher starts, independent random starts and whole-task holdouts. Freeze its seeds and protocol before launch. Save each training group before expensive CPU analysis so an analysis failure cannot erase finished GPU work. Do not infer recovery from this infrastructure test. The overall research goal remains active.
