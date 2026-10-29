# Toy research continuation — active, 2026-10-29

Goal: complete the toy experiments, scale them as justified, and derive/test interesting theoretical results. The prior native-MLP research remains paused. This project is separate and must not be redirected automatically into the old featurizer loop.

Previous goal turn: progress (readout/optimization controls and singular geometry). Current turn: progress (fresh multi-task transfer, explicit graph recovery, generic component-identifiability proof, and exact fiber/Jacobian certificate). All producers/auditors are terminal and passed; no live job or blocker. No GPU/native-MLP work started.

## Evidence current through this batch

- Main report MULTITASK_AND_GRAPH_RESULTS.md; TASK_SPANS_AND_GRAPH_IDENTIFICATION.md; GENERIC_BILINEAR_COMPONENT_IDENTIFIABILITY.md; GENERIC_FIBERS_AND_LOCAL_RLCT.md.
- results/multitask_transfer: 72 fits, 4176 held-out task records, 3 independent teachers x 2 optimizer seeds per teacher regime. Inputs d8, hidden h4, 12 source/4 held-out heads, source N64/256, 3000steps, fixed transfer penalties0/.001 with exactly8/32/128 adaptation labels and no held-out validation. Independent direct-forward/QR coefficient and Gaussian-metric audit passes.
- results/instrumented_graph_recovery: 8 known graph templates x20 seeds =160teachers,1280 probe-subset records; all-three probes identify160/160. Independent literal DAG interpreter and compatible-set audit pass. Known operations/target slots/noiseless access; not blind recovery of trained neural graphs.
- results/component_recovery: 12 trained shared students assessed with gauge-aware matching; independent quadratic-form metric audit passes. At N256, mean matched cosine .777 versus .238initial;13/24 correlated components exceed .95. Not full recovery. Three teacher factor matrices satisfy independence and task-head rank10.
- results/generic_rank: integer coefficient Jacobian3960x184;8 independent exact gauge vectors; rank176 certificate, independently checked 176-minor determinant714573 modulo1000003; autograd agreement1.59e-16. Generic fiber dimension8. Under the explicitly specified local realizable Gaussian likelihood/prior, regular quotient gives local RLCT88,multiplicity1; not an empirical or global singular-model RLCT estimate.
- Earlier readout/longer-training/singular-path/quotient studies remain valid and closed. Source snapshots and hashes preserved. Do not rerun existing output directories.

## Findings that change the next action

At sourceN256 on shared teachers, shared students beat both separate controls in all3 teacher blocks for adaptationk32/128, including the fixed OOD covariance shift. Not consistent against polynomial baseline (paired2/3wins at32,1/3at128), not uniform at8labels orsourceN64. This is exploratory3-teacher evidence, not confirmation. Wrong sharing assumption has a proven rank-tail source-error floor; independent tasks transfer poorly even using true source-output features.

The exact component theorem: with2h independent teacher linear factors and full task-product span h(h+1)/2, an exact h-feature bilinear competitor must match teacher quadratic functions up to scaling/permutation. Proof uses square-root subspace membership plus rank-two axes. Does not identify cached execution, apply to overcomplete competitors, or give approximate/finite-sample guarantees. Counterexamples cover incomplete task span and dependent factors; no literature novelty claim.

## Next bounded stage (not started)

Confirm the transfer signal on five fresh teachers with separately varied optimizer seeds. Suggested disjoint seed ranges: teachers90000–90004,source inputs91000–91004,optimizer seeds92000–92001,adaptation93000–93004. Freeze protocol before outputs. Preserve sourceN256,12training/4held-out tasks,3000steps, same two teacher regimes, adaptation8/32/128, fixed penalties0/.001. Primary comparisons k32/128 specified from development. Do not introduce held-out validation labels or test selection.

Add a wider shared trunk h10 (820 parameters) near the separate h4 model's888 parameters, alongside sharedh4(184),separateh4(888),separateh1(204). Main h4 width pair can start at identical functions; wide shared initialization is not automatically matched and needs recorded scale diagnostics. The h4 identifiability theorem does not extend to overcomplete h10 at d8. This strengthens budget controls but still changes architecture/class.

Compare teacher-block paired means; components/tasks/optimizer seeds are not independent teacher replications. Preserve all failure/negative conditions. If a robust signal survives, define representative depth/width/input scaling and inspect GPU availability without interrupting other work. A source-side meta-validation scheme, if later introduced, must use additional separate meta-task heads; preserve fixed-penalty primary results.

## Remaining scope — not complete

- E1/multi-task learning: independent five-teacher confirmation with stronger budget controls; broader sample/task-count behavior and the role of useful component recovery.
- E2: existing exact metric projection, optimization-budget controls, and singular-feature conditioning results are complete for their stated settings. Tiny-network spectrum/scaling comparisons may accompany depth experiments; do not reopen old sweeps without a new discriminating question.
- E3: known-family instrumented graph recovery and generic exact feature identifiability are now established under explicit assumptions. Learned models show only partial feature recovery. Need confirmation/robustness and operational transfer beyond this small family, not an unsupported universal graph claim.
- Theory: generic fiber dimension and a local regular-quotient RLCT are derived with an exact rank witness; scalar singular examples remain in MATHEMATICS.md. Degenerate/global RLCTs and robust approximate-identifiability bounds are not solved or claimed.
- E4: larger dimensions, hidden widths, and depth2–5 CPU→GPU experiments remain outstanding. Use representative, frozen configurations and appropriate evaluation when full quartic expansion no longer scales. Report compute, memory, failures, and approximation limits.
- Final synthesis must distinguish parameter tying, input identification, physical execution reuse, functional component recovery, and generalization. Preserve negative results and resource comparisons. The full goal is not achieved.

Runtime: /home/loganriggs/Coding/neuron_circuits/.venv/bin/python with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1. Output directories refuse overwrite. Publication authorized. Repository /home/loganriggs/Coding/reverse_engineering_llms, subdirectory shared_tensor_networks. Native-MLP project remains paused.
