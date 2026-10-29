# Toy research continuation — active, 2026-10-29

Goal: complete the toy experiments, scale them as justified, and derive/test interesting theoretical results. The prior native-MLP research remains paused. This project is separate and must not be redirected automatically into the old featurizer loop.

Previous goal turn: progress (readout/optimization controls and singular geometry). Current turn: progress (fresh multi-task transfer, explicit graph recovery, generic component-identifiability proof, and exact fiber/Jacobian certificate). All producers/auditors are terminal and passed; no live job or blocker. No GPU/native-MLP work started.

## Latest completed controls and exact reuse family

Current goal turn: progress. FASHION_FLOW_RATE_RESULTS.md:18new calibratedSGDfits, all audited, no failure. Rate chosen from initialtrainingprobe to equalize instantaneouslossdecrease, notoutputspeed;3disjointtrainingprobes assess transfer without tuning. Correctedmedianaccuracydepth2/3/4 now77.50/77.42/76.88%, versus73.70/67.02/60.58% atbase.01. Freenormalizedmedians78.64/76.20/74.56%. Finite-budget3seedcontrol, no robust architecture ranking. Producer21914/auditor70954terminal. No livejob/GPU.

REUSE_COMPLEXITY_AND_FLATNESS.md adds exact scalar repeated-squaring minimalmultiplicationD/formula2^D-1certificate, closed-formgradientflow and specifiedGaussianlikelihood localRLCT=1/2^(D+1). Six derivative/gradient/flow checks passed. Classical elementary results, notnovelty; changingdegree isexplicitconfound. NextA: fixed-degree multivariate repeatedvsdistinctfactor teachers with known sharing and matchedfunctionscale/metrics. Nextscaling: seed/sample-size controls, SVHN verifiedsource outstanding. Avoid unbounded one-epoch SGD tuning. Goalactive.

## Latest theory-to-image experiment: gradient-flow control complete

Current goal turn: progress. FASHION_FLOW_RESULTS.md:9cases×4arms=36fits,55ktraining,215steps,depth2/3/4,threefreshseeds,width16,float64SGDlr.01. Initial functions equal; correctedtied vsprojectedtree maximum recorded parametererror3.272e-13. Initial disagreement fractionsdepth2.363–.479,depth3.738–.776,depth4.838–.920. IndependentNumPy audit final36validationmetrics passes; finite-difference initial loss derivatives pass. Producer85540 and auditor57282terminal; no livejob. NoGPU/testevaluation. Sources/models/results saved. GenericCPU+pinnedcore0 run didnotcrash, not a definitive runtimefix.

Ordinarytied/free medianaccuracy near eachother atdepth3/4; corrected/projected substantiallyslower atsamebaselr. This is a finite-budget metric/control result, notconvergence/generalization evidence. Next bounded B control: independentlyfixed learning-rate/function-speed controls and multipleprobes; A certifiedreuse-complexity teacher experiments remainqueued. Scaling/seedreplication andSVHNloader provenance remainoutstanding. Goalactive.

## Explicit theory directions A and B — active

User requested reuse-aware rank/complexity and sharing gradient-flow proofs. SHARING_COMPLEXITY_AND_FLOW.md now defines grammar-dependent circuit/formula complexity and resource profiles, proves q^(2^L) small-DAG/large-formula/large-flattening separation, and derives tangent-kernel PSD decomposition and multiplicity bounds. These are classical/elementary results applied here, not novelty claims. check_sharing_complexity_flow.py passed exact rational moment/coefficient/rank fixtures and4actual-network Jacobian/kernel checks with/withoutRMSNorm. E2 existing20trajectories verified correctedSGD/projectedSGD equivalence. A did not previously have an implemented minimal-circuit measure.

Next theory experiments explicitly queued in note: A controlled reusable-polynomial teachers with degree/reuse controls and certified bounds; B image-probe disagreement gradients and ordinary/corrected/projected/freeSGD from equal functions, multipledepths/seeds. No large new sweep running. Do not let image accuracy replace these theoretical questions; both contribute to broadergoal alongside scaling. Exact minimum general circuit complexity and full circuit recovery remain unsolved here.

## Latest completed work: full-pool Fashion-MNIST

Current goal turn: progress. All10 full-pool classifiers complete at30epochs,55k training/5k validation; independent NumPy replay of train/validation metrics and saved logits passes. Producer34085 crashed SIGILL, resume43211 crashed SIGILL, generic26747 crashed SIGSEGV after budget-tree epoch16; pinned generic resume18656 completed. Final auditor32389 terminal passed. No active experiment. Sources/checkpoints/curves/crash traces saved; report FASHION_FULL_RESULTS.md and figures/fashion_full.png. Runtime root cause unresolved; genericCPU alone did not fix it. CPU affinity0 plus genericCPU completed, but that is not proof of a fix. NoGPU, officialtest unused.

Depth2/3/4 final stack accuracies88.50/88.90/89.04%; equalwidthtree89.12/89.16/89.06%; budgettree89.32/87.90/87.00%. Equalwidthtree trainaccuracy97–97.5%, stacks95.3–95.4%, little validation gain. Late validationCE mostly worsens while trainingCE improves. One seed only: no significance or universal ranking claim. Pilot vsfull changes bothdata andupdatecount, not an isolated samplecomplexity measurement.

Next: nested data-size/seed replication with explicit epoch and update budgets, preserving officialtest, and SVHN standard versusstandard+extra. svhn_data.py loader prepared with torchvision-reference checksums and layout/label assertions, but officialhost HTTP andHTTPS connections refused; noSVHN data downloaded or training started. Obtain a checksum-matching trusted mirror or restored officialsource; avoid arbitrary unverified data. Existing Fashiondata allows independent progress. Preserve synthetic frozen-hidden-feature transfer as deferred work. Overallgoal incomplete.

## Latest user direction: real-image occurrence-tree comparison

User requested Fashion-MNIST, then potentially SVHN/ImageNet, depths2–4 with RMSNorm as a controlled factor. Implemented vision_circuits.py and eight float64 correctness conditions (independent occurrence storage, function/input-gradient equality, summed occurrence gradients), all passed. Fashion-MNIST publisher checksums verified; cache outside repository.

Pilot: 2048 training / 5000 validation from official training split; official test unused. One seed, five epochs, 19 planned fits, shared stack / equal-width untied tree / narrower near-budget tree, normalization on/off, ReLU MLP baseline. Producer session76751 exited139 (segmentation fault); results/fashion_pilot/results.json now explicitly interrupted. Saved rows preserved. No producer remains live. Cause undiagnosed; independent replay audit not yet run. Do not present pilot as complete or sample-complexity evidence. Next: diagnose crash, independently audit saved rows, resume missing configurations without overwriting completed evidence, then freeze multi-seed/sample-size study. No GPU started.

The synthetic frozen-hidden-feature transfer experiment below is deferred in favor of this explicit user steering.

## Latest completed representative scaling sweep

Current goal turn: progress. SCALING_RESULTS.md:72fits complete, no nonfinite failures. Producer47326 and auditor28762 terminal. Independent replay all72passes(maxrelative3.16e-15); design audit and summary/figure complete.70/72source metrics and470/576learnedtransfer task metrics pass10%batchCVgate. NoGPU. Near-budget shared advantage only consistently paired in d16h8depth2 at128labels(3/3); no broad scaling replication. Fullstate/proposal/data archives retained.

New exact diagnosis: Wick105pairing Gram for shared depth2teachers avoids input-monomial expansion. Independent quadrature passes. Exact source-output projection floors: d16h8 .391/.413/.442; d32h16 .739/.806/.779. Twelve source outputs under-span36/136hidden products. This is NOT a lower bound for all learned hidden representations or necessarily approximate source predictions.

Next bounded experiment: transfer directly from frozen students' hidden feature dictionaries vs their12source outputs, on same32/128adaptation labels, with independent feature RMS and fixedpenalties; include true teacher hidden dictionary oracle. No retraining necessary. Preserve heldouttask isolation and use efficient dualridge for wide dictionaries. Only afterward consider source-task-count changes. Overall goal incomplete; publishing current batch authorized.

## Latest prefix controls — depth5 detour closed

Current goal turn: progress. DEPTH5_PREFIX_RESULTS.md:9audited fits copying1/3/4teacher trunk layers while retaining random readouts and independently recalibrating output energy. Prefix1 population errors1.94/9.45/2.02 despite falling training loss; prefix3 .669/.328/.125; prefix4 .00218/.00327/.00389 (all3 residual precision gates fail). Independent replay and exact prefix/calibration/hash audit pass. Alljobs terminal, noGPU.

Do not turn teacher-copy success into an unprivileged learning or computation-discovery claim. Fixed training proposal may cover modified student errors differently. This bounds the current depth5 detour: coverage, calibration, smallerLRs, localbasin and prefixes are tested. Return to representative feasible depth2/3 sharing/transfer scaling instead of more privileged rescues.

Next: implement and finalize SCALING_GRID_DRAFT.json (not launched/frozen):d16h8/d32h16,depth2/3,bothteacherregimes,3teacherreplicates,3studentarchitectures,width/budgetcontrols,12source+4whole-taskholdouts,expected72fits. Resolve independent calibration, teacher/source-only proposal isolation, exact budgets and residual-focused evaluation. Include oracle source-output span because12tasks do not span larger hidden dictionaries. Save fullstates/replay; report precisionfailures. Main goal remains incomplete.

## Latest depth5 basin controls

Current goal turn: progress. DEPTH5_BASIN_RESULTS.md records9 audited fits on frozen depth5 importance datasets. Random-start lr1e-4/1e-5 still yields relativeerror~1; teacher+1%RMSnoise positive controls give.000258/.000672/.001686. Two absolute precision gates fail and remain flagged. NumPy evaluation/objective replay, perturbation replay and input hashes pass. All producers/auditors terminal; no GPU/live training jobs.

Norm traces: random readout shrinks markedly, trunk norms barely change. Do not describe simultaneous all-parameter shrinkage as observed mechanism. Initial random empirical alignment~1e-6. Exact power-function pairing formula demonstrates that input-degree32 can make moderate directional overlap nearly function-orthogonal; independently checked against exact circle quadrature. This is illustrative, not a theorem about these general networks.

Next bounded step: initialization/curriculum control manipulating early feature alignment, with teacher access labeled privileged. Avoid unbounded LR sweeps. Determine whether deep learning can be stabilized for sharing/task-transfer comparisons, or document a depth limit and complete representative feasible scaling. Independent-teacher, budget controls and whole-task transfer remain outstanding in the deep regime. Overall goal remains incomplete.

## Latest coverage-training stage

Current goal turn: progress. COVERAGE_AND_COLLAPSE_RESULTS.md:12paired depth3/5 fits(3fresh teachers, uniform vs exact weighted importance sampling),1024examples600Adamsteps. Independent output calibration and fresh final proposals; full states/data saved. Producers and audits terminal, no jobs live. NumPy replay/weighted gradient audit passes. Depth3 importance improves all3pairs; depth5 predictions lose alignment and collapse nearzero. Four of six depth5 absolute-energy precision gates fail and remain reported. Cross-fitted scalar correction does not recover depth5 signal.

The simultaneous parameter-scaling path has degree2^(depth+1)-3 (=61atdepth5), verified alongside its loss derivative. Inputdegree is32; do not conflate them. This gives a candidate flatness mechanism, not measured causal explanation of Adam trajectories. Next bounded experiment: reuse exact frozen depth5 datasets/calibrated states, compare smaller learning rates and teacher-near initialization positive control, record norm/alignment/gradient traces and fresh evaluations. Then broaden to independent teachers, budget controls, task holdouts and representative scaling. Full goal remains incomplete; no GPU work.

## Latest directional evaluation stage

Current goal turn: progress. See DIRECTIONAL_SAMPLING_RESULTS.md. Rank-one angular central Gaussian mixture with20%uniform floor implemented and validated against exact moments, independent full covariance densities and an exact d2 depth5 circle oracle. Refined d16 depth4/5 proposals(64pilot starts,200steps,stretches32/128,12x32768fresh draws) meet predeclared batchCV<10% gate; depth5 errorCV4.28%, median errorESS570. All jobs terminal, aggregate and density/circle audits pass. No training/GPU jobs active.

Important new finding: earlier4096-point uniform calibration underestimates teacher energy by8.7x(depth4) and234x(depth5) for these fixed models. Next bounded step: independent importance-weighted calibration, then paired uniform-vs-weighted-importance training to distinguish rare-direction coverage from optimizer failure, with fresh per-model evaluation proposals. Save full states for replay. Then representative multi-seed sharing/budget/task-holdout scaling. Methodological evidence uses the same fixed teachers; no claim of deep learned sharing advantage yet. Full goal remains incomplete.

## Latest evaluation/scaling feasibility stage

Current goal turn: progress. HOMOGENEOUS_EVALUATION_AND_DEPTH.md records exact radial integration and a variance lower bound for naive Gaussian evaluation. Six independent-factor quartic comparisons passed; observed batch SD reduction2–4x. DeepMultiTask at d16/h8/depth2–5 passes NumPy forward, degree, matched-function and cached/recomputed gradient checks. Sixteen100-step CPU smoke fits completed with finite training; aggregate audit passed. All processes terminal (producer session90759 exited0). No GPU work.

Critical finding: at depth5(degree32),8192sphere samples yield error effective sample sizes1–6. Do not treat smoke MSEs as reliable learning results or launch a large conventional sweep. Next: implement and validate exact directional importance sampling with independent pilot/proposal/evaluation data and a uniform mixture floor; first against exact quartic moments, then deep repeated evaluation. Calibrated teacher energy also needs scrutiny. Then freeze representative scaling and whole-task transfer grid. Main goal remains incomplete.

## Latest completed stage

Five-teacher confirmation completed:80fits,4560transfer rows, all independently audited. See CONFIRMATION_RESULTS.md and results/multitask_confirmation. Component confirmation:10trained h4 models,24/40components above.95, mean cosine.7845, independent audit passed. All producer/auditor handles terminal; no live jobs. No GPU/native-MLP work started. Current goal turn is progress. Prior heartbeat removed an obsolete reminder; it did not restart experiments.

Shared h10 wins all5 paired shared-teacher blocks against shared h4, separate h4/h1 and polynomial at32/128labels. Independent-teacher transfer medians remain worse than zero prediction. Next stage: validate Monte Carlo population evaluation against exact quartic metrics, build depth2–5 smoke cases, then freeze representative input/width/depth scaling. Preserve whole-task holdouts and all failure/negative conditions. Full goal not complete.

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

## Completed confirmation protocol (historical)

Confirm the transfer signal on five fresh teachers with separately varied optimizer seeds. Suggested disjoint seed ranges: teachers90000–90004,source inputs91000–91004,optimizer seeds92000–92001,adaptation93000–93004. Freeze protocol before outputs. Preserve sourceN256,12training/4held-out tasks,3000steps, same two teacher regimes, adaptation8/32/128, fixed penalties0/.001. Primary comparisons k32/128 specified from development. Do not introduce held-out validation labels or test selection.

Add a wider shared trunk h10 (820 parameters) near the separate h4 model's888 parameters, alongside sharedh4(184),separateh4(888),separateh1(204). Main h4 width pair can start at identical functions; wide shared initialization is not automatically matched and needs recorded scale diagnostics. The h4 identifiability theorem does not extend to overcomplete h10 at d8. This strengthens budget controls but still changes architecture/class.

Compare teacher-block paired means; components/tasks/optimizer seeds are not independent teacher replications. Preserve all failure/negative conditions. If a robust signal survives, define representative depth/width/input scaling and inspect GPU availability without interrupting other work. A source-side meta-validation scheme, if later introduced, must use additional separate meta-task heads; preserve fixed-penalty primary results.

## Remaining scope — not complete

- E1/multi-task learning: fresh confirmation with stronger budget controls is complete; broader sample/task-count behavior and the role of useful component recovery remain.
- E2: existing exact metric projection, optimization-budget controls, and singular-feature conditioning results are complete for their stated settings. Tiny-network spectrum/scaling comparisons may accompany depth experiments; do not reopen old sweeps without a new discriminating question.
- E3: known-family instrumented graph recovery and generic exact feature identifiability are now established under explicit assumptions. Learned models show only partial feature recovery. Need confirmation/robustness and operational transfer beyond this small family, not an unsupported universal graph claim.
- Theory: generic fiber dimension and a local regular-quotient RLCT are derived with an exact rank witness; scalar singular examples remain in MATHEMATICS.md. Degenerate/global RLCTs and robust approximate-identifiability bounds are not solved or claimed.
- E4: representative d16/32,h8/16 plus near-budget widths21–46,depth2/3 sharing/transfer sweep is complete; depth4/5 feasibility and optimization limits are documented. Frozen-hidden-feature transfer diagnosis and final synthesis remain. No general d256/h512 scaling success is claimed.
- Final synthesis must distinguish parameter tying, input identification, physical execution reuse, functional component recovery, and generalization. Preserve negative results and resource comparisons. The full goal is not achieved.

Runtime: /home/loganriggs/Coding/neuron_circuits/.venv/bin/python with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1. Output directories refuse overwrite. Publication authorized. Repository /home/loganriggs/Coding/reverse_engineering_llms, subdirectory shared_tensor_networks. Native-MLP project remains paused.
