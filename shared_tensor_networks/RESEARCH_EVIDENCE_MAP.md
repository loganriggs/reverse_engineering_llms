# Current evidence map and completion status

Start with [the research synthesis](RESEARCH_SYNTHESIS.md). This file maps the original E0–E4/theory requirements to their present evidence. Historical reports retain their original “next” sections; those are not live job queues. Historical status superseded by the latest update below. The broad goal remains active.

| Requirement | Authoritative evidence | Status and scope |
|---|---|---|
| E0 derivatives, Hessian pullbacks and multiplicity | experiment.py; MATHEMATICS.md; SHARING_COMPLEXITY_AND_FLOW.md; stored checks | Supported for stated models. No global network RLCT inferred from Hessians. |
| E1 both teacher regimes, width/budget controls, N/seeds | RESULTS.md; e1_sweep/e1_matched audit240records each | Completed bounded sweeps. Replayed arms are not independent replication; pilot endpoints not archived. |
| E1 useful finite-label transfer/recovery | CONFIRMATION_RESULTS.md; FINITE_READOUT_CONFIRMATION_RESULTS.md; FINITE_READOUT_H4_RESULTS.md | Five-teacher confirmations, fixed budgets, held-out tasks. Benefits conditional; no universal tying advantage. |
| E2 matched initial functions/metrics and spectra | SHARING_COMPLEXITY_AND_FLOW.md; FASHION_FLOW_RATE_RESULTS.md; bilinear/GN controls | Completed specified checks. No global optimizer-independent convergence claim. |
| E2 alternating/curvature/variable-projection dynamics | PROFILED_CURVATURE_RESULTS.md plus prior bounded panels |20trajectories/654steps audited; four improvements/two regressions when removing one curvature term. |
| E3 distinguish parameter/input/computation reuse | TASK_SPANS_AND_GRAPH_IDENTIFICATION.md; instrumented_graph_recovery/audit.json |1280instrumented checks; known templates. Blind physical execution recovery is unidentifiable from functions alone. |
| E3 component recovery | GENERIC_BILINEAR_COMPONENT_IDENTIFIABILITY.md; finite recovery results | Exact fixed-width theorem, local certificates and empirical recovery; not overcomplete/global-circuit identification. |
| E3 approximate identification | BOUNDED_APPROXIMATE_IDENTIFICATION.md | New qualitative compactness result with bounded readouts. Quantitative thresholds and practical robust guarantees remain open. |
| E4 d16/32,h8/16,depth2/3 controls | SCALING_RESULTS.md,72audited fits | Complete bounded sweep; limited useful scaling and some precision failures. |
| E4 frozen hidden representation diagnosis | LARGER_HIDDEN_TRANSFER_RESULTS.md; DEPTH3_HIDDEN_TRANSFER_RESULTS.md | All saved larger models covered,192fits per depth. Depth2exact moments; depth3sampling qualifications. |
| E4 measurement/calibration replication | DEPTH3_MEASUREMENT_RESULTS.md | Weak regime original-fit precision20/96→80/96; remaining failures retained, no endless escalation. |
| E4 depth4/5 feasibility | Coverage/basin/prefix reports | Documented learning/evaluation limitations; privileged copying is not discovery. |
| E4 GPU/image scaling | SVHN_CONFIRMATION_RESULTS.md |30audited fits,five seeds; accuracy/CE disagree. One split, no official test or ImageNet result. |
| Proposed d256/h512/depth5/100k envelope | No full grid exists | Partial relative to the proposed envelope. Do not declare universal scaling achieved. |
| Reuse-aware complexity/rank | SHARING_COMPLEXITY_AND_FLOW.md | Fixed grammar, circuit/formula/flattening separations; not a general algorithm for minimal reusable rank. |
| Singular geometry and learning coefficients | MATHEMATICS.md; generic-rank/local-likelihood notes | Explicit toy/local calculations; no global RLCT or universal singular-learning result. |

## Evidence and audit limits

audit_research_evidence.py verifies selected audit counts, retained result hashes, and referenced SVHN checkpoint hashes. Later frozen-transfer and measurement audit counts are included. This inventory is not a fresh numerical replay of every historical experiment or a proof that every theorem is correct; theorem claims rest on the arguments and explicitly scoped checks in their notes.

A completed negative experiment is still a completed experiment. Conversely, a report or green audit does not turn an unmeasured range, imprecise estimate or unproved generalization into a finished claim. The present evidence does not pass a full-objective completion audit.

## Next substantive question

At a larger feasible fixed width, distinguish insufficient source task coverage from insufficient examples and optimization. Earlier scaling used12source tasks for36/136hidden products; the full task-span identifiability condition was not satisfied. The direct-hidden diagnosis now shows the learned dictionaries remain inadequate, while the true-feature oracle is useful. A predeclared task-count/sample-count control with exact population metrics addresses that gap more directly than another optimizer sweep. The larger proposed envelope and quantitative robustness remain explicit, not silently replaced by the small successful family.

The complete task-product-span assumption is sufficient, not known necessary in the larger family. Having12tasks below36/136products does not by itself prove the hidden features cannot be identified or learned; the next experiment must test that hypothesis rather than assume it.

Update: TASK_COVERAGE_H8_RESULTS.md now tests the larger feasible task-count/input-count cross (60fits), with mixed equal-label-budget evidence. This completes that bounded d16/h8grid, not the entire proposed scaling envelope.

Latest scaling update: TASK_COVERAGE_H16_RESULTS.md adds60d32/h16fits. At equal69632labels,34tasks×2048inputs beats136tasks×512inputs in5/5teachers. This supports input coverage alongside task diversity, while all low-input random fits have explicit lower-loss teacher-start candidates. The proposed larger envelope is still not fully executed.

## Readout learning timescale extension

[READOUT_TIMESCALE_RESULTS.md](READOUT_TIMESCALE_RESULTS.md) completes 30 finite-time two-timescale flows and independent checks. Faster readout adaptation approaches reduced gradient flow in all six tested teacher/initialization sequences. [READOUT_TIMESCALE_THEORY.md](READOUT_TIMESCALE_THEORY.md) states the conditioning-sensitive tracking bound and uniform local assumptions. This is an ODE mechanism result, not evidence that faster readout learning improves feature recovery or transfer. Finite-dwell alternating recovery tests and larger-width scaling remain open.

## Finite readout speed and recovery

[READOUT_SPEED_RECOVERY_RESULTS.md](READOUT_SPEED_RECOVERY_RESULTS.md) reports 70 paired fits: moderate-speed exact readout-flow splitting improves aggregate transfer over joint speed1 on all five archived teachers, but beats reduced training on only three. Fast explicit joint steps lose hidden-learning time to stability-driven backtracking. These are finite-budget results with independent metric audits; fresh-teacher and matched-time confirmation remain open.

## Fresh-teacher readout-speed result: advantage did not replicate

[READOUT_SPEED_CONFIRMATION_RESULTS.md](READOUT_SPEED_CONFIRMATION_RESULTS.md) adds 70 fresh-teacher/fresh-initialization fits, all at accumulated hidden-step time20. Split speed10 loses its previous aggregate advantage: transfer median .2517 versus joint1 .0913 and reduced .1906; wins only2/5 and1/5 teacher blocks respectively. These primary arms already had matching clocks in the old panel. High-speed joint and split methods have similar aggregate errors but substantially different gradient work. This supports stiff-block integration as a computational tool, not a reliable feature-recovery improvement. Larger d64/h32 scaling remains open.

## d64/h32 scaling and sampled information

[TASK_COVERAGE_H32_RESULTS.md](TASK_COVERAGE_H32_RESULTS.md) completes 60 audited GPU fits and 240 adaptation fits. At equal labels, more inputs/fewer tasks wins all five teachers (.0483 versus1.5217 median transfer). Every teacher start recovers all32components; random starts recover zero full sets at low input coverage and3/10at high coverage. Thirty-six of40random endpoints have explicit same-condition lower training-loss witnesses. This extends the coverage/optimization distinction to larger width without proving a universal scaling law.

[SAMPLED_HIDDEN_INFORMATION.md](SAMPLED_HIDDEN_INFORMATION.md) derives profiled finite-sample rank bounds, including sampled hidden-value basis changes that readouts absorb. Sixty-three derivative/rank fixtures and three finite basis-change tests pass. The bounds do not explain the current grid by a simple shortage of constraints; conditioning and optimization remain material. d128/h64 requires verified memory/storage adaptation before launch; the broad goal remains active.

## Verified infrastructure for d128/h64

[ACTIVE_COVERAGE_RESULTS.md](ACTIVE_COVERAGE_RESULTS.md) verifies independent-case grouping against the original full-batch updates and tests all four larger shapes on GPU (largest allocated6.62GiB). Deterministic block regeneration and lossless checkpoint chunks pass. An initial post-check shutdown abort is retained; cleanup replay exited successfully. This prepares larger scientific runs; it supplies no new recovery result at h64.


## Exact sampled-information extension; h64 training active

SAMPLED_HIDDEN_INFORMATION.md now includes exact rank certificates for d4/h2 and three product tasks: seven generic inputs leave two local non-scaling ambiguities, whereas eight give local identification modulo scaling. Thirty integer fixtures retain nongeneric small-range deficits alongside broader-range witnesses. This supplies no conditioning, noise or global-recovery guarantee. The d128/h64 producer was directly confirmed live; evaluation/audit and scientific results remain pending. Aggregation code has fixture tests but no h64 scientific outcome is claimed.


## Profiled information and conditioning

PROFILED_INFORMATION_AND_CONDITIONING.md adds fixed-weight information monotonicity, exact linearized noise risk, a local reduced-flow rate and a certified path showing arbitrarily poor conditioning despite full rank. Checks cover60 configurations,85 PSD comparisons and15 linear-noise simulations; no nonlinear recovery experiment is claimed. This motivates measuring spectra rather than relying only on dimension counts. The h64 producer remains live; scientific outcomes remain pending.
