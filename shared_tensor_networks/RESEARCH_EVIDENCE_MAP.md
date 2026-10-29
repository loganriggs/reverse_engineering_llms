# Current evidence map and completion status

Start with [the research synthesis](RESEARCH_SYNTHESIS.md). This file maps the original E0–E4/theory requirements to their present evidence. Historical reports retain their original “next” sections; those are not live job queues. No experiment is running. The broad goal remains active.

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
