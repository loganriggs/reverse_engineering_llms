# Research synthesis and completion evidence

Update: LARGER_HIDDEN_TRANSFER_RESULTS.md now closes the depth2portion of the frozen-hidden-transfer gap identified below (192fits); depth3remains. The review below records the evidence and remaining scope at its original audit point.

Status: the broad goal is NOT complete. This review checks the original E0–E4 direction against saved results, rather than declaring completion because the recent optimizer panel ended. Historical reports describe their own time point; their old “next” sections are not current job queues. No experiment is currently running.

## What the research presently says

1. Sharing is both a function-class restriction and a parameter metric. Correcting occurrence multiplicities exactly reproduces projected occurrence SGD. A same-learning-rate architecture comparison can conflate these effects.
2. A compact reusable circuit need not have small tensor rank or be easy to learn. The repeated-squaring examples provide resource separations and explicit slow/flat dynamics; they do not prove fixed-degree sharing is intrinsically difficult.
3. Diverse tasks can identify reusable features under explicit algebraic assumptions and improve empirical recovery. More tasks are not automatically better at a fixed label budget: spreading too few examples over many tasks can lose the benefit.
4. Prediction accuracy is not a circuit-recovery metric. Scalar-output models can fit exactly with different hidden subspaces. Shared versus recomputed execution is functionally indistinguishable without execution-sensitive observations.
5. Optimizers strongly affect apparent architecture advantages. ALS, GN, alternating blocks, factor balancing and readout elimination all have both favorable and unfavorable regimes. The latest controlled curvature intervention identifies a genuine local geometric difference, not a universal winning optimizer.
6. Scaling is mixed. SVHN has a replicated data/capacity result; wider shared stacks outperform equal-budget trees on accuracy in that panel while trees win cross-entropy. Larger synthetic models have optimization/evaluation limitations and an unresolved representation-measurement bottleneck.

## Original experiment requirements and evidence

| Requirement | Current authoritative evidence | Scope / remaining gap |
|---|---|---|
| E0: derivative and Hessian pullbacks; multiplicity and scalar singularities | experiment.py and results/e0 plus later checked identities in SHARING_COMPLEXITY_AND_FLOW.md; REUSE_COMPLEXITY_AND_FLATNESS.md | Exact local identities and scalar likelihood examples. No global network RLCT claim. |
| E1: tied/untied teachers; matched width and parameter budget; sample-size and seed sweeps | RESULTS.md; results/e1_sweep and e1_matched, each240audited records | Equal-function starts only for width pair. Some replayed arms are identical, not independent replications. Initial pilot lacks saved endpoints and is not covered by the later480record audit. |
| E1: useful transfer and recovery under finite noisy supervision | FINITE_READOUT_CONFIRMATION_RESULTS.md and FINITE_READOUT_H4_RESULTS.md; h4audit120fits | Five-teacher confirmation, whole-task holdouts, fixed scalar-label budgets. Conditional task-diversity benefits, not universal sharing gains. |
| E2: metric controls and dynamics | FASHION_FLOW_RATE_RESULTS.md; bilinear alternation/GN/balance reports; PROFILED_CURVATURE_RESULTS.md | Last panel20fits/654audited steps isolates curvature. Bounded panels closed; global convergence and optimizer-independent advantage unproved. |
| E3: separate tying, input reuse, computation reuse | TASK_SPANS_AND_GRAPH_IDENTIFICATION.md; results/instrumented_graph_recovery/audit.json |1280known-intervention checks. This is known-template instrumented identification, not blind recovery of arbitrary trained circuits. |
| E3: recover actual reusable features | GENERIC_BILINEAR_COMPONENT_IDENTIFIABILITY.md; READOUT_LOCAL_IDENTIFICATION.md; finite recovery panels | Exact restricted-family theorem, local rank certificates, noisy empirical feature matching. Approximate global uniqueness, overcomplete competitors and unconstrained circuit recovery unresolved. |
| E4: depth/width/input scaling | SCALING_RESULTS.md;72audited d16/32,h8/16,depth2/3fits with budget controls | Some Monte Carlo metrics miss precision gate. Direct hidden-feature transfer for these saved larger models is still unfinished. |
| E4: depths4/5 | DEPTH5_PREFIX_RESULTS.md and earlier coverage/basin studies | Documented failures and privileged controls. Do not count teacher-prefix success as learned recovery. Three full-prefix evaluations miss precision gate. |
| E4: larger image data and GPU | SVHN_CONFIRMATION_RESULTS.md;30audits and checkpoint hashes verified | Five seeds, one split; depth3confirmation, depth2/4development. No official-test or ImageNet claim; no generality beyond these settings. |
| E4: proposed d256/h512/depth5/100k grid | No complete grid exists | Proposal was a conditional scaling envelope, not an executed result. Feasibility failures warrant limits, not invented completion. |
| Theory: reuse-aware rank/complexity | SHARING_COMPLEXITY_AND_FLOW.md | Operation grammar and circuit/formula/flattening bounds. Not a general computable minimal reusable-network rank or claimed novel lower bound. |
| Theory: learning geometry and singularities | Gradient/pullback proofs, generic rank/local-fiber certificates, specified scalar/regular local likelihood integrals | No universal/global RLCT, generic global circuit identification or novelty certification. |

## Provenance review

audit_research_evidence.py checks selected audit counts and retained result hashes, plus all30SVHN audit statuses and referenced checkpoint hashes. It verifies the120fit h4panel,72fit scaling panel,9prefix controls,1280instrumented checks and20fit/654step curvature audit coverage. Its JSON explicitly marks the goal incomplete. This is a provenance/coverage review, not a fresh replay of every historical experiment. Report existence alone is not counted as a theorem or a passed experiment.

## The actionable gap found by this review

The larger scaling study transfers through12source predictions. At h8/h16 the teacher hidden-product dictionaries have36/136directions. Exact source-output-span projection floors are approximately .39–.44 and .74–.81, respectively. Those floors apply to the exact source outputs, NOT to arbitrary learned hidden representations. Thus poor source-output transfer cannot tell us whether the hidden dictionary was learned.

The later HIDDEN_EXTRACTION_RESULTS.md and EMPIRICAL_EXTRACTION_RESULTS.md address d8/h4 models; they do not close this older d16/32 gap. CONTINUATION.md explicitly deferred the larger frozen-feature study during the shift to image experiments. This review restores that unfinished deliverable rather than adding another optimizer sweep.

## Next experiment and its limits

Reuse saved larger model checkpoints and32/128adaptation inputs; no source retraining. Compare12source-output features against direct hidden-product dictionaries, with fixed normalization/regularization and teacher oracle controls. Preserve within-head products for separate models; do not silently add cross-head products. Report feature dimensions and label counts because larger representations can overfit even when they lower population projection error.

Start with depth2, where exact Gaussian fourth-order-feature cross-Grams avoid expanding every input monomial. quartic_cross_gram.py supplies memory-bounded cross-dictionary Wick moments. Independent degree-eight Gauss–Hermite tests, block-size/transposition checks and agreement with the earlier structured Gram pass in d2/3/4 (maxrelative9.71e-16). Sphere Gram equals Gaussian Gram divided by d(d+2)(d+4)(d+6), from radial independence and degree-eight homogeneity; the radial moment is independently checked.

Depth3remains a separate evaluation problem, since its features have higher degree. Completing the depth2diagnosis must not be presented as completing all72models or the entire research goal. Next-stage protocol should freeze shared/independent teacher coverage, exact normalization versus the older sampled normalization, and all oracle privileges before producing transfer rankings.
