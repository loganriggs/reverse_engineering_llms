# M61: all18 matched response-objective fits completed and audited

**Training is complete; evaluation is pending.** All18 fixed final states were retained: three objectives × scalar/block/shared representations × two seeds. Every fit executed132updates, for2376updates total. No candidate or checkpoint was selected on evaluation outcomes.

## What ran

The M60 protocol is unchanged. Each fit alternates66natural updates with66task-pair updates. All11training families receive six pair updates per seed, including both endpoints. Intact task KL is final-position KL; natural retention is all-position mean KL. Causal arms use the same fixed native source/final directions and raw versus training-cell-scaled loss. Global multipliers and the scale floor remain exactly as calibrated before training. No optimizer/LR/schedule/weight changes or retries were made.

The runner binds every final and resume checkpoint to the protocol, calibration, audits, loss/runtime code and initial state hashes. Resume files are saved every11updates. Native model parameters remain frozen; decoder columns are normalized after each update. This run completed uninterrupted; numerical equivalence of an interrupted/resumed optimizer trajectory was not independently tested.

## Natural branch preflight

Before training, all six starts passed exact native-output identity, independent NumPy all-position KL scoring, finite selected-support parameter gradients, and12directional finite differences in write space. Maximum relative finite-difference discrepancy was0.00836 (0.836%), below the fixed3%tolerance. There were42preflight forwards and no optimizer steps. These checks complement the M60 task-loss checks; they do not establish derivatives across discrete support changes.

## Training audit

8298forward calls completed the fixed fits. The independent post-run audit verifies all18checkpoint hashes and receipts, all2376step indices and scheduled inputs, exact family counts, per-step objective arithmetic, unchanged mean/scale/membership tensors, finite final states, decoder unit-column norms and active-coordinate limits. Every final state is preserved in the local hash-linked model manifest. Large state files remain outside the shareable report packet.

Online losses are computed on changing examples. First-versus-last online loss is not a valid retention or transfer comparison, and no claims are made from those values. The audit verifies execution and state integrity, not successful representation learning or an independent optimizer replay.

## Frozen evaluation next

`validation/balanced_response_evaluation_protocol.json` was saved while training was incomplete, before inspecting final-fit outcomes. It requires every final state on:

- The complete M57/M58training target bank:140recipients and418eligible native MLP1conditions, with the two ineligible source assignments explicit.
- The complete M59source-group development holdout bank:28recipients and84native conditions, with absent family coverage reported.
- All32natural evaluation documents32–63 for intact output KL, CE, top1 agreement, local reconstruction and actual active support.

Raw error and native response energy, pooled ratios, equal-family scores, margin fidelity and intact prediction are retained together. Source and final-position targets remain separate. Both gains and regressions matter; there is no posthoc winner criterion or follow-up hyperparameter search. These are familiar development evaluations, not pristine whole-task transfer or new-task discovery.

The experiment tests whether balancing causal objectives improves coverage rather than merely changing which errors dominate. Until evaluation finishes, there is no evidence that scaling improved fidelity, shared representations, meaningful manifolds, native redundancy or computational-unit recovery.
