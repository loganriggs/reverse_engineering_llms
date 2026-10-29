# Sixteen input-featurizer supervision fits complete

M87 is complete. All16fixed final models are retained and audited; held-out evaluation has not started. This is completion of the controlled training comparison, not evidence that causal supervision improves transfer.

## Matched training

The comparison contains four feature-selection structures (dense, scalar, disjoint block, fixed overlapping groups), two seeds, and two supervision arms. Within each structure/seed, both arms start from the exact same audited full-population warmup state. The architecture has443,776trainable parameters in every case; active coordinates differ and will be reported.

Both arms use the same132-update sequence:66natural-text and66task updates, alternating natural first. Natural loss is identical all-position output KL. The control task loss is intact KL. The joint task loss adds the calibrated0.2589969666times the balanced three-corner response error. Both task endpoints contribute with weightone-half. Every arm uses freshAdamstate,lr1e-4, and decoder column renormalization. Fixed input/output statistics and group memberships remain unchanged. No model, seed, checkpoint or task was selected based on results.

Training completed2,112optimizer updates and9,480native-model forwards. All loss/gradient finiteness and native-model-no-gradient checks passed. No runtime failure, extra update, restart or early stopping occurred.

## Audits

The structural audit verifies every final state against its saved progress, all132Adamsteps/model, initial-state hashes, defaults, fixed buffers, unit decoder norms, all scheduled examples and each recorded loss's arithmetic. This does not independently replay the optimizer trajectory.

A separate native check evaluates all16final models on the first fixed training pair endpoint0 and natural training document0. Independent NumPy scoring checks intact KL, all three response conditions and their normalization, and natural KL. The intact-only implementation exactly matches the joint loss's intact-KL term. Maximum metric discrepancy is9.77e-15. These144additional native forwards are correctness checks on training examples, not transfer measurements. Totalnativeforwards in M87:9,624.

## Frozen evaluation candidates

The manifest contains all16executable predictors plus16full-dictionary least-squares projections using each model's decoder. These privileged projections consume native writes explicitly and refuse to run when a native write is absent. They provide local-MSE references for the decoder spans; they are neither input-computable models nor upper bounds on behavioral fidelity.

All32adapters passed training-document preflight: independent bilinear/gating/decoder arithmetic for input predictors, independent least-squares projection and residual orthogonality for output references. Every model remains included. No native forwards or held-out scoring were needed for this adapter check.

## Next: evaluate the actual hypothesis

Run the predeclared32natural held-out documents,14source-group held-out pairs, and56whole-task pairs covering7families/14formats. Retain all weak and ineligible cases. Compare native/predicted backgrounds and changes at source/final positions, including identity controls, and independently audit saved outputs. Report natural retention, task accuracy, local writes, causal responses, parameter/active costs and seeds separately.

The primary question is whether joint supervision improves causal fidelity on excluded tasks while retaining ordinary behavior. Neither training completion, low loss on a fixed training example, nor privileged projection quality establishes reusable computational units, semantic manifolds or new-task discovery.

Artifacts: validation/bilinear_input_joint_training_protocol.json, bilinear_input_joint_training_receipt.json, bilinear_input_joint_training_results.json, bilinear_input_joint_training_audit.json, bilinear_input_final_loss_audit.json, bilinear_input_evaluation_models.json, and bilinear_input_evaluation_preflight.json. All model/optimizer states and fixed audit outputs are retained in the research artifact directory.
