# M69: all six encoded-change fits complete; evaluation pending

All six prospectively fixed fits completed and passed the saved-state audit. **No performance improvement is claimed yet.** The question remains whether training the effects of encoded donor-minus-recipient differences improves causal fidelity beyond the earlier supplied-direction objective, while preserving ordinary behavior.

## What ran

Scalar, block and fixed shared-dictionary featurizers, each at seeds71 and72, started from the same six M36 KL-only states as M61. Each received exactly132 Adam updates at learning rate1e-4:66 natural-text retention updates alternating with66 training task-pair updates. Each task update averaged both endpoint losses. Donor and recipient were independently encoded with their own supports; gradients flowed through both. The global encoded-response weight remained0.0930050359836775, calibrated before fitting.

The native model, normalization mean/scale and group memberships stayed fixed. Decoder columns were normalized after each optimizer step. Every final state was retained; there was no checkpoint, task, mask or seed selection. Total:792 optimizer updates and2766 native forward calls. No process is still running.

## What was checked

The audit verifies all six initial/final hashes and receipts, every update's scheduled document or task pair, both endpoint identities, eleven-family sampling counts, loss arithmetic and fixed loss coefficient. Each family appears six times per seed's task schedule. All final parameters are finite, decoder columns have unit norm within tolerance, and mean/scale/membership match their starts exactly. Recipient and donor support counts stay within the declared maximum32.

Optimizer defaults were independently compared with the original saved M61 optimizer states and match. Current PyTorch2.14.1+cu130 matches the historical cached-load smoke artifact. The M61 training receipt did not itself record its runtime version; we therefore do not claim a directly recorded historical-version match or a complete old optimizer-trajectory replay. M68 did replay every initial task-K calibration value exactly and verify the unchanged natural retention implementation.

Progress snapshots include optimizer state and exact receipts for safe resumption, but this run completed uninterrupted. Numerical equivalence after interrupt/resume has not been separately tested. Online losses refer to changing examples and are not evidence of improved held-out fidelity.

## Frozen next evaluation

`validation/encoded_change_evaluation_models.json` binds all six new final states and twelve fixed controls: KL-only and balanced supplied-direction controls for each architecture/seed. All18 checkpoint hashes are verified. The evaluation protocol was frozen before fitting.

Evaluate the same14 familiar source-group holdout pairs using the complete native/reconstructed background × native/reconstructed direction factorial, plus training fidelity and all32 natural evaluation documents. Report raw, pooled and equal-family response errors, answer margins, local direction error, intact KL/CE/top1 and active support costs. Replay existing controls to verify comparable semantics. Keep training and evaluation results separate and preserve all whole-task training exclusions.

This tests an objective mismatch, not whether the features are already semantic manifolds or discovered computational units. Those claims require successful causal fidelity and subsequent structural and task-transfer tests.
