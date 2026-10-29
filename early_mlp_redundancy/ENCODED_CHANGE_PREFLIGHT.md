# M68: encoded-change causal objective is implemented and verified

The preflight is complete with **no optimizer updates**. The new loss asks a featurizer to produce a donor-minus-recipient direction whose downstream effect matches the native change. Earlier supplied-direction supervision gave the representation the exact native direction and tested its reconstructed background; these are different objectives.

## Fixed comparison

We retain the same six M36 KL-only starts (scalar/block/shared × two seeds), M60 training schedule, native family/region denominators, final-position task KL, all-position natural-text KL, and whole-task exclusions. Donor and recipient writes are encoded independently with their own selections. Their decoded difference is applied at the fixed aligned positions in the original native recipient background. Both encoding paths receive gradients. The native model and teacher are frozen.

Training-only calibration uses the same792 initial scheduled model/endpoint rows as before. Repeated recipients are evaluated once per model but retain their scheduled multiplicity in calibration. The mean initial task KL is0.07943739390118659, the mean balanced encoded-response loss0.8541192749511752, giving one global coefficient0.0930050359836775. No evaluation effects, teacher norm matching or architecture-specific weights are used.

This coefficient equalizes initial mean task-K and causal-loss contributions, as in the earlier experiment. It is a declared calibration convention, not an optimized weight or proof of equal gradient influence.

## Checks completed

- All418 eligible training address mappings reproduce the native deltas, including valid final addresses for unequal-length pairs. Ineligible source alignments stay absent from the active source loss.
- All792 task-K values replay the old calibration exactly. The natural retention runtime and old task-loss hashes match the M61 training receipt.
- Six native-write identity checks reproduce intact/changed teacher outputs exactly and have zero loss. Identity directions and response losses remain zero.
- Independent NumPy checks cover all792 calibration rows,456 unique model/recipient cases and1818 saved final vectors. Maximum loss discrepancy is1.65e-14. Every encoded delta is reconstructed independently from saved donor/recipient predictions and address maps.
- Twenty-four write-space directional finite differences check source and final response gradients through both endpoints at all six starts. Maximum relative discrepancy is0.425%.
- Eighteen full-parameter directional finite differences use three predeclared step sizes at each start. All18 retain the same selected supports; maximum relative discrepancy is0.304%. These verify smooth derivatives within those supports, not differentiability across hard selection boundaries.
- Encoder/decoder parameter gradients are finite, both endpoint gradient norms are nonzero, native-model gradients remain absent, and perturbed parameters were restored exactly. No optimizer was invoked.

Calibration used1380 native forwards, endpoint gradient checks180, and parameter checks126. All processes completed. The natural branch was not changed or reimplemented; its previous numerical preflight remains bound by its original code hash.

## What is and is not established

The loss now trains the specific behavior that failed in the M63 encoded-direction tests, and the implementation passes addressing, forward and derivative checks. We have not trained a new candidate or shown better fidelity, transfer, manifolds, semantic units or discovery. Hard selection boundaries remain a possible architectural limitation.

The next stage is one fixed six-model fit: same132 updates, Adam learning rate1e-4, schedule, normalization and memberships as M61. Keep all final states. Compare against the fixed M61 KL-only and balanced supplied-direction controls; freeze factorial/retention evaluation now. No architecture, weight, learning-rate, task or seed search follows the initial results.
