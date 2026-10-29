# M60: matched response-balance experiment frozen and task loss verified

**The prospective design, training-only calibration, and task-loss gradient checks pass. No optimizer updates have run.** This prepares a direct test of whether family/region scaling improves broad causal fidelity, instead of inferring a training mechanism from uneven evaluation scores.

## Fixed comparison

Six starting checkpoints: scalar, block, fixed shared groups × seeds71/72, all from the M36 intact-KL arm. Every start receives three arms:

1. Intact prediction KL only.
2. Intact KL plus raw native causal-response squared error.
3. Intact KL plus family/region-scaled native response error.

All arms share the same examples, native deltas, optimizer, learning rate0.0001 and132updates:66 natural-text updates alternating with66 task-pair updates. Each of11 training families receives six pair updates per seed, with both endpoints retained. Pairs are sampled from the existing training inventory using fixed seeded schedules; no held-out pair is eligible. Natural documents are restricted to0–31. All fixed final states will be evaluated, without checkpoint selection or a weight sweep.

Task retention uses final-position KL, matching the task-response site. This deliberately differs from historical M36's all-position task KL and is held identical across the new arms. Therefore new-versus-old M36 comparisons cannot attribute changes solely to response balancing. Natural-text retention remains all-position mean KL. The natural branch still needs live validation in the training runner before fitting.

Response targets are the unchanged M57 native changed-input-position and final-position swaps. The two unequal-length color source assignments remain ineligible; their final targets remain. Every endpoint averages its eligible nonidentity response terms. Identity targets remain mandatory exact-zero controls, excluded from the loss mean because they are identically zero. No task/effect/competence filtering.

## Theoretical distinction made testable

Raw response loss prioritizes absolute native-probability-weighted error. The scaled arm divides each term by a family/region native-response energy estimated from the entire training bank. This approximates equal relative fidelity across cells while retaining the same endpoint sampling. It is a different fidelity objective, not a neutral gauge fix or proof of semantic structure.

Near-zero response energy can make inverse-energy weighting unstable. Before fitting, the floor was fixed to10% of the median positive training cell energy; no alternative floors are searched. Each denominator is the larger of its cell mean and that floor. Floor=0.0005846621. Six of22 cells are floored: final-position capital_recall, country_language, entity_binding, number_successor, object_color and pronoun_gender. This limits amplification but does not establish an optimal normalization.

Global response multipliers match each causal arm's pooled initial response term to pooled initial task KL, using all six starts and their exact scheduled training endpoints. The multipliers differ because the response units differ; this is not a result-based hyperparameter choice.

| Calibration quantity | Value |
|---|---:|
| Scheduled endpoint/model rows | 792 |
| Mean initial task KL | 0.0794374 |
| Mean raw response error | 0.0566501 |
| Mean scaled response error | 1.1265504 |
| Raw response multiplier | 1.4022465 |
| Scaled response multiplier | 0.0705138 |

The calibration reuses independently audited M58 training vectors only. The prospective schedules and scaling rule were frozen before any new fit. Old holdout results motivated the question; no holdout losses or gradients enter the scales, schedules, or multipliers. This remains research on previously examined development material, not pristine confirmation.

## Checks completed

- Training schedule: six updates per family per seed, natural documents0–31 only, original source-group evaluation IDs excluded.
- Independent reaggregation of all22 training energy cells and792 scheduled initial losses; global weights reproduced.
- Native capture cache: all140 training endpoint vectors replay M57 exactly; stored writes and native targets are hash-linked.
- Live task loss: first scheduled endpoint for all six starts reproduces saved initial K/raw/scaled losses; native reconstruction yields zero intact and response losses.
- Both causal objectives:24 directional finite differences in reconstructed-write space agree with autograd; maximum relative discrepancy0.001065 (about0.107%). All selected-support featurizer parameter gradients are finite; native model parameters receive no gradients.

These checks do not establish gradients across discrete support changes, optimizer success, or improved causal fidelity. The current gradient checks cover task loss, not the not-yet-implemented natural-text retention branch. No training result is claimed.

## Next execution step

Implement the fixed runner with per-fit resume records binding protocol, calibration, loss source and initial checkpoint hashes. Verify the all-position natural KL branch, identity and parameter gradients on training documents; verify branch scheduling and unit-column normalization before starting the18fits. Use all arms and final states; do not alter loss weights, floor, schedule or budget in response to evaluation. If implementation checks fail, diagnose correctness before fitting.

After training, measure common-target causal fidelity with both raw-energy pooling and equal-family summaries, plus intact task/natural prediction and actual active support. Preserve all historical whole-task exclusions. Improved normalized scores would not by themselves establish shared computational units, meaningful manifolds, native redundancy, or genuine task discovery.
