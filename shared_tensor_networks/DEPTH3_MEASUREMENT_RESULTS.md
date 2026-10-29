# Depth-three measurement and calibration replication

Bounded diagnostic replication on the weakest prior regime: d32independent teachers, three replicates, four model/oracle types. Original128label readouts remain fixed; two fresh source-only calibration draws add48readouts. Evaluation uses two independently optimized task-specific proposals, four8192point batches each, all six predictors measured on common points. Same total65536evaluation samples/model as the prior study, but substantially more proposal-optimization work and256rather than32axes per proposal. Not an equal-compute variance comparison.

## Separating measurement from calibration

Each table entry is source-output / hidden-feature relative MSE. Medians over three teacher-level means of four held-out tasks. Zero prediction scores1. “Original norms, new evaluation” is exactly the SAME predictor as the preceding column; only its estimator changed. The other columns change calibration and refit ridge from the same adaptation data.

| Model | Prior evaluation | Original norms, new evaluation | Calibration repeat1 | Calibration repeat2 |
|---|---|---|---|---|
| shared | 1.2061 / 48.203 | 1.212 / 48.631 | 1.212 / 46.076 | 1.2122 / 45.414 |
| separate | 1.1362 / 4.3479 | 1.1447 / 4.4336 | 1.1448 / 4.5005 | 1.1449 / 4.513 |
| shared_budget | 1.1299 / 2.0593 | 1.1288 / 2.0583 | 1.1288 / 2.0489 | 1.1287 / 2.0593 |
| oracle_source | 1.1038 / 1.2847 | 1.1048 / 1.2825 | 1.1047 / 1.3001 | 1.1048 / 1.2871 |

The broad poor-transfer conclusion survives. Calibration changes some hidden errors, particularly the narrow shared model, but none of these median hidden errors becomes competitive with zero prediction. This is a selected diagnostic regime, not an independent confirmation cohort. The independent teacher-source oracle contains only source-task trunks and is not a held-out-feature oracle.

## Precision improved, not guaranteed

For original fixed predictors, 80/96task/error estimates pass the numerator/target batchCV<10%gate, versus20/96previously. Across all three calibrations,240/288pass. Both proposals retain20%uniform support. Every failed estimate remains in the table. These are descriptive batch-variation diagnostics, not rigorous confidence intervals or proof all rare events were sampled. Fitting separate axes per target/error makes precision better here, but does not certify an optimal importance sampler.

The saved summary also gives paired hidden-minus-source mean differences and standard errors using common samples, plus discrepancy between the two proposal panels. With only eight batches these remain diagnostic, not a formal multiple-comparison analysis. No favorable calibration or proposal was selected by test score.

## Calibration has an exact estimator interpretation

CALIBRATION_AS_REGULARIZATION.md derives the equivalent raw-coordinate ridge penalty lambda CᵀD²C. For two normalization draws, the coefficient change satisfies

H1(C1-C0) = -lambda(D1²-D0²)C0,

with H1=FᵀF/N+lambda D1². All48finite-change identities pass. Thus calibration affects the effective regularizer while raw feature functions remain identical. It is distinct from evaluation noise. This does not imply a monotonic generalization effect.

## Audit and decision

Independent NumPy replay verifies all96evaluation batches,96fresh calibration batches (two draws×four batches×12cases),72SVDreadout solutions (24original,48new), and pooled ratios/SEs/precision flags. Original norms and readouts are byte-identical to the preceding study. Checkpoint/prior-fit hashes match. The evaluation proposals were fitted only after the six readouts were frozen; no evaluation data changes any fitted predictor.

Close this bounded replication. The remaining16/96imprecise original-predictor estimates stay qualified, rather than triggering an endless sampling escalation. Together with the exact depth2diagnosis and stable d16depth3cases, the evidence supports a limited conclusion: exposing a larger hidden dictionary alone does not fix these trained models' transfer, and measurements must distinguish representation, finite-label estimation, calibration and evaluation. No universal sharing advantage or impossibility theorem is claimed.

Next consolidate these findings into the overall research synthesis and completion audit, preserving the unresolved precision and theoretical scope limits.
