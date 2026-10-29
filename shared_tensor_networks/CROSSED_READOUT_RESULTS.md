# Crossed readout dynamics: normalization versus regularization

60newfits plus20archived fits complete the four-cell panel on identical five teachers, two starts each,512source examples, ten source tasks and four held-out tasks with128adaptation labels. Two methods per cell: joint versus exact readout elimination. TRF200evaluation cap, no rescues or new tuning. Teacher-mean transfer averages both starts before taking the median across five teachers. Do not pool this with the earlier d6 pilot.

| Coordinates / penalty | Method | All4features cosine>.99 | Median teacher transfer MSE | Max raw row norm | Hit cap |
|---|---|---:|---:|---:|---:|
| raw_zero | joint | 8/10 | 0.000584226 | 1.793 | 0/10 |
| raw_zero | reduced | 8/10 | 0.000584226 | 756.8 | 0/10 |
| raw_ridge | joint | 8/10 | 0.000584227 | 2.114 | 10/10 |
| raw_ridge | reduced | 7/10 | 0.00760719 | 527.2 | 0/10 |
| unit_zero | joint | 9/10 | 0.000373538 | 1 | 0/10 |
| unit_zero | reduced | 5/10 | 0.0246931 | 1 | 0/10 |
| unit_ridge | joint | 9/10 | 0.000373547 | 1 | 0/10 |
| unit_ridge | reduced | 6/10 | 0.0246965 | 1 | 0/10 |

## Interpretation

Removing ridge does not remove the unit-factor reduced method's recovery deficit: joint9/10 versus reduced5/10 without ridge,9/10 versus6/10 with ridge. Raw/no-ridge methods both recover8/10. Equal counts and median errors do not imply identical trajectories or cases. Thus the normalization/coordinate package materially changes method behavior, and the small ridge term alone cannot explain the reduced failures. This package includes a radial penalty as well as normalized predictions; the experiment does not isolate those ingredients separately.

Raw/ridge is a diagnostic of a known scale escape, not a recommended estimator: every joint run reaches the cap, while reduced raw norms grow as high as527. Raw/no-ridge reduced norms reach757. Algorithm termination does not mean global optimality. No failed or capped case is removed.

The functions available with nonzero factors are unchanged by row normalization when readouts can compensate. At zero ridge, the reduced objective is invariant to nonzero factor scaling, yet its parameter-space trust-region dynamics need not be invariant. Coordinate choice is consequential even after eliminating the readout.

## Verification

All60new checkpoints independently replay direct versus polynomial predictions, exact Gaussian source errors, penalized objectives, QR readout optima (reduced), transfer OLS and exact population errors. All adaptation designs have rank10. Checkpoint and source-data hashes match. Eight crossed objective/Jacobian fixtures agree with independent autograd to maximum relative error9.69e-14. The20archived controls retain their prior independent audit. Unit-factor controls contain no activation RMSNorm.

A separate exact-geometry check in10initial teacher/penalty fixtures establishes a mechanism worth isolating next: see PROFILED_GAUSS_NEWTON_GEOMETRY.md. It is not yet a causal explanation for the final success counts.

## Decision

Close this normalization/ridge sweep. Do not search more penalties or seeds to recover a preferred ranking. The specific next optimizer control is to compare the full reduced-residual Gauss–Newton matrix with the profiled joint-linearization matrix, using the same reduced objective, gauge convention, damping and line search. This isolates the newly derived positive-semidefinite term. Preserve failed starts and held-out transfer; a change in source loss alone is insufficient.
