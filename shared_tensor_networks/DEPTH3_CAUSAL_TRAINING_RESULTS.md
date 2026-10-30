# Learning with internal-ablation supervision: mixed result

Twenty predeclared fits completed and passed the independent audit. Fixed known first layer, two bilinear second-layer features, three product-task outputs; five teacher fixtures and two random initializations. The fixtures share the same original feature-product span but specify different internal axes and output readouts. This is a depth-three causal-supervision mechanism test, not a broad sample-efficiency or end-to-end recovery benchmark.

| Supervision | Ordinary population MSE | Ablated population MSE | Ordinary fit <1e-8 | Ordered component recovery | Successful solver status |
|---|---:|---:|---:|---:|---:|
| Ordinary only | 0.00905165 | 0.73343 | 5/10 | 1/10 | 10/10 |
| Ordinary + ablated | 0.0413492 | 0.0354071 | 3/10 | 3/10 | 9/10 |

Errors are medians of five teacher means, averaging both starts and three tasks within teacher. Component recovery requires both Gaussian feature cosines>.99. Permutation-matched recovery counts are the same as ordered counts here. Ablated errors use the original task normalization; each ordinary teacher task has Gaussian RMS1, while its ablated target need not.

## What changed, and what did not

Adding the specified ablation labels reduces teacher-mean ablated error in5/5fixtures, but raises teacher-mean ordinary error in5/5. It increases axis recovery from1/10to3/10 while reducing near-exact ordinary fits from5/10to3/10. The three successful causal exact fits recover both axes to numerical precision. Extra supervision is informative, but this fixed optimizer does not reliably exploit it. It would be misleading to call this a general improvement in learnability.

All five near-exact observation-only fits recover the hidden feature plane (projector distances below6e-12), yet only one meets the component cosine threshold. This agrees with the exact ambiguity analysis: output matching constrains the plane without uniquely selecting its axes. Crossing a cosine threshold by chance does not make an observation-only solution identifiable.

The harder second initialization fails the ordinary fit threshold for all five teachers in both arms. Causal supervision also loses two of the five near-exact ordinary fits available from the first initialization. The capped causal case is teacher263103/start264002, ending at300function evaluations; it remains in every aggregate. A solver success flag is not a global-minimum or exact-stationarity certificate. No cases were restarted or tuned.

## Verification

Independent literal bilinear forwards agree with polynomial coefficient predictions. A625-point tensor Gauss-Hermite rule replays population errors and feature inner products; it is polynomial-exact here because every squared error has degree at most eight in each Gaussian input coordinate. Independent weighted feature SVDs reproduce hidden-span distances. Teacher hidden mixtures and exact task normalization are verified. Initial/final analytic Jacobians for both observation modes agree with centered forward differences; maximum relative discrepancy is9.18e-12. All20final objectives and metrics pass. The first auditor run reached JSON serialization and failed on a NumPy integer; converting that count to a Python integer fixed reporting without changing scientific inputs or outcomes.

Training uses exact noiseless population loss and dense trust-region least squares. The causal arm has six output losses versus three and changes the objective; this is not an equal-information or equal-compute optimizer contest. The audit does not replay internal trust-region decisions.

## Decision

Close this bounded test. It confirms that informative internal measurements can select computational axes when optimization succeeds, while exposing optimization failures even in a17-parameter model. Do not search rates or restarts to force10/10recovery. Keep span, axes and intervention errors separate in deeper studies. The ongoing h64 task/input grid remains the scaling priority; general deep circuit recovery and the broader goal remain unfinished.
