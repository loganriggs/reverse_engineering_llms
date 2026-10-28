

## H375 — Selected local-oracle tails are low-probability extrapolation failures repairable within the same span

Select H374’s worst moment/donor conditions4/11 at site1007 (document251, fourth sampled position), the actual response at that same site, and the median-H374-oracle16-error site within each selected condition. These are six outcome-selected diagnostic cases, including two failures in the same context, not six independent samples or a population confirmation. Freeze amplitude grid, rare-token threshold and repair gates before tracing or fitting.

Scale both native target delta and local-oracle16 patch by t in {0,1/64,1/16,1/8,1/4,1/2,3/4,1}. Compare native capped logits with a separate path that linearizes the cap around baseline but keeps softmax exact. The latter changes both target and patch and is a diagnostic, not an uncapped model. Optimize full-amplitude target-to-patch KL within the identical16 directions using LBFGS with strong-Wolfe line search, at most200 iterations and three starts: zero, Euclidean coefficients, and local Fisher-oracle coefficients. Full target distribution is known throughout; this remains an oracle.

|Selected case|Local-oracle finite KL|Best finite-fit KL|Finite-fit recovery|
|---|---:|---:|---:|
|moment_site1007|5.40038|6.24243e-05|0.583|
|moment_median|0.000316913|0.000316868|0.303|
|donor_site1007|5.53076|6.18559e-05|0.590|
|donor_median|0.000307249|0.000307147|0.539|
|actual_site1007|1.63932e-05|1.18573e-05|0.878|
|actual_median|0.000217307|0.000217057|0.839|

All selected-case gates pass: {'rare_token_takeover': True, 'finite_fit_repairs_tail': True, 'finite_fit_beats_baseline': True}. For both tail cases, more than99.5% of patch probability lands on tokens assigned baseline probability below1e-6. Best finite fits reduce KL by more than99.99% and beat the unchanged baseline, recovering58.3%/59.0% of the native change. This proves existence of much better finite coefficients within the same span at these cases, not global optimality or universal span sufficiency.

Token trace: the token “ to” has baseline probability1.48e-9 and native target probability about1.2e-10. The moment/donor local oracle adds38.65/39.14 to its raw logit and24.58/24.70 after the cap, raising its probability to98.63%/98.18%. The token “ of” receives most of the remaining mass. The intact distribution is instead concentrated on “ank” (99.36%) and “anks” (0.50%) after a context ending in “Kevin D”. The actual response at this same site does not produce the takeover. These token identities describe a selected failure, not a semantic label for the16 directions.

At amplitude1/16, tail KL is about1e-6; at1/2 it grows to0.037/0.044 nats; at full amplitude it reaches5.40/5.53. With a locally linearized cap and exact softmax, full-amplitude KL is even larger (19.29/19.78). Thus cap curvature is not necessary for this observed instability and the exact cap reduces its size relative to that linearized path. The mechanism is consistent with the local probability metric giving little cost to initially unlikely-token logit changes that become consequential after a finite softmax shift.

Numerical checks: selected H374 KL replay has maximum relative error5.3e-14. All three finite-fit starts converge to nearly identical objective values at the two tails (differences below1e-12 nats), with gradient norms below1e-8. This is strong agreement among these starts, not proof of a global minimum of capped finite KL. Median-case fits change the already reasonable local-oracle errors only slightly. Complete traces, token values, fit histories, contexts and final coefficients are saved. Native weights unchanged.

Hypothesis update: the extreme selected errors are not intrinsic inability of the16-dimensional span to represent any useful finite response. Local Fisher optimality permits dangerous finite extrapolation into low-probability tokens. Rank16 can still be insufficient for high fidelity, since repaired recovery at these tails remains only58–59%; the diagnostic does not eliminate span limitations.

Next: evaluate a fixed finite-KL safeguard over the full panel, including all tail cases, rather than extrapolating these repairs to aggregate performance. A prespecified line search from zero toward local-oracle coefficients provides a cheap response-dependent lower bound on what the span can do and guarantees no worse than the baseline among evaluated candidates. Compare with unguarded rank16/64 and then assess whether any coefficient rule can be predicted without the full native target. Broader goal remains active.
