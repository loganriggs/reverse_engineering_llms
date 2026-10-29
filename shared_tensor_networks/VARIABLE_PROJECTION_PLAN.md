# Exact readout elimination: bounded development comparison

Question: does eliminating linear readouts improve hidden-feature optimization over joint trust-region least squares? This is an optimizer control, not new-task discovery.

Freeze three fresh synthetic teachers/seeds 231101–231103, input d6, hidden h3, six training heads, 256 Gaussian training inputs, label noise SD .05 relative to training-target RMS. Independent student A/B initialization. Both methods start with identical hidden factors and the exact same optimal unregularized readout. Compare joint versus reduced scipy TRF, maximum 100 function evaluations, tolerances 1e-10, no restarts. Reduced readouts solved by augmented least squares; rank loss raises an explicit error rather than silently changing derivative assumptions. No RMSNorm.

Report final training MSE, clean error on fixed 4096 fresh Gaussian examples, nfev/njev and elapsed time. The finite test estimate is a development diagnostic; no model selection uses it, no convergence/generalization ranking from three teachers. Equal evaluation caps do not mean equal compute. Audit final readout normal equations and model forward replay; reduced Jacobian has already passed independent autograd, finite difference and envelope-gradient checks in six fixtures. Next larger comparison should retain whole-task holdouts and population coefficient evaluation.
