# Nonlinear local fits approach the information prediction

All450 predeclared fits completed with successful solver status, and the independent audit passed. Five fresh input/readout fixtures, three sample counts, ten paired noise draws and three noise scales. Each fit starts at the true hidden representation; these are privileged local estimation tests, not discovery or global optimization experiments.

The primary comparison is the hidden-coordinate estimate against the linearized prediction for the SAME noise draw. Relative discrepancy decreases in150/150 paired three-scale sequences. Medians below average ten draws within each fixture, then take the median across five fixtures.

| Inputs | Noise SD | Relative nonlinear/linear discrepancy | Empirical risk / predicted risk |
|---:|---:|---:|---:|
| 16 | 1e-03 | 0.00447678 | 1.2330 |
| 16 | 1e-04 | 0.000454958 | 1.2308 |
| 16 | 1e-05 | 4.52673e-05 | 1.2305 |
| 64 | 1e-03 | 0.000295561 | 0.9549 |
| 64 | 1e-04 | 2.95576e-05 | 0.9545 |
| 64 | 1e-05 | 2.09898e-06 | 0.9545 |
| 256 | 1e-03 | 8.16078e-05 | 1.0994 |
| 256 | 1e-04 | 7.2422e-06 | 1.0993 |
| 256 | 1e-05 | 6.55238e-07 | 1.0993 |

At N16 the median discrepancy falls from about0.448% to0.00453% as noise decreases100-fold. The largest individual discrepancy falls from8.19% to0.0965%. For N64/256 the discrepancies are smaller. This supports the local linearization in these fixtures; it does not establish a uniform asymptotic rate or a usable basin radius.

The risk ratio compares only ten noise draws per fixture with an expectation, so its deviations from one are not evidence against the covariance identity. Reusing the same standardized noise across scales preserves finite-draw variation; reducing sigma does not eliminate that Monte Carlo discrepancy. The earlier4000-draw linear checks address the expectation more precisely.

## Solver and verification limits

All fits use the same100-evaluation cap; observed maximum nfev=19. Total measured solve time was 0.985s on one pinned CPU core, excluding audits and fixture preparation. Successful solver status includes relative-step/loss termination: the largest final gradient norm is 2.55e-07, so this is not an exact-stationarity certificate.

Independent literal quadratic products and SVD readout fits reproduce saved outputs, losses, coefficients and linear responses. Fresh RNG replay verifies all inputs/readouts/noise. All450 final residual Jacobians are checked by centered differences for gradient-norm replay; maximum discrepancy 4.44e-11. Noiseless profiled derivatives agree with independent finite differences to relative error 4.88e-10. These checks do not replay every internal trust-region decision.

## Interpretation and next step

The information matrix now has support from both exact linear-noise checks and local nonlinear estimator behavior. The remaining explanatory gap is between this teacher-neighborhood geometry and random-start learning. Measuring spectra at trained states requires accounting for nonzero residual curvature; the exact-fit information Hessian identity cannot simply be applied everywhere. Keep this local test closed and complete the ongoing h64 training/evaluation/audit before selecting a larger follow-up. Depth3–5 reusable recovery and the broader scaling objective remain unfinished.
