# h64 transfer: representation error and finite-label readout error

All60completed h64models were analyzed without retraining or adding labels. The reference readout minimizes Gaussian population squared error for each frozen learned product dictionary. It uses teacher-known moments and is a privileged diagnostic, not a practical estimator.

| Inputs | Tasks | Start | Actual transfer MSE | Population refit error | Excess readout error |
|---:|---:|---|---:|---:|---:|
|8192|520|random|1.43575|0.958019|0.479566|
|8192|520|teacher|0.00106435|1.73615e-05|0.00104678|
|8192|2080|random|0.0320563|0.0212519|0.0108044|
|8192|2080|teacher|0.00104946|5.52133e-06|0.00103977|
|32768|520|random|0.0160153|0.00984821|0.00616711|
|32768|520|teacher|0.00103836|3.83519e-06|0.00103495|
|32768|2080|random|0.0237113|0.0152053|0.00850601|
|32768|2080|teacher|0.00104716|1.8884e-06|0.00104307|

Each column is the median across five teacher means, averaging random starts and held-out tasks within teacher. Separately computed medians need not add exactly. Primary transfer uses8,192existing adaptation examples.

The high-input random models retain population refit errors around.010–.015, while their actual transfer errors are around.016–.024. Both incomplete features and finite-label readout estimation matter. Teacher-start controls have refit errors of only a few parts per million, with most of their roughly.001transfer error attributable to the readout. Poor low-input/low-task models remain near the zero-predictor error1even after population refitting; their observed error above1also includes substantial readout estimation error.

This does not show that the observed model is a local minimum, that all feature error is caused by finite-budget optimization, or that readout excess is noise variance alone. Finite-design misspecification and estimation contribute too. Feature matching, quadratic-span error, quartic task approximation, and sampled transfer are distinct measurements.

## Numerical verification

All Gram matrices retain rank2080at both1e-10and1e-12relative cutoffs; changing the cutoff changes no refit error. Maximum cross term in the exact error decomposition is2.79e-15; independent least-squares replay differs by at most7.44e-15. Saved coefficients reproduce audited transfer errors, all60checkpoint hashes and input records were verified, and all nonnegativity checks passed. Gaussian moment identities are evaluated in float64; these are not arbitrary-precision optimality certificates.

## Decision

Close this descriptive panel. The remaining high-input transfer gap is not solely an underfit readout that better coefficient estimation would eliminate. Preserve the frozen h128grid; no new optimizer rescue follows. Future mechanistic claims about the missed features need interventions or optimization diagnostics beyond this error decomposition.
