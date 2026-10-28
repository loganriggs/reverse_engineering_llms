

## H298 — Shared local descent exists; step-zero selection did not establish objective conflict

At the original weighted rank32 partner basis, accumulate full training-set gradients of separately normalized actual and angle cross-response losses, in256-example chunks. Same first768 discovery documents and payloads as H296–297; no stochastic gradient estimate. Project gradients into the horizontal tangent space orthogonal to both frozen core96 and partner32. Test two unit-Frobenius directions: negative sum of unit-normalized family gradients, and negative sum of raw family gradients. Fixed steps.001/.003/.01/.03/.1/.3/1, followed by QR retraction. Score both families on training and the existing256-document validation panel. No candidate is promoted or tested fresh here.

Actual tangent gradient norm.0237077, angle.0400893, cosine+.0900593. The gradients are weakly aligned rather than opposing. Unit-sum direction loss derivatives are−.0175025 actual and−.0295963 angle; raw-mean direction derivatives−.0133874/−.0349902. Central finite differences at±.003 agree to relative<.02%. This verifies the local first-order prediction at the tested scale. Native execution/gradients use float32 with double loss accumulation; this is a numerical diagnostic, not an exact symbolic proof.

| Direction | Step | Actual training gain | Angle training gain | Actual validation gain | Angle validation gain |
|---|---:|---:|---:|---:|---:|
|unit_sum|0.1|0.166pp|0.277pp|0.086pp|0.207pp|
|unit_sum|0.3|0.442pp|0.723pp|0.220pp|0.526pp|
|unit_sum|1.0|0.917pp|1.544pp|0.336pp|1.066pp|
|mean_loss|0.1|0.126pp|0.325pp|0.069pp|0.241pp|
|mean_loss|0.3|0.335pp|0.839pp|0.171pp|0.604pp|
|mean_loss|1.0|0.682pp|1.784pp|0.224pp|1.252pp|

Locked diagnostic gates: common_training_descent: PASS, validation_both_gain: PASS. Common training descent requires at least one fixed step improve both training fidelities. Validation-both-gain requires at least.1pp for each family at some fixed step. These are small local diagnostic thresholds, not the previous5/10pp or80% compactness gates. No such compactness success is claimed.

**Inference.** H297 selected step zero from its Adam checkpoint grid, but the proxy is not locally Pareto-stationary for these training objectives: a direction with negative derivatives for both losses exists, and actual small steps improve both training and validation. Thus unavoidable local objective conflict is not sufficient to explain the earlier training outcome. This keeps optimizer/preconditioning, step scale and sparse checkpoint selection as plausible factors. The audit does not distinguish those factors individually, establish a global compact optimum, or prove conflict cannot arise elsewhere.

The largest tested unit-sum step yields validation actual fidelity16.93% versus16.60% baseline and angle16.38% versus15.31%. Gains remain small and absolute coverage low. Validation is reused across many experiments; reporting all fixed steps does not turn this into fresh confirmation. No intact or downstream-KL measurements are made. Partner rank and native cross coefficients are unchanged; dense compilation still costs3,538,944 output coefficients.

**Next discriminator.** Run a short, fixed-budget full-gradient common-descent procedure with training-only backtracking. At each iteration form the unit-gradient-sum tangent direction, choose a step only if BOTH training losses decrease, and retract. Keep a maximum iteration/line-search budget and log rejected steps, gradient norms and family cosine. Validation monitors/selects the final frozen candidate but must not choose training step lengths. Compare with proxy and previous Adam runs, then use fresh documents only if gains merit confirmation. This tests whether repeated controlled steps accumulate useful coverage or quickly stall. Do not infer fundamental units from successful optimization alone.

All jobs terminal. Full gradients, candidate bases, scores and derivative checks saved under bilin18_l17_partner_gradient_audit*. Original native model/compiler/base unchanged. Broad goal remains active.
