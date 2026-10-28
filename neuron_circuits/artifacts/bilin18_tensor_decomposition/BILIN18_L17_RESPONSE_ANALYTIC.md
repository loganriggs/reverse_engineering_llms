

## H288 — Exact response-only optima reveal little remaining pooled headroom in these64 features

Compute training-only float64 normal-equation eigensolves for joint actual+angle/all-head responses, each head group separately, and each payload family separately. Same768 training/256 validation documents,64 fixed features, normalized feature differences and native residual-response targets as H287. No intact objective, penalty or new intercept. Primary relative Gram eigenvalue cutoff1e-12; sensitivities1e-8 and1e-4. Each solves a finite empirical output-MSE problem, not downstream KL. Full sufficient statistics are now saved for future audits without recollecting native outputs.

All five Gram matrices have rank64 at every cutoff; condition numbers range858–3028. The largest relative normal-equation residual is below5e-15. All cutoff variants yield bitwise-identical float32 exported matrices. Float64 and exported-float32 training objectives agree to displayed precision. A focused test verifies the eigensolve against direct least squares and the quadratic excess-loss identity. These checks support a numerical global minimum for each finite training linear least-squares objective. They do not bound distribution-wide error, larger feature classes, or semantic representations. Inputs/targets were collected using float32 native execution, then accumulated/solved in float64.

This is a diagnosis on the EXISTING validation panel, not a fresh confirmatory test. No fit uses validation targets and no cutoff is selected by validation. Energy pooling weights each group by its base response-error energy. Compare matched specialists only on their own group; their pooled performance on all groups is not their specialist optimum.

| Model | Training pooled error/base | Validation pooled error/base | Validation intact KL |
|---|---:|---:|---:|
|base|1.000000|1.000000|0.056147|
|state_response|0.981070|0.982240|0.051253|
|joint_r1e-12|0.960995|0.969821|0.058870|
|h280|1.046488|1.045775|0.052036|

The joint response-only optimum lowers validation response MSE by only1.26% relative to H287, while worsening intact KL by14.86%. Relative to the frozen base, it removes3.02% of remaining response error; H287 removes1.78%. This is additional correction of base error, not total native-response fidelity.

| Split | Joint response ratio | Matched head-specialist ratio | Matched payload-specialist ratio |
|---|---:|---:|---:|
|train|0.960995|0.945555|0.946035|
|validation|0.969821|0.968867|0.963964|

Validation joint-versus-matched-head-specialist excess error is0.10%. Specialist comparisons describe separate learned functions with known intervention family/group; they are not a single intact readout or an input-only routing algorithm.

Locked diagnostic gates: headroom: FAIL, state_retention: FAIL, small_interference: PASS. No confidence intervals or fresh replication claimed for these validation diagnostics.

**Interpretation.** Removing the intact-state objective does not reveal a large pool of additional average response accuracy within this fixed64-feature linear correction. H287 is already near this empirical response-only optimum in aggregate validation MSE, while preserving much better intact accuracy. This weakens optimizer failure or too-small response weight as the dominant explanation for the remaining pooled error in this class. It does not prove all useful information has been extracted from these reads: nonlinear/context-dependent maps, new directions, different response distributions and other objectives remain open. The specialists probe distribution dependence separately; do not collapse their results into a claim that fundamental units are independent heads.

**Next discriminator.** Quantify which response errors the exact64-feature solution leaves inaccessible. Add a controlled number of input directions outside the current span and solve the same response-only least-squares problem with identical train/validation interventions. Compare nested64/96/128 feature spans, preferably separating natural-state variance directions from response-informed directions learned only on discovery. Report parameter cost, conditioning, train/validation gap and intact KL. A large held-out response improvement would locate a missing-direction bottleneck; little improvement would motivate nonlinear/context-dependent features. Freeze feature choices before a new confirmation panel. The current result alone is not grounds to claim intrinsic64-dimensional computation.

All jobs terminal. Existing native model/compiler/base remain unchanged; analytic fitted adapters and statistics stored separately. Goal remains active.
