## H556 — Small L8 responses are dominated by bilinear linear cross terms

Reuse all 80 H554 prompts and destination-phase edge deltas, at ±1 and ±1/16 amplitudes. Capture the actual residual x entering L8 MLP normalization; define z=RMSNorm(x), e=RMSNorm(x+delta)−z, and B(z)=Down(Left(z)*Right(z))+bias. The exact finite algebra is:

B(z+e)−B(z) = Down(Left(z)*Right(e)+Left(e)*Right(z)) + Down(Left(e)*Right(e)).

The first term is linear in normalized displacement e but depends on native context z. The second is quadratic in e. This does not make the full model globally linear: normalization, context and later responses remain variable.

Test five local responses: full, linear normalized term only, quadratic only, fixed native RMS denominator, and the full first-order tangent including the RMS derivative. Replace only the final-position L8 MLP output response; preserve the attention perturbation and live later computation. This yields 1,600 causal interventions. The fixed-denominator counterfactual uses (x+delta)/s0 rather than renormalizing, so its difference from full is not a unique additive normalization attribution.

| Family, +1/16 amplitude | Quadratic / full response norm | Fixed-denominator relative error | Tangent relative error | Linear-only output-margin MAE |
|---|---:|---:|---:|---:|
|binding|0.21%|1.71%|0.22%|0.0000030|
|color_prose|0.51%|7.19%|0.47%|0.0000326|
|color_record|0.43%|10.38%|0.37%|0.0000236|
|name_prose|0.51%|5.42%|0.49%|0.0000255|
|name_record|0.65%|7.02%|0.61%|0.0000511|

At +1/16, quadratic response norm averages 0.21–0.65% of the full response across families. Linear-only replacements preserve the live margin closely (mean absolute deviations 3e−6 to 5.1e−5 nats). Omitting the linear term causes much larger deviations. The context-dependent linear cross term is therefore a useful local approximation for this selected MLP response.

At full positive amplitude, quadratic fractions grow to 3.4–11.1%, and linear-only margin errors reach 0.00076–0.01260 nats. Finite-amplitude corrections matter. Freezing the RMS denominator changes the local response by 1.7–10.4% at +1/16; normalization contributes but does not account for most of the response under this comparison. The smallest positive perturbations average 0.069–0.232% of the pre-MLP residual norm, giving a concrete scale for the local approximation.

Audits: reconstructed pre-normalized input matches captured normalization exactly; full-response causal replay agrees with H555 within 8.59e−6 nats. Float32 bilinear decomposition has max coordinate error 3.37e−4 and maximum relative vector error 7.26e−4, consistent with subtracting large native outputs. A separate audit promotes the same normalized endpoints and bf16-rounded weights to float64: the algebra agrees to max coordinate error 5.30e−12 and relative error 2.53e−12. This verifies the formula numerically; it does not establish that L8 alone explains the downstream reversal. Source unchanged, all results finite.

Next H557 tests whether the local bilinear response operator transfers between native contexts: substitute the base context z in the linear term while retaining the recipient's actual displacement e, quadratic term and native output. Compare opposite tokens, formats, vocabularies and a leave-pair-out family mean. This separates context dependence of the bilinear operator from the recipient normalization change; it is not a whole-state activation transplant.

No new prompts or FineWeb documents. Total corpus 26,112. Broad goal active and incomplete.
