

## H330 — Behavioral scale of exact response components

Reuse H327 documents and interventions. Exact edited-minus-intact final residual difference is direct residual change plus MLP response. Split MLP response into compiler response and residual linear, within-core, core/partner cross, missing cross, and complement quadratic terms. For each term t, measure KL(native edited distribution || distribution at native edited state−t). Include full source, full MLP and full compiler-residual removals. All other terms remain native. These are interventions on an exact algebraic decomposition, not independent additive behavioral contributions.

|Target|Removed term|Mean KL (nats)|Top1 agreement|Term energy / full-source energy|
|---|---|---:|---:|---:|
|actual|source|0.0012926|98.89%|1.00000|
|actual|mlp|0.00061479|99.35%|0.43128|
|actual|residual|0.000212992|99.31%|0.02323|
|actual|direct|0.00164038|98.81%|0.35547|
|actual|compiled|0.000409201|99.57%|0.39761|
|actual|linear|3.529e-05|99.76%|0.00257|
|actual|within|0.000109406|99.63%|0.00948|
|actual|partner_cross|2.4193e-05|99.80%|0.00200|
|actual|missing_cross|3.44039e-05|99.72%|0.00539|
|actual|complement|3.82976e-05|99.54%|0.00411|
|angle|source|0.00176532|98.46%|1.00000|
|angle|mlp|0.000717663|99.39%|0.33142|
|angle|residual|0.000264018|99.46%|0.04348|
|angle|direct|0.00169685|98.35%|0.48645|
|angle|compiled|0.000647389|99.65%|0.28761|
|angle|linear|4.68824e-05|99.78%|0.00586|
|angle|within|9.20187e-05|99.80%|0.00828|
|angle|partner_cross|4.00654e-05|99.78%|0.00372|
|angle|missing_cross|4.96239e-05|99.65%|0.00955|
|angle|complement|8.21849e-05|99.61%|0.01509|

|Target|Missing-cross KL divided by|Ratio [conditional95%]|
|---|---|---:|
|actual|source|2.662% [2.424,2.915]|
|actual|mlp|5.596% [4.827,6.498]|
|actual|residual|16.153% [13.775,18.343]|
|angle|source|2.811% [2.593,3.060]|
|angle|mlp|6.915% [6.102,7.750]|
|angle|residual|18.796% [17.506,20.271]|

Locked gates: missing_under1pct_source: FAIL, missing_under1pct_mlp: FAIL. Ratios must be below1% in both families for the respective gate. Ratios are comparisons of intervention magnitudes, not fractions in a causal variance decomposition. Bootstrap resamples128 documents jointly within each family and conditions on payloads and decomposition.

**Interpretation limits.** Source removal returns to the intact state; MLP-response removal retains the direct residual intervention but keeps the intact MLP write. Individual-term removals are oracle diagnostics. Algebraic terms depend on chosen core, partner and compiler, so apparent behavioral roles are not unique intrinsic units. Small means may hide rare important contexts; per-document/head rows are saved. This panel is reused diagnostically and no semantic or intact replacement claim follows.

**Audit.** Exact source-response reconstruction and H329 missing-component KL replay passed: {'response_identity_relative': 8.701478509465232e-05, 'missing_kl_replay_relative': 0.0}. Frozen geometry/corpus hashes checked; native weights unchanged.

**Next discriminator.** Prioritize terms with material behavioral effects and inspect head/context concentration before further residual compression. Test whether compact source-conditioned computations preserve the larger response effects, with causal controls and fresh confirmation. Broad research goal remains active.
