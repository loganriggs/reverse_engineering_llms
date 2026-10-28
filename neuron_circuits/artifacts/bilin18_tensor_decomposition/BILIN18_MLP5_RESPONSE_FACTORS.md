## H626 — MLP5 response variation chiefly follows its bilinear operating context, not variation in normalization response

The previous turn was progress: H625 located the first response divergence immediately after normalized MLP5, while leaving input normalization and the bilinear derivative unresolved. H626 captures72 native MLP5 inputs and analytically factors their response to the frozen injected vector v. No new intervention direction or downstream target is fitted.

For raw residual x, let y=RMSNorm(x) and t=D(RMSNorm)_x[v]. The ungated bilinear MLP derivative is B(y,t)=Down[(Left(y)·Right(t))+(Left(t)·Right(y))]. Down bias contributes no derivative. The final MLP5 response depends on both its normalized operating context y and its normalized perturbation t. B is bilinear in these two arguments, allowing exact factor substitutions and a baseline-anchored expansion.

Use first-four capital form0 discovery examples to form mean y and mean t, then evaluate held-out items. Compare native y with shared t, shared y with native t, both shared, and corresponding pooled/per-cell versions. The injected vector v stays fixed. These are local analytic counterfactual response predictions, not yet downstream behavioral interventions.

**Sharing the normalization response works much better than sharing the bilinear context.** Keeping native y and substituting the capital mean t gives held-out MLP-response error0.007 on original capital form and0.021–0.077 across language forms. Keeping native t but substituting capital mean y gives0.086 on original capital form and0.673–1.563 across language forms. Other capital/plural constructions show the same asymmetry. Thus variation in the normalized operating context is the main differentiator under these tested substitutions at this site.

| Held-out cell | Shared context/native normalization error | Native context/shared normalization error | Both shared error | Native context/no radial correction error | Per-cell context/native normalization error |
|---|---:|---:|---:|---:|---:|
|capital/0|0.0859|0.0068|0.0838|0.0491|0.0859|
|capital/1|0.9668|0.0393|0.9221|0.0418|0.1966|
|capital/2|0.8293|0.0523|0.8052|0.0515|0.0511|
|language/0|0.6726|0.0209|0.6690|0.0332|0.1172|
|language/1|1.5629|0.0658|1.4853|0.0181|0.1636|
|language/2|0.9914|0.0766|0.9470|0.0345|0.0480|
|plural/0|0.8168|0.0241|0.8042|0.0237|0.2425|
|plural/1|1.0686|0.0256|1.0545|0.0278|0.1814|
|plural/2|1.0236|0.0402|0.9903|0.0329|0.0934|

Relative errors compare predicted MLP5 write derivatives with central-difference measured derivatives. Values above1 are possible; these are not additive causal shares. Residual block5 errors, which include the unchanged injected skip vector v, are also stored but are a different denominator. Per-cell context means improve accuracy but introduce known cell labels and additional parameters; the table does not establish a universal low-dimensional context representation.

**Input normalization changes the perturbation less across these prompts.** Held-out normalization-tangent cosine with the capital mean is0.9953–1.0000. Raw residual RMS ranges roughly243–265 across cell means. Omitting only the radial correction and using t≈v/RMS(x) produces MLP-response errors0.018–0.052 across cells. This does not make normalization unimportant generally; it says its variation and radial correction are comparatively modest for this vector, layer and panel. Sharing y while retaining only recipient RMS scaling still fails, so scalar normalization gain does not explain the bilinear-context mismatch.

**Euclidean context similarity hides functional differences.** Normalized-context cosine with the capital mean is still0.945–0.9995 across cell means. For “People in … speak”, context cosine is0.9608 but shared-context response error is1.563; its normalization-tangent cosine is0.9953 and shared-normalization response error only0.066. A context can look close in raw vector geometry yet differ in directions that this bilinear operator reads strongly. This supports measuring equivalence through the relevant operator or downstream behavior rather than Euclidean context distance alone.

| Held-out cell | Context cosine to capital mean | Normalization-tangent cosine | Raw RMS | Radial correction / normalized perturbation norm |
|---|---:|---:|---:|---:|
|capital/0|0.9995|1.0000|259.6304|0.1216|
|capital/1|0.9793|0.9986|253.4058|0.0690|
|capital/2|0.9498|0.9992|248.9429|0.1018|
|language/0|0.9876|0.9991|265.1705|0.0804|
|language/1|0.9608|0.9953|248.9816|0.0229|
|language/2|0.9452|0.9982|242.6740|0.0669|
|plural/0|0.9766|0.9986|263.3040|0.0701|
|plural/1|0.9669|0.9983|263.3314|0.0663|
|plural/2|0.9459|0.9983|253.2753|0.0692|

An exact expansion around discovery means is also saved: B(y,t)=B(y0,t0)+B(y−y0,t0)+B(y0,t−t0)+B(y−y0,t−t0). It reconstructs the analytic response to5.59e-15 relative error. These terms depend on the chosen baseline and can cancel; their norms must not be interpreted as causal percentages. The mean of native response products and product of mean factors are separately included in the results.

Most likely local account: the same residual perturbation enters almost the same normalized perturbation direction, but encounters different native bilinear factors. Those factors convert it into different MLP writes. The operating context is therefore part of the computation being described; a direction alone does not specify its function. This is an explicit local mechanism for conditional behavior, but it does not prove that MLP5 generates most of the final answer-emission difference. H625 showed additional transformations downstream and at readout.

Next perform causal response replacement at MLP5. Preserve each prompt's native output, inject the same small signed Attention5 perturbation, and replace only MLP5's first-order response with (a) native analytic response, (b) shared-context response, or (c) shared-normalization response. Keep later layers live and compare final vocabulary tangents. The native analytic replacement is the necessary control for linearization/patching error. If shared-context replacement reduces the final cross-context difference, MLP5 contributes causally; if later layers restore the difference, local factorization alone is insufficient. Do not interpret any change as an additive layer share.

Limitations: same adaptive72-prompt panel and item0–3 discovery convention; the shared y need not be an attainable normalized state, so factor swaps are analytic operator interventions rather than ordinary prompts. The original plural panel is not competent when singular alternatives are included; its derivatives remain valid measurements, not successful task execution. No new response-fit parameters were optimized against output logits.

Audits:72 native forwards; exact native logit replay; reconstructed normalized input error6.99e-7; analytic MLP derivative maximum relative error0.000107 and block derivative0.0000800; exact factor expansion5.58e-15; all source/manifest/vector/trace/runner hashes unchanged and finite metrics. FineWeb remains26,112 natural-text documents. Broad research goal active and incomplete.
