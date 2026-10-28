## H625 — Context dependence arises inside downstream computation as well as in final readout

The previous turn was progress: H624 showed different small-dose vocabulary responses to one fixed injected vector, but did not distinguish intermediate transformations from final normalization. H625 captures native and±0.02 trajectories for the same72 prompts, recording Attention5 output and downstream attention/MLP/block outputs through block17. This adds216 model evaluations. Fit indices remain items0–3 within each cell, with items4–7 held out; capital-only discovery uses four capital form0 prompts.

Reconstruct final readout analytically. For residual x, r=sqrt(mean(x²)+epsilon), normalized y=x/r, and tangent dx, the normalization derivative is dy=dx/r−x·mean(x·dx)/r³. Unembedding maps dy linearly; the final30·tanh(raw/30) contributes factor1−tanh²(raw/30). Use float32 machine epsilon, matching the model's RMS default, and the actual unembedding weights. The analytic response agrees with measured logit tangents within0.00050 relative weighted error at worst; native logit reconstruction has maximum error2.04e-5.

**A shared pre-readout residual response plus native readout is insufficient.** Replace each prompt's final residual tangent with the mean capital-discovery tangent, then apply that prompt's own analytic RMS/unembedding/tanh derivative. Held-out response errors remain1.41–1.82 across language forms, and1.44 on capital questions. Thus the previous logit-response differences cannot be explained solely by applying different final readouts to this shared response. This rejects the tested common-vector hypothesis, not every possible low-dimensional representation.

**The native readout also matters.** Conversely, pass each prompt's actual final residual tangent through the mean readout derivative of the four capital-discovery prompts. Errors are0.40–0.82 across language forms, and0.17–0.68 across capital forms. Both the incoming residual response and the readout depend on context. Their errors are not additive and cannot be read as percentages of the total effect.

| Held-out cell | Own response/native readout audit | Capital mean response/native readout | Pooled response/native readout | Cell mean response/native readout | Own response/shared capital readout |
|---|---:|---:|---:|---:|---:|
|capital/0|0.0001|0.2875|0.8796|0.2875|0.1703|
|capital/1|0.0001|0.6668|0.8729|0.4280|0.4258|
|capital/2|0.0001|1.4370|0.9035|1.1107|0.6756|
|language/0|0.0002|1.4085|1.0171|0.6304|0.5252|
|language/1|0.0003|1.6305|1.2634|0.3524|0.8198|
|language/2|0.0002|1.8222|0.9350|0.3710|0.4031|
|plural/0|0.0001|1.1640|0.9274|0.5046|1.1992|
|plural/1|0.0001|1.9479|0.8323|0.3446|1.9563|
|plural/2|0.0002|2.5353|1.0522|0.7781|0.7318|

These errors are native-probability-weighted centered logit norms relative to the true response, with common logit offsets removed. “Shared readout” means the average Jacobian of the four native capital readouts; it is not a trained replacement output layer. Mean residual fits are simple averages, not output-error-optimal fits. Per-cell fits have more parameters and known relation/form labels, and even those leave material errors. The own/native column is a reconstruction sanity check, not predictive generalization.

**Divergence begins immediately after the injection, before final readout.** Attention5's injected tangent is the same vector across prompts, verified to1.18e-5 relative error. The next subcomputation is residual normalization followed by MLP5. At block5 output, the capital-discovery-mean response has held-out error0.059 on the original capital form,0.551 on short capital statements,0.626 on capital questions, and0.429/0.925/0.714 on the three language forms. Since the preceding perturbation is fixed and the native residual has no alpha dependence before the injection, this first divergence occurs in the normalized MLP5 response.

This does not yet distinguish the derivative of the RMS normalization entering MLP5 from the bilinear MLP's own context dependence. Nor does it show that MLP5 is the dominant cause of the final difference: later layers also transform the response, with possible amplification or cancellation.

| Held-out cell | Block5 capital-mean error | Block5 cosine | Block8 error | Block11 error | Block17 error | Block17 cosine |
|---|---:|---:|---:|---:|---:|---:|
|capital/0|0.0595|0.9982|0.2218|0.2716|0.4159|0.9307|
|capital/1|0.5513|0.8665|0.7386|0.8265|0.6464|0.8448|
|capital/2|0.6256|0.7908|1.0440|1.3212|1.9527|-0.2738|
|language/0|0.4290|0.9186|0.6782|0.9805|1.5028|0.5757|
|language/1|0.9246|0.6539|1.8687|1.4969|2.0250|-0.3530|
|language/2|0.7140|0.7315|1.0014|1.2786|1.7867|-0.5278|
|plural/0|0.5193|0.8808|1.0612|1.0778|1.1815|0.1096|
|plural/1|0.6537|0.8183|1.2098|1.2583|1.4939|-0.0859|
|plural/2|0.7280|0.7347|1.2390|1.4383|2.2645|-0.0398|

The language “speak” form has block5 cosine0.654, block8 cosine0.053 and final cosine−0.353 relative to the capital mean. Language questions end near−0.528. Original capital-form excluded items remain near0.931 at the final residual. These are Euclidean residual-space comparisons, distinct from the probability-weighted logit metric above. A negative cosine is evidence of a changed direction under that metric, not a semantic sign label. Layer trajectories are descriptive local derivatives, not an additive attribution of causal responsibility.

For every layer, the tangent accounting identity reproduces residual skip contribution plus attention and MLP write derivatives, with maximum relative discrepancy3.54e-5. This makes a hook bookkeeping error or unexplained tangent source unlikely at the measured precision. Final-position perturbations cannot causally affect earlier token positions in this model; the captured final-position path is therefore the relevant downstream response trajectory, while native earlier-position context remains available to attention.

Research interpretation: the reusable intervention direction does not carry one fixed vocabulary effect through the network. Its effect is transformed according to the recipient's state starting at the first normalized MLP and continuing downstream, with a further context-dependent readout. That supports describing a computational unit jointly with its operating context and readout, rather than equating a residual vector with a universal semantic feature. It still does not determine whether a compact conditional operator, a small set of context regimes, or a higher-dimensional family best captures this dependence.

Next isolate the first response transformation analytically: factor MLP5's Jacobian into its incoming RMS derivative and the bilinear derivative at its native normalized input. Swap or average those two factors separately on the declared discovery subset and test held-out responses. This can distinguish normalization geometry from context-dependent bilinear multiplication without changing the injected vector or selecting successful prompts. Later-layer causality can then be tested with targeted response replacement if the first factorization identifies a useful mechanism.

Audits: all source/vector/manifest/prior/runner hashes unchanged;216 forwards; prior native logits reproduce exactly; injected tangent discrepancy1.17e-5; residual accounting3.54e-5; native final-readout reconstruction2.04e-5; analytic tangent max relative error0.000499; all metrics finite. No new natural-text documents; FineWeb remains26,112. Broad goal active and incomplete.
