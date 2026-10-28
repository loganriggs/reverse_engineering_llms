## H1022 — Cached head13.0 writes factor exactly into a fixed value map and a nearly constant routing coefficient

Previous turn was progress: H1021isolated head13.0as the stable consumer carrying57–63%of the natural country-swap effect through its cached-value input. This turn reconstructs that conditional write from weights and routing, then tests simple routing and positional approximations against the native finite effect.

For a cache change at source position s, head0at layer13writes

    delta_write(t) = lambda13 * P13,0(t,s) * W_O13,0 * delta_v1,0(s)
    P(t,s) = (Q_t . K_s /128) * (Q2_t . K2_s /128), with causal mask.

W_O13,0is the1152x128slice of native output weights;lambda13is the native cache mixing coefficient. The pattern is the product of two normalized/rotary query-key scores, not a softmax probability; its coefficients can be negative. Since the intervention changes only the cache at layer13, layer13's own Q/Kdo not change. This conditional linearity is architecture-derived, not a discovered absence of nonlinear computation. Later residual, attention, normalization and MLPresponses remain live.

Exact native versus reconstructed head-write difference:3.12e-7relative maximum. Adding the compiled delta write to unedited native attention reproduces the finite output effect within4.77e-6absolute. The isolated-cache effect exactly replays H1021. Native prefix/QKand baseline head/suffix still supply inputs and behavior; this is a conditional port-to-write executor, not a replacement for the complete attention head or full model.

| Approximation | Discovery | Familiar/templateB | New languages,A | New languages,B |
|---|---:|---:|---:|---:|
| Discovery-average routing |2.82%|9.55%|2.26%|4.21%|
| Exact routing, final-token write only |0.25%|0.22%|0.19%|0.15%|
| Exact routing, source-token write only |99.92%|99.96%|99.89%|99.92%|

Scores are relative L2error against the head13.0isolated cached-input finite effect, not against the complete country-swap effect. Both source and final positions are downstream of the changed country token; all earlier positions have zero edit by causal masking. Almost all behavioral effect comes from the final-position write, despite a nonzero source-position write.

Fit two mean routing coefficients from the8discovery examples only (source/self offset0 and final-token offset1). Frozen means are-0.0620and-0.6738. Final-token routing mean/std is-0.6738/0.0177discovery,-0.6750/0.0130newlexicalA,-0.6568/0.0169newlexicalB,-0.6282/0.0273familiar/templateB. This explains why a fixed scalar works on this restricted template family and where template variation costs fidelity. It does not establish context-independent routing on arbitrary language tasks, long contexts, or competing country mentions.

Interpretation: the tested reusable object is a signed scalar routing rule applied to a fixed128-dimensional cached lexical-value-to-residual map. Vocabulary-specific content remains rich, while the operation can be stable across countries. This supplies a concrete alternative to forcing all language concepts into one small learned direction span. It aligns with the prior repository's weight-space/transport formulation and should not be described as discovery of attention algebra itself. Whether this map is a reusable computation across tasks still needs testing; current success is within one task family.

Audits: independent outer-product reconstruction confirms3.12e-7maxrelative write error; independentTorch finite-effect metrics agree; discovery-only routing means independently checked;meanfile hash unchanged;finite; no evaluation fitting.32pre13partial continuations,64localhead13calls,192suffix sequences;no newgradients. Historical boundary states reused, no reservedBSFtest data, natural-document pool26112unchanged. Saved per-example routing, fixed payload vectors, exact/native write tensors and effect records. No live process,H961pending,broad goal active.

Next fold the token-local first-layer value map into this fixed head13.0OVmap and export a small executable source-embedding-to-write program with explicit input/cost accounting. With fixed discovery routing, the write should no longer require native QKqueries; verify that reduction rather than merely claim it. Then challenge routing with varied contexts/competing countries and test another task that uses head13.0. Keep full-model contribution, isolated-path fidelity, parameter compression and semantic interpretability separate. This is progress toward a reusable operation, not resolution of the full featurizer/manifold objective.
