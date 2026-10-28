## H628 — Validated local-operator propagation identifies an early downstream region affecting language sign

The previous turn was progress: H627 showed that replacing MLP5's first-order output does not eliminate downstream context dependence. H628 builds a different intervention: propagate a tangent through each block's local operator, then substitute context-specific operators while preserving the evolving incoming tangent.

Capture native block inputs, embeddings and persistent value bus for72 prompts. Starting from the verified analytic block5 tangent, use forward-mode automatic differentiation to apply each block6–17 Jacobian to a perturbation at its final input position. Earlier positions, original embeddings and persistent bus remain fixed. This restriction is valid for the studied injection because a causal final-position perturbation cannot alter earlier token positions and the persistent bus was created before the injection. Apply each recipient's analytic native final normalization/readout to the resulting tangent.

**The native chain reproduces end-to-end measured responses.** Maximum residual-tangent error across prompts and blocks is0.000401; maximum final weighted logit-response error is0.000563 relative to the±0.02 finite-difference reference. Native logits replay exactly. Agreement supports the implementation of the chain rule; it is not itself discovery of a compact model, since the chain still uses the model's actual native states and weights.

Then replace selected block operators with the arithmetic mean Jacobian action from four capital form0 discovery contexts. Each reference Jacobian acts on the same incoming recipient tangent before averaging. This is an operator substitution, unlike injecting a fixed response vector. Test all blocks6–17, early6–9, middle10–13, or late14–17, with all other operators and final readout native. The initial block5 tangent remains recipient-specific. Evaluate item4–7 separately from discovery.

**The early downstream region affects the language sign reversal.** For language prompts ending in “speak”, the native intended-token log-probability derivative is−0.235. Replacing blocks6–9 operators with the capital reference mean changes it to+0.037. Middle-only and late-only replacement leave it negative at−0.079 and−0.154. This makes early context-dependent transformations a concrete target for finer dissection. It does not prove they alone cause the native behavior: the replacement is a particular constructed operator change, and other regions contribute too.

| Held-out cell | Native answer derivative | Shared early6–9 | Shared middle10–13 | Shared late14–17 | Shared all6–17 |
|---|---:|---:|---:|---:|---:|
|capital/0|0.5448|0.6067|0.6057|0.7579|0.6204|
|capital/1|2.8760|3.0736|2.2574|3.6283|1.9519|
|capital/2|0.4922|0.4102|0.3019|0.4326|0.5628|
|language/0|0.1148|0.0284|0.0682|0.2697|0.0588|
|language/1|-0.2353|0.0370|-0.0786|-0.1539|-0.0679|
|language/2|0.0904|-0.0031|0.0669|0.2126|-0.0314|
|plural/0|0.2220|-0.0637|0.1416|0.9307|-0.2721|
|plural/1|0.3563|-0.1535|0.1110|0.1391|-0.4637|
|plural/2|0.1178|0.0664|0.1152|0.1746|-0.3934|

Derivatives use addition of the frozen Attention5 vector as the positive sign. They are local log-probability changes, not finite-dose answer accuracy. Language form1 is the “People in … speak” construction; language form2 is a question followed by “Answer:”. Plural results remain distribution measurements on prompts that fail the corrected singular-versus-plural competence check.

**Region effects are not additive.** Although early replacement flips the “speak” derivative positive, replacing all downstream blocks leaves it negative at−0.068. For language questions, early replacement makes the derivative nearly zero (−0.003), whereas late replacement increases it to+0.213. Operator composition changes the perturbation that reaches subsequent operators. Consequently, neither differences nor response-error norms can be interpreted as percentages of the native effect attributable to a region.

| Held-out cell | Early replacement response change | Middle replacement change | Late replacement change | All replacement change |
|---|---:|---:|---:|---:|
|capital/0|0.1963|0.1360|0.3700|0.3034|
|capital/1|0.2185|0.3390|0.5193|0.5580|
|capital/2|0.6772|0.5843|0.4830|0.9136|
|language/0|0.7269|0.5216|0.9449|1.0844|
|language/1|1.2735|0.8187|0.6673|1.0797|
|language/2|0.9689|0.4154|0.8641|1.3968|
|plural/0|0.9751|0.6992|0.7438|1.1143|
|plural/1|0.9233|0.5676|1.5076|1.4484|
|plural/2|1.1562|0.7150|0.7493|2.1259|

Response change is the native-probability-weighted centered norm of the difference between replacement and measured native logit tangents, divided by native tangent norm. It measures the intervention's effect on the response, not fidelity of a trained predictor or a causal share. Replacing all downstream operators gives errors1.08–1.40 on language and1.11–2.13 on plurals, versus0.303 on held-out original capital form. This demonstrates sensitivity to the chosen operator context; it does not show that a different shared or conditional operator approximation could never work.

Reference operators use their donors' complete native sequence contexts and final token positions, even when recipient length or final token differs. All operate on the same1152-dimensional final residual tangent, but semantic roles need not align. Averaging Jacobians is a linear operator intervention and need not correspond to any attainable native prompt. Native forward states are not changed, only first-order propagation. This is a controlled local mechanistic test rather than ordinary activation patching at finite amplitude.

Current interpretation: the evolving perturbation is transformed by a sequence of recipient-dependent operators. A reusable residual direction alone is insufficient to specify its eventual behavior, and one downstream region cannot be assigned a fixed additive contribution. The early6–9 region is particularly informative for the language reversal under the tested reference substitution, while other regions modulate or counteract its effect.

Next factor early blocks into their attention and normalized-MLP operator stages. Test reference-context substitutions one stage at a time and across the early region, always propagating the actual incoming tangent. Verify that the unfactored and factored native chains agree before interpreting differences. This can distinguish attention's context routing from bilinear MLP transformation within the region without fitting a new feature direction or selecting a successful prompt subset.

Audits:72 native forwards and660 batched JVP calls; all source/manifest/vector/trace/factor/runner hashes unchanged; four reference IDs explicitly verified; exact native logits; maximum residual-chain error0.000401 and weighted logit-chain error0.000563; finite responses. JVP calls are derivative evaluations and are not counted as ordinary whole-model prompt evaluations. No new natural-text documents; FineWeb remains26,112. Broad research goal active and incomplete.
