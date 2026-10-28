## H799 — Identical value-direction interventions have context-dependent effects

Previous turn was progress: H798 showed that native edge removal can lower intended-word confidence under repeated keys yet raise it under different keys, despite similarly negative routing. H799 removes the confound that native edge coefficients differ across contexts.

Use only the competent matching-value cells from H797. For each item/format, choose one canonical first-source mixed-value vector V in L2H5. Audit V across same/different second keys and both later queries. Inject delta*V at the second-source head output before c_proj, with delta=-.5,-.25,0,.25,.5, using exactly the same V, delta and token position within the matched item/format. These deltas are additive coefficient changes, not multiplicative scales of the native edge. Native edge coefficients and all other background states remain those of their respective prompts.

### Exact intervention matching

The native first-source V vectors are exactly equal across key/query contexts, as expected because those changed tokens occur later. Canonicalization therefore introduces no approximation. Output projection weights are shared, so the corresponding residual-stream perturbation is also identical across the compared contexts. This rules out unequal injected direction or magnitude as an explanation of the response difference. It does not make the local background state identical: native edge coefficients and other context-dependent computation can still affect how the perturbation is processed.

### Behavioral result

Mean intended-token log-probability changes in natural-log units:

| Format | Additive coefficient delta | Same keys | Different keys | Paired different-minus-same response |
|---|---:|---:|---:|---:|
| Vertical bars | +.25 | -.1006 | +.0071 | +.1077 |
| Vertical bars | +.50 | -.4317 | -.0534 | +.3783 |
| Colon/newline | +.25 | +.0007 | +.0866 | +.0860 |
| Colon/newline | +.50 | -.1164 | +.1087 | +.2250 |

At+.50, descriptive paired item-bootstrap intervals for the context difference are [.2503,.5010] and[.1401,.3164]. The same directional context difference appears when querying the first key or the second key separately, rather than arising only from pooling queries. For colon/newline+.50, same-key and different-key mean effects have opposite signs with intervals below and above zero respectively.

This establishes a context-dependent finite response for an identical injected value direction on these prompts. It does not identify a semantic gate or prove that key equality, rather than other correlated local/context states, is the directly used variable.

Negative deltas lower intended log-probability in all four context/format groups. At-.50, same/different-key changes are-.141/-.150 in vertical bars and-.208/-.273 in colon/newline. Positive and negative perturbations therefore do not act as a simple sign-reversed linear response, and a single global coefficient-to-behavior mapping would miss this behavior.

Full-vocabulary accuracy stays at its native96.9–100% across the tested conditions. Distributional effects remain measurable: at+.50, mean KL(native||perturbed) is .1217/.0267 for same/different keys in vertical bars, and .0204/.0181 in colon/newline. Similar KL magnitudes need not imply the same intended-token effect, just as unchanged top1 need not imply unchanged confidence.

### Verification and limits

728 model forwards:104 native matching-value contexts,520 intervention runs,104 independent post-projection checks. Native V invariance max absolute and relative error are exactly zero. Zero-delta logits replay exactly. Native margins match H797 within6.20e-6. Pre-projection delta injection and equivalent post-projection injection agree within1.41e-5 maximum full-vocabulary logit error. All KL values finite and nonnegative; model/input/reference/runner hashes unchanged. Original reserved test untouched; natural-document pool remains26,112.

The panel contains16 aligned vertical-bar items and10 colon/newline items. Query variants are averaged within item for bootstrap; same-key queries are duplicate prompts, not independent evidence. Intervals are descriptive without multiplicity correction. Directions differ across lexical items and formats; exact matching is within an item/format. Finite interventions may leave native activation distributions, so response differences alone do not establish manifold geometry.

### Implication for representation learning

A content vector can be exactly shared while its behavioral response changes with computational background. This supports separating the reusable content representation from the context-conditioned transformation or readout. It does not imply that shared dictionaries cannot work; it means an invariant semantic label or average output effect is an inadequate validation target. A faithful representation must preserve the relevant family of finite responses across contexts, in addition to reconstructing the vector.

Next cross the local second-source residual state with the surrounding key/query context: transplant the pre-MLP residual from same-key or different-key runs, then apply the identical value perturbation in each context. Keep each transplanted baseline and zero-intervention reference explicit. If the response follows local state, local normalization/bilinear geometry may explain much of the modulation; if it follows surrounding context, later readers or other-token paths matter. This is a diagnostic hybrid experiment, not a semantic attribution by itself. Its result can guide which conditioning variables a compact shared representation or computational DAG needs.
