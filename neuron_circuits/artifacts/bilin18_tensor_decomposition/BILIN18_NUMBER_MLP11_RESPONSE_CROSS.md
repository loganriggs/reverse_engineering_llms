## H591 — MLP11's response depends jointly on contextual state and incoming perturbation

Following H590's discovery-selected suppressor, capture MLP11's normalized native input z and the input change e induced by the same signed Attention5 perturbation. The model's ungated bilinear MLP obeys exactly:

B(z+e)-B(z) = Down[Left(z) Right(e) + Left(e) Right(z)] + Down[Left(e) Right(e)].

The first term is linear in e but depends on native context z; the second is quadratic in e. The output bias cancels. The identity is applied after input normalization, so it does not separately explain the effect of normalization upstream.

For each recipient, independently choose its own or matched unfronted z and its own or matched unfronted e. Replace only MLP11's response around the recipient's native output, keeping the Attention5 perturbation and other computation live. These are counterfactual post-normalization combinations, not asserted natural states. Four crossings at two doses and both signs over 192 prompts give 3,072 intervention forwards; native/perturbed capture uses 960 forwards. Unfronted recipient cells give exact duplicate controls.

| Type/frame, small dose | Own context/input | Unfronted context only | Unfronted input response only | Both unfronted |
|---|---:|---:|---:|---:|
|regular/relative_clause|1.2251|1.1588|1.3181|1.2527|
|regular/fronted_preposition|0.6522|0.7065|0.6652|0.7765|
|irregular/relative_clause|1.0620|1.0616|1.1955|1.1803|
|irregular/fronted_preposition|0.6711|0.7290|0.6791|0.8074|

In fronted prompts, unfronted context alone increases central response modestly; unfronted incoming response alone changes it less. Replacing both has a larger effect than either alone: irregular 0.671 native, 0.729 context-only, 0.679 incoming-only, 0.807 both. The factorial interaction is not assignable solely to the local bilinear term, because later computation is nonlinear too. The same qualitative pattern persists at half dose.

Relative clauses differ. On irregular nouns, native 1.062 becomes 1.062 with context-only replacement, 1.196 with incoming-only replacement, and 1.180 with both. Thus the explanation cannot be a universal fixed MLP11 suppression gate: different frames change the contextual operator and incoming response in different proportions. This also does not explain all downstream suppression—H590's selected operations accounted for only part of the frame gap.

The quadratic term's relative norm averages 0.0129 at small dose (maximum 0.0501), and 0.1029 at half dose (maximum 0.4349). This makes a context-conditioned linear approximation plausible locally, but does not prove the quadratic term is causally negligible. Earlier experiments already showed that norm size and behavioral influence can rank components differently. Small perturbation nonlinearity also does not eliminate the bilinear dependence between native context and response direction.

This is a more concrete candidate computational description than “a number neuron”: a shared number-associated direction is transformed by a context-conditioned local operator, with incoming state and later readout jointly determining its effect. It remains a local mechanistic account on a narrow agreement panel, not a complete circuit or general feature ontology.

Next causally separate the linear-in-response and quadratic terms: replace MLP11's response with each term alone, their sum, or zero at both doses; retain all frames and lexical types. Compare actual output shifts, KL and answer-group loss instead of inferring importance from norm. Then use the surviving causal term to identify which context directions or upstream writes alter its action. Do not treat successful compression or an algebraic identity alone as the explanation.

Audits: exact-form response matches direct MLP output differences within 2.33e-4 relative error under float32 subtraction; native response replacement replays H588 within 1.53e-5 margin; unfronted duplicate crossings match exactly; finite metrics and unchanged source/basis. No new prompts or natural-text documents; FineWeb total 26,112. Broad goal active and incomplete.
