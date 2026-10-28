## H592 — The context-conditioned linear term is causally sufficient locally, but quadratic response matters at larger dose

The previous goal turn made progress by identifying MLP11's contextual and incoming-response dependence. Now intervene on its exact bilinear expansion rather than infer importance from component norms. At the same 192 prompts and +/-0.0625, +/-0.5 discovery-scaled Attention5 perturbations, replace MLP11's response around its native output by the linear-in-input-change term, the quadratic term, their sum, or zero. Keep all other computation live. Total 3,072 interventions; cached normalized inputs and native writes from H591 are reused.

For B(z)=Down[Left(z)Right(z)]+bias and normalized-input change e, linear response is Down[Left(z)Right(e)+Left(e)Right(z)], and quadratic response is Down[Left(e)Right(e)]. The linear term still depends on native context z, and e can be a nonlinear function of the earlier Attention5 perturbation. This is not a claim that the whole MLP or network is globally linear.

Compare each approximation with the sum/full response. Normalize mean absolute margin error by the sum-versus-zero-response error, and reference KL by KL(sum || zero). This denominator isolates the MLP11 response contribution; it is different from normalizing by the entire network perturbation. Retain intended-answer-group NLL error separately. Central averages and per-sign errors are both saved, so cancellation across signs does not define fidelity.

| Type/frame | Small-dose linear margin error / MLP contribution | Small-dose relative KL | Half-dose linear margin error / MLP contribution | Half-dose relative KL |
|---|---:|---:|---:|---:|
|regular/preposition|0.0450|0.00278|0.3725|0.22161|
|regular/relative_clause|0.0623|0.00500|0.4262|0.19643|
|regular/fronted_preposition|0.0310|0.00078|0.2872|0.11014|
|irregular/preposition|0.0446|0.00209|0.3575|0.16501|
|irregular/relative_clause|0.0567|0.00288|0.3936|0.20418|
|irregular/fronted_preposition|0.0323|0.00076|0.2873|0.09236|

At small dose, dropping the quadratic term gives only 3.1–6.2% of the error caused by removing the entire MLP response, and 0.076–0.500% of its reference KL. Mean group-NLL error is 1.26e-5–7.24e-5 nats. Quadratic-only response is nearly indistinguishable from the zero-response clamp in this local regime. This is causal support for the contextual linear approximation, rather than a conclusion based solely on norm.

At half dose, omitting the quadratic term produces 28.7–42.6% relative margin error and 9.2–22.2% relative KL, with group-NLL errors 0.00135–0.00766 nats. Its response norm was only about 10% on average in H591, but that did not guarantee small behavioral error. The linear approximation therefore has a material dose limit. Neither term's behavioral effect is an independent additive share once later nonlinear computation runs.

Example: regular fronted small-dose central response is 0.6522 for the sum, 0.6516 for linear-only, and 0.8421 for zero. At half dose it is 0.7493 for the sum, 0.7164 for linear-only, and 0.8711 for zero. Mean central responses alone understate some per-sign fidelity errors, which are reported separately.

Numerical correction: the first metric pass used float32 log-softmax and tiny KL comparisons reached its cancellation floor, including a small negative estimate. Recomputed KL and group-NLL in float64 from the unchanged float32 model logits and reran the same interventions. All reported KL values are nonnegative. This changes metric precision, not prompts, bases, doses or hypotheses.

Next test whether the context-conditioned action itself has a compact reusable form. The response change under a context switch dz at fixed e is exactly Down[Left(dz)Right(e)+Left(e)Right(dz)]; its quadratic-in-e term cancels. Fit context and incoming-response subspaces on regular discovery items only, then test rank-one and larger bilinear interactions on irregular nouns and relative-clause contexts. Compare the live full-context-switch intervention, not just vector norms, and include norm-matched random controls. A successful factorization would identify a candidate context-by-number computation; failure would show that the contextual operator requires richer structure. Actual donor coefficients and already-explored versus fresh validation must remain explicit.

Audits: sum replay against H591 <=1.44e-5 margin; zero response replay against H590 <=1.91e-5; bilinear reconstruction max relative error 2.33e-4 from float32 subtraction; finite metrics and unchanged source/basis. No new prompts or natural-text documents; FineWeb remains 26,112. Broad goal active and incomplete.
