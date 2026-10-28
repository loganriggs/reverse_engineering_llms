# Preregistration: Shapley split of answer verb and semantic residual

For each pure-rule transition and layer 1–3, use the discovery-Fisher-selected
scalar from the preceding assay to define

`A = scale × generic_answer_delta`

and

`R = semantic_delta − A`.

Thus `A + R` is exactly the original semantic activation patch. On held-out
identities/template 6, evaluate the four exact finite coalitions: neither,
answer only, residual only, and both. Compute vector Shapley effects for `A` and
`R`, their signed Fisher projections onto the complete semantic-patch effect,
absolute projection shares, and the non-additive interaction norm.

This coalition game is conditional on the chosen answer ray and scale. It asks
whether the remaining activation correction is causally important; it does not
claim a unique allocation of the underlying rule computation. Exact vector
recomposition is a mandatory implementation check.
