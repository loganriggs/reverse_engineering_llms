# Preregistration: state contraction of route and full sensitivity operators

The prior red-team found high rank-16 route/full projector overlap but imperfect
finite agreement when removing the current state's projection.  Freeze those
examples and ask whether the missing object is the contraction of the operator
with the current residual state.

For each route and full suffix operator, measure:

- sign-aligned leading eigenvector cosine and exact finite effect at 1% and 5%;
- the raw active vector `P x`, its norm fraction, route/full cosine, and norm
  ratio;
- exact effects of literal `-P x` removal with no norm matching;
- exact effects after separately normalizing those directions to 1% and 5%;
- literal-removal KL for 32 seeded random rank-16 subspaces.

The projector is not declared sufficient if its action on the current state is
unstable.  Conversely, a leading direction that transfers while `P x` does not
would distinguish stable potential sensitivity from the state-selected active
component.  Literal random controls test natural component use; normalized
controls test local causal sensitivity.  These are reported separately.
