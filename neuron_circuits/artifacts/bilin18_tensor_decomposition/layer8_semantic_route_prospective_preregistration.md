# Preregistration: prospective semantic-route transfer

Freeze Layer-4 64-edge routes using only the original nineteen lexical pairs and
their already designated FineWeb discovery contexts. Before evaluating any new
pair, declare seventeen new one-token substitutions across the same five
families. Each source token has enough occurrences for three independent
fixed-position contexts in both FineWeb and TinyStories.

For each new example, compare:

- the route learned from all old pairs in its declared family;
- each of the four wrong-family routes;
- a route learned from all old pairs;
- eight cyclic head shifts of the correct route;
- the per-example oracle-64 route.

All routes preserve exact edge addresses and are evaluated by finite removal at
Layer 4 with MLP and suffix recomputation. The primary metric is signed Fisher
projection onto the complete natural counterfactual effect.

This is a prospective falsifier of semantic-route reuse. Correct-family failure
with global success favors a generic broadcast verb. Failure of both frozen
routes favors context-local routing. Success requires both a meaningful effect
gap and broad example-level wins; post-hoc pair removal or relabeling is not
allowed.
