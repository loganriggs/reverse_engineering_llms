# Preregistration: semantic-family versus generic attention routes

Use the existing five hand-declared lexical families, fixed position-47 edits,
position-63 predictions, and Layer 4 selected by the prior frozen-route study.
For every held-out lexical pair, use only the second FineWeb context for route
discovery and evaluate the first, independently selected context on FineWeb and
TinyStories.

Rank exact attention edges by mean absolute local projection and freeze 64-edge
routes from four discovery pools:

1. other lexical pairs in the same semantic family;
2. all pairs outside that family;
3. all pairs except the held-out pair;
4. the held-out pair's own disjoint discovery context (lexical upper bound).

Compare these with a per-example oracle and cyclic head-shift controls that keep
source and destination positions fixed.  The primary finite statistic is signed
removal projection onto the complete natural counterfactual output effect after
recomputing the Layer-4 MLP and full suffix.

A same-family advantage is evidence for a reusable family-level transport
operator, not proof that the route stores the concept.  Equal family,
outside-family, and global performance supports a generic broadcast verb whose
content is supplied by values/state.  A pair-specific advantage supports lexical
address reuse.  Oracle superiority means frozen routes remain incomplete.
