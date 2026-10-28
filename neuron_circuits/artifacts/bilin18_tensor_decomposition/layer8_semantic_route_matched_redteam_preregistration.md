# Preregistration: data-matched red-team of semantic-family routes

The leave-one-pair-out family route used only two or three discovery examples,
whereas the outside-family route used about fifteen. The family advantage was
0.054 on FineWeb and 0.035 on TinyStories, narrowly missing the locked 0.05 OOD
threshold. Freeze the same pairs, contexts, layer, route size, and causal metric.

For each held-out pair, draw sixteen seeded outside-family discovery subsets
with exactly the same number of examples as its same-family discovery pool.
Learn a 64-edge route from each subset and evaluate all routes on the same frozen
FineWeb and TinyStories contexts. Compare the same-family route with the
within-pair median matched-outside effect; report pair win rate and effects by
family. Retain cyclic head shifts of the family route.

This red-team can rescue only a modest family-level route claim. It cannot make
the human categories correct, prove that the route represents the concept, or
erase the already observed oracle gap. Failure after data matching is stronger
evidence for generic transport; success motivates replication with more pairs.
