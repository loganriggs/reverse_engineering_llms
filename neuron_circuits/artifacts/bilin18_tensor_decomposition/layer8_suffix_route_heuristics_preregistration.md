# Preregistration: can simple layer evidence select the local route?

The exhaustive atlas shows that a three-layer suffix route usually exists but
is not stable by tensor query. This test asks whether its changing identity has
a short weight-derived rule.

For every frozen atlas site, compute three size-3 masks:

1. **Singleton utility:** rank layers by the objective, under the full Fisher
   Gram, of the program selected when only that layer's attention JVP is active.
2. **Leaveout necessity:** rank layers by the Frobenius change between the full
   Gram and the Gram with that layer's attention JVP removed.
3. **Self-contained energy:** rank layers by the Frobenius change between the
   singleton-layer Gram and the direct/no-attention Gram. This selector never
   uses the full Gram as its scoring target.

The chosen masks already have exact finite effects in the exhaustive atlas, so
no mask is reevaluated or selected by its finite answer here.

A selector passes if median full-reference fidelity is at least 0.95 and its
10th percentile is at least 0.80 on both corpora. Strong synergy is supported if
all three fail and the exhaustive weight-objective selector improves the 10th
percentile by at least 0.10 on either corpus.

The first two methods use the full suffix diagnostically and are not compressed
interpreters. The third is cheaper and self-contained. Failure of these three
heuristics does not prove no learnable router exists.
