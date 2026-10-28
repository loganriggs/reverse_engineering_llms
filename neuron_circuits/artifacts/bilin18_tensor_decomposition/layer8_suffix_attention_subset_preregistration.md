# Preregistration: which later attention blocks select the Layer-8 program?

## Question

The direct tensor Jacobian propagates Layer-8 mode tangents through residual,
RMSNorm, and later bilinear MLP derivatives but clamps all later attention
derivatives. It identifies substantially worse programs than the exact full
tensor Jacobian, especially out of distribution. Is the missing correction a
small reusable routing module, or a distributed suffix computation?

## Frozen design

- Source layer: 8. Candidate attention-JVP blocks: 9–17.
- Native forward states and attention writes are always identical. A mask only
  controls which attention *derivatives* carry the Layer-8 perturbation.
- The candidate program contains 16 Layer-8 tensor modes.
- Eight FineWeb discovery sites are selected by the pre-existing tensor query
  procedure. All 130 masks of cardinality zero through three are evaluated.
- For each cardinality, choose the mask maximizing median objective efficiency
  under the exact full-tensor Fisher response. Ties are broken lexicographically.
- Freeze those masks. Evaluate on disjoint FineWeb rows and on WikiText, using
  eight query-selected sites per corpus. Evaluate a cumulative broad curve by
  greedily adding remaining layers on discovery data only.
- The primary causal endpoint is Fisher fidelity between the exact finite
  5%-norm intervention selected by the masked response and the intervention
  selected by the full response. Secondary endpoints are full-response
  objective efficiency and support recall.

## Locked interpretations

**Compact reusable correction.** The best frozen size-3 mask reaches at least
0.90 median finite-effect fidelity and objective efficiency on both corpora.

**Broad suffix routing.** No size-at-most-3 mask reaches 0.80 fidelity on both
corpora, the full mask reaches at least 0.98, and the frozen cumulative curve
needs at least six layers to cross 0.90 on either corpus.

An intermediate outcome supports neither sharp version and will be retained as
such rather than relabeled.

## Negative-result red team

1. The empty mask must match `include_attention=False`; the nine-layer mask
   must match `include_attention=True`, elementwise within floating tolerance.
2. Native logits must be identical for every derivative mask.
3. Exhaustive size-3 search prevents a false negative from greedy synergy.
4. Discovery and evaluation rows are disjoint; WikiText is an OOD corpus.
5. Eight seeded size-matched random masks calibrate selection advantage.
6. A synthetic 10% interpolation between direct and full tangents must be
   detectable and lie below the full endpoint, guarding against an insensitive
   response metric.
7. Full finite forward interventions, not only tangent similarity, determine
   the primary result.
8. Bootstrap intervals over sites will be shown. With only eight sites per
   corpus, failure with a wide interval is inconclusive, not falsification.

The result localizes a derivative route. It is not a percentage allocation of
stored information to attention blocks.
