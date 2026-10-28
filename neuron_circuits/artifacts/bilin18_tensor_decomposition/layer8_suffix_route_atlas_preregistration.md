# Preregistration: reusable suffix route or local route atlas?

The global FineWeb-discovered suffix mask `[9, 10, 11]` has excellent median
fidelity but severe lower-tail failures on the expanded 32-site panels. This
experiment exhausts all 84 three-block subsets of layers 9–17 at every site.

Three levels are separated:

1. **Global route:** the already frozen `[9, 10, 11]` mask.
2. **Query-frozen route:** for each of 16 tensor queries, choose the best mask
   on its first FineWeb site, freeze it, and evaluate on the second FineWeb site
   and both WikiText sites.
3. **Sitewise ceiling:** choose the best mask after seeing that site's effect.
   This is explicitly a hindsight existence bound, not an interpreter.

Every mask selects its 16-mode Layer-8 program from its exact weight-derived
masked Fisher Gram. Candidate finite 5%-norm interventions are recomputed
through the full model. The primary reference is the finite program selected by
the full nine-layer tensor Jacobian; the prior backprop-Fisher oracle is a
secondary independent reference.

## Locked outcomes

- Sitewise compact route: median fidelity >=0.98 and 10th percentile >=0.90 on
  both corpora.
- Query-recurrent route: median >=0.95 and 10th percentile >=0.80 on held-out
  FineWeb sites and WikiText.
- Local route atlas: sitewise criterion passes but query-recurrent criterion
  fails.
- Broad-required tail: hindsight size-3 10th percentile <0.80 anywhere.

## Red-team controls

- All 84 masks are evaluated, eliminating greedy subset-search failure.
- Query masks use one FineWeb site only; no WikiText selection is allowed.
- Results from hindsight finite selection and realizable weight-objective
  selection are reported separately.
- Native logits, full-mask reference indices, and prior oracle indices are
  checked against frozen artifacts.
- Median, 10th percentile, minimum, per-query failures, and bootstrap intervals
  are retained. A high median cannot conceal catastrophic route failures.
- Hindsight success cannot establish a usable router; it only establishes that
  a compact route exists at each state.
