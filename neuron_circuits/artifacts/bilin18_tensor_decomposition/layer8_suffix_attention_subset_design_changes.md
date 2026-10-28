# Layer-8 suffix-attention subset design changes

## Locked pilot

The original preregistered pilot used queries 0–7 and one selected site per
query. The FineWeb-discovered mask `[9, 10, 11]` reached median full-reference
finite-effect fidelity 0.996 on FineWeb and 0.989 on WikiText. It did not beat
random triples on WikiText. The bootstrap interval on WikiText included a large
negative value, and individual-site inspection found one fidelity of -1.942.
The no-attention route also reached median 0.972 on this small WikiText panel.

The pilot artifacts are preserved as `*_pilot.*`. The result is classified as
an underpowered positive, not validation of a compact reusable module.

## Red-team expansion, locked before rerun

Keep discovery fixed at the original eight sites and do not reselect the
compact masks. Expand held-out evaluation to all 16 pre-existing queries and
two selected sites per query, exactly matching the panel in
`layer8_full_tensor_jacobian.json`. Add a second independent reference: the
per-example backprop-Fisher oracle indices frozen in that prior artifact.

The expanded result must report, for every mask:

1. fidelity to the exact finite effect selected by the full tensor Jacobian;
2. fidelity to the exact finite effect selected by the independent backprop
   oracle;
3. site-level intervals and lower-tail failures;
4. matched random triples.

The original C1 label remains tied to its preregistered median/full-reference
criterion. A scientific claim of robust compact routing additionally requires
median oracle-effect fidelity at least 0.90 on both corpora, a 10th-percentile
full-reference fidelity at least 0.80, and an advantage over the median random
triple on both corpora. Failure of this stricter audit demotes C1 to metric-
specific sufficiency rather than falsifying the existence of useful compact
routes.
