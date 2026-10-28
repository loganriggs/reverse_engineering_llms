# Preregistration: role-separated final-MLP tangent quotient

## Question

The prior symmetric tensor quotient used one state-PCA basis on both input
indices. Its tensor factorization was exact inside that quotient but the
quotient retained little native behavior. This experiment distinguishes two
explanations:

1. the final-MLP backward computation is intrinsically broad; or
2. the quotient erased the computation by conflating three statistically and
   computationally distinct roles: natural context, incoming tangent, and
   outgoing write.

For a natural normalized state `x` and an incoming tangent `d`, the bilinear
MLP derivative is

```text
J_f(x)d = Down[(Left d) * (Right x) + (Left x) * (Right d)].
```

The test fits separate uncentered PCA bases to `x`, `d`, and `J_f(x)d`. The
bases are learned on the first 192 FineWeb discovery sites and frozen before
using the final 64 discovery sites, disjoint FineWeb, WikiText, or TinyStories.

## Comparisons

- output-write projection only;
- context projection only;
- tangent projection only;
- the three correctly assigned role bases together;
- the previous shared-state construction, with one state basis on both input
  roles and a PCA basis of endpoint adjoints on output;
- context and tangent bases deliberately swapped;
- eight matched triples of isotropic random rank-64 bases.

At rank 64 the role-separated factors are also contracted into an explicit
`output × tangent × context` tensor. Its direct execution must match the
projected native CP factors before any compression result is interpreted.

## Interpretation boundaries

Success would establish a compact chart on the observed tangent bundle, not a
globally low-rank final MLP and not a semantic explanation. The tangent and
natural-write bases use causal-path observations and are therefore oracle
mechanism-discovery objects. They must transfer to disjoint and OOD sites, and
a later experiment must ask whether weights or cheaper observables recover
them.

Failure would be evidence against this particular global rank-64/128
role-separated chart. It would not establish intrinsic breadth until checking
local or routed charts, because a single basis can still average over
context-dependent low-rank mechanisms.

The locked thresholds and all paths are in
`configs/bilin18_layer17_tangent_bundle_quotient.json`.
