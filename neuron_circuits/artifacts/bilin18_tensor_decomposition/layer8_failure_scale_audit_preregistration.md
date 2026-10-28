# Preregistration: scale and support audit of blind-replication failures

Freeze the twelve `<0.8` finite-effect failures from the blind FineWeb/TinyStories
replication.  For every failure, choose without replacement the passing triple at
the same site with the nearest cost-whitened Gram error.  Do not search new sites
or redefine the failure label.

Recompute the exact triple response and its complete singleton/pair Mobius
reconstruction.  Independently select the same 16-mode greedy programs as
before, then distinguish four objects that the previous aggregate conflated:

1. reconstruction error on a *fixed exact-selected support*;
2. reconstruction error on a *fixed reconstructed-selected support*;
3. the exact infinitesimal effects of the two independently selected supports;
4. their finite full-model effects at input edit norms 0.1%, 0.25%, 0.5%, 1%,
   2%, and 5%.

For each support and scale, also compare its finite effect with the exact analytic
tangent for that same support.  This checks the analytic implementation and tells
apart local program disagreement from finite nonlinear divergence.

Locked interpretations:

- fixed-support agreement plus support-to-support disagreement means the response
  reconstruction is sound but sparse program selection is non-identifiable or
  discontinuous;
- agreement at 0.1% followed by deterioration toward 5% implicates finite
  nonlinear propagation;
- failure of a support's own tangent at 0.1% triggers a numerical/analytic bug
  investigation before any mechanistic conclusion;
- matched controls test whether any effect is merely an expected consequence of
  comparing independently selected supports.

No thresholds, scales, controls, or hypotheses will be changed after inspecting
the new scale results.

## Implementation correction recorded after the first run

The first implementation incorrectly used the three-layer route-masked tangent
for item 3 and for the own-tangent finite check.  Finite interventions run the
complete native suffix, so those comparisons require the full suffix analytic
Jacobian.  The route-masked tangent remains correct for items 1 and 2: defining
and reconstructing the explanatory route used to select a program.  This was
found because several passing controls had excellent between-program agreement
but poor own-tangent fidelity.  Results were regenerated with the full Jacobian;
no locked thresholds, scales, examples, controls, or hypotheses were changed.
