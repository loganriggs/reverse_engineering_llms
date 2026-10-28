# M33: native context sensitivity persists near the intact computation

2026-10-28. Follow-up to [M32](NATIVE_CONTEXT_RESPONSE_GEOMETRY.md), with frozen
intervention directions and no featurizer fitting or amplitude search.

Context-dependent native response directions persist under small symmetric interventions. Of 627 active direction/context cases, 626 pass the fixed two-scale agreement threshold; median relative scale discrepancy is 0.37%. At amplitude .01, centered cross-context change/reference energy is .122–.314 and 55–86% of that change cannot be explained by scalar gain. Full-removal curvature is therefore insufficient to explain the aggregate context dependence.

## Why this test

A large component removal can produce effects unlike a small change to that component.
M32 alone could not distinguish local sensitivity from finite-removal curvature. We now apply
symmetric perturbations to the same MLP1 task suffix at amplitudes .1 and .01. Positive amplitude
follows the removal direction: write minus h times delta. Negative amplitude adds it.
The central difference is [logits(+h) - logits(-h)]/(2h), centered using the reference intact
next-token probability weights. This measures finite-difference local sensitivity, not an exact
Jacobian. No featurizer parameters, intervention groups, prompts or task exclusions change.

## Coverage and numerical checks

All 720 scheduled direction/context rows remain, including 93 exact zero interventions. All
218 intact final-token vectors exactly replay M32. There were 2,726 native forward passes.
The 627 active cases all pass the preset output-storage resolution diagnostic (minimum ratio
22.68 versus threshold 10). This compares the symmetric output difference with float32 output
spacing; it is NOT a bound on all internal numerical error.

The preset relative weighted-norm discrepancy threshold between scales is 10%. It passes in
626/627 active cases; median discrepancy is .003662. One entity_binding/block72/passage case
(target 470) has discrepancy .103729 and remains inconclusive, rather than triggering a new
amplitude. Its local response energy is 1.74e-5. Zero cases are not counted as successful local
convergence. Results include all cases and the predefined jointly usable reference/context subset.
Two-scale consistency provides evidence of local behavior, not a proof of convergence as h→0.

## Cross-context effects at small and full amplitude

Each entry is the ratio of summed squared response differences to summed reference response
energy. It is not a percent error rate or fraction of explained behavior. Generator names denote
the frozen scalar/block source of intervention directions; the native model is identical.
The table retains all rows. Removing the sole nonconverged paired case changes its aggregate
by less than .00001, leaving the conclusion intact.

| Generator | Context | Small (.01) change/reference | Full removal change/reference | Small orthogonal fraction |
|---|---|---:|---:|---:|
| scalar 71 | passage | 0.215 | 0.204 | 85.8% |
| scalar 71 | sample | 0.122 | 0.113 | 76.9% |
| scalar 72 | passage | 0.216 | 0.240 | 64.2% |
| scalar 72 | sample | 0.184 | 0.196 | 55.0% |
| block 71 | passage | 0.150 | 0.131 | 64.0% |
| block 71 | sample | 0.122 | 0.103 | 81.3% |
| block 72 | passage | 0.314 | 0.265 | 73.6% |
| block 72 | sample | 0.215 | 0.177 | 72.8% |

Here orthogonal fraction is the change that cannot be explained by projecting the context
response onto the reference response and rescaling it. Median small-response cosines remain
.846–.947: the responses remain substantially aligned, while their differences are real and
not generally just gain changes. This is not evidence that the entire response is orthogonal.

## Research decision

Close the wrapper/amplitude diagnostic. A component's native downstream causal role is locally
context dependent on this development panel. It is inappropriate to impose an identical final
output response across contexts as a requirement for a meaningful feature. Conditional reuse
remains plausible, not proved. The existing native-response supervision already targets the
native effect in its own context; this result supports that target, not a newly invented loss.
It also does not explain all student fidelity failures or show blocks outperform scalar features.

Next return to the joint scalar/block/shared-dictionary comparison. First reconcile the existing
shared-dictionary implementation and its synthetic recovery audits with the current native-write
training setup; do not introduce a new architecture until that inventory is checked. The concrete
deliverable is one common, capacity-accounted comparison protocol and verified union decoding,
using diverse competent tasks and context-specific native causal targets. Separate ordinary
reconstruction/output fidelity, causal fidelity, reusable structure and task discovery. Stop the
architecture audit when those interface/budget/split checks are satisfied, then execute the joint
comparison rather than doing more wrapper or optimizer variants.

All examples remain development data. Preserve whole-task training exclusions sequence_copy,
spatial_inverse, weekday_successor, role_binding and tense; past_tense is separately allowed.
Wrapper meaning and position remain confounded. Neither semantic manifolds, new tasks nor
unseen-combination transfer have been established by this diagnostic. The user's capacity/noise
toy questions remain distinct from this native local-sensitivity result.

## Reproduction and audit

- [Frozen protocol](validation/native_local_protocol.json)
- [Run inventory and vector-cache hash](validation/native_local_inventory.json)
- [All row, pair and aggregate results](validation/native_local_results.json)
- [Independent audit](validation/native_local_audit.json)

Run probe_native_local_sensitivity.py, analyze_native_local_sensitivity.py and
 audit_native_local_sensitivity.py in the established single-thread environment. The probe
requires the pinned model, existing M32 local vector cache and expansion data, refuses to
overwrite completed results, and saves progress every 30 rows. The new full-vector cache remains
local at the path recorded in the inventory; it is not bundled in the small shareable packet.

NumPy analysis was independently rescored with Torch: 720 rows, 1,440 paired geometries,
48 aggregates, all 218 intact baselines, all delta hashes and full-response agreement with M32.
Maximum absolute metric discrepancy is 9.46e-11 (including large resolution ratios). The
transformer is not independently implemented by this audit.
