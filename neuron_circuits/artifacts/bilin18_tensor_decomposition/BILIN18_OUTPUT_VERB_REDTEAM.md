# Matched-null red team for recurring output verbs

The bus-aware atlas found leading output directions much closer than Haar-random vectors. This post-result red team asks whether that recurrence survives three stronger controls: random directions inside each native c_proj output subspace, permutation of learned input maps across output projections, and whitening by the aggregate output-operator covariance.

## Locked hypotheses

- **R1_verbs_exceed_output_subspace_null: supported.**
- **R2_verbs_depend_on_learned_input_output_pairing: supported.**
- **R3_verbs_survive_global_output_whitening: not supported.**
- **R4_verbs_are_not_just_top_cproj_directions: supported.**

## Results

- Observed median nearest verb cosine: `0.213`.
- Native-output-subspace null median/p95: `0.076/0.080`.
- Input/output-pairing null median/p95: `0.103/0.114`.
- Whitened observed versus Haar p95: `0.102/0.077`.
- Combined-operator versus c_proj-only leading-direction alignment: median `0.727`.

Strongest unique pairs:

- L4H2 ↔ L5H7: verb `0.705`, whitened `0.350`, normalized output-subspace overlap `0.133`
- L1H1 ↔ L5H7: verb `0.641`, whitened `0.205`, normalized output-subspace overlap `0.120`
- L2H4 ↔ L5H0: verb `0.448`, whitened `0.197`, normalized output-subspace overlap `0.132`
- L1H5 ↔ L4H1: verb `0.383`, whitened `0.122`, normalized output-subspace overlap `0.147`
- L2H6 ↔ L3H6: verb `0.374`, whitened `0.086`, normalized output-subspace overlap `0.193`
- L4H7 ↔ L5H6: verb `0.349`, whitened `0.180`, normalized output-subspace overlap `0.177`
- L3H7 ↔ L4H3: verb `0.339`, whitened `0.089`, normalized output-subspace overlap `0.198`
- L0H3 ↔ L2H6: verb `0.295`, whitened `0.139`, normalized output-subspace overlap `0.159`
- L2H5 ↔ L3H8: verb `0.280`, whitened `0.107`, normalized output-subspace overlap `0.199`
- L2H0 ↔ L4H0: verb `0.275`, whitened `0.073`, normalized output-subspace overlap `0.157`
- L0H7 ↔ L1H8: verb `0.275`, whitened `0.198`, normalized output-subspace overlap `0.142`
- L1H3 ↔ L2H8: verb `0.269`, whitened `0.147`, normalized output-subspace overlap `0.169`

## Guardrails

These are post-result controls and therefore diagnose, rather than independently confirm, the original recurrence claim. Output-subspace overlap is not itself a semantic function. Whitening can erase a genuinely reused direction if global reuse is exactly what created the anisotropy; its purpose is to distinguish a discrete family from one broad shared residual geometry. Only a pair that survives matched controls should advance to finite causal replacement.
