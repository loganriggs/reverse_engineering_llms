# Finite native test of the L8H2-cue to L9H8-boundary circuit

The actual L8H2 cue-edge write is replaced by its counterfactual write inside the original prefix, after which the Layer-8 MLP and every later block are recomputed. At Layer 9, the proposed downstream edge is either left native or frozen to its value before the upstream edit. Matched upstream head/source and downstream head/source controls receive the same treatment.

## Locked results

- **C0_native_branch_recomposes: supported.**
- **C1_native_L8H2_cue_edge_transfers_style: supported.**
- **C2_upstream_effect_is_specific: supported.**
- **C3_L9H8_boundary_change_mediates: supported.**
- **C4_downstream_mediation_is_specific: supported.**
- **C5_declared_contrast_beats_random: supported.**
- **C6_native_two_hop_circuit_is_supported: supported.**

| Fold/split | native win/mean | frozen mean | mediated fraction | random percentiles native/mediated | control mediation H7/nonboundary |
|---|---:|---:|---:|---:|---:|
| 0/0 | 0.867/+0.07105 | +0.03021 | 0.575 | 1.000/1.000 | 0.000/0.000 |
| 0/1 | 0.844/+0.07159 | +0.03019 | 0.578 | 1.000/1.000 | 0.000/0.000 |
| 1/0 | 0.906/+0.08215 | +0.03999 | 0.513 | 1.000/1.000 | 0.000/0.000 |
| 1/1 | 0.867/+0.06522 | +0.02955 | 0.547 | 1.000/1.000 | 0.000/0.000 |
| 2/0 | 0.891/+0.08347 | +0.03875 | 0.536 | 1.000/1.000 | 0.001/0.000 |
| 2/1 | 0.906/+0.09762 | +0.04546 | 0.534 | 1.000/1.000 | -0.000/0.000 |

## Interpretation boundary

Replacing one upstream edge and freezing one downstream edge establishes a finite path-specific dependency. It does not imply that either complete head is monosemantic, nor that this route explains every contextual consequence of the spelling cue. The counterfactual upstream write comes from a full swapped-prefix execution, so the edge packages both changed routing and changed value; a subsequent factorization is needed to distinguish lexical payload from route selection.
