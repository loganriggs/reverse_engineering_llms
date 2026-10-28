# Does L17H2 read a literal low-rank spelling echo?

An uncentered value-delta basis is fit only on fold 0/split 0. On all cells, native H2 post-boundary value changes are projected into ranks 1, 4, and 16 while native routing is retained, then the final MLP and output are recomputed. Eight frozen random subspaces per rank are controls.

## Locked results

- **V0_discovery_basis_is_frozen: supported.**
- **V1_rank4_is_a_literal_value_echo: supported.**
- **V2_rank4_is_causally_sufficient: supported.**
- **V3_rank4_beats_random_subspaces: supported.**
- **V4_rank1_is_nearly_sufficient: supported.**
- **V5_routing_alone_is_incomplete: supported.**
- **V6_literal_low_rank_echo_ontology: supported.**

| Fold/split | raw value R1/R4/R16 | causal value R1/R4/R16 | routing/full | best random R4 |
|---|---:|---:|---:|---:|
| 0/0 | 0.828/0.893/0.943 | 1.000/1.000/1.000 | -0.108 | 0.176 |
| 0/1 | 0.731/0.840/0.896 | 0.999/1.000/1.000 | -0.172 | 0.171 |
| 1/0 | 0.791/0.855/0.906 | 0.999/1.000/1.000 | -0.101 | 0.180 |
| 1/1 | 0.757/0.861/0.907 | 0.999/1.000/1.000 | -0.210 | 0.182 |
| 2/0 | 0.751/0.864/0.912 | 0.999/1.000/1.000 | -0.130 | 0.192 |
| 2/1 | 0.790/0.865/0.911 | 0.999/1.000/1.000 | -0.128 | 0.188 |

## Interpretation boundary

High raw and causal fidelity would support a literal repeated value direction. Low raw but high causal fidelity would instead identify a compact output-relevant quotient. Low fidelity on both would mean that the sparse H2 read function binds genuinely different positional value transforms. The test concerns intervention-induced differences, not the full semantic content of source states.
