# Upstream head chain of the singular ordered path

Layer-14 head 6 and its QK2/value-bus factor are held fixed. The path is decomposed by layer-13 head, then by layer-11 head after choosing the dominant failure component.

## Locked hypotheses

- **C0_nested_head_decompositions_recompose: supported.**
- **C1_layer13_is_head_localized: supported.**
- **C2_layer11_is_head_localized: not supported.**
- **C3_full_ordered_head_chain_is_context_amplified: supported.**

Selected ordered head chain: `[4, 3, 6]` for layers `[11, 13, 14]`. Maximum recomposition error: `1.693e-05`. Fixed-chain failure/control amplification: `20.2x`.

| Stage | Top head/share | Effective support | Control top head/share |
|---|---:|---:|---:|
| Layer 13 | 3/0.640 | 2.18 | 3/0.497 |
| Layer 11 | 4/0.319 | 4.12 | 6/0.389 |

A localized chain is a compact exceptional circuit even though its gain is distributed. A diffuse upstream stage would instead make the final clean head the endpoint of a broad subspace computation.
