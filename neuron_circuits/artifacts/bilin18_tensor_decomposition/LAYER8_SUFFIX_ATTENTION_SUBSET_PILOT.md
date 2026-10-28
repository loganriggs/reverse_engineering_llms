# Layer-8 suffix attention-subset localization

The native forward pass is fixed. Only selected attention derivatives in blocks 9–17 are allowed to carry a Layer-8 perturbation. All subsets of up to three blocks were searched on separate FineWeb discovery rows; the masks and wider cumulative order were then frozen before FineWeb and WikiText evaluation.

## Locked hypotheses

- **C0_endpoints_reproduce: supported.**
- **C1_compact_reusable_router: supported.**
- **C2_broad_suffix_required: not supported.**
- **C3_selection_transfers_ood: not supported.**
- **C4_attention_signal_is_detectable: supported.**

Best size-3 discovery mask: `[9, 10, 11]`. Frozen order: `[9, 10, 11, 16, 12, 17, 13, 14, 15]`.

![Suffix attention subset curve](layer8_suffix_attention_subset.png)

| Corpus | Layers | Effect fidelity (95% bootstrap interval) | Objective efficiency | Support recall |
|---|---:|---:|---:|---:|
| fineweb | 0 | 0.215 [-1.447, 0.980] | 0.764 | 0.281 |
| fineweb | 1 | 0.984 [0.468, 0.989] | 0.943 | 0.750 |
| fineweb | 2 | 0.980 [-1.558, 0.986] | 0.970 | 0.750 |
| fineweb | 3 | 0.996 [0.993, 0.999] | 0.993 | 0.875 |
| fineweb | 4 | 0.999 [0.995, 1.000] | 0.996 | 0.906 |
| fineweb | 5 | 0.999 [0.995, 1.000] | 0.997 | 0.938 |
| fineweb | 6 | 0.999 [0.993, 1.000] | 0.998 | 0.938 |
| fineweb | 7 | 1.000 [0.998, 1.000] | 1.000 | 0.969 |
| fineweb | 8 | 1.000 [0.998, 1.000] | 1.000 | 1.000 |
| fineweb | 9 | 1.000 [1.000, 1.000] | 1.000 | 1.000 |
| wikitext | 0 | 0.972 [0.887, 0.985] | 0.958 | 0.719 |
| wikitext | 1 | 0.974 [0.877, 0.992] | 0.980 | 0.750 |
| wikitext | 2 | 0.989 [0.865, 0.997] | 0.990 | 0.812 |
| wikitext | 3 | 0.989 [0.793, 0.997] | 0.999 | 0.812 |
| wikitext | 4 | 0.989 [0.822, 0.997] | 0.998 | 0.875 |
| wikitext | 5 | 0.993 [0.980, 0.998] | 0.998 | 0.875 |
| wikitext | 6 | 0.996 [0.980, 1.000] | 0.998 | 0.875 |
| wikitext | 7 | 0.994 [0.984, 1.000] | 0.999 | 0.875 |
| wikitext | 8 | 0.996 [0.991, 1.000] | 1.000 | 0.938 |
| wikitext | 9 | 1.000 [1.000, 1.000] | 1.000 | 1.000 |

## Interpretation guardrails

This is a derivative-route localization, not an allocation of stored information. A compact mask would show that a few later attention Jacobians are sufficient to choose the same Layer-8 intervention program. A broad curve would instead favor a distributed suffix route. Neither result alone establishes whether the selected modes have semantic meanings.

The empty/full endpoint checks, exhaustive compact search, held-out corpora, random masks, weak-signal control, exact finite interventions, and site bootstrap intervals are retained in the JSON output.
