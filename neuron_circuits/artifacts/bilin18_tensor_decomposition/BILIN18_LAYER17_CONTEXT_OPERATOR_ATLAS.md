# Context-conditioned tangent operator atlas

The rank-16 context coefficients contract with the output × tangent × context tensor to produce one local 16×16 operator per token site. The full hidden state is an exact control register; token lookup and coarse labels are lossy router candidates.

## Locked hypotheses

- **C0_context_tensor_recomposes_native_factors: supported.** The contracted output-by-tangent-by-context tensor matches direct projected native factors to relative error at most 1e-5.
- **C1_context_axes_have_transferable_semantics: supported.** Frozen per-axis context explanations have median AUROC at least 0.60 on FineWeb, WikiText, and TinyStories.
- **C2_context_axes_are_more_semantic_than_rotations: not supported.** Native context axes exceed the same-subspace rotation p95 median AUROC by at least 0.03 on every corpus.
- **C3_local_operators_are_low_rank: supported.** Median per-site 90%-energy rank of the 16-by-16 tangent-to-output operator is at most 8 on every corpus.
- **C4_eight_context_modes_are_sufficient: supported.** The first eight tensor-ordered context coordinates retain at least 0.90 local-operator fidelity on every corpus.
- **C5_current_token_routes_operator: supported.** A discovery-frozen exact-current-token lookup retains at least 0.50 operator fidelity on FineWeb and 0.30 on both OOD corpora, with coverage at least 0.50.
- **C6_coarse_token_routes_operator: supported.** A discovery-frozen coarse current-token lookup retains at least 0.20 operator fidelity on every corpus.
- **C7_mean_operator_is_insufficient: supported.** A single discovery-mean operator retains at most 0.80 fidelity on every corpus.
- **C8_full_state_is_an_exact_control_register: supported.** The observed context coordinates reproduce every local operator to numerical precision by construction; lexical routers are judged against this explicit upper bound.

Projected-native/tensor control error: `3.94e-07`.

## Operator and router summary

| Corpus | semantic axes / rotation p95 | local R90 / effective rank | context k=4/8 | mean operator | token operator / coverage | coarse operator |
|---|---:|---:|---:|---:|---:|---:|
| fineweb | 0.946/0.957 | 3.0/2.16 | 0.970/0.993 | 0.711 | 0.846/0.674 | 0.790 |
| wikitext | 0.951/0.945 | 3.0/2.16 | 0.969/0.994 | 0.721 | 0.808/0.571 | 0.798 |
| tinystories | 0.984/0.983 | 3.0/2.12 | 0.974/0.993 | 0.736 | 0.893/0.738 | 0.809 |

## Context-mode examples

| Mode | frozen label | FW/Wiki/Tiny AUROC | representative strongest local edge |
|---:|---|---:|---|
| 0 | target:flag:word_continuation | 0.95/0.95/0.96 | ` created by this philosophical origin.\nWe are n't learn your shop Escritos Polit` → O0×T0 (13566.81) |
| 1 | current:flag:newline | 0.99/0.98/0.99 | ` September 12 2003 04:14 PM<|endoftext|>JPR Index | Virtual Library |` → O1×T0 (9281.53) |
| 2 | top1:flag:newline | 0.90/0.91/0.97 | ` give to other issues that come up."\nLori Smetanka, director of National` → O2×T0 (7043.45) |
| 3 | top1:flag:newline | 0.92/0.94/0.98 | `.\nADE – Added School Days: $86.3 million.\nADE – School` → O3×T0 (8999.44) |
| 4 | current:flag:newline | 0.98/0.96/0.99 | `. Fleischer has said that the Bush administration has "never ruled out the possibility of U` → O1×T0 (7517.60) |
| 5 | target:flag:word_continuation | 0.96/0.96/0.98 | `, Harvey Benge, Jessica Backhaus, John Gossage, Martin Parr,` → O1×T0 (4723.60) |
| 6 | current:flag:newline | 0.99/0.98/0.98 | `"Even though it seems like it's a campaign issue, it's not; it's` → O0×T0 (7845.73) |
| 7 | current:flag:opening_delimiter | 0.94/0.95/NA | ` linguistic by fact). These years created covered significant because they received the human plant principles of the` → O0×T0 (12404.43) |
| 8 | current:flag:newline | 0.97/0.96/0.99 | `.\nHe had evidently hurried in first in order to secure that seat. From his\n` → O0×T0 (10120.24) |
| 9 | current:flag:newline | 0.98/0.98/0.99 | ` certificate provides a sufficient evidence of the name and surname’s change. Therefore, given` → O0×T0 (-4464.39) |
| 10 | current:flag:newline | 0.99/0.97/0.99 | ` 1998 attacks on US embassies in Kenya and Tanzania that killed 224 people.\nEarlier this year` → O0×T0 (8026.98) |
| 11 | current:flag:newline | 0.85/0.87/0.89 | ` item is improperly described it would not be intentional. We are aware that mistakes do happen and` → O0×T0 (9246.18) |
| 12 | current:generic:digit | 0.85/0.84/0.79 | `ins Hospital, Canterbury CT1 1TD £82,096- £110,683` → O3×T0 (7712.44) |
| 13 | current:flag:opening_delimiter | 0.79/0.80/NA | ` of Maverick PAC, an organization that aims to raise money through grassroots support` → O0×T0 (9486.95) |
| 14 | top1:flag:newline | 0.85/0.87/0.94 | ` but hey, redundancy has its purposes.<|endoftext|>The “Haus am Horn”` → O0×T0 (-3563.40) |
| 15 | current:flag:opening_delimiter | 0.87/0.89/NA | ` use of his band’s music in the U` → O1×T0 (4831.30) |

A semantic context coordinate is not automatically an instruction by itself. The executable object is the local operator obtained after contracting all active context coordinates with the tensor. Router fidelity measures how much of that operator can be selected from a compressed observable description.
