# Hierarchical module-to-neuron circuit analysis

Whole attention and MLP outputs were localized first. Individual MLP neurons were then searched only inside the three strongest MLP layer-position modules for each counterfactual.

## Capability screen

| Task | Full-vocabulary top-1 |
|---|---:|
| maximum | 98/100 (98%) |
| minimum | 92/100 (92%) |
| second largest | 12/100 (12%) |
| argmax | 36/100 (36%) |
| second argmax | 29/100 (29%) |
| second largest few shot | 50/100 (50%) |
| second argmax few shot | 26/100 (26%) |

Second-largest and second-argmax were not interpreted because the model failed the capability screen; max and min were reliable enough to analyze.

## Working circuit hypothesis

1. **Local feature encoding:** early MLP neurons at the changed list element or outer bracket encode the token identity. For max/min, the layer-0 input-site neurons have local-value η² = 1.0.
2. **Selection/routing:** attention outputs at the final answer position, especially layers 16 and 20, aggregate the relevant input. Brackets also repeatedly use an earlier attention output near the bracket sequence.
3. **Answer writeout:** late MLP neurons at the final position are selective for the resulting digit or closing bracket and causally move the answer margin.

This is a supported circuit skeleton, not yet a complete max/min comparator proof: head-level path patching is still needed to identify exactly how candidate values are compared and routed.

## Brackets

- attention output: target top-1 in 4/4 pairs; median crossover 1.5 module outputs.
- mlp output: target top-1 in 4/4 pairs; median crossover 2.0 module outputs.
- Localized neurons: target top-1 in 4/4 pairs; median crossover 11.0 neurons.
- Recurrent attention output locations: L20/P17 (4/4), L17/P17 (4/4), L16/P17 (4/4), L4/P16 (3/4).
- Recurrent mlp output locations: L20/P17 (4/4), L22/P17 (3/4), L23/P17 (3/4), L0/P15 (3/4).

| Neuron | Sparse pairs | Mean exact effect | Answer η² | Local-value η² | Winner Δ |
|---|---:|---:|---:|---:|---:|
| L20/P17/N855 | 4 | 1.422 | 0.931 | — | — |
| L20/P17/N2988 | 4 | 0.812 | 0.767 | — | — |
| L20/P17/N2412 | 4 | 0.703 | 0.977 | — | — |
| L0/P15/N4295 | 2 | 0.906 | 0.860 | — | — |
| L0/P15/N3384 | 2 | 0.781 | 0.695 | — | — |

- `L20/P17/N855`: high `({} ->` (+1.094); low `[{} ->` (-8.812).
- `L20/P17/N2988`: high `{[[]] ->` (+2.172); low `[{} ->` (-4.375).
- `L20/P17/N2412`: high `(([]) ->` (+4.250); low `[[] ->` (-0.645).
- `L0/P15/N4295`: high `{() ->` (+3.078); low `[({}) ->` (-0.019).
- `L0/P15/N3384`: high `([()] ->` (+3.016); low `[(()) ->` (-0.096).

## Maximum

- attention output: target top-1 in 4/4 pairs; median crossover 2.0 module outputs.
- mlp output: target top-1 in 4/4 pairs; median crossover 2.0 module outputs.
- Localized neurons: target top-1 in 4/4 pairs; median crossover 4.0 neurons.
- Recurrent attention output locations: L20/P19 (4/4), L16/P19 (4/4), L22/P19 (4/4), L16/P15 (3/4).
- Recurrent mlp output locations: L22/P19 (4/4), L21/P19 (3/4), L20/P19 (3/4), L0/P9 (2/4).

| Neuron | Sparse pairs | Mean exact effect | Answer η² | Local-value η² | Winner Δ |
|---|---:|---:|---:|---:|---:|
| L21/P19/N2550 | 2 | 2.062 | 0.708 | — | — |
| L22/P19/N4741 | 2 | 2.031 | 0.737 | — | — |
| L20/P19/N412 | 2 | 1.531 | 0.374 | — | — |
| L0/P9/N4489 | 2 | 0.500 | 0.053 | 1.000 | -0.435 |
| L0/P6/N1570 | 2 | 0.312 | 0.046 | 1.000 | 1.383 |

- `L21/P19/N2550`: high `Find the maximum number in [1, 8, 6, 0]. Answer:` (+23.500); low `Find the maximum number in [6, 1, 7, 0]. Answer:` (-1.172).
- `L22/P19/N4741`: high `Find the maximum number in [0, 6, 1, 3]. Answer:` (+0.805); low `Find the maximum number in [1, 2, 7, 0]. Answer:` (-17.250).
- `L20/P19/N412`: high `Find the maximum number in [0, 2, 4, 1]. Answer:` (+0.102); low `Find the maximum number in [1, 8, 7, 6]. Answer:` (-29.250).
- `L0/P9/N4489`: high `Find the maximum number in [2, 0, 5, 6]. Answer:` (-0.010); low `Find the maximum number in [9, 7, 0, 6]. Answer:` (-3.297).
- `L0/P6/N1570`: high `Find the maximum number in [8, 4, 0, 9]. Answer:` (+2.281); low `Find the maximum number in [0, 1, 9, 2]. Answer:` (-0.570).

## Minimum

- attention output: target top-1 in 4/4 pairs; median crossover 2.5 module outputs.
- mlp output: target top-1 in 4/4 pairs; median crossover 3.5 module outputs.
- Localized neurons: target top-1 in 4/4 pairs; median crossover 6.0 neurons.
- Recurrent attention output locations: L20/P19 (4/4), L16/P19 (4/4), L14/P19 (3/4), L21/P19 (3/4).
- Recurrent mlp output locations: L20/P19 (3/4), L22/P19 (3/4), L21/P19 (2/4), L16/P19 (2/4).

| Neuron | Sparse pairs | Mean exact effect | Answer η² | Local-value η² | Winner Δ |
|---|---:|---:|---:|---:|---:|
| L22/P19/N4584 | 2 | 0.688 | 0.369 | — | — |
| L21/P19/N4580 | 2 | 0.562 | 0.869 | — | — |
| L22/P19/N38 | 2 | 0.500 | 0.402 | — | — |
| L0/P6/N2206 | 1 | 2.625 | 0.145 | 1.000 | 0.790 |
| L0/P9/N2206 | 1 | 2.500 | 0.017 | 1.000 | 0.078 |

- `L22/P19/N4584`: high `Find the minimum number in [8, 7, 0, 2]. Answer:` (+0.660); low `Find the minimum number in [3, 0, 8, 9]. Answer:` (-13.688).
- `L21/P19/N4580`: high `Find the minimum number in [4, 5, 6, 0]. Answer:` (+12.125); low `Find the minimum number in [5, 1, 4, 2]. Answer:` (-0.621).
- `L22/P19/N38`: high `Find the minimum number in [2, 3, 4, 1]. Answer:` (+0.664); low `Find the minimum number in [2, 0, 9, 8]. Answer:` (-4.656).
- `L0/P6/N2206`: high `Find the minimum number in [2, 8, 3, 4]. Answer:` (+4.219); low `Find the minimum number in [9, 1, 3, 2]. Answer:` (+0.021).
- `L0/P9/N2206`: high `Find the minimum number in [6, 2, 8, 0]. Answer:` (+4.250); low `Find the minimum number in [4, 9, 1, 5]. Answer:` (-0.004).

η² is the fraction of activation variance explained by the answer or local list value. Winner Δ is mean activation when that list position contains the max/min minus its mean otherwise. These are descriptive selectivity measures; exact patches supply the causal evidence.
