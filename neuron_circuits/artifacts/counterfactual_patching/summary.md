# Counterfactual neuron patching

Strict crossover means that the counterfactual answer became the model's full-vocabulary top-1 token.

| Task | Activation instances, targeted | Unique neurons, targeted | Instances by largest change | Instances, random |
|---|---:|---:|---:|---:|
| IOI | 6.5 | 4.5 | 983 | 758784 |
| Agreement | 8.5 | 8.5 | 831.5 | 642048 |
| Brackets | 3.5 | 3.5 | 15 | 420250 |

Counts are first observed crossovers along each nested ranking, refined to integer resolution. They are not certified global minima.

The targeted ranking uses the base-run gradient of `logit(counterfactual answer) - logit(original answer)` multiplied by the actual source-minus-base activation change. It therefore answers a pair-specific causal question and is intentionally answer-aware.

## Directionality
- **IOI:** toward first name: median 4, range 2–13; toward second name: median 51, range 17–111
- **Agreement:** are → is: median 14.5, range 11–40; is → are: median 4, range 4–6
- **Brackets:** ] → ): median 9, range 7–9; ) → }: median 4.5, range 3–6; ] → }: median 2, range 2–2; ) → ]: median 3.5, range 3–4; } → ]: median 1.5, range 1–2; } → ): median 7, range 7–7

## Recurring neurons
- **IOI:** layer 15 / channel 796 in 12/12 pairs, layer 14 / channel 1412 in 11/12 pairs, layer 16 / channel 2973 in 8/12 pairs
- **Agreement:** layer 20 / channel 4662 in 12/12 pairs, layer 18 / channel 1303 in 11/12 pairs, layer 21 / channel 1653 in 10/12 pairs
- **Brackets:** layer 22 / channel 1943 in 9/12 pairs, layer 22 / channel 3011 in 6/12 pairs, layer 23 / channel 4291 in 6/12 pairs
