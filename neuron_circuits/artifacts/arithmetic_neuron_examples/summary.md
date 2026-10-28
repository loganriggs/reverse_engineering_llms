# Arithmetic sparse-neuron examples

Counterfactuals preserve the operation and left operand while changing the right operand. Sparse means no layers 20–23, no final token position, and a strict crossover of at most 100 scalar activations.

| Operation | Median crossover | Range | ≤100 activations |
|---|---:|---:|---:|
| Add | 8.5 | 2–153 | 10/12 |
| Subtract | 6.5 | 2–13 | 12/12 |
| Multiply | 6 | 3–971,137 | 11/12 |

Most recurrent units across the 33 sparse pairs: L15/P6/N1606 (8/33), L0/P6/N2234 (8/33), L0/P6/N3180 (6/33), L13/P6/N4251 (6/33), L0/P6/N902 (6/33)

## Top positive neuron–example combinations

| Rank | Unit | Operation | Counterfactual | Answers | Score | Activation change |
|---:|---|---|---|---|---:|---:|
| 1 | L0/P6/N3180 | multiply | `What is 5 * 1? Answer:` → `What is 5 * 0? Answer:` | 5 → 0 | 1.452 | -3.626 |
| 2 | L15/P6/N1606 | multiply | `What is 1 * 6? Answer:` → `What is 1 * 4? Answer:` | 6 → 4 | 1.102 | 6.34 |
| 3 | L13/P6/N4251 | subtract | `What is 2 - 0? Answer:` → `What is 2 - 2? Answer:` | 2 → 0 | 0.9772 | 3.6 |
| 4 | L0/P6/N3180 | multiply | `What is 2 * 3? Answer:` → `What is 2 * 0? Answer:` | 6 → 0 | 0.7981 | -3.715 |
| 5 | L0/P6/N2234 | subtract | `What is 8 - 4? Answer:` → `What is 8 - 3? Answer:` | 4 → 5 | 0.7917 | 2.203 |

## Top negative neuron–example combinations

| Rank | Unit | Operation | Counterfactual | Answers | Score | Activation change |
|---:|---|---|---|---|---:|---:|
| 1 | L0/P6/N3180 | subtract | `What is 9 - 5? Answer:` → `What is 9 - 0? Answer:` | 4 → 9 | -0.3678 | -3.656 |
| 2 | L0/P6/N3180 | multiply | `What is 1 * 0? Answer:` → `What is 1 * 5? Answer:` | 0 → 5 | -0.3672 | 3.686 |
| 3 | L0/P6/N3180 | multiply | `What is 1 * 0? Answer:` → `What is 1 * 6? Answer:` | 0 → 6 | -0.1357 | 3.706 |
| 4 | L0/P6/N902 | multiply | `What is 5 * 1? Answer:` → `What is 5 * 0? Answer:` | 5 → 0 | -0.125 | 2.876 |
| 5 | L15/P6/N1606 | multiply | `What is 1 * 1? Answer:` → `What is 1 * 5? Answer:` | 1 → 5 | -0.03309 | -0.3242 |

Positive scores locally support the donor answer; negative scores locally resist it. Scores are first-order attributions, not single-unit patch effects.
