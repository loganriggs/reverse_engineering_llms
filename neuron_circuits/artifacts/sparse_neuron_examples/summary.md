# Positive and negative examples for recurrent sparse neurons

Sparse means the strictest restriction—no layers 20–23 and no final token position—with a strict top-1 crossover of at most 100 scalar activations.

A positive signed causal score means copying the donor activation locally pushes toward the donor answer. A negative score means the same donor activation locally pushes against that answer. This is a first-order score at the base run, not the result of patching the unit alone.

## IOI

Most recurrent units: L15/P8/N3036 (7/8 sparse pairs), L13/P8/N3541 (7/8 sparse pairs), L13/P8/N3593 (7/8 sparse pairs), L10/P8/N1567 (6/8 sparse pairs), L11/P8/N1911 (6/8 sparse pairs)

### Top positive neuron–example combinations

| Rank | Unit | Counterfactual | Answers | Score | Activation change |
|---:|---|---|---|---:|---:|
| 1 | L13/P8/N3541 | `Dave and Robert went to the store. Dave gave the book to` → `Dave and Robert went to the store. Robert gave the book to` | Robert → Dave | 0.6471 | 1.57 |
| 2 | L15/P8/N3036 | `Alice and Carol went to the store. Alice gave the book to` → `Alice and Carol went to the store. Carol gave the book to` | Carol → Alice | 0.5869 | -1.59 |
| 3 | L13/P8/N3541 | `Alice and Carol went to the store. Alice gave the book to` → `Alice and Carol went to the store. Carol gave the book to` | Carol → Alice | 0.576 | 1.117 |
| 4 | L13/P8/N3541 | `John and Mary went to the store. John gave the book to` → `John and Mary went to the store. Mary gave the book to` | Mary → John | 0.4773 | 1.49 |
| 5 | L10/P8/N1567 | `Dave and Robert went to the store. Dave gave the book to` → `Dave and Robert went to the store. Robert gave the book to` | Robert → Dave | 0.4399 | 1.522 |

### Top negative neuron–example combinations

| Rank | Unit | Counterfactual | Answers | Score | Activation change |
|---:|---|---|---|---:|---:|
| 1 | L13/P8/N3541 | `Bob and Emma went to the store. Emma gave the book to` → `Bob and Emma went to the store. Bob gave the book to` | Bob → Emma | -0.9434 | -1.312 |
| 2 | L13/P8/N3541 | `James and Robert went to the store. Robert gave the book to` → `James and Robert went to the store. James gave the book to` | James → Robert | -0.8859 | -1.473 |
| 3 | L15/P8/N3036 | `Bob and Emma went to the store. Emma gave the book to` → `Bob and Emma went to the store. Bob gave the book to` | Bob → Emma | -0.4728 | 1.066 |
| 4 | L15/P8/N3036 | `Mary and James went to the store. James gave the book to` → `Mary and James went to the store. Mary gave the book to` | Mary → James | -0.4353 | 1.018 |
| 5 | L13/P8/N3593 | `Bob and Emma went to the store. Emma gave the book to` → `Bob and Emma went to the store. Bob gave the book to` | Bob → Emma | -0.2905 | 0.6328 |
## AGREEMENT

Most recurrent units: L18/P7/N1303 (9/9 sparse pairs), L17/P7/N4 (8/9 sparse pairs), L16/P7/N4197 (7/9 sparse pairs), L17/P7/N4047 (6/9 sparse pairs), L17/P7/N1479 (5/9 sparse pairs)

### Top positive neuron–example combinations

| Rank | Unit | Counterfactual | Answers | Score | Activation change |
|---:|---|---|---|---:|---:|
| 1 | L18/P7/N1303 | `Complete with is or are: The key near the cats` → `Complete with is or are: The keys near the cat` | is → are | 1.188 | -4.346 |
| 2 | L18/P7/N1303 | `Complete with is or are: The keys near the picture` → `Complete with is or are: The key near the pictures` | are → is | 0.887 | 4.346 |
| 3 | L18/P7/N1303 | `Complete with is or are: The book near the pictures` → `Complete with is or are: The books near the picture` | is → are | 0.8636 | -2.68 |
| 4 | L18/P7/N1303 | `Complete with is or are: The book near the dogs` → `Complete with is or are: The books near the dog` | is → are | 0.8479 | -2.68 |
| 5 | L18/P7/N1303 | `Complete with is or are: The reports near the bottle` → `Complete with is or are: The report near the bottles` | are → is | 0.7151 | 2.883 |

### Top negative neuron–example combinations

| Rank | Unit | Counterfactual | Answers | Score | Activation change |
|---:|---|---|---|---:|---:|
| 1 | L16/P7/N4197 | `Complete with is or are: The reports near the bottle` → `Complete with is or are: The report near the bottles` | are → is | -0.1259 | 5.342 |
| 2 | L17/P7/N4047 | `Complete with is or are: The reports near the bottle` → `Complete with is or are: The report near the bottles` | are → is | -0.08958 | -0.4609 |
| 3 | L16/P7/N4197 | `Complete with is or are: The book near the dogs` → `Complete with is or are: The books near the dog` | is → are | -0.03642 | -4.232 |
| 4 | L16/P7/N4197 | `Complete with is or are: The book near the pictures` → `Complete with is or are: The books near the picture` | is → are | -0.02557 | -4.232 |
| 5 | L17/P7/N4047 | `Complete with is or are: The book near the dogs` → `Complete with is or are: The books near the dog` | is → are | -0.004505 | -1.102 |
## BRACKETS

Most recurrent units: L4/P16/N3818 (6/11 sparse pairs), L0/P15/N105 (5/11 sparse pairs), L2/P15/N1089 (5/11 sparse pairs), L5/P16/N3363 (5/11 sparse pairs), L0/P15/N4295 (5/11 sparse pairs)

### Top positive neuron–example combinations

| Rank | Unit | Counterfactual | Answers | Score | Activation change |
|---:|---|---|---|---:|---:|
| 1 | L0/P15/N4295 | `[[] ->` → `{[] ->` | ] → } | 1.718 | 3.097 |
| 2 | L4/P16/N3818 | `({()} ->` → `{{()} ->` | ) → } | 1.504 | 4.938 |
| 3 | L0/P15/N4295 | `((()) ->` → `{(()) ->` | ) → } | 0.9766 | 1.938 |
| 4 | L0/P15/N4295 | `[(()) ->` → `{(()) ->` | ] → } | 0.9486 | 1.935 |
| 5 | L5/P16/N3363 | `((()) ->` → `{(()) ->` | ) → } | 0.8864 | 4.803 |

### Top negative neuron–example combinations

| Rank | Unit | Counterfactual | Answers | Score | Activation change |
|---:|---|---|---|---:|---:|
| 1 | L0/P15/N4295 | `[{} ->` → `({} ->` | ] → ) | -0.1215 | 1.472 |
| 2 | L5/P16/N3363 | `[[] ->` → `{[] ->` | ] → } | -0.07004 | -1.062 |
| 3 | L4/P16/N3818 | `[[{}] ->` → `([{}] ->` | ] → ) | -0.06603 | -3.445 |
| 4 | L0/P15/N105 | `([()] ->` → `[[()] ->` | ) → ] | -0.06243 | -0.3828 |
| 5 | L5/P16/N3363 | `{{[]} ->` → `[{[]} ->` | } → ] | -0.05776 | 1.148 |

Full top-five positive and negative lists for all five recurrent units per family are in `examples.json` and `examples.csv`.
