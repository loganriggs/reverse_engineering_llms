# Natural positive and negative examples for recurrent sparse neurons

These are intact, unpatched dataset examples. Positive and negative mean the raw sign of the post-gating MLP activation—not counterfactual influence, gradient attribution, or answer support.

## IOI

Scanned 132 natural prompts.

### L15/P8/N3036

Recurred in 7 sparse counterfactual circuits. At this coordinate, 0/132 prompts had positive activation and 132/132 had negative activation.

| Sign/rank | Natural prompt | Answer | Position token | Activation |
|---|---|---:|---|---:|
| −1 | `Alice and Carol went to the store. Carol gave the book to` | Alice | `ĠCarol` | -2.2031 |
| −2 | `Bob and Carol went to the store. Carol gave the book to` | Bob | `ĠCarol` | -2.0156 |
| −3 | `Alice and Emma went to the store. Emma gave the book to` | Alice | `ĠEmma` | -1.8047 |
| −4 | `John and Carol went to the store. Carol gave the book to` | John | `ĠCarol` | -1.7578 |
| −5 | `Bob and Emma went to the store. Emma gave the book to` | Bob | `ĠEmma` | -1.7578 |

### L13/P8/N3541

Recurred in 7 sparse counterfactual circuits. At this coordinate, 132/132 prompts had positive activation and 0/132 had negative activation.

| Sign/rank | Natural prompt | Answer | Position token | Activation |
|---|---|---:|---|---:|
| +1 | `Bob and Laura went to the store. Laura gave the book to` | Bob | `ĠLaura` | 2.1875 |
| +2 | `Emma and Laura went to the store. Laura gave the book to` | Emma | `ĠLaura` | 2.1562 |
| +3 | `Bob and Carol went to the store. Carol gave the book to` | Bob | `ĠCarol` | 2.1406 |
| +4 | `John and Carol went to the store. Carol gave the book to` | John | `ĠCarol` | 2.0625 |
| +5 | `Emma and Robert went to the store. Robert gave the book to` | Emma | `ĠRobert` | 2.0625 |

### L13/P8/N3593

Recurred in 7 sparse counterfactual circuits. At this coordinate, 66/132 prompts had positive activation and 66/132 had negative activation.

| Sign/rank | Natural prompt | Answer | Position token | Activation |
|---|---|---:|---|---:|
| +1 | `Mary and Emma went to the store. Mary gave the book to` | Emma | `ĠMary` | 0.46484 |
| +2 | `Bob and James went to the store. Bob gave the book to` | James | `ĠBob` | 0.41016 |
| +3 | `Bob and Laura went to the store. Bob gave the book to` | Laura | `ĠBob` | 0.39844 |
| +4 | `Dave and Tom went to the store. Dave gave the book to` | Tom | `ĠDave` | 0.39648 |
| +5 | `Bob and Dave went to the store. Bob gave the book to` | Dave | `ĠBob` | 0.37891 |
| −1 | `Emma and James went to the store. James gave the book to` | Emma | `ĠJames` | -0.57031 |
| −2 | `Alice and Dave went to the store. Dave gave the book to` | Alice | `ĠDave` | -0.53125 |
| −3 | `Dave and James went to the store. James gave the book to` | Dave | `ĠJames` | -0.51562 |
| −4 | `Alice and Laura went to the store. Laura gave the book to` | Alice | `ĠLaura` | -0.51172 |
| −5 | `Dave and Tom went to the store. Tom gave the book to` | Dave | `ĠTom` | -0.51172 |

### L10/P8/N1567

Recurred in 6 sparse counterfactual circuits. At this coordinate, 124/132 prompts had positive activation and 8/132 had negative activation.

| Sign/rank | Natural prompt | Answer | Position token | Activation |
|---|---|---:|---|---:|
| +1 | `Dave and Robert went to the store. Robert gave the book to` | Dave | `ĠRobert` | 1.6016 |
| +2 | `James and Robert went to the store. Robert gave the book to` | James | `ĠRobert` | 1.5938 |
| +3 | `Carol and Robert went to the store. Robert gave the book to` | Carol | `ĠRobert` | 1.5625 |
| +4 | `Laura and Robert went to the store. Robert gave the book to` | Laura | `ĠRobert` | 1.4766 |
| +5 | `John and Robert went to the store. Robert gave the book to` | John | `ĠRobert` | 1.4609 |
| −1 | `Tom and Emma went to the store. Tom gave the book to` | Emma | `ĠTom` | -0.070801 |
| −2 | `Bob and Laura went to the store. Bob gave the book to` | Laura | `ĠBob` | -0.054932 |
| −3 | `Emma and Robert went to the store. Emma gave the book to` | Robert | `ĠEmma` | -0.043701 |
| −4 | `Tom and James went to the store. Tom gave the book to` | James | `ĠTom` | -0.040527 |
| −5 | `Bob and Sarah went to the store. Bob gave the book to` | Sarah | `ĠBob` | -0.031982 |

### L11/P8/N1911

Recurred in 6 sparse counterfactual circuits. At this coordinate, 66/132 prompts had positive activation and 66/132 had negative activation.

| Sign/rank | Natural prompt | Answer | Position token | Activation |
|---|---|---:|---|---:|
| +1 | `Bob and Dave went to the store. Bob gave the book to` | Dave | `ĠBob` | 0.94531 |
| +2 | `Mary and Laura went to the store. Mary gave the book to` | Laura | `ĠMary` | 0.92578 |
| +3 | `Bob and Laura went to the store. Bob gave the book to` | Laura | `ĠBob` | 0.92188 |
| +4 | `Bob and Carol went to the store. Bob gave the book to` | Carol | `ĠBob` | 0.875 |
| +5 | `Emma and Laura went to the store. Emma gave the book to` | Laura | `ĠEmma` | 0.83594 |
| −1 | `Alice and Robert went to the store. Robert gave the book to` | Alice | `ĠRobert` | -1.0703 |
| −2 | `John and Robert went to the store. Robert gave the book to` | John | `ĠRobert` | -1.0312 |
| −3 | `Tom and Robert went to the store. Robert gave the book to` | Tom | `ĠRobert` | -1.0312 |
| −4 | `Bob and James went to the store. James gave the book to` | Bob | `ĠJames` | -1.0078 |
| −5 | `John and Tom went to the store. Tom gave the book to` | John | `ĠTom` | -0.99219 |

## AGREEMENT

Scanned 112 natural prompts.

### L18/P7/N1303

Recurred in 9 sparse counterfactual circuits. At this coordinate, 56/112 prompts had positive activation and 56/112 had negative activation.

| Sign/rank | Natural prompt | Answer | Position token | Activation |
|---|---|---:|---|---:|
| +1 | `Complete with is or are: The report near the keys` | is | `Ġreport` | 0.6875 |
| +2 | `Complete with is or are: The report near the dogs` | is | `Ġreport` | 0.6875 |
| +3 | `Complete with is or are: The report near the cats` | is | `Ġreport` | 0.6875 |
| +4 | `Complete with is or are: The report near the pictures` | is | `Ġreport` | 0.6875 |
| +5 | `Complete with is or are: The report near the bottles` | is | `Ġreport` | 0.6875 |
| −1 | `Complete with is or are: The keys near the dog` | are | `Ġkeys` | -3.9062 |
| −2 | `Complete with is or are: The keys near the cat` | are | `Ġkeys` | -3.9062 |
| −3 | `Complete with is or are: The keys near the report` | are | `Ġkeys` | -3.9062 |
| −4 | `Complete with is or are: The keys near the picture` | are | `Ġkeys` | -3.9062 |
| −5 | `Complete with is or are: The keys near the bottle` | are | `Ġkeys` | -3.9062 |

### L17/P7/N4

Recurred in 8 sparse counterfactual circuits. At this coordinate, 56/112 prompts had positive activation and 56/112 had negative activation.

| Sign/rank | Natural prompt | Answer | Position token | Activation |
|---|---|---:|---|---:|
| +1 | `Complete with is or are: The pictures near the key` | are | `Ġpictures` | 0.22363 |
| +2 | `Complete with is or are: The pictures near the dog` | are | `Ġpictures` | 0.22363 |
| +3 | `Complete with is or are: The pictures near the cat` | are | `Ġpictures` | 0.22363 |
| +4 | `Complete with is or are: The pictures near the report` | are | `Ġpictures` | 0.22363 |
| +5 | `Complete with is or are: The pictures near the bottle` | are | `Ġpictures` | 0.22363 |
| −1 | `Complete with is or are: The picture near the bottles` | is | `Ġpicture` | -2.8438 |
| −2 | `Complete with is or are: The picture near the trees` | is | `Ġpicture` | -2.8438 |
| −3 | `Complete with is or are: The picture near the books` | is | `Ġpicture` | -2.8438 |
| −4 | `Complete with is or are: The picture near the keys` | is | `Ġpicture` | -2.8125 |
| −5 | `Complete with is or are: The picture near the dogs` | is | `Ġpicture` | -2.8125 |

### L16/P7/N4197

Recurred in 7 sparse counterfactual circuits. At this coordinate, 56/112 prompts had positive activation and 56/112 had negative activation.

| Sign/rank | Natural prompt | Answer | Position token | Activation |
|---|---|---:|---|---:|
| +1 | `Complete with is or are: The key near the dogs` | is | `Ġkey` | 0.52734 |
| +2 | `Complete with is or are: The key near the cats` | is | `Ġkey` | 0.52734 |
| +3 | `Complete with is or are: The key near the reports` | is | `Ġkey` | 0.52734 |
| +4 | `Complete with is or are: The key near the pictures` | is | `Ġkey` | 0.52734 |
| +5 | `Complete with is or are: The key near the bottles` | is | `Ġkey` | 0.52734 |
| −1 | `Complete with is or are: The reports near the key` | are | `Ġreports` | -4.8438 |
| −2 | `Complete with is or are: The reports near the dog` | are | `Ġreports` | -4.8438 |
| −3 | `Complete with is or are: The reports near the cat` | are | `Ġreports` | -4.8438 |
| −4 | `Complete with is or are: The reports near the picture` | are | `Ġreports` | -4.8438 |
| −5 | `Complete with is or are: The reports near the bottle` | are | `Ġreports` | -4.8438 |

### L17/P7/N4047

Recurred in 6 sparse counterfactual circuits. At this coordinate, 63/112 prompts had positive activation and 49/112 had negative activation.

| Sign/rank | Natural prompt | Answer | Position token | Activation |
|---|---|---:|---|---:|
| +1 | `Complete with is or are: The reports near the key` | are | `Ġreports` | 0.87109 |
| +2 | `Complete with is or are: The reports near the dog` | are | `Ġreports` | 0.87109 |
| +3 | `Complete with is or are: The reports near the cat` | are | `Ġreports` | 0.87109 |
| +4 | `Complete with is or are: The reports near the picture` | are | `Ġreports` | 0.87109 |
| +5 | `Complete with is or are: The reports near the bottle` | are | `Ġreports` | 0.87109 |
| −1 | `Complete with is or are: The pictures near the bottle` | are | `Ġpictures` | -1.6094 |
| −2 | `Complete with is or are: The pictures near the tree` | are | `Ġpictures` | -1.6094 |
| −3 | `Complete with is or are: The pictures near the book` | are | `Ġpictures` | -1.6094 |
| −4 | `Complete with is or are: The pictures near the key` | are | `Ġpictures` | -1.5625 |
| −5 | `Complete with is or are: The pictures near the dog` | are | `Ġpictures` | -1.5625 |

### L17/P7/N1479

Recurred in 5 sparse counterfactual circuits. At this coordinate, 91/112 prompts had positive activation and 21/112 had negative activation.

| Sign/rank | Natural prompt | Answer | Position token | Activation |
|---|---|---:|---|---:|
| +1 | `Complete with is or are: The cat near the keys` | is | `Ġcat` | 0.69141 |
| +2 | `Complete with is or are: The cat near the dogs` | is | `Ġcat` | 0.69141 |
| +3 | `Complete with is or are: The cat near the reports` | is | `Ġcat` | 0.69141 |
| +4 | `Complete with is or are: The cat near the pictures` | is | `Ġcat` | 0.69141 |
| +5 | `Complete with is or are: The cat near the bottles` | is | `Ġcat` | 0.69141 |
| −1 | `Complete with is or are: The cats near the key` | are | `Ġcats` | -0.34375 |
| −2 | `Complete with is or are: The cats near the dog` | are | `Ġcats` | -0.34375 |
| −3 | `Complete with is or are: The cats near the report` | are | `Ġcats` | -0.34375 |
| −4 | `Complete with is or are: The cats near the picture` | are | `Ġcats` | -0.34375 |
| −5 | `Complete with is or are: The cats near the bottle` | are | `Ġcats` | -0.34375 |

## BRACKETS

Scanned 36 natural prompts.

### L4/P16/N3818

Recurred in 6 sparse counterfactual circuits. At this coordinate, 22/36 prompts had positive activation and 14/36 had negative activation.

| Sign/rank | Natural prompt | Answer | Position token | Activation |
|---|---|---:|---|---:|
| +1 | `{{} ->` | } | `}` | 3.2031 |
| +2 | `{[] ->` | } | `[]` | 2.5781 |
| +3 | `{{()} ->` | } | `()}` | 2.5312 |
| +4 | `{[()] ->` | } | `[` | 2.4531 |
| +5 | `{[{}] ->` | } | `[` | 2.4531 |
| −1 | `[{()} ->` | ] | `()}` | -2.9688 |
| −2 | `({} ->` | ) | `}` | -2.7812 |
| −3 | `({()} ->` | ) | `()}` | -2.4531 |
| −4 | `([{}] ->` | ) | `{}]` | -2.2188 |
| −5 | `[{[]} ->` | ] | `[]}` | -2.1094 |

### L0/P15/N105

Recurred in 5 sparse counterfactual circuits. At this coordinate, 28/36 prompts had positive activation and 8/36 had negative activation.

| Sign/rank | Natural prompt | Answer | Position token | Activation |
|---|---|---:|---|---:|
| +1 | `[() ->` | ] | `[` | 2.2188 |
| +2 | `[[] ->` | ] | `[` | 2.2188 |
| +3 | `[{} ->` | ] | `[` | 2.2188 |
| +4 | `[({}) ->` | ] | `[` | 2.2188 |
| +5 | `[{{}} ->` | ] | `[` | 2.2188 |
| −1 | `{{{}} ->` | } | `{{{` | -0.36914 |
| −2 | `({{}} ->` | ) | `({` | -0.015869 |
| −3 | `({} ->` | ) | `({` | -0.015503 |
| −4 | `({()} ->` | ) | `({` | -0.015503 |
| −5 | `({[]} ->` | ) | `({` | -0.015503 |

### L2/P15/N1089

Recurred in 5 sparse counterfactual circuits. At this coordinate, 16/36 prompts had positive activation and 20/36 had negative activation.

| Sign/rank | Natural prompt | Answer | Position token | Activation |
|---|---|---:|---|---:|
| +1 | `{(()) ->` | } | `{(` | 0.96094 |
| +2 | `{{{}} ->` | } | `{{{` | 0.87109 |
| +3 | `[(()) ->` | ] | `[(` | 0.83594 |
| +4 | `[([]) ->` | ] | `[(` | 0.83594 |
| +5 | `[{()} ->` | ] | `[{` | 0.42188 |
| −1 | `({} ->` | ) | `({` | -0.77344 |
| −2 | `({()} ->` | ) | `({` | -0.77344 |
| −3 | `({[]} ->` | ) | `({` | -0.77344 |
| −4 | `({{}} ->` | ) | `({` | -0.77344 |
| −5 | `([()] ->` | ) | `([` | -0.73047 |

### L5/P16/N3363

Recurred in 5 sparse counterfactual circuits. At this coordinate, 21/36 prompts had positive activation and 15/36 had negative activation.

| Sign/rank | Natural prompt | Answer | Position token | Activation |
|---|---|---:|---|---:|
| +1 | `{{{}} ->` | } | `}}` | 6.0625 |
| +2 | `{(()) ->` | } | `())` | 5.25 |
| +3 | `[(()) ->` | ] | `())` | 4.8438 |
| +4 | `[[()] ->` | ] | `()]` | 4 |
| +5 | `[[[]] ->` | ] | `[]` | 3.875 |
| −1 | `[{{}} ->` | ] | `{{` | -2.5312 |
| −2 | `{[()] ->` | } | `[` | -1.0156 |
| −3 | `{[{}] ->` | } | `[` | -1.0156 |
| −4 | `{[[]] ->` | } | `[` | -1 |
| −5 | `({{}} ->` | ) | `{` | -0.57812 |

### L0/P15/N4295

Recurred in 5 sparse counterfactual circuits. At this coordinate, 28/36 prompts had positive activation and 8/36 had negative activation.

| Sign/rank | Natural prompt | Answer | Position token | Activation |
|---|---|---:|---|---:|
| +1 | `{() ->` | } | `{` | 3.0781 |
| +2 | `{[] ->` | } | `{` | 3.0781 |
| +3 | `{([]) ->` | } | `{` | 3.0781 |
| +4 | `{({}) ->` | } | `{` | 3.0781 |
| +5 | `{[()] ->` | } | `{` | 3.0781 |
| −1 | `[({}) ->` | ] | `[` | -0.018921 |
| −2 | `[{{}} ->` | ] | `[` | -0.018921 |
| −3 | `[() ->` | ] | `[` | -0.018799 |
| −4 | `[[] ->` | ] | `[` | -0.018799 |
| −5 | `[{} ->` | ] | `[` | -0.018799 |

## ARITHMETIC

Scanned 152 natural prompts.

### L15/P6/N1606

Recurred in 8 sparse counterfactual circuits. At this coordinate, 15/152 prompts had positive activation and 137/152 had negative activation.

| Sign/rank | Natural prompt | Answer | Position token | Activation |
|---|---|---:|---|---:|
| +1 | `What is 0 * 4? Answer:` | 0 | `4` | 5.4062 |
| +2 | `What is 1 * 4? Answer:` | 4 | `4` | 5.4062 |
| +3 | `What is 1 + 4? Answer:` | 5 | `4` | 4.625 |
| +4 | `What is 2 * 4? Answer:` | 8 | `4` | 4.375 |
| +5 | `What is 0 + 4? Answer:` | 4 | `4` | 4.2188 |
| −1 | `What is 0 * 6? Answer:` | 0 | `6` | -0.92969 |
| −2 | `What is 1 * 6? Answer:` | 6 | `6` | -0.92969 |
| −3 | `What is 0 + 6? Answer:` | 6 | `6` | -0.91797 |
| −4 | `What is 7 - 6? Answer:` | 1 | `6` | -0.85938 |
| −5 | `What is 9 - 6? Answer:` | 3 | `6` | -0.85547 |

### L0/P6/N2234

Recurred in 8 sparse counterfactual circuits. At this coordinate, 0/152 prompts had positive activation and 152/152 had negative activation.

| Sign/rank | Natural prompt | Answer | Position token | Activation |
|---|---|---:|---|---:|
| −1 | `What is 2 * 4? Answer:` | 8 | `4` | -2.5 |
| −2 | `What is 2 + 4? Answer:` | 6 | `4` | -2.4844 |
| −3 | `What is 0 * 4? Answer:` | 0 | `4` | -2.4844 |
| −4 | `What is 3 + 4? Answer:` | 7 | `4` | -2.4688 |
| −5 | `What is 4 + 4? Answer:` | 8 | `4` | -2.4688 |

### L0/P6/N3180

Recurred in 6 sparse counterfactual circuits. At this coordinate, 8/152 prompts had positive activation and 144/152 had negative activation.

| Sign/rank | Natural prompt | Answer | Position token | Activation |
|---|---|---:|---|---:|
| +1 | `What is 8 - 7? Answer:` | 1 | `7` | 0.0068359 |
| +2 | `What is 7 - 7? Answer:` | 0 | `7` | 0.0058899 |
| +3 | `What is 9 - 7? Answer:` | 2 | `7` | 0.0058899 |
| +4 | `What is 1 * 7? Answer:` | 7 | `7` | 0.0058899 |
| +5 | `What is 0 * 7? Answer:` | 0 | `7` | 0.0053711 |
| −1 | `What is 0 * 0? Answer:` | 0 | `0` | -3.75 |
| −2 | `What is 7 * 0? Answer:` | 0 | `0` | -3.75 |
| −3 | `What is 8 * 0? Answer:` | 0 | `0` | -3.75 |
| −4 | `What is 9 * 0? Answer:` | 0 | `0` | -3.75 |
| −5 | `What is 0 + 0? Answer:` | 0 | `0` | -3.7344 |

### L13/P6/N4251

Recurred in 6 sparse counterfactual circuits. At this coordinate, 61/152 prompts had positive activation and 91/152 had negative activation.

| Sign/rank | Natural prompt | Answer | Position token | Activation |
|---|---|---:|---|---:|
| +1 | `What is 6 - 6? Answer:` | 0 | `6` | 3.9688 |
| +2 | `What is 9 - 9? Answer:` | 0 | `9` | 3.9062 |
| +3 | `What is 2 - 2? Answer:` | 0 | `2` | 3.8125 |
| +4 | `What is 8 - 8? Answer:` | 0 | `8` | 3.5156 |
| +5 | `What is 5 - 5? Answer:` | 0 | `5` | 3.3906 |
| −1 | `What is 0 + 1? Answer:` | 1 | `1` | -1.0156 |
| −2 | `What is 0 + 7? Answer:` | 7 | `7` | -1.0078 |
| −3 | `What is 0 + 6? Answer:` | 6 | `6` | -1 |
| −4 | `What is 0 + 3? Answer:` | 3 | `3` | -0.98047 |
| −5 | `What is 0 + 2? Answer:` | 2 | `2` | -0.91406 |

### L0/P6/N902

Recurred in 6 sparse counterfactual circuits. At this coordinate, 0/152 prompts had positive activation and 152/152 had negative activation.

| Sign/rank | Natural prompt | Answer | Position token | Activation |
|---|---|---:|---|---:|
| −1 | `What is 0 * 1? Answer:` | 0 | `1` | -2.9219 |
| −2 | `What is 2 * 1? Answer:` | 2 | `1` | -2.9219 |
| −3 | `What is 4 * 1? Answer:` | 4 | `1` | -2.9219 |
| −4 | `What is 5 * 1? Answer:` | 5 | `1` | -2.9219 |
| −5 | `What is 6 * 1? Answer:` | 6 | `1` | -2.9219 |

The full prompt, model prediction, correctness, operation, and activation are retained in `examples.json` and `examples.csv`.
