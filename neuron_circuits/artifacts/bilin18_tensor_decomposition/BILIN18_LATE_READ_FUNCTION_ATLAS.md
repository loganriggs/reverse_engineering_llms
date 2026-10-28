# Large-corpus atlas of late read functions

This test asks whether late heads selected by an exact local derivative program admit shorter, reusable descriptions on two disjoint 98,304-token FineWeb panels. A function invocation is described by relative-distance address, source-variable class, and its native head transform. Human token classes are compared with equally sized frequency bins and human labels shuffled within frequency strata.

## Locked results

- **H0_exact_partitions: supported.**
- **H1_ranked_profiles_are_stable: supported.**
- **H2_human_grammar_beats_controls: not supported.**
- **H3_derivative_importance_predicts_readability: supported.**
- **H4_ranked_heads_have_compact_programs: supported.**
- **H5_route_types_replicate: supported.**
- **H6_ranked_heads_are_more_stable_than_rest: not supported.**

## Ranked heads

| Head | joint cos | literal fit | best simple program | simple fit | route Jaccard | leading route |
|---|---:|---:|---|---:|---:|---|
| L9H7 | 0.998 | 0.996 | rank1 | 0.991 | 0.882 | 1 / function_word |
| L7H8 | 0.999 | 0.997 | rank1 | 0.994 | 0.778 | 1 / function_word |
| L7H0 | 0.998 | 0.995 | rank1 | 0.993 | 1.000 | 32-127 / lowercase |
| L9H8 | 0.997 | 0.993 | rank1 | 0.973 | 0.778 | 8-15 / period |
| L8H1 | 0.997 | 0.994 | rank1 | 0.988 | 0.882 | 1 / lowercase |
| L8H2 | 0.990 | 0.980 | additive | 0.979 | 0.778 | 32-127 / lowercase |
| L9H1 | 0.997 | 0.994 | rank1 | 0.988 | 0.882 | 2-3 / lowercase |
| L8H8 | 0.992 | 0.985 | rank1 | 0.958 | 0.882 | 0 / lowercase |

## Concrete high-effect edges

### L9H7
- `\n` (newline, distance 5) -> ` Institute`; attention `-0.518`; context: `at Beohar\nBluhm Cardiovascular Institute, Northwestern`
- `'t` (other, distance 1) -> ` share`; attention `-0.568`; context: ` music if we can't share it with`
- `?` (question_exclamation, distance 1) -> ` Step`; attention `-0.415`; context: ` not fun to do? Step on a`
- `|` (code_operator, distance 1) -> `Option`; attention `-0.494`; context: `17 AM|\n|Option||Are`

### L7H8
- ` joining` (suffix, distance 0) -> ` joining`; attention `0.591`; context: ` Hall of Fame, joining sister Cheryl`
- ` may` (function_word, distance 1) -> ` be`; attention `0.665`; context: ` license, GSK may be required to`
- ` keeping` (suffix, distance 1) -> ` a`; attention `0.635`; context: ` your first employee, keeping a few fundamental`
- ` involves` (suffix, distance 1) -> `\n`; attention `0.726`; context: ` laser liposuction involves\nless bruising`

### L7H0
- ` debate` (lowercase, distance 1) -> `,`; attention `-0.553`; context: ` tonight’s debate, Obama can`
- ` campus` (suffix, distance 2) -> `\n`; attention `-0.492`; context: ` and international travel to campus.\nAs you`
- `ways` (suffix, distance 2) -> ` a`; attention `-0.503`; context: ` tree-lined fairways at a good value`
- ` writers` (suffix, distance 5) -> `�`; attention `-0.481`; context: ` bullying among young adult writers, I thought I’d`

### L9H8
- `\n` (newline, distance 0) -> `\n`; attention `0.514`; context: ` found them Aug.\nSavannah`
- `�` (non_ascii, distance 15) -> `�`; attention `0.557`; context: ` all call the terrorists’ “actions evil, and denounce them without equivocation.” But`
- `\n` (newline, distance 2) -> ` talking`; attention `0.528`; context: `$ 17.40\nWhile talking heads debate`
- `<|endoftext|>` (other, distance 0) -> `<|endoftext|>`; attention `0.611`; context: ` a good thing !<|endoftext|>OAK`

### L8H1
- ` Harry` (capitalized, distance 1) -> ` S`; attention `0.500`; context: ` as Albert Einstein, Harry S. Truman`
- ` Brooke` (capitalized, distance 1) -> ` C`; attention `0.539`; context: ` the Vice Squad, Brooke Cessna`
- ` Judd` (capitalized, distance 1) -> ` Green`; attention `0.486`; context: ` Ted Hearne, Judd Greenstein,`
- ` Sarah` (capitalized, distance 1) -> ` Silver`; attention `0.352`; context: `,... 56\n- Sarah Silverman:`

### L8H2
- ` Canberra` (capitalized, distance 0) -> ` Canberra`; attention `-0.344`; context: `<|endoftext|>On Thursday in Canberra, Australia`
- ` India` (capitalized, distance 5) -> `\n`; attention `-0.212`; context: ` Bank of India to the banks.\nThe earlier`
- ` Australia` (capitalized, distance 7) -> `\n`; attention `-0.208`; context: `- State of South Australia\n- Boscombe Hotels\n- Province`
- ` metres` (suffix, distance 39) -> `.`; attention `-0.211`; context: `.5 trillion cubic metres of gas overall here – we could also set up another export centre focused on the Asia-Pacific region, an area in which we should concentrate above all on developing liquefied natural gas exports. Let me`

### L9H1
- ` years` (suffix, distance 1) -> ` we`; attention `0.541`; context: `\nFor over 50 years we’`
- ` page` (lowercase, distance 4) -> `le`; attention `0.654`; context: `\nOn the Facebook page Kurk Gayle observed the`
- ` later` (lowercase, distance 1) -> ` I`; attention `0.593`; context: `s that a year later I’`
- ` arrived` (suffix, distance 1) -> ` they`; attention `0.596`; context: ` police. When police arrived they entered the`

### L8H8
- `�` (non_ascii, distance 22) -> `�`; attention `-0.494`; context: ` would say, ‘Do you know that guy (pointing at their friend or relative) who went is a KGB agent.’\n`
- `�` (non_ascii, distance 3) -> `�`; attention `-0.450`; context: ` established reputation for ‘high quality’ care`
- `�` (non_ascii, distance 4) -> `�`; attention `-0.474`; context: `.\n‘The Next Revolution’ host`
- ` corrections` (suffix, distance 2) -> ` been`; attention `0.597`; context: `OCR). No corrections have been made to`

## Interpretation guardrails

The route score is the squared norm of a signed attention-edge effect on a frozen weight-derived local operator sketch. It is not attention weight alone and not final-token loss. Squared grouped effects contain within-type cross terms, so scores are useful route profiles rather than an additive conservation law. A stable literal table is a reproducible lookup, not automatically a readable feature. A simple distance/category model is evidence for a shorter program only when it survives the frequency and shuffled-label controls. High-effect examples are selected after measuring magnitude and therefore illustrate, rather than independently validate, a label.
