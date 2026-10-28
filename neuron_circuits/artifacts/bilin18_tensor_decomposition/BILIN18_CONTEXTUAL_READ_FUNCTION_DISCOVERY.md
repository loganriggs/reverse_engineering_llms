# Contextual read-function discovery

A bounded 16-leaf readable tree predicts the top 2% of exact local attention-edge effects. Rules are selected on one 61,440-token FineWeb split and scored on another. Baseline rules see only distance, absolute source position, and source/destination frequency. Lexical rules add 50 preregistered static properties for each endpoint. Contextual rules add endpoint relations, repetition, induction-like matches, and intervening boundaries.

## Locked results

- **C0_targets_are_replicable: supported.**
- **C1_contextual_rules_predict_effects: supported.**
- **C2_context_adds_beyond_position_frequency: not supported.**
- **C3_relations_add_beyond_static_lexicon: not supported.**
- **C4_rules_are_split_stable: supported.**
- **C5_ranked_heads_are_more_readable: supported.**
- **C6_variables_are_reused_by_distinct_functions: supported.**

## Head results

| Head | group | baseline | lexical | contextual | split-rule Jaccard |
|---|---|---:|---:|---:|---:|
| L9H7 | ranked | 0.979 | 0.982 | 0.979 | 0.533 |
| L7H8 | ranked | 0.982 | 0.985 | 0.980 | 0.500 |
| L7H0 | ranked | 0.852 | 0.854 | 0.867 | 0.571 |
| L9H8 | ranked | 0.892 | 0.927 | 0.930 | 0.533 |
| L8H1 | ranked | 0.936 | 0.938 | 0.931 | 0.562 |
| L8H2 | ranked | 0.823 | 0.832 | 0.841 | 0.600 |
| L9H1 | ranked | 0.940 | 0.942 | 0.939 | 0.375 |
| L8H8 | ranked | 0.883 | 0.884 | 0.911 | 0.353 |
| L8H0 | bottom | 0.862 | 0.861 | 0.860 | 0.467 |
| L10H7 | bottom | 0.925 | 0.931 | 0.926 | 0.375 |
| L10H6 | bottom | 0.745 | 0.824 | 0.854 | 0.444 |
| L8H5 | bottom | 0.839 | 0.891 | 0.888 | 0.643 |
| L10H0 | bottom | 0.881 | 0.879 | 0.895 | 0.312 |
| L10H8 | bottom | 0.784 | 0.809 | 0.844 | 0.333 |
| L9H2 | bottom | 0.958 | 0.958 | 0.958 | 0.300 |
| L8H6 | bottom | 0.823 | 0.826 | 0.855 | 0.562 |

## Reused candidate variables

- `context:punctuation_between`: L9H7, L7H8, L7H0, L9H8, L8H1, L8H2, L9H1, L8H8; median |OV cosine| `0.026`.
- `relation:function_word_to_alphabetic`: L7H8, L7H0, L8H2, L8H8; median |OV cosine| `0.006`.
- `context:newline_between`: L9H7, L9H1, L8H8; median |OV cosine| `0.029`.
- `destination:determiner`: L9H1, L8H8; median |OV cosine| `0.029`.
- `destination:period`: L7H8, L9H1; median |OV cosine| `0.080`.
- `source:alphabetic`: L7H0, L9H8; median |OV cosine| `0.120`.
- `source:leading_space`: L7H8, L9H1; median |OV cosine| `0.080`.
- `source:long_piece`: L8H1, L8H2; median |OV cosine| `0.010`.
- `source:lowercase`: L9H7, L9H1; median |OV cosine| `0.048`.
- `source:punctuation`: L9H7, L8H1; median |OV cosine| `0.124`.

## Selected rules for ranked heads

- **L9H7**: `context:punctuation_between, distance:2-3, distance:1-1, distance:4-7, distance:8-15, distance:16-31, distance:4-7, context:newline_between, source:lowercase, distance:8-15, source:lowercase, source:punctuation, source:lowercase, destination:non_ascii, destination:lowercase`
- **L7H8**: `context:punctuation_between, distance:2-3, distance:1-1, distance:8-15, distance:4-7, distance:16-31, distance:4-7, destination:leading_space, destination:period, destination:leading_space, destination:suffix_s, distance:8-15, relation:both_alphabetic, relation:function_word_to_alphabetic, source:leading_space`
- **L7H0**: `context:punctuation_between, relation:function_word_to_alphabetic, distance:64-128, distance:32-63, relation:function_word_to_alphabetic, source:alphabetic, source_frequency_bin:7, distance:0-0, source_frequency_bin:7, distance:8-15, distance:16-31, distance:1-1, source:suffix_s, source:alphabetic, distance:4-7`
- **L9H8**: `context:punctuation_between, source:alphabetic, context:latest_occurrence_of_source_token, distance:2-3, distance:4-7, source:numeric, distance:32-63, distance:8-15, relation:same_coarse_category, source:period, source:alphabetic, distance:8-15, distance:16-31, source_position:0-15, distance:4-7`
- **L8H1**: `context:punctuation_between, distance:4-7, distance:2-3, distance:8-15, distance:8-15, distance:1-1, distance:16-31, source:long_piece, source:punctuation, source:punctuation, source:determiner, distance:4-7, destination:capitalized, source:determiner, source:long_piece`
- **L8H2**: `context:punctuation_between, relation:function_word_to_alphabetic, distance:64-128, distance:32-63, distance:16-31, relation:function_word_to_alphabetic, source:capitalized, source:capitalized, distance:8-15, distance:0-0, distance:4-7, distance:16-31, source:long_piece, destination:alphabetic, distance:8-15`
- **L9H1**: `context:punctuation_between, context:newline_between, distance:32-63, distance:64-128, distance:16-31, distance:16-31, distance:2-3, distance:8-15, distance:4-7, distance:4-7, source:leading_space, destination:determiner, context:newline_between, destination:period, source:lowercase`
- **L8H8**: `context:punctuation_between, context:newline_between, relation:function_word_to_alphabetic, distance:32-63, distance:64-128, relation:function_word_to_alphabetic, distance:8-15, distance:8-15, distance:4-7, relation:source_previous_equals_destination, distance:16-31, source_frequency_bin:7, destination:determiner, distance:4-7, distance:8-15`

## Guardrails

These trees explain the magnitude of a frozen local operator effect, not the sign, output semantics, or final loss. The positive class is defined separately as the top 2% in each split, so AUROC measures rank transfer rather than calibration. Predicates encode substantial human prior knowledge. Greedy trees can select correlated proxies and predicate identity can be unstable even when predictions transfer. A reused predicate with low OV cosine is evidence that one readable variable participates in distinct native transforms; it does not prove that the predicate is the model's unique internal ontology.
