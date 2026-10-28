# Prospective matched tests of candidate read relations

Candidate relations were frozen from earlier descriptive examples, then tested on two later FineWeb blocks. Every candidate edge is matched to a non-candidate edge at the same destination position, log-distance bin, source-frequency bin, and destination-frequency bin. AUROC asks whether exact local operator-effect magnitude ranks candidate edges above controls.

## Locked results

- **P1_adjacent_name_is_specific: not supported.**
- **P2_boundary_read_is_specific: supported.**
- **P3_function_word_read_is_specific: supported.**
- **P4_long_range_capitalized_is_specific: not supported.**
- **P5_some_new_relation_is_validated: supported.**
- **P6_unsupervised_relation_candidates_exist: not supported.**

## Expected pairs

| Candidate | Expected head | split AUCs | median-other AUCs | pass |
|---|---|---:|---:|---|
| adjacent_name_piece | L8H1 | 0.518 / 0.513 | 0.475 / 0.474 | no |
| adjacent_function_word | L9H7 | 0.616 / 0.633 | 0.400 / 0.392 | yes |
| boundary_source | L9H8 | 0.777 / 0.773 | 0.529 / 0.540 | yes |
| long_range_capitalized | L8H2 | 0.500 / 0.530 | 0.483 / 0.493 | no |

## Full head matrix (mean across splits)

| Candidate | L9H7 | L7H8 | L7H0 | L9H8 | L8H1 | L8H2 | L9H1 | L8H8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| adjacent_name_piece | 0.275 | 0.485 | 0.475 | 0.439 | 0.516 | 0.612 | 0.359 | 0.604 |
| adjacent_function_word | 0.625 | 0.788 | 0.336 | 0.387 | 0.406 | 0.346 | 0.483 | 0.401 |
| boundary_source | 0.556 | 0.475 | 0.577 | 0.775 | 0.520 | 0.535 | 0.574 | 0.431 |
| long_range_capitalized | 0.474 | 0.497 | 0.465 | 0.498 | 0.495 | 0.515 | 0.454 | 0.513 |
| repeated_same_token | 0.573 | 0.576 | 0.584 | 0.527 | 0.514 | 0.510 | 0.571 | 0.532 |
| induction_like_previous_match | 0.519 | 0.545 | 0.549 | 0.498 | 0.541 | 0.538 | 0.525 | 0.543 |

## Guardrails

This is a prospective validation of local read-effect magnitude, not a final-loss intervention. Matching is exact on coarse declared strata but not on token identity, syntax, or all correlated lexical properties. The candidate hypotheses came from selected high-effect examples, so only these later corpus blocks are confirmatory. Failure means the proposed readable relation is not a sufficiently specific description under this metric; it does not prove the head lacks other semantics.
