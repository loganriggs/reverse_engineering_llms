# Are spectral Jacobian modes semantic verbs?

The top 16 FineWeb transport-Jacobian PCA modes were frozen. Each mode selected one simple current-token, next-token, or position predicate and polarity using only discovery data. Those exact explanations were evaluated on disjoint FineWeb, WikiText, and TinyStories. Twenty random rotations received the identical label-search budget. Modes 17–32 are a spectral-bulk control.

## Locked hypotheses

- **E0_outlier_modes_have_simple_semantics: not supported.**
- **E1_individual_semantics_transfer_ood: supported.**
- **E2_pca_axes_are_semantically_privileged: not supported.**
- **E3_outlier_modes_are_more_semantic_than_bulk: supported.**
- **E4_mode_labels_are_diverse: supported.**
- **E5_mode_subspace_contains_semantic_information: supported.**
- **E6_position_does_not_explain_the_modes: supported.**

| Corpus | Top-mode median / >=.80 | Bulk median | Random-rotation p95 | Top-subspace decoder |
|---|---:|---:|---:|---:|
| fineweb | 0.703 / 5 | 0.604 | 0.876 | 0.853 |
| wikitext | 0.774 / 7 | 0.669 | 0.871 | 0.792 |
| tinystories | 0.814 / 6 | 0.608 | 0.929 | 0.869 |

## Frozen top-mode explanations

| Mode | Predicate | Discovery | FineWeb | WikiText | TinyStories |
|---:|---|---:|---:|---:|---:|
| 1 | `next:non_ascii` | 0.765 | 0.719 | 0.772 | n/a |
| 2 | `current:newline` | 0.989 | 0.993 | 0.984 | 0.990 |
| 3 | `current:newline` | 0.862 | 0.823 | 0.962 | 0.946 |
| 4 | `current:newline` | 0.939 | 0.941 | 0.972 | 0.950 |
| 5 | `next:word_continuation` | 0.771 | 0.796 | 0.833 | 0.674 |
| 6 | `current:newline` | 0.724 | 0.779 | 0.774 | 0.915 |
| 7 | `current:newline` | 0.854 | 0.910 | 0.930 | 0.947 |
| 8 | `current:newline` | 0.881 | 0.870 | 0.841 | 0.814 |
| 9 | `current:comma_period` | 0.712 | 0.676 | 0.609 | 0.758 |
| 10 | `next:digit` | 0.626 | 0.674 | 0.493 | n/a |
| 11 | `next:digit` | 0.699 | 0.651 | 0.773 | n/a |
| 12 | `next:non_ascii` | 0.778 | 0.607 | 0.546 | n/a |
| 13 | `current:newline` | 0.679 | 0.687 | 0.486 | 0.679 |
| 14 | `next:digit` | 0.777 | 0.680 | 0.815 | n/a |
| 15 | `next:suffix_ed` | 0.710 | 0.632 | 0.577 | 0.551 |
| 16 | `current:newline` | 0.623 | 0.529 | 0.739 | 0.685 |

Individual-mode success would support a basis-dependent verb vocabulary. Subspace-only success with random rotations matching PCA would instead mean semantic information is present but mode coordinates are conventional. Failure of both would leave the spectral outliers as reusable causal geometry without a simple token-local semantic description.
