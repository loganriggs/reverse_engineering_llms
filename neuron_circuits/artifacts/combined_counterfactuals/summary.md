# Combined counterfactual circuits

The same 36 counterfactual pairs were tested with scalar MLP activation patches and six attention intervention types. Strict crossover means that the donor answer became the full-vocabulary top-1 token.

| Task | MLP scalar | Query | Key | Value | Attention row | Message | Head output |
|---|---:|---:|---:|---:|---:|---:|---:|
| IOI | 6.5 (12/12) | 5 (12/12) | — (0/12) | 1 (12/12) | 4 (11/12) | 2.5 (12/12) | 2 (12/12) |
| Agreement | 8.5 (12/12) | — (0/12) | — (0/12) | 1 (12/12) | — (0/12) | 1 (12/12) | 1 (12/12) |
| Brackets | 3.5 (12/12) | 182 (3/12) | 1 (1/12) | 2 (12/12) | 95 (2/12) | 2.5 (12/12) | 1 (12/12) |

Counts are targeted medians with successful pairs in parentheses. MLP units are scalar activation instances. Attention query, key, value, message, and head-output units are 64-dimensional vectors; an attention row is a routing distribution. Counts are therefore causal intervention sizes, not equal-dimensional circuit sizes.

## Interpretation

- **IOI:** both routing and transported content can redirect the answer. The John/Mary example flips with five query rows, five attention rows, two messages, two head outputs, or six MLP scalar activations.
- **Agreement:** the result is carried by content. Query, key, attention-row, and joint Q+K patches never flip a pair, while one value, one message, or one head output flips every pair.
- **Brackets:** content patches are reliable; routing-only patches are rare and usually large. One head output, two values, or 2.5 messages is the typical targeted crossover.

## Representative routes

- **IOI:** MLP L15/N796 appears in 12/12 pairs. In John/Mary, attention messages L20/H9 repeated `John` → final `to` and L23/H9 initial `Mary` → final `to` jointly flip the answer.
- **Agreement:** MLP L20/N4662 appears in 12/12 pairs. In book/pictures, L19/H4 sends the subject `book` to the final prediction position.
- **Brackets:** MLP L22/N1943 appears in 9/12 pairs. In the shown example, L20/H4 and L20/H5 send the unmatched opener to the final arrow.
