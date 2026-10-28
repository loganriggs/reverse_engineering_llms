# Attention counterfactual patching

Strict crossover means the donor answer became the full-vocabulary top-1 token in a fresh forward pass.

| Task | Query | Key | Value | Pattern | Message | Head output | MLP scalar |
|---|---:|---:|---:|---:|---:|---:|---:|
| IOI | 5 (12/12) | — (0/12) | 1 (12/12) | 4 (11/12) | 2.5 (12/12) | 2 (12/12) | 6.5 |
| Agreement | — (0/12) | — (0/12) | 1 (12/12) | — (0/12) | 1 (12/12) | 1 (12/12) | 8.5 |
| Brackets | 182 (3/12) | 1 (1/12) | 2 (12/12) | 95 (2/12) | 2.5 (12/12) | 1 (12/12) | 3.5 |

Entries are median targeted units followed by successful pairs. One value or head-output unit is a 64-dimensional head vector; one message is a 64-dimensional source-to-destination contribution. An MLP unit is one scalar activation instance, so the counts are not dimension-for-dimension equivalents.

Agreement is the cleanest separation: query, key, pattern, and Q+K patches never flipped any pair, while exactly one value vector, one message, or one head-output vector flipped every pair. IOI uses both routing and content. Bracket behavior is predominantly content-controlled in this intervention family.

## IOI example

For John/Mary, two messages flip Mary to John: layer 20 head 9 from the repeated-subject token `John` to the final `to`, and layer 23 head 9 from the initial `Mary` token to the final `to`.
