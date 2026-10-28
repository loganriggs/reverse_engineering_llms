# L8 source-span response charts

Source projector P yields Qhat=PQ+QP-PQP, preserving every interaction touching P. Rank r needs at most 2r quadratic reads plus one exact linear read. Shared P comes from even-head discovery perturbations; conditional P comes only from declared head c_proj weight columns, with no held-out-head data fit. Conditional charts are source-indexed response explanations, not one common forward scalar function.

Locked gates: `{'source_transfer': False, 'family_transfer': False, 'conditional_advantage': True, 'natural_retention': False, 'larger_budget': True}`.

| Method | Natural R2 | Fit heads | Held-out heads | Half | Pairs |
|---|---:|---:|---:|---:|---:|
| natural64 | 0.859 | 0.515 | 0.534 | 0.499 | 0.512 |
| shared_source32 | 0.148 | 0.384 | 0.018 | 0.101 | 0.046 |
| natural128 | 0.921 | 0.645 | 0.696 | 0.658 | 0.672 |
| shared_source64 | 0.197 | 0.569 | 0.077 | 0.191 | 0.141 |
| conditional_source32 | 0.132 | 0.529 | 0.548 | 0.566 | 0.452 |
| conditional_source64 | 0.184 | 0.729 | 0.828 | 0.828 | 0.736 |

Conditional natural R2 is the mean over single-head charts. Each chart is an integrable quadratic function, but choosing a different chart for each changed source does not recover a single global state function. Rank32 source charts and natural64 have the same maximum 65-read budget per output; rank64 source charts and natural128 have 129 reads. No decoder consumes the native write.
