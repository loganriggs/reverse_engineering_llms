# Is midpoint circuit compilation semantic-specific?

Every declared semantic edit is paired with a cross-family target chosen to match its normalized embedding-edit distance from the same source token. Both edits use identical contexts, rank, midpoint-JVP budget, realized-delta patch, and output metric.

## Locked hypotheses

- **S0_semantic_edits_have_better_midpoint_compilation: not supported.**
- **S1_semantic_edits_have_more_predictable_support: supported.**
- **S2_matched_controls_also_compile: supported.**
- **S3_matched_control_direction_is_predictable: not supported.**
- **S4_distance_matching_is_tight: not supported.**
- **S5_any_semantic_advantage_survives_pair_bootstrap: not supported.**

| Layer | Corpus | Semantic midpoint/oracle/support/cosine | Control midpoint/oracle/support/cosine | Semantic-control pair median (95% CI) | Distance mismatch |
|---:|---|---:|---:|---:|---:|
| 0 | fineweb | 0.673/0.661/0.858/0.996 | 0.729/0.729/0.784/0.988 | -0.013 (-0.238, +0.110) | 0.220 |
| 0 | tinystories | 0.665/0.724/0.855/0.995 | 0.695/0.696/0.763/0.988 | -0.072 (-0.211, +0.064) | 0.220 |
| 1 | fineweb | 0.667/0.744/0.727/0.977 | 0.665/0.696/0.561/0.939 | -0.043 (-0.080, +0.164) | 0.220 |
| 1 | tinystories | 0.738/0.739/0.721/0.981 | 0.661/0.770/0.561/0.945 | +0.033 (-0.010, +0.184) | 0.220 |

If controls compile equally well, midpoint routing is a generic finite-edit property of the tensor architecture. It remains a valid circuit tool but is not evidence that the selected lexical relations share a privileged semantic representation.
