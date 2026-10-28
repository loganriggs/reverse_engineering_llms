# Downstream-sensitivity fingerprint router

The router sees only singleton attention-JVP response norms and their pairwise signed cosines, plus static mask bits. It never sees the full suffix response or candidate finite effects at evaluation time.

## Locked hypotheses

- **J0_downstream_fingerprint_validates: not supported.**
- **J1_fingerprint_beats_raw_state: not supported.**
- **J2_fingerprint_is_used: supported.**
- **J3_pairwise_downstream_geometry_is_insufficient: supported.**

Selected ridge alpha `0.1`.

| Split | Median fidelity | 10th percentile | Minimum |
|---|---:|---:|---:|
| fineweb_cv | 0.984 | 0.624 | -1.538 |
| wikitext | 0.963 | -0.870 | -3.686 |
| wikitext_shuffled | 0.966 | -1.729 | -3.520 |

The shuffled-site control tests whether dynamic fingerprints matter beyond global mask biases. Failure here rejects only singleton norms/cosines; head-level, factor-level, or higher-order path features remain possible.
