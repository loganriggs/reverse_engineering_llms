# Oracle residual restoration and discovery scaling

New 128-document test, all maps frozen from1024 discovery documents.

| Restoration | Rank | KL error recovered | Mean CE increase | Native top1 agreement | Residual energy recovered |
|---|---:|---:|---:|---:|---:|
| Energy | 16 | 0.274 | 0.1160 | 0.863 | 0.364 |
| Energy | 128 | 0.622 | 0.0630 | 0.883 | 0.602 |
| Energy | 256 | 0.740 | 0.0408 | 0.910 | 0.701 |
| Downstream sensitive | 16 | 0.446 | 0.0845 | 0.879 | 0.267 |
| Downstream sensitive | 128 | 0.732 | 0.0461 | 0.920 | 0.545 |
| Downstream sensitive | 256 | 0.815 | 0.0301 | 0.943 | 0.647 |

Replication gates: {'small_residual_repairs': False, 'fisher_directional_advantage': True, 'broad_restoration_still_incomplete': True}.

Frozen H2511024-document restoration maps and H248 compiled model, no fitting/recalibration/selection. New128 documents disjoint from discovery and old test. Oracle coefficients still required. Native coefficients do not establish a forward solution. Cached same-corpus text; no new model or external-domain replication.
