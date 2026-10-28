# Blind replication of Layer-8 regular/singular path regimes

The second-order reconstruction, failure threshold, and cost-whitened error certificate were frozen before evaluating new FineWeb rows and TinyStories.

## Locked hypotheses

- **R0_pairwise_bulk_replicates: supported.**
- **R1_frozen_certificate_retains_precision: supported.**
- **R2_rejected_region_is_failure_enriched: not supported.**
- **R3_no_broad_third_order_regime: not supported.**

Frozen certificate threshold: `0.115970`.

| Corpus | Fidelity median/q10 | Failures/triples | Failure sites | Certificate coverage/precision | Accepted/rejected failure rate | Enrichment |
|---|---:|---:|---:|---:|---:|---:|
| fineweb_new | 1.000/1.000 | 8/1344 | 4/16 | 1.000/0.994 | 0.0060/nan | nanx |
| tinystories | 1.000/1.000 | 4/1344 | 3/16 | 1.000/0.997 | 0.0030/nan | nanx |

Bulk replication supports the low-order regular regime. Certificate enrichment is the stronger test of whether the singular boundary transfers. Zero failures or zero rejections cannot establish enrichment and are left unsupported.
