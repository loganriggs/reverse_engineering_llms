# Stability boundaries of Layer-8 sparse program extraction

Rare failures of second-order response reconstruction are tested as ill-conditioned greedy selection boundaries. The causal failure labels are frozen from exact finite interventions.

## Locked hypotheses

- **S0_failures_are_rare_and_clustered: supported.**
- **S1_low_greedy_margin_predicts_failure: not supported.**
- **S2_cost_whitened_error_predicts_failure: supported.**
- **S3_global_response_cosine_is_inadequate: not supported.**
- **S4_stable_pairwise_regime_is_certifiable: supported.**

| Corpus | Failures/triples | Failure sites | AUROC: -gap | AUROC: whitened error | AUROC: raw Gram error | AUROC: -cosine | Certificate coverage/precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| fineweb | 2/1344 | 2/16 | 0.434 | 0.951 | 0.949 | 0.949 | 1.000/0.999 |
| wikitext | 11/1344 | 1/16 | 0.520 | 0.988 | 0.992 | 0.992 | 0.985/0.998 |

FineWeb certificate: minimum gap `0.000e+00`, maximum whitened error `1.160e-01`.

A useful certificate separates the regular second-order regime from singular states. Failure means these two scalar diagnostics do not safely identify the boundary; it does not make the bulk reconstruction false.
