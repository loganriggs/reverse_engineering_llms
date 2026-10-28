# M5: cross-document token tables change downstream predictions

2026-10-28. Follows the fixed M4 causal plan on the SAME64 reused BSF training documents. Token means for each document are estimated from the opposite32-document half, on deduplicated M4 native product writes. No new text, table refit, scientific holdout exposure or native-channel pruning.

## Intervention

For each MLP0/1/2 separately, replace the product write at covered positions with its token mean and retain Down bias. At uncovered positions preserve the live native write exactly. A norm-only control scales the live native product write to the predicted token-mean product norm, retaining its direction and Down bias. All downstream computations recompute live, including effects on later positions. The control requires the original computation and is not a compressed model.

Coverage averages75.594/128positions (59.06%). Output KL and native top1 agreement use all128positions and all50304logits; CE uses127next-token targets. Metrics include uncovered positions, which can still experience downstream effects. All64windows have equal length, so equal-document means also equal position-weighted means. Baseline next-token CE is3.231744nats.

## Results

| Layer | Replacement | Output KL | Native top1 agreement | Change in CE (nats) | Local relative squared write error |
|---|---|---:|---:|---:|---:|
| MLP0 | table | 0.048414 | 89.82% | +0.048124 | 0.1495 |
| MLP0 | norm_only | 0.000137 | 99.39% | -0.000022 | 0.0091 |
| MLP1 | table | 0.183539 | 79.30% | +0.192442 | 0.3481 |
| MLP1 | norm_only | 0.001428 | 98.03% | +0.001597 | 0.0704 |
| MLP2 | table | 0.106199 | 83.12% | +0.110063 | 0.4550 |
| MLP2 | norm_only | 0.001951 | 97.27% | +0.002489 | 0.0765 |

MLP1 table replacement has the largest downstream effect, despite MLP2 having larger local reconstruction error. MLP1 predictions match native top1 only79.30% of the time; its CE increases0.19244nats. Thus local error or contextual variance alone does not rank downstream sensitivity.

For every layer, table replacement changes the output distribution much more than changing only product-write magnitude to the same table norm. This supports preserving directional information beyond current-token means. It does not attribute an additive fraction of KL to direction, establish semantic selectivity, or show that magnitudes never matter. The two interventions have different local error magnitudes, and downstream normalization can make scale changes relatively innocuous.

These errors mix context dependence with finite-sample token-mean estimation error. Thirty-two source documents provide limited examples per token. Consequently this experiment does not isolate a pure contextual component or contradict historical token-table CE-stake recovery numbers collected with different data, tables and denominators. A large residual in write space and a causal feature are different claims.

## Implication for the research loop

A featurizer optimized only for raw reconstruction can miss downstream weighting: on this cohort, MLP1 has less local table error than MLP2 but more output damage. Preserve downstream causal fidelity when comparing representations. This result does not yet nominate meaningful manifolds or a unique computational unit.

Next bounded control before a semantic interpretation: at MLP1 only, compare the table-direction change to random tangent-direction changes at the SAME per-position final product norm and SAME angular distance from the native product. Preserve covered-position mask/bias and use a fixed small seed set with all results reported. That matches both norm and local product error, unlike the radial control here. Its deliverable is a reference for whether token-table errors preferentially affect downstream-sensitive directions; stop after the fixed comparison and use the result to choose a causal-featurizer target. Do not expand width/noise grids or infer semantic selectivity just because generic perturbations cause damage.

## Audit and reproduction

448native/intervention forwards completed. All192layer/document native product writes replay M4 exactly. Production invariants verify uncovered writes stay unchanged and native product norms are nonzero. Independent NumPy audit covers all six conditions on document0 and32: 12full-output comparisons, opposite-half token means, masks, local interventions and write errors. All384record aggregates checked separately. Maximum metric difference8.88e-16, intervention relative difference4.41e-8, table-mean difference0. Local squared-error discrepancy2.46e-10 is explained by production float32 subtraction before double accumulation versus independent double subtraction; the initial1e-10 audit threshold was revised to1e-7 with this documented precision reason, not an experiment change.

Run `probe_token_table_causal.py`, then `audit_token_table_causal.py` in the research Python environment. Requires M4 cached writes and cohort plus the pinned native model. Full audit samples are archived as `emlp_m5_causal_samples.pt` in the primary artifact directory and are not bundled in the packet.

[Pre-run protocol](validation/token_table_causal_protocol.json), [all results](validation/token_table_causal.json), [independent audit](validation/token_table_causal_audit.json), [M4 context](NATURAL_TOKEN_CONTEXT.md).
