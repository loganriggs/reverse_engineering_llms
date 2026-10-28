# Task-blind MLP multiview atlas

The same natural FineWeb sites are organized three ways after a frozen rank-64 compression: canonicalized native product allocation, exact residual write, and channelwise next-token CE allocation. Clustering uses no task labels or decoded text. Coarse labels and examples are inspected only after all coordinates and memberships are frozen.

## Locked hypotheses

- **A0_discovery_is_task_blind: supported.** Observed: `{'views': ['canonical_allocation', 'exact_write', 'ce_allocation'], 'semantic_fields_used_after_selection_only': True}`
- **A1_model_derived_clusters_recur: supported.** Observed: `{'layers_beating_null': {'canonical_allocation': 8, 'exact_write': 8, 'ce_allocation': 8}}`
- **A2_one_cluster_ontology_spans_views: not supported.** Observed: `{'layer_median_ari': {'0': 0.22831234328647396, '1': -0.01039715051626213, '3': -0.004384075249543678, '6': 0.005685055250776707, '8': 0.022146130693651938, '12': 0.12091691662930816, '16': 0.013672986328168492, '17': 0.12773176684786672}}`
- **A3_clusters_are_not_coarse_label_restatements: supported.** Observed: `{'passing': 20, 'cells': 24}`
- **A4_discrete_codes_beat_continuous_subspaces: not supported.** Observed: `{'cluster_wins': 0, 'cells': 24}`
- **A5_k4_is_a_privileged_vocabulary: not supported.** Observed: `{'passing': 4, 'cells': 24}`
- **A6_replicated_unlabeled_candidates_exist: supported.** Observed: `{'eligible': 51, 'selected': 12}`

## Primary k=4 cells

| Layer | View | Center cosine/null p95 | Margin D/R | Cluster/PCA fidelity | Max coarse-label ARI | Rank-64 energy D/R |
|---:|---|---:|---:|---:|---:|---:|
| 0 | canonical_allocation | 0.785/0.220 | 0.451/0.480 | 0.231/0.483 | 0.398 | 0.515/0.194 |
| 0 | exact_write | 0.778/0.283 | 0.547/0.506 | 0.241/0.503 | 0.351 | 0.750/0.528 |
| 0 | ce_allocation | 0.565/0.165 | 0.219/0.172 | 0.391/0.608 | 0.039 | 0.915/0.027 |
| 1 | canonical_allocation | 0.802/0.202 | 0.410/0.445 | -0.014/0.375 | 0.265 | 0.557/0.092 |
| 1 | exact_write | 0.842/0.193 | 0.428/0.462 | -0.064/0.374 | 0.240 | 0.699/0.349 |
| 1 | ce_allocation | 0.257/0.125 | 0.510/0.107 | 0.311/0.536 | 0.006 | 0.893/0.022 |
| 3 | canonical_allocation | 0.563/0.230 | 0.328/0.381 | -0.135/0.310 | 0.268 | 0.458/0.066 |
| 3 | exact_write | 0.891/0.195 | 0.337/0.405 | -0.199/0.281 | 0.172 | 0.634/0.293 |
| 3 | ce_allocation | 0.251/0.144 | 0.314/0.176 | 0.183/0.446 | 0.011 | 0.841/0.026 |
| 6 | canonical_allocation | 0.943/0.183 | 0.246/0.339 | 0.422/0.696 | 0.028 | 0.545/0.171 |
| 6 | exact_write | 0.637/0.114 | 0.222/0.440 | 0.176/0.515 | 0.055 | 0.679/0.339 |
| 6 | ce_allocation | 0.755/0.149 | 0.246/0.300 | 0.350/0.618 | 0.074 | 0.782/0.088 |
| 8 | canonical_allocation | 0.653/0.212 | 0.333/0.396 | -0.200/0.275 | 0.056 | 0.496/0.048 |
| 8 | exact_write | 0.841/0.269 | 0.305/0.339 | -0.181/0.348 | 0.095 | 0.612/0.208 |
| 8 | ce_allocation | 0.386/0.178 | 0.244/0.180 | -0.090/0.330 | 0.014 | 0.724/0.028 |
| 12 | canonical_allocation | 0.643/0.176 | 0.386/0.348 | 0.027/0.438 | 0.180 | 0.504/0.063 |
| 12 | exact_write | 0.809/0.192 | 0.347/0.365 | -0.037/0.404 | 0.107 | 0.618/0.207 |
| 12 | ce_allocation | 0.582/0.161 | 0.251/0.249 | 0.042/0.488 | 0.071 | 0.693/0.056 |
| 16 | canonical_allocation | 0.938/0.411 | 0.399/0.529 | 0.522/0.836 | 0.147 | 0.740/0.496 |
| 16 | exact_write | 0.941/0.052 | 0.472/0.567 | 0.636/0.918 | 0.153 | 0.951/0.897 |
| 16 | ce_allocation | 0.792/0.129 | 0.432/0.243 | 0.350/0.679 | 0.153 | 0.843/0.494 |
| 17 | canonical_allocation | 0.933/0.413 | 0.433/0.490 | 0.454/0.838 | 0.119 | 0.909/0.810 |
| 17 | exact_write | 0.954/0.444 | 0.451/0.441 | 0.532/0.910 | 0.142 | 0.986/0.969 |
| 17 | ce_allocation | 0.937/0.242 | 0.285/0.300 | 0.150/0.510 | 0.178 | 0.852/0.539 |

## Cross-view agreement

| Layer | Allocation↔write | Allocation↔CE | Write↔CE |
|---:|---:|---:|---:|
| 0 | 0.874 | 0.228 | 0.215 |
| 1 | 0.647 | -0.011 | -0.010 |
| 3 | 0.459 | -0.013 | -0.004 |
| 6 | 0.374 | -0.002 | 0.006 |
| 8 | 0.346 | 0.022 | 0.001 |
| 12 | 0.431 | 0.121 | 0.083 |
| 16 | 0.839 | 0.014 | 0.012 |
| 17 | 0.412 | 0.128 | 0.119 |

## Frozen candidates

| Candidate | Center cosine | D/R size | Member cosine | Strongest post-hoc enrichment |
|---|---:|---:|---:|---|
| L16 exact_write C0 | 0.971 | 80/100 | 0.872 | current:specialized (0.03, 1.9x) |
| L17 exact_write C3 | 0.985 | 62/92 | 0.782 | next:other (0.57, 1.7x) |
| L16 canonical_allocation C1 | 0.974 | 81/102 | 0.788 | current:specialized (0.03, 1.9x) |
| L17 exact_write C2 | 0.980 | 53/73 | 0.752 | next:continuation (0.22, 3.3x) |
| L6 canonical_allocation C2 | 0.986 | 67/65 | 0.745 | next:continuation (0.20, 3.0x) |
| L6 ce_allocation C0 | 0.983 | 90/95 | 0.735 | current:specialized (0.03, 2.0x) |
| L6 canonical_allocation C3 | 0.975 | 77/97 | 0.734 | current:specialized (0.03, 2.0x) |
| L16 exact_write C2 | 0.920 | 64/54 | 0.771 | next:numeric (0.11, 3.6x) |
| L16 ce_allocation C1 | 0.980 | 108/93 | 0.716 | current:specialized (0.04, 2.8x) |
| L16 ce_allocation C0 | 0.987 | 93/110 | 0.710 | loss:q4 (0.55, 2.0x) |
| L0 ce_allocation C1 | 0.994 | 149/194 | 0.685 | next:continuation (0.09, 1.3x) |
| L16 exact_write C3 | 0.882 | 69/50 | 0.766 | current:function (0.42, 1.5x) |

## Guardrails

Stable clusters can quantize a continuous density; the cluster-versus-PCA and k-sweep tests address that directly. Canonical allocation fixes bilinear channel scale/sign with each Down column, exact write removes channel-coordinate choice entirely, and CE allocation is prediction-relative. Disagreement is therefore evidence for purpose-relative ontologies, not automatically polysemanticity. The selected JSON examples are post-hoc interpretation material and do not validate a semantic name.
