# Distance-matched causal test of attention source roles

Eight stable head/source-feature enrichments were frozen from earlier FineWeb panels and tested on a later untouched split. At each eligible destination, all edges from matching source tokens are removed from one head. Eight controls remove equal numbers of sources from the same relative-distance bins in the same sequence. Both the immediate bilinear-operator sketch and the exact final-logit KL are recomputed.

Maximum native-branch error: `6.34e-05`.

## Locked hypotheses

- **E0_edge_intervention_reproduces_native: supported.**
- **E1_source_features_have_privileged_local_effect: supported.**
- **E2_source_features_win_sitewise: not supported.**
- **E3_source_features_have_privileged_downstream_effect: supported.**
- **E4_source_features_explain_material_head_effect: not supported.**
- **E5_semantic_routing_survives_distance_control: supported.**

| Candidate source role | Sites | Local effect ratio / win | Downstream KL ratio / win | Local/downstream fraction of full head |
|---|---:|---:|---:|---:|
| L0 H8 reads `opening_delimiter` | 128 | 14.04 / 0.66 | 3.08 / 0.72 | 0.20/0.09 |
| L1 H6 reads `non_ascii` | 128 | 19.93 / 0.38 | 5.70 / 0.41 | 0.36/0.07 |
| L1 H5 reads `modal` | 128 | 1.46 / 0.51 | 3.36 / 0.51 | 0.09/0.07 |
| L2 H0 reads `opening_delimiter` | 128 | 4.81 / 0.80 | 2.98 / 0.80 | 0.25/0.34 |
| L2 H4 reads `colon_semicolon` | 128 | 5.59 / 0.73 | 3.82 / 0.73 | 0.07/0.05 |
| L3 H7 reads `numeric` | 128 | 1.32 / 0.41 | 1.32 / 0.51 | 0.16/0.17 |
| L4 H1 reads `programming_keyword` | 128 | 2.42 / 0.52 | 2.92 / 0.51 | 0.04/0.01 |
| L5 H4 reads `suffix_s` | 128 | 1.84 / 0.60 | 3.01 / 0.59 | 0.08/0.09 |

## Guardrails

Controls match source count and coarse relative-distance bin, not exact token frequency, overlapping lexical categories, or value direction. Squared local sketch change and KL are effect magnitudes, so they establish privileged routing but not whether the source feature is promoted or suppressed. Candidates were chosen from stable enrichment on earlier panels; this split is prospective for causal effect. A failure can mean the enrichment is descriptive routing with small behavioral consequence, not that the measured QK preference was absent.
