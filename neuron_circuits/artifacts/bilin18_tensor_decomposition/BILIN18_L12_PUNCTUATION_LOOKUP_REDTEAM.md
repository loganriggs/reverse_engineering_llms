# L12 punctuation chart: token-lookup and repeated-row red team

The frozen punctuation axes are tested against source-token lookup, within-token distance discrimination, and a one-edge-per-FineWeb-row selection rule.

## Locked hypotheses

- **T0_axes_are_frozen_from_discovery: supported.** Observed: `{}`
- **T1_source_token_lookup_is_insufficient: supported.** Observed: `{'r2': [-0.10024151138790716, -0.052751835733967356]}`
- **T2_axis0_tracks_distance_within_token: supported.** Observed: `{'auc': 0.7122252747252747, 'groups': 203, 'pairs': 13104}`
- **T3_profiles_survive_one_edge_per_row: supported.** Observed: `{'mean_cosine': 0.9399824413146316, 'cosines': [0.9677892895558394, 0.973308657594378, 0.8461245125623684, 0.9727073055459403]}`
- **T4_axis0_separates_local_and_long_range_poles: supported.** Observed: `{'positive_median': 2.0, 'negative_median': 22.5}`

| Axis | Token-lookup R2 | Within-token long-range AUROC | + pole distance ordinary/unique | - pole distance ordinary/unique | Profile cosine +/- |
|---:|---:|---:|---:|---:|---:|
| 0 | -0.100 | 0.712 | 3.0/2.0 | 42.5/22.5 | 0.968/0.973 |
| 1 | -0.053 | 0.575 | 3.0/2.0 | 5.5/2.5 | 0.846/0.973 |

## Guardrail

A within-token distance signal rejects literal source-token lookup, but does not uniquely identify a punctuation concept. Distance, repeated delimiter structure, and document layout remain entangled in this observational chart.
