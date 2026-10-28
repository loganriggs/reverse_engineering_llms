# Scale and power red team for semantic attention-head causality

The six frozen candidates are retested on larger, later FineWeb calibration and evaluation panels. Exact head-write removal is scaled from 10 to 100 percent. Every scale uses the same examples, while the causal sign is independently fixed on calibration data. This distinguishes a nonlinear whole-head-ablation failure from a genuine lack of stable semantic causal specialization.

Maximum native-branch error: `6.34e-05`.

## Locked hypotheses

- **R0_manual_native_matches_checkpoint: supported.**
- **R1_small_ablation_recovers_probe_roles: not supported.**
- **R2_selected_heads_beat_controls_locally: not supported.**
- **R3_causal_role_is_scale_stable: not supported.**
- **R4_full_ablation_null_was_low_power: not supported.**
- **R5_broad_head_semantic_ontology_remains_falsified: supported.**

| Candidate | AUROC at 0.1/0.25/0.5/1.0 | Rank at 0.1/1.0 | Calibration sign stable |
|---|---:|---:|---:|
| L0 H8 `numeric` | 0.475/0.480/0.487/0.494 | 9/6 | True |
| L0 H6 `opening_delimiter` | 0.555/0.554/0.551/0.536 | 2/5 | True |
| L1 H7 `negation` | 0.506/0.505/0.498/0.497 | 7/5 | True |
| L1 H6 `non_ascii` | 0.465/0.460/0.547/0.564 | 9/2 | False |
| L2 H1 `dash` | 0.465/0.462/0.461/0.466 | 7/8 | True |
| L3 H4 `numeric` | 0.536/0.534/0.528/0.515 | 4/4 | True |

## Guardrails

Scaling the same ablation checks local-versus-large intervention behavior but does not create an infinitesimal derivative: numerical disagreement and class-score noise become relatively more important at 10 percent. Larger panels improve sign calibration for rare labels but the six candidates remain selected from earlier exploratory probes. A stable source-routing role can survive even if this broad output-class role fails.
