# Cross-initialization alignment in task-margin coordinates

This adaptive red-team repeats the answer-vector analysis after projecting every exact channel effect onto the shared correct-answer-minus-mean-alternative margin. It tests the narrower ontology that a reusable native channel is a signed discriminator across facts, even when its collateral eight-logit write differs.

## Locked hypotheses

- **M0_complete_margin_discrimination_recurs: not supported.**
- **M1_margin_effect_subspace_recurs: not supported.**
- **M2_individual_margin_verbs_recur: not supported.**
- **M3_margin_coalition_selector_recurs: not supported.**
- **M4_margin_alignment_tracks_interface_fidelity: not supported.**
- **M5_margin_channel_matching_transfers: not supported.**

| Seed | Gate | Full margin cos | Subspace 16/32/64/128 | Channel cos disc/eval | Random p95 | Site top32 |
|---:|---|---:|---:|---:|---:|---:|
| 101 | pass | 0.076 | 0.074/0.140/0.255/0.504 | 0.165/0.001 | 0.007 | 0.188 |
| 102 | fail | 0.070 | 0.072/0.130/0.255/0.504 | 0.169/0.001 | 0.009 | 0.182 |
| 103 | fail | 0.085 | 0.070/0.138/0.263/0.507 | 0.164/0.003 | 0.011 | 0.190 |
| 104 | pass | 0.109 | 0.070/0.136/0.266/0.513 | 0.168/0.014 | 0.010 | 0.189 |
| 105 | fail | 0.110 | 0.053/0.113/0.242/0.501 | 0.163/0.008 | 0.007 | 0.188 |
| 106 | fail | 0.066 | 0.061/0.123/0.250/0.505 | 0.168/-0.005 | 0.008 | 0.183 |

This is an adaptive red-team prompted by failure of the preregistered answer-vector alignment, not an independent confirmatory result. Failure still cannot exclude a rotated distributed basis or a different semantic perturbation; success would show only recurring margin discrimination.
