# Cross-pair latent-variable capability test

A cue pair is swapped before the latest boundary, while the output contrast is built only from disjoint pairs in the same declared family. The two folds reverse cue and target sets. Spelling is the positive control. Thirty-one frozen balanced signings of the target tokens test contrast specificity.

## Locked results

- **C0_all_cells_are_balanced: supported.**
- **C1_spelling_positive_control_transfers: supported.**
- **C2_human_semantic_families_are_transferable_variables: not supported.** (3/5 families)
- **C3_declared_family_contrasts_beat_random_signings: supported.**
- **C4_transfer_strength_tracks_short_terminal_programs: not supported.**

Capability-win versus terminal-k90 correlation: `-0.433`.

| Family/fold/split | N | win/mean | random percentile | endpoint energy |
|---|---:|---:|---:|---:|
| spelling/0/0 | 80 | 0.975/+1.08 | 1.000 | 0.00749 |
| spelling/0/1 | 80 | 0.963/+1.04 | 1.000 | 0.00848 |
| spelling/1/0 | 80 | 0.963/+1.53 | 1.000 | 0.0107 |
| spelling/1/1 | 80 | 0.950/+1.09 | 1.000 | 0.00996 |
| sentiment/0/0 | 80 | 0.800/+0.0961 | 1.000 | 0.00445 |
| sentiment/0/1 | 80 | 0.863/+0.11 | 1.000 | 0.00562 |
| sentiment/1/0 | 80 | 0.787/+0.0851 | 1.000 | 0.00873 |
| sentiment/1/1 | 80 | 0.850/+0.116 | 1.000 | 0.00369 |
| gender/0/0 | 80 | 0.863/+0.932 | 1.000 | 0.0352 |
| gender/0/1 | 80 | 0.838/+0.678 | 1.000 | 0.00969 |
| gender/1/0 | 80 | 0.863/+0.517 | 1.000 | 0.00386 |
| gender/1/1 | 80 | 0.838/+0.444 | 1.000 | 0.00261 |
| tense/0/0 | 80 | 0.762/+0.0931 | 1.000 | 0.00266 |
| tense/0/1 | 80 | 0.688/+0.0958 | 1.000 | 0.005 |
| tense/1/0 | 80 | 0.812/+0.0622 | 1.000 | 0.00401 |
| tense/1/1 | 80 | 0.762/+0.0648 | 1.000 | 0.00207 |
| increasing_scale/0/0 | 80 | 0.388/-0.0106 | 0.000 | 0.00284 |
| increasing_scale/0/1 | 80 | 0.388/-0.00567 | 0.161 | 0.00307 |
| increasing_scale/1/0 | 80 | 0.463/-0.000539 | 0.226 | 0.00224 |
| increasing_scale/1/1 | 80 | 0.425/-0.0142 | 0.194 | 0.00524 |
| polarity/0/0 | 80 | 0.537/+0.0195 | 1.000 | 0.00178 |
| polarity/0/1 | 80 | 0.600/+0.0176 | 1.000 | 0.00285 |
| polarity/1/0 | 80 | 0.738/+0.0462 | 1.000 | 0.0055 |
| polarity/1/1 | 80 | 0.625/+0.0274 | 1.000 | 0.00287 |

## Interpretation boundary

A failed family is not evidence that the model lacks sentiment, gender, tense, scale, or polarity information. It means this particular single-token edit does not establish a transferable family-level variable that predicts disjoint target-token logits at the fixed destination. Without that behavioral capability, broad or unstable terminal mechanisms cannot adjudicate how a successfully computed semantic variable would be represented.
