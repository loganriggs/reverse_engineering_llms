# Prospective final-position semantic banks in Layers 0 and 1

Banks are frozen on the original nineteen substitutions using only each MLP's write at the final query position and discovery contexts `[0, 1]`. Seventeen declared new substitutions, three contexts per pair, are evaluated on FineWeb and TinyStories. The causal target is the complete final-position MLP change with all other MLP positions held at the base run.

## Locked hypotheses

- **F0_final_bank_transfers: not supported.**
- **F1_correct_family_wins: not supported.**
- **F2_final_bank_beats_sequence_bank: not supported.**
- **F3_native_basis_is_privileged: not supported.**
- **F4_oracle_gap_remains: supported.**
- **F5_ood_retention: supported.**
- **F6_scopes_share_nonrandom_but_distinct_channels: not supported.**
- **F7_final_patch_has_measurable_effect: supported.**

![Final-position banks](layer01_final_position_banks.png)

| Layer | Corpus | Correct / wrong / advantage (win) | Sequence bank | Global | Rotated | Oracle | Local correct | Target Fisher energy |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | fineweb | 0.154 / 0.133 / 0.015 (0.53) | 0.162 | 0.168 | 0.083 | 0.751 | 0.115 | 2.41e-06 |
| 0 | tinystories | 0.180 / 0.051 / 0.116 (0.65) | 0.106 | 0.062 | 0.076 | 0.729 | 0.104 | 1.29e-06 |
| 1 | fineweb | 0.065 / 0.128 / -0.034 (0.43) | 0.104 | 0.140 | 0.102 | 0.763 | 0.087 | 1.67e-05 |
| 1 | tinystories | 0.064 / 0.100 / -0.009 (0.47) | 0.064 | 0.089 | 0.089 | 0.745 | 0.096 | 6.46e-06 |

Final/sequence bank Jaccard by family:

- Layer 0: sentiment=0.032, gender=0.089, tense=0.082, increasing_scale=0.045, polarity=0.056
- Layer 1: sentiment=0.045, gender=0.030, tense=0.111, increasing_scale=0.032, polarity=0.045

This separates a position-local feature claim from the earlier distributed-sequence circuit claim. High oracle but weak family transfer would mean native channels are causally useful without supporting reusable semantic banks at the query position.
