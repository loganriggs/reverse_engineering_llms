# Transfer test for selecting among random rotated bases

The earlier best-of-seven control chose a different rotated basis after seeing each held-out example's downstream fidelity. Here one rotation must be frozen using other lexical pairs, or selected on FineWeb and transferred unchanged to TinyStories. Native and rotated programs retain the same rank-256 execution budget; only the basis-selection information changes.

## Locked hypotheses

- **T0_every_fixed_rotation_is_weaker_in_the_median: supported.**
- **T1_leave_one_pair_out_rotation_selection_does_not_erase_native_privilege: supported.**
- **T2_fineweb_selected_rotation_does_not_erase_ood_native_privilege: not supported.**
- **T3_per_example_oracle_is_strictly_more_powerful_than_frozen_selection: supported.**
- **T4_rotation_choice_is_not_cross_corpus_stable: supported.**

| Layer | Corpus | Native / best fixed rotation | Native−LOPO rotation median (95% CI) | Oracle-best−LOPO median (95% CI) | Best fixed seed |
|---:|---|---:|---:|---:|---:|
| 0 | fineweb | 0.814/0.706 | +0.110 (+0.087, +0.256) | +0.204 (+0.086, +0.282) | 20261412 |
| 0 | tinystories | 0.801/0.746 | +0.035 (+0.017, +0.152) | +0.056 (+0.002, +0.190) | 20261614 |
| 1 | fineweb | 0.837/0.728 | +0.177 (+0.096, +0.201) | +0.157 (+0.080, +0.200) | 20261513 |
| 1 | tinystories | 0.746/0.694 | +0.056 (+0.004, +0.154) | +0.197 (+0.074, +0.308) | 20261715 |

FineWeb-selected basis transferred to TinyStories:

- Layer 0: seed 20261412, native advantage +0.130 (+0.084, +0.188).
- Layer 1: seed 20261513, native advantage +0.028 (-0.019, +0.201).

A per-example best-of-seven result answers whether a larger adaptive dictionary contains a coordinate system well suited to each edit. The frozen-selection result answers whether one rotated coordinate system is a reusable alternative to the native neuron basis. These are different definitions of basis privilege and should not be merged.
