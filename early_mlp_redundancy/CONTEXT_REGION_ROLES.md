# M16: restricting interventions to task tokens does not restore stable roles

2026-10-28. The fixed prefix/task localization experiment is complete. Most answer-margin effect energy lies on the original task tokens, and geometric block correspondence remains partial there. The large fitted-gain failure moves between regions across the two preambles. Dropping prefix interventions is therefore not a general solution.

## Design and verification

Reuse all 27 M15 native-passing pairs (54 endpoints), frozen four M11 mixed-supervision featurizers, scalar contiguous/block contiguous/block shuffled groups, all 32 groups and M13 matching/amplitudes. The exercise plural cohort remains excluded by the prior native competence gate and its failure remains reported in M15. No new task or untouched holdout claim.

Before running, verify every input token sequence equals six prefix token IDs followed by exactly the original task token IDs. Freeze complementary masks. For each group, separately remove its decoded contribution from prefix positions or task positions, holding all other codes/gates/biases fixed. Compare to intact reconstruction, and reuse M15 all-position interventions as the joint reference. Replayed intact answer margins match M15 within 1e-8.

The run used 19,630 actual forwards: 54 native, 216 intact and 19,360 nonzero region ablations. There are 20,736 region records including 1,376 exact no-ops. No restarts, refits, gain clipping or group selection.

## Where do the effects occur?

For actual block groups, squared margin-effect energy over individual endpoint/group interventions, divided by the all-position energy:

| Context | Seed | Prefix | Task | Interaction |
|---|---:|---:|---:|---:|
| exercise | 71 | 0.085 | 0.831 | 0.0040 |
| exercise | 72 | 0.130 | 0.918 | 0.0115 |
| example | 71 | 0.190 | 0.757 | 0.0062 |
| example | 72 | 0.248 | 0.779 | 0.0164 |

These are squared-norm ratios, **not additive fractions of responsibility**. Interaction is the finite margin effect of removing both regions minus the sum of their separate effects. Its squared norm is small in aggregate (.4–1.6% of whole effect energy for actual blocks), but that does not establish linearity for every group or larger interventions. KL effects are reported separately and never treated as additive.

Task effects have greater energy than prefix effects in every model/grouping/seed condition. This does not imply greater per-token importance: region lengths differ, and task positions are closer to the answer. It does show that the previous aggregate effects were not mostly direct damage from deleting the newly added prefix region.

## Does task-only intervention restore correspondence?

Frozen geometric cross-seed relative squared prediction error, zero predictor=1:

| Grouping | Exercise prefix | Exercise task | Example prefix | Example task |
|---|---:|---:|---:|---:|
| scalar contiguous | 0.879 | 1.031 | 1.120 | 1.067 |
| block contiguous | 1.051 | 0.840 | 0.751 | 0.747 |
| block shuffled | 1.079 | 1.121 | 1.309 | 1.170 |

Actual-block task-only errors .840/.747 are close to M15 all-position .854/.726. This is not a return to the stronger original-context result. Equal-family task-only errors are .978/.866; without antonyms they are .950/.865. Antonym contributes 64%/58% of task-only target energy. Thus the task-region advantage is limited, especially for exercise, rather than uniform evidence for stable task units.

Scores average effects within each family before predicting them; their weighting differs from the individual-endpoint energy table. The frozen amplitudes were fitted on all-position training interventions, not these regions. Failure therefore tests transport of that entire predictor, not whether any new region-specific fit could succeed. We deliberately do not fit such a rescue predictor.

## Calibration failure changes location

The unchanged functional match has source group15→target29 and amplitude93.841. Post-result error accounting finds:

- Exercise: both matched prefix role vectors are zero. Task-only functional error is9.683, with 89.0% of its error from this pair.
- Example: prefix-only functional error is57.413, with 97.5% from this pair. Task-only error is1.519, and the pair supplies only0.23% of that error.

The large source response is task-local in one context and prefix-local in the other. This does not establish that an underlying semantic computation migrated: group occupancy, fixed matching, downstream sensitivity and imperfect reconstructed baselines all remain possible contributors. It does establish that a fixed large gain learned from nearly silent training effects is not portable by simply excluding one input region. No gain repair or cherry-picked group removal was applied.

## Consequence for the featurizer loop

Close this localization detour. Current blocks have some reproducible causal correspondence but are not validated semantic manifolds. Broader context coverage is the next deliverable, rather than another matching or gain variant on these 14 lexical evaluation pairs.

Prepare a fixed, metadata-audited context expansion across all ten allowed training task families. Preserve the existing source-token split and original whole-task exclusions, and reserve entire context templates for evaluation. Screen native competence before training or selecting intervention targets; retain failed cohorts instead of replacing prompts. Freeze the expansion and a matched training comparison before outcomes. Keep raw reconstruction, native-output fidelity, task prediction, and intervention fidelity separate.

Both prefix and task regions must remain in the causal validation set. The present role intervention acts on imperfect reconstructed outputs; before claiming native computational units, explicitly compare corresponding interventions on the native model as well. This is a requirement for the next training design, not evidence supplied by M16. New-task discovery and held-out-combination structure remain unachieved. Stop after the fixed context comparison and its causal validation; no further predictor sweeps on this panel.

## Audit and reproduction

```bash
python probe_context_region_roles.py --prepare-only
python probe_context_region_roles.py
python analyze_context_region_roles.py
python audit_context_region_roles.py
python audit_context_region_analysis.py
python analyze_region_calibration.py
```

The probe refuses to overwrite completed results or alter its frozen protocol. Large samples/checkpoints remain in the primary artifacts directory. The packet includes scripts, protocol, all metrics and audit receipts.

Independent NumPy checks cover24 sampled masked decoder contributions/full-output metrics, all20,736 unique intervention records/no-ops, all216 native/intact replay baselines, and complementary region masks. Maximum sampled metric error1.99e-8; masked decoding relative error5.77e-8. A separate scalar-loop audit checks all12 energy/KL summaries,36 family matrices and36 frozen prediction scores. This is sampled numerical replay, not independent rerunning of every model intervention. Post-result calibration accounting is descriptive and not new confirmation.

[Protocol](validation/context_region_protocol.json), [all interventions](validation/context_region_roles.json), [analysis](validation/context_region_analysis.json), [intervention audit](validation/context_region_audit.json), [analysis audit](validation/context_region_analysis_audit.json), [calibration accounting](validation/context_region_calibration.json).
