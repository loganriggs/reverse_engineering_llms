# M15: some geometric role correspondence survives context changes

2026-10-28. Frozen featurizers, matches and training amplitudes were tested under two fixed preambles. Geometrically matched block effects remain partly predictable across seeds, but prediction degrades. This is evidence for limited causal correspondence, not stable semantic units.

## Fixed challenge and competence gate

Append neither new answers nor examples: prepend `A short exercise follows.\n` or `Here is another example.\n` to the existing 14 lexical evaluation pairs. These are familiar development tasks with new preambles, not new-task discovery or untouched evaluation data. The original whole-task exclusions remain unused.

Before ablation outcomes, require each family/context cohort to have at least 75% pairs with both binary answer comparisons correct and 50% endpoints with the correct answer as the native full-vocabulary top choice. Retain every pair in a passing cohort, including individual errors. Require three passing families for an aggregate. No replacement prompt search.

All 12 cohorts pass the binary criterion. Eleven pass both criteria. Plural nouns under the exercise preamble fail full-vocabulary top1 (0%, one pair) and are retained in the competence report but excluded from causal aggregates. Thus exercise has five eligible families/13 pairs, example six families/14 pairs. Native top1 on eligible endpoints is 80.8% and 89.3%, respectively.

Use the four M11 mixed-supervision models (scalar/block, seeds 71/72). Remove each of 32 decoded groups at every position, holding other codes/gates and biases fixed. Compare against intact reconstructed computation, not the native MLP. Scalar contiguous groups and shuffled block groups are controls. Retain the M13 geometry, functional and fixed-random assignments and their training-fitted amplitudes without alteration.

## Causal prediction under context change

Relative squared error predicts seed72 group effects from matched seed71 effects. Zero prediction has error 1; lower is better. This is not the percentage of model computation understood. Compare each new context to the original context restricted to exactly the same passing families.

| Grouping | Matching | Exercise | Original, same families | Example | Original, same families |
|---|---|---:|---:|---:|---:|
| scalar contiguous | functional | 1.154 | 0.945 | 1.199 | 0.968 |
| scalar contiguous | geometric | 1.053 | 0.957 | 1.069 | 0.978 |
| scalar contiguous | random | 0.987 | 0.976 | 1.118 | 0.980 |
| block contiguous | functional | 8.473 | 0.860 | 14.843 | 0.881 |
| block contiguous | geometric | 0.854 | 0.515 | 0.726 | 0.551 |
| block contiguous | random | 8.619 | 0.887 | 15.554 | 0.907 |
| block shuffled | functional | 1.343 | 1.089 | 1.445 | 1.110 |
| block shuffled | geometric | 1.123 | 0.989 | 1.195 | 0.997 |
| block shuffled | random | 1.010 | 0.952 | 1.090 | 0.996 |

Actual-block geometric matching beats zero prediction in both contexts, but degrades from .515 to .854 for exercise and .551 to .726 for example. Scalar and shuffled geometric controls exceed 1. This supports some grouping-dependent correspondence within this small experiment; it does not establish superiority across models or tasks.

Weighting families equally gives geometric errors 1.044 (exercise) and .747 (example). Omitting antonyms gives .934 and .789. Thus the exercise advantage is weak and not uniform across families. These are descriptive sensitivities, not extra independent replications.

## Why the enormous functional/random errors?

A post-result accounting, without refitting, locates 85–92% of their squared error in one source group, 15, matched to target group 29 by both assignments. Its frozen fitted amplitude is 93.841. The source's training role-vector norm was .000431, versus .227 for the target; in the new contexts source norms are .0905/.1357 and target norms .1707/.2026. These vectors have different family counts across train/evaluation, so do not interpret norm ratios as a matched-sample growth estimate. They reveal a near-silent training source with a poorly transferable large gain.

The inflated predictions have 7.7–14.6 times target energy. This is direct evidence of fragile training calibration. It is not evidence that the underlying group is meaningless, nor a license to clip the gain and relabel this test successful. Functional and random matching share this problematic pair, so their failures are not independent evidence. All original scores remain intact.

## Reconstruction remains a material limitation

On the eligible exercise endpoints, intact featurizer replacement has mean native-output KL .119–.184; for example .042–.079. Block seed72 has only 92.3% binary correctness on exercise, although the native model has 100%. Other binary rates are 100%. Reconstructed top1 is 69.2–76.9% for exercise and 71.4–78.6% for example. The original unprefixed evaluation KL was roughly .038–.045 over all six families; that is not a matched-family or matched-length control for these context KLs.

Role interventions describe the reconstructed model. A matched group can therefore have consistent effects while the replacement itself is imperfect. The preamble also adds intervention positions, changes token positions, and changes downstream state. These are confounds for interpreting the degradation as a change in semantic identity.

## Next discriminating experiment and stopping rule

Stop tuning matches or gains on this panel. Decompose the existing intervention into prefix-only and original-task-only group removals, keeping the all-position result as reference. Include every group and the same scalar/shuffled controls, fixed models and passing cohorts; do not select the striking group15 alone. Measure final answer-margin effects and final-position KL, and retain the nonadditive interaction: whole effect minus prefix-only minus task-only. Report reconstruction fidelity separately.

This separates damage caused at newly added positions from changed roles on the original task tokens; it will not by itself separate positional shifts from semantic context. It is one bounded localization test, not another predictor search. Its featurizer deliverable is an explicit choice of intervention sites and targets for any subsequent causal-supervision objective. If correspondence remains weak on task-only effects, do not promote current blocks to semantic manifolds; broaden training/validation contexts before retraining. Preserve whole-task exclusions and reserve genuinely new task discovery as unachieved.

## Reproduction and evidence

Run the scripts in this packet with the research Python/dependencies and pinned native model. Competence and group probes depend on the frozen protocols and local M11 checkpoints. Large logits/checkpoints remain in the primary artifact directory, not this shareable packet.

```bash
python probe_context_competence.py
python audit_context_competence.py
python probe_context_group_roles.py
python analyze_context_role_transfer.py
python audit_context_roles.py
python analyze_context_role_concentration.py
```

The native screen uses 56 forwards. The group test records 10,368 interventions (206 exact no-ops), requiring 10,432 forwards including 54 native and 216 intact baselines. Independent audits check all 56 competence endpoints/12 gates, 24 sampled full-output/group-decode interventions, all record identities/no-ops, 216 native replays, and all 18 frozen matching scores. Maximum sampled metric error is 1.99e-8 and group-decode relative error 5.66e-8. This is sampled output verification, not independent replay of every native forward. The post-result concentration script verifies the cited fitted gain and error accounting.

[Challenge protocol](validation/context_challenge_protocol.json), [competence](validation/context_competence.json), [competence audit](validation/context_competence_audit.json), [intervention protocol](validation/context_group_roles_protocol.json), [all interventions](validation/context_group_roles.json), [transfer scores](validation/context_role_transfer.json), [role audit](validation/context_roles_audit.json), [concentration](validation/context_role_concentration.json).
