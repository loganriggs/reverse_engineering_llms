# M13: partial causal correspondence, not a stable task-unit dictionary

2026-10-28. Frozen mixed-trained scalar/block models, seeds71/72. One role-training pair per each10families chosen by minimumSHA256(M13:+pairID), without outcomes. All14source-separated evaluation pairs remain evaluation-only for matching. This gives20training and28evaluation endpoints. Original task exclusions remain unused.

## Intervention and controls

At each endpoint, first compute the intact featurizer reconstruction. Remove each group's decoded contribution at every position while holding every other code/gate fixed and retaining decoder mean/bias and native Down bias. Compare outputs to THAT intact reconstructed model. This is not deletion of native MLP neurons, and all conclusions are conditional on an imperfect reconstruction.

All32groups are evaluated: actual block groups, the same block models' fixed shuffled atom groups, and arbitrary4atom groupings of scalar models. No group filtering or candidate selection. Report actual active-coordinate cost and removed energy. These are not state-by-state energy-matched interventions; different group coordination can change removal energy. Zero-contribution groups produce explicit no-op records.

Signed answer-margin loss(intact minus ablated) is averaged within each family, yielding32×10training signatures and32×6evaluation signatures. Function-based Hungarian matching maximizes cosine of training signatures across seeds; compare with M12geometry-based matching and one fixed random assignment. All three fit a nonnegative scalar amplitude per matched group on training signatures ONLY, then predict seed72evaluation effects from seed71effects. No evaluation fitting or assignment choice.

## Cross-seed causal prediction

Relative squared error uses the squared seed72evaluation effects as denominator. Predicting zero effect gives1; lower is better. It is NOT centered variance explained or fraction of model computation understood.

| Grouping | Matching | Training mean cosine | Evaluation relative squared error | Evaluation flattened cosine | Mean per-group evaluation cosine |
|---|---|---:|---:|---:|---:|
| scalar / contiguous | functional | 0.666 | 0.968 | 0.384 | 0.314 |
| scalar / contiguous | geometric | 0.132 | 0.978 | 0.215 | 0.009 |
| scalar / contiguous | random | 0.036 | 0.980 | 0.175 | 0.134 |
| block / contiguous | functional | 0.670 | 0.881 | 0.431 | 0.245 |
| block / contiguous | geometric | 0.254 | 0.551 | 0.689 | 0.115 |
| block / contiguous | random | 0.081 | 0.907 | 0.306 | 0.087 |
| block / shuffled | functional | 0.673 | 1.110 | 0.401 | 0.397 |
| block / shuffled | geometric | 0.205 | 0.997 | 0.222 | 0.126 |
| block / shuffled | random | 0.124 | 0.996 | 0.218 | 0.114 |

Actual block geometry matching predicts aggregate evaluation effects better(error.551) than random(.907) and function-based matching(.881). Shuffled groups are near zero-predictor performance under geometry/random matching(.997/.996), and scalar arbitrary groups are also near it(.978/.980). This supports some cross-seed causal correspondence in the actual learned grouping despite its low absolute geometric overlap.

Function matching achieves training cosine~.67 for actual, scalar and shuffled groups alike, but transfers much less impressively. Optimizing correspondence on one pair per family can match broad sensitivity patterns without recovering stable mechanisms. This experiment cannot distinguish all causes of that failure. Mean per-group cosine tells another part of the story: actual blocks' functionalmatch.245 exceeds geometricmatch.115, even though geometry wins energy-weighted prediction. Therefore the geometric advantage is not uniform across groups.

## Concentration sensitivity, explicitly post-result

For actual block targets, antonym contributes59.8%of evaluation squared effect energy; its evaluation comprises only ONE pair. Two target groups contribute57.8%of the energy across families. This concentration limits interpretation of the aggregate.

Without changing any matching or amplitude, omitting the largest-energy family gives relative errors: geometry.682, function.724, random1.058. Equal-family mean normalized errors are.723/.812/1.021 respectively. Geometry's advantage survives these descriptive checks but shrinks. These are outcome-informed sensitivity calculations, not fresh confirmation or grounds to select those high-effect groups for later claimed validation.

## What is supported, and the next discriminator

Some geometric correspondence carries transferable causal information. We have not recovered a clean dictionary of stable task units: individual block agreement is weak, causal effects are concentrated, small evaluation families dominate some metrics, and scalar/shuffled group controls are not energy-matched per state. Margin effects may reflect generic disruption rather than semantic selectivity. Function-level equivalence can still span multiple groups rather than one-to-one matches.

Next use the EXISTING cached effects to test a simple removed-energy predictor before calling group roles selective. Fit any coefficients on the10family training signatures only; score the same evaluation signatures without refitting/matching. Compare with the fixed geometry/functional predictions. This asks how much apparent causal-role prediction can be explained by local error magnitude and common downstream sensitivity. No new group/seed sweep, model fitting or native forwards is needed. If stronger role evidence survives, design independent contextual/task tests rather than repeatedly selecting on the14pair development panel.

## Reproduction and audit

9216group-intervention records completed, including683exactnoops. 8773actualforwards:48native+192intact reconstructed+8533nonzero ablations. Independent NumPy checks24saved sample group decodes and full-output score calculations across training/evaluation, all record uniqueness/noops,12family-signature matrices and9train-amplitude/evaluation-prediction receipts. Maximum metric error9.27e-16, group-decode discrepancy5.55e-8relative, prediction discrepancy4.45e-16. Full native-model/optimizer implementations are not independently recreated.

Run `probe_group_roles.py`, `analyze_group_roles.py`, `audit_group_roles.py`; the separately labeled concentration diagnostic is `analyze_group_role_concentration.py`. Full audit samples are archived as `emlp_m13_group_role_samples.pt` in primary artifacts.

[Pre-run protocol](validation/group_roles_protocol.json), [all ablations](validation/group_roles.json), [matching predictions](validation/group_role_matching.json), [audit](validation/group_roles_audit.json), [post-result concentration](validation/group_role_concentration.json), [M12 geometry](BLOCK_STRUCTURE_AUDIT.md).
