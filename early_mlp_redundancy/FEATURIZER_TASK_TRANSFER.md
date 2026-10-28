# M9: causal-supervision gains transfer unevenly across tasks

2026-10-28. Eight M8 models frozen; no additional fitting, selection or tuning. Evaluate all80previously opened training pairs/160endpoints across10families. These tasks were absent from M7/M8 featurizer fitting, but previously examined throughout the research. This is representation transfer to existing development tasks, not pristine whole-task validation or new-task discovery. Original sequence_copy, spatial_inverse, weekday_successor, role_binding and tense exclusions remain unused; lexical past_tense here is distinct from excluded tense-operation tasks.

Every position's native MLP1 product write is reconstructed and Down bias retained. Evaluate full50304logit distributions, prompt-continuation CE, answer-group logsumexp margins, binary answer preference and full-vocabulary top1 correctness. All failures retained, including native errors. Aggregate below weights endpoints equally, not tokens; family sizes differ. Equal-family macros are also reported.

## Overall transfer

Native binary answer accuracy98.75%, native task top1 correctness86.25%. These differ from native top1 agreement, which measures whether reconstruction preserves the model's prediction regardless of correctness.

| Model / seed | Objective | Output KL | Binary task accuracy | Task top1 correctness | Both endpoints correct |
|---|---|---:|---:|---:|---:|
| scalar / 71 | reconstruction | 0.1302 | 91.88% | 59.38% | 87.50% |
| scalar / 71 | downstream_KL | 0.1068 | 93.12% | 65.62% | 87.50% |
| block / 71 | reconstruction | 0.1722 | 95.62% | 57.50% | 92.50% |
| block / 71 | downstream_KL | 0.1492 | 95.00% | 65.00% | 90.00% |
| scalar / 72 | reconstruction | 0.1535 | 89.38% | 56.88% | 81.25% |
| scalar / 72 | downstream_KL | 0.1221 | 96.25% | 65.62% | 93.75% |
| block / 72 | reconstruction | 0.1925 | 89.38% | 56.88% | 81.25% |
| block / 72 | downstream_KL | 0.1478 | 95.00% | 65.00% | 93.75% |

Output KL improves13–23% under downstream training in all four model comparisons, smaller than the59–63%natural-text improvement. Task top1 correctness improves in all four, but binary accuracy, pair correctness and margin fidelity do not improve uniformly: block71 loses binary/pair accuracy and worsens mean absolute margin error. Natural-text prompt-CE improvement also does not imply improvement in CE of these authored prompt continuations.

## Every family retained

Percent reduction in mean output KL, continued-MSE → downstream-KL training. Negative means worse:

| Family | Scalar71 | Scalar72 | Block71 | Block72 |
|---|---:|---:|---:|---:|
| agreement | +9.3% | +21.3% | +28.7% | +9.8% |
| sentiment | +26.8% | +48.6% | +23.9% | +53.4% |
| entity_binding | +46.4% | +42.8% | +37.9% | +34.7% |
| pronoun_gender | +3.9% | +1.3% | +22.5% | +4.7% |
| capital_recall | -23.0% | -73.0% | -55.1% | -4.5% |
| antonym | +21.9% | +34.5% | +17.2% | +29.3% |
| plural_noun | +9.9% | -21.7% | -26.2% | -41.5% |
| past_tense | +24.8% | -11.7% | +12.0% | +19.2% |
| country_language | +27.8% | +10.8% | -2.4% | +18.7% |
| number_successor | -12.6% | -2.3% | -42.0% | -35.7% |

Capital recall and number successor worsen in all four all-position KL comparisons while retaining100%binary accuracy. Capital recall is the stronger task-level failure: final-position KL and answer-margin error worsen in all four, and full-vocabulary task top1 correctness falls or stays at zero (scalar71 .50→.375, scalar72 .0→.0, block71 .0→.0, block72 .125→.0). Number successor's final-position KL does NOT worsen in all four; block72 improves at the final position despite worse overall KL. Do not infer the same mechanism from the shared overall-KL sign.

Antonym remains weak despite improvements: final task top1 ranges12.5–37.5%under causal training. Each small lexical family has only4pairs, so percentages are coarse and uncertainty large. Family outcomes are development evidence for coverage gaps, not reliable population rankings.

Equal-family macro mean KL also improves in all four comparisons, but unevenly:

| Model / seed | MSE macro KL | Downstream-trained macro KL |
|---|---:|---:|
| scalar71 | 0.1444 | 0.1201 |
| scalar72 | 0.1561 | 0.1409 |
| block71 | 0.1829 | 0.1797 |
| block72 | 0.2017 | 0.1688 |

## Research consequence

Causal training on natural text helps the average task distribution but does not preserve every competent behavior. It is reasonable to broaden causal training coverage; it is not justified to declare the learned blocks useful manifolds or new computational units yet. The objective and the distribution supplying it both matter.

Next design a bounded diverse-supervision comparison using the already-approved training families. Audit lexical source groups and context duplicates before splitting: paired endpoints and repeated source-token pairs must stay together for lexical-transfer claims, and a family with only one lexical group cannot support such a claim. Freeze family balancing, training budget and evaluations before fitting. Compare added diverse task KL supervision against continued natural-text KL from identical checkpoints, retaining natural-text evaluation to detect forgetting. Keep original excluded families reserved and explicitly label all reused development data. Do not cherry-pick capital/antonym examples into training and then present their fitted performance as transfer.

## Validation and reproduction

1440forwards completed (160native+1280reconstructions). Native answer margins replay the earlier D1 bank exactly within the asserted tolerance. Active coordinate count32in every reconstruction. Independent NumPy scores80saved full-output samples (one endpoint per family×8models), including margins, binary and top1 answers and localwriteerrors. Maximum metric discrepancy6.82e-8, reflecting float32 agreement means in production. All1280rows/88summaries and equal-family macros independently aggregated. No fresh model fitting or original task-holdout reads.

Run `probe_featurizer_task_transfer.py` and `audit_featurizer_task_transfer.py`. Requires M8 states and the bundled task metadata. Full audit logits are archived as `emlp_m9_task_transfer_samples.pt` in primary artifacts, not bundled here.

[Protocol](validation/featurizer_task_transfer_protocol.json), [all family/pair results](validation/featurizer_task_transfer.json), [audit](validation/featurizer_task_transfer_audit.json), [M8](CAUSAL_FEATURIZER_OBJECTIVE.md).
