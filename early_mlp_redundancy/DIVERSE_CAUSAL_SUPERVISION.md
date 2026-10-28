# M11: diverse causal supervision transfers, with a natural-text tradeoff

2026-10-28. Executes the exact M10protocol: four M8causal checkpoints(scalar/block,seed71/72), each continued for120updates under natural-only KL or mixed natural/task KL. Same initialweights, train normalization, active32, freshAdam.0003 and schedules; all eight runs retained. No resplitting, tuning or checkpoint selection.

Mixed training replaces60of120natural updates with60taskpair updates, exactly6per10families. Pair endpoints receive equal weight; their native-output KL is averaged over their own positions and full50304logit vocabulary. This is causal output-distribution supervision, not answer-label training. Updates/initialweights are matched, tokens/FLOPs are not.

The audited task split has66trainingpairs and14evaluationpairs. No changed source-input token in either side occurs in the other side's task prompts, including distractors. Sixfamilies support evaluation; agreement/sentiment/pronoungender/numbersuccessor aretraining-only. All data remain previously examined development material; this is not pristine confirmation or new-task discovery. Original excluded families stayunused.

## Source-separated task evaluation and natural-text fidelity

Task evaluation contains28endpoints/14pairs. Native binary accuracy100%; native full-vocabulary tasktop1correctness78.57%. Natural evaluation is unchanged32documents. All rows below use fixed final models.

| Model / seed | Natural KL: original → natural-only → mixed | Task KL: original → natural-only → mixed | Task top1: natural-only → mixed |
|---|---|---|---|
| scalar71 | 0.1212 → 0.1154 → 0.1218 | 0.1255 → 0.1146 → 0.0447 | 39.3% → 71.4% |
| scalar72 | 0.1177 → 0.1106 → 0.1139 | 0.1524 → 0.1405 → 0.0404 | 46.4% → 67.9% |
| block71 | 0.1543 → 0.1413 → 0.1520 | 0.2010 → 0.1842 → 0.0441 | 42.9% → 71.4% |
| block72 | 0.1659 → 0.1459 → 0.1612 | 0.1669 → 0.1610 → 0.0377 | 39.3% → 64.3% |

Mixed training reduces taskKL61–77%relative to continued natural-only training, across all four comparisons. Tasktop1correctness rises to64–71%, still below native78.57%. Binary accuracy is100%for three mixed models and96.43%for scalar72. Stronger task fidelity does not mean exact preservation.

Natural-text KL is3–10%worse under mixed than under natural-only continuation. Relative to the original M8checkpoint, mixed scalar71 is about0.5%worse and the other three slightly better. Thus most of the cost is foregone natural-only improvement, not large absolute forgetting. More complete forgetting claims would require broader natural data and tasks.

## Family effects and small evaluation counts

All six evaluated families improve overall KL in all four mixed-versus-natural comparisons. The strongest improvements are capital recall, antonym and country language. Capital tasktop1 reaches100%on its TWOevaluationpairs, and antonym reaches100%on its ONEevaluationpair, in all four mixed models. Those are encouraging transfers to source-input-separated examples, not evidence of universal recovery.

Entity binding is less uniform by decision metrics: scalar72 tasktop1falls50.0→41.7%despite improved KL; block72 remains41.7%. Other tasktop1scores can stay unchanged while KL improves. Keep distribution fidelity, answer correctness and margins separate. All66trainingpair results and per-family evaluation failures remain in the JSON; the fourtraining-only families have no lexical evaluation estimate.

Block mixed taskKL.0377–.0441 is now competitive with scalar.0404–.0447 on this small panel. Block naturalKL.1520–.1612 remains worse than scalar.1139–.1218. Two seeds and fourteen evaluation pairs cannot establish a general block advantage. The stronger conclusion is that supplying diverse causal training examples matters substantially for both architectures.

## What this means for computational units

This completes a concrete link in the intended loop: broader competent task coverage → causal supervision → improved representation transfer. It has NOT established meaningful manifolds, reusable computations or genuine new-task discovery. Better distribution preservation can arise from many noncanonical internal representations.

Freeze the final models rather than extending task-mixture optimization. Next inspect the learned blocks: conditional within-block dimensionality, occupancy, seed-to-seed subspace/support agreement, and selective causal effects. Use scalar coordinate groupings and shuffled block assignments as controls; account for sign/permutation/within-block rotations. A stable or multidimensional block is only a candidate, not a semantic unit. Use existing training data for candidate selection and clearly separated evaluation for claims, preserving the original whole-task exclusions. Do not spend the tiny lexical evaluation panel repeatedly choosing interpretable-looking blocks.

## Audit and reproduction

2960native/intervention forwards:224teacher baselines,1200training forwards,1536evaluation forwards. Eight120-update histories exactly match the frozen schedules; tasktraining reads only66authorizedtrainingpairs. All model/source split hashes checked. Independent NumPy covers56sample featurizer reconstructions/full-output score evaluations across eightmodels and the natural+six task-evaluation domains. Maximum reconstruction discrepancy2.09e-7relative, metric discrepancy6.82e-8absolute, no near-tie support exceptions. All1536records and family summaries independently aggregated; immutable M8/M9baseline summaries rechecked.

Run `train_diverse_supervision.py`, then `audit_diverse_supervision.py`. Large model states are under primary artifacts `emlp_m11_diverse_supervision/`; audit logits are `emlp_m11_audit_samples.pt`. Saved states/hashes and full update-key histories are recorded; large files are not bundled in the packet.

[Frozen protocol](validation/diverse_supervision_protocol.json), [all results including training/evaluation separation](validation/diverse_supervision_results.json), [audit](validation/diverse_supervision_audit.json), [split rationale](DIVERSE_SUPERVISION_SPLIT.md).
