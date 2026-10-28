# M18: broader context fitting improves prediction more consistently than causal fidelity

2026-10-28. Eight paired fits and the fixed native-intervention validation are complete. Context expansion strongly improves output distributions on training wrappers with held-out lexical items. Transfer to withheld wrappers is uneven, and improvement in native causal response fidelity is less consistent. Current features remain candidate representations, not validated semantic units.

## Frozen comparison

Four M11 starts (scalar/block × seeds71/72), each continued in original-context and expanded-context arms. Fresh Adam at.0003,120updates, unit decoder columns and frozen mean/scale. Sixty natural-text updates use paired documents0–31; sixty task updates use paired source examples, six per family. Only the expanded arm receives the specified M17 wrappers. Native-gated training eligibility uses train/train data only. Past tense has one available template; antonym/plural two; other families four. Initial checkpoints are evaluated as baselines.

This matches starting weights, source examples and updates, not tokens/FLOPs: expanded prompts are longer. The objective is native-output KL, not an intervention loss. All480 expanded pairs,80 original pairs and32 natural evaluation documents are evaluated for each of12 checkpoints. The three nontraining quadrants separate lexical, template and joint transfer. Every native failure remains in the all-row analysis; native-passing summaries are also saved.

## Output fidelity

Mean all-position KL, original-context control → expanded-context training. All rows retained; lower is better.

| Model | Held-out lexical, training templates | Training lexical, withheld templates | Both lexical and templates withheld | Natural evaluation |
|---|---:|---:|---:|---:|
| scalar71 | 0.0886 → 0.0410 | 0.0819 → 0.0624 | 0.0833 → 0.0748 | 0.1193 → 0.1181 |
| scalar72 | 0.0767 → 0.0297 | 0.0910 → 0.0610 | 0.0788 → 0.0590 | 0.1137 → 0.1146 |
| block71 | 0.0859 → 0.0336 | 0.0858 → 0.0866 | 0.0847 → 0.0851 | 0.1461 → 0.1477 |
| block72 | 0.0960 → 0.0414 | 0.0859 → 0.0738 | 0.0789 → 0.0783 | 0.1567 → 0.1569 |

Training-template lexical KL improves54–61% in allfour comparisons. Withheld-template improvements are smaller: scalar models improve on both nontraining template quadrants, block71 slightly worsens, and block72's joint-transfer gain is negligible. Natural KL changes about−1% to+1% versus control. Initial-checkpoint comparisons and all family results are retained in the JSON.

Lower KL is not necessarily higher answer accuracy. On the joint lexical/template evaluation, top1 falls from.732→.696 and.750→.714 for scalar71/72, and.768→.750 and.821→.786 for block71/72. Original unprefixed lexical KL for block72 worsens.0343→.0518. These outcomes prevent a general claim of improved task understanding.

## Native intervention fidelity, independently of fitted coordinates

Before fitting, select one native-passing pair per family/nontraining quadrant using fixed hashes:22pairs44endpoints. Select two groups per endpoint/seed, and intervene separately on prefix/task positions. Four initial models yield704 fixed native directions,96 exactly zero. Each delta is decoded from the **initial** featurizer using native MLP1 writes. Apply the identical delta to the native write and to each initial/final reconstructed write, preserve biases, and recompute downstream. The trained model cannot change the intervention to make this test easier.

Compare margin changes relative to each model's own intact baseline. Relative squared error is summed response discrepancy divided by summed native effect energy; zero response prediction scores1. Also report raw RMSE. A second metric compares final-logprob responses, weighted by intact native probabilities, with corresponding native response energy normalization. Intact and perturbed output KL are recorded separately. No gain or match fitting.

| Model | Region | Margin error: initial | Original control | Expanded context | Logprob-response error: control → expanded |
|---|---|---:|---:|---:|---:|
| scalar71 | prefix | 0.832 | 0.605 | 0.551 | 1.107 → 0.898 |
| scalar71 | task | 4.099 | 3.008 | 3.682 | 0.829 → 0.919 |
| scalar72 | prefix | 2.416 | 1.602 | 1.881 | 1.508 → 1.635 |
| scalar72 | task | 0.277 | 0.340 | 0.366 | 0.621 → 0.502 |
| block71 | prefix | 0.682 | 0.377 | 0.325 | 2.251 → 1.090 |
| block71 | task | 1.246 | 1.330 | 1.678 | 0.877 → 1.038 |
| block72 | prefix | 1.062 | 0.920 | 1.166 | 0.947 → 0.984 |
| block72 | task | 1.314 | 1.490 | 0.854 | 1.248 → 0.772 |

Expanded-context training improves aggregate margin-response error in3/8 model/region comparisons and logprob-response error in4/8. Equal-family margin aggregation improves4/8, rather than3/8, showing sensitivity to effect weighting. Family ratios can be large when native effect energy is small; raw sums, denominator energies and zero-denominator counts are available. Do not read these errors as percentages of computation understood.

There is a useful qualification: restricted to the joint withheld-lexical/withheld-template causal subset, margin fidelity improves6/8 comparisons (both regions in scalar71, block71 and block72; neither in scalar72). The broad aggregate combines three evaluation regimes and hides that difference. This subset contains only one pair per six families, so it is an informative scope distinction, not a robust population claim. It does not rescue the uneven all-regime result.

The fixed perturbations are different across initial featurizers, so within-model arm comparisons are cleaner than architecture rankings. Sampling two groups per endpoint does not cover every learned unit. Perturbation transport assesses downstream sensitivity around reconstructed versus native states; it does not establish semantic labels, final-feature identity or manifold recovery.

## What this changes and next bounded test

We now have a direct native causal-fidelity benchmark, not just cross-seed agreement between imperfect reconstructions. Better context fitting alone does not reliably preserve that benchmark. The next comparison should explicitly supervise native intervention responses while retaining intact-output fidelity, against continued intact-output training from the same checkpoints. Use only training contexts and source pairs for new targets; preserve all evaluation templates and whole-task exclusions. Freeze intervention directions from each starting checkpoint so shrinking directions cannot reduce the training target. Define normalization using training data only and fix the loss/schedule before outcomes.

This is a distinct objective test, not another search for better block matching. Keep scalar and block controls, both regions, natural-text forgetting and all three evaluation quadrants. Success must improve independently selected evaluation perturbations, not merely trained response targets. Even success would show better causal featurization, not new-task discovery or complete reverse engineering. The broad goal remains open.

## Reproduction and audit

```bash
python prepare_context_training.py
python audit_context_training_protocol.py
python train_context_supervision.py
python audit_context_training.py
python probe_context_native_fidelity.py
python audit_context_native_fidelity.py
python analyze_context_fidelity_comparison.py
```

Training uses16,448forwards (1,184teachers,1,440training,13,824evaluation). Native response validation uses3,004forwards, including skipped exact-zero directions, and records2,112 response comparisons. The training probe refuses to overwrite existing checkpoint files; inspect a stopped run before restarting.

Independent training audit checks all paired schedules,12 checkpoint hashes and initial tensor equality, full evaluation inventories/aggregates, and228 NumPy featurizer/output samples. Max forward relative error2.09e-7, metric error6.81e-8; zero TopK tie exceptions. The native audit independently reconstructs72 sampled initial codes/masks/deltas and output-response metrics, checks all704 fixed native directions/2,112 records/no-ops/shared-teacher invariants and336 aggregates. Max decoder error2.09e-7, response metric discrepancy1.78e-15; zero tie exceptions. The audit script's schedule-variable shadowing was corrected before it passed; no training data or result was changed. These are sample/full-accounting audits, not independent implementations of optimization or the native transformer.

[Protocol](validation/context_training_protocol.json), [protocol audit](validation/context_training_protocol_audit.json), [training results](validation/context_training_results.json), [training audit](validation/context_training_audit.json), [native responses](validation/context_native_fidelity.json), [native audit](validation/context_native_fidelity_audit.json), [weighted/equal-family comparison](validation/context_fidelity_comparison.json).

Large states: `artifacts/bilin18_tensor_decomposition/emlp_m18_context_training/`; audit samples: `emlp_m18_context_training_samples.pt` and `emlp_m18_native_fidelity_samples.pt`. These local model/sample files are not bundled in the Git packet.
