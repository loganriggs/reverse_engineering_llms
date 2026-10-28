# M8: downstream supervision improves fidelity while worsening reconstruction

2026-10-28. Fixed fine-tuning comparison from the SAME M7 scalar/block checkpoints, both seeds retained. This is an actual downstream training objective, not just evaluation with a causal metric.

## Gradient verification and fixed design

The native language model is frozen. Invoke the underlying native trace with autograd enabled and replace the MLP1 product write with the featurizer reconstruction, retaining Down bias. Native-write replay produces exactly identical logits. On training document0, the KL derivative with respect to the replacement write agrees with central finite differences at two step sizes to7.05e-6 and3.62e-5relative error. All four featurizer parameter tensors receive finite nonzero gradients; native model gradients stay absent. TopK selections themselves remain discrete: this checks selected-support gradients, not a derivative of the hard index changes.

For each of four saved models, start two arms from identical weights and frozen training normalization: continue normalized raw-write MSE, or optimize pure native-output KL over all50304logits and128positions. Fresh Adam.0003,128updates, one full training document per step, four paired permutations of documents0–31, unit decoder columns after updates. No mixture-weight search, evaluation-based checkpoint selection or new data. Both arms retain32active coordinates and identical parameter count. KL training requires substantially more computation; updates/data are matched, FLOPs are not.

Evaluation uses all positions of documents32–63, already used development documents with disjoint document hashes from the fitting half. It remains neither pristine scientific holdout nor whole-task transfer. Both arms are evaluated by live native-model reconstruction interventions.

## Results

| Architecture / seed | Objective | Raw relative squared error | Within-token relative error | Output KL | Change in CE (nats) |
|---|---|---:|---:|---:|---:|
| scalar / 71 | reconstruction | 0.428 | 0.629 | 0.300 | +0.303 |
| scalar / 71 | downstream_KL | 0.458 | 0.659 | 0.121 | +0.120 |
| block / 71 | reconstruction | 0.456 | 0.663 | 0.416 | +0.414 |
| block / 71 | downstream_KL | 0.487 | 0.691 | 0.154 | +0.150 |
| scalar / 72 | reconstruction | 0.427 | 0.629 | 0.289 | +0.291 |
| scalar / 72 | downstream_KL | 0.454 | 0.655 | 0.118 | +0.117 |
| block / 72 | reconstruction | 0.459 | 0.670 | 0.421 | +0.424 |
| block / 72 | downstream_KL | 0.487 | 0.693 | 0.166 | +0.166 |

KL supervision reduces evaluation output KL by59–63% versus the matched continued-MSE controls in all four comparisons. It also improves next-token CE and native top1 agreement. Scalar remains better than block: causal scalar KL.118–.121 versus causal block.154–.166. This does not establish a statistically certified architecture ranking from two seeds.

Raw write reconstruction and within-token reconstruction both worsen under KL training. Therefore the improvement is not explained by a general increase in Euclidean reconstruction quality. The native downstream computation supplies information about which errors matter. This is consistent with M6's unequal effects for equal-size errors, but does not prove a particular feature is semantic or uniquely causal.

Neither representation is faithful enough to call the computation reverse-engineered. Finite capacity, limited data and optimization remain unresolved. A lower KL can preserve distributional behavior while changing internal implementation; it is evidence for an output-fidelity objective, not a canonical ontology, stable manifold, reusable computation, or discovered task. The model still requires the original MLP write as encoder input.

## Next: frozen transfer to diverse competent tasks

Freeze all eight final models. Evaluate them without fitting on the existing80-pair, ten-family training task panel from earlier focused studies. These task families were not part of M7/M8 featurizer fitting, but were already examined elsewhere in this research, so call this representation transfer to an existing development panel, not fresh task discovery or pristine whole-task generalization. Preserve the original excluded task families.

Report full-output KL and native top1 agreement, pair/task decision fidelity and changes in margins where the panel's answer definitions support them, alongside local reconstruction. Keep both seeds and every family, including failures. This tests whether the natural-text causal-supervision benefit generalizes beyond those documents before interpreting feature groups or allocating further supervised tasks. No further tuning on the natural-text evaluation half.

## Audit and numerical boundary case

Eight128update fits completed;832native/intervention forwards in the main run (64teacher baselines,512KL-training forwards,256evaluation interventions), plus the separately recorded gradient smoke test. Native writes in all64baseline documents replay M4 exactly. Saved checkpoints reproduce every evaluation reconstruction exactly under the original CUDA arithmetic.

Initial independent float64 NumPy forward audit found a0.000785relative reconstruction discrepancy in the scalar71MSE arm. Investigation isolated ONE TopK boundary row among8×4096evaluated rows: relative score gap1.94e-7, causing float32/float64 to select different coordinates. All other supports agree. The audit retains that unconditioned discrepancy and permits production support only for explicitly near-cutoff rows (relative gap<1e-5). Conditional independent decoder error is at most4.33e-7; all output score audits agree within8.89e-16. Token/context partitions agree within1.86e-16relative; all256record aggregates checked. This is a documented numerical-discontinuity exception, not an unqualified independent exact gate replay. No training or experiment result was changed to resolve it.

Run `check_causal_gradients.py`, `finetune_causal_featurizers.py`, then `audit_causal_featurizers.py`. Large saved states/reconstructions are under primary artifacts `emlp_m8_causal_featurizers/`, audit logits in `emlp_m8_causal_samples.pt`; hashes are retained. Models and samples are not bundled in this packet.

[Gradient check](validation/causal_gradient_check.json), [fixed protocol](validation/causal_featurizer_protocol.json), [all results](validation/causal_featurizers.json), [audit](validation/causal_featurizers_audit.json), [M7 initial fits](WRITE_FEATURIZER_PILOT.md).
