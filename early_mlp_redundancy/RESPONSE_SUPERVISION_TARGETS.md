# M19: native-response training targets and gradients verified

2026-10-28. The next objective is implemented and validated, but no featurizer has yet been optimized with it. This stage produces native causal supervision rather than treating better output KL as evidence of better computational units.

## Frozen comparison

Start from all four M18 expanded-context checkpoints. Continue each under intact-output KL alone or intact-output KL plus response matching: eight fits,120updates, fresh Adam at.0003, unit decoder columns. Reuse the paired M18 task/source/natural schedules, sixty natural and sixty task updates, six per task family. Each family gets exactly three prefix-region and three task-region target updates. Training inputs remain in the native-passing train/train M17 cohorts; evaluation templates and original whole-task exclusions remain unused for target construction.

Each task endpoint receives one hash-selected four-atom group. Decode its perturbation from the starting M18 checkpoint using native MLP1 writes and freeze it, including its region mask. Keep inactive directions; do not select groups by response strength. Initial-state perturbations remain fixed as learned features change.

For final-token probability vectors, let the native response be the change in log probabilities after perturbation, and the featurizer response the corresponding change relative to its own intact reconstruction. The response loss is the squared difference of those responses, weighted by the intact native probabilities. Gradients flow through **both** reconstructed forwards. Native targets, weights and perturbation vectors are detached.

Ordinary loss K is all-position native-output KL. Response loss R is final-position and all-vocabulary. Before optimization, set lambda=mean_initial_K/mean_initial_R over all120 scheduled task endpoints for each start, including repeated examples and inactive directions. The rule was fixed before target generation and uses training data only. Natural updates use K; task updates compare K against K+lambda*R, with equal endpoint weights. This matches initial average contributions, not gradient norms or total objective scale. Treatment needs additional downstream forwards/backwards; updates are matched, not FLOPs.

## Collected training targets

480 scheduled endpoint perturbations,158 unique native input sequences and400 unique model/input reconstructions required1,404 native forwards. These are repeated scheduled training examples, not480 independent examples.

| Start | Nonzero / scheduled directions | Mean initial KL | Mean initial response error | Frozen response weight |
|---|---:|---:|---:|---:|
| block71 | 90/120 | 0.028541 | 0.002505 | 11.395977 |
| block72 | 93/120 | 0.024543 | 0.007959 | 3.083833 |
| scalar71 | 120/120 | 0.020104 | 0.004298 | 4.677775 |
| scalar72 | 120/120 | 0.022848 | 0.002317 | 9.858933 |

All57 inactive directions belong to block models; they remain in training and calibration. Consequently architectures do not receive equal numbers of nonzero perturbation targets. Compare treatment/control within each starting model before drawing architecture conclusions.

Family-balanced update counts also do not imply balanced response-loss magnitude. As a descriptive pretraining diagnostic, past tense supplies50.2% of scalar71's initial response error and45.1% of block72's. Past tense has only one passing training wrapper. We retain the prescribed loss, report this concentration, and require family-level evaluation; we do not retune weights after seeing it.

## Verification

The independent target audit reconstructs every one of400 initial featurizer forwards, all480 masks/deltas, every cached intact KL and response loss, eligibility, hash-selected groups, family/region counts, and allfour calibrated weights. Maximum decoding relative discrepancy is3.42e-7, with zero TopK tie exceptions. The native transformer itself is not independently reimplemented.

Eight gradient cases cover each model and both regions, using the first scheduled nonzero perturbation with positive response loss. All sixteen directional finite differences pass; maximum relative discrepancy is0.00482 (0.482%), within the predeclared3% tolerance. Setting the reconstructed write equal to the native write gives response error at most5.58e-12. Both reconstructed branches have nonzero gradients, featurizer parameter gradients are finite, and native weights/deltas remain frozen. This validates selected-support gradients, not derivatives across TopK index changes or full optimizer behavior.

## Transfer test and next action

Execute the frozen eight fits, then evaluate the unchanged M18 output inventory and the **same M18/M11-based704 native perturbation directions**. Training directions come from the newer M18 starts, while evaluation directions stay anchored to M11; do not accidentally regenerate evaluation deltas from the new initial checkpoints. Preserve the old selection, sites, group indices and native responses, verifying that benchmark invariants replay. Report initial baselines, both final arms, both regions, three nontraining quadrants, weighted/equal-family effects, raw response errors and natural-text forgetting.

This is a fixed objective comparison: no response-weight, group, template or width sweep. An intervention-training improvement must transfer to the independently selected benchmark, rather than just fitting its training targets. Even a successful result would establish better causal featurization, not semantic manifolds, feature identity, new-task discovery or full reverse engineering. The broad research goal remains active.

## Reproduction and artifacts

```bash
python prepare_response_supervision.py
python collect_response_targets.py
python audit_response_targets.py
python check_response_gradients.py
```

Protocol and target generation refuse to overwrite incompatible/completed results. The full tensor cache is local at `artifacts/bilin18_tensor_decomposition/emlp_m19_response_training_targets.pt`; it is required for subsequent training and is not bundled with Git. Scripts and JSON receipts are bundled.

[Protocol](validation/response_supervision_protocol.json), [targets and calibration](validation/response_training_targets.json), [full target audit](validation/response_target_audit.json), [gradient checks](validation/response_gradient_checks.json).
