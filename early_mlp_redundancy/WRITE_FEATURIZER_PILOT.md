# M7: scalar features lead this small MLP1-write pilot; none is faithful yet

2026-10-28. Reconnects the M4–M6 natural-text causal benchmark to actual featurizer fitting. Six fixed fits, two seeds per architecture, no evaluation-based selection or tuning. No new corpus acquisition or original whole-task/BSF reserved-test exposure.

## What was fitted

Use native MLP1 product writes, excluding Down bias. Fit only on distinct causal prefixes from documents0–31 of M4's reused training cohort; evaluate all4096positions of documents32–63. Document hashes are disjoint. These evaluation documents have already been examined in M4–M6, so this is a development comparison, not pristine scientific validation. Common first-token prefixes or related text can still occur across documents.

All models have128atoms and296192trainable parameters: signed linear encoder with bias, unit-column decoder and output bias. Training-only mean/global RMS normalization, Adam.001,1000steps,batch128, seeds71/72, identical initial coordinate weights and minibatch streams per seed, fixed final step. Objective is RAW write reconstruction MSE only; no task labels or causal supervision.

Scalar selects32individual coordinates by absolute magnitude. Block selects8of32disjoint groups of4 by coordinate L2 energy. Fixed-shared selects4of32groups of8, with each atom appearing in exactly2groups; support is the exact union and each shared coordinate decodes once. Groups are fixed random memberships, not learned semantic concepts. Gates use coordinate energy, not the older SharedGroups learned gate encoder. There is no Grassmannian within-block orthogonality, hierarchy or DAG. Shared unions activate fewer than32coordinates, so the comparison matches parameter count and maximum active cost, not exact active cost. These models encode already-computed writes; they do not replace the original MLP computation from its inputs.

## Reconstruction and causal evaluation

Replace the native MLP1 product write at EVERY evaluation position by the featurizer reconstruction and retain Down bias. Recompute downstream outputs live. This differs from M5's covered-position-only table replacement, so their KL values are not a matched architecture comparison.

| Model / seed | Active coordinates, mean | Raw relative squared error | Within-token relative squared error | Output KL | Change in CE (nats) | Native top1 agreement |
|---|---:|---:|---:|---:|---:|---:|
| scalar / 71 | 32.00 | 0.438 | 0.642 | 0.304 | +0.307 | 72.0% |
| block / 71 | 32.00 | 0.465 | 0.674 | 0.424 | +0.424 | 67.0% |
| shared_fixed / 71 | 28.26 | 0.486 | 0.700 | 0.567 | +0.576 | 62.1% |
| scalar / 72 | 32.00 | 0.434 | 0.639 | 0.292 | +0.295 | 71.8% |
| block / 72 | 32.00 | 0.466 | 0.679 | 0.434 | +0.434 | 66.7% |
| shared_fixed / 72 | 28.44 | 0.490 | 0.716 | 0.524 | +0.531 | 64.0% |

Scalar beats block on both reconstruction and downstream metrics in both seeds, at exactly32active coordinates. The reconstruction gap is modest relative to the causal gap. Fixed-shared is worse, but its smaller union budget complicates attribution. All128atoms are used on evaluation for every model; this run does not reproduce the historical learned-membership collapse. Costs and both seeds remain reported.

The within-token error score subtracts each evaluation token group's mean reconstruction error, and divides by native within-token energy on that evaluation set. It measures fidelity to observed contextual variation; singleton groups contribute no within-token target energy. This descriptive error partition is not semantic supervision or held-out token-mean prediction.

None of these models preserves native behavior closely. With only32training documents and128dictionary atoms, optimization, capacity, coverage and architectural assumptions remain confounded. Fixed1000steps do not establish convergence; no width/step sweep follows. This pilot does not disprove meaningful blocks/shared computations, and it does not validate any manifold or discover a new task.

## Next: test an actual causal training objective

Keep scalar as the fidelity baseline and block as the matched-active-cost comparison. Do not rescue fixed-shared by tuning group counts based on these outcomes. The next bounded experiment should compare continued reconstruction-only training against an explicit downstream native-output KL objective, starting from the SAME saved checkpoints with the same training-document stream and fixed optimizer budget. Use only documents0–31 for teacher targets and fitting; freeze objective scaling before evaluation. Verify differentiable native replay and gradient paths before training. Evaluate raw/token/context reconstruction and downstream KL/CE on documents32–63 without selecting checkpoints from them. A supervised benefit must beat the continued-training control; merely evaluating KL is not causal supervision.

This asks whether loss choice rather than group architecture is the current bottleneck. It does not yet test semantic labels, diverse-task transfer or reusable computational DAGs; original whole-task exclusions stay reserved for future representation transfer tests.

## Audit and reproduction

Six1000step fits and224native/intervention forwards completed. Native writes on all32evaluation documents replay M4 exactly. Independent NumPy implements all six encoders, support selection, memberships, exact union and decoder over all4096evaluation positions: maximum relative reconstruction discrepancy4.28e-7. Token/context partitions agree within1.84e-16relative. Sixfull-output scores on document32 agree within8.88e-16absolute; all192causal-record aggregates are independently checked. Audit verifies unit costs and group membership structure. It does not independently reproduce optimizer steps or the full native model.

Run `train_write_featurizers.py`, then `audit_write_featurizers.py` in the research environment. Model states and evaluation reconstructions live in primary artifacts under `emlp_m7_write_featurizers/`; their hashes are in the results. Full causal audit sample is `emlp_m7_causal_samples.pt`. Large states/samples are not bundled in the shareable packet.

[Fixed protocol](validation/write_featurizer_protocol.json), [results](validation/write_featurizers.json), [audit](validation/write_featurizers_audit.json), [matched-direction benchmark](MATCHED_ANGLE_CONTROL.md).
