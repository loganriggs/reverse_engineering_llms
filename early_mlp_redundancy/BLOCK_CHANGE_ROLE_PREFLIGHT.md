# Testing block roles in the native background

M105 setup and preflight, 2026-10-29. **Role transfer has not yet been evaluated.**

## Why this test, given the existing work?

M12 found multidimensional blocks but weak individual block stability. M13–M16 found some cross-seed causal correspondence, mostly through group removals from imperfect reconstructed models. M49 tested selected shared-group atom removal in native backgrounds. M63–M71 subsequently tested full encoded changes and showed that encoded-change supervision can improve training effects without reliably transferring. These are distinct results; none evaluates all newer block groups' task-changing contributions in native backgrounds with the comparison below.

The question is whether a learned block's encoded task change has a reproducible causal role across seeds and tasks, even when the full featurizer is imperfect. Positive correspondence would still be conditional evidence about an external intervention, not proof of native units. Negative correspondence would limit the current block interpretation without establishing that sparse structure is impossible.

## Frozen candidates and intervention

Use four previously trained block models: seeds71/72, each KL-only or KL-plus-encoded-change supervision. No new parameter fitting. Compare their 32 trained groups of4 atoms with fixed shuffled groups of the same atoms. Shuffling changes grouping, not the encoded reconstruction; active coordinate costs and removed/added energies are not statewise matched.

For each donor/recipient pair, independently encode both native MLP1 product writes. A group's change is its decoded donor contribution minus its decoded recipient contribution at the selected positions. Add that change to the **native recipient write**, retaining all other native computation. Means and decoder biases cancel. This is neither native-neuron deletion nor replacement of the full MLP by a featurizer.

All disjoint group changes sum to the encoded change in real arithmetic. Their downstream responses need not add; measure the sum-versus-joint response discrepancy explicitly. Include complete encoded-change and native-change controls and identity/no-op checks. Source and final-position effects remain separate.

## Data and correspondence

Training signatures use one pair per each of11 training families, selected by fixed hash without outcomes. Preserve all14 historical source-group evaluation pairs and all56 whole-task pairs. These whole-task families were excluded from fitting but evaluated in prior research; they are not pristine blind discovery data. Retain ineligible alignments explicitly and report native competence separately.

Freeze cross-seed Hungarian assignments using only decoder-subspace overlap, separately for each objective and trained/shuffled grouping. Compare100 fixed random permutations. No task-effect matching, amplitude fitting, group filtering, or gain repair. Report raw signed family/site effect prediction, cosine, pooled and equal-family relative errors, and effect energy/active costs. Similarity to a task role is not semantic specificity by itself.

## Completed correctness checks

The preflight uses one prespecified eligible training pair. All four states pass independent NumPy encoder and Top8 block-gate replay. All32 group changes per site/endpoint/grouping pass independent decoder arithmetic and sum checks. Two prescribed groups per condition are checked through both the callback intervention and explicit native MLP override:64 conditions,130 Transformer forwards, all exact output agreements. Historical native baselines replay exactly.

Maximum relative discrepancies are 1.01e-7 for independent selected codes, 1.24e-7 for group decodes, and 1.80e-6 between the summed group changes and differences of separately rounded reconstructions. The frozen tolerance is1e-5. This is float32 arithmetic agreement, not bit-exact group summation. Saved preflight outputs remain available for inspection.

Weight-only matching is frozen. All4,096 overlap scores agree between NumPy QR and Torch SVD subspace constructions (maximum difference5.56e-16).

| Objective | Trained-group matched overlap | Shuffled-group matched overlap |
|---|---:|---:|
| KL-only | 0.117576 | 0.080325 |
| Encoded-change supervision | 0.117568 | 0.080243 |

Random-assignment means are about0.036. These nearly unchanged geometric scores are descriptive; causal role transfer is still unmeasured by M105. This avoids treating a small geometry difference as evidence that supervision found better units.

## Next and stopping rule

Implement one resumable producer for the frozen training/source-group/whole-task cohorts, save every group response and complete controls, and independently audit the saved responses and decompositions before drawing conclusions. Then compute the already-frozen matching comparisons. No further matches, gains, architectures, or rank searches follow a disappointing result. This is a direct featurizer-validation deliverable, not another geometry-only detour.

[Protocol](validation/block_change_roles_protocol.json), [preflight](validation/block_change_roles_preflight.json), [weight-only matching](validation/block_change_role_matching.json).
