# M75: broader validation coverage, with competence limits preserved

We returned from the local-metric diagnostic to the featurizer validation loop. A fixed cohort now covers seven explicit training-excluded operations, with eight contrast pairs and two formats per family. All56pairs were selected from frozen historical manifests without consulting their outcome scores during selection. All56 also have verified historical native-evaluation records: this is broader validation, not a pristine test or task discovery.

## Training and exposure audit

The70-pair current causal-training inventory and the historical80-pair task panel contain none of these seven family labels, and none of the112selected token sequences exactly matches those training prompts. The audit saves source hashes and operational definitions. This does not establish absence of related computation from natural-text featurizer fitting or model pretraining.

- Sequence copying: repeated key/value continuation.
- Spatial inversion: reverse left/right or above/below when the queried entities reverse.
- Weekday successor: next weekday in a sequence or yesterday/today prose. Its historical formats were adaptively developed; that exposure is disclosed.
- Role binding: agent/patient swaps with active/passive clauses, distinct from but related to trained recipient-completion entity binding.
- Temporal cues (`tense`): Yesterday/Tomorrow with was/will contrast; distinct from trained explicit past-tense word lookup. Completions are not unique.
- Continent recall: country-to-continent retrieval, related to trained capital/language facts. Historical America labels are broad.
- Object function: use/purpose completion with non-exhaustive, sometimes awkward answer labels.

Thus “whole-task exclusion” describes the explicit training operation and prompt inventory; it does not imply absence of shared vocabulary, knowledge, or subcomputations.

## Native competence

The predeclared gate requires at least6/8pairs to have the intended answer group beat its paired alternative on both endpoints. Original, capitalized and uppercase single-token answers are accepted with exact token-boundary checks. Every pair is retained, including failures. Top1 is measured over the full vocabulary separately.

| Family | Both endpoints correct, pairs /8 | Gate | Intended top1, endpoints /16 | Source-aligned pairs /8 |
|---|---:|---|---:|---:|
| Sequence copy | 8 | Pass | 8 | 8 |
| Spatial inversion | 4 | Fail | 8 | 8 |
| Weekday successor | 7 | Pass | 12 | 7 |
| Role binding | 2 | Fail | 8 | 6 |
| Temporal cues | 8 | Pass | 3 | 8 |
| Continent recall | 6 | Pass | 6 | 7 |
| Object function | 8 | Pass | 4 | 8 |

Five families pass the forced-choice contrast gate. This is not uniform free-form competence. In particular, temporal cues and object functions pass every pair while rarely putting the declared answer first in the vocabulary. Other valid completions can account for some failures, so low top1 also does not prove inability.

Spatial inversion splits4/4correct pairs for left/right and0/4for above/below. Role binding has0/4active-format pairs and2/4passive-format pairs. We will not rescue those formats or keep only successful examples after seeing the results.

## Native causal coverage

We captured both endpoints and intervened on MLP1 product-write changes at changed input positions, the final position, and an empty identity control. Down bias and the model's native downstream paths are preserved. Equal-length source addresses are not a guarantee of semantic alignment, especially for role swaps. Four unequal-length pairs produce eight retained source-ineligible assignments; final positions remain explicitly alignable.

| Family | Source response energy, summed | Final response energy, summed |
|---|---:|---:|
| Sequence copy | 3.05695 | 0.819448 |
| Spatial inversion | 0.34855 | 0.000480 |
| Weekday successor | 0.14937 | 0.017110 |
| Role binding | 0.24413 | 0.001625 |
| Temporal cues | 1.51686 | 0.019834 |
| Continent recall | 0.37734 | 0.003944 |
| Object function | 1.22623 | 0.007423 |

Response energy is the original-output-probability-weighted squared change in final log probabilities, not task accuracy. Source interventions can touch multiple positions; final interventions touch one. The table therefore does not establish which site performs the task or compare equal intervention budgets. Very small final targets make relative fidelity errors particularly sensitive; raw energies and margin changes will accompany normalized errors.

## Verification and artifacts

- Frozen56pairs,112endpoints,336assignments: all56are token-valid;52have equal-length nonempty source changes.
-328native forward executions;328evaluated response vectors,8source-ineligible assignments,112exact identity controls.
- Independent NumPy audit checks all saved responses and deltas, every address, all21family/site aggregates and forward counts. Maximum absolute metric discrepancy7.45e-9 includes large local delta energies. The transformer itself was not independently reimplemented.
- All56selected historical IDs are matched to actual prior evaluation records with hashed result files, rather than inferring exposure from names alone.
- The initial builder invocation lacked the existing tiktoken dependency path and failed before writing artifacts. Rerunning with the established dependency path succeeded; no prompt changes or outcome-based repair occurred. No native experimental failure occurred.

Scripts: build_whole_task_validation.py, probe_whole_task_native.py, audit_whole_task_native.py, analyze_whole_task_native.py, prepare_whole_task_comparison.py. Protocols, source/exposure audits, raw metrics, summaries and hashes are under validation/whole_task_*.json. The full native samples remain in the local artifact directory. No featurizer was trained or evaluated in M75.

## Frozen next comparison

M76 compares all18original sparse encoders, the18dense least-squares versions of the same dictionaries, and PCA32/PCA128 on this entire cohort. It retains all three measured background/change corners, both intervention sites and identity, all weak families and all ineligible assignments. It reports every family and format, intact predictions, full-distribution fidelity, causal response fidelity and local reconstruction separately. Existing natural retention measurements remain fixed references.

The conceptual question is whether gains in retained distributed directions transfer to different computations and formats, or depend on the familiar task/background geometry. Passing contrast tasks alone will not establish reusable manifolds, native-neuron redundancy, or semantic units. One fixed joint comparison comes next, with no task-specific seed choice, fitting, rank sweep or prompt rescue.
