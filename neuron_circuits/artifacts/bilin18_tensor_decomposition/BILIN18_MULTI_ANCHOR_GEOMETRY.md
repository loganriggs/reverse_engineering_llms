## H1006 — Preserving multiple causal anchors partly recovers old-task fidelity

Previous turn was progress: H1005 confirmed the new anchor's advantage over random constraints and exposed mixed geometry's old-task causal loss. H1006 is the final dense-geometry comparison before sparse allocation.11648forwards complete, comprising256natural reference forwards,6272whole-state evaluations,640full-donor references and4480paired candidate edits. Existing development panels and natural documents reused.

The H994six-task discovery causal selections contain48memberships but35unique oldtaskPCA atoms. Preserve their union plus H1001nounrank1, for36orthogonal constrained directions. Fill the remaining92dimensions with leading mixed-discovery PCA directions orthogonal to these constraints. Totalrank128. The dictionary explicitly retains a mapping from each original task group to its shared atoms, counting overlaps once. Controls keep the same noun anchor and35seeded randomly selected oldtaskPCA atoms, then fit the same92dimensional complement. This matches constraint count, old-task source span and new-anchor inclusion, rather than comparing against completely unrelated ambient directions. Seeds10061/10062 fixed; no evaluation fitting or outcome selection.

| Dictionary | Old whole-state KL | New whole-state KL | Natural whole-state KL | Old paired relative KL | New paired relative KL |
|---|---:|---:|---:|---:|---:|
| Task-only128 |0.003093|0.033368|0.033326|0.0913|0.7741|
| Mixed+noun128 |0.003371|0.026423|0.028605|0.1951|0.3433|
| Mixed+allanchors128 |0.003255|0.028404|0.028856|0.1219|0.3558|
| Matchedrandom1 |0.003359|0.028509|0.028010|0.1646|0.3750|
| Matchedrandom2 |0.003175|0.029971|0.029058|0.1693|0.3519|

Whole-state KL is KL(native || reconstruction); paired relative KL uses the full donor edit as reference, normalized by its native-reference divergence within cohort, then macro-averaged. Lower is better for both but they measure different interventions. Primary summaries cover400old states/50native-passing cohorts,40new states/fivepassing cells, and256natural states/64documents. Diagnostics remain saved; original whole-task holdouts untouched.

Selecting old directions by their discovery causal profiles reduces old paired error from0.195to0.122 (~37.5%), outperforming bothrandommatchedconstraints0.165/0.169. It does not recover task-only0.091. New paired fidelity remains strong relative to task-only but is slightly worse than noun-only mixed; one random control is marginally better on this new metric. Whole-state differences across constrained bases are modest and mixed: one random seed has better old/natural KL than the selected union. Thus the selected anchors have evidence for preserving targeted old causal contrasts, not universal superiority.

This is a concrete multi-task supervision benefit at matchedrank: prior causal measurements can guide which directions a broad reconstruction representation must retain. It does not prove each selected atom is a semantic unit. Capital/binding anchors were previously weak in some assays, signed discovery selection can be noisy, and a union preserves directions without learning sparse groups, hierarchy or computation reuse. Overlapping membership alone is not evidence of shared computation.

Decision and next step: freeze these dictionaries and move to sparse allocation now. Compare individual-coordinate selection with overlapping task-group selection and shuffled-group controls, matching active dimension or reporting per-state matched scalar controls when group unions vary. Decode shared atoms once. Evaluate both whole-state and paired fidelity on the same locked panels, preserving dense task-only and dense multi-anchor baselines to expose losses. No more dense geometry tuning before this test. Group names remain provisional task labels; meaningful manifolds and reusable computations require further validation.

Audits:11648/11648forwards; source/basis hashes unchanged; identity logits3.96e-5maxerror and natural prefix capture relative3.78e-6. Independent42whole-state and28paired saved-logit samples replay with zero metric error; all summaries/macros replayed; unique state/paired keys verified. Independent constrained fit projector error1.99e-15 and constraint retention error2.50e-16. Verified all seven group-to-atom maps;35oldatoms+oneanchor=36constraints. No reserved-test evaluation or validation fitting. Natural-document pool26,112 unchanged. H961 context-bridge evaluation pending; broad goal active.
