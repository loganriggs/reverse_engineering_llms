## H766 — Sensitivity-weighted geometry does not consistently preserve copying interventions

The frozen natural-data Fisher geometries were evaluated on the H761 copying panel: 32 pairs across semicolon, newline, colon and equals formats. All four formats passed native competence checks. Both regularization strengths, fixed and adaptive coordinate selection, and budgets 8, 16, 32, 64 and 128 were retained. No task labels were used to fit these geometries. This panel is reused evaluation data, not a new confirmatory holdout.

At L5H5, interventions add reconstructed donor minus reconstructed receiver activations to the native head. Query, source and joint positions are tested, with reciprocal answer changes, nuisance changes and identity controls. Comparisons below are query-position answer-change errors against the full head edit, in absolute answer-margin RMS units; lower is better.

| Format | Adaptive write PCA, 8 | Adaptive Fisher .01, 8 | Adaptive Fisher .1, 8 | Adaptive write PCA, 64 | Adaptive Fisher .01, 64 | Adaptive Fisher .1, 64 |
|---|---:|---:|---:|---:|---:|---:|
| Semicolon | .2477 | .2091 | .2147 | .0815 | .0991 | .0941 |
| Newline | .3686 | .2520 | .2104 | .0660 | .0693 | .1066 |
| Colon | .3034 | .2463 | .2324 | .0524 | .0668 | .0676 |
| Equals | .2288 | .1648 | .1651 | .0713 | .0809 | .0689 |

Adaptive Fisher improves all four mean errors at budget 8. That advantage does not persist across budgets: at 64, ridge .01 is worse in all four formats, and ridge .1 is worse in three. Fixed Fisher at 16 is worse in all three formats with full intervention RMS above .25; fixed 64 improves colon and equals but worsens newline. Intermediate errors are nonmonotonic. No single frozen Fisher geometry is established as generally better. These descriptive comparisons have small, shared panels and should not be interpreted as independent significance tests.

Full query edit RMS is .1688 for semicolon, .2616 for newline, .2528 for colon and .2539 for equals. Relative errors remain suppressed for the semicolon task effect under the preexisting .25 threshold. Source-only full effects remain small. Complete query/source/joint, task/nuisance/identity comparisons are saved in the companion JSON, without choosing a winning ridge or budget post hoc.

Together with H765's natural-text replacement KL results, this rules out adopting the current global Fisher geometry as an already-verified improvement. It does not rule out output-aware featurization. The metric averages local sensitivity across contexts and ignores its correlation with individual activation errors; reconstruction also produces finite changes that need not follow the local quadratic approximation.

Audits: 8,064 whole-model forwards; full intervention replay and all rank-128 margin replays exactly match their references. Identity logit discrepancy <=2.17e-5 and full donor-head discrepancy <=1.77e-5. All outputs finite and source/capture/model hashes unchanged. Original reserved test untouched; unique natural document pool remains 26,112.

Next information-gaining experiment: compare actual KL along scaled reconstruction-error directions with local per-context and globally averaged quadratic predictions. Small steps test the local approximation; full steps test finite-change failure; the difference between local and global predictions tests context averaging. Lock directions and contexts before inspecting outcomes. Then revisit feature learning according to the diagnosed failure. Manifold verification remains the downstream objective: stable structure across seeds and data, held-out combination transfer, and selective causal effects across diverse competent tasks. Reconstruction alone is insufficient.
