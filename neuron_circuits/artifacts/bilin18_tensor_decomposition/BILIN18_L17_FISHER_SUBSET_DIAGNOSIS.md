

## H409 — Separate subset reconstruction, context transfer and finite-edit mismatch

Reuse all H408 frozen orders and both panels, actual head6 only, ranks16/64/256. No reselection or new documents. The discovery panel was used to choose Fisher order, so its accuracy is selection-conditioned. Evaluation documents are disjoint from selection but previously studied. Native weights remain fixed.

Let t be the complete centered logit derivative, k its selected-factor component, o=t−k, and p the intact native output probabilities. Store F=E_p[t²], K=E_p[k²], O=E_p[o²], C=2E_p[ko], A=E_p[kt]. Then F=K+O+C, A=K+C/2, and subset recovery R=1−O/F=2A/F−K/F. These are pooled energies, not averaged site ratios. The final RMS derivative uses native float32 epsilon with double arithmetic; capped-logit derivatives and softmax centering are included. H408 finite KL recovery uses a different, finite-intervention metric; their difference diagnoses predictive mismatch, not an exact causal decomposition.

Freeze one joint4 max-absolute-deviation95% band from4000 document bootstraps, independently resampling the two panels and pairing methods/finite outcomes within evaluation. Gates: discovery Fisher64 recovery lower>.9; evaluation−discovery tangent interval wholly within±.05; evaluation tangent−finite unit interval wholly within±.05; evaluation Fisher64−contrast64 tangent improvement lower>.05. Numerical energy closure and H408 full-Fisher replay must be<1e-8. No post-result threshold changes.

|Primary statistic|Estimate|Joint95% interval|
|---|---:|---|
|fisher64_discovery_tangent_recovery|0.668427|[0.6028517709018489, 0.7340029497335109]|
|fisher64_evaluation_minus_discovery_tangent_recovery|0.004796|[-0.060779871266043714, 0.07037130756561831]|
|fisher64_evaluation_tangent_minus_finite_unit_recovery|0.017520|[-0.048055519293461676, 0.08309565953820035]|
|fisher64_minus_contrast64_evaluation_tangent_recovery|-0.038380|[-0.10395531398497775, 0.02719586484668428]|

Gates: {'energy_identity': True, 'H408_discovery_Fisher_replay': True, 'discovery90': False, 'transfer_equivalence05': False, 'finite_equivalence05': False, 'tangent_over_contrast05': False}.

|Ranking|Factors|Discovery tangent recovery|Evaluation tangent recovery|Finite unit recovery|Discovery attribution fraction|Discovery cross/F|
|---|---:|---:|---:|---:|---:|---:|
|fisher|16|0.474619|0.473484|0.484263|0.432394|0.084449|
|fisher|64|0.668427|0.673223|0.655703|0.770835|-0.204816|
|fisher|256|0.771148|0.770847|0.757383|0.953875|-0.365454|
|contrast|16|0.273237|0.284030|0.296272|0.263273|0.019928|
|contrast|64|0.707502|0.711603|0.711173|0.681523|0.051958|
|contrast|256|0.803631|0.809715|0.803387|0.782236|0.042791|
|individual_energy|16|0.294549|0.304659|0.316423|0.256792|0.075515|
|individual_energy|64|0.701608|0.711905|0.712161|0.677536|0.048144|
|individual_energy|256|0.844099|0.854300|0.845411|0.863899|-0.039600|

Audits: {'max_relative_energy_closure': 3.262952747621803e-14, 'H408_discovery_full_energy_relative_replay': 9.317843437784663e-15}. Independent saved-array checks: {'energy_identity_max_relative': 3.262952747621803e-14, 'all_discovery_subset_attributions_replay_H408_scores_max_absolute': 2.168404344971009e-18, 'recovery_identity': True}. The complete per-site retained, omitted, cross and attribution energies and all bootstrap samples are saved. Nonprimary table comparisons are descriptive.

Outcome: numerical identity and replay gates pass; all four scientific gates fail. Fisher64 tangent recovery is already only0.668427 on the discovery panel, with joint95% interval[0.602852,0.734003]. Thus neither cross-document transfer nor finite-edit nonlinearity is necessary for the low reconstruction: the deficiency is present in the local metric on the documents used to rank factors.

Evaluation tangent recovery is0.673223. Evaluation−discovery is0.004796,[−0.060780,0.070371]; tangent−finite unit recovery is0.017520,[−0.048056,0.083096]. These point differences are small, but the frozen±.05 equivalence gates fail. Do not claim equivalence or rule out meaningful transfer/nonlinearity contributions. Evaluation Fisher64−contrast64 tangent recovery is−0.038380,[−0.103955,0.027196]; this local comparison does not independently establish the ordering, unlike H408's paired finite comparison.

The signed-attribution discrepancy is algebraically visible. On discovery, Fisher64 attribution/full energy is0.770835, retained energy/full is0.873243, and retained–omitted cross/full is−0.204816. Hence reconstruction is2×0.770835−0.873243=0.668427: the selected response is partly canceled by omitted factors. At256, attribution fraction0.953875 coexists with reconstruction0.771148 and retained energy1.136602; cross/full is−0.365454. On evaluation the same qualitative pattern remains: Fisher64 attribution0.778944, reconstruction0.673223, cross/full−0.211442. These accounting comparisons are descriptive, without newly selected confidence claims. They demonstrate why summed signed attribution cannot be treated as subset fidelity; they do not prove a unique sparse compensator set or that cancellation is the only limitation.

Raw contrast64 has positive discovery cross/full0.051958, and individual-energy64 has0.048144, compared with the negative Fisher64 value. At16 factors Fisher has positive cross/full0.084449 and descriptively outperforms raw rankings; the signed-ranking failure develops as the retained set grows. This supports considering combinations of factors, rather than identifying a fixed number of highest individual scores with computational units.

Hypothesis narrowing: #5 (factor cancellation/cooperation) now has direct support as the accounting explanation of the signed-attribution/reconstruction gap. #13 (context-dependent support) is not required for this failure, although transfer equivalence remains unproven. #17 (finite-edit mismatch) is likewise not required here. #3/#4 remain unresolved in native-output space: these are failures of particular fixed subsets, not proof that no small subset exists. #16 remains valid: the choice of output metric matters, but a better metric alone does not make a correlation-based ranking a good subset objective.

Next discriminating experiment: freeze the Fisher64 core and rank omitted factors on discovery by their signed correlation with the core's omitted response in the same Fisher metric. Add64 factors without changing native output vectors, and compare at128 total factors with the original Fisher128, raw-energy128, contrast128 and six random64 additions to the identical core. Include a 64-factor control chosen to maximize the original full-response attribution among omitted factors (identical to Fisher128) to isolate the benefit of residual-aware ranking. Score exact tangent reconstruction on both panels and finite native KL/clamping on evaluation. A successful targeted rescue would support computational groupings with compensating factors; failure would favor more distributed support or the need to account for candidate self-energy. Rank by residual correlation alone is not an exact greedy squared-error optimizer, so label it accordingly. Freeze improvement gates and selection before results, and use fresh-document confirmation before promoting any successful grouping.

The broad research goal remains open. No new documents were consumed; the next fresh panel must exclude7872 complete-document hashes. No experiment processes remain running.
