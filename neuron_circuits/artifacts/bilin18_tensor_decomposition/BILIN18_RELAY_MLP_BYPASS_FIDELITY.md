## H945 — Relay MLP changes matter downstream, with substantial residual-path interaction

Replay all312 H944 full/rank1 task jobs in the complete model. At the selected layer4 relay position, independently add (a) raw input difference to the residual output, (b) exact normalized-MLP finite update difference, or (c) both. Their sum is the exact full preMLP replacement. Student arms substitute their predicted MLP difference while retaining the native recipient MLP baseline, with/without the exact residual difference. This isolates finite-change fidelity; it is not native replacement by the student.

| Family/arm | Residual-only margin RMS | MLP-only RMS | Joint RMS | Joint-minus-sum RMS |
|---|---:|---:|---:|---:|
| Agreement full |.466|.657|2.099|1.092|
| Agreement rank1 |.459|.511|1.864|.983|
| Sentiment full |.102|.697|.933|.154|
| Sentiment rank1 |.108|.612|.841|.134|

These RMS magnitudes are not additive causal percentages. Downstream nonlinear interaction is substantial, especially agreement. The local residual-plus-MLP vector decomposition is exact; nonadditivity arises after that state enters the rest of the model. MLP-only changes are behaviorally significant, especially relative to residual-only sentiment changes, so the proposed anchors are not merely local geometric targets.

The effect-trained H943 students miss95.1–97.3% of isolated MLP-induced answer-margin effects across tasks/arms/seeds. Preserving the exact residual delta while substituting student MLP deltas still leaves71.7–75.0% joint-effect error for agreement and83.4–86.2% for sentiment. Native/endpoint-trained fits and all per-cell records remain available; no best model was selected. This confirms that H944's local failure has downstream consequences on these established mechanisms.

Full-distribution KL to exact MLP-only interventions is .00355–.00636 for agreement and .0162–.0208 for sentiment. Joint-intervention KL is .0250–.0356 agreement and .0228–.0290 sentiment. These are absolute discrepancies, not KL improvements relative to dropping the MLP change; no zero-change KL baseline is claimed. Future trained-anchor evaluations should retain that full-distribution baseline explicitly as well as native replacement fidelity.

Verification:4992 expected/actual whole-model forwards, identity-logit replay max3.43e-5, exactjoint historical full/rank1 answer-margin replay1.79e-5. All hashes unchanged, all finite, no refitting and no reserved-test evaluation. All competence-passing H849 cases retained. Full-model computation downstream of the edited state is live; persistent value bus is unchanged. Natural-document pool remains26,112.

Next train the same student with explicit original-discovery mechanism anchors, keeping natural-data losses and matched endpoint controls. Only original H848 discovery contexts may supply supervised anchors; H849 lexical/context variants stay evaluation-only. Include per-task normalization so one task does not dominate by scale. Report supplied-mechanism recovery as supervised validation and evaluate generalization, native/full-output damage, residual-path interactions and unrelated-probe performance separately. A successful anchor fit will still not establish unsupervised discovery or general reusable units.
