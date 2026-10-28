## H650 — Residual correction is predictable before the modeled segment

H649 predicted three correction coefficients using receiver contexts throughout MLP10–17. H650 holds the discovery items, rank3 basis, eight-item mean shifts, coefficient targets, and fixed ridge regularization alpha1 constant, and replaces those features with the receiver final-position residual leaving block9. This state exists before the segment whose operator change is modeled. Eight discovery examples train each known family/directed-template group. Features are centered and the linear kernel is divided by its discovery mean diagonal. A shuffled-label control rotates targets by four items.

Prediction uses only receiver block9 state and known task/template identity. Fresh donor contexts supply evaluation targets, not predictor inputs. The predictor still depends on the original model and is not an end-to-end substitute. All64 H642 prompts and four probes remain in evaluation, including unsuccessful free-generation cases. This panel has been inspected previously: the comparison is retrospective, despite discovery-only fitting and no hyperparameter sweep.

**Upstream features retain useful predictive information.** Coefficient relative RMSE is sqrt(sum squared prediction error / sum squared oracle-coefficient norm); zero correction has error1:

| Condition | Block9 predictor | Shuffled labels |
|---|---:|---:|
|Copy matched|0.652|1.043|
|Copy broad|0.566|1.079|
|Fact matched|0.731|1.512|
|Fact broad|0.399|1.331|

All four groups beat zero correction and shuffled labels. Matched factual coefficient error is worse than H649's later-state predictor (0.731 versus0.367), even though output-effect predictions remain similar. Geometric coefficient accuracy and functional accuracy are different quantities.

Functional error compares predicted versus exact context-replacement logit derivatives, divided by the magnitude of the exact context-replacement effect. It uses receiver-probability-weighted centered logits. No-change error is1. Lower is better; these numbers are not accuracy or causal shares. The geometric oracle projects true donor context residuals onto the fixed rank3 basis; it is not an output-optimal lower bound.

**Receiver incoming tangent and readout:**

| Condition | Mean8 only | Geometric oracle | Later-state predictor | Block9 predictor | Shuffled block9 |
|---|---:|---:|---:|---:|---:|
|Copy matched|0.365|0.320|0.352|0.341|0.372|
|Copy broad|0.388|0.345|0.358|0.353|0.385|
|Fact matched|0.372|0.348|0.353|0.353|0.390|
|Fact broad|0.204|0.195|0.198|0.196|0.212|

Block9 prediction improves 206/256 prompt/probe cases over mean8. These repetitions are not independent statistical trials.

**Donor incoming tangent and readout:**

| Condition | Mean8 only | Geometric oracle | Later-state predictor | Block9 predictor | Shuffled block9 |
|---|---:|---:|---:|---:|---:|
|Copy matched|0.371|0.326|0.358|0.348|0.380|
|Copy broad|0.419|0.379|0.397|0.396|0.427|
|Fact matched|0.374|0.351|0.356|0.356|0.388|
|Fact broad|0.198|0.190|0.193|0.192|0.206|

Block9 prediction improves 199/256 prompt/probe cases over mean8. These repetitions are not independent statistical trials.

Block9 prediction improves all four cell means over the mean-only and shuffled-label baselines at both endpoints. Its output errors are close to or better than later-state prediction; receiver matched facts are slightly worse. The fixed-alpha comparison does not establish optimal feature sets or equivalence. It does establish that useful prediction need not read states inside the modeled segment for these tasks and templates.

The working hypothesis is a shared template-conditioned operator change plus a smaller receiver-dependent correction whose useful coefficients can be specified upstream. This remains a limited description of one component of local computation. The three-dimensional correction leaves substantial error; task/form identity is supplied, the original model is retained, and the experiments replace local derivative operators at fixed native states rather than demonstrating finite behavioral interventions. No unique canonical form or general semantic unit has been established.

Next freeze H650 predictor weights and H648 bases/means, then test another item panel with new probe directions before inspecting its results. Reuse all four template contrasts, retain every item after tokenizer eligibility checks, and evaluate coefficient error, local effect error, shuffled controls, and native competence separately. Keep copy and factual groups separately reported; broader task diversity remains necessary before a general claim. This blind replication has priority over more feature engineering on the already-inspected H642 panel.

Audits: no new whole-model forwards;512 fresh and512 discovery cached attention evaluations;768 batched JVP calls. Mean reconstruction exactly matches H648; ridge normal-equation residual7.11e-15; all source, runner and predictor hashes unchanged; all metrics finite. FineWeb unique document count remains26,112. Broad research goal active and incomplete.
