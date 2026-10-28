## H719 — Task-transfer positive controls emerge at middle and late blocks

Previous goal turn classification: progress. H718 established that neither full early residual replacement nor its retained/complementary span supplied a task-transfer positive control. H719 maps the same intervention across depth instead of treating that failure as evidence against all featurizers.

Capture the final-position residual after each of18blocks on the fixed80competence-eligible task pairs and nuisance variants. At one block replace only the receiver's final-position residual with the donor's; preserve the receiver's other positions, persistent value bus and input pathway. Evaluate both task directions and nuisance0<-2for every block. This tests whole-state sufficiency at a specific interface, not necessity or the origin of the computation.

First test block17, the final block: donor residual replacement must reproduce donor logits because only the shared output normalization/readout remains. It does, within2.01e-5maximum absolute logit error over all240swaps. This is an architectural sanity check, not a discovered mechanism.

Define answer-margin recovery as mean donor-directed margin change divided by the mean available native donor-minus-receiver margin difference. This is a ratio of means, not accuracy or recovered mechanistic information. Freeze a discovery rule: earliest block with at least50%margin recovery and positive task-minus-absolute-nuisance score. Evaluate that same block on transfer wording without reselection.

| Family | Discovery-selected block (zero-indexed) | Discovery margin recovery | Transfer margin recovery | Transfer donor Top1 | Native donor Top1 | Transfer specificity95%interval |
|---|---:|---:|---:|---:|---:|---|
|agreement|11|93.9%|94.0%|50.0%|56.2%|[6.725,8.127]|
|capital_recall|13|97.8%|74.6%|56.2%|87.5%|[2.686,7.854]|
|entity_binding|14|93.5%|95.1%|100.0%|100.0%|[6.833,7.621]|
|sentiment|9|52.8%|54.7%|43.8%|68.8%|[0.938,1.413]|
|sequence_copy|9|50.3%|63.5%|25.0%|56.2%|[2.744,5.240]|

All five discovery-selected blocks retain more than50%margin recovery and positive specificity on transfer wording. Pair-bootstrap intervals resample8transfer pairs, keeping both task directions and nuisance together;10000resamples, exploratory and not multiplicity-adjusted. Native donor competence constrains meaningful Top1transfer: agreement's semantic answer sets achieve only56.25%nativeTop1, for example. Margin recovery is not equivalent to switching the final answer: copy reaches63.5%margin recovery atblock9but only25%donorTop1.

![Task transfer across depth](/home/loganriggs/Documents/Codex/2026-10-21/he/outputs/task_depth_transfer.png)

The early last-position residual is a poor transfer interface for these tasks. Responses strengthen later, with pronounced transitions nearblock11for agreement,13for capitals and14for binding; sentiment and copying develop more gradually and cross the discovery criterion atblock9. Copy's alternate wording shows an earlier rise aroundblock8, illustrating context dependence. The full curves are retained, including nonmonotonic changes and modest overshoot beyond100%on some discovery points.

This gives a useful causal target for representation work: candidate featurizers can be tested at sites where whole-state transplantation already transfers a predefined answer contrast and survives alternate wording. It does not establish that the task is first computed at the selected layer. Earlier computation could occur at other positions or in the value bus, and subsequent attention can collect it into the final-position state. Nor is an entire transferable residual a fundamental unit; it may contain distributed information, output preparation and incidental state.

Next decompose the relevant block transitions into their actual incoming-state, attention and MLP contributions, following the model's exact update equations rather than assuming a standard residual block. Audit that the contributions reconstruct the native state. Then intervene on those contributions at discovery-selected blocks, with the same task/nuisance and transfer controls. This distinguishes where information is read from where it is transformed, and identifies computationally relevant activations for the next manifold/featurizer comparison. Keep early-layer/value-bus pathways in scope; do not equate later final-position sufficiency with absence of earlier structure.

Audits:4560whole-model prefix evaluations,240native and4320swaps. Native-logit replay againstH717maxabsolute1.86e-5; block1state replay relative8.49e-7; block1swap answer-margin replay againstH718maxabsolute8.59e-6. Final-block donor-logit replay2.01e-5. All source/model/input hashes unchanged, finite metrics. No original corpus reserved-test evaluation or new unique research documents, pool26112. Broad goal active.
