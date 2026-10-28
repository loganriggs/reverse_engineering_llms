## H643 — The transferable context-shift effect is distributed and composition-dependent

H642 was progress: frozen context shifts predicted component effects on fresh items and independently sampled directions without refitting. H643 tests whether that transferable effect belongs mostly to one later layer or to a distributed composition.

Keep the frozen H641 shifts, H64264-prompt panel and four probes. Evaluate eight fixed single-layer interventions and eight fixed leave-one-layer-out interventions across MLP10–17, with both receiver/donor incoming tangents and readouts. Retain the cached no-shift and full-shift controls and replay the full intervention independently on the original probe. No subset is selected, no parameters are fitted and no prompts are removed. This is descriptive localization on a now-observed test panel, not a new blind generalization test.

Primary ablation error is norm(ablated response − full predicted response) divided by norm(full predicted response − no-shift response), using receiver-probability-weighted centered logits. No shift has error1 and the full predictor error0. A separate metric measures error relative to the exact donor-context intervention, preserving the distinction between the predictor's effect and the true context-replacement effect.

**No single layer reproduces most of the full effect at high fidelity.** Under donor incoming/readout conditions, even the lowest mean single-layer error is0.825 for matched copying,0.885 for broad copying,0.889 for matched facts and0.910 for broad facts. These minima are descriptive across a fixed tested set, not selected models with validated future performance. The remaining single-layer errors are generally closer to1. A small local intervention can contribute to the complete effect without being sufficient in isolation.

**Every leave-one-layer-out condition changes the full predictor appreciably on average.** Mean omission errors range0.198–0.453 for matched copying,0.258–0.437 for broad copying,0.263–0.427 for matched facts and0.247–0.382 for broad facts. This supports distribution over several layers but does not prove eight layers are a minimal sufficient circuit: multiple omissions can interact or cancel, and a smaller jointly refitted representation was not tested.

| Omitted layer | Copy matched error | Copy broad error | Fact matched error | Fact broad error |
|---|---:|---:|---:|---:|
|10|0.453|0.437|0.427|0.310|
|11|0.404|0.382|0.367|0.301|
|12|0.328|0.341|0.311|0.282|
|13|0.280|0.297|0.265|0.247|
|14|0.246|0.258|0.279|0.249|
|15|0.221|0.262|0.263|0.280|
|16|0.198|0.260|0.264|0.373|
|17|0.213|0.337|0.299|0.382|

The relative influence varies by contrast: earlier layers10–11 are especially consequential for matched copying and facts, while layers16–17 are substantial for broad factual transfer. Endpoint choices also affect rankings. Receiver-only incoming/readout conditions yield the same broad conclusion: best single-layer mean errors remain0.825–0.889, and omission errors span roughly0.19–0.45 across cells. Thus distributedness is not solely an artifact of supplying donor endpoints.

**Adding independent single-layer effects does not reproduce their composition.** Construct the additive control as the native response plus the sum of eight single-layer response changes. The final readout is linear in the propagated tangent, so adding final tangents gives exactly that additive local-output control. Compare with the actual full chain:

| Condition, donor endpoints | Additive error versus full predicted effect | Additive error versus exact context effect | Full predictor error versus exact context effect |
|---|---:|---:|---:|
|Copy matched|0.276|0.495|0.416|
|Copy broad|0.646|0.763|0.454|
|Fact matched|0.326|0.486|0.388|
|Fact broad|0.626|0.634|0.202|

Matched instruction changes have a substantial but smaller nonadditive remainder (0.276/0.326) than broad format changes (0.646/0.626). The full composition predicts the exact intervention better than the additive approximation in all four averaged cells. Under receiver endpoints, additive-versus-full errors are0.278/0.619 for matched/broad copying and0.325/0.680 for matched/broad facts. This interaction is therefore present without donor input or readout.

The interaction arises within a local linear response calculation. Each layer's derivative changes linearly with its context shift, but multiplying shifted layer maps introduces products of shifts from different layers. It should not be described as finite-amplitude activation curvature or as an irreducible semantic synergy. Additive effects still carry useful information; they simply do not preserve the full predicted response closely enough in the broad-format contrasts.

The current result favors a reusable distributed transformation over a single localized “instruction neuron” or one independent layer contribution. Yet the eight vectors are a parameterization, not a demonstrated fundamental unit. The experiment says how this predictor composes, not which biological or semantic interpretation of those vectors is uniquely correct. All task-competence and component-versus-full-computation limitations from H642 remain in force.

Next test interaction order directly. Because each fixed-state MLP Jacobian is affine in a scalar multiplying its learned context shift, the eight-layer tangent chain is a polynomial of degree at most eight in that scalar. Propagate exact coefficients and compare first-, second- and third-order truncations against the full chain, with the full polynomial as a numerical recovery control and first order checked against this additive result. This can determine whether pairwise cross-layer composition captures the transferable effect or whether higher-order interactions remain necessary, without selecting a favorable subset on test results.

Audits:no new whole-model forwards;512 cached attention-stage reconstructions;6240 batched JVP calls; full-predictor replay exactly matches H642; all source/runner/model hashes unchanged; finite metrics. All fixed ablations, prompts, probes and endpoint combinations retained. No new natural-text documents; FineWeb remains26,112. Broad goal active and incomplete.
