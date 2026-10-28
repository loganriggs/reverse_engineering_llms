## H649 — Receiver contexts predict some residual coefficients without fresh donor access

H648 was progress: increasing discovery coverage improved both the mean shift and a fixed-rank residual span, but the residual corrections still used oracle donor-informed coefficients. H649 tests coefficient prediction from receiver information alone.

Keep the H648 eight-item means and rank3 residual bases fixed. For each family and directed template contrast, use the eight discovery receivers' concatenated normalized MLP contexts10–17 as features. Targets are their three residual-basis coefficients computed from discovery donor–receiver shifts. Center receiver features within that discovery group; form a linear kernel divided by its discovery mean diagonal; solve ridge regression with fixed regularization alpha1. No regularization sweep, fresh-data tuning or output-response fitting occurs. A control rotates discovery coefficient labels by four item positions, leaving feature geometry and regularization unchanged.

Evaluate on all64 H642 fresh prompts and four probes. Predicted contexts equal the eight-item mean shift plus the predicted rank3 correction, added to the receiver's own native MLP contexts. Fresh donor contexts are used only to score coefficient accuracy and provide the exact/oracle comparison targets. They do not enter the ridge prediction. Known family/template identity chooses the predictor; this is not task discovery or prediction for unseen instruction contrasts. Receiver native states from all eight later layers remain required.

The fresh panel has been inspected in prior experiments, so this is a retrospective evaluation with discovery-only fitting and a predeclared regularization rule, not a new blind replication. The basis and all predictor parameters are learned only from H640 discovery items. No held-out donor target enters fitting or parameter selection.

**Coefficient prediction has signal beyond a zero correction.** Aggregate coefficient relative RMSE (sqrt of summed squared prediction error divided by summed squared oracle-coefficient norm) is:

| Condition | Receiver ridge | Shuffled-label control |
|---|---:|---:|
|Copy matched|0.797|1.045|
|Copy broad|0.585|0.994|
|Fact matched|0.367|1.218|
|Fact broad|0.516|1.157|

Zero coefficients, corresponding to the mean-only model, have coefficient RMSE1. The correct-label model beats that baseline in each cell. Shuffled labels are generally near or worse than1; broad copying is just below1, so that control is not uniformly harmful by every metric. Coefficient error and functional error must be evaluated separately.

**The predicted coefficients produce modest functional improvement.** Error is measured against the exact context-replacement effect, normalized by that effect's magnitude under receiver-probability-weighted centered logits. Donor incoming/readout results:

| Condition | Eight-item mean | Oracle rank3 correction | Receiver-predicted correction | Shuffled-label prediction |
|---|---:|---:|---:|---:|
|Copy matched|0.371|0.326|0.358|0.379|
|Copy broad|0.419|0.379|0.397|0.420|
|Fact matched|0.374|0.351|0.356|0.382|
|Fact broad|0.198|0.190|0.193|0.203|

Receiver-only incoming/readout results:

| Condition | Eight-item mean | Oracle rank3 correction | Receiver-predicted correction | Shuffled-label prediction |
|---|---:|---:|---:|---:|
|Copy matched|0.365|0.320|0.352|0.372|
|Copy broad|0.388|0.345|0.358|0.382|
|Fact matched|0.372|0.348|0.353|0.383|
|Fact broad|0.204|0.195|0.198|0.208|

The receiver-predicted correction improves199/256 donor-endpoint cases and211/256 receiver-endpoint cases over the eight-item mean. It improves all four cell means in both conditions and outperforms shuffled prediction in each cell. These are repeated prompt/probe evaluations, not independent trials or a statistical significance claim. Shuffled prediction slightly improves receiver-endpoint broad copying over the mean, which cautions against treating every small cell improvement as mechanistic evidence by itself.

The learned correction captures part of the oracle correction's benefit, especially in factual cells, but does not eliminate residual mismatch. Oracle coefficients optimize Euclidean context reconstruction rather than output fidelity, so their functional scores are a comparison, not a guaranteed lower bound. The predictor is not chosen to beat the oracle on individual examples, and no fresh labels are used to adjust it.

This moves the residual component beyond donor lookup: a small coefficient map learned from receiver contexts transfers to different items and new probe directions. The original mean model already captured a stable component; the receiver-dependent correction adds limited item-specific structure. However, the model still uses full native receiver activations from the very segment whose operator change it describes, plus known template/family and the original network. It is a conditional description of counterfactual computation, not an end-to-end surrogate, cheap forward replacement or unique canonical feature system.

A practical question now is whether those useful coefficients can be predicted earlier. Access to all normalized MLP inputs10–17 may include information developed inside the modeled segment. That is not donor leakage, but it limits causal and computational interpretations of the predictor. Next compare the same fixed-regularization coefficient model using only the receiver residual leaving block9 against the current later-context feature set. Keep the discovery means/bases, label definitions, fitting items and fresh evaluation fixed, and do not tune on fresh outputs. If upstream features retain the benefit, the correction can be specified before the later computation; if they do not, local state dependence remains an essential part of the description. A future blind item replication is still needed before treating this new predictor as prospectively established.

Audits:512 fresh plus512 discovery cached attention-stage evaluations;768 batched JVP calls; no new whole-model forwards. Expanded mean reconstruction matches H648 exactly; ridge normal-equation residual below3.56e-15; fixed alpha1; all source/runner/predictor hashes unchanged; finite metrics. All fresh contexts, probes and endpoint choices retained. No new natural-text documents; FineWeb remains26,112. Existing competence and local-derivative limitations remain. Broad goal active and incomplete.
