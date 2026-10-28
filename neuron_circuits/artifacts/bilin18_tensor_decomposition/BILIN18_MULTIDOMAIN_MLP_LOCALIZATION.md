## H906 — Language MLP influence localizes later than number; binding remains weak at single tested sites

The previous turn made progress: H905 improved binding contrast geometry but found weak L12 language/binding target-margin influence even under full MLP donor substitution. H906 screens all prespecified layers8–17, plus their combined substitution, on the same96 transfer endpoints. No layer is dropped or silently selected. Discovery/transfer naming follows earlier work; these examples are reused diagnostic data, not untouched confirmation.

### Intervention scope

At the final-query MLP site, replace the recipient's live normalized input contribution with its matched opposite-source donor's native normalized input contribution, using the exact bilinear output difference. Recipient attention, residual context, other positions and all unpatched computation remain live. For the combined condition, each layer receives its corresponding fixed donor-native contribution; this is a coordinated intervention, not the sum of individual-layer effects or a natural donor trajectory. It is not a whole-prompt swap or an ablation of all computation for the task.

### RMS expected-answer margin changes

RMS prevents positive and negative effects from canceling. It measures influence, not necessarily helpfulness or task necessity. Each cell summarizes16 endpoints (four pairs, two templates, both orders), with endpoint dependence retained.

|MLP intervention|Agreement task|Agreement copy|Language task|Language copy|Binding task|Binding copy|
|---|---:|---:|---:|---:|---:|---:|
|layer_8|0.292|0.591|0.149|0.331|0.088|0.253|
|layer_9|0.822|0.759|0.161|0.341|0.122|0.255|
|layer_10|0.285|0.264|0.099|0.386|0.161|0.204|
|layer_11|0.353|0.559|0.120|0.291|0.063|0.230|
|layer_12|2.166|0.732|0.120|0.277|0.070|0.216|
|layer_13|0.380|0.874|1.103|0.419|0.149|0.246|
|layer_14|0.211|0.392|0.950|0.553|0.168|0.229|
|layer_15|0.628|0.530|0.507|0.385|0.144|0.264|
|layer_16|0.422|0.685|0.511|0.421|0.170|0.366|
|layer_17|0.865|1.099|0.511|0.515|0.197|0.552|
|all|2.438|4.274|3.109|2.252|0.785|1.953|

Language task influence rises from L12 RMS0.120 to L13 1.103 and L14 0.950. Mean signed margin reductions there are1.009/0.790, and native-to-edited KL0.0351/0.0217 versus L12 0.00225. Matched-copy RMS effects are0.419/0.553. This supports later MLP sites as more informative language probes, though task/copy amplitude differences alone are not a normalized selectivity test.

Agreement retains its largest tested single-site RMS effect at L12 (2.166). Some late substitutions increase rather than reduce its expected-answer margin: mean signed drops at L15/L16/L17 are−0.343/−0.388/−0.629. A large RMS or KL effect is therefore not automatically evidence of removing a helpful component.

Binding's individual task RMS effects remain0.063–0.197, with the largest at L17. Combined L8–17 substitution reaches0.785 RMS and mean drop0.741, KL0.0413. This is compatible with distributed contributions or with important computation outside these final-query MLP sites. It does not establish either explanation, and it does not make single-site low-effect reconstructions convincing computational-unit evidence.

### Combined conditions and performance

All-layer task RMS effects are2.438 agreement,3.109 language and0.785 binding; corresponding KL values0.3269,0.3070,0.0413. All task endpoints retain correct binary target preference after these combined substitutions. Thus strong margin influence does not demonstrate sufficiency to switch the answer or necessity for task success.

Combined copying interventions are stronger than many single-site probes: binary correctness falls to12.5% for agreement-copy and93.75% for language-copy, while binding-copy remains100%. These effects reinforce the need for copying controls rather than interpreting every contextual perturbation as task-specific. No additive mediation fraction is inferred from single-versus-combined differences.

### Research decision

L12 inherited from the number investigation was a poor language localization choice. Next compare discovery paired-change and pooled subspaces at L13/L14, keeping L12 as a reference and explicitly treating the new layer choice as screening-derived. Ask whether low-dimensional subspaces recover actual donor-intervention effects there, with copying controls and held-out pairs. Later independent validation is required before a generalization claim.

For binding, avoid assuming the same MLP-local computation. Return to the previously identified attention/value-bus pathways or test a broader site intervention before investing further in a low-rank MLP unit. Shared-dictionary, manifold, and reusable-computation claims require evidence across these different mechanisms, not only a number case study.

### Verification

1,344 full-model forwards. Identity intervention yields zero logit difference. Replacing all ten MLP sites with each recipient's own cached native input gives maximum logit error1.17e−5; cached activation relative replay error5.98e−7. L12 target margins reproduce H905 exactly. Source/model/capture hashes unchanged and all KL finite. No natural or reserved-test evaluation; document pool remains26,112. Broad goal active.
