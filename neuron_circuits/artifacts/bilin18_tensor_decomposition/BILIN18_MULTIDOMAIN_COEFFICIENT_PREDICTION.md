## H929 — Context-specific coefficient means transfer, but generic positional rules miss causal behavior

H928 verified token-identity aggregation and full-distribution fidelity using donor routing. H929 asks whether simple coefficients fitted on discovery examples can replace live routing for the persistent component. This removes oracle donor coefficients from the prediction model, while retaining native token-value vectors and current-layer values.

### Design

Use agreement, language and binding with matched copying controls and two templates each. Fit the first four lexical pairs per domain (96 examples); evaluate the other four (96 examples). All evaluation examples were used in earlier research and are not pristine holdouts.

Three rules are fixed before evaluation. Context mean knows domain, task/copy family and template and averages each absolute position's coefficient. Distance mean pools discovery examples by distance from the final token. Family-distance adds only task/copy knowledge to that positional rule. Zeroing and native routing are controls. All rules act jointly on the persistent components of the previously fixed L8H3, L13H0 and L14H4 readers, across all source positions, retaining live current-value contributions.

Evaluate each rule with native persistent values and with values from the opposite lexical endpoint. The donor shares task/template and token alignment. Native-routing donor interventions remain live at later readers after earlier edits. For each rule, intervention effect is its donor-value output minus its own native-value output; compare this with the native-routing intervention effect. This distinguishes matching baseline predictions from preserving causal response.

Context recognition is supplied metadata here. No claim is made that the model has learned or that the experiment has discovered those context labels.

### Transfer results

Native KL compares native model with predicted native-value output. Edited KL compares the reference native-routing donor intervention with the rule's donor intervention. Effect error is answer-margin RMSE divided by reference intervention RMS; zeroing has ratio1 because it removes the value-dependent intervention response. Top1 is correctness of the predicted native-value output.

|Domain/family|Rule|Native KL|Edited KL|Reference effect RMS|Effect error ratio|Top1|
|---|---|---:|---:|---:|---:|---:|
|agreement task|zero|0.00085|0.00082|0.0056|1.000|87.50%|
|agreement task|context_mean|0.00024|0.00026|0.0056|2.318|87.50%|
|agreement task|distance_mean|0.00367|0.00361|0.0056|1.763|87.50%|
|agreement task|family_distance|0.00111|0.00108|0.0056|1.332|87.50%|
|language task|zero|0.95824|1.02424|9.8324|1.000|0.00%|
|language task|context_mean|0.03279|0.04767|9.8324|0.167|100.00%|
|language task|distance_mean|0.71962|0.84513|9.8324|0.890|12.50%|
|language task|family_distance|0.41026|0.58322|9.8324|0.728|25.00%|
|agreement copy|zero|0.03550|0.03781|0.5963|1.000|81.25%|
|agreement copy|context_mean|0.01287|0.01113|0.5963|0.325|93.75%|
|agreement copy|distance_mean|0.00667|0.00716|0.5963|0.473|93.75%|
|agreement copy|family_distance|0.01956|0.01809|0.5963|0.365|100.00%|
|language copy|zero|0.13949|0.32611|3.9181|1.000|100.00%|
|language copy|context_mean|0.02626|0.03366|3.9181|0.451|100.00%|
|language copy|distance_mean|0.08047|0.13948|3.9181|0.563|100.00%|
|language copy|family_distance|0.08589|0.06014|3.9181|0.444|100.00%|
|binding copy|zero|0.13198|0.11309|4.4726|1.000|100.00%|
|binding copy|context_mean|0.00564|0.00175|4.4726|0.100|100.00%|
|binding copy|distance_mean|0.05726|0.05702|4.4726|0.673|100.00%|
|binding copy|family_distance|0.03657|0.01410|4.4726|0.388|100.00%|
|binding task|zero|0.31996|0.10962|4.6246|1.000|37.50%|
|binding task|context_mean|0.00644|0.00419|4.6246|0.065|81.25%|
|binding task|distance_mean|0.05373|0.09406|4.6246|0.826|81.25%|
|binding task|family_distance|0.04460|0.05907|4.6246|0.670|81.25%|

For binding task, context means achieve native KL0.00644, edited KL0.00419 and intervention error6.5%. The generic distance rule has82.6% intervention error despite the same81.25% native top1. Family-distance remains weaker at67.0%. Therefore equal task accuracy can conceal very different causal fidelity.

Language task also benefits from context-specific means: native KL0.03279 and effect error16.7%, versus88.9% for distance and72.8% for family-distance. Binding copying transfers well (10.0% error); language copying is less well modeled (45.1%). The copying examples have identical template structure across domains but different lexical inventories, so differences among domain-conditioned means can also capture lexical-category effects. The experiment does not isolate abstract task context from lexical statistics.

Agreement task has reference intervention RMS only0.00563 logits at these selected readers. Its large relative errors are denominator-sensitive: context-mean absolute effect error is about0.0131 logits. Low native KL there does not establish an adequate agreement mechanism; earlier experiments implicated other pathways. Keep this low-signal control separate from the stronger language and binding effects.

### Implication and next experiment

A small contextual coefficient rule can preserve substantial behavior without compressing the lexical vectors themselves. However, one generic position rule is inadequate across these contexts. This motivates testing discrete or low-dimensional context states as activation features, rather than treating all lexical directions as one small manifold.

Next capture the final-token attention query activations at the fixed readers and fit task-blind query prototypes on discovery examples. Use predetermined prototype counts, multiple seeds and held-out lexical evaluation, without task/template labels in fitting or gating. Compare prototype-based coefficient prediction with the context-mean upper reference, generic positional rule, and native routing on both native KL and value-intervention fidelity. Audit the query/key attention formula before interpreting any learned groups. This tests whether useful unsupervised context labels can be recovered from model state, directly returning to the featurizer/manifold objective. Success would still need cross-seed stability, new-template transfer and selective interventions before a semantic manifold claim.

### Verification

1,056 full-model forwards. Identity replay exactly zero; native-cache replay passes1e-4. Fit/evaluation indices disjoint and saved; every prediction coefficient is fitted from discovery only. Source hashes unchanged; all metrics finite. Fits, donor-control outcomes and endpoint metrics retained. No reserved test or added natural documents; pool remains26,112. Broad goal active.
