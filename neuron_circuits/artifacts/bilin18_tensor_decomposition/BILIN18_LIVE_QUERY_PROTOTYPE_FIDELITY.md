## H931 — Live task-blind query prototypes preserve substantial causal behavior on familiar templates

H930 found useful query clusters but tested only local writes. H931 freezes those prototype centers and coefficient tables and evaluates live joint replacement of the persistent components at L8H3, L13H0 and L14H4. At each reader, select the nearest prototype from the current normalized pre-rotary query pair, so earlier edits can alter later gates. Gating uses no task/template labels. Current-layer values and the rest of the model remain live.

Evaluate1,4,8 prototypes across all three seeds, plus native routing, zeroing and the H929 supplied-context/positional baselines. All96 transfer examples, both templates and all task/copy families are retained. No best-seed or best-count selection occurs. These are familiar templates and previously studied lexical pairs; this is not a fresh generalization test.

### Native and intervention fidelity

Native KL measures the predicted native-value distribution against the unmodified model. Edited KL compares the predicted donor-value output with the native-routing donor-value intervention. Effect error compares the donor-minus-native answer-margin response, normalized by the reference effect RMS. Table ranges cover all three seeds.

|Domain/family|Prototypes|Native KL|Edited KL|Effect error ratio|
|---|---:|---:|---:|---:|
|agreement task|4|0.0005–0.0007|0.0005–0.0007|1.1389–2.9572|
|agreement task|8|0.0002–0.0004|0.0003–0.0004|1.9168–2.3178|
|agreement copy|4|0.0092–0.0117|0.0088–0.0118|0.2391–0.2919|
|agreement copy|8|0.0093–0.0103|0.0081–0.0090|0.2780–0.3098|
|language task|4|0.0325–0.0353|0.0467–0.0541|0.1630–0.1795|
|language task|8|0.0312–0.0329|0.0447–0.0467|0.1588–0.1630|
|language copy|4|0.0889–0.0934|0.0603–0.0643|0.4701–0.4853|
|language copy|8|0.0266–0.0300|0.0361–0.0464|0.4344–0.4936|
|binding task|4|0.0134–0.0160|0.0130–0.0137|0.0890–0.1442|
|binding task|8|0.0117–0.0144|0.0114–0.0137|0.0810–0.0852|
|binding copy|4|0.0335–0.0355|0.0116–0.0122|0.3574–0.3683|
|binding copy|8|0.0061–0.0066|0.0018–0.0026|0.0998–0.1049|

Eight prototypes preserve binding task effects with8.1–8.5% error, compared with82.6% for the generic distance rule and6.5% for supplied-context means. Language task error is15.9–16.3%, near supplied-context16.7% and far below generic88.9%. These results establish useful causal fidelity for this partial surrogate, rather than merely stable clustering.

Copying is uneven. Binding copying improves from roughly36% error with four prototypes to10.0–10.5% with eight, near the supplied-context10.0%. Language copying remains43.4–49.4% at eight prototypes. Agreement task still has a tiny reference effect (0.00563 logits RMS) at these readers; its large relative errors should not be interpreted as evidence about the complete agreement circuit.

The one-prototype models exactly reproduce the pooled-distance baseline for all seeds, which is an independent implementation sanity check. All H929 baseline metrics also replay exactly. Thus gains at larger prototype counts cannot be attributed to a changed baseline implementation.

### Live gates and limitations

Only20 of5,184 prototype gate evaluations switch cluster relative to the unedited query state; none require a fallback for an uncovered distance. This verifies that gates were recomputed, but also shows these interventions offer a weak stress test of gate dynamics. More challenging edits may cross boundaries more often.

The surrogate is task-blind in fitting and gating but still uses native query computations, pretrained token-value vectors and native current-layer values. It approximates a persistent attention component; it does not replace whole heads or explain how query states arise. Prototype count, stable partitions and intervention fidelity do not yet demonstrate semantic manifolds. Surface form, final-token identity and lexical category can still explain the clusters.

### Next test

Refit with one template excluded and evaluate the excluded template with the other lexical half held out as well. Run both template directions, preserve all predetermined prototype counts/seeds and account for any unsupported relative positions without fitting on evaluation data. Compare with the same models' within-template lexical transfer. This distinguishes template memorization from reusable context organization. Independently refit on disjoint discovery samples to test partition stability beyond random initialization. Failure should identify which context distinctions need richer featurizers rather than justify adding labels after the fact.

### Verification

2,784 full-model forwards. Identity replay exactly zero; native-cache replay within1e-4; fit/evaluation indices disjoint. Frozen prototype fit indices match the original discovery split. All H929 baseline metrics and one-prototype/positional equivalence checks match exactly. Source hashes unchanged; all outputs finite. Every seed/count, gate decision and endpoint metric is saved. No reserved test or added natural documents; pool remains26,112. Broad goal active.
