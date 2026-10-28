## H967 — MLP12 predominantly transforms an already-present sentiment interaction

Previous turn was progress: H966 supplied a full adjective-polarity by negation panel and established a composed-valence signature with narrow native competence. H967 captures the actual normalized MLP12 input and the residual immediately before its RMS normalization on all96 locked states. The four familiar original-format quartets remain the sole native-gate-passing cohort; other cohorts remain diagnostics.

For a,n in {-1,+1}, write normalized input x=c0+a*ca+n*cn+a*n*ci and define B(u,v)=Down((Left u)*(Right v)). The actual MLP is ungated bilinear with an output bias. Its exact factorial interaction is:

C_out = [B(c0,ci)+B(ci,c0)] + [B(ca,cn)+B(cn,ca)].

The first bracket transforms interaction already present in normalized input (inherited); the second creates interaction from separately indexed adjective/negation components (local cross product). Bias contributes only to the constant. This is an exact finite-panel algebraic decomposition, not a globally identified causal ontology. Inherited includes any interaction introduced by RMS normalization, so it cannot all be assigned to earlier layers without a further trace.

On the competent familiar-A cohort, signed projection of inherited interaction onto total output interaction is99.620%; local cross-product projection is0.380%. These shares sum to one by algebra but are not fractions of whole-model computation or variance explained; they can be negative or exceed100% in general. In the frozen sentiment output coordinate, the local term similarly accounts for0.394% of the summed interaction coefficient. Raw pre-normalization input already has53.08% of its centered factorial squared norm in the interaction component, versus53.45% after normalization. Similar fractions do not prove normalization is irrelevant, but establish that interaction is present before it.

Each state's native MLP write was then edited by subtracting a*n times one or both output components, retaining native residual and attention paths. These interventions are algebraic output edits and may be off-manifold; they do not remove upstream interaction itself.

| Removed output interaction | Mean loss in intended-sentiment margin | Mean KL(native||edited) | Binary accuracy, 16 competent-cohort endpoints |
|---|---|---|---|
| None (identity) |0|0|100%|
| Inherited component |0.4062|0.006951|93.75%|
| Local cross product |0.00234|0.00000236|100%|
| Both |0.4086|0.007083|93.75%|

The inherited-term intervention has the much larger effect, consistent with the exact decomposition. Most endpoints still retain correct binary answers after removal, so MLP12 is neither the only relevant pathway nor established as necessary for every answer. Across failed native cohorts, inherited vector shares range98.98–99.62%; these are model-mechanism diagnostics, not evidence of semantic competence.

Independent verification: reconstructed actual writes using all weights and Down_bias (max relative error8.22e-7). Exact bilinear interaction identity holds to5.82e-15 relative. Separate direct evaluations of the actual double-precision MLP on synthetic inputs recover the local term by a four-corner contrast with ci omitted, and the inherited term by a symmetric difference around c0 along ci. These144 local MLP evaluations match saved terms within2.71e-8 relative. Thus the central result does not rely solely on a self-check of the expanded polynomial formula.

Runtime audits:480/480 whole-model forward examples (96 captures,384 edits), plus144 local MLP evaluations; capture replay exact; identity logit error2.19e-5; raw-to-normalized replay1.91e-6; exact four-state reconstruction9.09e-13 absolute; independent summary aggregation; finite outputs and unchanged sources. No training or reserved-test access; natural-document pool26,112.

Interpretation: on this panel, MLP12 primarily transforms already-composed input into a sentiment output direction. A strong causal site and a clean output direction did not identify the site that originally composes adjective polarity with negation. This distinction should inform featurizer objectives: result directions and reusable operations need separate validation.

Next: trace the locked quartet interaction through earlier residual, attention-output and MLP-output states. Verify the additive residual recurrence, including learned residual scales and normalization, before selecting an earlier source for finite interventions. This can distinguish attention importing composition from earlier-token states, MLP-local cross products, and normalization effects. H961 frozen context-bridge predictor evaluation remains pending.
