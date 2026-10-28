## H617 — Missing directions matter on unseen constructions; nonlinear effects grow with dose

H616 showed frozen copying coordinates transfer poorly across constructions despite perfect copying accuracy. To distinguish behaviorally irrelevant residual directions from nonlinear interactions, decompose every Attention5 contextual interaction I into P (its projection onto frozen H615 joint64) and Q=I−P. Subtract P, Q or I across body positions at doses 0.25, 0.5, 1 and 2. Use the same 128 combined-condition recipients, giving 1,536 interventions plus128 native baseline evaluations. No fit, selection or additional natural text.

The previous goal turn was progress: H616 completed new experiments and changed the hypothesis ranking. This turn directly tests the alternatives raised by that evidence. The broad goal remains incomplete.

**The complement is not generally behaviorally negligible.** At dose1, complement-only native-reference KL is 0.5–1.3% of full-removal KL in familiar constructions, but 29–36% in the unseen sentence and 22–23% in the list. Ratios compare mean KL within each cell; they are not fractions of an additive causal effect. All copied answers remain top1. Thus output distributions reveal an effect that copied-answer accuracy misses.

| Names | Form | P-only KL / full KL | Q-only KL / full KL | Internal cross-term / full shift | Weighted logit cross-term / full shift | Full KL (nats) |
|---|---|---:|---:|---:|---:|---:|
|Familiar|Sent a letter|0.9978|0.0097|0.0089|0.0117|0.004371|
|Familiar|Letter delivered|0.9679|0.0117|0.0096|0.0134|0.003875|
|Familiar|Mailed a parcel|1.3414|0.3627|0.0607|0.1724|0.001161|
|Familiar|Name list|0.3152|0.2323|0.0889|0.1006|0.007740|
|Different|Sent a letter|1.0367|0.0125|0.0113|0.0158|0.002995|
|Different|Letter delivered|0.9682|0.0051|0.0125|0.0132|0.004115|
|Different|Mailed a parcel|1.2802|0.2880|0.0618|0.1539|0.001217|
|Different|Name list|0.2995|0.2151|0.0854|0.0758|0.009722|

The factorial cross-term at a downstream vector z is z(full removal)−z(P removal)−z(Q removal)+z(native). A zero cross-term means the two separate perturbation effects add. Internal cross-terms use the normalized MLP11 input; logit cross-terms remove the probability-weighted common offset and use native-probability-weighted RMS over the vocabulary. This weighting is a local distribution-sensitive diagnostic, not exact KL. A second check compares actual full-removal logits with logits predicted by adding the two separate shifts, using exact softmax KL.

**Finite-strength nonlinearity is present but is not the whole failure.** In unseen constructions, relative logit cross-terms are about 1.7–4.4% at quarter dose, 7.6–17.2% at dose1, and 14.5–30.6% at dose2. Internal cross-terms likewise grow from about 1.6–2.3% to 6.1–8.9% to 11.5–16.8%. The missing directions already produce appreciable KL at quarter dose, where vector/logit additivity is closer. That weakens an explanation in which only large-intervention nonlinearity creates the transfer gap.

| Names | Unseen form | Dose | Logit cross relative | Additive prediction KL | Full native-reference KL |
|---|---|---:|---:|---:|---:|
|Familiar|Mailed a parcel|0.25|0.0438|0.00000014|0.00007710|
|Familiar|Mailed a parcel|0.5|0.0878|0.00000214|0.00029983|
|Familiar|Mailed a parcel|1.0|0.1724|0.00003086|0.00116060|
|Familiar|Mailed a parcel|2.0|0.3063|0.00036499|0.00473328|
|Familiar|Name list|0.25|0.0232|0.00000015|0.00036517|
|Familiar|Name list|0.5|0.0482|0.00000289|0.00159048|
|Familiar|Name list|1.0|0.1006|0.00006227|0.00773990|
|Familiar|Name list|2.0|0.1865|0.00134969|0.04671363|
|Different|Mailed a parcel|0.25|0.0386|0.00000012|0.00008583|
|Different|Mailed a parcel|0.5|0.0775|0.00000184|0.00032749|
|Different|Mailed a parcel|1.0|0.1539|0.00002681|0.00121679|
|Different|Mailed a parcel|2.0|0.2818|0.00032547|0.00461949|
|Different|Name list|0.25|0.0171|0.00000013|0.00048593|
|Different|Name list|0.5|0.0356|0.00000239|0.00208166|
|Different|Name list|1.0|0.0758|0.00004891|0.00972242|
|Different|Name list|2.0|0.1449|0.00097304|0.05342816|

For the unseen sentence, P-only KL exceeds full-removal KL (1.28–1.34 times at dose1), consistent with partial cancellation between component effects. For the list, both separate KLs are smaller than full-removal KL. Neither comparison alone measures nonlinearity: KL is nonlinear even if logits add exactly. The direct factorial logit test above is what diagnoses nonadditivity. A weighted cosine between separate logit shifts would be a more direct cancellation measurement; it was not computed here.

At dose1, additive-logit prediction KL is around 0.000027–0.000062 nats on unseen forms, compared with full native-reference KL about0.0012–0.0097. At dose2 the prediction discrepancy grows to about0.00033–0.00135 nats. These are small absolute distribution changes; describing relative errors without that scale would exaggerate their practical size. No task accuracy change is established.

Updated hypothesis ranking: (1) construction-dependent output directions omitted by a familiar-format fit remains best supported; (2) downstream sensitivity weights those directions differently, so an internal Euclidean basis does not automatically preserve output KL; (3) finite-dose component interactions contribute, especially under format shift; (4) the complement is universally irrelevant is weakened; (5) preservation of this interaction is necessary for correct copying remains unsupported. This is a conditional representation of one interaction, not a discovered universal feature ontology or executable compression of the model.

Next move the same frozen-space/complement test to genuinely different tasks: grammatical agreement, entity binding, capital recall, sentiment, with copying anchors. Use fixed templates and initial-case/prefix conditions, audit token alignment before inference, report candidate-choice and full-vocabulary competence separately, retain failed competence cells, and do not fit new bases to rescue failures. This distinguishes copying-format dependence from broader task dependence. Any excluded token-incompatible prompt must be reported before model-output selection; use semantic-group NLL where multiple accepted tokens are needed and keep that distinct from single-token CE.

Audits: all source/manifest/basis/runner hashes unchanged; native replay exact; maximum P/Q cosine6.70e-7 and sum relative error2.03e-8; 1,536 interventions and512 factorial comparisons finite; dose1 span results reproduce H616 internal error and relative KL within1e-6; every copied answer stays correct. The runner was rerun after adding explicit full-reference omission-KL denominators to the output schema; this changed no experimental conditions. FineWeb count remains26,112. Broad goal active and incomplete.
