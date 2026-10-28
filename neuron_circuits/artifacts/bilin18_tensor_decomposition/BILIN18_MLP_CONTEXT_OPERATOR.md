## H557 — Context-dependent local operators transfer partially; output signs hide internal differences

Use H556's local bilinear formula on the same 80 prompts at ±1/16 edge amplitude. Replace only the native base context z in the linear response term with a donor context, retaining recipient normalized displacement e, its quadratic term, baseline MLP output, and live downstream computation. Donor rules are fixed before evaluation: self; opposite endpoint token; same source token in another format; matched item/endpoint in another vocabulary; and family-mean context excluding both endpoints of the recipient pair. No model fitting or output-based donor selection.

For binding, the other-format donor is name-prose copying; for copying it is the other copying format. Binding has no color counterpart, so its other-vocabulary cells are explicitly omitted. This gives 768 interventions, not a balanced full factorial. Donors change the bilinear response operator, not the entire residual state or the task answer. Recipient normalization dependence is deliberately retained.

| Family | Donor | Relative local-response error | Response cosine | Live-margin MAE | Live-effect sign agreement |
|---|---|---:|---:|---:|---:|
|binding|other_token|0.307|0.953|0.000322|100.0% (32)|
|binding|other_format|1.343|0.087|0.000829|100.0% (32)|
|binding|family_mean_leave_pair_out|0.208|0.978|0.000192|100.0% (32)|
|color_prose|other_token|0.491|0.863|0.002004|93.8% (32)|
|color_prose|other_format|0.812|0.808|0.003058|100.0% (32)|
|color_prose|other_vocabulary|0.881|0.541|0.005119|93.8% (32)|
|color_prose|family_mean_leave_pair_out|0.509|0.864|0.002663|93.8% (32)|
|color_record|other_token|0.523|0.842|0.001872|100.0% (32)|
|color_record|other_format|0.595|0.805|0.002129|100.0% (32)|
|color_record|other_vocabulary|0.921|0.432|0.005260|87.5% (32)|
|color_record|family_mean_leave_pair_out|0.553|0.848|0.002682|100.0% (32)|
|name_prose|other_token|0.477|0.878|0.001846|73.3% (30)|
|name_prose|other_format|0.836|0.702|0.001509|81.5% (27)|
|name_prose|other_vocabulary|0.814|0.620|0.002837|58.6% (29)|
|name_prose|family_mean_leave_pair_out|0.425|0.904|0.002040|72.4% (29)|
|name_record|other_token|0.658|0.756|0.004006|56.2% (32)|
|name_record|other_format|0.737|0.701|0.002399|62.5% (32)|
|name_record|other_vocabulary|0.968|0.466|0.005233|43.8% (32)|
|name_record|family_mean_leave_pair_out|0.585|0.810|0.004116|56.2% (32)|

A family-mean operator gives mean relative response errors of 21% for binding and 43–59% for copying. Opposite-token and cross-format operators also leave substantial error. Therefore a single average operator is not a faithful local-response model at these settings. This does not establish irreducibility: low-dimensional context modulation or other structured approximations have not been tested.

Vocabulary crossings produce larger local-response differences than several within-family controls, consistent with H554's vocabulary dependence. Name-copying output effects are particularly sensitive: vocabulary-swapped operators preserve only 17/29 prose and 14/32 record directions above the fixed 1e−4 threshold. Other tiny effects remain in magnitude metrics. For colors, signs remain mostly stable despite 88–92% mean relative local-response error under vocabulary crossings. These are correlated probes, not independent population estimates.

Binding offers a clear example of why fidelity metrics must be separated. A name-copying context yields 134% mean relative local-response error and cosine 0.087, yet retains all 32 live-effect directions. Its margin MAE is 0.000829 nats against mean absolute native effect 0.003561, so even this behavioral agreement is limited: it preserves signs, not exact margins, logits, KL or loss. A mean binding context is closer internally (21% relative error, cosine 0.978) and has smaller margin MAE, 0.000192.

This connects to data-based canonical forms: many internal responses could share a coarse output statistic while differing internally. A useful representation should state its equivalence criterion and evaluation distribution. Current results support neither a unique fundamental basis nor a KL/CE-equivalent simplification. They identify a concrete, testable candidate object: a context-conditioned local response operator whose value depends on whether we require internal reconstruction or downstream behavioral fidelity.

Next H558 should test whether a small number of context modulation directions predicts these operators on held-out lexical pairs. Fit only on discovery pairs; evaluate internal vector error, output margins and full-distribution KL separately. Include the family mean and exact operator as baselines, freeze ranks/criteria before evaluation, and retain binding/copying and vocabulary crossings. This tests compactness only after the causal object has been specified. Continue to track downstream layers and physical-position controls as unresolved; no small L8 model alone explains the complete circuit.

Audits: self operator reconstructs the prior live margins within 9.54e−6; all finite, source unchanged. H556's separate float64 audit verifies the bilinear algebra, while this test remains float32. H556–H557 complete 2,368 interventions on reused prompts, with no new FineWeb documents; tracked total 26,112. Broad goal active and incomplete.
