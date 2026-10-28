## H616 — Frozen coordinates transfer across names much better than constructions

Freeze H615's joint rank32/rank64 output bases and random64 control, with no refitting. Cross eight familiar name pairs and eight different pairs with two familiar constructions, a new sentence ("She mailed a parcel to …"), and a list ("Name: …"). Each has prefix presence × initial capitalization, giving 512 native prompts and 128 combined-condition recipients. Subtract full or projected Attention5 contextual-value interaction across all body positions, giving 640 interventions.

**Construction change is the main generalization failure in this panel.** Different names in familiar constructions have rank64 write errors 0.066/0.061 and later internal-effect errors 0.078/0.072. The unseen sentence has internal-effect error about 0.41; the list has about 0.76–0.77, similarly for both name groups. This weakens the idea that name memorization alone caused the previous format dependence. It does not isolate syntax from lexical scaffolding, punctuation, length, or position: these change together across constructions.

“Rank64” means 64 directions in the 1,152-dimensional output space. Write error compares the projected interaction with its full vector. Internal-effect error compares the downstream normalized MLP11 input after projected removal with that after full removal, divided by the change caused by full removal. Relative KL compares full output distributions against full removal, normalized by doing no removal. Relative token CE compares the intended answer's negative log probability on the same reference scale. These are approximation errors for this intervention, not task error rates, computational shares, or percentages of semantic understanding.

| Names | Form | Rank | Write error | Internal-effect error | Relative KL | Relative token CE | Omission KL (nats) |
|---|---|---:|---:|---:|---:|---:|---:|
|Familiar|Sent a letter|32|0.1045|0.1125|0.0282|0.2355|0.0042|
|Familiar|Sent a letter|64|0.0520|0.0607|0.0098|0.1055|0.0042|
|Familiar|Letter delivered|32|0.0863|0.0963|0.0339|0.2950|0.0039|
|Familiar|Letter delivered|64|0.0496|0.0574|0.0132|0.1893|0.0039|
|Familiar|Mailed a parcel (unseen)|32|0.4403|0.4431|0.3405|0.7937|0.0012|
|Familiar|Mailed a parcel (unseen)|64|0.3862|0.4093|0.5102|1.3087|0.0012|
|Familiar|Name list (unseen)|32|0.8012|0.8409|0.3231|0.4278|0.0077|
|Familiar|Name list (unseen)|64|0.7513|0.7641|0.2931|0.4707|0.0077|
|Different|Sent a letter|32|0.1099|0.1289|0.0688|0.4713|0.0029|
|Different|Sent a letter|64|0.0663|0.0778|0.0127|0.1574|0.0029|
|Different|Letter delivered|32|0.0914|0.1012|0.0148|0.1987|0.0041|
|Different|Letter delivered|64|0.0611|0.0724|0.0054|0.0991|0.0041|
|Different|Mailed a parcel (unseen)|32|0.4422|0.4480|0.2906|0.5617|0.0012|
|Different|Mailed a parcel (unseen)|64|0.3887|0.4057|0.3915|0.9821|0.0012|
|Different|Name list (unseen)|32|0.8058|0.8346|0.2578|0.3961|0.0095|
|Different|Name list (unseen)|64|0.7557|0.7675|0.2659|0.4553|0.0095|

The familiar-name/familiar-form cells are replay anchors and include H615 discovery names; they are not fresh held-out evidence. All other exact prompt strings are absent from H609's manifest. Different names means different from H615's fit, not unseen in the model's training data or all earlier experiments. Sixteen endpoints per form/name cell are related paired cases, not sixteen independent task families. The two new constructions cannot support a population-wide estimate.

**Internal reconstruction and behavioral fidelity separate.** Rank64 improves internal errors over rank32 on both unseen forms, but worsens relative KL for the unseen sentence: 0.341→0.510 for familiar names and 0.291→0.391 for different names. The sentence's omission KL is only about 0.0012 nats, so relative errors describe a small absolute output effect. For lists, internal-effect error remains around 0.77 while relative KL is about 0.27–0.29. Directions useful for vector reconstruction are not automatically best for output probability fidelity. Intended-token CE has another ordering; these objectives should remain explicit.

All 32 native competence cells have perfect paired candidate choice and full-vocabulary top1; all 640 interventions preserve correct copied-name top1. This demonstrates a representational transfer gap in a task the model can perform. It does not establish that this interaction is necessary for copying. The task is still copying throughout; construction diversity is not task diversity. Earlier diverse-task studies remain relevant, but this run alone cannot support cross-task claims.

Most likely explanations now are (1) shared but construction-dependent output directions, with coverage determining compactness; (2) context-dependent downstream sensitivity causing the KL/internal-fidelity mismatch; and (3) a substantial interaction component that is behaviorally dispensable for this easy task. A universal rank64 space covering unseen constructions is weakened. Name identity as the dominant source of failure is weakened on this panel.

The next discriminating experiment should separate omitted directions from nonlinear downstream sensitivity: remove the frozen in-span and out-of-span components separately and jointly at multiple doses, retaining all forms and names. If complement removal causes little output change despite large internal shifts, an output-based quotient could be much smaller. If combinations are strongly nonadditive, separate component importance is misleading. Follow this with the same predeclared comparison across agreement, copying, factual retrieval and controlled relational reasoning, using competence checks per task without deleting failed cells. Do not infer a universal computational unit from further copying-only fits.

Tokenization correction: the first attempt stopped during interaction construction because capitalized "Recipient" tokenized into two tokens while lowercase "recipient" used one. No downstream transfer metrics were produced. The list scaffold was changed to "Name:" using tokenization alone. All 512 final prompts satisfy prefix/body ID equality and an initial-case change affecting exactly one token. No prompts were selected using model outputs. Final basis, manifest and runner hashes remained fixed through inference.

Audits: 512 native prompts, 640 interventions, all frozen hashes pass, native Attention5 replay relative error 2.12e-7, factorial interaction replay 8.34e-6, identity logit replay exact, finite metrics. No new natural-text documents; FineWeb remains 26,112. Broad research goal remains active and incomplete.
