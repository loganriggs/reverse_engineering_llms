## H505 — Natural own-source replacement and observed-token loss

Move from donor-intervention prediction to compressing each head's own contribution on ordinary text. On the512 H499 receiving contexts, collect each head's layer17 attention write, center it by the frozen calibration mean, and reconstruct that centered source with a frozen rank4 map. Replace only the MLP input contribution: p becomes p−h+hhat at the MLP input, while the direct residual bypass remains p. The final state is p+MLP(p−h+hhat). Each head is replaced separately. This is a conditional source-interface approximation, not complete attention-write or layer replacement.

Compare five fixed reconstructions: calibration-mean replacement (hhat=0), PCA, orthogonal projection into the original Fisher write span, original16-receiver oblique Fisher, and pooled64-receiver oblique Fisher. No maps are fitted or selected on these natural-replacement outcomes. Measure KL(original output || replacement output), raw MLP write squared error, and observed-next-token CE change. Original mean CE is3.443114 nats per evaluated token. Targets are document token64 for model position63. These are reused documents and descriptive results, not new confirmation or CE-trained models.

For comparability within a head, KL recovery below is1−replacement KL / calibration-mean-replacement KL. It is not accuracy or a percent of total model computation preserved. The reference distribution and operation differ from earlier donor-intervention experiments, so their recovery percentages are not directly interchangeable.

| Head | PCA KL recovery | Orthogonal-write | Original Fisher | Pooled Fisher |
|---|---:|---:|---:|---:|
|0|30.90%|25.87%|26.28%|29.28%|
|1|85.11%|84.91%|87.02%|87.44%|
|2|35.89%|26.99%|40.08%|55.19%|
|3|44.46%|50.75%|52.21%|55.02%|
|4|92.52%|92.87%|92.61%|92.46%|
|5|24.37%|29.66%|39.46%|40.94%|
|6|98.68%|98.91%|99.04%|98.98%|
|7|51.89%|53.91%|57.55%|58.63%|
|8|85.46%|86.55%|88.78%|88.85%|

Probability-aware representations preserve more KL fidelity in several natural-source settings. Head2 improves from35.89% PCA to55.19% pooled; head5 from24.37% to40.94%; head7 from51.89% to58.63%. Head6 remains nearly preserved at about99%. Heads0 and4 do not exhibit a general Fisher advantage. Head5's natural-source recovery is lower than its earlier donor-intervention recovery, underscoring the scope of those results; this comparison changes source/receiver association, operation, and KL reference together, so it cannot isolate a single cause.

Observed-token CE is a different objective. Positive change means worse loss; negative means better loss. The following descriptive changes are in millinats per token (0.001 nat). Parentheses show the unadjusted paired standard error, not a confidence interval or a multiple-comparison test.

| Head | PCA ΔCE (SE) | Original Fisher ΔCE (SE) | Pooled Fisher ΔCE (SE) |
|---|---|---|---|
|0|-1.425 (0.925)|-1.454 (0.946)|-1.087 (0.913)|
|1|-0.476 (0.626)|-0.198 (0.542)|-0.174 (0.519)|
|2|+0.672 (1.372)|-0.812 (1.393)|-0.241 (1.114)|
|3|+0.104 (0.463)|+0.062 (0.447)|+0.123 (0.428)|
|4|-0.066 (0.294)|-0.079 (0.319)|-0.226 (0.310)|
|5|+0.349 (0.255)|+0.170 (0.229)|+0.233 (0.223)|
|6|-0.155 (0.324)|-0.016 (0.285)|+0.067 (0.298)|
|7|+0.216 (0.314)|+0.538 (0.286)|+0.424 (0.264)|
|8|-0.146 (0.392)|-0.210 (0.377)|-0.175 (0.382)|

CE changes are tiny compared with baseline CE, and generally comparable to their paired standard errors. No generalization improvement or equivalence claim is supported. Objective rankings can nevertheless differ descriptively: head7's Fisher maps have lower KL error than PCA but a larger positive observed-token CE change in this sample; head2's original Fisher map has worse KL fidelity than pooled but a more negative sample CE change. This illustrates why matching average observed-token loss is not the same as preserving the model's distribution. It does not demonstrate a reliable CE/faithfulness tradeoff frontier; that requires separately trained objectives and a larger independent evaluation.

Teacher-distribution cross-entropy would differ from KL only by the original distribution's fixed entropy. The CE measured here instead uses actual next tokens from the dataset. The original model is the zero-KL reference, but it is not guaranteed to minimize empirical observed-token CE among altered computations. Negative sample CE changes should not be promoted into a better-model claim.

Absolute KL errors, mean-source reference errors, raw write errors, all five CE results, and per-case data are archived. The natural source writes and next-token labels are also retained for subsequent joint-source tests. Audits: head-wise writes sum to the attention write with relative error4.29e-7; original receiver states exactly replay H499; token alignment is explicitly asserted; all arrays are finite and aggregate KL values replay. No documents were added; tracked total remains19,456.

Next H506: replace all nine centered source contributions jointly using the same frozen per-head maps, with mean replacement as a comparator. Evaluate two explicit computation paths: MLP-input-only replacement (original bypass retained) and whole-pre-MLP replacement (both MLP input and direct residual use the reconstructed attention contribution). Compare original output KL and observed-token CE, retaining all prescribed methods and reporting absolute errors. This tests composition and the limitation that the current metric was trained for the MLP path only. A failure on the full path would identify a metric-scope mismatch, not disprove data-based canonical forms. The test should remain diagnostic on reused contexts. If needed, a full-path metric must include the direct residual derivative before claiming actual attention-write compression.
