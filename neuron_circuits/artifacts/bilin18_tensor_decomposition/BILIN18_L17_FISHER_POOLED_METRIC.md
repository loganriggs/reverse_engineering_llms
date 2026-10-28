## H504 — Pooling the probability metric helps some heads, not all

Average the four equal-sized H503 Fisher matrices, then compute the rank-4 source map from the pooled metric. The pooled estimator uses64 distinct training receivers; the reference uses the original16. Source calibration covariance, rank, model, donors, and evaluation receivers are fixed. Average metrics before eigentruncation, not truncated source operators. No best individual fit is selected. This is a diagnostic on reused H500 donor/H499 receiver pairs, not fresh confirmation.

| Head | Original16 recovery −/+ | Pooled64 recovery −/+ | Gain in percentage points −/+ |
|---|---|---|---|
|0|21.11% / 27.56%|21.16% / 25.12%|+0.05 / -2.43|
|1|82.78% / 81.19%|82.91% / 81.46%|+0.13 / +0.28|
|2|48.54% / 53.30%|56.43% / 58.08%|+7.89 / +4.78|
|3|68.23% / 64.94%|68.67% / 64.08%|+0.44 / -0.86|
|4|82.54% / 82.50%|82.25% / 82.20%|-0.29 / -0.30|
|5|57.79% / 57.52%|58.57% / 57.89%|+0.78 / +0.38|
|6|98.69% / 98.46%|98.60% / 98.35%|-0.09 / -0.11|
|7|51.03% / 51.69%|53.77% / 55.18%|+2.74 / +3.48|
|8|82.62% / 82.38%|83.21% / 82.90%|+0.59 / +0.52|

Head2 benefits most, gaining7.89/4.78 percentage points; head7 gains2.74/3.48 points. Head0 is not rescued: negative improves only0.05 points and positive regresses2.43 points. Several already well-captured heads change little, and heads4/6 regress slightly. Pooling does not guarantee finite held-out improvement even though its own calibration-weighted local objective is no worse than the original map's.

The result supports metric-estimation sensitivity as one limitation, especially for head2, but not as a complete explanation of all low-rank failures. A single pooled fit does not demonstrate reduced variance across independently pooled samples. The JSON retains all four individual-fit recoveries for context; their maximum is not a selected comparator. There are no fresh intervals or universal superiority claims.

Audits: each pooled metric is the mean of the four equal-size archived matrices; pooled rank4 training objective is no worse than the original map on the same pooled objective; map idempotence passes1e-10; native effect denominators replay H500; finite per-case results reproduce aggregate recoveries. Pooled matrices/operators/eigenspaces are saved. No new documents were used; tracked total remains19,456.

Next H505 changes the evaluation question instead of only increasing estimator size. On ordinary H499 receiver documents, collect each head's own source contribution entering the layer17 MLP. Replace its centered contribution h by a rank4 reconstruction hhat, giving MLP input p−h+hhat while keeping the direct residual bypass p fixed. Compare PCA, orthogonal projection into the original Fisher write span, original16 oblique, and pooled64 oblique maps. All maps and calibration means remain frozen; evaluate each head separately, not all heads simultaneously. Measure KL(original output || replacement output), change in observed-next-token cross-entropy, and raw MLP write error. Include original output as the reference and mean-source replacement hhat=0 as a scale comparator.

This directly tests natural source replacement, not approximation of a donor-removal intervention. CE here uses the actual next token in the receiver text; it is distinct from teacher-distribution cross-entropy, which differs from KL only by fixed entropy. Reused documents mean descriptive results only, not evidence of out-of-sample CE improvement. These are existing Fisher-trained maps, not CE-optimized solutions, so the experiment compares evaluation objectives without claiming to compare optimized objective frontiers. The native remaining MLP and direct residual remain dependencies. This is a more relevant next step toward data-based canonical computation on ordinary text and clarifies the remaining gap to full layer replacement.
