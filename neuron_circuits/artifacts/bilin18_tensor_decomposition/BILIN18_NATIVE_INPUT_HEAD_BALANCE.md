## H674 — Native signed gates mostly turn opposing directions into reinforcement

H673 found both boosting and suppressing carried-value directions across heads. H674 includes the actual signed bilinear attention coefficient for the input position, separates weighted local and carried values, and sums the corresponding direct residual/readout derivatives. It uses the same128 contexts across four domains, copy/task instructions, four items and four forms. No fitting or selection.

These coefficients are not attention probabilities. A negative coefficient reverses a suppressive value direction into a boosting write. This distinction materially changes the interpretation of the previous direction probes.

| Family | Context | Net input slope | Positive head sum | Negative head sum | Mean abs(net)/sum(abs(head)) | Head4 input slope | Mean head4 coefficient |
|---|---|---:|---:|---:|---:|---:|---:|
|plural|copy|2.2261|2.2276|-0.0015|0.9987|1.2804|-0.3028|
|plural|task|2.1492|2.1527|-0.0035|0.9964|1.2369|-0.2950|
|antonym|copy|1.7411|1.7597|-0.0185|0.9758|0.4615|-0.2168|
|antonym|task|1.8513|1.8597|-0.0083|0.9901|0.5220|-0.2354|
|capital|copy|0.8689|0.9877|-0.1188|0.7082|0.2557|-0.0664|
|capital|task|0.5718|0.6652|-0.0934|0.7608|0.1797|-0.0475|
|successor|copy|1.5767|1.6630|-0.0863|0.8982|0.3018|-0.1148|
|successor|task|0.6016|0.8597|-0.2581|0.5414|-0.0741|0.0295|

Head effects in this table include both local and carried input-token writes. Positive/negative sums classify each combined head effect before summing across heads. The cancellation ratio is descriptive scalar arithmetic, not a fraction of causal responsibility. It describes only the current-input logit and this token position.

Most contexts exhibit reinforcement, not substantial opposition: plural and antonym family means have ratios near1. Head4's mean coefficient is negative under both copying and task instructions in those domains and in capitals, so its native direct contribution boosts the current input despite the suppressive direction seen when increasing the coefficient. Capitals show some cancellation. Month task instructions have the strongest mean cancellation and a much smaller net boost than month copying. Head4 changes from mean input boost+0.3018 in copy contexts to suppression−0.0741 in successor contexts, with mean coefficient moving from−0.1148 to+0.0295.

This updates the leading hypothesis: the unit under investigation behaves more like a signed, context-controlled content readout than a permanently suppressive head. Earlier intervention findings remain valid: increasing its coefficient suppresses input-associated content, but the native operation can have the opposite sign. Opposing available readout directions do not establish native cancellation. The native weighted evidence is essential.

These are derivatives at the native final residual of scaling each input-token write along the direct skip path. They are not an additive decomposition of finite output logits, do not include later branch responses, and do not establish behavioral necessity. Differences between copy/task means also include different native final readout states; they are not isolated causal effects of changing only the gate. Other token positions remain part of the native attention output but are absent from the reported input-position accounting.

The next test is finite removal of the measured native input-token writes: head4 alone and all nine heads together, followed by full downstream computation. Compare copying versus task instructions, read current/answer logits, output KL and correct-target loss separately. This tests whether the direct sign reversal predicts actual behavior and whether reducing input support helps the requested operation or merely damages useful copying.

Audits:128 cached attention evaluations and zero whole-model forwards. Reconstructing complete layer14 attention output over all positions gives maximum relative error3.840e-7. Input-position head4 coefficients replay saved factors with maximum absolute error1.491e-7. All tracked hashes unchanged, finite effects. FineWeb remains26,112 unique documents. Prior goal turn was progress; broad goal remains active and incomplete.
