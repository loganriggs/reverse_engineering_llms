## H666 — Frozen routing-change × receiver-value rule replicates on new items

H665 identified a useful conditional description for layer14 head4: mean discovery coefficient change multiplied by the receiver's actual input-value vector. H666 freezes that factor model, role means and second-wording boundary/attention means before evaluating new items. No means, coefficients, amplitudes, token-role rules or weights are refitted.

The pre-inference panel has four items per family:

| Family | New items / intended answers |
|---|---|
|plural|cup/cups; room/rooms; train/trains; flower/flowers|
|antonym|open/closed; full/empty; rich/poor; clean/dirty|
|capital|Canada/Ottawa; Ireland/Dublin; Egypt/Cairo; Belgium/Brussels|
|successor|September/October; October/November; November/December; December/January|
|addition|6+1=7;6+2=8;6+3=9;7+3=10|
|agreement|The cup/is; The rooms/are; The train/is; The flowers/are|

Each item uses the same four second-wording forms and task/copy/constant instructions, totaling288 native prompts. All targets pass single-token eligibility; all prompt strings are disjoint from H653/H655. Some words and countries appeared elsewhere in this research, so novelty is relative to this predictor's fitting/evaluation panel, not the entire project or model training. No item is removed after inference. The frozen predictor artifact hashes, panel runner and design are saved before native evaluation; the transfer runner's methods/configuration are saved before intervention evaluation.

For96 copy/task pairs, compare the frozen unconditional input-write mean, mean coefficient change × fresh receiver values, fixed product of discovery means, and exact input-write control, with predicted and exact boundary settings. Native copying and full exact recovery provide references. The predictive modes use no fresh donor factors. The exact-input/full-recovery controls use donor states for diagnosis only. All earlier attention/head components remain frozen mean predictions.

**Predicted boundary, KL(task||edited), nats:**

| Family | Mean-write baseline | Frozen value-conditioned rule | Fixed product | Exact input-write control |
|---|---:|---:|---:|---:|
|addition|0.005830|0.005544|0.005840|0.005022|
|agreement|0.005381|0.005205|0.005382|0.005149|
|antonym|0.011770|0.010701|0.011780|0.011046|
|capital|0.014240|0.011058|0.014327|0.009674|
|plural|0.015235|0.014250|0.015336|0.015591|
|successor|0.076205|0.019267|0.076155|0.021183|

The frozen rule improves every family mean relative to both mean-write baseline and fixed-product control. It improves79/96 individual prompt cases over the baseline with predicted boundary and78/96 with exact boundary; these cases share items/forms and are not independent statistical trials. Improvements in several families are small. No significance claim is made.

**Exact boundary:**

| Family | Mean-write baseline | Frozen value-conditioned rule | Exact input-write control |
|---|---:|---:|---:|
|addition|0.003255|0.003001|0.002531|
|agreement|0.002040|0.001911|0.001875|
|antonym|0.007377|0.005957|0.006152|
|capital|0.010267|0.007145|0.004167|
|plural|0.006467|0.006039|0.007084|
|successor|0.073562|0.012964|0.015503|

Exact-input repair can be worse than the learned approximation for total output KL because other prediction errors remain and can compensate. The exact component is a reconstruction reference, not a global output optimum.

**Month transfer, predicted boundary:**

| Input | Baseline KL | Frozen-rule KL | Exact-input KL |
|---|---:|---:|---:|
|September|0.06692|0.01454|0.01777|
|October|0.06783|0.01518|0.01637|
|November|0.08747|0.02336|0.02535|
|December|0.08260|0.02398|0.02524|

Succession improves0.07620→0.01927 overall, and every new month improves, including the December→January boundary. The factor model was fitted on January–April discovery items; no new month coefficient is learned. This is prospective evidence that preserving receiver input identity through its value vector explains variation that a fixed write mean missed.

**Behavior is separate from fidelity:** natural-task candidate/top1 accuracy is addition25%/0%, agreement50%/0%, antonyms68.75%/0%, capitals100%/0%, plurals100%/0%, succession56.25%/31.25%. Under the frozen rule, succession candidate/top1 accuracy is37.5%/12.5%, compared with25%/0% for the mean-write baseline; other families' correctness is unchanged by the new rule. Thus the improvement includes a limited behavioral gain for months, but is not successful recovery of the task prompt or an established month algorithm. Candidate sets differ from older panels, so their rates are not directly comparable across panels.

Target loss need not track fidelity: for capitals the frozen rule improves KL while slightly worsening target CE5.376→5.401 relative to the mean-write baseline. For months it improves both KL and CE2.225→2.018. Continue reporting these objectives separately.

The supported pattern is conditional routing × content: a simple learned coefficient change acts on an item-specific receiver value vector. This is stronger than the retrospective H665 result, but still one model, one head, known forms/families and fixed templates, with full-model receiver baselines. The coefficient change could encode a generic difference between copying and task instructions rather than a distinct operation. Neither a unique canonical form nor a semantic primitive has been established.

Next test that distinction by replacing task-specific coefficient means with discovery-only coefficients pooled from other families, matched by form, keeping receiver values and all other predictions unchanged. For the four single-token-input families this is a direct comparison of scalar routing changes without positional ambiguity. Treat multi-token addition/agreement separately or explicitly test a broadcast scalar, rather than silently pooling incompatible ordinal roles. If pooled coefficients work similarly, the candidate unit is more generic input routing; if task-specific coefficients win, operation-conditioned gating remains plausible. This test should precede more elaborate coefficient prediction.

Audits:288 native panel forwards plus960 intervention/reference forwards=1248 whole-model forwards;1728 cached attention-stage evaluations/reconstructions. All preregistered and runtime hashes unchanged; no refit; all prompt pairs/targets explicit, complete input-token alignment and frozen ordinal counts verified. Head reconstruction maximum relative error4.77e-7. Native-copy replay max absolute logit error1.20e-5; full exact recovery1.53e-5. All scores finite, all items retained. FineWeb remains26,112 unique documents. Previous goal turn was progress. Broad research goal active and incomplete.
