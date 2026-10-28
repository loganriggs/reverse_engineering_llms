## H668 — Directional gating is domain-dependent and insufficient to switch the task

H667 found different routing-change coefficients across task families, but input domain and task identity were confounded. H668 crosses next versus previous on identical inputs in two domains: all12 months (cyclic) and numbers2–9 (arithmetic ±1). Each input has next, previous and copy instructions under four Give/Provide × Answer:/Result: forms, yielding240 prompts. Within every next/previous pair, token length is identical and exactly one instruction token changes. Input text, suffix and domain-wide candidate set are identical.

Discovery months are January, April, July, October; discovery numbers are2,4,6,8. Remaining whole items are transfer, so every form of an evaluated item is excluded from coefficient fitting. All answers are explicit, single-token and checked before inference. Inputs and some next-month templates were previously studied; this is a newly controlled direction comparison, not an entirely novel-input replication.

Reconstruct layer14 head4's input-token coefficient and projected effective value. Fit mean copy→next and copy→previous coefficient changes on discovery items. On held-out next/previous receivers, patch only the head4 input contribution at layer14, retaining receiver values and all other native computation. Two patches use either the discovery mean coefficient difference between operations or the exact donor-minus-receiver coefficient difference. Neither swaps values or other components. Native output is the reference. Exact coefficient controls use donor information; learned patches do not.

**Discovery mean coefficient changes relative to copying:** coefficients are signed bilinear products, not probabilities.

| Domain | Operation | Form0 | Form1 | Form2 | Form3 |
|---|---|---:|---:|---:|---:|
|month|next|0.13178|0.17042|0.18375|0.22426|
|month|previous|0.01532|0.08546|0.08807|0.14760|
|number|next|-0.05601|-0.05446|-0.04694|-0.05474|
|number|previous|-0.05442|-0.04229|-0.05172|-0.04610|

Month next-instructions produce larger positive coefficient changes than previous-instructions in every form. Number coefficients are negative and much closer between directions, with direction differences changing sign across forms. Equal token length rules out a simple length explanation for these paired differences, but the words next/previous and domain remain part of the context. This is evidence of context-sensitive gating, not a domain-invariant next/previous scalar code.

**Held-out coefficient relative RMSE, using own-operation versus opposite-operation discovery means:**

| Domain | Operation | Own mean | Opposite mean |
|---|---|---:|---:|
|month|next|0.3118|0.5294|
|month|previous|0.3532|1.1630|
|number|next|0.2266|0.2953|
|number|previous|0.3254|0.3064|

Own-operation means clearly help months. For number-previous, the opposite mean is slightly better, reinforcing that these small differences do not establish a universal direction encoding.

**Finite coefficient swaps: KL(opposite-task distribution||edited receiver), nats.** Only head4's input coefficient changes; downstream computation runs normally.

| Domain | Direction | Native receiver | Mean coefficient swap | Exact coefficient swap |
|---|---|---:|---:|---:|
|month|next_to_previous|0.032951|0.031919|0.031044|
|month|previous_to_next|0.030911|0.029796|0.028956|
|number|next_to_previous|0.011688|0.011649|0.011675|
|number|previous_to_next|0.011903|0.011864|0.011892|

Mean and exact swaps reduce KL slightly, but leave most of the natural next/previous distribution difference. Even the exact coefficient is insufficient to switch the output distribution. This is a materially narrower result than saying the head implements next or previous. Coefficient-level direction discrimination does not establish an algorithmic role by itself.

**Held-out native competence:**

| Domain | Instruction | Shared-candidate accuracy | Exact next-token accuracy |
|---|---|---:|---:|
|month|next|18.75%|18.75%|
|month|previous|0.00%|0.00%|
|month|copy|100.00%|100.00%|
|number|next|0.00%|0.00%|
|number|previous|0.00%|0.00%|
|number|copy|87.50%|87.50%|

The previous-month and both number operations fail exact next-token accuracy on this held-out set. Candidate sets are shared across operations, unlike prior restricted target sets, making some earlier competence numbers incomparable. Month previous→next coefficient swaps improve opposite-answer exact accuracy from6.25% to9.375%, still below native next-prompt18.75%. Other swap directions remain0%. These are small repeated-template counts, not robust algorithmic success. Mean swapping next→previous slightly worsens opposite target CE while improving KL, another example of objective divergence.

The leading description is a domain- and instruction-sensitive gate on input content, with other components carrying much of the direction-specific behavior. It is not a general successor/predecessor unit. A useful next question is what the projected input-value write actually favors after downstream processing. Measure signed changes in current-item, next-item and previous-item logits when perturbing this input contribution, across copy/next/previous contexts and both domains. This can distinguish copying/repetition bias from a relation-specific output preference and avoids inferring function solely from the coefficient's size. Keep finite amplitudes fixed and native task competence separate.

Audits:240 native panel forwards plus288 held-out intervention forwards=528 whole-model forwards;240 cached attention-stage reconstructions. Exactly one changed instruction token and equal token length asserted for every next/previous pair. Explicit cyclic/arithmetic answers, shared candidate sets, full-item discovery exclusions and single input-token extraction verified. Head reconstruction maximum relative error4.77e-7; native intervention replay exactly matches stored logits. All hashes unchanged; KL finite.96 held-out directional/form rows retained. FineWeb remains26,112 unique documents. Previous goal turn was progress. Broad research goal active and incomplete.
