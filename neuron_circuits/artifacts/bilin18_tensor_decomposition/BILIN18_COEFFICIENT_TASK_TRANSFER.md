## H667 — Routing coefficient changes are not interchangeable across task families

H666 prospectively replicated mean routing change × receiver input value. H667 asks whether the routing change is generic across tasks. Keep the same H666 evaluation panel and all receiver values, boundary predictions, other attention predictions, and frozen discovery models. Replace only the coefficient applied to layer14 head4 input-token values.

Primary comparison: plural, antonym, capital and successor have exactly one input token in the fitted/evaluated examples. For each known form, compare its own discovery coefficient change with the mean of the other three families' discovery coefficient changes, and with each individual source family's coefficient. Every coefficient is fitted only on the original four H665 discovery items. A zero-change coefficient keeps the receiver's input write unchanged while removing its original mean input-shift prediction. It is therefore distinct from the unconditional mean-write baseline.

Addition and agreement have multiple input-token positions. Their own predictor retains separate ordinal coefficients; transferred single-token coefficients are explicitly broadcast to every input position. Those are stress tests, not matched ordinal-role controls. No output fitting or source-family selection occurs. Because H666 results were already inspected, this is retrospective family-transfer diagnosis, not a new blind replication.

**Primary families, predicted boundary, KL(task||edited):**

| Target | Own coefficient | Other-family pooled | Zero change | From plural | From antonym | From capital | From successor |
|---|---:|---:|---:|---:|---:|---:|---:|
|plural|0.014250|0.017104|0.015390|0.014250|0.018048|0.017621|0.037511|
|antonym|0.010701|0.014922|0.011769|0.012127|0.010701|0.011815|0.026104|
|capital|0.011058|0.013639|0.014302|0.015924|0.014239|0.011058|0.027916|
|successor|0.019267|0.062248|0.061969|0.062725|0.070385|0.055154|0.019267|

Each of the four single-token target families has its lowest cell-mean KL with its own coefficient among these tested sources, under both predicted and exact boundary conditions. This is not evidence that the own coefficient is globally output-optimal. For months, own-task KL0.01927 is much better than pooled-other0.06225 or zero0.06197; capital coefficients partially transfer (0.05515) but leave a large mismatch. A generic coefficient is insufficient for this panel.

**Exact boundary primary comparison:**

| Family | Own coefficient | Other-family pooled | Zero change |
|---|---:|---:|---:|
|plural|0.006039|0.010918|0.006414|
|antonym|0.005957|0.011013|0.007379|
|capital|0.007145|0.011963|0.010259|
|successor|0.012964|0.060003|0.059296|

**Frozen discovery coefficient changes:** forms0/1 use Answer: with Give/Provide; forms2/3 use Result: with Give/Provide. These signed bilinear coefficients are not probabilities.

| Family | Form0 | Form1 | Form2 | Form3 |
|---|---:|---:|---:|---:|
|plural|0.01750|0.02323|-0.02260|-0.00682|
|antonym|-0.05084|-0.03328|-0.02004|-0.00882|
|capital|-0.03466|-0.01495|0.02444|0.07077|
|successor|0.13274|0.17419|0.19477|0.23618|

Successor changes are large and positive in all forms, while other families' changes are smaller, negative or sign-changing. The form dependence itself cautions against treating the scalar as an invariant semantic label. Pooling coefficients is an intentionally simple generic-routing hypothesis; its failure does not exclude a shared conditional rule that also depends on input or context.

**Multi-token broadcast stress tests, predicted boundary:**

| Target | Own ordinal coefficients | Pooled single-token scalar | Zero change | Best individual transferred source (descriptive) | Its KL |
|---|---:|---:|---:|---|---:|
|addition|0.005544|0.007004|0.005838|antonym|0.005260|
|agreement|0.005205|0.005142|0.005364|capital|0.005074|

Addition benefits slightly more from the antonym scalar than its own coefficients, while agreement benefits slightly more from capital or pooled coefficients. These small improvements can reflect error compensation and non-equivalent token roles; they should not be labeled semantic transfer. All tested sources are reported rather than deploying the best retrospective source.

The supported description is input-dependent value content multiplied by a coefficient conditioned on the task/form. It is not simply one generic copying→instruction routing change. But operation identity is still confounded with lexical instruction and input domain: months, countries, nouns and adjectives differ. The other boundary/attention components also retain target-family information, so this is a localized head-write test, not task discovery or a task-independent model.

Next cross operation direction on identical inputs: next versus previous month, and next versus previous number, with copying controls and matched answer forms. Use explicit item/answer pairs and token-length audits. Measure native competence separately; decompose coefficient changes before choosing a predictor. This tests whether head4 distinguishes relational direction or mainly the broader input/domain and instruction type. A shared large coefficient for both next and previous would weaken a succession-specific interpretation even if both differ from copying. Independent item splits and an additional domain are needed to avoid relabeling an input-domain effect as semantics.

Audits:1728 whole-model forwards;1728 cached attention-stage evaluations/reconstructions. H666 baseline and own-coefficient logit arrays reproduce identically. All frozen model/preregistration and runtime hashes unchanged; no refit. Primary single-token counts asserted, complete input alignment preserved; native replay and exact recovery deviations below1.20e-5 and1.53e-5, head reconstruction below4.77e-7 relative. All metrics finite. The new result concerns KL fidelity; no claim of improved task correctness is inferred from coefficient transfer. FineWeb remains26,112 unique documents. Previous goal turn was progress. Broad research goal active and incomplete.
