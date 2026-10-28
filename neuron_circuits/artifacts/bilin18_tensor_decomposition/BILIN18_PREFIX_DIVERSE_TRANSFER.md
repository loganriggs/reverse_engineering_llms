## H618 — Diverse-task transfer fails; capital effects implicate continuation mode more than paired fact discrimination

The previous turn was progress: H617 established that omitted directions can affect outputs and that nonlinear effects grow with dose. H618 extends the frozen H615 copying space to five families: copying, agreement, entity binding, capital recall and sentiment, each with two templates and eight pairs. Prefix presence × initial capitalization gives640 native prompts. On160 combined-condition recipients, compare native, full Attention5 contextual interaction removal, joint32, joint64, joint64 complement and random64:960 interventions. No basis refit or competence-based filtering. Entity-binding prompts receive a fixed "Today," lead-in so case changes affect a scaffold token rather than a name. No answer tokens or pairs were excluded.

**The copying space is not a general representation of this interaction across tasks.** Rank64 internal-effect errors are about0.66–0.72 on agreement,0.82–0.83 on binding,1.05–1.07 on capital recall, and0.70–0.82 on sentiment. The same copying anchor has error0.061; another copying template has0.417. This supports dependence on the prompt distribution, but task and wording change together, so these numbers do not isolate task identity as the cause.

| Task | Template | Write error | Internal-effect error | Relative reference KL | Complement native KL / full native KL | Full native KL (nats) |
|---|---:|---:|---:|---:|---:|---:|
|sequence_copy|0|0.0520|0.0607|0.0098|0.0097|0.004371|
|sequence_copy|1|0.4586|0.4170|0.1524|0.1522|0.003746|
|agreement|0|0.6126|0.6577|0.4058|0.2511|0.025153|
|agreement|1|0.6314|0.7179|0.4743|0.3051|0.032508|
|entity_binding|0|0.8303|0.8316|0.5809|0.5449|0.003150|
|entity_binding|1|0.8068|0.8157|0.5489|0.5007|0.007872|
|capital_recall|0|0.7579|1.0703|1.2218|1.3472|0.665777|
|capital_recall|1|0.7500|1.0528|1.3338|1.1994|0.489093|
|sentiment|0|0.7086|0.8190|0.7654|0.8232|0.002406|
|sentiment|1|0.7608|0.6974|0.4821|0.5071|0.002879|

Write error compares vectors at Attention5. Internal-effect error compares the downstream normalized MLP11 input against full interaction removal, normalized by the full removal's effect. Reference KL compares output distributions against full removal and normalizes by no removal. Native KL instead compares the original model with an intervention. Its component ratios are comparisons, not additive causal shares. Values above1 are possible because components can counteract one another and downstream transformations are nonlinear. Group NLL and single-token CE are both stored, separately; for agreement/sentiment the latter uses the first predeclared accepted token and is not a complete task metric.

**Competence is uneven and is retained in the report.** The table below gives paired candidate-choice accuracy/full-vocabulary accepted-token top1 for all four native conditions. Prefix/case conditions are ordered00,01,10,11; 01 is unprefixed lowercase. A cell passes the predeclared gate when candidate accuracy≥0.75 and accepted-token top1≥0.5. The gate measures this answer set, not general semantic ability.

| Task | Template | 00 | 01 | 10 | 11 |
|---|---:|---|---|---|---|
|sequence_copy|0|1.000/1.000|1.000/1.000|1.000/1.000|1.000/1.000|
|sequence_copy|1|1.000/1.000|1.000/1.000|1.000/1.000|1.000/1.000|
|agreement|0|1.000/0.875|1.000/0.000 fail|1.000/1.000|1.000/1.000|
|agreement|1|1.000/0.500|1.000/0.000 fail|1.000/1.000|1.000/0.938|
|entity_binding|0|1.000/0.875|1.000/0.875|1.000/0.875|1.000/0.625|
|entity_binding|1|1.000/0.438 fail|1.000/0.438 fail|1.000/0.438 fail|1.000/0.438 fail|
|capital_recall|0|1.000/1.000|1.000/0.000 fail|1.000/1.000|1.000/1.000|
|capital_recall|1|1.000/0.250 fail|1.000/0.000 fail|1.000/1.000|1.000/1.000|
|sentiment|0|1.000/0.625|1.000/0.500|1.000/0.500|1.000/0.562|
|sentiment|1|0.000/0.000 fail|0.000/0.000 fail|0.000/0.000 fail|0.000/0.000 fail|

All paired candidate tests pass except sentiment template1, where they all fail. Several unrestricted-answer gates fail despite correct paired choice. For unprefixed lowercase agreement, the model often prefers punctuation or another continuation. For unprefixed lowercase capital prompts, it prefers generic continuation tokens such as "the". Sentiment answer sets are incomplete: "Excellent" and multiword starts such as "Very" can express the intended sentiment but are outside the frozen accepted set. No post hoc expansion rescues reported scores. These failures prevent claiming that all factorial corners execute the same task in the same response mode.

**Capital recall has a large answer-selection effect that the copying span misses.** Combined recipients initially give correct capital top1 throughout. Full interaction removal reduces top1 to0.375/0.125 across templates; complement removal gives0.125/0.125. Span-only removal preserves1.0/1.0. Full native-reference KL is0.666/0.489 nats, substantially larger than the copying examples.

A targeted128-forward completion audit then shows that the correct capital remains preferred over the paired wrong capital in every native/full/span/complement case. The winning tokens after damaging interventions are usually "the", "a" or "not", not the paired wrong capital. Intended-capital CE rises from1.031/1.381 nats natively to2.583/3.096 after full removal and2.883/3.266 after complement removal. Span-only CE is0.998/1.267. This supports a change in propensity to emit the direct factual answer, while the narrow paired fact distinction remains intact. It does not prove that all factual ranking, knowledge or retrieval is preserved: the audit only compares the specified alternative. Country prompts repeat across neighboring pairs, so the16 endpoints per template are not16 independent countries.

The native completion audit also exposes a confound: the unprefixed lowercase corner often uses a generic continuation mode already. Subtracting its factorial interaction may transfer that behavior into an otherwise competent prefixed prompt. Therefore the intervention is causal for answer emission on these recipients, but cannot yet be named a country–capital feature or a universal retrieval unit. It may reflect a broader context-dependent continuation mechanism.

Most likely current hypotheses: (1) compact output coordinates depend on distribution coverage; (2) directions outside copying coordinates can control answer-emission behavior in other prompts; (3) response mode and answer identity are partly separable here; (4) a factual-memory interpretation is weaker than an answer-emission interpretation given preserved paired discrimination; (5) task identity alone is not isolated from scaffold and case effects. We should now prioritize disambiguating these mechanisms rather than fitting more spans to explain away failures.

Next control: compare two explicit prefixes crossed with initial case, preserving body alignment, and include a fixed direct-answer format alongside free continuation. Predeclare strings using tokenization alone and retain all competence failures. Measure correct-capital versus paired alternative separately from correct-capital versus generic continuation probability, and repeat span/complement/full removals at small and full dose. If the effect disappears when response mode is held stable, the earlier failure was primarily a continuation-mode interaction. If it persists with stable competence and alters factual discrimination, a stronger retrieval-specific hypothesis survives. Keep copying/agreement anchors so this is not inferred from capital prompts alone.

Audits: source basis and runner/manifest frozen; all640 prompts satisfy boundary ID equality and case changes exactly one token;960 interventions finite; Attention5 native replay relative error2.45e-7, interaction replay8.56e-6, identity logit error1.53e-5. The independent capital audit reproduces all main top1 rates. The completion audit is descriptive and was not used to filter or refit the main panel. FineWeb remains26,112 documents. Broad research goal active and incomplete.
