## H1051 — Content preservation does not solve residual-dependent whole-task transfer

Previous goal turn was progress: H1050 validated the exact token-content program and showed that factual rescue was chiefly an explicit content bypass. This turn applied the same conditional contract to all other eight H1048whole-family crossfit folds,72pairs/144endpoints. No new fit or family selection.

Every method receives the exact donor embedding/cache outputs from the H1050program and the native recipient residual. Only the residual change is represented sparsely (max16coordinates) or densely (128). Encoder/decoder models were fitted excluding each evaluated family. Initial and final decoder checkpoints are retained, alongside zero-residual-change control. These are conditional edits with2304exact content scalars/token, not a compressed whole model. All retained program/embedding/native-suffix costs remain as documented in H1050.

Paired relative KL, lower better:

| Excluded family | Zero residual edit | Trained scalar16 | Trained block max16 | Trained shared max16 | Dense128 |
|---|---:|---:|---:|---:|---:|
| Agreement |0.9839|0.8169|0.8876|0.9383|0.5308|
| Antonym |0.6983|0.6114|0.6227|0.6678|0.5046|
| Entity binding |0.0949|0.0531|0.0565|0.0677|0.0543|
| Number successor |0.8115|0.7585|0.7754|0.7951|0.6739|
| Past tense |0.9446|0.7181|0.8102|0.8415|0.5607|
| Noun plural |0.9878|0.7328|0.8028|0.8783|0.4719|
| Pronoun gender |0.6571|0.2109|0.4140|0.4831|0.0745|
| Sentiment |0.8418|0.6778|0.7589|0.7820|0.5983|

The content program covers much of binding, consistent with its cached-value dependence, but leaves substantial contextual computation unexplained elsewhere. Scalar improves over zero edits, particularly pronouns, yet most task errors remain high. Dense residual projection is better but still imperfect on most families; it is not a behaviorally optimal upper bound for arbitrary encoders in the span.

Equal-family macro relative KL: zero0.7525; scalar initial0.5766→trained0.5724; block0.6458→0.6410; shared0.6802→0.6817; dense0.4336. Thus nearly all sparse benefit was already present before action-decoder fitting. Macro donor-answer top1is11.98%zero,18.62%trained scalar,13.54%block,12.37%shared,22.53%dense. No strong learned cross-family contextual representation is established.

This closes the generality check without promoting the factual result into an overall success. The current decoders were trained against all-stream reconstruction of changes; they were not trained under the exact-content conditional contract used here. That objective mismatch is a remaining testable explanation, alongside weak encoder coverage, fixed random groups, small training inventory, and limited capacity. These results alone cannot isolate those explanations or prove that context-conditioned manifolds are absent.

Verification:1440suffix calls, zero fitting rows, all8excluded families absent from their fitted geometry, source/checkpoint/program hashes unchanged. Program content matches native inputs exactly and identity replacement reproduces native logits exactly on all evaluated endpoints. Max16sparse residual coordinates. Independently recomputed KL for64saved examples, max2.54e-14. Native errors and all panels retained. No new corpus, untouched external validation, or new-task discovery.

Next bounded comparison: train only residual action decoders under this exact conditional objective, preserving whole-family exclusions and the same fixed support rules. All architectures should have147456learnable normalized decoder parameters, identical exact content access, and16visits per remaining training family. Require full-training initial/final loss improvement, report zero-edit and untrained controls, and compare with the existing frozen all-stream-trained residual decoders. This tests whether directing supervision at the actual unexplained contextual change helps; do not infer success merely because the exact content component is strong. If cross-family gains remain absent, stop this decoder-only family before a new representation hypothesis.
