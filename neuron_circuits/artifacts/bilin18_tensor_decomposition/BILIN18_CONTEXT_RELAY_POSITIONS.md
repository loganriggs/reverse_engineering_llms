## H847 — A single unchanged token just after the edited content carries substantial context

**Main result.** Layer4 contextual transfer is concentrated at one intermediate token in each tested task. In agreement, swapping the preMLP state at ` near`/` beside` transfers36.0% of the native answer gap. In sentiment, swapping the state at the period after the adjective transfers37.2%. These are token-identical, position-identical donor swaps, so this is contextual state rather than direct token-identity substitution.

**Design.** All31 equal-length agreement/sentiment pairs from H846, both directions. At preMLP4, intervene separately at every same-token position at or after the first token difference. No position excluded based on effects. Recipient original embeddings, persistent values, other tokens and later computation remain native. Layer4 and task families were chosen after H846, so this is exploratory localization, not fresh confirmation.

|Task|Position from end|Token(s)|Gap transferred|Paired95% interval|
|---|---:|---|---:|---|
|agreement|-3|' beside', ' near'|36.0%|[32.5%, 39.3%]|
|agreement|-2|' the'|2.2%|[1.8%, 2.7%]|
|agreement|-1|' artist', ' book', ' bottle', ' cat', ' doctor', ' dog', ' driver', ' farmer', ' key', ' picture', ' report', ' singer', ' student', ' teacher', ' tree', ' worker'|1.0%|[0.4%, 1.5%]|
|sentiment|-4|'.'|37.2%|[34.0%, 40.7%]|
|sentiment|-3|' Sent'|1.9%|[1.6%, 2.2%]|
|sentiment|-2|'iment'|-0.7%|[-1.8%, 0.2%]|
|sentiment|-1|':'|-2.0%|[-3.2%, -0.8%]|

The aggregate same-token/nonfinal H846 effects were40.9% for agreement and39.8% for sentiment. Thus the largest single-position effects are close in magnitude to the joint nonfinal effect, but ratios should not be called unique mediation percentages: interventions may interact, and single effects need not sum to the joint effect.

**Interpretation.** In both templates the strongest position immediately follows the altered lexical content: after the subject noun in agreement, and after the sentiment adjective in the review. This suggests a local contextual relay or previous-token feature copied into the next position. It does not establish a general grammar/polarity manifold or a shared semantic computation. Adjacent-token routing, punctuation/format-specific behavior, position effects and a generic one-token-delay representation all remain live alternatives. Sentiment's `Sentiment` label is tokenized into ` Sent` and `iment`; the strong signal sits at the earlier period, not at those label pieces. Agreement uses two different prepositions across the two formats.

**Next high-information tests.** First distinguish a compact cross-context direction from a high-dimensional lexical trace: fit low-rank contrast directions using only the discovery pairs at this correctly matched preMLP4 site, freeze them, and test reconstruction and causal transfer on the held-out format/lexicon split. Call that supervised contrast compression, not an unsupervised discovered manifold. Compare ranks and matched-rank random controls, preserve reconstruction residual, and keep native full-state effects as denominator. Then use new prospectively fixed distance/format variants to separate adjacency/position heuristics from portable contextual computation, checking competence and exact tokenization first. Natural-data featurizers at this site can subsequently be judged against this causal target; earlier-site dictionaries cannot be transplanted here without fitting/validation.

**Audits.**264 full forwards. Identity logits agree within3.25e-5; source/model hashes unchanged; all effects finite; no reserved-test use; natural-document count26,112 unchanged. Full per-item/direction measurements, split summaries and paired intervals retained. This turn made progress by eliminating the changed-token confound and locating a concrete intermediate contextual target. The broad feature/computation goal remains active.
