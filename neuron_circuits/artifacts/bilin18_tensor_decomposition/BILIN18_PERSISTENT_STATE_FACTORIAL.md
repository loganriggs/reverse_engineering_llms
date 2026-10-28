## H844 — Persistent layer0 values explain much of the weak early residual-only task transfer

**Main result.** This model has materially different routes for different tasks. At early boundaries, donor layer0 values alone transfer about95% of the capital-recall answer gap,97% of entity binding, and87–88% of copying. Agreement and sentiment transfer more through the current residual sequence. The original embedding anchor alone contributes modest donor transfer in these tests.

This is a concrete reason not to judge early residual featurizers by final-position task swaps alone. Much of the task-relevant donor difference can bypass that representation through a persistent architectural input.

**Architecture and terms.** Each later attention layer mixes its newly computed values with a persistent value tensor computed at layer0. Call that tensor V. Each block also mixes the current residual sequence R with the original normalized embeddings E (`x0`). We independently choose donor or recipient R,V,E at boundaries1/2 and resume the remaining network. All8 combinations run on all76 equal-token-length pairs from H843, both directions. These are controlled mixed-state interventions, not natural mediation fractions.

Reported transfer is donor-directed answer-margin change divided by the native donor-recipient margin gap, pooled over both directions. It is neither accuracy nor percentage of information stored in that route. Pair counts:16 agreement,13 capital,16 binding,15 sentiment,16 copying; unequal-length examples remain explicit exclusions from H843.

|Boundary|Task|Residual R alone|Persistent values V alone|Embedding E alone|
|---|---|---:|---:|---:|
|1|agreement|74.9%|6.5%|9.4%|
|1|capital_recall|0.4%|94.8%|3.5%|
|1|entity_binding|2.5%|97.0%|0.5%|
|1|sentiment|58.3%|29.9%|5.4%|
|1|sequence_copy|7.2%|87.5%|4.7%|
|2|agreement|77.8%|5.6%|9.3%|
|2|capital_recall|0.5%|94.7%|3.5%|
|2|entity_binding|2.6%|97.1%|0.5%|
|2|sentiment|61.7%|28.6%|5.2%|
|2|sequence_copy|7.0%|87.7%|4.3%|

The complete R+V+E control transfers100%, as required by determinism. The full factorial and paired-item bootstrap intervals are retained. Routes need not add linearly: for example L1 agreement R+V transfers90.6%, versus74.9% and6.5% individually. But do not read complementary-mask symmetry as independent evidence: averaging bidirectional swaps algebraically makes complementary subsets sum to approximately100% of the native gap. Any interaction claim needs direction-specific analysis rather than interpreting that identity as redundancy or mechanism.

**What this says about features.** The persistent layer0 value at a token is a token-local transform of its embedding, before any attention aggregation. Repeated identical tokens therefore have identical value vectors in this model (up to numerical effects). A strong V intervention primarily shows that later computation reads this persistent token-content channel; it does not show that a contextual task feature or semantic manifold is already computed inside V. The meaningful computation may lie in the later attention read/routing pattern and its interaction with contextual query/key states. This aligns with the earlier observation that value0 same-token clouds are degenerate.

For agreement and sentiment, early residual states at changed content positions are a better causal starting point than the prediction position. However, changing token identities also changes the training-token mean that H836 featurizers subtract. Their learned contextual residual coordinates need not capture that lexical contribution. The next component test must explicitly compare token-mean replacement, learned reconstruction contribution, and reconstruction residual, alongside the whole-state control.

**Updated experimental priority.** Pursue two complementary branches within the same broad goal: (1) localize where later attention reads the persistent V channel for copying/binding/recall, separating source-token content from context-dependent routing; (2) at informative early content positions for agreement/sentiment, test whether frozen featurizer components explain causal transfer beyond token means. First inspect earlier value-bus/head intervention records to avoid repeating already resolved routing work. Retain matched controls and cross-format/seed replication, and use fresh confirmation after exploratory choices are frozen. A task label becomes useful only when a component predicts and transfers the relevant computation under controlled conditions.

**Audits.**2,432 suffix forwards starting at captured boundaries, no new whole-model forwards. Native and donor replay max logit error4.49e-5; repeated H843 residual/complete margin effects agree within8.59e-6. All hashes unchanged and metrics finite. No reserved-test use or additional natural documents; count stays26,112. Broad feature/computation goal remains active. This turn added a depth/position map and a complete architectural-state factorial, changing the next experiment from blind candidate labeling to route-specific representation tests.
