## H842 — The weak-transfer bottleneck extends to the early final-position residual state

**Main result.** Whole block-output donor swaps at sites1/2 also transfer little or negative average donor answer preference. Thus H841's weak candidate result is not explained simply by preserving an informative final-position residual skip around the MLP. The early final-position state itself is a weak donor-transfer location for these five task families under these interventions.

This does **not** prove that these layers contain no task information or do no useful computation. Task information could remain at earlier content positions, require a joint intervention with the recipient's context/value bus, or become available at the prediction position later. Failure of single-position transplantation is weaker than information absence or causal irrelevance.

**Design.** Same80 pairs and two directions as H841, sites1/2. Swap either the donor's entire final-position MLP output or the donor's entire final-position block output. All other positions and recipient attention bus remain native. This is a post-H841 diagnostic on reused tasks, not a new confirmation dataset. All five competent families and both format/lexicon splits retained.

**Whole block-output effects.** Donor-directed answer margin changes, fraction of native donor-recipient gap, and descriptive95% paired-item bootstrap intervals:

|Site|Task|Margin change|Native-gap fraction|Margin interval|
|---|---|---:|---:|---|
|1|agreement|-0.00605|-0.069%|[-0.02493, 0.01170]|
|1|capital_recall|0.00441|0.036%|[-0.00702, 0.01778]|
|1|entity_binding|0.00295|0.038%|[-0.00102, 0.00664]|
|1|sentiment|-0.00619|-0.224%|[-0.01963, 0.00761]|
|1|sequence_copy|-0.05281|-0.645%|[-0.12150, 0.01029]|
|2|agreement|0.00320|0.037%|[-0.02401, 0.02962]|
|2|capital_recall|0.01066|0.087%|[0.00005, 0.02210]|
|2|entity_binding|0.00769|0.099%|[0.00010, 0.01461]|
|2|sentiment|-0.03161|-1.142%|[-0.05625, -0.00711]|
|2|sequence_copy|-0.09260|-1.131%|[-0.13096, -0.05534]|

No block-output cell transfers even0.1% of the native gap in the positive direction on average. Negative values mean movement away from the donor preference, not evidence of useful transfer. These summaries can hide item-level heterogeneity, so full records and paired intervals are retained. Output KL is separate and can be nonzero despite small answer-directed effects.

**Implementation check.** Whole MLP-output donor swaps reproduce H841's whole normalized-input donor swap margins within1.63e-5. Since an MLP is deterministic and token-local, these operations should implement the same function; their agreement supports the patching implementation. Identity block/MLP replacements reproduce logits exactly.1,440 actual forwards, all finite, all source/checkpoint hashes unchanged, no reserved-test evaluations. Natural-document count stays26,112.

**Research-direction update.** The latest experiments distinguish three questions:

1. Do training seeds recover similar directions/subspaces? Some scalar directions do, and some blocks partially do.
2. Do those components occur consistently on competent tasks? H840 found substantial sparsity and cross-seed block-coordinate disagreement.
3. Does the tested location causally transfer task information at all? H841–842 show weak transfer even for whole MLP/block states at the early final position.

Consequently, interpreting these tiny component effects as evidence against all early-layer manifolds would be premature. The next high-information experiment should locate causal task-state availability before refining candidate labels: sweep final-position block states through depth, then compare early whole-sequence/content-position swaps on token-aligned, equal-length pairs. A final-block donor swap provides an exact positive control because its final hidden state directly determines logits; earlier swaps retain recipient context and can fail for substantive reasons. Keep unmatched pairs as explicit exclusions for sequence-aligned interventions, not as model failures. Once a causal route is established, test the frozen components there or train correctly controlled featurizers at the informative site.

The broad representation/computation goal remains unresolved and active. Previous goal turn was progress: it produced a diverse frozen task screen; this turn added6,685 actual forwards across H841–842 and ruled out a simple residual-skip explanation for weak final-position early-MLP transfer.
