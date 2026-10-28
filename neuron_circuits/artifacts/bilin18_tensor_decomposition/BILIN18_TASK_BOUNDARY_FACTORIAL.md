## H582 — Small format changes improve competence, but threshold-passing formats need not transfer

The previous goal turn made progress: H581 established a diverse causal panel and exposed second-template failures. Before further causal interpretation, preregister task-specific two-by-two format factors over discovery and lexical-transfer sets. Each condition has eight attempted pairs with both answer endpoints and both same-answer nuisance versions. Factors: agreement beside/near x empty/usually ending; recall national capital/capital city x is/is the city of ending; binding school/library x gave/handed; sentiment Text/Review x Overall sentiment/Sentiment; copying He/She x spoke with/talked to.

Score 1,248 prompts. Eight condition-pairs (32 prompts) fail single-token answer eligibility: the Dennis/Jacqueline transfer pair in binding and copying across four conditions. These exclusions happen before model scoring, not from performance. Some discovery prompts repeat H581. Transfer lexical combinations are new to this panel, not claimed unseen by the model or all earlier studies.

Gate stays >=4 valid pairs, >=75% both-endpoint candidate correctness and >=50% intended-group full-vocabulary top1. Before transfer inference, choose the lexicographically first discovery-passing condition, not the best transfer condition. Keep every result and the earlier failed panel.

| Task | Locked condition | Discovery top1 | Transfer top1 | Discovery nuisance top1 | Transfer nuisance top1 | Transfer gate |
|---|---|---:|---:|---:|---:|---|
|agreement|01|1.000|0.938|1.000|0.938|pass|
|capital_recall|10|1.000|1.000|1.000|1.000|pass|
|entity_binding|00|0.500|0.357|0.438|0.357|fail|
|sentiment|01|0.500|0.438|0.250|0.188|fail|
|sequence_copy|00|1.000|1.000|1.000|1.000|pass|

Agreement: adding “usually” after the beside phrase raises discovery top1 from 0.438 to 1.000 and transfer from 0.563 to 0.938. Both-endpoint candidate accuracy was already 1.000. This is consistent with making a finite-verb continuation more likely, not adding number knowledge.

Recall: changing “national capital” to “capital city” yields 1.000 top1 in both splits. Appending “is the city of” is not a universal fix: with “national capital” it reaches only 0.375 discovery top1 but 0.875 transfer. A direct boundary explanation alone is insufficient; wording and lexical context interact.

Binding: selected condition 00 barely passes discovery (0.500) and fails transfer (0.357). Other cells perform better, but choosing them after inspecting transfer would invalidate this particular locked selection test. Sentiment: selected condition 01 similarly falls from 0.500 to 0.438, and its nuisance variants perform worse. Review+Sentiment performs better in both splits, but remains an alternative observed condition rather than a replacement for the locked primary. Copying passes all cells on both splits.

This demonstrates why mere threshold passing on one small discovery set is weak evidence for generalization. It also explains why candidate-only scores can conceal continuation-format differences. Factor changes can alter the underlying computation; this is not a pure answer-boundary causal intervention.

Next H583 retests fixed writer operations on these locked conditions, reporting three competent transfer tasks and retaining the two failed cells. This preserves the selection rule. Future stronger format selection should use discovery-only robustness criteria and a genuinely new confirmation set, not retrofit this transfer set.

Audits: 256 repeated prompt+answer-set comparisons replay H581 within 1.91e-5 margin; all finite; locked selection matches; source/manifest hashes unchanged. An initial comparison keyed only by prompt text incorrectly mixed different capital candidate pairs; fixed to key by text and answer set and reran with unchanged prompts and selection rule. No new natural-text documents; FineWeb count remains 26,112. Broad goal active and incomplete.
