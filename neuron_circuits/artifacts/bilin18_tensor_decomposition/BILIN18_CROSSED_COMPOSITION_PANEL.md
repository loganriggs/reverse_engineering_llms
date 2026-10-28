## H1053 — Crossed combination panel isolates a competent residual-dependent agreement test

Previous goal turn was progress: H1052 completed24conditional fits, found narrow transfer gains, and stopped the decoder-only whole-family sequence. Current state remained H1052 with no running GPU jobs. This turn froze and evaluated a new content×context checkerboard before fitting any representation.

Design: four tasks (agreement, pronoun gender, sentiment, antonym), four content pairs per task, four context templates per task:64paired combinations/128unique endpoint prompts. Train if(content index+context index) is even; reserve odd parity. Each content factor and context factor occurs twice in training and twice in the reserved half, while the exact prompt combinations are disjoint. This tests lexical/context recombination, not composition of independently identified computational primitives or transfer to a wholly new task.

Factors are explicit. Agreement changes singular/plural number with noun stem as content and location phrase as context. Pronouns change gendered referents with referent pair as content and location phrase as context. Sentiment changes adjective polarity with adjective pair as content and reviewed object as context. Antonyms change queried lexical identity with opposite pair as content and framing prefix as context. These contrasts do not all isolate the same kind of semantic operation.

All64pairs have valid disjoint single-token answer groups and equal token lengths. Answer groups for agreement and sentiment use the previously fixed semantic sets; pronoun/antonym use exact alternatives. Exact overlap with the four scanned historical task banks is0, and train/reserved overlap0. The historical scan is scoped to listed banks, not a claim that every file in the entire research history was searched. No historical reserved prompts were moved into training.

Fixed gate: all8aligned pairs in a family/split,>=75%both-endpoint binary accuracy,>=50%unrestricted semantic/exact top1. Training eligibility uses training outcomes only; reserved failures remain visible.

| Family | Train binary / top1 | Reserved binary / top1 | Status |
|---|---:|---:|---|
| Agreement |100% /56.25%|100% /56.25%|Both pass|
| Pronoun gender |100% /56.25%|100% /43.75%|Training eligible; reserved diagnostic|
| Sentiment |25% /0%|25% /6.25%|Training ineligible|
| Antonym |75% /43.75%|62.5% /50%|Training ineligible|

Do not reinterpret a correct forced-choice contrast as unrestricted task competence. The lower pronoun reserved top1is retained rather than relaxing the gate. Failed sentiment/antonym grids are not evidence of inability on all formats, and will not be repaired repeatedly to manufacture a broad passing benchmark.

The zero-residual-edit control transfers exact donor token content but leaves recipient contextual residual unchanged. Reserved relative KL is0.9935agreement and0.7253pronoun. Thus agreement's content bypass alone is essentially no help: a successful conditional representation must explain a residual-dependent change. Full native boundary transplant replays exactly. No feature, manifold, or decoder trained in this experiment.

Deliverables: complete frozen factor manifest, all128native boundary states/logits, all64native outcome records, zero-residual causal targets, and training eligibility for16pairs (eightagreement+eightpronoun). Agreement is the primary competent composition endpoint; pronoun remains diagnostic. This is a deliberately easier complementary test to whole-family exclusion, not a replacement success criterion for the broader goal.

Verification:128prefix and384suffix executions, content program exact on all inputs, full-boundary logit replay error0, manifest unchanged. Every factor appears twice per split. Independently recomputed all64native pair outcomes and four saved control-logit KL samples; maximum discrepancy below1e-10. No fitting rows, no new natural documents, and no newly validated whole-task discovery.

Next fixed comparison: add only the16eligible checkerboard training pairs to the existing historical training bank, keeping all reserved combinations and original whole-task exclusions out of every fitted statistic. Compare a historical-only control against the augmented bank under the same conditional residual representation and equal compute, with enough prespecified per-family visits to cover every training pair. This avoids confusing tiny-grid rank deficiency with compositional failure. Evaluate primary agreement and diagnostic pronouns separately, include untrained and zero-residual controls, and report both full training fit and reserved-combination fidelity. No outcome-based rank or checkpoint selection, and no claims about the two ineligible grids.
