## H952 — Frozen-candidate fresh-context panel exposes competence and alignment limits

Prospective 64 prompt pairs: eight source pairs each for agreement/sentiment, four contexts. Agreement includes baseline, relative clause with opposite-number embedded subject, opposite-number prepositional distractor, and extra locative context. Sentiment includes baseline, negation with expected polarity reversed, opposing opening clause, and alternate phrasing. Layer4 relay token/site and all H949 checkpoints were frozen. No replacements or selection based on model outputs. Exact full prompts are disjoint from H849 and original discovery-source examples; this is not a pretraining or entire-research disjointness claim.

53 pairs survive answer-token and equal-length/same-position requirements. All eight relative-clause pairs fail token-length alignment, and painter/painters fails in the other three agreement contexts. This is an untested condition, not a model competence failure. Remaining 106 endpoints were captured and identity-replayed; four intervention arms per directed pair (full contrast, own rank1, rank1 complement, other-task rank1) yield 424 causal evaluations, 636 whole-model forwards total.

Native competence gate was fixed in advance: at least four pairs, >=75% both-endpoint correct binary choices, and >=50% intended full-vocabulary top1. All token-valid pairs were intervened upon, including failed cells.

| Task | Context | Pairs | Both endpoints binary correct | Intended full-vocabulary top1 | Gate |
|---|---|---|---|---|---|
| agreement | baseline | 7 | 100.0% | 85.7% | True |
| agreement | composition | 0 | 0.0% | 0.0% | False |
| agreement | distractor | 7 | 100.0% | 50.0% | True |
| agreement | alternate | 7 | 100.0% | 42.9% | False |
| sentiment | baseline | 8 | 100.0% | 68.8% | True |
| sentiment | composition | 8 | 100.0% | 18.8% | False |
| sentiment | distractor | 8 | 37.5% | 81.2% | False |
| sentiment | alternate | 8 | 100.0% | 50.0% | True |

Four of eight cells pass. Negation preserves binary polarity discrimination but fails the full-vocabulary gate; the conflicting sentiment clause fails binary consistency despite often predicting one intended label. These are meaningful limitations of the teacher on the new contexts. Alternate agreement narrowly misses the full-vocabulary threshold. Do not silently discard these conditions or interpret student errors there as failures to reproduce a demonstrated task capability.

Audit: identity replay max absolute error zero, same token and position for all included donor pairs, no exact prior full-prompt overlap, manifest/checkpoint/input hashes unchanged, all effects finite. No reserved-test evaluations, no training, natural-document pool unchanged at26,112. A future relative-clause supplement must fix input alignment prospectively; it cannot be claimed tested here.

Next: evaluate frozen complete and top-eight student finite MLP effects on every valid pair, then live downstream fidelity grouped by context and competence status. Keep all penalties/seeds; no fresh-panel refitting or replacement.
